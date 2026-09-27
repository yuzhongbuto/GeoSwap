"""geoswap_rounds.py -- 多轮「搜索在环」的交换提案头训练（560 实例版）。

方案（用户 2026-09-26 定稿）
---------------------------------------------------------------
两个头，唯一变量是"有没有 Transformer 的表征"（2026-09-26 诊断后定稿）：
    头 A（有 Transformer）  输入 = 骨干 encoder memory 256 维      （只用 memory）
    头 B（不用 Transformer）输入 = 手工特征 13 维                   （不用骨干）

    诊断结论（写进代码的理由）：
      * 手工各列 std 相差 5 个数量级（aa≈5365 vs 位置≈0.2），不标准化时换种子结果就翻
        （prec@1 sd 0.035，组间差 0.046）-> 两个头一律做逐列标准化，mu/sd 存进 checkpoint。
      * 手工13 + memory256 拼起来（0.354）反而不如只用 memory（0.396）-> 头 A 只用 memory，
        hand+mem 作为每轮的消融行 headAB_both 一起报。

轮次：
    第 1 轮（点火，不计入迭代轮）：提案器 = 随机。
        跑 560 实例 x 40 次 BLF 评估 -> 迭代曲线 + 训练样本 -> 训出头 A_r1 / 头 B_r1
    第 2..R 轮（默认 R=6，即 5 个迭代轮）：每轮两个头各跑一遍搜索（提案不同、轨迹分叉，
        不能共用），各自产出自己的迭代曲线与样本，再各自训出新一版头。

一次 BLF 评估 = 一条训练样本 = 一个迭代点：
    - 迭代曲线：第 k 次评估后的利用率 u_k（k=1..40，接不接受都记）
    - 训练样本：(状态深度, (i,j), 手工13 + 借来的分4 + memory256, ΔU)

样本跨轮累积：第 r 轮的头 A 用 r1_random + r2_mem + ... + rr_mem 全部样本训练，
头 B 同理用 hand 那条链。第 1 轮的随机样本两边都当点火数据。

产物（逐实例落盘，全部不覆盖；断电后重跑同一命令即从断点续）
---------------------------------------------------------------
    results/geoswap_<tag>/
        round1_random/  search.jsonl  curve.jsonl  samples/<inst>.npz
        round2_mem/     ...          round2_hand/ ...
        headA_r1.pth ... headA_rR.pth   headB_r1.pth ... headB_rR.pth
        offline_r1.txt ... offline_rR.txt
        eval/  {mem,hand,borrowed,random}/curve.jsonl
        state.json   run.log
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import common as C  # noqa: E402

C.prep_env()
C.utf8_stdout()
os.environ.setdefault('PACKING_FAST_BLF', '1')

_POOL_MAX = 60
HAND = 13
LPF = 4                       # lp_i, lp_j, lp_i+lp_j, lp_i-lp_j
MEMD = 128
OFF_LP = HAND                 # 13
OFF_MEM = HAND + LPF          # 17
XDIM = OFF_MEM + 2 * MEMD     # 273

COLS_A = np.arange(OFF_MEM, XDIM)                       # 头 A: 只用 memory (256)
COLS_BOTH = np.concatenate([np.arange(0, HAND), np.arange(OFF_MEM, XDIM)])   # 消融: 手工13+memory (269)
COLS_B = np.arange(0, HAND)                             # 头 B: 只用手工 (13)

# 逐列标准化（2026-09-26 诊断加）：不标准化时手工各列 std 相差 5 个数量级
# （aa≈5365 vs 位置≈0.2），同一份数据换种子结果就翻（prec@1 sd 0.035，组间差 0.046）。
# 标准化后 sd 掉到 0.006~0.012，命中率翻倍。mu/sd 必须存进 checkpoint 供推理复用。
YPRUNE_HINT = 'PACKING_BLF_YPRUNE'


def log(msg):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), msg), flush=True)


def _winit():
    """池初始化：每个 worker 单线程跑 torch，避免线程争抢。必须是模块级函数（spawn 要 pickle）。"""
    import torch
    torch.set_num_threads(1)


# ===================================================================== 多池
def run_sharded(tasks, fn, n_workers, on_result):
    """多池并发（Windows 单池 <=60 handle）。"""
    import threading
    import multiprocessing as mp
    ctx = mp.get_context('spawn')
    n_workers = max(1, int(n_workers))
    if n_workers <= 1:
        import torch
        torch.set_num_threads(1)
        for i, t in enumerate(tasks, 1):
            on_result(fn(t), i, len(tasks))
        return
    n_pools = max(1, (n_workers + _POOL_MAX - 1) // _POOL_MAX)
    per_pool = max(1, n_workers // n_pools)

    chunks = [tasks[i::n_pools] for i in range(n_pools)]
    lock = threading.Lock()
    state = {'done': 0}
    errs = []

    def _worker(chunk):
        if not chunk:
            return
        try:
            with ctx.Pool(processes=min(per_pool, len(chunk)), initializer=_winit) as pool:
                for r in pool.imap_unordered(fn, chunk, chunksize=1):
                    with lock:
                        state['done'] += 1
                        kk = state['done']
                    on_result(r, kk, len(tasks))
        except Exception:
            import traceback
            errs.append(traceback.format_exc()[-800:])

    log('并发 %d -> %d 池 x %d' % (n_workers, n_pools, per_pool))
    ths = [threading.Thread(target=_worker, args=(c,)) for c in chunks]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    for e in errs:
        log('POOL ERROR\n%s' % e)


# ===================================================================== 特征
def build_rows(n, order, areas, lp, mem, pairs):
    """向量化构造样本行。

    pairs: (P,2) 位置对。返回 hand (P,13) 与 memblock (P,256)。
    手工 13 维的定义与 ranksteer/probe 完全一致：
        n, i/(n-1), j/(n-1), (j-i)/(n-1), aa, ab, ab-aa, (ab-aa)/aa,
        min/max, log aa, log ab, 1[ab<aa], |i-j|
    """
    i = pairs[:, 0].astype(np.float64)
    j = pairs[:, 1].astype(np.float64)
    order = np.asarray(order, dtype=np.int64)
    a = order[pairs[:, 0]]
    b = order[pairs[:, 1]]
    aa = areas[a]
    ab = areas[b]
    dn = float(max(n - 1, 1))
    h = np.empty((len(pairs), HAND), dtype=np.float32)
    h[:, 0] = float(n)
    h[:, 1] = i / dn
    h[:, 2] = j / dn
    h[:, 3] = (j - i) / dn
    h[:, 4] = aa
    h[:, 5] = ab
    h[:, 6] = ab - aa
    h[:, 7] = (ab - aa) / np.maximum(aa, 1e-9)
    h[:, 8] = np.minimum(aa, ab) / np.maximum(np.maximum(aa, ab), 1e-9)
    h[:, 9] = np.log(np.maximum(aa, 1e-9))
    h[:, 10] = np.log(np.maximum(ab, 1e-9))
    h[:, 11] = (ab < aa).astype(np.float32)
    h[:, 12] = np.abs(i - j)
    lpblk = np.empty((len(pairs), LPF), dtype=np.float32)
    li = lp[pairs[:, 0]]
    lj = lp[pairs[:, 1]]
    lpblk[:, 0] = li
    lpblk[:, 1] = lj
    lpblk[:, 2] = li + lj
    lpblk[:, 3] = li - lj
    mb = np.concatenate([mem[pairs[:, 0]], mem[pairs[:, 1]]], axis=1).astype(np.float32)
    return h, lpblk, mb


def backbone_step(actor, feats, order):
    """给定顺序跑一次骨干，返回 (log_softmax 分数, encoder memory)。"""
    import torch
    from ranksteer.supervised_pretrain import START_IDX
    src = torch.tensor(np.asarray(feats)[order], dtype=torch.float32).unsqueeze(0)
    tgt = torch.tensor([[START_IDX]], dtype=torch.long)
    with torch.no_grad():
        mem = actor.encode(src)                  # [1,n,128]
        dec = actor.decode_step(mem, tgt)
        ol = actor.forward_logits(mem, dec)
    lg = torch.nan_to_num(ol[0, -1, :len(order)], nan=0.0, posinf=50.0, neginf=-50.0)
    lp = torch.log_softmax(lg.float(), dim=-1).numpy()
    return lp, mem[0, :len(order)].numpy().astype(np.float32)


def mlp(dim, hidden=256, seed=0):
    import torch
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(dim, hidden), torch.nn.GELU(),
        torch.nn.Linear(hidden, hidden // 2), torch.nn.GELU(),
        torch.nn.Linear(hidden // 2, 1))


def load_head(path):
    """返回 (net, mu, sd, cols)。mu/sd 是训练时算的逐列标准化统计量。"""
    import torch
    ck = C.torch_load(path, 'cpu')
    net = mlp(int(ck['dim']), int(ck['hidden']))
    net.load_state_dict(ck['sd'])
    net.eval()
    mu = ck.get('x_mu')
    sd = ck.get('x_sd')
    cols = ck.get('cols')
    if cols is None:
        cols = np.arange(int(ck['dim']))
    return net, mu, sd, np.asarray(cols, dtype=np.int64)


def predict(net, X, mu=None, sd=None):
    """mu/sd 非 None 时先做逐列标准化，必须与训练时用的是同一套。"""
    import torch
    if mu is not None:
        X = (X - np.asarray(mu, dtype=np.float32)) / np.asarray(sd, dtype=np.float32)
    with torch.no_grad():
        out = []
        for s in range(0, len(X), 16384):
            out.append(net(torch.tensor(X[s:s + 16384], dtype=torch.float32)).numpy().ravel())
    return np.concatenate(out) if out else np.zeros(0)


# ===================================================================== 搜索
def search_task(task):
    """一个实例的完整搜索：预算 = BLF 评估次数。

    起点 = 面积序。每步：当前提案器给所有"还没试过的"交换对打分 -> 取最高分那对
    -> 交换 -> BLF 真跑一遍 -> ΔU>0 接受，否则换下一对。被拒的对在同一局内不再重试。
    每一次评估都产出一个迭代点 + 一条样本。
    """
    import torch
    torch.set_num_threads(1)
    import common as C2

    (inst_path, name, kind, headA_path, headB_path, budget, seed, anchor) = task
    try:
        t0 = time.time()
        env = C2.env_for(inst_path)
        n = int(env.n)
        feats = env.features.numpy() if hasattr(env.features, 'numpy') else np.asarray(env.features)
        areas = np.asarray(env.residual_raw[:, 0], dtype=float)
        rng = random.Random(seed)

        actor, _ = C2.build_actor(C2.get_actor_sd(C2.torch_load(anchor)), 'cpu')
        head = None
        h_mu = h_sd = None
        if kind == 'headA':
            head, h_mu, h_sd, _ = load_head(headA_path)   # 输入 = 只用 memory (256)
        elif kind == 'headB':
            head, h_mu, h_sd, _ = load_head(headB_path)   # 输入 = 只用手工 (13)

        order = [int(x) for x in C2.area_order(env)]
        u, _ = C2.rollout_given_order(env, order)
        u_area = float(u)

        tried = set()
        curve = []          # (k, u_before, u_after, du, accepted, depth)
        S_hand, S_lp, S_mem, S_du, S_meta = [], [], [], [], []
        n_acc = 0
        cand = None
        ptr = 0
        state_key = None

        while len(curve) < int(budget):
            key = tuple(order)
            if key != state_key:
                lp, mem = backbone_step(actor, feats, order)
                pairs = np.asarray([(a, b) for a in range(n) for b in range(a + 1, n)
                                    if (a, b) not in tried], dtype=np.int64)
                # 剔除"空操作"交换对：两个零件的特征逐位相同 -> 交换后布局逐位不变，ΔU 恒等于 0。
                # 为什么必须剔：模型学到"面积比越接近 1 越好"这条先验，会把这类对排在最前面，
                # 于是整个评估预算全砸在空操作上。公开集 shapes0 只有 4 种零件（43 个 = 15+12+9+7），
                # 25% 以上的对是空操作，2026-09-26 实测：不剔 = 40 次评估一步不涨；剔了 = +1.93pp。
                # 训练集由随机尺寸生成、几乎没有重复零件，所以这个坑在训练时看不见。
                if len(pairs):
                    sig = np.unique(np.asarray(feats, dtype=np.float32), axis=0,
                                    return_inverse=True)[1]
                    oarr = np.asarray(order, dtype=np.int64)
                    keep = sig[oarr[pairs[:, 0]]] != sig[oarr[pairs[:, 1]]]
                    pairs = pairs[keep]
                if pairs.size == 0:
                    break
                h, lpblk, mb = build_rows(n, order, areas, lp, mem, pairs)
                if kind == 'random':
                    perm = np.arange(len(pairs))
                    rng.shuffle(perm)                      # 随机提案 = 随机候选顺序
                else:
                    if kind == 'borrowed':
                        sc = lpblk[:, 2]                   # 免费基线：lp_i + lp_j
                    elif kind == 'headA':
                        sc = predict(head, mb, h_mu, h_sd)          # 只用 memory
                    else:
                        sc = predict(head, h, h_mu, h_sd)           # 只用手工
                    perm = np.argsort(-sc, kind='stable')
                cand = pairs[perm]
                cand_h = h[perm]
                cand_lp = lpblk[perm]
                cand_m = mb[perm]
                ptr = 0
                state_key = key
            if ptr >= len(cand):
                break
            a, b = int(cand[ptr, 0]), int(cand[ptr, 1])
            hrow, lprow, mrow = cand_h[ptr], cand_lp[ptr], cand_m[ptr]
            ptr += 1
            tried.add((a, b))
            new = list(order)
            new[a], new[b] = new[b], new[a]
            u_new, _ = C2.rollout_given_order(env, new)
            n_eval = len(curve) + 1
            du = float(u_new - u)
            acc = 1 if du > 0 else 0
            curve.append((n_eval, float(u), float(u_new), du, acc, int(n_acc)))
            S_hand.append(hrow)
            S_lp.append(lprow)
            S_mem.append(mrow)
            S_du.append(du)
            S_meta.append((int(n_acc), a, b, int(order[a]), int(order[b])))
            if du > 0:
                order, u = new, float(u_new)
                n_acc += 1

        return {'ok': True, 'instance': name, 'n': n, 'u_area': u_area, 'u_final': float(u),
                'improve': float(u - u_area), 'n_acc': n_acc, 'n_eval': len(curve),
                'rho': C2.order_rho(order, env), 'order': order,
                'curve': curve,
                'hand': np.asarray(S_hand, dtype=np.float32),
                'lp': np.asarray(S_lp, dtype=np.float32),
                'mem': np.asarray(S_mem, dtype=np.float32),
                'du': np.asarray(S_du, dtype=np.float64),
                'meta': S_meta,
                'seconds': time.time() - t0}
    except Exception:
        import traceback
        return {'ok': False, 'instance': name, 'error': traceback.format_exc()[-500:]}


# ===================================================================== 阶段
def stage_dir(out, stage):
    return C.out_dir(os.path.join(out, stage))


def done_instances(sdir):
    """已完成的实例名集合。search.jsonl 是**实例收尾时才写**的，所以它是权威进度标记；
    断电时最后一行可能写坏，json 解析失败的那行会被丢弃 -> 该实例重跑（安全）。"""
    p = os.path.join(sdir, 'search.jsonl')
    done = set()
    if not os.path.exists(p):
        return done
    with open(p, encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)['instance'])
            except Exception:
                continue
    return done


def run_stage(out, stage, insts, kind, headA_path, headB_path, budget, workers, seed, anchor):
    import threading
    sdir = stage_dir(out, stage)
    done = done_instances(sdir)
    todo = [p for p in insts if os.path.basename(p) not in done]
    log('stage=%s 提案器=%s  待跑 %d/%d 实例' % (stage, kind, len(todo), len(insts)))
    if not todo:
        log('stage=%s 已全部完成，跳过' % stage)
        return
    tasks = [(p, os.path.basename(p), kind, headA_path, headB_path, budget,
              seed * 7919 + i, anchor) for i, p in enumerate(todo)]
    sp = os.path.join(sdir, 'search.jsonl')
    cp = os.path.join(sdir, 'curve.jsonl')
    smp = C.out_dir(os.path.join(sdir, 'samples'))
    stats = {'ok': 0, 'fail': 0, 'evals': 0, 'sum_u': 0.0, 'sum_start': 0.0,
             'improved': 0, 'acc': 0}
    t0 = time.time()
    ntot = len(tasks)
    wlock = threading.Lock()

    def _on(r, k, ntot_):
        if not r.get('ok'):
            stats['fail'] += 1
            log('FAIL %s :: %s' % (r.get('instance'), r.get('error', '')[-200:]))
            return
        stats['ok'] += 1
        stats['evals'] += r['n_eval']
        stats['sum_u'] += r['u_final']
        stats['sum_start'] += r['u_area']
        stats['acc'] += r['n_acc']
        if r['improve'] > 1e-9:
            stats['improved'] += 1
        if len(r['du']):
            np.savez_compressed(os.path.join(smp, r['instance'].replace('.txt', '') + '.npz'),
                                hand=r['hand'], lp=r['lp'], mem=r['mem'], du=r['du'],
                                meta=np.asarray(r['meta'], dtype=np.int64))
        rec = {'instance': r['instance'], 'n': r['n'], 'u_area': r['u_area'],
               'u_final': r['u_final'], 'improve': r['improve'], 'n_acc': r['n_acc'],
               'n_eval': r['n_eval'], 'rho': r['rho'], 'seconds': r['seconds'],
               'order': r['order']}
        with wlock:
            C.append_jsonl(sp, rec)
            for c in r['curve']:
                C.append_jsonl(cp, {'instance': r['instance'], 'k': c[0], 'u_before': c[1],
                                    'u_after': c[2], 'du': c[3], 'accepted': c[4],
                                    'depth': c[5]})
        if k % 5 == 0 or k == ntot_:
            el = time.time() - t0
            eta = el / max(k, 1) * (ntot_ - k)
            log('  %s %d/%d (%.1f%%)  均值u %.4f (起点 %.4f)  改进 %d  平均接受 %.1f  '
                'BLF %d  已用 %.1fm  预计剩余 %.1fm'
                % (stage, k, ntot_, 100.0 * k / ntot_,
                   stats['sum_u'] / max(stats['ok'], 1),
                   stats['sum_start'] / max(stats['ok'], 1),
                   stats['improved'], stats['acc'] / max(stats['ok'], 1),
                   stats['evals'], el / 60.0, eta / 60.0))

    run_sharded(tasks, search_task, workers, _on)
    el = time.time() - t0
    log('stage=%s 完成：实例 %d 失败 %d  BLF %d  用时 %.1f 分钟  吞吐 %.1f 次/秒'
        % (stage, stats['ok'], stats['fail'], stats['evals'], el / 60.0,
           stats['evals'] / max(el, 1e-6)))
    C.atomic_json_dump({'stage': stage, 'kind': kind, 'tasks': ntot, **stats,
                        'seconds': el}, os.path.join(sdir, 'stage_summary.json'))


def load_stage_samples(out, stages):
    H, L, M, D, K, I = [], [], [], [], [], []
    for st in stages:
        d = os.path.join(out, st, 'samples')
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith('.npz'):
                continue
            z = np.load(os.path.join(d, f), allow_pickle=True)
            H.append(z['hand']); L.append(z['lp']); M.append(z['mem']); D.append(z['du'])
            K.append(z['meta'][:, 0])
            I.append(np.asarray([f.replace('.npz', '')] * len(z['du'])))
    if not D:
        return None
    return {'hand': np.concatenate(H), 'lp': np.concatenate(L), 'mem': np.concatenate(M),
            'du': np.concatenate(D), 'depth': np.concatenate(K), 'inst': np.concatenate(I)}


def _metrics(scores, du, ks=(1, 3, 5, 10)):
    o = np.argsort(-scores, kind='stable')
    out = {}
    for k in ks:
        top = o[:k]
        out['prec@%d' % k] = float((du[top] > 1e-12).mean()) if len(top) else 0.0
    out['harv@5'] = float(du[o[:5]][du[o[:5]] > 1e-12].sum())
    return out


def train_round(out, r, samples, epochs, lr, hidden, seed, device):
    """用累积样本训第 r 轮的两个头；同时给离线报告。"""
    import torch
    H, L, M, D, K, I = (samples['hand'], samples['lp'], samples['mem'], samples['du'],
                        samples['depth'], samples['inst'])
    # 头 A = 只用 memory（诊断：手工13 + memory256 拼起来 0.354 < 只用 memory 0.396，
    # 手工特征对 memory 是冗余加噪）。COLS_BOTH 保留成每轮的消融行。
    XV = {'headA_mem': M, 'headAB_both': np.concatenate([H, M], axis=1), 'headB_hand': H}
    XA = M
    XB = H
    names = sorted(set(I.tolist()))
    idx = {nm: i for i, nm in enumerate(names)}
    iid = np.asarray([idx[x] for x in I])
    y = (D > 0).astype(np.float32)
    log('训练第 %d 轮：样本 %d  实例 %d  正样本率 %.2f%%  设备=%s'
        % (r, len(D), len(names), 100 * y.mean(), device))
    rng = np.random.RandomState(seed + r)
    folds = np.array_split(rng.permutation(len(names)), 5)
    lines = []
    lines.append('第 %d 轮离线报告  样本 %d  实例 %d  正样本率 %.2f%%（含逐列标准化）' %
                 (r, len(D), len(names), 100 * y.mean()))
    lines.append('%-14s %8s %8s %8s %11s' % ('proposer', 'prec@1', 'prec@3', 'prec@5', 'harv@5'))

    def _rand(cnt):
        return np.random.RandomState(999 + cnt).rand(cnt)

    res = {}
    for tag, X in XV.items():
        acc = {'learned': [], 'rand': [], 'borrowed': [], 'hand': []}
        for fi in range(5):
            te = folds[fi]
            mte = np.isin(iid, te)
            if not mte.any():
                continue
            Xtr, ytr = X[~mte], y[~mte]
            if len(Xtr) < 8 or ytr.sum() < 2:
                continue
            tf = time.time()
            net, mu, sd = _fit(Xtr, ytr, X.shape[1], epochs, lr, hidden, seed + fi, device)
            log('  %s fold%d/5 拟合完成 %.1fs' % (tag, fi + 1, time.time() - tf))
            Xte, dte = X[mte], D[mte]
            pr = predict(net, Xte, mu, sd)
            # 手工规则：面积比接近 1 且位置间隔小（与旧脚本同定义）
            hrule = -(np.abs(np.log(np.clip(H[mte, 8], 1e-9, 1.0))) / 0.15 + H[mte, 3] / 0.05)
            for nm in te:
                m = iid[mte] == nm
                if m.sum() < 4:
                    continue
                acc['learned'].append(_metrics(pr[m], dte[m]))
                acc['rand'].append(_metrics(_rand(int(m.sum())), dte[m]))
                acc['borrowed'].append(_metrics(L[mte][m, 2], dte[m]))
                acc['hand'].append(_metrics(hrule[m], dte[m]))
        res[tag] = acc
        for nm in ('rand', 'hand', 'borrowed', 'learned'):
            rows = acc[nm]
            if not rows:
                continue
            lines.append('%-14s %8.3f %8.3f %8.3f %+11.5f'
                         % ('%s/%s' % (tag, nm),
                            np.mean([x['prec@1'] for x in rows]),
                            np.mean([x['prec@3'] for x in rows]),
                            np.mean([x['prec@5'] for x in rows]),
                            np.mean([x['harv@5'] for x in rows])))
        lines.append('')

    txt = '\n'.join(lines)
    with open(os.path.join(out, 'offline_r%d.txt' % r), 'w', encoding='utf-8') as f:
        f.write(txt + '\n')
    log('第 %d 轮离线报告：\n%s' % (r, txt))

    # 全量重训并保存（不覆盖：文件名带轮次）。mu/sd 必须一起存，推理要用同一套。
    for tag, X, cols, kind in (('headA', XA, COLS_A, 'mem'), ('headB', XB, COLS_B, 'hand')):
        tf = time.time()
        net, mu, sd = _fit(X, y, X.shape[1], epochs, lr, hidden, seed, device)
        C.atomic_torch_save({'sd': net.state_dict(), 'dim': int(X.shape[1]),
                             'hidden': hidden, 'cols': np.asarray(cols, dtype=np.int64),
                             'x_mu': mu, 'x_sd': sd, 'round': r, 'kind': kind,
                             'note': 'geoswap_rounds %s r%d（含逐列标准化）' % (tag, r)},
                            os.path.join(out, '%s_r%d.pth' % (tag, r)))
        log('存 %s_r%d.pth（%d 维，%d 样本，全量重训 %.1fs；已存 mu/sd）'
            % (tag, r, X.shape[1], len(X), time.time() - tf))
    return res


def _fit(X, y, dim, epochs, lr, hidden, seed, device):
    """训练打分头，返回 (net, mu, sd)。

    做两件事，都是 2026-09-26 诊断的结论：
      1) 逐列标准化（mu/sd 由训练集算，必须随 checkpoint 一起保存，推理时复用）；
      2) 显式放开线程并优先用 GPU —— worker 池跑在 OMP_NUM_THREADS=1 下，
         主进程训练会继承成单线程，22k 样本 x 300 轮会变成几小时。
    """
    import torch
    torch.set_num_threads(max(1, min(64, (os.cpu_count() or 4))))
    dev = torch.device(device)
    mu = X.mean(0).astype(np.float32)
    sd = X.std(0).astype(np.float32)
    sd = np.where(sd < 1e-6, 1.0, sd).astype(np.float32)
    Xs = (X - mu) / sd
    net = mlp(dim, hidden, seed).to(dev)
    pw = float((1 - y.mean()) / max(y.mean(), 1e-6))
    posw = torch.tensor([min(pw, 10.0)], device=dev)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=posw)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    Z = torch.tensor(Xs, dtype=torch.float32, device=dev)
    T = torch.tensor(y, dtype=torch.float32, device=dev).view(-1, 1)
    for _ in range(int(epochs)):
        opt.zero_grad()
        loss = lossf(net(Z), T)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    net.eval()
    return net.to('cpu'), mu, sd


def pick_train_device():
    import torch
    if os.environ.get('GSW_TRAIN_DEVICE'):
        return os.environ['GSW_TRAIN_DEVICE']
    return 'cuda' if torch.cuda.is_available() else 'cpu'


# ===================================================================== 评测
def eval_task(task):
    import torch
    torch.set_num_threads(1)
    import common as C2
    (inst_path, name, kind, headA_path, headB_path, budget, seed, anchor) = task
    try:
        r = search_task(task)
        if not r.get('ok'):
            return r
        return {'ok': True, 'instance': name, 'n': r['n'], 'u_area': r['u_area'],
                'u_final': r['u_final'], 'improve': r['improve'], 'n_acc': r['n_acc'],
                'n_eval': r['n_eval'], 'rho': r['rho'], 'curve': r['curve'],
                'seconds': r['seconds']}
    except Exception:
        import traceback
        return {'ok': False, 'instance': name, 'error': traceback.format_exc()[-400:]}


def run_eval(out, tag, headA_path, headB_path, budget, workers, seed, anchor, test30, esicup):
    ed = C.out_dir(os.path.join(out, 'eval'))
    sets = [('30inst', C.test30_dir(test30)), ('esicup', C.esicup_dir(esicup))]
    variants = [('mem', 'headA'), ('hand', 'headB'), ('borrowed', 'borrowed'), ('random', 'random')]
    for label, d in sets:
        if not d:
            log('评测集 %s 找不到，跳过' % label)
            continue
        insts = C.list_instances(d)
        for vname, kind in variants:
            cp = os.path.join(ed, '%s_%s_curve.jsonl' % (label, vname))
            sp = os.path.join(ed, '%s_%s_search.jsonl' % (label, vname))
            done = set()
            if os.path.exists(sp):
                with open(sp, encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                done.add(json.loads(line)['instance'])
                            except Exception:
                                pass
            todo = [p for p in insts if os.path.basename(p) not in done]
            if not todo:
                log('eval %s/%s 已完成，跳过' % (label, vname))
                continue
            log('eval %s/%s 待跑 %d 实例 预算 %d' % (label, vname, len(todo), budget))
            tasks = [(p, os.path.basename(p), kind, headA_path, headB_path, budget,
                      seed * 104729 + i, anchor) for i, p in enumerate(todo)]
            rows = []
            t0 = time.time()
            import threading
            wlock = threading.Lock()

            def _on(r, k, ntot_):
                if not r.get('ok'):
                    log('eval FAIL %s :: %s' % (r.get('instance'), r.get('error', '')[-200:]))
                    return
                rows.append(r)
                with wlock:
                    C.append_jsonl(sp, {kk: r[kk] for kk in
                                        ('instance', 'n', 'u_area', 'u_final', 'improve',
                                         'n_acc', 'n_eval', 'rho', 'seconds')})
                    for c in r['curve']:
                        C.append_jsonl(cp, {'instance': r['instance'], 'k': c[0],
                                            'u_before': c[1], 'u_after': c[2], 'du': c[3],
                                            'accepted': c[4], 'depth': c[5]})

            run_sharded(tasks, eval_task, workers, _on)
            if rows:
                ua = np.mean([r['u_area'] for r in rows])
                uf = np.mean([r['u_final'] for r in rows])
                log('eval %-8s %-9s n=%d  area %.4f -> %.4f (%+.4f)  改进 %d/%d  %.1f 分钟'
                    % (label, vname, len(rows), ua, uf, uf - ua,
                       sum(1 for r in rows if r['improve'] > 1e-9), len(rows),
                       (time.time() - t0) / 60.0))


# ===================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['run', 'eval'])
    ap.add_argument('--tag', default='gsw560')
    ap.add_argument('--rounds', type=int, default=6, help='总轮数，含点火轮（默认 6 = 点火 + 5 迭代）')
    ap.add_argument('--instances', type=int, default=560)
    ap.add_argument('--budget', type=int, default=40, help='每实例 BLF 评估预算')
    ap.add_argument('--workers', type=int, default=120)
    ap.add_argument('--eval_workers', type=int, default=60)
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--anchor', default=C.DEFAULT_ANCHOR)
    ap.add_argument('--train_dir', default=None)
    ap.add_argument('--test30', default=None)
    ap.add_argument('--esicup', default=None)
    ap.add_argument('--max_hours', type=float, default=0.0, help='>0 时超时安全退出')
    ap.add_argument('--no_eval', action='store_true', help='跑完轮次后不做最终评测（本机通路验证用）')
    args = ap.parse_args()

    out = C.out_dir(os.path.join('rlfine', 'results', 'geoswap_%s' % args.tag))
    anchor = os.path.abspath(args.anchor)
    t_start = time.time()
    log('=' * 78)
    log('geoswap_rounds  tag=%s  轮数=%d  实例上限=%d  预算=%d  workers=%d'
        % (args.tag, args.rounds, args.instances, args.budget, args.workers))
    log('输出目录 %s' % out)
    log('anchor %s' % anchor)

    if args.mode == 'eval':
        headA = os.path.join(out, 'headA_r%d.pth' % args.rounds)
        headB = os.path.join(out, 'headB_r%d.pth' % args.rounds)
        log('评测用头：%s / %s' % (headA, headB))
        run_eval(out, args.tag, headA, headB, args.budget, args.eval_workers, args.seed,
                 anchor, args.test30, args.esicup)
        return

    d = C.train_dir(args.train_dir)
    insts = C.list_instances(d)
    if args.instances and args.instances < len(insts):
        insts = insts[:args.instances]
    log('训练实例 %d 个（%s）' % (len(insts), d))

    for r in range(1, int(args.rounds) + 1):
        if args.max_hours > 0 and (time.time() - t_start) / 3600.0 > args.max_hours:
            log('已超过 --max_hours=%.1f，安全退出（重跑同命令即续）' % args.max_hours)
            return
        if os.path.exists(os.path.join(out, 'round%d_done.json' % r)):
            log('第 %d 轮已完成（round%d_done.json），跳过' % (r, r))
            continue
        if r == 1:
            stages = [(('r1_random'), 'random', None, None)]
        else:
            stages = [('r%d_mem' % r, 'headA',
                       os.path.join(out, 'headA_r%d.pth' % (r - 1)),
                       os.path.join(out, 'headB_r%d.pth' % (r - 1))),
                      ('r%d_hand' % r, 'headB',
                       os.path.join(out, 'headA_r%d.pth' % (r - 1)),
                       os.path.join(out, 'headB_r%d.pth' % (r - 1)))]
        for stage, kind, hA, hB in stages:
            if kind != 'random' and not os.path.exists(hA if kind == 'headA' else hB):
                log('缺上一轮的头 %s，停在第 %d 轮 %s' % (hA if kind == 'headA' else hB, r, stage))
                return
            run_stage(out, stage, insts, kind, hA, hB, args.budget, args.workers,
                      args.seed, anchor)
        # 累积样本：第 1 轮（随机）是点火数据，之后每轮的 mem / hand 两条链都并进来。
        # 每个阶段对每条样本都存了 hand / lp / mem 三块，所以两个头用的是同一批样本，
        # 差别只在各自取哪几列（头 A = 手工13+memory256；头 B = 手工13）。
        all_stages = ['r1_random']
        for x in range(2, r + 1):
            all_stages += ['r%d_mem' % x, 'r%d_hand' % x]
        samples = load_stage_samples(out, all_stages)
        if samples is None:
            log('第 %d 轮没有样本，停' % r)
            return
        train_round(out, r, samples, args.epochs, args.lr, args.hidden, args.seed,
                    pick_train_device())
        C.atomic_json_dump({'round': r, 'done': True, 'stages': [s[0] for s in stages]},
                           os.path.join(out, 'round%d_done.json' % r))

    if args.no_eval:
        log('--no_eval：不做最终评测。总用时 %.1f 分钟' % ((time.time() - t_start) / 60.0))
        return
    log('全部轮次完成，开始评测')
    run_eval(out, args.tag, os.path.join(out, 'headA_r%d.pth' % args.rounds),
             os.path.join(out, 'headB_r%d.pth' % args.rounds), args.budget,
             args.eval_workers, args.seed, anchor, args.test30, args.esicup)
    log('ALL DONE  总用时 %.1f 分钟' % ((time.time() - t_start) / 60.0))


if __name__ == '__main__':
    main()
