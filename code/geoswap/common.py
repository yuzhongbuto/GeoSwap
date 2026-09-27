"""rlfine 公共工具（路径解析 / 模型构建 / 原子保存 / 评测原语）。

统一约定（与论文"修复版"协议一致，训练/评测/部署三处必须相同）：
  * 特征：PackingEnvV2(feature_scale='maxnorm')，128 维，第 0 维 = area/max(area)
  * 不设 PACKING_NO_AREA
  * state -> logits：tgt = 已放置零件的**原索引**序列；t=0 时 tgt=[[START_IDX]]；
    取 logits[0, -1, :n_rem]；动作 = 该零件在"剩余列表"中的位置（与 env.step 一致）
"""
import os
import sys
import json
import time
import random
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

DEFAULT_ANCHOR = os.path.join('checkpoints', 'sup128_v8', 'anchor_sup128.pth')
AUX_TAGS = ('_feat', '_order', '_angle', 'summary', 'best', 'experiment')


# ---------------------------------------------------------------- 环境
def prep_env():
    """统一运行时环境。必须在 import torch 之前调用（OMP 线程数）。"""
    os.environ['PACKING_FEATURE_SCALE'] = 'maxnorm'
    os.environ.pop('PACKING_NO_AREA', None)
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')


def utf8_stdout():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(name='auto'):
    import torch
    if name == 'cpu':
        return 'cpu'
    if name == 'cuda':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return 'cuda' if torch.cuda.is_available() else 'cpu'


# ---------------------------------------------------------------- 路径
def list_instances(d):
    if not d or not os.path.isdir(d):
        return []
    out = []
    for f in sorted(os.listdir(d)):
        if not f.endswith('.txt'):
            continue
        if any(t in f for t in AUX_TAGS):
            continue
        out.append(os.path.join(d, f))
    return out


def resolve_dir(cli, env_key, candidates):
    cands = []
    if cli:
        cands.append(cli)
    if os.environ.get(env_key):
        cands.append(os.environ[env_key])
    cands.extend(candidates)
    for c in cands:
        if c and os.path.isdir(c) and list_instances(c):
            return os.path.abspath(c)
    return None


def test30_dir(cli=None):
    """30 实例主 benchmark。原脚本把它写死在项目外，这里允许覆盖。"""
    return resolve_dir(cli, 'PACKING_TEST30', [
        os.path.join(REPO, 'test_set'),
        os.path.join(REPO, 'datasets', 'test30'),
        os.path.join(REPO, 'data', 'test30'),
        r'D:\Transformer\实验集合7',
    ])


def esicup_dir(cli=None):
    return resolve_dir(cli, 'PACKING_ESICUP', [os.path.join(REPO, 'data', 'public_benchmark')])


def train_dir(cli=None):
    return resolve_dir(cli, 'PACKING_TRAIN', [os.path.join(REPO, 'data', 'generated', 'train')])


def out_dir(p):
    p = p or os.path.join(REPO, 'rlfine', 'results', 'default')
    p = p if os.path.isabs(p) else os.path.join(REPO, p)
    os.makedirs(p, exist_ok=True)
    return p


# ---------------------------------------------------------------- 保存
def atomic_json_dump(obj, path):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix='.tmp')
    os.close(fd)
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def atomic_torch_save(obj, path):
    import torch
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix='.tmp')
    os.close(fd)
    torch.save(obj, tmp)
    os.replace(tmp, path)


def append_jsonl(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')


def load_json(path, default=None):
    """读 JSON。用 utf-8-sig 容忍 Windows 工具写入的 BOM（否则 json.load 会直接崩）。"""
    if not os.path.exists(path):
        return default
    with open(path, encoding='utf-8-sig') as f:
        return json.load(f)


def torch_load(path, map_location='cpu'):
    import torch
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


# ---------------------------------------------------------------- 模型
def get_actor_sd(ckpt):
    """兼容三种格式：anchor {'actor_sd':..} / 训练断点 / 裸 state_dict。"""
    if isinstance(ckpt, dict) and 'actor_sd' in ckpt:
        return ckpt['actor_sd']
    if isinstance(ckpt, dict) and 'model_sd' in ckpt:
        return ckpt['model_sd']
    if isinstance(ckpt, dict) and 'input_proj.weight' in ckpt:
        return ckpt
    raise KeyError('checkpoint 里找不到 actor state_dict（既无 actor_sd 也无 input_proj.weight）')


def build_actor(sd, device='cpu'):
    """按 state_dict 自动判断 pointer，构建 PPOActor 并加载。返回 (actor, loaded_keys)。"""
    from ranksteer.ranksteer_train import PPOActor
    has_ptr = 'pointer_q.weight' in sd
    actor = PPOActor(n_residual=1, use_pointer=has_ptr)
    own = actor.state_dict()
    filt = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    actor.load_state_dict(filt, strict=False)
    actor.eval()
    return actor.to(device), filt


def env_for(path, feature_scale='maxnorm'):
    from training.rl_env_v2 import PackingEnvV2
    return PackingEnvV2(path, placement_mode='blf', feature_scale=feature_scale)


# ---------------------------------------------------------------- 评测原语
def greedy_episode(env, actor, device='cpu', mode='pure', w_area=0.5, temperature=0.0,
                   explore='softmax', eps=0.2, topk=3):
    """跑一条 episode。

    mode='pure'   -> logits 直接用骨干输出
    mode='resid'  -> logits + w_area * area_norm（论文头条口径）
    temperature=0 -> 纯 argmax（贪心）
    explore='softmax' : 按 softmax(logits/T) 全分布采样（加热式，探索范围大、代价高）
    explore='eps_topk': 以 eps 概率在 top-k 里采样、否则 argmax
                        （针对性探索：轨迹贴着贪心策略，利用率不塌，同时产生可比较的差异）
    返回 (utilization, order_of_original_indices)
    """
    import torch
    from config import MAX_SEQ_LEN, LOGIT_CLAMP
    state = env.reset()
    done = False
    order = []
    with torch.no_grad():
        while not done:
            rem_feat, tgt, rem_res, sky, af, ldx = state
            nr = rem_feat.size(0)
            rft = rem_feat.unsqueeze(0).to(device)
            tgt_t = tgt.unsqueeze(0).to(device) if tgt.numel() > 0 else \
                torch.tensor([[MAX_SEQ_LEN]], dtype=torch.long, device=device)
            ol, dec = actor(rft, tgt_t)
            lo = ol[0, -1, :nr]
            lo = torch.nan_to_num(lo, nan=0.0, posinf=LOGIT_CLAMP, neginf=-LOGIT_CLAMP)
            lo = lo.clamp(-LOGIT_CLAMP, LOGIT_CLAMP)
            if mode == 'resid':
                lo = lo + w_area * rem_res[:, 0].to(device)
            if temperature and temperature > 0 and explore == 'eps_topk':
                if random.random() < eps:
                    k = min(max(int(topk), 2), lo.numel())
                    top = torch.topk(lo, k)
                    j = int(torch.distributions.Categorical(
                        torch.softmax(top.values, dim=-1)).sample().item())
                    a = int(top.indices[j].item())
                else:
                    a = int(torch.argmax(lo).item())
            elif temperature and temperature > 0:
                a = int(torch.distributions.Categorical(
                    torch.softmax(lo / temperature, dim=-1)).sample().item())
            else:
                a = int(torch.argmax(lo).item())
            order.append(int(env.remaining_indices[a]))
            state, r, done, info = env.step(a)
    return float(info.get('utilization', 0.0)), order


def rollout_given_order(env, order):
    """按给定的原索引顺序喂给环境（用于面积降序基线 / LNS 解）。

    返回 (utilization, n_steps)。顺序不完整时返回当前 util（不强行归零）。
    """
    env.reset()
    done = False
    info = {}
    steps = 0
    for idx in order:
        if idx not in env.remaining_indices:
            continue
        pos = env.remaining_indices.index(idx)
        state, r, done, info = env.step(pos)
        steps += 1
        if done:
            break
    return float(info.get('utilization', 0.0)), steps


def area_order(env):
    import numpy as np
    return [int(i) for i in np.argsort(-env.residual_raw[:, 0].astype(float))]


def spearman(a, b):
    """无 scipy 依赖的 Spearman（秩相关）。"""
    import numpy as np
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2:
        return float('nan')
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = (ra.std() * rb.std())
    return float((ra * rb).mean() / den) if den > 0 else float('nan')


def order_rho(order, env):
    """模型/方法给出的顺序 与 面积降序 的 Spearman ρ。"""
    import numpy as np
    n = env.n
    seq_rank = np.full(n, -1.0)
    for pos, idx in enumerate(order):
        if 0 <= idx < n:
            seq_rank[idx] = pos
    if (seq_rank < 0).any():
        return float('nan')
    area_rank = np.argsort(np.argsort(-env.residual_raw[:, 0].astype(float)))
    return spearman(seq_rank, area_rank)


def fmt(x, nd=4):
    try:
        return ('%.' + str(nd) + 'f') % float(x)
    except Exception:
        return str(x)


def now():
    return time.strftime('%Y-%m-%d %H:%M:%S')
