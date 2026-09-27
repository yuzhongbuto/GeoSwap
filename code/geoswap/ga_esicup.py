"""ga_esicup.py -- ESICUP 上补跑遗传算法（带精英保留 + 逐代 best-so-far）。

为什么要重跑
------------
原实现 ranksteer/benchmark_ga.py::order_ga 有 2026-09-26 查出的两个问题：
  1. **没有精英保留**：每一代整个替换成子代，父代全部丢弃，最后只从末代里挑最好的。
     后果 (a) 多跑几代不保证更好；(b) 白白浪费评估（弱基线）。
  2. **只报最后一代**：拿不到质量-预算曲线，而审稿意见（AE report）明确要求
     "quality-time tradeoff analysis ... under comparable computational budgets"。

本脚本的改动
------------
  * 精英保留：当前最优个体直接进下一代（保证 best-so-far 单调不降）。
  * 逐代记录 best-so-far：一次 GA@(pop x (1+gens)) 的跑，同时产出
    10 / 20 / 30 / ... / 110 次评估的整条曲线。
  * 其余（面积降序起步、交换变异 / 段交叉、BLF 利用率当适应度、pop=10 gens=10 seed=7）
    与原实现保持一致，以便与既有 GA 数字对照。

产物
----
    results/ga_esicup/
        curve.jsonl     每实例每代一行：instance / evals / best_so_far / gen / seconds
        search.jsonl    每实例一行：最终利用率、序列、与面积序的 rho、耗时
        summary.json    汇总：各预算点的均值、vs 面积序、vs 我们的方法
"""
import argparse
import json
import math
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
os.environ.setdefault('PACKING_FAST_BLF', '1')      # 向量化 BLF：快 6.4x，利用率逐位一致（已验）

import geoswap_rounds as G  # noqa: E402


def log(msg):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), msg), flush=True)


def eval_seq(env, seq):
    """按给定顺序跑一遍 BLF，返回利用率。一次评估 = 一次 eval_seq。"""
    u, _ = C.rollout_given_order(env, seq)
    return float(u)


def order_ga_elite(env, pop=10, gens=10, seed=7, on_record=None):
    """GA + 精英保留 + 逐代 best-so-far。

    返回 (best_seq, best_util, curve)，curve 是 [(evals, best_so_far), ...]。
    """
    rng = random.Random(seed)
    n = env.n
    areas = env.residual_raw[:, 0]
    base = [int(x) for x in np.argsort(-areas)]

    pop_seqs = [base[:]]
    for _ in range(pop - 1):
        s = base[:]
        for _ in range(rng.randint(1, 4)):
            i, j = rng.randrange(n), rng.randrange(n)
            s[i], s[j] = s[j], s[i]
        pop_seqs.append(s)

    t0 = time.time()
    fit = [eval_seq(env, s) for s in pop_seqs]
    evals = pop
    bi = int(np.argmax(fit))
    best_seq, best_u = pop_seqs[bi][:], float(fit[bi])
    curve = [(evals, best_u)]
    if on_record:
        on_record(evals, best_u, time.time() - t0)

    for g in range(gens):
        order = sorted(range(pop), key=lambda i: fit[i], reverse=True)
        new_pop = [pop_seqs[order[0]][:]]                 # <<< 精英保留：最优父代直接进下一代
        while len(new_pop) < pop:
            p1 = pop_seqs[order[rng.randrange(max(1, pop // 2))]]
            p2 = pop_seqs[order[rng.randrange(max(1, pop // 2))]]
            if rng.random() < 0.5:
                child = p1[:]
                i, j = sorted(rng.sample(range(n), 2))
                child[i:j] = p2[i:j]
                from collections import deque
                missing = deque(x for x in p1 if x not in child)
                seen = set()
                for k in range(n):
                    if child[k] in seen and missing:
                        child[k] = missing.popleft()
                    seen.add(child[k])
            else:
                child = p1[:]
                i, j = rng.randrange(n), rng.randrange(n)
                child[i], child[j] = child[j], child[i]
            new_pop.append(child)
        pop_seqs = new_pop
        fit = [eval_seq(env, s) for s in pop_seqs]
        evals += pop
        bi = int(np.argmax(fit))
        if fit[bi] > best_u:                               # 因为精英保留，这里必然只增不减
            best_u, best_seq = float(fit[bi]), pop_seqs[bi][:]
        curve.append((evals, best_u))
        if on_record:
            on_record(evals, best_u, time.time() - t0)

    return best_seq, best_u, curve


def ga_task(task):
    import torch
    torch.set_num_threads(1)
    import common as C2

    inst_path, pop, gens, seed = task
    name = os.path.basename(inst_path)
    try:
        t0 = time.time()
        env = C2.env_for(inst_path)
        h_util, _ = C2.rollout_given_order(env, [int(x) for x in np.argsort(-env.residual_raw[:, 0])])
        rows = []

        def rec(evals, u, el):
            rows.append((evals, float(u), el))

        seq, u, curve = order_ga_elite(env, pop=pop, gens=gens, seed=seed, on_record=rec)
        return {'ok': True, 'instance': name, 'n': int(env.n),
                'heuristic': float(h_util), 'ga': float(u), 'ga_seq': [int(x) for x in seq],
                'rho': C2.order_rho(seq, env), 'curve': curve, 'rows': rows,
                'seconds': time.time() - t0}
    except Exception:
        import traceback
        return {'ok': False, 'instance': name, 'error': traceback.format_exc()[-500:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='ga_esicup')
    ap.add_argument('--pop', type=int, default=10)
    ap.add_argument('--gens', type=int, default=10)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--esicup', default=None)
    a = ap.parse_args()

    d = C.esicup_dir(a.esicup)
    insts = C.list_instances(d)
    out = C.out_dir(os.path.join('rlfine', 'results', a.tag))
    log('=' * 78)
    log('GA(带精英保留)  ESICUP %d 个实例  pop=%d gens=%d seed=%d  每实例评估 %d 次'
        % (len(insts), a.pop, a.gens, a.seed, a.pop * (1 + a.gens)))
    log('输出 %s' % out)
    log('实例集 %s' % d)

    tasks = [(p, a.pop, a.gens, a.seed) for p in insts]
    cp = os.path.join(out, 'curve.jsonl')
    sp = os.path.join(out, 'search.jsonl')
    for f in (cp, sp):
        if os.path.exists(f):
            os.remove(f)
    st = {'ok': 0, 'fail': 0}
    t0 = time.time()

    def _on(r, k, ntot):
        if not r.get('ok'):
            st['fail'] += 1
            log('FAIL %s :: %s' % (r.get('instance'), r.get('error', '')[-200:]))
            return
        st['ok'] += 1
        for ev, u, el in r['rows']:
            C.append_jsonl(cp, {'instance': r['instance'], 'n': r['n'], 'evals': ev,
                                'best_so_far': u, 'seconds': el})
        C.append_jsonl(sp, {'instance': r['instance'], 'n': r['n'],
                            'heuristic': r['heuristic'], 'ga': r['ga'],
                            'rho': r['rho'], 'seconds': r['seconds'],
                            'ga_seq': r['ga_seq']})
        log('  %-14s n=%-3d  面积序 %.4f -> GA %.4f (%+.4f)  %.1f 分钟'
            % (r['instance'], r['n'], r['heuristic'], r['ga'], r['ga'] - r['heuristic'],
               r['seconds'] / 60.0))

    G.run_sharded(tasks, ga_task, a.workers, _on)
    el = time.time() - t0

    # 汇总：各预算点的均值
    per = {}
    if os.path.exists(cp):
        for line in open(cp, encoding='utf-8'):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            per.setdefault(r['evals'], {})[r['instance']] = r['best_so_far']
    ref_area = 0.7349
    summary = {'instances': st['ok'], 'failed': st['fail'], 'seconds': el,
               'pop': a.pop, 'gens': a.gens, 'seed': a.seed, 'budget_curve': {}}
    log('-' * 78)
    log('%-8s %12s %14s' % ('评估次数', 'GA 均值', 'vs 面积序'))
    for ev in sorted(per):
        vals = list(per[ev].values())
        m = float(np.mean(vals))
        summary['budget_curve'][ev] = {'mean': m, 'n': len(vals),
                                       'vs_area': m - ref_area}
        log('%-8d %12.4f %+13.4f' % (ev, m, m - ref_area))
    C.atomic_json_dump(summary, os.path.join(out, 'summary.json'))
    log('总用时 %.1f 分钟   存到 %s' % (el / 60.0, out))


if __name__ == '__main__':
    main()
