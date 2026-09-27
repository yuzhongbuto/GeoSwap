"""路径 A：改进标签生成（LNS：从面积降序出发，局部搜索优化 util）

为什么：监督标签 = 纯面积降序，信号里没有"形状互补/偏离排序"知识——
模型不可能学出超过启发式的东西。GA/LNS 优化过的排序 ≈ 面积优先 + 互补调整，
监督信号第一次包含"超过面积排序"的知识。

口径：评估用 rl_env_v2 的 BLF 放置器（与训练/评测完全一致）。
用法：
  python ranksteer/gen_improved_labels.py [--iters 80] [--workers 8] [--limit 0]
输出：ranksteer/results/labels_v2.json  {base_name: [order...]}
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from multiprocessing import Pool

from config import GENERATED_DIR
from data.preprocess import parse_instance_file
from training.rl_env_v2 import PackingEnvV2
from ranksteer.supervised_pretrain import list_instances


def eval_sequence(fp, seq, feature_scale='maxnorm'):
    """按给定序列放置，返回 util。"""
    env = PackingEnvV2(fp, placement_mode='blf', feature_scale=feature_scale)
    state = env.reset()
    done = False
    for idx in seq:
        remaining = env.remaining_indices
        pos = remaining.index(idx)
        state, r, done, info = env.step(pos)
        if done and not info.get('success', True):
            return 0.0
    return info.get('utilization', 0.0)


def lns_one(args):
    fp, iters, seed = args
    rng = np.random.RandomState(seed)
    parts, w, h = parse_instance_file(fp)
    n = len(parts)
    from geometry import polygon_area
    areas = np.array([polygon_area(v) for v in parts])
    seq = list(np.argsort(-areas))
    # 复用 env（NFP 缓存），每次候选用 env.reset() 重新模拟
    env = PackingEnvV2(fp, placement_mode='blf', feature_scale='maxnorm')

    def eval_seq(s):
        state = env.reset()
        done = False
        info = {}
        for idx in s:
            remaining = env.remaining_indices
            pos = remaining.index(idx)
            state, r, done, info = env.step(pos)
            if done:
                break
        return info.get('utilization', 0.0)

    best_util = eval_seq(seq)
    best_seq = seq[:]
    for it in range(iters):
        cand = best_seq[:]
        k = rng.randint(1, 3)
        for _ in range(k):
            i, j = rng.randint(0, n), rng.randint(0, n)
            cand[i], cand[j] = cand[j], cand[i]
        u = eval_seq(cand)
        if u > best_util:
            best_util, best_seq = u, cand[:]
    return os.path.basename(fp), [int(x) for x in best_seq], best_util


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=40)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--limit', type=int, default=0,
                        help='只处理前 N 个实例（0=全部，测速用）')
    parser.add_argument('--out', type=str, default='ranksteer/results/labels_v2.json')
    args = parser.parse_args()

    files = list_instances(os.path.join(GENERATED_DIR, 'train'))
    if args.limit:
        files = files[:args.limit]

    # 断点续跑：已完成的实例从输出 json 恢复
    labels = {}
    if os.path.exists(args.out):
        with open(args.out, encoding='utf-8') as f:
            labels = json.load(f)
        print(f'resume: {len(labels)} labels already done')

    todo = [fp for fp in files if os.path.basename(fp) not in labels]
    print(f'todo: {len(todo)} instances (total {len(files)})')

    t0 = time.time()
    tasks = [(fp, args.iters, i * 1000 + 7) for i, fp in enumerate(todo)]
    done = len(labels)
    with Pool(args.workers) as pool:
        for i, (name, seq, util) in enumerate(pool.imap_unordered(lns_one, tasks)):
            labels[name] = seq
            done += 1
            if done % 20 == 0:
                # 周期性保存（崩溃/中断不丢进度）
                with open(args.out, 'w', encoding='utf-8') as f:
                    json.dump(labels, f)
                print(f'  {done}/{len(files)}  elapsed {time.time()-t0:.0f}s', flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(labels, f)
    print(f'DONE {len(labels)} labels -> {args.out} ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
