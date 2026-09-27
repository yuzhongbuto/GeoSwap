"""P1: part55/60 LNS 标签双机分片生成（断点续跑）

从 data/generated/train 中按零件规模分片跑 LNS（iters 轮局部搜索），
输出 {basename: [order...]} json，支持断点续跑（已完成实例跳过）。

用法:
  分片 A: python code/train_first_model/gen_labels_subset.py --sizes 055,060 --range 0:210 \
             --workers 8 --iters 15 --out labels_5560_a.json
  分片 B: python code/train_first_model/gen_labels_subset.py --sizes 055,060 --range 210:240 \
             --workers 2 --iters 15 --out labels_5560_b.json

完成后合并:
  python code/train_first_model/gen_labels_subset.py --merge labels_5560_a.json,labels_5560_b.json
  → 写入 data/labels_v2.json（与既有 380 个合并）
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from multiprocessing import Pool

from config import GENERATED_DIR
from data.preprocess import parse_instance_file
from training.rl_env_v2 import PackingEnvV2
from geometry import polygon_area


def eval_seq_util(env, s):
    """按序列放置（drop 协议，env.step 内部处理退化/失败零件），返回 util。"""
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


def lns_one(args):
    """LNS：面积降序起步，随机交换 + 接受 util 提高（与 gen_improved_labels 同逻辑）。"""
    fp, iters, seed = args
    rng = np.random.RandomState(seed)
    parts, w, h = parse_instance_file(fp)
    n = len(parts)
    areas = np.array([polygon_area(v) for v in parts])
    seq = list(np.argsort(-areas))
    env = PackingEnvV2(fp, placement_mode='blf', feature_scale='maxnorm')
    best_util = eval_seq_util(env, seq)
    best_seq = seq[:]
    for it in range(iters):
        cand = best_seq[:]
        k = rng.randint(1, 3)
        for _ in range(k):
            i, j = rng.randint(0, n), rng.randint(0, n)
            cand[i], cand[j] = cand[j], cand[i]
        u = eval_seq_util(env, cand)
        if u > best_util:
            best_util, best_seq = u, cand[:]
    return os.path.basename(fp), [int(x) for x in best_seq], best_util


def list_train_files():
    d = os.path.join(GENERATED_DIR, 'train')
    return sorted(os.path.join(d, f) for f in os.listdir(d)
                  if f.endswith('.txt') and '_feat' not in f
                  and '_order' not in f and '_angle' not in f
                  and 'summary' not in f and 'best' not in f
                  and 'experiment' not in f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sizes', type=str, default='055,060',
                    help='零件规模前缀，逗号分隔（如 055,060）')
    ap.add_argument('--range', type=str, default='0:240',
                    help='每规模排序后取 [start:end]（如 0:210 / 210:240）')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--iters', type=int, default=15)
    ap.add_argument('--out', type=str, default='ranksteer/results/labels_5560_local.json')
    ap.add_argument('--merge', type=str, default=None,
                    help='合并多个 json（逗号分隔）进 labels_v2.json 并退出')
    args = ap.parse_args()

    if args.merge:
        merged = {}
        for p in args.merge.split(','):
            p = p.strip()
            if os.path.exists(p):
                with open(p, encoding='utf-8') as f:
                    d = json.load(f)
                merged.update(d)
                print(f'  merge {p}: {len(d)} labels')
        target = 'ranksteer/results/labels_v2.json'
        if os.path.exists(target):
            with open(target, encoding='utf-8') as f:
                old = json.load(f)
            n_old = len(old)
            old.update(merged)
            merged = old
            print(f'  existing {target}: {n_old} labels')
        with open(target, 'w', encoding='utf-8') as f:
            json.dump(merged, f, ensure_ascii=False)
        print(f'MERGE DONE -> {target}: {len(merged)} labels total')
        return

    sizes = [s.strip() for s in args.sizes.split(',')]
    start, end = (int(x) for x in args.range.split(':'))

    files = [fp for fp in list_train_files()
             if os.path.basename(fp)[4:7] in sizes]
    by_size = {}
    for fp in files:
        by_size.setdefault(os.path.basename(fp)[4:7], []).append(fp)
    todo = []
    for s in sizes:
        sl = sorted(by_size.get(s, []))
        todo.extend(sl[start:end])

    labels = {}
    if os.path.exists(args.out):
        with open(args.out, encoding='utf-8') as f:
            labels = json.load(f)
        print(f'resume: {len(labels)} done')

    todo = [fp for fp in todo if os.path.basename(fp) not in labels]
    print(f'sizes={sizes} range={args.range} -> todo={len(todo)} '
          f'(workers={args.workers}, iters={args.iters})')

    t0 = time.time()
    tasks = [(fp, args.iters, i * 1000 + 7) for i, fp in enumerate(todo)]
    done = len(labels)
    with Pool(args.workers) as pool:
        for name, seq, util in pool.imap_unordered(lns_one, tasks):
            labels[name] = seq
            done += 1
            if done % 10 == 0:
                with open(args.out, 'w', encoding='utf-8') as f:
                    json.dump(labels, f)
                print(f'  {done}  elapsed={time.time()-t0:.0f}s  '
                      f'last util={util:.4f}', flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(labels, f)
    print(f'DONE {len(labels)} labels -> {args.out} ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
