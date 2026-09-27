"""监督预训练 — 单头 128 维 PPOActor 骨干（2026-08-15 本阶段，恢复旧论文监督配方）

为什么做（见 results/research_direction_analysis.md 与 experiment_log.md 本阶段章节）:
  Probe A 证明当前管线骨干（课程学习 RL）ρ≈0，从未学会排序；旧论文证明
  真·监督预训练（CE on GA/面积降序标签）能让 Transformer 骨干学会排序（+3.16pp vs 启发式）。
  本脚本用**单头 128 维 PPOActor**（不是旧论文的双头 131 维 PackingActor）做监督预训练：
  - 特征: extract_shape_vector(normalize=False)（保留尺寸）+ 实例级 z-score（与 rl_env_v2 一致）
  - 标签: 面积降序索引序列（后续可换 GA 标签升级）
  - 训练: teacher forcing CE，per-position mask 已选零件（贴近推理分布）
  - 输出: 直接是 anchor 格式 {'actor_sd', 'residual_weights'}，供 ppo_train / ranksteer_train 微调

用法:
  python ranksteer/supervised_pretrain.py [--epochs 90] [--batch_size 32]
                                          [--out_dir checkpoints/sup128]
输出:
  checkpoints/sup128/supervised_best.pth   骨干 actor state_dict
  checkpoints/sup128/anchor_sup128.pth    anchor 格式（PPO 直接加载）
  checkpoints/sup128/supervised_checkpoint.pth  断点续训
  results/sup128/supervised_training_log.csv
"""
import os, sys, time, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from config import (
    D_FEAT, D_MODEL, NHEAD, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS,
    DIM_FEEDFORWARD, MAX_SEQ_LEN, DEVICE, LR_SUPERVISED, ZSCORE_MIN_STD,
    MODEL_DIR, RESULT_DIR, GENERATED_DIR,
)
from geometry import extract_shape_vector, polygon_area
from data.preprocess import parse_instance_file
from ranksteer.ranksteer_train import PPOActor  # fork 版（单头 128 维，无 steer 时与原版一致）

PAD_IDX = MAX_SEQ_LEN + 1
START_IDX = MAX_SEQ_LEN
INSTANCE_DIR = os.path.join(GENERATED_DIR, 'train')


def list_instances(data_dir):
    files = sorted(os.path.join(data_dir, f) for f in os.listdir(data_dir)
                   if f.endswith('.txt') and '_feat' not in f
                   and '_order' not in f and '_angle' not in f
                   and 'summary' not in f and 'best' not in f
                   and 'experiment' not in f)
    return files


def extract_instance_data(filepath, feature_scale='maxnorm', no_area=False):
    """与 rl_env_v2（feature_scale=maxnorm）完全一致的协议（LayerNorm 穿透）：
    形状 = per-part normalize（与面积解耦，LN 安全）+ 面积列 area/max(area) 放第 0 维。
    返回 (feats[n,128], order 面积降序索引, n)

    no_area=True（A5 消融①）：不再把第 0 维覆盖为显式面积列，第 0 维保持
    per-part 归一化的形状值 → 即"128 维纯归一化特征"。评测端由环境变量
    PACKING_NO_AREA=1 同步（rl_env_v2 读同一开关），保证训练/评测口径一致。
    """
    parts, plate_w, plate_h = parse_instance_file(filepath)
    areas = np.array([polygon_area(v) for v in parts])
    if feature_scale == 'maxnorm':
        vecs = np.array([extract_shape_vector(v, D_FEAT, normalize=True)
                         for v in parts], dtype=np.float32)
        a_max = max(areas.max(), 1e-6)
        if not no_area:
            vecs[:, 0] = areas / a_max
        feats = vecs
    else:  # zscore（原文协议）
        vecs = np.array([extract_shape_vector(v, D_FEAT, normalize=False)
                         for v in parts], dtype=np.float32)
        mean = vecs.mean(axis=0, keepdims=True)
        std = vecs.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        feats = (vecs - mean) / std
    order = np.argsort(-areas).astype(np.int64)  # 面积降序
    return feats, order


def build_batch(batch_data, device):
    """batch_data: list of (src_feats[n_rem,128], prefix(list), label_pos, n_rem)
    逐步样本为静态决策：tgt 恒为 [START]，label 在位置 0（=剩余中最大）。"""
    max_n = max(d[3] for d in batch_data)
    B = len(batch_data)
    src = torch.zeros(B, max_n, D_FEAT)
    tgt = torch.full((B, 1), START_IDX, dtype=torch.long)
    label = torch.full((B, 1), -100, dtype=torch.long)
    src_mask = torch.zeros(B, max_n, dtype=torch.bool)
    for b, (feats, prefix, label_pos, n_rem) in enumerate(batch_data):
        src[b, :n_rem] = torch.tensor(feats)
        src_mask[b, n_rem:] = True
        label[b, 0] = label_pos  # 从剩余中选面积最大者
    return (src.to(device), tgt.to(device), label.to(device),
            src_mask.to(device), None, max_n)


def build_logits_mask(orders, n_classes, device, feat_counts=None):
    """per-position mask：位置 t 的已选零件（order[:t]）与无效索引（>=零件数）的 logits 置 -inf。

    orders: [B, max_n] long（-100 为 pad）
    n_classes: logits 最后一维（pointer 模式 = max_n；原模式 = MAX_SEQ_LEN+2）
    feat_counts: [B] 每个样本的零件数（src 有效行数；None 时用 label 有效数近似）
    """
    B, T = orders.size()
    mask = torch.zeros(B, T, n_classes, dtype=torch.bool, device=device)
    for b in range(B):
        if feat_counts is not None:
            n_feat = int(feat_counts[b])
        else:
            n_feat = int((orders[b] >= 0).sum())
        if n_feat < n_classes:
            mask[b, :, n_feat:] = True  # 无效索引
        for t in range(T):
            if t == 0:
                continue
            prev = orders[b, :t]
            prev = prev[prev >= 0]
            if len(prev):
                mask[b, t, prev] = True  # 已选零件
    return mask


def build_seq_samples(feats, order, n, k_per_instance=10, seed=0):
    """逐步剩余样本（2026-08-15 本阶段，与部署协议 100% 一致）：
    对每个实例随机采 k 个位置 t，样本 = (剩余零件特征[原始顺序], 已选前缀, 剩余中最大位置)。
    分裂测试证明：全部输入协议 ρ=0.89 vs 剩余输入协议 ρ=0.29（第二步起乱）——
    必须用"剩余输入 + 前缀"训练，部署（env/RL worker）才能正确。
    返回 list of (src_feats[n_rem,128], prefix(list), label_pos, n_rem)
    """
    samples = []
    rng = np.random.RandomState(seed)
    all_idx = list(range(n))
    for _ in range(k_per_instance):
        t = int(rng.randint(0, max(n - 1, 1)))  # 至少剩 2 个零件
        picked = set(order[:t])
        remaining = [i for i in all_idx if i not in picked]
        pos_of = {int(i): p for p, i in enumerate(remaining)}
        label_pos = pos_of[int(order[t])]
        samples.append((feats[remaining], list(map(int, order[:t])),
                        label_pos, len(remaining)))
    return samples


def build_full_samples(feats, order, n, k_per_instance=10, seed=0):
    """旧协议（全实例状态）——A5 消融③ 专用。

    每个样本都是"全部零件在场"的特征矩阵（prefix 为空），标签 = 面积最大者
    的位置。部署时环境给的是"剩余零件"，因此状态分布不匹配（这就是被诊断的
    瓶颈三）。k_per_instance 份内容相同的样本：保持样本数与修复版一致，使
    "只换协议"成为唯一变量（seed 参数保留是为了签名一致，未使用）。
    返回 list of (src_feats[n,128], prefix(list), label_pos, n_rem)
    """
    label_pos = int(order[0])          # 全部零件在场时，位置 == 原始索引
    return [(feats, [], label_pos, n) for _ in range(max(1, k_per_instance))]


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=90)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=LR_SUPERVISED)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--no_mask', action='store_true',
                        help='不用 per-position 已选 mask（旧论文全 vocab CE 做法）')
    parser.add_argument('--fixed_lr', action='store_true',
                        help='固定 LR（不用 ReduceLROnPlateau，防止 LR 过早衰减）')
    parser.add_argument('--subset_aug', type=int, default=0,
                        help='每实例生成的子集增强样本数（0=不启用）')
    parser.add_argument('--xavier', action='store_true',
                        help='xavier 初始化（旧 PackingActor 有，PPOActor 缺失）')
    parser.add_argument('--pointer', action='store_true',
                        help='Pointer 选择机制（逐零件打分，per-token 实验 96.9%% 可学）')
    parser.add_argument('--labels', type=str, default=None,
                        help='外部改进标签 json（{basename: [order...]}，路径 A）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（多种子统计用）')
    parser.add_argument('--out_dir', type=str,
                        default=os.path.join(MODEL_DIR, 'sup128'))
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--no_area', action='store_true',
                        help='A5 消融①：去掉显式面积列（第 0 维保持形状值）；'
                             '同时置 PACKING_NO_AREA=1 使评测端口径一致')
    parser.add_argument('--full_protocol', action='store_true',
                        help='A5 消融③：训练状态改回"全实例"（旧协议），'
                             '样本数仍为 subset_aug 份以隔离协议这一个变量')
    args = parser.parse_args()

    if args.no_area:
        os.environ['PACKING_NO_AREA'] = '1'

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(RESULT_DIR, 'sup128'), exist_ok=True)

    print(f'监督预训练（单头 128 维 PPOActor, normalize=False + maxnorm 保留尺寸）')
    print(f'  epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}, '
          f'patience={args.patience}, subset_aug={args.subset_aug}, '
          f'xavier={args.xavier}')
    print(f'  数据: {INSTANCE_DIR}')

    # ---- 1. 数据（逐步剩余样本：与部署协议 100% 一致）----
    files = list_instances(INSTANCE_DIR)
    print(f'  实例文件: {len(files)}')
    ext_labels = None
    if args.labels:
        import json as _json
        with open(args.labels, encoding='utf-8') as f:
            ext_labels = _json.load(f)
        print(f'  外部标签: {len(ext_labels)}（路径 A 改进标签）')
    data = []
    for fi, fp in enumerate(files):
        try:
            feats, order = extract_instance_data(fp, no_area=args.no_area)
        except Exception as e:
            print(f'  skip {os.path.basename(fp)}: {e}')
            continue
        n = len(order)
        if ext_labels is not None:
            bn = os.path.basename(fp)
            if bn in ext_labels:
                o = list(ext_labels[bn])
                if len(o) == n and sorted(o) == list(range(n)):
                    order = np.array(o, dtype=np.int64)
                else:
                    print(f'  [warn] label invalid for {bn}, keep area-sort')
            else:
                print(f'  [warn] no label for {bn}, keep area-sort')
        builder = build_full_samples if args.full_protocol else build_seq_samples
        data.extend(builder(feats, order, n,
                            k_per_instance=args.subset_aug,
                            seed=fi * 1000 + args.seed))
    proto = '全实例（旧协议，消融③）' if args.full_protocol else '逐步剩余（部署协议）'
    area_tag = 'NO-AREA（消融①）' if args.no_area else '含面积列'
    print(f'  可用样本: {len(data)}（{proto} / {area_tag}）')
    split = int(len(data) * 0.85)
    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(len(data))
    train_data = [data[i] for i in idx[:split]]
    val_data = [data[i] for i in idx[split:]]
    print(f'  train={len(train_data)}, val={len(val_data)}, seed={args.seed}')

    # ---- 2. 模型 ----
    actor = PPOActor(n_residual=1, use_area_encoding=False,
                     use_rank_bias=False, use_rank_steer=False,
                     use_pointer=args.pointer).to(DEVICE)
    if args.xavier:
        for p in actor.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        print('  xavier 初始化已应用')
    optimizer = optim.Adam(actor.parameters(), lr=args.lr)
    scheduler = None
    if not args.fixed_lr:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                         factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    start_epoch = 1
    best_val_loss = float('inf')
    patience_counter = 0
    if args.resume_from and os.path.exists(args.resume_from):
        ck = torch.load(args.resume_from, map_location=DEVICE, weights_only=False)
        actor.load_state_dict(ck['actor_sd'])
        optimizer.load_state_dict(ck['optimizer_sd'])
        if scheduler is not None and 'scheduler_sd' in ck:
            scheduler.load_state_dict(ck['scheduler_sd'])
        start_epoch = ck['epoch'] + 1
        best_val_loss = ck['best_val_loss']
        patience_counter = ck['patience_counter']
        print(f'  从断点恢复: epoch {start_epoch}')

    ckpt_path = os.path.join(out_dir, 'supervised_checkpoint.pth')
    best_path = os.path.join(out_dir, 'supervised_best.pth')
    anchor_path = os.path.join(out_dir, 'anchor_sup128.pth')
    log_path = os.path.join(RESULT_DIR, 'sup128', 'supervised_training_log.csv')
    if start_epoch == 1 or not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write('epoch,train_loss,val_loss,val_acc,lr\n')

    def forward_loss(logits, label, max_n, feat_counts):
        """logits [B, 1, nc]（pointer: nc=剩余数），label [B, 1]（位置 0 = 剩余中最大）。"""
        nc = logits.size(-1)
        logits_v = logits
        if not args.no_mask:
            lm = build_logits_mask(label, nc, DEVICE, feat_counts)
            logits_v = logits_v.masked_fill(lm, float('-inf'))
        return criterion(logits_v.reshape(-1, nc), label.reshape(-1))

    def batch_acc(logits, label, max_n, feat_counts):
        nc = logits.size(-1)
        logits_v = logits
        if not args.no_mask:
            lm = build_logits_mask(label, nc, DEVICE, feat_counts)
            logits_v = logits_v.masked_fill(lm, float('-inf'))
        pred = logits_v.argmax(-1)
        ok = (pred == label) & (label >= 0)
        return ok.sum().item(), (label >= 0).sum().item()

    def evaluate():
        actor.eval()
        losses, accs, cnts = [], [], []
        with torch.no_grad():
            for i in range(0, len(val_data), args.batch_size):
                batch = val_data[i:i + args.batch_size]
                src, tgt, label, sm, tm, max_n = build_batch(batch, DEVICE)
                fc = (~sm).sum(1)
                mem = actor.encode(src, src_key_padding_mask=sm)
                dec = actor.decode_step(mem, tgt, tgt_key_padding_mask=tm,
                                        memory_key_padding_mask=sm)
                logits = actor.forward_logits(mem, dec)
                losses.append(forward_loss(logits, label, max_n, fc).item())
                a, c = batch_acc(logits, label, max_n, fc)
                accs.append(a)
                cnts.append(c)
        return float(np.mean(losses)), (sum(accs) / sum(cnts) if cnts else 0.0)

    def train_acc_first():
        """剩余输入单步预测 acc（与部署协议一致）：抽样样本，预测剩余中最大。"""
        actor.eval()
        hits, tot = 0, 0
        rng3 = np.random.RandomState(12345)
        sub = rng3.choice(len(train_data), min(128, len(train_data)), replace=False)
        with torch.no_grad():
            for j in sub:
                feats, prefix, label_pos, n_rem = train_data[j]
                src = torch.tensor(feats, device=DEVICE).unsqueeze(0)
                mem = actor.encode(src)
                tgt = torch.tensor([[START_IDX]], dtype=torch.long, device=DEVICE)
                dec = actor.decode_step(mem, tgt)
                lo = actor.forward_logits(mem, dec)[0, -1]
                if lo.size(-1) > n_rem:
                    lo = lo[:n_rem]
                pick = int(torch.argmax(lo).item())
                hits += (pick == int(label_pos))
                tot += 1
        return hits / max(tot, 1), hits / max(tot, 1)

    # ---- 3. 训练循环 ----
    print(f'\n训练 {start_epoch} -> {args.epochs}'
          f'  (no_mask={args.no_mask}, fixed_lr={args.fixed_lr})')
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        actor.train()
        total_loss, n_batch = 0.0, 0
        rng2 = np.random.RandomState(epoch * 1000 + args.seed)
        order_idx = rng2.permutation(len(train_data))
        for i in range(0, len(order_idx), args.batch_size):
            bi = order_idx[i:i + args.batch_size]
            if len(bi) < 2:
                continue
            batch = [train_data[j] for j in bi]
            src, tgt, label, sm, tm, max_n = build_batch(batch, DEVICE)
            fc = (~sm).sum(1)
            mem = actor.encode(src, src_key_padding_mask=sm)
            dec = actor.decode_step(mem, tgt, tgt_key_padding_mask=tm,
                                    memory_key_padding_mask=sm)
            logits = actor.forward_logits(mem, dec)
            loss = forward_loss(logits, label, max_n, fc)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 2.0)
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1

        val_loss, val_acc = evaluate()
        if scheduler is not None:
            scheduler.step(val_loss)
        tl = total_loss / max(n_batch, 1)
        lr_now = optimizer.param_groups[0]['lr']
        extra = ''
        if epoch % 5 == 0 or epoch == 1:
            t_acc, f_hit = train_acc_first()
            extra = f'  train_step_acc={t_acc:.4f} first_is_largest={f_hit:.4f}'
        print(f'E{epoch:3d}/{args.epochs}  train={tl:.4f}  val={val_loss:.4f}  '
              f'val_acc={val_acc:.4f}  lr={lr_now:.2e}  T={time.time()-t0:.0f}s{extra}',
              flush=True)
        with open(log_path, 'a') as f:
            f.write(f'{epoch},{tl:.6f},{val_loss:.6f},{val_acc:.6f},{lr_now:.2e}\n')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(actor.state_dict(), best_path)
            # anchor 格式（residual_weights 初始 W_AREA_INIT=0.5）
            torch.save({'actor_sd': {k: v.cpu() for k, v in actor.state_dict().items()},
                        'residual_weights': torch.tensor([0.5]),
                        'note': 'supervised pretrain (single-head 128d, '
                                'normalize=False, area-desc labels)'},
                       anchor_path)
        else:
            patience_counter += 1

        ck_data = {'epoch': epoch,
                   'actor_sd': actor.state_dict(),
                   'optimizer_sd': optimizer.state_dict(),
                   'best_val_loss': best_val_loss,
                   'patience_counter': patience_counter}
        if scheduler is not None:
            ck_data['scheduler_sd'] = scheduler.state_dict()
        torch.save(ck_data, ckpt_path)

        if patience_counter >= args.patience:
            print(f'  >>> EARLY STOP (patience={args.patience})')
            break

    print(f'\nDONE. best_val_loss={best_val_loss:.4f}')
    print(f'  骨干: {best_path}')
    print(f'  anchor: {anchor_path}')


if __name__ == '__main__':
    train()
