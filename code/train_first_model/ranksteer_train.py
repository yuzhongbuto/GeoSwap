"""
D1 RankSteer — 最小参数导向（tied 多位置注入）PPO 训练（ppo_train.py 的 fork）

核心思想（2026-08-15 本阶段，详见 results/research_direction_analysis.md §3 D1）:
  论文已证明 policy gradient 的容量定律 ≈ 1-2 个标量参数（A1-A3/S1/F1），
  Rank Bias 在 logits 处有效但做不了序列级规划（logit-only 天花板 4.2pp）。
  RankSteer: 把**同一个** w_area 和 λ（仍只有 2 个可训练参数，tied）同时注入
  编码器输入 token（固定方向 steer_dir，默认 const），让冻结的 self-attention
  有机会"看到"每个零件的面积/排名结构：
      src_emb_i = Linear(centroid_i) + w_area·z_area_i·dir + λ·rank_frac_i·dir
      logits_i  = u_i + w_area·a_norm_i − λ·(1 − rank_i/(N−1))     （原 Rank Bias）
  可训练参数仍 = 2（residual_weights[0] + lambda_raw），满足梯度容量定律。

用法（镜像 5b Step2 配方）:
  python ranksteer/ranksteer_train.py --n_residual 1 --freeze \
      --use_rank_bias --rank_steer const --tag rs_pilot --patience 10 --epochs 30
  --rank_steer: none|const|dim0|dim63|random（注入方向；none=纯 5b 复现）

旧 ppo_train.py 的用法（4b/4c/4d/5a/5b/A3）请仍用 experiments/ppo_train.py。
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys, json, math, random, time, argparse, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import torch, torch.nn as nn, torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import concurrent.futures

from config import *
from models.positional import PositionalEncoding
from training.rl_env_v2 import PackingEnvV2
from models.critic import CriticNetwork


# ==================== 路径 ====================
PPO_DIR = os.path.join(RESULT_DIR, 'ppo')
CKPT_PPO_DIR = os.path.join(MODEL_DIR, 'ppo')
ANCHOR_CKPT = os.path.join(MODEL_DIR, 'step1', 'longest_edge_ratio_checkpoint.pth')
os.makedirs(PPO_DIR, exist_ok=True)
os.makedirs(CKPT_PPO_DIR, exist_ok=True)

# ==================== PPO 超参数 ====================
CLIP_EPSILON = 0.2          # 策略裁剪系数
GAE_LAMBDA = 0.95           # GAE λ
PPO_EPOCHS = 4              # 每批数据 PPO 更新轮数
ENTROPY_COEF = 0.01         # 熵正则系数

# ==================== 训练配置 ====================
UNFREEZE_PREFIX = 'transformer.decoder.layers.2'
RESIDUAL_LR = 5e-4          # 残差权重学习率
UNFREEZE_DECODER_LR = 1e-6  # 解冻层学习率（PPO 下可以比 REINFORCE 激进）
UNFREEZE_WD = 1e-4          # 解冻层权重衰减
CRITIC_LR = 5e-5            # Critic 学习率（PPO 下 critic 需要多训）

# 止损
DEVIATION_SURGE = 5.0
TRAIN_R_DROP = 0.05
W_AREA_FLOOR = 0.5
QUICK_CHECK_EPOCHS = 5
REGULAR_CHECK_EPOCHS = 10

# 1 维残差用的特征列 (area=0)，2 维用 (area=0, longest_edge=5)
FEAT_COLS = {1: [0], 2: [0, 5]}

# ==================== 面积编码 (Size Encoding) ====================
N_AREA_BUCKETS = 32           # 面积分桶数
AREA_SCALE_INIT = 0.1         # alpha 初始值（保守，避免冲乱原有表征）
AREA_ENCODING_LR = 5e-4       # area_scale / area_embedding 学习率


# ==================== 排名偏置 (Rank Bias) ====================
LAMBDA_INIT = 0.0             # lambda_raw 初始值，softplus(0)≈0.69
RANK_BIAS_LR = 5e-4           # lambda_raw 学习率


# ==================== RankSteer（tied 多位置注入） ====================
STEER_MODES = ('none', 'const', 'dim0', 'dim63', 'random')


def _atomic_torch_save(obj, path):
    """原子写入 checkpoint：先写 .tmp 再 os.replace，避免中途被杀留下半截文件。"""
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _seed_all(seed):
    """统一设置 random / numpy / torch 种子（2026-09-08 多种子实验用）。

    背景：原脚本只有 random.seed(42)，而策略 rollout 用 Categorical(...).sample()
    走 torch 默认 RNG（未播种 → 每次运行结果都不同）。多种子统计必须把 torch
    也钉住，否则"种子方差"里混着不可复现的噪声。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_steer_dir(mode, d=D_MODEL, seed=42):
    """生成固定注入方向（不可学习，保证可训练参数仍只有 2 个）。

    const:   全 1 归一化向量（各向同性）
    dim0/dim63: 单位基向量 e_0 / e_63（测试"单维携带面积信号"是否够用）
    random:  固定种子随机单位向量
    """
    if mode == 'const':
        return torch.ones(d) / math.sqrt(d)
    if mode == 'dim0':
        v = torch.zeros(d); v[0] = 1.0; return v
    if mode == 'dim63':
        v = torch.zeros(d); v[63] = 1.0; return v
    if mode == 'random':
        g = torch.Generator().manual_seed(seed)
        v = torch.randn(d, generator=g)
        return v / v.norm()
    raise ValueError(f'unknown steer mode: {mode}')


def compute_rank_penalty(areas, lam):
    """计算排名惩罚：最大零件 penalty≈0，最小 penalty≈λ。

    Args:
        areas: [N] 面积值（z-score 归一化，只保留序关系）
        lam: 标量 λ≥0，惩罚力度

    Returns:
        penalty: [N] tensor，最大值 0，最小值 ≈lam
    """
    if areas.numel() <= 1:
        return torch.zeros(areas.size(0), device=areas.device)
    n = areas.size(0)
    ranks = areas.argsort().argsort().float()  # 0=最小, N-1=最大
    return lam * (1.0 - ranks / max(n - 1, 1))


def compute_area_buckets(areas, n_buckets=N_AREA_BUCKETS):
    """百分位分桶：将面积值映射到 [0, n_buckets-1]。

    桶 0 = 当前剩余零件中面积最小的
    桶 N-1 = 当前剩余零件中面积最大的

    Args:
        areas: [N] 面积值（可以是 z-score 归一化或原始值，只保留序关系）
        n_buckets: 分桶数

    Returns:
        buckets: [N] long tensor
    """
    if areas.numel() <= 1:
        return torch.zeros(areas.size(0), dtype=torch.long, device=areas.device)
    n = areas.size(0)
    ranks = areas.argsort().argsort().float()
    buckets = (ranks / max(n - 1, 1) * (n_buckets - 1)).long().clamp(0, n_buckets - 1)
    return buckets


# ==================== 模型 ====================
class PPOActor(nn.Module):
    """可配置残差维度的 Transformer Actor，支持面积编码 (Size Encoding)。

    面积编码: 将零件相对面积编码为 d_model 维向量，加到 Encoder 输入上，
    使 self-attention 能感知零件大小差异。
    """

    def __init__(self, n_residual=2, use_area_encoding=False,
                 n_area_buckets=N_AREA_BUCKETS, freeze_area_embedding=True,
                 use_rank_bias=False, use_rank_steer=False,
                 rank_steer_mode='const', use_pointer=False):
        super().__init__()
        self.d_model = D_MODEL
        self.max_seq_len = MAX_SEQ_LEN
        self.vocab_size = MAX_SEQ_LEN + 2
        self.pad_idx = MAX_SEQ_LEN + 1
        self.n_residual = n_residual
        self.use_area_encoding = use_area_encoding
        self.use_rank_steer = use_rank_steer
        self.rank_steer_mode = rank_steer_mode
        # ---- Pointer 选择机制（2026-08-15 本阶段）----
        # 决定性实验（_per_token_test.py）：per-token 打分头 96.9% vs 原 decoder
        # 固定投影 ~0%——"从 memory 中选最大"必须逐零件打分。
        # pointer: logits_i = (W_q h_dec) · mem_i（decoder 历史上下文 + 逐零件打分）
        self.use_pointer = use_pointer

        self.input_proj = nn.Linear(D_FEAT, D_MODEL)
        self.token_embedding = nn.Embedding(self.vocab_size, D_MODEL, padding_idx=self.pad_idx)
        self.pos_encoder = PositionalEncoding(D_MODEL, max_len=MAX_SEQ_LEN)
        self.transformer = nn.Transformer(
            d_model=D_MODEL, nhead=NHEAD,
            num_encoder_layers=NUM_ENCODER_LAYERS,
            num_decoder_layers=NUM_DECODER_LAYERS,
            dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT, batch_first=True)
        self.order_head = nn.Linear(D_MODEL, self.vocab_size)
        if use_pointer:
            # pointer query 投影：logits_i = (W_q·h_dec) · mem_i
            self.pointer_q = nn.Linear(D_MODEL, D_MODEL)

        self.use_rank_bias = use_rank_bias
        init_vals = [W_AREA_INIT] + [0.05] * (n_residual - 1)
        self.residual_weights = nn.Parameter(torch.tensor(init_vals))

        # ---- 排名偏置 (Rank Bias) ----
        if use_rank_bias:
            self.lambda_raw = nn.Parameter(torch.tensor(LAMBDA_INIT))

        # ---- RankSteer: 固定注入方向（buffer，不可学习，tied 复用 w_area/λ） ----
        if use_rank_steer:
            self.register_buffer('steer_dir', make_steer_dir(rank_steer_mode))

        # ---- 面积编码 (Size Encoding) ----
        if use_area_encoding:
            self.n_area_buckets = n_area_buckets
            self.area_embedding = nn.Embedding(n_area_buckets, D_MODEL)
            self.area_scale = nn.Parameter(torch.tensor(AREA_SCALE_INIT))
            self.freeze_area_embedding = freeze_area_embedding
            self._init_area_embedding()
            if freeze_area_embedding:
                self.area_embedding.weight.requires_grad = False

    def _init_area_embedding(self):
        """正弦初始化：相邻桶 embedding 相似，大小桶天然可分。

        桶 i 的位置 t = i/(K-1) ∈ [0,1]，embedding[j] = sin/cos(t × freq_j)。
        PPO 不需要从零学"什么是大"，只需要学"多大程度上利用面积信号"。
        """
        with torch.no_grad():
            for i in range(self.n_area_buckets):
                t = i / max(self.n_area_buckets - 1, 1)
                for j in range(self.d_model):
                    freq = (j // 2 + 1) * math.pi
                    val = math.sin(t * freq) if j % 2 == 0 else math.cos(t * freq)
                    self.area_embedding.weight[i, j] = val

    def encode(self, src, src_key_padding_mask=None, area_buckets=None,
               steer_areas=None):
        src_emb = self.input_proj(src) * (self.d_model ** 0.5)
        if self.use_area_encoding and area_buckets is not None:
            src_emb = src_emb + self.area_scale * self.area_embedding(area_buckets)
        if self.use_rank_steer and steer_areas is not None:
            # tied 注入：复用 w_area（residual_weights[0]）与 λ（softplus），
            # 让 self-attention 在输入层就能感知"面积/排名"结构。
            # scale_i = w_area·z_area_i + λ·rank_frac_i,  rank_frac∈[0,1]
            n = steer_areas.size(0)
            z = steer_areas.to(src_emb.device)
            if n > 1:
                ranks = z.argsort().argsort().float()
                rank_frac = ranks / max(n - 1, 1)
            else:
                rank_frac = torch.zeros_like(z)
            lam = self.rank_lambda().to(src_emb.device)
            scale = self.residual_weights[0] * z + lam * rank_frac
            src_emb = src_emb + scale.unsqueeze(-1) * self.steer_dir
        src_emb = torch.clamp(src_emb, -10, 10)
        src_emb = self.pos_encoder(src_emb)
        return self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

    def decode_step(self, memory, tgt, tgt_key_padding_mask=None,
                    memory_key_padding_mask=None):
        tgt_emb = self.token_embedding(tgt) * (self.d_model ** 0.5)
        tgt_emb = torch.clamp(tgt_emb, -10, 10)
        tgt_emb = self.pos_encoder(tgt_emb)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        return self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask,
                                        tgt_key_padding_mask=tgt_key_padding_mask,
                                        memory_key_padding_mask=memory_key_padding_mask)

    def forward_logits(self, mem, dec):
        """统一 logits 入口：pointer = (W_q·h_dec)·mem（逐零件打分）；否则固定投影。"""
        if self.use_pointer:
            q = self.pointer_q(dec)                    # [B, T, D]
            return torch.bmm(q, mem.transpose(1, 2))   # [B, T, n]
        return self.order_head(dec)

    def forward(self, src, tgt, area_buckets=None, steer_areas=None):
        mem = self.encode(src, area_buckets=area_buckets, steer_areas=steer_areas)
        dec = self.decode_step(mem, tgt)
        return self.forward_logits(mem, dec), dec

    @torch.no_grad()
    def get_weights(self):
        w = self.residual_weights.detach().cpu().numpy().tolist()
        if self.use_area_encoding:
            w.append(self.area_scale.item())
        if self.use_rank_bias:
            w.append(self.get_lambda())
        return w

    @torch.no_grad()
    def get_area_scale(self):
        return self.area_scale.item() if self.use_area_encoding else 0.0

    def rank_lambda(self):
        """softplus 包裹，保证 λ≥0。不可导时用 @torch.no_grad()。"""
        return torch.nn.functional.softplus(self.lambda_raw) if self.use_rank_bias else torch.tensor(0.0)

    @torch.no_grad()
    def get_lambda(self):
        """推理时获取 λ 值。"""
        return torch.nn.functional.softplus(self.lambda_raw).item() if self.use_rank_bias else 0.0

    def compute_deviation_norm(self, init_params):
        total = 0.0
        for name, param in self.named_parameters():
            if name in init_params and param.requires_grad:
                total += (param - init_params[name]).norm().item() ** 2
        return total ** 0.5


# ==================== Worker ====================
def ppo_worker(args):
    """
    单实例 rollout。额外存储 log_prob 和 critic_value 用于 PPO。
    自动检测面积编码：若 actor_sd 含 area_embedding 则启用。
    """
    inst_file, actor_sd, critic_sd, feat_cols = args
    has_ae = 'area_embedding.weight' in actor_sd
    has_rb = 'lambda_raw' in actor_sd
    has_rs = 'steer_dir' in actor_sd
    has_ptr = 'pointer_q.weight' in actor_sd
    actor = PPOActor(n_residual=len(feat_cols), use_area_encoding=has_ae,
                     use_rank_bias=has_rb, use_rank_steer=has_rs,
                     use_pointer=has_ptr)
    actor.load_state_dict(actor_sd)
    actor.eval()

    critic = CriticNetwork(d_feat=D_FEAT, d_model=D_MODEL, nhead=NHEAD,
                           num_encoder_layers=NUM_ENCODER_LAYERS,
                           num_decoder_layers=NUM_DECODER_LAYERS,
                           dim_feedforward=DIM_FEEDFORWARD,
                           max_seq_len=MAX_SEQ_LEN)
    critic.load_state_dict(critic_sd)
    critic.eval()

    env = PackingEnvV2(inst_file, placement_mode='blf')
    trajs = []

    with torch.no_grad():
        for _ in range(TRAJECTORIES_PER_INSTANCE):
            state = env.reset()
            sm, am, rm = [], [], []  # state_memories: (feat, tgt, res2, log_prob, value)
            done = False
            while not done:
                rem_feat, tgt, rem_res, sky, af, ldx = state
                nr = rem_feat.size(0)

                res2 = rem_res[:, feat_cols]  # [N, n_residual]

                rft = rem_feat.unsqueeze(0)
                tgt_t = tgt.unsqueeze(0) if tgt.numel() > 0 else \
                        torch.tensor([[MAX_SEQ_LEN]], dtype=torch.long)

                # 面积编码：实例内百分位分桶
                ab = None
                if has_ae:
                    ab = compute_area_buckets(rem_res[:, 0]).unsqueeze(0)  # [1, N]

                sa = rem_res[:, 0] if has_rs else None  # RankSteer: 面积列
                ol, dec = actor(rft, tgt_t, area_buckets=ab, steer_areas=sa)
                smk = torch.zeros(1, rft.size(1), dtype=torch.bool)

                lo = ol[0, -1, :nr]
                lo = torch.clamp(torch.nan_to_num(lo, nan=0, posinf=LOGIT_CLAMP,
                                 neginf=-LOGIT_CLAMP), -LOGIT_CLAMP, LOGIT_CLAMP)
                lo = lo + (res2 * actor.residual_weights.unsqueeze(0)).sum(dim=-1)
                if has_rb:
                    lo = lo - compute_rank_penalty(rem_res[:, 0], actor.rank_lambda())
                lo = lo - lo.max()

                dist = Categorical(torch.softmax(lo, dim=-1))
                oa = dist.sample().item()
                log_prob = dist.log_prob(torch.tensor(oa))

                # Critic value prediction
                value = critic(rft, tgt_t, src_key_padding_mask=smk).item()

                sm.append((
                    rem_feat.clone(),
                    tgt.clone() if tgt.numel() > 0 else tgt,
                    res2.clone(),
                    log_prob.clone(),
                    value,
                ))
                am.append(oa)
                rm.append(torch.tensor(value, dtype=torch.float32))  # placeholder, overwritten by real reward

                ns, r, done, _ = env.step(oa)
                rm[-1] = r  # replace with real reward
                state = ns
            trajs.append((sm, am, rm))
    return trajs


# ==================== GAE ====================
def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """
    GAE advantage 计算。

    Args:
        rewards:   list of float, length T
        values:    list of float, length T (V(s_t) for each step)
        gamma, lam: discount and GAE lambda

    Returns:
        advantages: tensor [T]
        returns:    tensor [T]
    """
    T = len(rewards)
    advantages = torch.zeros(T, dtype=torch.float32)
    gae = 0.0
    for t in reversed(range(T)):
        next_value = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = advantages + torch.tensor(values, dtype=torch.float32)
    return advantages, returns


# ==================== PPO Update ====================
def ppo_update(actor, critic, actor_opt, critic_opt, all_traj,
               feat_cols, ppo_epochs=4, clip_eps=0.2, entropy_coef=0.01,
               gamma=0.99, lam=0.95):
    """
    PPO clipped surrogate objective + critic MSE。
    Critic 比 actor 多训一倍。
    """
    actor.train()
    critic.train()

    # ---- 1. 展开轨迹，提取 advantage ----
    all_sm, all_am, all_rm = [], [], []
    for sm, am, rm in all_traj:
        # sm[i] = (feat, tgt, res2, log_prob, value)
        rewards = torch.tensor(rm, dtype=torch.float32)
        values = torch.tensor([m[4] for m in sm], dtype=torch.float32)
        advantages, returns = compute_gae(rewards.tolist(), values.tolist(), gamma, lam)
        all_sm.append(sm)
        all_am.append(am)
        all_rm.append((advantages, returns))

    # ---- 2. Advantage 归一化 ----
    all_adv = torch.cat([r[0] for r in all_rm])
    adv_mean = all_adv.mean()
    adv_std = all_adv.std() + 1e-8
    norm_adv = [(r[0] - adv_mean) / adv_std for r in all_rm]
    norm_ret = [r[1] for r in all_rm]

    # 打包并 shuffle
    paired = list(zip(all_sm, all_am, norm_adv, norm_ret))
    random.shuffle(paired)

    total_al, total_cl, total_ent, n_updates = 0.0, 0.0, 0.0, 0

    # ---- 3. PPO epochs ----
    critic_updates_per_epoch = 2  # critic 多训一倍
    for ppo_ep in range(ppo_epochs):
        for sm, am, adv, ret in paired:
            n_updates += 1

            # ---- Actor loss (clipped surrogate) ----
            lp_new_list, ent_list = [], []
            for i, (rf, tg, res2, lp_old, _) in enumerate(sm):
                rft = rf.unsqueeze(0).to(DEVICE)
                tgt_t = tg.unsqueeze(0).to(DEVICE) if tg.numel() > 0 else \
                        torch.tensor([[MAX_SEQ_LEN]], dtype=torch.long, device=DEVICE)
                # 面积编码
                ab = None
                if actor.use_area_encoding:
                    ab = compute_area_buckets(res2[:, 0]).unsqueeze(0).to(DEVICE)
                sa = res2[:, 0].to(DEVICE) if actor.use_rank_steer else None
                ol, dec = actor(rft, tgt_t, area_buckets=ab, steer_areas=sa)
                nr2 = rft.size(1)
                lo = ol[0, -1, :nr2]
                lo = torch.clamp(torch.nan_to_num(lo, nan=0, posinf=LOGIT_CLAMP,
                                 neginf=-LOGIT_CLAMP), -LOGIT_CLAMP, LOGIT_CLAMP)
                lo = lo + (res2.to(DEVICE) * actor.residual_weights.unsqueeze(0)).sum(dim=-1)
                if actor.use_rank_bias:
                    lo = lo - compute_rank_penalty(
                        res2[:, 0].to(DEVICE), actor.rank_lambda())
                lo = lo - lo.max().detach()

                dist = Categorical(torch.softmax(lo, dim=-1))
                oa = am[i]
                lp_new = dist.log_prob(torch.tensor(oa, device=DEVICE))
                lp_new_list.append(lp_new)
                ent_list.append(dist.entropy())

            lpn = torch.stack(lp_new_list)  # [T]
            lpo = torch.tensor([m[3] for m in sm], dtype=torch.float32, device=DEVICE)  # [T]
            adv_t = adv.to(DEVICE)
            ent = torch.stack(ent_list).mean()

            ratio = torch.exp(lpn - lpo)
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_t
            actor_loss = -torch.min(surr1, surr2).mean() - entropy_coef * ent

            actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.residual_weights, 2.0)
            torch.nn.utils.clip_grad_norm_(
                [p for n, p in actor.named_parameters()
                 if p.requires_grad and 'residual_weights' not in n],
                1.0
            )
            if actor.use_rank_bias:
                torch.nn.utils.clip_grad_norm_(actor.lambda_raw, 1.0)
            actor_opt.step()

            total_al += actor_loss.item()
            total_ent += ent.item()

            # ---- Critic loss (update 2x more) ----
            for _ in range(critic_updates_per_epoch):
                vs_list = []
                for rf, tg, res2, _, _ in sm:
                    rft = rf.unsqueeze(0).to(DEVICE)
                    tgt_t = tg.unsqueeze(0).to(DEVICE) if tg.numel() > 0 else \
                            torch.tensor([[MAX_SEQ_LEN]], dtype=torch.long, device=DEVICE)
                    smk = torch.zeros(1, rft.size(1), dtype=torch.bool, device=DEVICE)
                    vs_list.append(critic(rft, tgt_t, src_key_padding_mask=smk))

                vt = torch.cat(vs_list)  # [T]
                ret_t = ret.to(DEVICE)
                critic_loss = (vt - ret_t).pow(2).mean()

                critic_opt.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                critic_opt.step()

                total_cl += critic_loss.item()

    n = max(n_updates, 1)
    return {
        'actor_loss': round(total_al / n, 4),
        'critic_loss': round(total_cl / (n * critic_updates_per_epoch), 4),
        'entropy': round(total_ent / n, 4),
    }


# ==================== 工具：加载 PPO 模型 ====================
def load_ppo_actor(ckpt_path, device='cpu'):
    """从 checkpoint 加载 PPOActor，自动检测面积编码。

    Returns:
        actor, feat_cols, residual_weights
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt['actor_sd']
    has_ae = 'area_embedding.weight' in sd
    has_rb = 'lambda_raw' in sd
    n_res = 1 if 'residual_weights' not in ckpt else len(ckpt['residual_weights'])
    feat_cols = FEAT_COLS.get(n_res, [0])

    actor = PPOActor(n_residual=n_res, use_area_encoding=has_ae,
                     use_rank_bias=has_rb)
    actor_sd = actor.state_dict()
    sd_f = {k: v for k, v in sd.items()
            if k in actor_sd and actor_sd[k].shape == v.shape}
    actor.load_state_dict(sd_f, strict=False)

    with torch.no_grad():
        actor.residual_weights.copy_(ckpt['residual_weights'])
        if has_ae and 'area_scale' in ckpt:
            actor.area_scale.copy_(torch.tensor(ckpt['area_scale']))
        if has_rb and 'lambda_raw' in sd:
            actor.lambda_raw.copy_(sd['lambda_raw'])
    actor.eval()
    actor = actor.to(device)
    return actor, feat_cols, ckpt.get('residual_weights')


# ==================== Benchmark ====================
def benchmark_model(actor, feat_cols, test_files):
    device = next(actor.parameters()).device
    results = {}
    for fname in test_files:
        try:
            env = PackingEnvV2(fname, placement_mode='blf')
            n = env.n
            state = env.reset()
            done = False
            t0 = time.time()
            with torch.no_grad():
                while not done:
                    rem_feat, tgt, rem_res, sky, af, ldx = state
                    nr = rem_feat.size(0)
                    res2 = rem_res[:, feat_cols].to(device)
                    rft = rem_feat.unsqueeze(0).to(device)
                    tgt_t = tgt.unsqueeze(0).to(device) if tgt.numel() > 0 else \
                            torch.tensor([[MAX_SEQ_LEN]], dtype=torch.long, device=device)
                    # 面积编码
                    ab = None
                    if actor.use_area_encoding:
                        ab = compute_area_buckets(rem_res[:, 0]).unsqueeze(0).to(device)
                    sa = rem_res[:, 0].to(device) if actor.use_rank_steer else None
                    ol, dec = actor(rft, tgt_t, area_buckets=ab, steer_areas=sa)
                    lo = ol[0, -1, :nr] + \
                         (res2 * actor.residual_weights.unsqueeze(0)).sum(dim=-1)
                    if actor.use_rank_bias:
                        lo = lo - compute_rank_penalty(rem_res[:, 0].to(device),
                                                       actor.get_lambda())
                    oa = torch.argmax(lo).item()
                    state, r, done, info = env.step(oa)
            util = info.get('utilization', 0)
            et = time.time() - t0
            basename = os.path.basename(fname)
            results[basename] = {'util': float(util), 'n': n, 'time_s': et}
            status = 'OK' if util > 0.01 else 'BLF_FAIL'
            print(f"    {basename:<20s} n={n:2d}  util={util:.4f}  [{et:.0f}s] {status}")
        except Exception as e:
            basename = os.path.basename(fname)
            results[basename] = {'util': 0.0, 'n': 0, 'time_s': 0, 'error': str(e)}
            print(f"    {basename:<20s}  ERROR: {e}")

    valid_utils = [v['util'] for v in results.values() if v['util'] > 0.01]
    avg = float(np.mean(valid_utils)) if valid_utils else 0.0
    return {'per_instance': results, 'mean': avg, 'n_valid': len(valid_utils)}


# ==================== 主函数 ====================
def main(epochs=30, n_residual=2, freeze=True, tag=None,
         test_dir=None, data_dir=None, skip_calibration=False,
         use_area_encoding=False, use_rank_bias=False,
         rank_steer='none', use_pointer=False, freeze_w_area=False,
         patience=0, anchor_ckpt=None, critic_ckpt=None, w_area_floor=None,
         reset_early_stop=False, seed=42):
    _seed_all(seed)
    if anchor_ckpt is None:
        anchor_ckpt = ANCHOR_CKPT
    if critic_ckpt is None:
        critic_ckpt = os.path.join(MODEL_DIR, 'phase1_checkpoint.pth')
    feat_cols = FEAT_COLS[n_residual]
    mode = 'freeze' if freeze else 'unfreeze'
    use_rank_steer = rank_steer != 'none'

    print("=" * 65)
    ae_tag = " + AreaEncoding" if use_area_encoding else ""
    rb_tag = " + RankBias" if use_rank_bias else ""
    rs_tag = f" + RankSteer({rank_steer})" if use_rank_steer else ""
    ptr_tag = " + Pointer" if use_pointer else ""
    print(f"PPO Training — Phase: {mode}, n_residual={n_residual}"
          f"{ae_tag}{rb_tag}{rs_tag}{ptr_tag}")
    print("=" * 65)
    print(f"Anchor: {anchor_ckpt}")
    print(f"Residual features: cols {feat_cols}")
    if use_area_encoding:
        print(f"Area Encoding: n_buckets={N_AREA_BUCKETS}, "
              f"scale_init={AREA_SCALE_INIT}, freeze_emb=True")
    if use_rank_bias:
        print(f"Rank Bias: lambda_init={LAMBDA_INIT} "
              f"(softplus(0)≈0.693), freeze_w_area={freeze_w_area}")
    if use_rank_steer:
        print(f"RankSteer: dir={rank_steer}, "
              f"trainable params still = 2 (w_area + lambda, tied)")
    print(f"PPO: clip={CLIP_EPSILON}, ppo_epochs={PPO_EPOCHS}, "
          f"gae_lambda={GAE_LAMBDA}, entropy={ENTROPY_COEF}")
    print(f"Critic LR: {CRITIC_LR}, Residual LR: {RESIDUAL_LR}")
    if not freeze:
        print(f"Unfreeze: {UNFREEZE_PREFIX}, lr={UNFREEZE_DECODER_LR}")
    print()

    _tag_suffix = f'_{tag}' if tag else ''
    _ckpt_name = f'training_checkpoint{_tag_suffix}.pth'
    _log_name = f'training_log{_tag_suffix}.json'

    # ---- 1. 加载锚点 ----
    if not os.path.exists(anchor_ckpt):
        print(f"ERROR: Anchor not found: {anchor_ckpt}")
        sys.exit(1)

    print("1. Loading anchor...")
    anchor = torch.load(anchor_ckpt, map_location='cpu', weights_only=False)
    anchor_sd = anchor['actor_sd']
    anchor_w = anchor['residual_weights']  # [2]: area, edge

    # ---- 2. 构建模型 ----
    print(f"\n2. Building PPOActor (n_residual={n_residual}, "
          f"area_encoding={use_area_encoding}, "
          f"rank_steer={rank_steer}, pointer={use_pointer})...")
    actor = PPOActor(n_residual=n_residual, use_area_encoding=use_area_encoding,
                     use_rank_bias=use_rank_bias,
                     use_rank_steer=use_rank_steer,
                     rank_steer_mode=rank_steer,
                     use_pointer=use_pointer)
    actor_sd = actor.state_dict()
    sd_filtered = {k: v for k, v in anchor_sd.items()
                   if k in actor_sd and actor_sd[k].shape == v.shape}
    actor.load_state_dict(sd_filtered, strict=False)
    print(f"   Loaded {len(sd_filtered)}/{len(anchor_sd)} keys "
          f"(area_embedding uses sinusoidal init)")

    # 残差权重：area 从锚点继承
    with torch.no_grad():
        actor.residual_weights[0] = anchor_w[0].item()
        if n_residual >= 2:
            actor.residual_weights[1] = anchor_w[1].item()
    w_info = actor.get_weights()
    print(f"   Residual weights: {w_info}")

    # ---- 3. 冻结/解冻 ----
    print("\n3. Configuring parameters...")
    for param in actor.parameters():
        param.requires_grad = False
    actor.residual_weights.requires_grad = True
    # 面积编码：area_scale 可训
    if use_area_encoding:
        actor.area_scale.requires_grad = True

    # 排名偏置：lambda_raw 可训，w_area 可选冻结（Step 1）
    rb_trainable = 0
    if use_rank_bias:
        actor.lambda_raw.requires_grad = True
        rb_trainable = 1
        if freeze_w_area:
            actor.residual_weights.requires_grad = False

    unfrozen_count = 0
    if not freeze:
        for name, param in actor.named_parameters():
            if UNFREEZE_PREFIX in name:
                param.requires_grad = True
                unfrozen_count += param.numel()
    if not (use_rank_bias and freeze_w_area):
        actor.residual_weights.requires_grad = True

    total_trainable = sum(p.numel() for p in actor.parameters() if p.requires_grad)
    ae_params = 1 if use_area_encoding else 0
    print(f"   Trainable: {total_trainable:,} (decoder {unfrozen_count:,} "
          f"+ residual {n_residual} + area_scale {ae_params}"
          f" + lambda {rb_trainable})")
    if use_rank_bias and freeze_w_area:
        print(f"   w_area FROZEN (Step 1: only train lambda)")

    # ---- 4. Optimizer 参数组 ----
    print("\n4. Setting up optimizers...")
    residual_params = [actor.residual_weights]
    unfrozen_params = [p for n, p in actor.named_parameters()
                       if p.requires_grad and 'residual_weights' not in n
                       and 'area_scale' not in n and 'area_embedding' not in n
                       and 'lambda_raw' not in n]

    opt_groups = [
        {'params': unfrozen_params, 'lr': UNFREEZE_DECODER_LR, 'weight_decay': UNFREEZE_WD},
        {'params': residual_params, 'lr': RESIDUAL_LR, 'weight_decay': 0.0},
    ]
    if use_area_encoding:
        area_scale_params = [actor.area_scale]
        opt_groups.append({'params': area_scale_params, 'lr': AREA_ENCODING_LR,
                           'weight_decay': 0.0})
        ae_params = [p for n, p in actor.named_parameters()
                     if 'area_embedding' in n and p.requires_grad]
        if ae_params:
            opt_groups.append({'params': ae_params, 'lr': AREA_ENCODING_LR * 0.1,
                               'weight_decay': 1e-5})
    if use_rank_bias:
        opt_groups.append({'params': [actor.lambda_raw], 'lr': RANK_BIAS_LR,
                           'weight_decay': 0.0})

    actor_opt = optim.Adam(opt_groups)
    ae_str = " + area_scale" if use_area_encoding else ""
    rb_str = " + lambda" if use_rank_bias else ""
    print(f"   Actor: {len(opt_groups)} groups, residual_lr={RESIDUAL_LR}, "
          f"decoder_lr={UNFREEZE_DECODER_LR}{ae_str}{rb_str}")

    # ---- 5. Critic ----
    print("\n5. Loading Critic...")
    critic = CriticNetwork(d_feat=D_FEAT, d_model=D_MODEL, nhead=NHEAD,
                           num_encoder_layers=NUM_ENCODER_LAYERS,
                           num_decoder_layers=NUM_DECODER_LAYERS,
                           dim_feedforward=DIM_FEEDFORWARD,
                           max_seq_len=MAX_SEQ_LEN)
    # 尝试从 phase1_checkpoint 加载 critic（V3 训练时保存）
    critic_ckpt_path = critic_ckpt
    if not os.path.exists(critic_ckpt_path):
        critic_ckpt_path = os.path.join(MODEL_DIR, 'phase1_best.pth')  # fallback
    if os.path.exists(critic_ckpt_path):
        ckpt_full = torch.load(critic_ckpt_path, map_location='cpu', weights_only=False)
        if 'critic' in ckpt_full:
            critic_sd = ckpt_full['critic']
            csd = critic.state_dict()
            critic_sd_f = {k: v for k, v in critic_sd.items()
                           if k in csd and csd[k].shape == v.shape}
            critic.load_state_dict(critic_sd_f, strict=False)
            print(f"   Loaded critic from phase1_best.pth ({len(critic_sd_f)} keys)")
        else:
            print("   No critic in checkpoint, using fresh critic")
    else:
        print("   No critic checkpoint, using fresh critic")

    critic_opt = optim.Adam(critic.parameters(), lr=CRITIC_LR)

    # ---- 6. 移到 device + init_params ----
    actor = actor.to(DEVICE)
    critic = critic.to(DEVICE)

    init_params = {}
    for name, param in actor.named_parameters():
        if param.requires_grad and 'residual_weights' not in name:
            init_params[name] = param.detach().clone()

    # ---- 7. 校准 ----
    test_files = []
    if test_dir is None:
        test_dir = r'D:\Transformer\测试集合1'
    test_files = sorted([os.path.join(test_dir, f) for f in os.listdir(test_dir)
                         if f.endswith('.txt') and '_feat' not in f
                         and 'summary' not in f and 'best' not in f
                         and 'experiment' not in f])
    random.seed(seed)
    random.shuffle(test_files)
    test_files = test_files[:6]

    if not skip_calibration:
        print("\n7. Calibration benchmark...")
        actor.eval()
        calib = benchmark_model(actor, feat_cols, test_files)
        print(f"   Calibration util: {calib['mean']:.4f}")
        with open(os.path.join(PPO_DIR, f'calibration{_tag_suffix}.json'), 'w', encoding='utf-8') as f:
            json.dump(calib, f, indent=2, ensure_ascii=False)

    # ---- 8. 训练数据 ----
    print("\n8. Preparing training data...")
    if data_dir is None:
        data_dir = os.path.join(GENERATED_DIR, 'train')
    instance_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
                      if f.endswith('.txt') and '_feat' not in f
                      and '_order' not in f and '_angle' not in f]
    print(f"   Training instances: {len(instance_files)}")

    order_path = os.path.join(RESULT_DIR, 'step1', 'instance_order.json')
    if os.path.exists(order_path):
        with open(order_path, 'r') as f:
            order_data = json.load(f)
            instance_order = [
                [os.path.join(data_dir, fname) for fname in ep]
                for ep in order_data.get('order', [])
            ]
        print(f"   Using S1 instance order: {len(instance_order)} epochs")
    else:
        rng = random.Random(seed)
        instance_order = []
        for e in range(epochs):
            ep = rng.sample(instance_files, min(RL_SAMPLES_PER_EPOCH, len(instance_files)))
            instance_order.append(ep)
        print(f"   Generated new order (seed=42)")

    # ---- 9. Resume ----
    print("\n9. Checking resume...")
    resume_ckpt_path = os.path.join(CKPT_PPO_DIR, _ckpt_name)
    start_epoch = 1
    baseline_reward = None
    epoch_log, deviation_history = [], []
    stopped_early, stop_reason = False, None

    if os.path.exists(resume_ckpt_path):
        resume_ckpt = torch.load(resume_ckpt_path, map_location='cpu', weights_only=False)
        saved_epoch = resume_ckpt['epoch']
        if saved_epoch >= epochs:
            print(f"   Already completed ({saved_epoch}/{epochs})")
            epoch_log = resume_ckpt.get('epoch_log', [])
            deviation_history = resume_ckpt.get('deviation_history', [])
            start_epoch = epochs + 1
        else:
            start_epoch = saved_epoch + 1
            # 恢复模型权重
            if 'actor_sd' in resume_ckpt:
                actor_sd_resume = resume_ckpt['actor_sd']
                csd = actor.state_dict()
                sd_f = {k: v for k, v in actor_sd_resume.items()
                        if k in csd and csd[k].shape == v.shape}
                actor.load_state_dict(sd_f, strict=False)
            if 'critic_sd' in resume_ckpt:
                critic_sd_resume = resume_ckpt['critic_sd']
                ccd = critic.state_dict()
                ccd_f = {k: v for k, v in critic_sd_resume.items()
                         if k in ccd and ccd[k].shape == v.shape}
                critic.load_state_dict(ccd_f, strict=False)
            try:
                actor_opt.load_state_dict(resume_ckpt['actor_optimizer_sd'])
                critic_opt.load_state_dict(resume_ckpt['critic_optimizer_sd'])
            except Exception:
                print("   Warning: optimizer state restore failed")
            for name in init_params:
                if name in resume_ckpt.get('init_params', {}):
                    init_params[name] = resume_ckpt['init_params'][name].to(DEVICE)
            epoch_log = resume_ckpt.get('epoch_log', [])
            deviation_history = resume_ckpt.get('deviation_history', [])
            baseline_reward = resume_ckpt.get('baseline_reward')
            if reset_early_stop:
                # A3 两段式：段 1 的 30 条记录会污染段 2 的 patience 早停窗口
                # （段 2 一启动 len(epoch_log)>=13 即满足条件，被段 1 的 R 历史误判
                #  "No improvement for 10 epochs"）。清空后早停只看段 2 自己的历史。
                epoch_log = []
                deviation_history = []
                baseline_reward = None
                print("   [reset_early_stop] epoch_log/baseline cleared "
                      "(early-stop window restarts)")
            print(f"   Resuming from epoch {start_epoch}")
    else:
        print("   Fresh start")

    # ---- 10. 训练 ----
    if start_epoch <= epochs:
        floor = W_AREA_FLOOR if w_area_floor is None else w_area_floor
        print(f"\n10. Starting PPO training ({epochs} epochs)...")
        print(f"    Stop-loss: w_area < {floor}, "
              f"R drop > {TRAIN_R_DROP*100:.0f}% after {REGULAR_CHECK_EPOCHS} epochs"
              + (f", patience={patience}" if patience else ""))
        if not freeze:
            print(f"    Stop-loss: deviation surge > {DEVIATION_SURGE} in "
                  f"first {QUICK_CHECK_EPOCHS} epochs")
        print()

        t_total_start = time.time()

        for epoch in range(start_epoch, epochs + 1):
            t0 = time.time()
            actor_cpu = {k: v.cpu() for k, v in actor.state_dict().items()}
            critic_cpu = {k: v.cpu() for k, v in critic.state_dict().items()}

            epoch_files = instance_order[(epoch - 1) % len(instance_order)]
            tasks = [(f, actor_cpu, critic_cpu, feat_cols) for f in epoch_files]

            all_traj = []
            with concurrent.futures.ProcessPoolExecutor(max_workers=1) as ex:
                for r in ex.map(ppo_worker, tasks):
                    all_traj.extend(r)

            stats = ppo_update(actor, critic, actor_opt, critic_opt, all_traj,
                               feat_cols, PPO_EPOCHS, CLIP_EPSILON, ENTROPY_COEF,
                               GAMMA, GAE_LAMBDA)
            et = time.time() - t0
            w = actor.get_weights()

            # 偏差监控
            dev_norm = actor.compute_deviation_norm(init_params) if not freeze else 0.0
            deviation_history.append(dev_norm)

            # 计算 avg reward
            avg_r = np.mean([sum(rm) for _, _, rm in all_traj]) if all_traj else 0

            epoch_info = {
                'epoch': epoch,
                'avg_reward': round(float(avg_r), 4),
                'actor_loss': stats['actor_loss'],
                'critic_loss': stats['critic_loss'],
                'entropy': stats['entropy'],
                'w_area': round(w[0], 6),
                'deviation_norm': round(dev_norm, 6),
                'time_s': round(et, 0),
            }
            if n_residual >= 2:
                epoch_info['w_edge'] = round(w[1], 6)
            if use_area_encoding:
                epoch_info['area_scale'] = round(actor.get_area_scale(), 6)
            if use_rank_bias:
                epoch_info['lambda'] = round(actor.get_lambda(), 6)
            epoch_log.append(epoch_info)

            # 打印
            w_str = f"w=[area={w[0]:.4f}"
            if n_residual >= 2:
                w_str += f"  edge={w[1]:.4f}"
            w_str += "]"
            if use_area_encoding:
                w_str += f"  α={actor.get_area_scale():.4f}"
            if use_rank_bias:
                w_str += f"  λ={actor.get_lambda():.4f}"
            dev_str = f"  dev={dev_norm:.4f}" if not freeze else ""
            print(f"  E{epoch:3d}/{epochs}  R={avg_r:.4f}  "
                  f"AL={stats['actor_loss']:.4f}  CL={stats['critic_loss']:.4f}  "
                  f"ent={stats['entropy']:.4f}  "
                  f"{w_str}  T={et:.0f}s{dev_str}")

            # 止损
            if not freeze and epoch <= QUICK_CHECK_EPOCHS and dev_norm > DEVIATION_SURGE:
                stopped_early = True
                stop_reason = f'Deviation surge: {dev_norm:.2f}'
                print(f"\n  >>> STOP-LOSS: {stop_reason}")
                break

            if baseline_reward is None and epoch >= 3:
                baseline_reward = np.mean([e['avg_reward'] for e in epoch_log[:3]])
                print(f"  --- Baseline R: {baseline_reward:.4f} ---")
            elif baseline_reward is not None and epoch >= REGULAR_CHECK_EPOCHS:
                recent = np.mean([e['avg_reward'] for e in epoch_log[-3:]])
                if recent < baseline_reward * (1 - TRAIN_R_DROP):
                    stopped_early = True
                    stop_reason = f'R drop: {recent:.4f} < {baseline_reward:.4f}*0.95'
                    print(f"\n  >>> STOP-LOSS: {stop_reason}")
                    break

            if w[0] < floor:
                stopped_early = True
                stop_reason = f'w_area={w[0]:.4f} < {floor}'
                print(f"\n  >>> STOP-LOSS: {stop_reason}")
                break

            # 早停：连续 patience 轮无改善
            if patience > 0 and len(epoch_log) >= patience + 3:
                recent_best = max(e['avg_reward'] for e in epoch_log[-patience:])
                older_best = max(e['avg_reward'] for e in epoch_log[-(patience+3):-patience])
                if recent_best <= older_best:
                    stopped_early = True
                    stop_reason = f'No improvement for {patience} epochs'
                    print(f"\n  >>> EARLY STOP: {stop_reason}")
                    break

            # 保存 checkpoint
            ckpt_data = {
                'epoch': epoch,
                'actor_sd': {k: v.cpu() for k, v in actor.state_dict().items()},
                'critic_sd': {k: v.cpu() for k, v in critic.state_dict().items()},
                'residual_weights': actor.residual_weights.detach().cpu(),
                'actor_optimizer_sd': actor_opt.state_dict(),
                'critic_optimizer_sd': critic_opt.state_dict(),
                'init_params': init_params,
                'epoch_log': epoch_log,
                'deviation_history': deviation_history,
                'baseline_reward': baseline_reward,
                'tag': tag,
            }
            if use_area_encoding:
                ckpt_data['area_scale'] = actor.area_scale.item()
            if use_rank_bias:
                ckpt_data['lambda_raw'] = actor.lambda_raw.item()
            _atomic_torch_save(ckpt_data, resume_ckpt_path)

            # 里程碑保存：30, 40, 50, 60 epoch 各存一份
            if epoch >= 30 and epoch % 10 == 0:
                milestone_path = resume_ckpt_path.replace('.pth', f'_epoch{epoch}.pth')
                torch.save(ckpt_data, milestone_path)
                print(f"    [Milestone saved: epoch {epoch}]")

        total_time = time.time() - t_total_start
    else:
        total_time = 0

    # ---- 11. Benchmark ----
    print(f"\n11. Final benchmark...")
    actor.eval()
    bench = benchmark_model(actor, feat_cols, test_files)

    # ---- 12. Save ----
    log = {
        'experiment': f'PPO — {mode}, n_residual={n_residual}'
                      f'{", +AreaEncoding" if use_area_encoding else ""}'
                      f'{", +RankBias" if use_rank_bias else ""}'
                      f'{", +RankSteer(" + rank_steer + ")" if use_rank_steer else ""}',
        'tag': tag,
        'anchor': anchor_ckpt,
        'n_residual': n_residual,
        'freeze': freeze,
        'use_area_encoding': use_area_encoding,
        'rank_steer_mode': rank_steer if use_rank_steer else None,
        'feat_cols': feat_cols,
        'ppo_params': {
            'clip_epsilon': CLIP_EPSILON,
            'gae_lambda': GAE_LAMBDA,
            'ppo_epochs': PPO_EPOCHS,
            'entropy_coef': ENTROPY_COEF,
        },
        'lr': {
            'residual': RESIDUAL_LR,
            'decoder': UNFREEZE_DECODER_LR if not freeze else None,
            'critic': CRITIC_LR,
            'area_encoding': AREA_ENCODING_LR if use_area_encoding else None,
        },
        'stopped_early': stopped_early,
        'stop_reason': stop_reason,
        'total_time_min': round(total_time / 60, 1) if total_time else 0,
        'final_weights': actor.get_weights(),
        'epoch_log': epoch_log,
        'deviation_history': deviation_history,
        'benchmark': bench,
    }
    if use_area_encoding:
        log['area_encoding'] = {
            'n_buckets': N_AREA_BUCKETS,
            'scale_init': AREA_SCALE_INIT,
            'freeze_embedding': True,
        }
    if use_rank_bias:
        log['rank_bias'] = {
            'lambda_init': LAMBDA_INIT,
            'freeze_w_area': freeze_w_area,
        }

    log_path = os.path.join(PPO_DIR, _log_name)
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    # 打印对比
    calib_util = calib.get('mean', 0) if not skip_calibration else 0
    ae_tag2 = " + AreaEncoding" if use_area_encoding else ""
    rb_tag2 = " + RankBias" if use_rank_bias else ""
    print(f"\n{'='*65}")
    print(f"PPO Results — {mode}, n_residual={n_residual}{ae_tag2}{rb_tag2}")
    print(f"{'='*65}")
    print(f"Calibration:              util={calib_util:.4f}")
    print(f"Final benchmark:          util={bench['mean']:.4f}")
    if calib_util:
        print(f"Delta:                    {bench['mean']-calib_util:+.4f}")
    print(f"Final weights:            {actor.get_weights()}")
    if use_area_encoding:
        print(f"Final area_scale:         {actor.get_area_scale():.6f}")
    if use_rank_bias:
        print(f"Final lambda:             {actor.get_lambda():.6f}")
    if not freeze:
        print(f"Max deviation norm:       {max(deviation_history) if deviation_history else 0:.4f}")
    if stopped_early:
        print(f"STOPPED EARLY: {stop_reason}")
    print(f"Log: {log_path}")

    return log


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPO Training')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--n_residual', type=int, default=2, choices=[1, 2],
                        help='1=area only, 2=area+longest_edge (default: 2)')
    parser.add_argument('--freeze', action='store_true', default=True,
                        help='Freeze transformer backbone (Phase 4b/4c)')
    parser.add_argument('--unfreeze', action='store_true', default=False,
                        help='Unfreeze decoder L2 (Phase 4d)')
    parser.add_argument('--use_area_encoding', action='store_true', default=False,
                        help='Enable sinusoidal area encoding (Size Encoding) added to encoder input')
    parser.add_argument('--use_rank_bias', action='store_true', default=False,
                        help='Enable rank-based logit penalty for large-first ordering')
    parser.add_argument('--rank_steer', type=str, default='none',
                        choices=STEER_MODES,
                        help='RankSteer tied input injection direction '
                             '(none|const|dim0|dim63|random)')
    parser.add_argument('--use_pointer', action='store_true', default=False,
                        help='Pointer 选择机制：logits_i = (W_q·h_dec)·mem_i '
                             '（逐零件打分，可学；2026-08-15 本阶段）')
    parser.add_argument('--freeze_w_area', action='store_true', default=False,
                        help='Freeze w_area, only train lambda (Step 1 of rank bias)')
    parser.add_argument('--patience', type=int, default=0,
                        help='Early stop if no reward improvement for N epochs (0=disabled)')
    parser.add_argument('--anchor_ckpt', type=str, default=None,
                        help='Anchor checkpoint（默认 S1 longest_edge）')
    parser.add_argument('--w_area_floor', type=float, default=None,
                        help='w_area 止损地板（默认 0.5）')
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--test_dir', type=str, default=None)
    parser.add_argument('--skip_calibration', action='store_true')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（多种子统计用；同时钉住 random/numpy/torch）')
    args = parser.parse_args()

    freeze = not args.unfreeze
    main(args.epochs, args.n_residual, freeze, args.tag,
         args.test_dir, args.data_dir, args.skip_calibration,
         args.use_area_encoding, args.use_rank_bias, args.rank_steer,
         args.use_pointer, args.freeze_w_area, args.patience,
         args.anchor_ckpt, None, args.w_area_floor, False, args.seed)
