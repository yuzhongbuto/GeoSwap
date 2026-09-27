"""
统一配置模块 — 所有超参数和路径集中管理
"""
import os
import torch

# ======================== 路径配置 ========================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
GENERATED_DIR = os.path.join(DATA_DIR, "generated")       # 自生成训练/验证数据
BENCHMARK_DIR = os.path.join(DATA_DIR, "benchmark")       # 标准benchmark (Terashima)
MODEL_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
RESULT_DIR = os.path.join(PROJECT_ROOT, "results")

for d in [GENERATED_DIR, BENCHMARK_DIR, MODEL_DIR, RESULT_DIR]:
    os.makedirs(d, exist_ok=True)

# ======================== 设备配置 ========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================== 几何配置 ========================
PLATE_WIDTH = 1000            # 固定板材宽度
NUM_RAYS = 128                # 质心射线数（特征维度）
NUM_ANGLES = 4                # 旋转角度数（0°, 90°, 180°, 270°）
ANGLE_DEGREES = [0, 90, 180, 270]
PLACEMENT_STEP = 1            # 扫描步长（评估用，step=1最精确）
PLACEMENT_STEP_FAST = 5       # 快速扫描步长（训练用）

# ======================== 特征维度 ========================
D_FEAT = 128                 # GNN 编码器输出维度

# ======================== 模型架构 ========================
D_MODEL = 128
NHEAD = 8
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DIM_FEEDFORWARD = 512
MAX_SEQ_LEN = 60
DROPOUT = 0.1

# ======================== 监督预训练 ========================
SUPERVISED_EPOCHS = 100
SUPERVISED_BATCH_SIZE = 32
LR_SUPERVISED = 1e-4
SUPERVISED_PATIENCE = 15      # early stopping

# ======================== RL微调 ========================
RL_EPOCHS = 600
LR_ACTOR = 5e-5
LR_CRITIC = 5e-5
LR_ANGLE_HEAD = 5e-4          # 角度头学习率更高
GAMMA = 0.99                  # 折扣因子
TRAJECTORIES_PER_INSTANCE = 1 # 每个实例采样轨迹数
RL_SAMPLES_PER_EPOCH = 30     # 每epoch随机采样实例数

# ======================== CNN 位置评分器 ========================
CNN_GRID_SIZE = 128           # BIN 栅格图尺寸
CNN_CANDIDATE_K = 50          # 每步候选位置数
CNN_LR = 1e-4                 # CNN 学习率
PHASE1_EPOCHS = 100           # Phase1: 序列+角度+BLF 训练轮数
PHASE2_EPOCHS = 100           # Phase2: CNN定位训练轮数
PHASE3_EPOCHS = 50            # Phase3: 联合微调轮数
GRAD_CLIP = 2.0               # 梯度裁剪
LOGIT_CLAMP = 20

# ======================== 角度策略（核心改进） ========================
ENTROPY_COEF_INIT = 0.05      # 初始熵系数（比原来的0.02更高）
ENTROPY_COEF_MAX = 0.2        # 熵系数上限
ENTROPY_COEF_MIN = 0.005      # 熵系数下限
TARGET_ENTROPY_RATIO = 0.7    # 目标熵 = ratio * ln(num_angles)
ENTROPY_ADJUST_STEP = 0.005   # 每epoch调整步长
DYNAMIC_TEMP_INIT = 2.0       # 初始温度
DYNAMIC_TEMP_MIN = 0.5        # 最低温度

# ======================== Sinkhorn辅助损失 ========================
SINKHORN_EPOCHS = 20          # 前N个epoch使用Sinkhorn辅助
SINKHORN_WEIGHT = 0.3         # 辅助损失权重

# ======================== 课程学习（面积排序引导） ========================
TEACHER_PROB_START = 0.9       # 初始 teacher 干预概率
TEACHER_PROB_END = 0.0         # 最终 teacher 干预概率
TEACHER_DECAY_EPOCHS = 80      # teacher 概率从初始衰减到最终的轮数（放缓，防断崖）
CURRICULUM_ENABLED = True      # 是否启用课程学习

# ======================== 面积先验残差 ========================
W_AREA_INIT = 0.5              # w_area 初始值
W_AREA_FREEZE_EPOCHS = 50      # 前N轮冻结 w_area，之后再放开学习
LR_W_AREA = 5e-6               # w_area 学习率（主网络的 1/10）

# ======================== V4 多特征残差注入 ========================
N_RESIDUAL_FEATURES = 7        # 残差特征数（area + fill_rate + inv_circ + concave + recess + edge + skymatch）
W_RESIDUAL_INIT = 0.05         # 残差权重初始值（保守，让几何特征逐步发挥作用）
L2_REG_RESIDUAL = 0.001        # L2 正则化系数（防止某个权重爆炸）
ZSCORE_MIN_STD = 1e-3          # z-score 最小标准差（防尾部失真）
SKYLINE_BINS = 16              # 天际线分辨率和context维度

# ======================== 评估配置 ========================
NUM_EVAL_RUNS = 10            # 每个实例运行次数
SEEDS = list(range(42, 52))   # 固定随机种子保证可复现

# ======================== 遗传算法配置 ========================
GA_POP_SIZE = 10              # 种群大小
GA_MAX_GEN = 10               # 最大代数
GA_CROSSOVER_PROB = 0.5
GA_MUTATION_PROB = 0.1
GA_EARLY_STOP = 5             # 连续无改善则提前停止

# ======================== 数据生成配置 ========================
TRAIN_INSTANCES_PER_SIZE = 80  # 每个规模的训练实例数
VAL_INSTANCES_PER_SIZE = 20    # 每个规模的验证实例数
PART_SIZES = [30, 35, 40, 45, 50, 55, 60]  # 零件数分布
MIN_VERTICES = 3
MAX_VERTICES = 8
AREA_CATEGORIES = {            # 小/中/大零件面积范围
    "small":  (500, 2000),
    "medium": (2000, 8000),
    "large":  (8000, 20000),
}
