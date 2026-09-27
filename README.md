# GeoSwap 复现包

**打包日期：2026-09-27**

本包包含 GeoSwap（二维不规则条形排样：骨干表征 + 交换提案 + BLF 验证的接受式爬坡）的**全部代码、数据、两个训练好的模型与结果数据**。

---

## 一、目录结构

```
GeoSwap_复现包_20260927\
├── README.md                       本文件
├── code\
│   ├── train_first_model\          ★ 第一个模型（修复后骨干）的训练代码
│   │   ├── supervised_pretrain.py      训练主脚本（--labels 读 LNS 标签）
│   │   ├── ranksteer_train.py          PPOActor 模型类 + 训练循环
│   │   ├── gen_improved_labels.py      ★ LNS 标签生成器（小/中实例，--iters 40）
│   │   └── gen_labels_subset.py        ★ LNS 标签生成器（大实例 55/60 件，--iters 15）
│   ├── geoswap\                    ★ 主方法
│   │   ├── geoswap_rounds.py           六轮闭环训练 + 评测（含 search_task / train_round / run_eval）
│   │   ├── ga_esicup.py                GA 基线（pop=10 × 11 代 = 110 次评估，精英保留，seed 7）
│   │   └── common.py                   环境/路径/工具（env_for, area_order, rollout_given_order, build_actor）
│   ├── env\                        排样环境与基础
│   │   ├── rl_env_v2.py                ★ PackingEnvV2：BLF 放置、128 维射线特征、NFP
│   │   ├── geometry.py                 多边形工具（polygon_area、NFP 构造）
│   │   ├── config.py                   全局配置（MODEL_DIR、MAX_SEQ_LEN、NUM_ANGLES…）
│   │   ├── preprocess.py               parse_instance_file
│   │   └── generate.py                 训练实例生成器
│   └── baseline\
│       └── genetic.py                  早期 GA 实现（⚠️ 无精英保留，其数字不可用，仅存档）
├── models\
│   ├── anchor_sup128.pth           ★ 第一个模型 = 修复后骨干（5.5 MB）
│   └── headA_r6.pth                ★ 第二个模型 = 表征头，256 维纯 memory（0.38 MB）
├── data\
│   ├── train\                      ★ 560 个训练实例（1681 文件 / 155.7 MB）
│   ├── esicup\                     ★ ESICUP 公开集 10 个实例（唯一与 GA 对比的测试集）
│   ├── test30\                     ★ 自建开发集 30 个实例（内部选型用）
│   └── labels_v2.json              ★ LNS 标签 540 个（{basename: [order...]}）
└── results\                        全部结果数据（jsonl，逐实例 + 逐步骤）
    ├── esicup\                     ESICUP 评测：4 档提案器 × (curve + search)
    │   ├── esicup_mem_curve.jsonl        表征头（我们的方法）
    │   ├── esicup_hand_curve.jsonl       手工头（消融：换成 13 维手工特征）
    │   ├── esicup_borrowed_curve.jsonl   现成分（骨干的 lp_i+lp_j，零训练）
    │   ├── esicup_random_curve.jsonl     随机试（随机提案顺序）
    │   └── *_search.jsonl                逐实例汇总
    ├── ga\
    │   ├── curve.jsonl               GA 质量-预算曲线（110 次评估，逐代 best-so-far）
    │   └── search.jsonl              GA 逐实例最终结果（含 ga_seq 排序）
    ├── train560\                     560 训练实例上的六轮闭环结果（r1_random … r6_mem/hand）
    └── test30\                       30 实例开发集评测
```

---

## 二、数据格式

**结果 jsonl**

`*_curve.jsonl` —— 每一步一行：
```json
{"instance": "blaz.txt", "k": 1, "u_before": 0.7251, "u_after": 0.7129,
 "du": -0.0122, "accepted": 0, "depth": 0}
```
- `k`：第几次 BLF 评估（计价单位）
- `u_before`：该次评估前的当前最优利用率
- `u_after`：交换后重跑 BLF 得到的利用率
- `du = u_after − u_before`；`accepted = 1` 表示接受（`du > 0`）
- `depth`：已接受的交换次数

`*_search.jsonl` —— 每个实例一行（最终利用率、接受次数、评估次数、耗时等）。

**实例文件**：第一行零件数，第二行条带宽高，随后每行一个多边形（顶点坐标交替）。

---

## 三、从零复现（五步）

```bash
# ① 生成训练数据（560 个实例）
python code/env/generate.py
  → data/train/                     每个实例 3 个文件（.txt / _feat.txt / _feat_all.txt）

# ② 生成 LNS 标签（两个脚本分别覆盖小/中 与 大实例）
python code/train_first_model/gen_improved_labels.py --iters 40   # 小/中实例
python code/train_first_model/gen_labels_subset.py   --iters 15   # 大实例（55/60 件）
  → 合并为 data/labels_v2.json（540 个）

# ③ 训练第一个模型（修复后骨干）
python code/train_first_model/supervised_pretrain.py --labels data/labels_v2.json
  → models/anchor_sup128.pth

# ④ GeoSwap 六轮闭环训练 + 评测
python code/geoswap/geoswap_rounds.py
  # 第 1 轮 = 随机点火（不计入轮次），第 2–6 轮用上一轮训出的头
  # 每轮产出两个头：headA_r{r}.pth（表征，256 维）/ headB_r{r}.pth（手工，13 维）
  → results/

# ⑤ GA 基线
python code/geoswap/ga_esicup.py
  → results/ga/
```

**评测口径**
- 计价单位 = **BLF 评估次数**（与机器无关）
- 两边都用同一个向量化 BLF：环境变量 **`PACKING_FAST_BLF=1`**（利用率与逐点实现**逐位一致**）
- **起点不同来源、几乎相同排序**：遗传算法从**面积序**起步（初始种群 = 面积序 + 9 个扰动）；GeoSwap 从**第一个模型产出的排序**起步。两者对面积序的 Spearman 相关为 **0.988**，所以比较把起点几乎固定住，只变评估怎么花
- 我们 40 次评估；GA 110 次评估

---

## 四、环境依赖

- Python 3.9+
- 核心：`numpy`、`torch`、`shapely`、`matplotlib`、`scipy`
- 本项目在 Python 3.9 + PyTorch (CPU) + shapely 下运行
- ⚠️ `code/env/rl_env_v2.py` 里的 BLF 是纯 Python/NumPy 实现，**不需要 GPU**

---

## 五、⚠️ 尚未包含（需从服务器补）

| 缺什么 | 说明 |
|---|---|
| **`models/headB_r6.pth`** | **手工头（消融用，13 维输入）**。最终轮的这一个只在服务器上 |
| `models/headA_r1..r5.pth`、`headB_r1..r5.pth` | 各中间轮的头（用于复现各轮结果） |
| `results/train560/r3_hand/`、`r4_mem/`、`r4_hand/`、`r5_hand/` 的 `search.jsonl` | 这几档只有 `curve.jsonl`，缺逐实例汇总 |
| 每实例逐次的 `search.jsonl` | 服务器上有 8420 文件的完整目录（199 MB），本包只含汇总层 |

**这些都在：**
```
10.0.0.1:C:\Users\Administrator\Desktop\ccm\Transformer_new\rlfine\results\geoswap_gsw560\
```

---

## 六、几个必须知道的坑

| 坑 | 说明 |
|---|---|
| **空操作交换对** | ESICUP 里同种零件重复出现（`shapes0` 只有 4 种零件，**25.2% 的交换对是两个完全相同的零件**，交换后布局逐位不变）。训练集和 30 实例开发集**都是 0%**。代码里已在候选池剔除（`geoswap_rounds.py::search_task`） |
| **`baseline/genetic.py` 作废** | 该实现**没有精英保留**（每代整体替换、只从末代挑），是弱搜索。`ga_esicup.py` 才是论文用的版本（带精英保留） |
| **`_gsw_diag` 的 `hand` 一词两义** | 离线报告里的 `hand` = 手写规则；闭环评测里的 `hand` = 手工头（13 维 MLP） |
| **checkpoint 的标准化统计键名** | 用 `x_mu` / `x_sd`，**不能叫 `sd`**（与 `state_dict` 键冲突） |
| **`ranksteer/eval_rs.py` 导入会改工作目录** | 模块顶层有 `os.chdir(仓库根)`。要用其中的 `run_episode` 需在 import 前后恢复 cwd |
| **30 实例 ≠ 公开集** | 30 实例是**自建开发集**（内部选型用）；**ESICUP 10 个才是与 GA 对比的公开集** |
