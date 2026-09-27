# GeoSwap 算法复杂度分析

> 建立日期：2026-09-27 ｜ 所有常量**已对着代码核实**，不是凭记忆写的。

---

## 〇、符号表

| 符号 | 含义 | 值 / 出处 |
|---|---|---|
| n | 零件数 | 12–60（ESICUP）/ 30–60（训练集） |
| d | 模型维度 `D_FEAT` | **128**（`config.py:30`） |
| D | 前馈维度 `DIM_FEEDFORWARD` | **512**（`config.py:37`） |
| L | Transformer 层数 | **6** = 3 编码 + 3 解码（`config.py:35–36`） |
| h | 提案头隐藏宽度 | **256**（`geoswap_rounds.py:187` `mlp(dim, hidden=256)`） |
| P̄ | 多边形平均顶点数 | 由实例决定 |
| X | 一次放置扫描的 x 栅格列数 | 由几何决定，**有早退出** |
| Y | 条带高度方向的栅格点数 | 常数（`bin_height / PLACEMENT_STEP`） |
| B | 搜索预算（BLF 评估次数） | **40**（与 n 无关） |
| A | 接受的交换次数 | A ≤ B |
| K | 旋转角度数 `NUM_ANGLES` | **4**（`config.py:24`） |

---

## 一、五个组成部分

### ① 骨干前向（`backbone_step`）

`actor.encode(src)` → `decode_step` → `forward_logits`，共 6 层。

- 自注意力：`QK^T` 是 n×n 矩阵，每层 **O(n²d)**；注意力加权 **O(n²d)**
- 前馈：**O(nDd)**
- 6 层合计：**O(n²d + nDd)**

因 d、D 是常数：**时间 O(n²)，存储 O(n²)**（注意力矩阵）。

每次 `backbone_step` 调一次；**只在候选池重建时调用**（见 §三）。

### ② 候选池构造与打分

- 候选对数量 **C(n,2) = O(n²)**
- `build_rows`：向量化构造每一行的特征。每行 `hand`(13 维) + `memblock`(2d = 256 维) → 每行 O(d)，共 **O(n²d)**
- 打分 `predict`：n² 行过三层 MLP（d→h→h/2→1），每行 **O(dh + h²)**，共 **O(n²dh)**
- 排序 `np.argsort`：**O(n² log n)**

合计：**O(n² log n)**（排序项主导）。

### ③ 一次 BLF 评估（= 一次完整放置）← **成本主项**

放置第 k 个零件时：

1. 取它与已放的 k−1 个零件之间的 NFP（`_get_nfp` 有缓存，key = (固定件,角度,移动件,角度)，首次构造后 O(1)）
2. 扫 x 栅格列：`for x in range(0, x_limit+1, step)`，最多 X 列
3. 每列遍历 NFPs（**x 包围盒剪枝**后剩 m_k ≤ k−1 个）
4. 对每个存活的 NFP 做**段内** `contains_xy`（**y 段剪枝**，点数 ≤ 该 NFP 的 y 跨度 ≤ Y），点入多边形测试 O(P̄)
5. **早退出**：某一列出现空闲点即返回

单次放置最坏：**O(X · (m_k + Y) · P̄)**
累加 n 次放置（m_k ≤ k−1）：

> **一次 BLF 评估 = O(n² · X · P̄ + n · X · Y · P̄)**

**最坏情形**：X 增长到「已用长度 / step」，故最坏可达 O(n³)。
**实际情形**：条带被填得密，第一个可行 x 就在前沿附近，**X 的有效值很小**，因此实测标度接近 O(n²) 量级（见 §四）。

### ④ 整个搜索

```
T_search = B × T_BLF  +  A × (T_backbone + T_pool)
```

- **B 次评估，每次一次完整 BLF**
- **只有接受时**才重建候选池（+1 次骨干前向 + 1 次池构造与打分）
- A ≤ B，B = 40

代入：

> **T_search = O( B·n²X P̄ + A·(n² + n² log n) )**

**因为 B = 40 与 n 无关，总复杂度 = 常数 × 单次 BLF 评估的成本（同阶）。**

### ⑤ 内存

| 项 | 规模 | n=60 时的量 |
|---|---|---|
| 注意力矩阵 | O(n²) | 3600 × 8 头 |
| 候选池特征 | C(n,2) × (13 + 256) × 4 B | 1770 × 269 × 4 ≈ **1.9 MB** |
| 一轮训练样本（560 实例） | 560 × 40 × 269 × 4 B | ≈ **24 MB**（实测 32 MB，含 lp 块） |

**内存不是瓶颈。**

---

## 二、核心结论（论文要的那句话）

> 搜索的复杂度是 **O(B) 次完整 BLF 放置**（外加至多 B 次候选池重建），**B = 40 与 n 无关**。
> 单次 BLF 放置为 O(n²X P̄)，因此 **GeoSwap 与"同预算的随机 LNS"复杂度完全同阶 —— 学习组件不改变复杂度阶**，只改变每次评估打在哪一对上。

**与之对照：**

| | 完整 BLF 放置次数 | 额外开销 |
|---|---|---|
| **GeoSwap** | **40** | ≤40 次骨干前向 + ≤40 次候选池重建 |
| **随机 LNS（同预算）** | 40 | 无 |
| **遗传算法** | **110** | 无 |

三者**同阶**；我们的常数因子更大（每次评估实测贵 **1.10 倍**），但**评估次数少 2.75 倍**。

**还有一个下界性质值得写**：任何"用 BLF 评估一条完整排序"的方法，单次评估都不可能快于一次完整放置。**我们的每次评估 = 1 次完整放置 + （仅在接受时）一次前向/重建，实测仅贵 10%** —— 也就是说，**我们几乎以 BLF 本身的成本在做学习式提案**。这是这个设计的经济性所在。

---

## 三、多少次前向 / 多少次池重建？

设 A = 接受的交换次数。一次搜索中：

- 骨干前向：**A + 1 次**（每次状态改变后重建一次）
- 候选池构造 + 打分：**A + 1 次**
- 完整 BLF 放置：**B = 40 次**

实测（ESICUP 10 实例，40 次评估）：**接受 29 次 / 400 次评估 = 7.25%**，平均 2.9 次/实例。所以 A ≈ 3，**前向与池重建各约 4 次**，而 BLF 放置 40 次 —— **成本几乎全是 BLF**。

---

## 四、经验标度（★ 已有数据，非新实验）

本机 10 个实例（n = 12…60，各 40 次评估）：

**每次评估耗时 ∝ n^1.87**（log-log 最小二乘拟合，相关系数 r = 0.92）

| n | 12 | 20 | 24 | 24 | 25 | 25 | 28 | 30 | 43 | 60 |
|---|---|---|---|---|---|---|---|---|---|---|
| 实测 秒/次 | 0.262 | 0.377 | 0.580 | 1.122 | 0.441 | 0.489 | 1.264 | 0.988 | 2.374 | 4.250 |
| 拟合 | 0.187 | 0.487 | 0.686 | 0.686 | 0.740 | 0.740 | 0.915 | 1.041 | 2.044 | 3.816 |

⚠️ **这个拟合不干净**：它是**跨实例**拟的，混了形状复杂度（同是 n=24，`marques` 0.58 秒/次、`albano` 1.12 秒/次，**差 2 倍**）。

**要干净的经验标度**：用同一生成器、同一规模档的实例（30/35/40/45/50/55/60 各取 ~20 个），只让 n 变。**成本：本机约 30–60 分钟。**

---

## 五、可直接用的英文（投稿草稿）

> **Complexity.**
> Let n be the number of parts, d the model width, P̄ the mean polygon vertex count, and X and Y the numbers of grid columns and grid points scanned by the placement routine; d is fixed at 128 and the encoder–decoder stack has 6 layers. One backbone forward pass is O(n²d), the candidate pool over all C(n,2) position pairs costs O(n²d + n²dh) with h = 256, and one bottom-left-fill evaluation — n placements, each scanning at most X columns against up to k−1 no-fit polygons with an early exit at the first feasible column — is O(n²X P̄) in the worst case and close to quadratic in practice.
>
> The search performs **B = 40** evaluations, rebuilding the candidate pool only when an exchange is accepted. With A ≤ B accepted exchanges the total cost is
> **O( B·n²X P̄ + A·(n² + n² log n) )**, where B is independent of n. **The learned component therefore does not change the complexity class**: GeoSwap is of the same order as a random-proposal local search at the same budget, differing only in which pair each evaluation is spent on. Against the genetic algorithm, which spends 110 full placements and no neural overhead, the same order holds, with a larger constant per evaluation (measured 1.10×) and 2.75× fewer evaluations.
>
> Memory is dominated by the O(n²) attention matrix and the O(n²) candidate pool (about 1.9 MB at n = 60). Since every evaluation costs at least one full placement, and GeoSwap adds a backbone pass and a pool rebuild only on acceptance — 29 of 400 evaluations (7.25%) in our runs — the method operates within about 10% of the placement routine's own cost.

---

## 六、数据出处

| 数字 | 来源 |
|---|---|
| 模型常量（d=128 / D=512 / 6 层 / 8 头 / 4 角度） | `config.py` L24–38 |
| 提案头结构（d→256→128→1） | `rlfine/geoswap_rounds.py` L187–193 |
| BLF 扫描结构（5 次 attempt、x 栅格、x/y 包围盒剪枝、早退出） | `training/rl_env_v2.py` L226–287 |
| NFP 缓存 | `training/rl_env_v2.py` `_get_nfp`（`nfp_cache` / `buffered_nfp_cache`） |
| 候选池构造 | `rlfine/geoswap_rounds.py::build_rows` |
| 每轮接受次数与接受率（29 次 / 7.3%） | `_gsw_diag/final/eval/esicup_mem_curve.jsonl` |
| 每次评估耗时（本机 10 实例） | `_gsw_diag/modelstart/search_model.jsonl` |
| 每次评估成本比 1.10× | 服务器 `esicup_mem_search.jsonl` vs `ga_esicup/search.jsonl` |
