# Complexity of GeoSwap

> Written 2026-09-27. Every constant below was read off the code, not recalled.

---

## 0. Notation

| Symbol | Meaning | Value / source |
|---|---|---|
| n | number of parts | 12–60 (ESICUP), 30–60 (training set) |
| d | model width, `D_FEAT` | **128** (`config.py:30`) |
| D | feed-forward width, `DIM_FEEDFORWARD` | **512** (`config.py:37`) |
| L | Transformer depth | **6** = 3 encoder + 3 decoder (`config.py:35–36`) |
| h | hidden width of the proposal head | **256** (`geoswap_rounds.py:187`, `mlp(dim, hidden=256)`) |
| P̄ | mean polygon vertex count | set by the instance |
| X | grid columns scanned along x per placement | set by the geometry, **with early exit** |
| Y | grid points across the strip height | constant (`bin_height / PLACEMENT_STEP`) |
| B | search budget in BLF evaluations | **40**, independent of n |
| A | number of accepted exchanges | A ≤ B |
| K | number of rotation angles, `NUM_ANGLES` | **4** (`config.py:24`) |

---

## 1. The five components

### 1.1 Backbone forward pass (`backbone_step`)

`actor.encode(src)` → `decode_step` → `forward_logits`, six layers in total.

- Self-attention: `QK^T` is an n×n matrix, **O(n²d)** per layer; the weighted sum is **O(n²d)**
- Feed-forward: **O(nDd)**
- Six layers together: **O(n²d + nDd)**

With d and D constant this is **O(n²) time and O(n²) memory** for the attention matrix.

One call per rebuild of the candidate pool, and none otherwise (see section 3).

### 1.2 Candidate pool: construction and scoring

- Candidate pairs: **C(n,2) = O(n²)**
- `build_rows`: `hand` (13 attributes) plus `memblock` (2d = 256 values) per row, so O(d) per row and **O(n²d)** in total
- Scoring: n² rows through a three-layer MLP (d→h→h/2→1), **O(dh + h²)** per row, **O(n²dh)** in total
- Sorting with `np.argsort`: **O(n² log n)**

Together: **O(n² log n)**, dominated by the sort.

### 1.3 One BLF evaluation, that is one complete placement — the dominant cost

Placing the k-th part:

1. Take the no-fit polygon between it and each of the k−1 placed parts. `_get_nfp` caches on the key (fixed part, its angle, moving part, its angle), so after the first construction this is O(1).
2. Scan the x grid: `for x in range(0, x_limit+1, step)`, at most X columns.
3. For each column, walk the no-fit polygons that survive an **x bounding-box prune**, at most k−1 of them.
4. For each surviving polygon run an **in-segment** `contains_xy` on its y span, whose length is at most Y, at O(P̄) per point-in-polygon test.
5. **Early exit** at the first feasible point in a column.

Worst case for one placement: **O(X · (m_k + Y) · P̄)**. Summing over the n placements, with m_k ≤ k−1:

> **One BLF evaluation = O(n² · X · P̄ + n · X · Y · P̄)**

X grows with the used length over the grid step, so the worst case reaches O(n³). In practice the strip fills densely and the first feasible column lies near the front, so the effective X is small and the measured scaling is close to quadratic.

### 1.4 The whole search

```
T_search = B × T_BLF  +  A × (T_backbone + T_pool)
```

- B evaluations, each one complete BLF placement
- The candidate pool is rebuilt **only on acceptance**, together with one backbone forward pass
- A ≤ B, and B = 40

Substituting:

> **T_search = O( B·n²X P̄ + A·(n² + n² log n) )**

**Because B = 40 is independent of n, the total is a constant multiple of the cost of one BLF evaluation.** The learned component does not change the complexity class; it changes where each evaluation is spent.

### 1.5 Memory

| Item | Size | At n = 60 |
|---|---|---|
| Attention matrix | O(n²) | 3600 × 8 heads |
| Candidate-pool features | C(n,2) × (13 + 256) × 4 B | 1770 × 269 × 4 ≈ **1.9 MB** |
| One round of training samples (560 instances) | 560 × 40 × 269 × 4 B | ≈ **24 MB** (measured 32 MB including the lp block) |

Memory is not the constraint.

---

## 2. The result the paper uses

> The search costs **B full BLF placements**, plus at most B candidate-pool rebuilds, and **B = 40 is independent of n**. One placement is O(n²X P̄), so **GeoSwap is of the same order as a random-proposal local search spending the same budget: the learned component does not change the order**, only which pair each evaluation is spent on.

For comparison:

| | Full BLF placements | Additional overhead |
|---|---|---|
| **GeoSwap** | **40** | ≤40 backbone passes, ≤40 pool rebuilds |
| Random local search, same budget | 40 | none |
| Genetic algorithm | **110** | none |

All three are the same order. The larger constant per evaluation is measured at **1.10×**, against **2.75× fewer evaluations**.

**A lower bound applies.** No method can evaluate a complete ordering for less than the cost of placing it. GeoSwap spends one complete placement per evaluation plus one forward pass and one pool rebuild, and only on acceptance, which is the measured 10% above the placement routine's own cost. The learning therefore runs at close to the cost of the placement it is steering.

---

## 3. How many forward passes and how many pool rebuilds

With A the number of accepted exchanges, one search performs:

- Backbone forward passes: **A + 1**
- Candidate-pool constructions and scorings: **A + 1**
- Complete BLF placements: **B = 40**

Measured on the ten ESICUP instances at 40 evaluations: **29 exchanges accepted in 400 evaluations, 7.25%**, or 2.9 per instance. So A ≈ 3, the forward pass and the pool rebuild are each paid about four times, and the 40 placements carry almost all of the cost.

---

## 4. Measured timing

Ten instances, n = 12 to 60, 40 evaluations each:

| n | 12 | 20 | 24 | 24 | 25 | 25 | 28 | 30 | 43 | 60 |
|---|---|---|---|---|---|---|---|---|---|---|
| seconds per evaluation | 0.262 | 0.377 | 0.580 | 1.122 | 0.441 | 0.489 | 1.264 | 0.988 | 2.374 | 4.250 |

These are per-instance timings, and instances of the same part count differ in shape
complexity, so they are reported as measurements rather than fitted to a scaling law.
The wall-clock comparison that the paper uses is the median over the ten public
instances at a fixed budget (Section 5.5 of the paper).

---

## 5. Statement for the manuscript

> **Complexity.**
> Let n be the number of parts, d the model width, P̄ the mean polygon vertex count, and X and Y the numbers of grid columns and grid points scanned by the placement routine; d is fixed at 128 and the encoder–decoder stack has 6 layers. One backbone forward pass is O(n²d), the candidate pool over all C(n,2) position pairs costs O(n²d + n²dh) with h = 256, and one bottom-left-fill evaluation — n placements, each scanning at most X columns against up to k−1 no-fit polygons with an early exit at the first feasible column — is O(n²X P̄) in the worst case and close to quadratic in practice.
>
> The search performs **B = 40** evaluations, rebuilding the candidate pool only when an exchange is accepted. With A ≤ B accepted exchanges the total cost is
> **O( B·n²X P̄ + A·(n² + n² log n) )**, where B is independent of n. **The learned component therefore does not change the complexity class**: GeoSwap is of the same order as a random-proposal local search at the same budget, differing only in which pair each evaluation is spent on. Against the genetic algorithm, which spends 110 full placements and no neural overhead, the same order holds, with a larger constant per evaluation (measured 1.10×) and 2.75× fewer evaluations.
>
> Memory is dominated by the O(n²) attention matrix and the O(n²) candidate pool (about 1.9 MB at n = 60). Since every evaluation costs at least one full placement, and GeoSwap adds a backbone pass and a pool rebuild only on acceptance — 29 of 400 evaluations (7.25%) in our runs — the method operates within about 10% of the placement routine's own cost.

---

## 6. Where each number comes from

| Number | Source |
|---|---|
| Model constants (d=128, D=512, 6 layers, 8 heads, 4 angles) | `code/env/config.py`, lines 24–38 |
| Proposal-head structure (dim→256→128→1) | `code/geoswap/geoswap_rounds.py`, lines 187–193 |
| BLF scan structure (5 attempts, x grid, x/y bounding-box pruning, early exit) | `code/env/rl_env_v2.py`, lines 226–287 |
| NFP cache | `code/env/rl_env_v2.py`, `_get_nfp` (`nfp_cache`, `buffered_nfp_cache`) |
| Candidate-pool construction | `code/geoswap/geoswap_rounds.py::build_rows` |
| Accepted exchanges and acceptance rate (29 in 400, 7.25%) | `results/esicup/esicup_mem_curve.jsonl` |
| Wall-clock per evaluation | `results/esicup/esicup_mem_search.jsonl` and `results/ga/search.jsonl` |
