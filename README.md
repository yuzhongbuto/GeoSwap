# GeoSwap — reproduction package

**Learning both decisions of a genetic algorithm for irregular strip packing**
Cunmeng Chen, Dayong Cao — Harbin University of Science and Technology

This package contains the code, the instance sets, the two trained checkpoints and the
result files behind every table and figure of the paper.

---

## 1. What the method does

Two-dimensional irregular strip packing places irregular parts into a strip of fixed
width so that the used length is minimal. The ordering of the parts decides the layout
quality. A genetic algorithm optimizes that ordering directly; GeoSwap keeps the same
separation that a genetic algorithm keeps, and learns both halves of it.

| Component | File | Role |
|---|---|---|
| First model (constructive backbone) | `models/anchor_sup128.pth` | Produces the ordering and a 128-d description (encoder memory) for every part |
| Second model (representation head) | `models/headA_r6.pth` | Reads the two parts' memories (256-d) and ranks the candidate exchange |
| Search | `code/geoswap/geoswap_rounds.py::search_task` | Applies the top-ranked exchange, measures it with the placer, keeps it when the utilization rises |
| Placer | `code/env/rl_env_v2.py` | Bottom-left-fill placement with no-fit polygons |

The description carried by the first model is what makes two parts comparable: it is the
one place in the pipeline where size and geometry survive the standard processing chain.

---

## 2. Directory layout

```
.
├── code/
│   ├── env/                    packing environment and geometry
│   │   ├── rl_env_v2.py            PackingEnvV2: BLF placement, 128-ray features, NFP
│   │   ├── geometry.py             polygon tools (polygon_area, NFP construction)
│   │   ├── config.py               global settings (model dir, sequence length, angles)
│   │   ├── preprocess.py           writes the 131-d per-part feature files
│   │   └── generate.py             training-instance generator
│   ├── train_first_model/      first model
│   │   ├── supervised_pretrain.py  training entry point (--labels reads the LNS labels)
│   │   ├── ranksteer_train.py      PPOActor class and training loop
│   │   ├── gen_improved_labels.py  LNS label generator, small and medium instances
│   │   └── gen_labels_subset.py    LNS label generator, large instances (55/60 parts)
│   ├── geoswap/                the method
│   │   ├── geoswap_rounds.py       six closed-loop training rounds and evaluation
│   │   ├── ga_esicup.py            the GA baseline used in the paper
│   │   └── common.py               environment and path helpers
│   └── baseline/
│       └── genetic.py              archived early GA, see the note in section 6
├── models/
│   ├── anchor_sup128.pth       first model, repaired backbone (5.5 MB)
│   └── headA_r6.pth            second model, representation head (0.38 MB)
├── data/
│   ├── train/                  560 training instances
│   ├── esicup/                 the ten public ESICUP instances
│   ├── test30/                 30-instance development set
│   └── labels_v2.json          540 LNS labels
└── results/
    ├── esicup/                 ESICUP evaluation, four proposer arms
    ├── ga/                     GA curves and per-instance results
    ├── train560/               the six closed-loop rounds on the 560 instances
    └── test30/                 development-set evaluation
```

---

## 3. File formats

**Result files.** `*_curve.jsonl` holds one line per evaluation:

```json
{"instance": "blaz.txt", "k": 1, "u_before": 0.7251, "u_after": 0.7129,
 "du": -0.0122, "accepted": 0, "depth": 0}
```

- `k` — index of the BLF evaluation, the unit in which the paper charges both methods
- `u_before` / `u_after` — utilization before and after the candidate exchange
- `du = u_after - u_before`; `accepted = 1` when the exchange is kept (`du > 0`)
- `depth` — number of accepted exchanges so far

`*_search.jsonl` holds one line per instance: final utilization, accepted count,
evaluation count and wall-clock time.

**Instance files.** First line the part count, second line the strip width, then one
polygon per line with alternating vertex coordinates.

---

## 4. Reproduction, step by step

```bash
# 1. generate the 560 training instances
python code/env/generate.py

# 2. generate the LNS labels (two scripts cover small/medium and large instances)
python code/train_first_model/gen_improved_labels.py --iters 40
python code/train_first_model/gen_labels_subset.py   --iters 15
#    merge the two outputs into data/labels_v2.json (540 labels)

# 3. train the first model
python code/train_first_model/supervised_pretrain.py --labels data/labels_v2.json
#    -> models/anchor_sup128.pth

# 4. six closed-loop rounds, then evaluation
python code/geoswap/geoswap_rounds.py
#    round 1 is a random ignition round; rounds 2-6 use the head from the previous round.
#    Each round writes headA_r{r}.pth (representation, 256-d) and headB_r{r}.pth
#    (handcrafted, 13-d).

# 5. the GA baseline
python code/geoswap/ga_esicup.py
```

**Accounting conventions**

- The unit of account is the **BLF evaluation**, which is machine independent.
- Both methods call the same vectorized BLF implementation
  (`PACKING_FAST_BLF=1`); it reproduces the point-wise implementation bit for bit.
- **The two methods start from different orderings that agree to 0.988.** The genetic
  algorithm starts from the area-descending sort (its initial population is that sort
  plus nine perturbed copies). GeoSwap starts from the ordering the first model produces.
  The comparison therefore holds the starting point nearly fixed and varies only how the
  evaluations are spent.
- GeoSwap is reported at 40 evaluations; the genetic algorithm at 110.

---

## 5. Dependencies

- Python 3.9+
- `numpy`, `torch`, `shapely`, `matplotlib`, `scipy`
- Tested on Python 3.9 with CPU-only PyTorch and `shapely`.
- The placer in `code/env/rl_env_v2.py` is pure Python and NumPy. **No GPU is needed to
  reproduce any result in the paper.**

---

## 6. Notes for anyone reusing this code

| Item | What you need to know |
|---|---|
| Degenerate exchange pairs | Parts repeat in ESICUP. On `shapes0`, 43 parts come in 4 distinct feature vectors, so 228 of its 903 candidate pairs consist of two identical parts, and exchanging them leaves the layout bit-identical. The candidate pool removes such pairs in `geoswap_rounds.py::search_task`. The training and development sets contain none, because the generator draws every part with an independent size. |
| `baseline/genetic.py` is archived | That implementation has no elitism (it replaces each generation wholesale and selects from the last generation only), which makes it a weak search. `ga_esicup.py` is the version used in the paper: population 10, ten generations, elitism, seed 7. |
| `hand` has two meanings | In the offline reports `hand` means a hand-written rule; in the closed-loop evaluation it means the handcrafted head (13-d MLP). |
| Checkpoint normalization keys | The stored per-column statistics use the keys `x_mu` and `x_sd`. They cannot be called `sd`, which collides with the `state_dict` keys. |
| `ranksteer/eval_rs.py` changes the working directory | The module runs `os.chdir` at import time. If you need its `run_episode`, save and restore the working directory around the import. |
| The 30 instances are not the public benchmark | `data/test30/` is a development set built in house, used for parameter selection. `data/esicup/` holds the ten public instances on which every comparison with the genetic algorithm is made. |

---

## 7. Complexity of one evaluation

The cost of a single evaluation is what separates the evaluation count from the
wall-clock time. With `n` parts, grid step `g`, model width `d = 128` and `P_in` the cost
of one point-in-polygon test against a cached no-fit polygon:

```
T_eval = O( (n * x_max / g) * P_in )     BLF placement, paid by every method
       + O( n^2 * d + n * d^2 )          backbone forward
       + O( n^2 * log n )                ranking the candidate pairs
```

The ranking is paid once per rebuild, not once per evaluation, because the candidate
order is recomputed only when the ordering changes. The budget is fixed ahead of time and
does not grow with the instance, so the order of the method comes from the placement
term. See `COMPLEXITY.md` for the full derivation.

---

## 8. Availability

Code, data, checkpoints and results: <https://github.com/yuzhongbuto/GeoSwap>.
The ESICUP instances are public and come from the OR-Datasets repository
(`Cutting-and-Packing/2D-Irregular/Datasets`); they are used after expanding each item's
demand into individual parts and scaling the strip width to 1000.
