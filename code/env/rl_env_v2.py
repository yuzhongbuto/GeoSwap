"""
RL 环境 V4 — 单头 Actor + BLF(角度+定位) + 多特征残差 + 天际线

V4 新增:
  - 6 维零件几何特征 (area/fill_rate/inv_circ/n_concave/recess/edge_ratio)
  - 天际线提取 (16维)
  - 天际线匹配度 (1维)
  - 全局统计 (剩余零件平均 fill_rate)
  - 上一步 delta_x 反馈
"""
import os, sys, math, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from shapely.geometry import Polygon, Point
from shapely.affinity import translate
from shapely.strtree import STRtree
from shapely.prepared import prep

from geometry import (
    polygon_centroid, polygon_area, rotate_polygon,
    extract_shape_vector, compute_residual_features,
    compute_skyline, compute_skyline_match_score,
)
from nfp_utils import calculate_nfp
from training.reward import compute_step_reward, compute_terminal_bonus
from config import (
    NUM_RAYS, NUM_ANGLES, D_FEAT, PLACEMENT_STEP_FAST,
    PLATE_WIDTH, SKYLINE_BINS, ZSCORE_MIN_STD,
)


class PackingEnvV2:
    """
    2D Irregular Strip Packing RL 环境 V4。
    模型管排序，BLF 遍历 4 角度管放置。
    """

    def __init__(self, instance_path, placement_mode='blf', use_rotation=True,
                 normalize_shape=None, feature_scale=None):
        self.instance_path = instance_path
        self.placement_mode = placement_mode
        self.use_rotation = use_rotation
        # normalize_shape: None → 读环境变量 PACKING_NORMALIZE_SHAPE（默认 True）
        #   True  = 按零件各自 max_dist 归一化（原文设置，抹掉绝对尺寸信息）
        #   False = 保留原始质心距离（保留尺寸信息，A 系列实验用）
        if normalize_shape is None:
            normalize_shape = os.environ.get('PACKING_NORMALIZE_SHAPE', '1') == '1'
        self.normalize_shape = normalize_shape
        # feature_scale: None → 读环境变量 PACKING_FEATURE_SCALE（默认 zscore）
        #   zscore  = 实例内逐维 z-score（原文设置——逐维标准化会抹掉尺寸信息！
        #             实测 rho(area, |z|)≈0，这是 A3 失败的第二层原因）
        #   maxnorm = 实例内除以全局最大值（保留相对尺寸，rho(area,|v|)=1.0，
        #             且尺度不变、数值稳定在 [0,1]，本阶段新管线用）
        if feature_scale is None:
            feature_scale = os.environ.get('PACKING_FEATURE_SCALE', 'zscore')
        self.feature_scale = feature_scale
        self._load_instance()

    def _load_instance(self):
        from data.preprocess import parse_instance_file
        self.orig_parts, self.plate_width, self.plate_height = parse_instance_file(self.instance_path)
        self.bin_width = PLATE_WIDTH
        self.bin_height = self.plate_height
        self.n = len(self.orig_parts)

        # 128维质心距离特征
        features_list = []
        for verts in self.orig_parts:
            dist_vec = extract_shape_vector(verts, NUM_RAYS,
                                            normalize=self.normalize_shape)
            features_list.append(dist_vec)
        self.features_raw = torch.tensor(np.array(features_list), dtype=torch.float32)
        if self.feature_scale == 'maxnorm':
            # ---- "形状解耦 + 显式面积列"协议（LayerNorm 穿透，2026-08-15 本阶段）----
            # 研究发现（三层归一化缺陷）：
            #   1) per-part normalize 抹掉尺寸（论文已诊断）
            #   2) 实例级逐维 z-score 再抹一层（A3 失败原因，rho(area,|z|)≈0）
            #   3) nn.Transformer 的逐 token LayerNorm 会把"范数编码"的尺寸信息
            #      完全抹掉（rho(area,|memory|)=0.000），甚至反相显式面积列
            #      （maxnorm 特征与面积耦合时 rho=-0.63/-0.34）。
            # 解法：形状特征用 per-part normalize（与面积**解耦**，LN 安全），
            # 面积用显式列 area/max(area) 放第 0 维（LN 不干扰正交维度）。
            # 这正是旧论文 131 维（128 形状+面积列）能学会的机制，但我们保持
            # 128 维 + 单头。注意：此协议强制形状 per-part normalize。
            shape_list = []
            for verts in self.orig_parts:
                shape_list.append(extract_shape_vector(verts, NUM_RAYS,
                                                       normalize=True))
            shapes = torch.tensor(np.array(shape_list), dtype=torch.float32)
            raw_areas = np.array([polygon_area(v) for v in self.orig_parts],
                                 dtype=np.float32)
            a_max = max(raw_areas.max(), 1e-6)
            # A5 消融①：PACKING_NO_AREA=1 时不覆盖第 0 维（即无显式面积列，
            # 第 0 维保持 per-part 归一化形状值）→ 与训练端 --no_area 一致。
            if os.environ.get('PACKING_NO_AREA', '0') != '1':
                shapes[:, 0] = torch.tensor(raw_areas / a_max, dtype=torch.float32)
            self.features = shapes
        else:  # 'zscore'（默认，保持原文行为）
            mean = self.features_raw.mean(dim=0, keepdim=True)
            std = self.features_raw.std(dim=0, keepdim=True)
            std[std == 0] = 1.0
            self.features = (self.features_raw - mean) / std

        # ---- V4: 6 维几何残差特征 (实例内 z-score, min_std=1e-3) ----
        raw_feats = [compute_residual_features(v) for v in self.orig_parts]
        self.residual_raw = np.array([
            [f['area'], f['fill_rate'], f['inv_circularity'],
             f['n_concave'], f['recess'], f['edge_ratio']]
            for f in raw_feats
        ], dtype=np.float32)

        # z-score normalize per instance
        self.residual_feats = np.zeros_like(self.residual_raw)
        for k in range(6):
            col = self.residual_raw[:, k]
            m = col.mean()
            s = max(col.std(), ZSCORE_MIN_STD)
            self.residual_feats[:, k] = (col - m) / s

        # 归一化面积（保持兼容）
        raw_areas = self.residual_raw[:, 0].copy()
        mean_area = raw_areas.mean()
        self.part_areas = raw_areas / max(mean_area, 1.0)

        # ---- 预计算旋转顶点 ----
        self.rotated_vertices = []
        self.part_bounds = []
        for i in range(self.n):
            verts_i, bounds_i = [], []
            n_angles = 1 if not self.use_rotation else NUM_ANGLES
            for ai in range(n_angles):
                angle = 0 if not self.use_rotation else ai * 90
                rotated = rotate_polygon(self.orig_parts[i], angle)
                verts_i.append(rotated)
                xs, ys = rotated[:, 0], rotated[:, 1]
                bounds_i.append((min(xs), min(ys), max(xs), max(ys)))
            self.rotated_vertices.append(verts_i)
            self.part_bounds.append(bounds_i)

        # NFP 缓存
        self.nfp_cache = {}
        self.buffered_nfp_cache = {}

        # ---- 天际线状态（每步更新）----
        self.skyline = np.zeros(SKYLINE_BINS, dtype=np.float32)
        self.last_delta_x = 0.0

    def reset(self):
        self.remaining_indices = list(range(self.n))
        self.placed_info = []
        self.placed_polys = []
        self.placed_tree = None
        self.current_max_x = 0.0
        self.total_area = 0.0
        self.skyline = np.zeros(SKYLINE_BINS, dtype=np.float32)
        self.last_delta_x = 0.0
        return self._get_state_v4()

    def _get_state(self):
        remaining_feat = self.features[self.remaining_indices]
        if self.placed_info:
            tgt = torch.tensor([info[0] for info in self.placed_info], dtype=torch.long)
        else:
            tgt = torch.tensor([], dtype=torch.long)
        return remaining_feat, tgt

    def _get_state_v4(self):
        """返回 V4 完整状态"""
        rem_feat, tgt = self._get_state()

        # 剩余零件残差特征 [N_rem, 6]
        rem_res = torch.tensor(self.residual_feats[self.remaining_indices], dtype=torch.float32)

        # 天际线
        skyline = self.skyline.copy()

        # 全局统计：剩余零件平均 fill_rate
        avg_fill = float(rem_res[:, 1].mean()) if rem_res.size(0) > 0 else 0.0

        # 上一步 delta_x
        last_dx = self.last_delta_x / max(self.bin_height, 1.0)

        return rem_feat, tgt, rem_res, skyline, avg_fill, last_dx

    @staticmethod
    def _compute_skyline_match_scores(part_indices, rotated_vertices, skyline,
                                       bin_width, bin_height):
        """
        为每个剩余零件计算天际线匹配分 (第7维残差特征)。
        """
        n_rem = len(part_indices)
        scores = np.zeros(n_rem, dtype=np.float32)
        cell_w = bin_width / SKYLINE_BINS

        for i, pid in enumerate(part_indices):
            verts = rotated_vertices[pid][0]  # angle 0 bbox
            part_w = max(verts[:, 0]) - min(verts[:, 0])
            bins_needed = max(1, int(part_w / cell_w + 0.5))
            best_valley = 1.0
            for j in range(SKYLINE_BINS - bins_needed + 1):
                valley_depth = max(skyline[j:j + bins_needed])
                best_valley = min(best_valley, valley_depth)
            scores[i] = 1.0 - best_valley

        # z-score normalize
        m = scores.mean()
        s = max(scores.std(), ZSCORE_MIN_STD)
        if s > 1e-6:
            scores = (scores - m) / s
        return scores

    def _get_nfp(self, fixed_idx, fixed_angle, moving_idx, moving_angle):
        key = (fixed_idx, fixed_angle, moving_idx, moving_angle)
        if key not in self.buffered_nfp_cache:
            poly_fixed = Polygon(self.rotated_vertices[fixed_idx][fixed_angle])
            poly_moving = Polygon(self.rotated_vertices[moving_idx][moving_angle])
            nfp = calculate_nfp(poly_fixed, poly_moving, ref_point=(0, 0))
            self.nfp_cache[key] = nfp
            self.buffered_nfp_cache[key] = nfp if nfp.is_empty else nfp.buffer(1e-6)
        return self.buffered_nfp_cache[key]

    def _blf_first_feasible_fast(self, part_idx, angle_idx, part_w, part_h, vx):
        """向量化版：按 x 升序、同一列内 y 升序，取第一个可行点。

        与原版逐点扫描**语义一致**（x 优先、y 次之、step 相同、NFP 集合相同），
        但把"逐点 Python 循环 + prep.contains"换成"按列向量化 contains_xy"，
        并先用 NFP 的 x/y 包围盒剪枝。2026-09-22 实测：同结果、速度提升一个量级。
        由 PACKING_FAST_BLF=1 开启（默认关闭，保证既有口径不变）。
        """
        from shapely import contains_xy
        nfps = []
        for j, aj, xj, yj in self.placed_info:
            nfp = self._get_nfp(j, aj, part_idx, angle_idx)
            if nfp.is_empty:
                continue
            g = translate(nfp, xj, yj)
            nfps.append((g.bounds, g))

        step = PLACEMENT_STEP_FAST
        y_max = int(self.bin_height - part_h)
        if y_max < 0:
            return None
        ys = np.arange(0, y_max + 1, step, dtype=np.float64)
        if ys.size == 0:
            return None
        y0, y1 = float(ys[0]), float(ys[-1])

        # y 方向剪枝（PACKING_BLF_YPRUNE=1，默认开）：
        # 原版对**整列** ys（~200 个点）算 contains_xy，但单个 NFP 的 y 跨度通常只有
        # 几个栅格 —— 绝大多数点必然为 False，白算。改成只对落在该 NFP y 包围盒内的
        # 那一段做测试，再把结果 OR 回对应区段。逐点判定完全一致，只是少算了空点。
        # 置 PACKING_BLF_YPRUNE=0 回退到原版逐列全算（逐位一致性验证用）。
        if os.environ.get('PACKING_BLF_YPRUNE', '1') == '1':
            nfps_pruned = []
            for (minx, miny, maxx, maxy), g in nfps:
                if maxy < y0 or miny > y1:
                    continue
                i0 = int(np.searchsorted(ys, miny, side='left'))
                i1 = int(np.searchsorted(ys, maxy, side='right'))
                if i1 > i0:
                    nfps_pruned.append((minx, maxx, i0, i1, g))
        else:
            nfps_pruned = [(minx, maxx, 0, ys.size, g)
                           for (minx, miny, maxx, maxy), g in nfps
                           if not (maxy < y0 or miny > y1)]

        for attempt in range(5):
            x_limit = int(self.current_max_x + part_w + 500 + attempt * 300)
            for x in range(0, x_limit + 1, step):
                blocked = None
                for minx, maxx, i0, i1, g in nfps_pruned:
                    if x < minx or x > maxx:
                        continue
                    inside = contains_xy(g, float(x), ys[i0:i1])
                    if not inside.any():
                        continue
                    if blocked is None:
                        blocked = np.zeros(ys.size, dtype=bool)
                    blocked[i0:i1] |= inside
                    if blocked.all():
                        break
                if blocked is None:
                    return (float(x), y0)
                free = np.flatnonzero(~blocked)
                if free.size:
                    return (float(x), float(ys[free[0]]))
        return None

    def _blf_placement_best_angle_fast(self, part_idx):
        """4 角度遍历 + 向量化扫描（与 _blf_placement_best_angle 输出相同）。"""
        best_result = None
        n_angles = 1 if not self.use_rotation else NUM_ANGLES
        for angle_idx in range(n_angles):
            vertices = self.rotated_vertices[part_idx][angle_idx]
            vx, vy = vertices[:, 0], vertices[:, 1]
            part_w = max(vx) - min(vx)
            part_h = max(vy) - min(vy)
            pos = self._blf_first_feasible_fast(part_idx, angle_idx, part_w, part_h, vx)
            if pos is None:
                continue
            x, y = pos
            new_max_x = max(self.current_max_x, max(vx) + x)
            delta_x = new_max_x - self.current_max_x
            if best_result is None or delta_x < best_result[3]:
                best_result = (x, y, angle_idx, delta_x)
        if best_result is not None:
            x, y, angle_idx, _ = best_result
            return (x, y), angle_idx, True
        return (0, 0), 0, False

    def _blf_placement_best_angle(self, part_idx):
        """BLF 遍历 4 角度 + 扩展搜索范围重试"""
        if os.environ.get('PACKING_FAST_BLF', '0') == '1':
            return self._blf_placement_best_angle_fast(part_idx)
        best_result = None

        n_angles = 1 if not self.use_rotation else NUM_ANGLES
        for angle_idx in range(n_angles):
            vertices = self.rotated_vertices[part_idx][angle_idx]
            xs, ys = vertices[:, 0], vertices[:, 1]
            part_w = max(xs) - min(xs)
            part_h = max(ys) - min(ys)

            active_nfps = []
            for j, aj, xj, yj in self.placed_info:
                nfp = self._get_nfp(j, aj, part_idx, angle_idx)
                if not nfp.is_empty:
                    active_nfps.append(translate(nfp, xj, yj))
            prep_nfps = [(nfp.bounds, prep(nfp)) for nfp in active_nfps]

            step = PLACEMENT_STEP_FAST
            found = False

            for attempt in range(5):
                x_limit = int(self.current_max_x + part_w + 500 + attempt * 300)
                for x in range(0, x_limit + 1, step):
                    for y in range(0, int(self.bin_height - part_h) + 1, step):
                        pt = Point(x, y)
                        ok = True
                        for bounds, p_nfp in prep_nfps:
                            minx, miny, maxx, maxy = bounds
                            if minx <= x <= maxx and miny <= y <= maxy and p_nfp.contains(pt):
                                ok = False; break
                        if ok:
                            new_max_x = max(self.current_max_x, max(xs) + x)
                            delta_x = new_max_x - self.current_max_x
                            if best_result is None or delta_x < best_result[3]:
                                best_result = (x, y, angle_idx, delta_x)
                            found = True; break
                    if found: break
                if found: break

        if best_result is not None:
            x, y, angle_idx, _ = best_result
            return (x, y), angle_idx, True
        return (0, 0), 0, False

    def step(self, rem_idx):
        """
        执行一步排样决策。

        Args:
            rem_idx: 在 remaining_indices 中的索引

        Returns:
            state_v4, reward, done, info

        P0 修复（2026-08-16）：无法放置 / 退化（零面积）零件不再终止整局，
        改为 drop 策略——跳过该零件、继续放置其余零件（所有方法共用同一协议）。
        根因：生成数据偶发重复顶点多边形（如 (529,279),(529,279),(534,255)），
        shapely 判定 invalid 且 area=0，旧逻辑把整局判死 → util=0。
        """
        part_idx = self.remaining_indices[rem_idx]
        old_max_x = self.current_max_x

        def _drop():
            """跳过当前零件：从 remaining 移除，不放置、不终止（除非无零件剩余）。"""
            self.remaining_indices.pop(rem_idx)
            done = len(self.remaining_indices) == 0
            reward = -1.0
            if done:
                reward += compute_terminal_bonus(
                    self.total_area, self.current_max_x, self.bin_height)
            util = self.total_area / (self.current_max_x * self.bin_height) \
                   if self.current_max_x > 0 else 0.0
            info = {'success': True, 'dropped': True, 'part_idx': part_idx,
                    'utilization': util, 'current_max_x': self.current_max_x}
            return self._get_state_v4(), reward, done, info

        (best_x, best_y), best_angle, success = self._blf_placement_best_angle(part_idx)

        if not success:
            return _drop()

        vertices = self.rotated_vertices[part_idx][best_angle]

        # 几何处理
        poly = Polygon(vertices + [best_x, best_y])
        if not poly.is_valid:
            poly = poly.buffer(0)
            if poly.is_empty or poly.area < 1e-12:
                return _drop()

        part_area = poly.area
        self.total_area += part_area

        # 接触面积
        total_contact = 0.0
        if self.placed_polys:
            new_bounds = poly.bounds
            candidates = (
                [self.placed_polys[i] for i in self.placed_tree.query(poly)]
                if self.placed_tree else self.placed_polys
            )
            for p in candidates:
                p_bounds = p.bounds
                if (new_bounds[2] + 0.1 < p_bounds[0] - 0.1 or
                    new_bounds[0] - 0.1 > p_bounds[2] + 0.1 or
                    new_bounds[3] + 0.1 < p_bounds[1] - 0.1 or
                    new_bounds[1] - 0.1 > p_bounds[3] + 0.1):
                    continue
                try:
                    inter = poly.intersection(p.buffer(0.1))
                    if not inter.is_empty:
                        total_contact += inter.area
                except Exception:
                    continue

        # 更新状态
        self.placed_info.append((part_idx, best_angle, best_x, best_y))
        self.placed_polys.append(poly)
        self.placed_tree = STRtree(self.placed_polys)
        self.remaining_indices.pop(rem_idx)
        self.current_max_x = float(max(self.current_max_x, max(vertices[:, 0]) + best_x))
        delta_x = max(0, self.current_max_x - old_max_x)
        self.last_delta_x = delta_x

        # 更新天际线
        self.skyline = compute_skyline(
            self.placed_polys, self.bin_width, self.bin_height, SKYLINE_BINS
        )

        # 奖励
        reward = compute_step_reward(delta_x, self.bin_height, total_contact, part_area)
        done = len(self.remaining_indices) == 0
        if done:
            bonus = compute_terminal_bonus(self.total_area, self.current_max_x, self.bin_height)
            reward += bonus

        if not np.isfinite(reward):
            reward = -0.5; done = True

        info = {
            'success': True,
            'current_max_x': self.current_max_x,
            'utilization': self.total_area / (self.current_max_x * self.bin_height)
                           if self.current_max_x > 0 else 0.0,
            'delta_x': delta_x,
            'contact_area': total_contact,
            'part_area': part_area,
        }
        return self._get_state_v4(), reward, done, info
