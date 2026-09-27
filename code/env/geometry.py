"""
几何计算模块 — 形心、旋转、特征提取
"""
import numpy as np
import math


def polygon_centroid(vertices):
    """
    计算多边形形心（面积加权平均法）

    Args:
        vertices: np.array of shape (N, 2) 或 list of (x, y)，逆时针顺序

    Returns:
        np.array([cx, cy])
    """
    vertices = np.array(vertices, dtype=float)
    x, y = vertices[:, 0], vertices[:, 1]
    area = 0.0
    cx = 0.0
    cy = 0.0
    n = len(vertices)

    for i in range(n):
        xi, yi = vertices[i]
        xi1, yi1 = vertices[(i + 1) % n]
        cross = xi * yi1 - xi1 * yi
        area += cross
        cx += (xi + xi1) * cross
        cy += (yi + yi1) * cross

    area *= 0.5
    if abs(area) < 1e-12:
        return np.mean(vertices, axis=0)
    cx /= (6 * area)
    cy /= (6 * area)
    return np.array([cx, cy])


def polygon_area(vertices):
    """使用 Shoelace 公式计算多边形面积"""
    vertices = np.array(vertices, dtype=float)
    x, y = vertices[:, 0], vertices[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def polygon_perimeter(vertices):
    """计算多边形周长"""
    vertices = np.array(vertices, dtype=float)
    n = len(vertices)
    return sum(
        np.linalg.norm(vertices[(i + 1) % n] - vertices[i])
        for i in range(n)
    )


def polygon_circularity(vertices):
    """
    计算圆形度 C = 4πA / P²
    C=1 表示正圆，越小越不圆
    """
    A = polygon_area(vertices)
    P = polygon_perimeter(vertices)
    if P < 1e-12:
        return 0.0
    return 4 * math.pi * A / (P ** 2)


def polygon_aspect_ratio(vertices):
    """
    基于 PCA 计算长宽比 = 较大特征值 / 较小特征值
    反映形状的"细长"程度
    """
    vertices = np.array(vertices, dtype=float)
    centered = vertices - np.mean(vertices, axis=0)
    if len(vertices) < 2:
        return 1.0
    cov = np.cov(centered.T)
    eig_vals = np.maximum(np.real(np.linalg.eigvals(cov)), 0)
    if eig_vals.min() < 1e-12:
        return 10.0  # 退化情况
    return np.sqrt(eig_vals.max()) / np.sqrt(eig_vals.min())


def rotate_points(points, angle_deg):
    """绕原点旋转点集"""
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    return np.dot(points, rot_mat.T)


def rotate_polygon(vertices, angle_deg):
    """
    绕形心旋转多边形，并平移使边界框左下角对齐原点。

    Args:
        vertices: list/np.array of (x, y)
        angle_deg: 旋转角度（度）

    Returns:
        np.array: 旋转+归一化后的顶点
    """
    vertices = np.array(vertices, dtype=float)
    centroid = polygon_centroid(vertices)
    centered = vertices - centroid
    rotated = rotate_points(centered, angle_deg)
    min_x = np.min(rotated[:, 0])
    min_y = np.min(rotated[:, 1])
    return rotated - [min_x, min_y]


def ray_segment_intersection(origin, direction, p1, p2):
    """
    计算射线与线段的交点。

    射线: O + t*d, t >= 0
    线段: p1 + s*(p2-p1), 0 <= s <= 1

    Returns:
        (t, s) 或 None（平行/不相交）
    """
    O = np.array(origin, dtype=float)
    d = np.array(direction, dtype=float)
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)

    v = p2 - p1
    A = np.column_stack([d, -v])
    try:
        t_s = np.linalg.solve(A, p1 - O)
    except np.linalg.LinAlgError:
        return None

    t, s = t_s[0], t_s[1]
    if t >= 0 and 0 <= s <= 1:
        return t, s
    return None


def ray_distance_to_polygon(origin, direction, vertices):
    """
    从原点沿方向发射射线，返回到多边形边界的最大距离
    （用于凸多边形质心在内部的情况，取最远交点）。

    Args:
        origin: (x, y) 射线起点（质心）
        direction: (dx, dy) 射线方向
        vertices: 多边形顶点

    Returns:
        float: 距离
    """
    O = np.array(origin, dtype=float)
    max_t = 0.0
    n = len(vertices)

    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]
        res = ray_segment_intersection(origin, direction, p1, p2)
        if res is not None:
            t, _ = res
            if t > 1e-8:
                max_t = max(max_t, t)

    if max_t < 1e-8:
        # 数值回退：到所有顶点的最大距离
        return np.max(np.linalg.norm(np.array(vertices) - O, axis=1))

    return max_t


def extract_shape_vector(vertices, num_rays=128, normalize=True):
    """
    提取质心-轮廓距离特征向量。

    从质心以等角间隔发射 num_rays 条射线，记录每条射线
    到多边形边界的最远交点的距离。

    Args:
        vertices: 多边形顶点坐标
        num_rays: 射线数（特征维度）
        normalize: True 时按每个零件的 max_dist 归一化到 [0, 1]（原文设置，
                   会抹掉零件的绝对尺寸信息）；False 时保留原始距离
                   （保留尺寸信息，A 系列实验用）。

    Returns:
        np.array of shape (num_rays,)
    """
    centroid = polygon_centroid(vertices)
    distances = []

    for j in range(num_rays):
        angle = 2 * math.pi * j / num_rays
        direction = np.array([math.cos(angle), math.sin(angle)])
        dist = ray_distance_to_polygon(centroid, direction, vertices)
        distances.append(dist)

    distances = np.array(distances, dtype=np.float32)
    if normalize:
        max_dist = np.max(distances)
        if max_dist > 0:
            distances /= max_dist

    return distances


def compute_geometric_features(vertices):
    """
    计算全局几何特征：面积、长宽比、圆形度。

    Returns:
        (area, aspect_ratio, circularity)
    """
    area = polygon_area(vertices)
    aspect = polygon_aspect_ratio(vertices)
    circ = polygon_circularity(vertices)
    return area, aspect, circ


def extract_full_features(vertices, num_rays=128, normalize=True):
    """提取特征：[num_rays维轮廓距离]"""
    shape_vec = extract_shape_vector(vertices, num_rays, normalize=normalize)
    return shape_vec.astype(np.float32)


# ======================== V4 几何残差特征 ========================

def _cross_2d(a, b):
    """2D 叉积: a_x*b_y - a_y*b_x"""
    return a[0] * b[1] - a[1] * b[0]


def compute_concave_vertices(vertices, angle_threshold_deg=210.0):
    """
    检测凹顶点（带角度阈值，过滤锯齿噪声）。

    仅当内角 > angle_threshold_deg 时计为凹点。默认 210°，
    忽略微小锯齿（~180-200°），只捕获真正的凹陷。

    Returns:
        n_concave: int, 凹顶点数
        max_recess_depth: float, 最大凹陷深度（凹点到凸包对应边的距离）
    """
    verts = np.array(vertices, dtype=float)
    n = len(verts)
    if n < 4:
        return 0, 0.0

    n_concave = 0
    max_recess = 0.0

    for i in range(n):
        prev = verts[(i - 1) % n]
        curr = verts[i]
        next_v = verts[(i + 1) % n]
        edge_in = curr - prev
        edge_out = next_v - curr
        cross = _cross_2d(edge_in, edge_out)
        if cross > 0:  # 右手系正 → curr 处内凹
            # 计算内角
            dot = np.dot(edge_in, edge_out)
            norms = np.linalg.norm(edge_in) * np.linalg.norm(edge_out)
            if norms > 1e-12:
                cos_a = max(-1.0, min(1.0, dot / norms))
                angle_deg = math.degrees(math.acos(cos_a))
                # angle_deg is the exterior angle of the bend, interior is 360-exterior
                interior_deg = 360.0 - angle_deg
                if interior_deg > angle_threshold_deg:
                    n_concave += 1

    return n_concave, 0.0  # max_recess requires convex hull which is costly — compute separately if needed


def compute_recess_depth(vertices):
    """
    用凸包法计算最大凹陷深度。
    凹点到其外接凸包对应边的最大垂直距离。

    Returns:
        max_recess: float, 归一化到 [0, 1]
    """
    from scipy.spatial import ConvexHull
    verts = np.array(vertices, dtype=float)
    n = len(verts)
    if n < 4:
        return 0.0
    try:
        hull = ConvexHull(verts)
        hull_verts = verts[hull.vertices]
    except Exception:
        return 0.0

    # 凸包边
    max_dist = 0.0
    m = len(hull_verts)
    for i in range(m):
        p1 = hull_verts[i]
        p2 = hull_verts[(i + 1) % m]
        edge = p2 - p1
        edge_len = np.linalg.norm(edge)
        if edge_len < 1e-8:
            continue
        # 遍历所有顶点，计算到该边的距离
        for v in verts:
            # 点到线段距离
            t = np.dot(v - p1, edge) / (edge_len ** 2)
            t = max(0.0, min(1.0, t))
            proj = p1 + t * edge
            dist = np.linalg.norm(v - proj)
            max_dist = max(max_dist, dist)

    # 归一化：除以凸包外接矩形最长边
    hs_x_min, hs_y_min = hull_verts.min(axis=0)
    hs_x_max, hs_y_max = hull_verts.max(axis=0)
    hs_diag = max(hs_x_max - hs_x_min, hs_y_max - hs_y_min, 1.0)
    return max_dist / hs_diag


def compute_fill_rate(vertices):
    """面积填充率 = 实际面积 / 外接矩形面积"""
    verts = np.array(vertices, dtype=float)
    area = polygon_area(verts)
    xs, ys = verts[:, 0], verts[:, 1]
    bbox_w = max(xs) - min(xs)
    bbox_h = max(ys) - min(ys)
    if bbox_w < 1e-8 or bbox_h < 1e-8:
        return 1.0
    return area / (bbox_w * bbox_h)


def compute_longest_edge_ratio(vertices):
    """最长边占比 = 最长边长度 / 总周长"""
    verts = np.array(vertices, dtype=float)
    n = len(verts)
    peri = polygon_perimeter(verts)
    if peri < 1e-12:
        return 0.0
    max_edge = max(
        np.linalg.norm(verts[(i + 1) % n] - verts[i]) for i in range(n)
    )
    return max_edge / peri


def compute_residual_features(vertices):
    """
    计算 6 维几何残差特征（原始值，不含归一化）。

    Returns:
        dict with keys:
          area       — 零件面积（raw）
          fill_rate  — 面积 / bbox面积
          circularity— 4πA/P²（可用 invert: P²/4πA）
          n_concave  — 凹顶点数
          recess     — 最大凹陷深度比
          edge_ratio — 最长边 / 周长
    """
    area = polygon_area(vertices)
    fill_rate = compute_fill_rate(vertices)
    circ = polygon_circularity(vertices)
    n_concave, _ = compute_concave_vertices(vertices)
    recess = compute_recess_depth(vertices)
    edge_ratio = compute_longest_edge_ratio(vertices)
    # 用 inverted circularity = P²/(4πA)，值越大形状越复杂（和面积无关）
    inv_circ = 1.0 / max(circ, 1e-8)
    return {
        'area': area,
        'fill_rate': fill_rate,
        'inv_circularity': inv_circ,
        'n_concave': n_concave,
        'recess': recess,
        'edge_ratio': edge_ratio,
    }


# ======================== V4 天际线特征 ========================

def compute_skyline(placed_polys, bin_width, bin_height, n_bins=16):
    """
    计算天际线特征：将板材宽度 n_bins 等分，每格取最高占用高度。

    Args:
        placed_polys: list of shapely Polygon
        bin_width:    板材宽度
        bin_height:   板材高度
        n_bins:       采样分辨率

    Returns:
        skyline: np.array (n_bins,) 归一化到 [0, 1]
    """
    if not placed_polys:
        return np.zeros(n_bins, dtype=np.float32)
    cell_w = bin_width / n_bins
    skyline = np.zeros(n_bins, dtype=np.float32)
    # 对每列采样
    for i in range(n_bins):
        x_center = (i + 0.5) * cell_w
        # 粗粒度：用最小外接矩形估算
        max_y = 0.0
        for p in placed_polys:
            bounds = p.bounds  # (minx, miny, maxx, maxy)
            if bounds[0] <= x_center <= bounds[2]:
                max_y = max(max_y, bounds[3])
        skyline[i] = max_y / max(bin_height, 1.0)
    return skyline


def compute_skyline_match_score(vertices, skyline, bin_width, bin_height, n_bins=16):
    """
    计算零件与天际线的匹配度。

    在 skyline 中找最低谷宽度，看零件 bbox 是否能嵌入低谷。

    Returns:
        match_score: float, 归一化到 [0, 1]
    """
    verts = np.array(vertices, dtype=float)
    xs = verts[:, 0]; ys = verts[:, 1]
    part_w = max(xs) - min(xs)
    cell_w = bin_width / n_bins
    bins_needed = max(1, int(part_w / cell_w + 0.5))

    best_valley = 1.0  # lower is better (deeper valley)
    for i in range(n_bins - bins_needed + 1):
        valley_depth = max(skyline[i:i + bins_needed])
        best_valley = min(best_valley, valley_depth)

    return 1.0 - best_valley  # 1.0 = perfect valley available
