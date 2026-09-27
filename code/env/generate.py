"""
极坐标法生成凸多边形训练/验证数据集
生成的数据与 Terashima benchmark 形状完全不重叠
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import math
import random
from config import (
    GENERATED_DIR, PLATE_WIDTH, MIN_VERTICES, MAX_VERTICES,
    PART_SIZES, TRAIN_INSTANCES_PER_SIZE, VAL_INSTANCES_PER_SIZE,
    AREA_CATEGORIES, SEEDS,
)


def generate_convex_polygon(num_vertices, target_area=None, seed=None):
    """
    使用极坐标法生成凸多边形。

    在随机中心周围，按等角顺序生成半径扰动，形成凸多边形顶点。
    可选地引入椭圆弧拟合增加形状多样性。

    Args:
        num_vertices: 顶点数 (3-8)
        target_area:  目标面积范围 (min, max) 或 None
        seed:         随机种子

    Returns:
        list of (x, y) 顶点坐标
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # 随机中心
    cx = random.uniform(100, 300)
    cy = random.uniform(100, 300)

    # 基础半径
    base_radius = random.uniform(50, 200)

    # 生成角度等分点 + 随机半径扰动
    angles = []
    radii = []
    for i in range(num_vertices):
        angle = 2 * math.pi * i / num_vertices + random.uniform(-0.2, 0.2)
        angles.append(angle)
        # 半径扰动（保留凸性：控制扰动幅度）
        r = base_radius + random.uniform(-base_radius * 0.4, base_radius * 0.6)
        radii.append(max(r, 10))

    # 可选：椭圆弧拟合增加形状多样性
    if random.random() < 0.3:
        eccentricity = random.uniform(0.3, 0.9)
        for i in range(num_vertices):
            angle = angles[i]
            r_eff = base_radius / math.sqrt(
                1 - eccentricity * math.cos(angle) ** 2
            )
            radii[i] = radii[i] * 0.5 + r_eff * 0.5

    # 生成顶点
    vertices = []
    for angle, r in zip(angles, radii):
        x = cx + r * math.cos(angle)
        y = cy + r * math.sin(angle)
        vertices.append((x, y))

    # 确保凸性：按角度排序
    centroid = np.mean(vertices, axis=0)
    vertices = sorted(vertices, key=lambda p: math.atan2(p[1] - centroid[1], p[0] - centroid[0]))

    # 缩放到目标面积
    if target_area is not None:
        from geometry import polygon_area
        current_area = polygon_area(vertices)
        min_area, max_area = target_area
        target = random.uniform(min_area, max_area)
        if current_area > 0:
            scale = math.sqrt(target / current_area)
            vertices = [(x * scale, y * scale) for x, y in vertices]

    return vertices


def generate_instance(n_parts, seed=None):
    """
    生成一个包含 n_parts 个零件的完整排样实例。

    Args:
        n_parts: 零件数量
        seed:    随机种子

    Returns:
        parts:   list of list of (x, y), 每个零件的顶点
        plate_w: 板材宽度 (X方向，固定=PLATE_WIDTH)
        plate_h: 板材高度 (Y方向，固定=PLATE_WIDTH，方形)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    parts = []

    # 按比例分配面积类别
    n_large = max(1, int(n_parts * 0.15))   # 15% 大零件
    n_small = int(n_parts * 0.45)           # 45% 小零件
    n_medium = n_parts - n_large - n_small  # 40% 中零件

    area_specs = (
        [AREA_CATEGORIES["large"]] * n_large +
        [AREA_CATEGORIES["medium"]] * n_medium +
        [AREA_CATEGORIES["small"]] * n_small
    )
    random.shuffle(area_specs)

    for i in range(n_parts):
        n_vert = random.randint(MIN_VERTICES, MAX_VERTICES)
        target_area = area_specs[i]
        vertices = generate_convex_polygon(n_vert, target_area, seed=seed + i if seed else None)
        parts.append(vertices)

    return parts, PLATE_WIDTH, PLATE_WIDTH


def write_instance_file(parts, plate_w, plate_h, filepath):
    """
    写入 Terashima 格式的实例文件。

    格式：
    第1行：零件数
    第2行：板材宽度 板材高度
    后续每行：顶点数 m, x1 y1 x2 y2 ... xm ym
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"{len(parts)}\n")
        f.write(f"{int(plate_w)} {int(plate_h)}\n")
        for verts in parts:
            coords = []
            for x, y in verts:
                coords.append(str(int(round(x))))
                coords.append(str(int(round(y))))
            f.write(f"{len(verts)} {' '.join(coords)}\n")


def generate_dataset(output_dir, part_sizes, n_instances_per_size, seed_start=0):
    """
    生成一批数据集。

    Args:
        output_dir:        输出目录
        part_sizes:        list of int, 零件数分布
        n_instances_per_size: 每个规模的实例数
        seed_start:        起始随机种子

    Returns:
        list of (filepath, n_parts) 生成的文件信息
    """
    os.makedirs(output_dir, exist_ok=True)
    generated = []

    for n_parts in part_sizes:
        for idx in range(n_instances_per_size):
            seed = seed_start + idx * 1000 + n_parts * 100
            parts, plate_w, plate_h = generate_instance(n_parts, seed=seed)

            filename = f"part{n_parts:03d}_{idx:03d}.txt"
            filepath = os.path.join(output_dir, filename)
            write_instance_file(parts, plate_w, plate_h, filepath)
            generated.append((filepath, n_parts))

    print(f"生成 {len(generated)} 个实例到 {output_dir}")
    return generated


def main():
    """生成训练集和验证集"""
    # 训练集
    train_dir = os.path.join(GENERATED_DIR, "train")
    print("生成训练集...")
    generate_dataset(
        train_dir, PART_SIZES, TRAIN_INSTANCES_PER_SIZE,
        seed_start=SEEDS[0]
    )

    # 验证集（不同种子！）
    val_dir = os.path.join(GENERATED_DIR, "val")
    print("\n生成验证集...")
    generate_dataset(
        val_dir, PART_SIZES, VAL_INSTANCES_PER_SIZE,
        seed_start=SEEDS[-1]  # 不同种子确保不重叠
    )

    print("\n数据生成完成！")


if __name__ == "__main__":
    main()
