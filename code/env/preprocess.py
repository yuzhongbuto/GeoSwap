"""
特征提取预处理：将原始多边形文件转换为特征向量
输出 _feat.txt 文件，每行一个零件的131维特征
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from geometry import extract_full_features, rotate_polygon
from config import NUM_RAYS, NUM_ANGLES, ANGLE_DEGREES, D_FEAT


def parse_instance_file(filepath):
    """
    解析 Terashima 格式实例文件。

    Returns:
        parts: list of list of (x, y)
        plate_w, plate_h: int
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    n_parts = int(lines[0])
    plate_w, plate_h = map(int, lines[1].split())
    parts = []

    for i in range(2, 2 + n_parts):
        tokens = list(map(int, lines[i].split()))
        m = tokens[0]
        coords = tokens[1:]
        vertices = [(coords[j], coords[j+1]) for j in range(0, 2*m, 2)]
        parts.append(vertices)

    return parts, plate_w, plate_h


def process_instance(input_filepath, output_dir, num_rays=NUM_RAYS):
    """
    处理单个实例文件：提取特征并保存。

    生成两个文件：
    - {name}_feat.txt: 每个零件的131维特征向量
    - {name}_feat_all_angles.txt: 每个零件4个角度的特征（如需要）

    Args:
        input_filepath: 原始多边形文件路径
        output_dir:     输出目录
        num_rays:       质心射线数
    """
    parts, plate_w, plate_h = parse_instance_file(input_filepath)
    base_name = os.path.splitext(os.path.basename(input_filepath))[0]

    features = []
    features_all_angles = []  # [n_parts, 4, D_FEAT]

    for vertices in parts:
        # 0° 特征
        feat = extract_full_features(vertices, num_rays)
        features.append(feat)

        # 所有4个角度的特征
        angle_feats = []
        for angle in ANGLE_DEGREES:
            rotated = rotate_polygon(vertices, angle)
            feat_rot = extract_full_features(rotated, num_rays)
            angle_feats.append(feat_rot)
        features_all_angles.append(angle_feats)

    features = np.array(features, dtype=np.float32)
    features_all_angles = np.array(features_all_angles, dtype=np.float32)

    # 保存
    feat_path = os.path.join(output_dir, f"{base_name}_feat.txt")
    np.savetxt(feat_path, features, fmt='%.6f')

    # 保存所有角度特征（供模型直接使用）
    # 格式：每个零件4行（4个角度），每行 D_FEAT 维
    feat_all_path = os.path.join(output_dir, f"{base_name}_feat_all.txt")
    reshaped = features_all_angles.reshape(-1, D_FEAT)
    np.savetxt(feat_all_path, reshaped, fmt='%.6f')

    return features, features_all_angles


def process_directory(input_dir, output_dir, num_rays=NUM_RAYS):
    """
    批量处理目录下所有 .txt 实例文件。
    """
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir)
             if f.endswith('.txt') and '_feat' not in f
             and not f.startswith('summary')]

    for f in sorted(files):
        filepath = os.path.join(input_dir, f)
        try:
            process_instance(filepath, output_dir, num_rays)
        except Exception as e:
            print(f"处理 {f} 失败: {e}")

    print(f"处理完成：{len(files)} 个实例 -> {output_dir}")


def normalize_features(features):
    """
    Z-score 归一化。

    Args:
        features: np.array of shape (N, D_FEAT)

    Returns:
        normalized: np.array
        mean, std:  用于后续归一化的统计量
    """
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (features - mean) / std, mean, std


if __name__ == "__main__":
    # 处理训练集
    train_raw = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "generated", "train"
    )
    train_out = train_raw  # 同目录输出
    process_directory(train_raw, train_out)

    # 处理验证集
    val_raw = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "generated", "val"
    )
    val_out = val_raw
    process_directory(val_raw, val_out)
