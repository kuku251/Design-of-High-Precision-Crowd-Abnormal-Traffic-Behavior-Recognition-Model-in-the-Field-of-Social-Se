# -*- coding: utf-8 -*-
# data_loader.py - 三分类最终优化版：支持 enhance_level 消融实验
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

# ===================== 全局配置 =====================
USE_SELF_DATASET = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEAT_PATH = "./chapter3_results/fused_multimodal_features_with_resnet50.csv"
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
TRAIN_NORMAL_DOWNSAMPLE_RATIO = 1.2
BATCH_SIZE = 32
RANDOM_SEED = 42
NUM_CLASSES = 3
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)

# ===================== 三分类标签映射 =====================
ANOMALY_LABELS = ["奔跑", "徘徊", "跌倒", "人群拥挤"]
def map_label(anomaly_type):
    if anomaly_type == "正常行走":
        return 0
    elif anomaly_type in ANOMALY_LABELS:
        return 1
    elif anomaly_type == "车辆闯入":
        return 2
    else:
        raise ValueError(f"未知标签: {anomaly_type}")

CLASS_NAMES = ['正常行走', '人体异常', '车辆闯入']
SCALER_SAVE_PATH = "./chapter4_results/models/standard_scaler.pth"
Path(SCALER_SAVE_PATH).parent.mkdir(exist_ok=True, parents=True)
BASE_SAVE = Path(SCALER_SAVE_PATH).parent  # 即 ./chapter4_results/models/

# ===================== 数据增强函数（强增强，原版） =====================
def augment_data_strong(X, y_cls, y_risk):
    """车辆增强3次，人体异常增强2次"""
    # 车辆(2)增强3次
    car_mask = (y_cls == 2)
    if car_mask.any():
        X_car, y_car, r_car = X[car_mask], y_cls[car_mask], y_risk[car_mask]
        for _ in range(3):
            noise = np.random.normal(0, 0.01, X_car.shape)
            X = np.concatenate([X, X_car + noise], axis=0)
            y_cls = np.concatenate([y_cls, y_car], axis=0)
            y_risk = np.concatenate([y_risk, r_car], axis=0)
    # 人体异常(1)增强2次
    human_mask = (y_cls == 1)
    if human_mask.any():
        X_human, y_human, r_human = X[human_mask], y_cls[human_mask], y_risk[human_mask]
        for _ in range(2):
            noise = np.random.normal(0, 0.008, X_human.shape)
            X = np.concatenate([X, X_human + noise], axis=0)
            y_cls = np.concatenate([y_cls, y_human], axis=0)
            y_risk = np.concatenate([y_risk, r_human], axis=0)
    return X, y_cls, y_risk

# ===================== 核心加载器（支持 enhance_level） =====================
def build_dataloader(batch_size=BATCH_SIZE, enhance_level=2):
    """
    enhance_level:
        0 - 基线 (baseline): 无数据增强，无类别权重，损失函数使用标准交叉熵
        1 - 数据增强 (data_aug): 强数据增强，类别权重按逆频率计算（无手动放大），损失函数使用 FocalLoss
        2 - 完整优化 (full_opt): 强数据增强 + 类别权重放大（人体×1.2，车辆×2.0）+ FocalLoss
    """
    df = pd.read_csv(FEAT_PATH, encoding="utf-8-sig").dropna().reset_index(drop=True)
    if not USE_SELF_DATASET:
        df = df[df["dataset"] != "dataset_self"].reset_index(drop=True)
    
    df["cls_label"] = df["anomaly_type"].apply(map_label)
    df["risk_label"] = df["cls_label"]
    
    # 全量视频划分
    all_videos = df["video_name"].unique()
    np.random.shuffle(all_videos)
    n_total = len(all_videos)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)
    
    train_videos = set(all_videos[:n_train])
    val_videos = set(all_videos[n_train:n_train+n_val])
    test_videos = set(all_videos[n_train+n_val:])
    
    train_df = df[df["video_name"].isin(train_videos)].reset_index(drop=True)
    val_df = df[df["video_name"].isin(val_videos)].reset_index(drop=True)
    test_df = df[df["video_name"].isin(test_videos)].reset_index(drop=True)
    
    # 训练集正常类下采样（保持原逻辑）
    train_class_counts = train_df["cls_label"].value_counts().sort_index()
    train_normal_count = train_class_counts[0] if 0 in train_class_counts.index else 0
    train_human = train_class_counts[1] if 1 in train_class_counts.index else 0
    train_car = train_class_counts[2] if 2 in train_class_counts.index else 0
    target_train_normal = int(train_human + train_car * 3)
    
    if train_normal_count > target_train_normal and target_train_normal > 0:
        train_normal_df = train_df[train_df["cls_label"] == 0].sample(n=target_train_normal, random_state=RANDOM_SEED)
        train_other_df = train_df[train_df["cls_label"] != 0]
        train_df = pd.concat([train_normal_df, train_other_df], ignore_index=True)
    
    # 特征提取
    exclude_cols = ["dataset","video_name","frame_num","anomaly_type","frame_path","crowd_density_y","avg_speed_y","centroid_y_ratio_y","cls_label","risk_label"]
    feat_cols = df.columns.difference(exclude_cols, sort=False)
    
    X_train = train_df[feat_cols].values.astype(np.float32)
    y_train = train_df["cls_label"].values.astype(np.int64)
    r_train = train_df["risk_label"].values.astype(np.int64)
    
    X_val = val_df[feat_cols].values.astype(np.float32)
    y_val = val_df["cls_label"].values.astype(np.int64)
    r_val = val_df["risk_label"].values.astype(np.int64)
    
    X_test = test_df[feat_cols].values.astype(np.float32)
    y_test = test_df["cls_label"].values.astype(np.int64)
    r_test = test_df["risk_label"].values.astype(np.int64)

    # 在标准化之前，保存一小部分原始特征（取前1000个样本，避免文件过大）
    sample_indices = np.random.choice(len(X_train), min(1000, len(X_train)), replace=False)
    X_train_sample_before = X_train[sample_indices].copy()
    np.save(BASE_SAVE / "X_train_sample_before_scaling.npy", X_train_sample_before)

    # 标准化
    scaler = StandardScaler()
    X_train_scaled  = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    import joblib
    joblib.dump(scaler, SCALER_SAVE_PATH)

    # 保存标准化前后的样本
    X_train_sample_after = X_train_scaled[sample_indices]
    np.save(BASE_SAVE / "X_train_sample_after_scaling.npy", X_train_sample_after)
    X_train = X_train_scaled

    # 记录增强前的类别分布
    pre_aug_dist = [np.sum(y_train==i) for i in range(NUM_CLASSES)]
    print(f"增强前训练集分布: {pre_aug_dist}")

    # ========== 根据 enhance_level 进行数据增强 ==========
    if enhance_level >= 1:
        X_train, y_train, r_train = augment_data_strong(X_train, y_train, r_train)
        print(f"✅ 增强等级 {enhance_level}: 已启用强数据增强")
    else:
        print(f"📌 增强等级 {enhance_level}: 无数据增强")
    
    # 记录增强后的类别分布
    post_aug_dist = [np.sum(y_train==i) for i in range(NUM_CLASSES)]
    print(f"增强后训练集分布: {post_aug_dist}")

    # 将分布数据保存为 npy 文件，供后续绘图
    np.save(BASE_SAVE / f"pre_aug_dist_enh{enhance_level}.npy", pre_aug_dist)
    np.save(BASE_SAVE / f"post_aug_dist_enh{enhance_level}.npy", post_aug_dist)

    # 打印最终分布
    print(f"\n📊 三分类最终划分结果 (enhance_level={enhance_level})：")
    print(f"训练集分布: {[np.sum(y_train==i) for i in range(NUM_CLASSES)]}")
    print(f"验证集分布: {[np.sum(y_val==i) for i in range(NUM_CLASSES)]}")
    print(f"测试集分布: {[np.sum(y_test==i) for i in range(NUM_CLASSES)]}")
    
    # 加载器
    train_loader = DataLoader(AnomalyDataset(X_train, y_train, r_train), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(AnomalyDataset(X_val, y_val, r_val), batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(AnomalyDataset(X_test, y_test, r_test), batch_size=batch_size, shuffle=False, num_workers=0)
    
    # ========== 根据 enhance_level 计算类别权重 ==========
    if enhance_level == 0:
        class_weights = None
        print(f"📌 增强等级 0: 不使用类别权重")
    else:
        train_class_counts_final = np.bincount(y_train)
        class_weights = 1.0 / train_class_counts_final
        if enhance_level == 2:
            # 手动放大小类权重
            class_weights[1] *= 1.2   # 人体异常
            class_weights[2] *= 2.0   # 车辆闯入
            print(f"✅ 增强等级 2: 类别权重已放大 (人体×1.2, 车辆×2.0)")
        else:
            print(f"📌 增强等级 1: 类别权重仅逆频率，无手动放大")
        class_weights = class_weights / class_weights.sum()
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
        print(f"   类别权重: {class_weights.cpu().numpy()}")
    
    return train_loader, val_loader, test_loader, class_weights

# ===================== 数据集类 =====================
class AnomalyDataset(Dataset):
    def __init__(self, X, y_cls, y_risk):
        self.X = torch.from_numpy(X).float()
        self.y_cls = torch.from_numpy(y_cls).long()
        self.y_risk = torch.from_numpy(y_risk).long()
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y_cls[idx], self.y_risk[idx]

if __name__ == "__main__":
    # 测试各等级
    for lvl in [0, 1, 2]:
        print(f"\n{'='*50}\n测试 enhance_level={lvl}\n{'='*50}")
        build_dataloader(enhance_level=lvl)