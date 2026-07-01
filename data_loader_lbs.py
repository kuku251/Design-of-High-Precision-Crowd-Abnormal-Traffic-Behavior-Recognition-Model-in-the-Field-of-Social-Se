# -*- coding: utf-8 -*-
# data_loader_lbs.py - 支持加载含模拟LBS特征的扩展数据，复用相同视频划分
import torch
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

USE_SELF_DATASET = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEAT_PATH = "./chapter3_results/fused_multimodal_features_with_lbs.csv"  # 3069维
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
BATCH_SIZE = 32
RANDOM_SEED = 42
NUM_CLASSES = 3
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)

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
SCALER_SAVE_PATH = "./chapter4_results/models/standard_scaler_lbs.pth"
Path(SCALER_SAVE_PATH).parent.mkdir(exist_ok=True, parents=True)
BASE_SAVE = Path(SCALER_SAVE_PATH).parent

def augment_data_strong(X, y_cls, y_risk):
    car_mask = (y_cls == 2)
    if car_mask.any():
        X_car, y_car, r_car = X[car_mask], y_cls[car_mask], y_risk[car_mask]
        for _ in range(3):
            noise = np.random.normal(0, 0.01, X_car.shape)
            X = np.concatenate([X, X_car + noise], axis=0)
            y_cls = np.concatenate([y_cls, y_car], axis=0)
            y_risk = np.concatenate([y_risk, r_car], axis=0)
    human_mask = (y_cls == 1)
    if human_mask.any():
        X_human, y_human, r_human = X[human_mask], y_cls[human_mask], y_risk[human_mask]
        for _ in range(2):
            noise = np.random.normal(0, 0.008, X_human.shape)
            X = np.concatenate([X, X_human + noise], axis=0)
            y_cls = np.concatenate([y_cls, y_human], axis=0)
            y_risk = np.concatenate([y_risk, r_human], axis=0)
    return X, y_cls, y_risk

def build_dataloader_lbs(batch_size=BATCH_SIZE, enhance_level=2):
    df = pd.read_csv(FEAT_PATH, encoding="utf-8-sig").dropna().reset_index(drop=True)
    if not USE_SELF_DATASET:
        df = df[df["dataset"] != "dataset_self"].reset_index(drop=True)
    df["cls_label"] = df["anomaly_type"].apply(map_label)
    df["risk_label"] = df["cls_label"]
    
    # 加载已保存的视频划分（保证与原始 data_loader 完全一致）
    split_path = BASE_SAVE / "video_split.json"
    if not split_path.exists():
        raise FileNotFoundError(f"视频划分文件不存在: {split_path}，请先运行 data_loader.py 生成")
    with open(split_path, "r", encoding="utf-8") as f:
        split_info = json.load(f)
    train_videos = set(split_info["train_videos"])
    val_videos = set(split_info["val_videos"])
    test_videos = set(split_info["test_videos"])
    
    train_df = df[df["video_name"].isin(train_videos)].reset_index(drop=True)
    val_df = df[df["video_name"].isin(val_videos)].reset_index(drop=True)
    test_df = df[df["video_name"].isin(test_videos)].reset_index(drop=True)
    
    # 训练集正常类下采样（与原始逻辑完全一致）
    train_class_counts = train_df["cls_label"].value_counts().sort_index()
    train_normal_count = train_class_counts[0] if 0 in train_class_counts.index else 0
    train_human = train_class_counts[1] if 1 in train_class_counts.index else 0
    train_car = train_class_counts[2] if 2 in train_class_counts.index else 0
    target_train_normal = int(train_human + train_car * 3)
    if train_normal_count > target_train_normal and target_train_normal > 0:
        train_normal_df = train_df[train_df["cls_label"] == 0].sample(n=target_train_normal, random_state=RANDOM_SEED)
        train_other_df = train_df[train_df["cls_label"] != 0]
        train_df = pd.concat([train_normal_df, train_other_df], ignore_index=True)
    
    exclude_cols = ["dataset","video_name","frame_num","anomaly_type","frame_path",
                    "crowd_density_y","avg_speed_y","centroid_y_ratio_y","cls_label","risk_label"]
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
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    import joblib
    joblib.dump(scaler, SCALER_SAVE_PATH)
    X_train = X_train_scaled
    
    if enhance_level >= 1:
        X_train, y_train, r_train = augment_data_strong(X_train, y_train, r_train)
    
    train_loader = DataLoader(AnomalyDataset(X_train, y_train, r_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(AnomalyDataset(X_val, y_val, r_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(AnomalyDataset(X_test, y_test, r_test), batch_size=batch_size, shuffle=False)
    
    if enhance_level == 0:
        class_weights = None
    else:
        train_class_counts_final = np.bincount(y_train)
        class_weights = 1.0 / train_class_counts_final
        if enhance_level == 2:
            class_weights[1] *= 1.2
            class_weights[2] *= 2.0
        class_weights = class_weights / class_weights.sum()
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    return train_loader, val_loader, test_loader, class_weights

class AnomalyDataset(Dataset):
    def __init__(self, X, y_cls, y_risk):
        self.X = torch.from_numpy(X).float()
        self.y_cls = torch.from_numpy(y_cls).long()
        self.y_risk = torch.from_numpy(y_risk).long()
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y_cls[idx], self.y_risk[idx]