# -*- coding: utf-8 -*-
# 08_extract_lbs_features.py - 从质心轨迹提取模拟LBS特征（6维）
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# 配置路径
FEAT_CSV_PATH = Path("./chapter3_results/fused_multimodal_features_with_resnet50.csv")
LBS_SAVE_PATH = Path("./chapter3_results/lbs_simulated_features.csv")
FUSED_WITH_LBS_PATH = Path("./chapter3_results/fused_multimodal_features_with_lbs.csv")

def read_frame_safely(frame_path: Path) -> np.ndarray:
    """安全读取帧（兼容中文路径和.tif格式），返回BGR图像"""
    try:
        img = Image.open(str(frame_path))
        frame = np.array(img)
        if len(frame.shape) == 2:  # 灰度图
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif len(frame.shape) == 3 and frame.shape[2] == 3:  # RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif len(frame.shape) == 3 and frame.shape[2] == 4:  # RGBA
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        return frame
    except Exception:
        return None

def extract_lbs_features_by_optical_flow(df):
    """
    基于光流主运动模拟轨迹，生成6维仿真LBS特征
    """
    lbs_feat_list = []
    grouped = df.groupby("video_name")
    for video, group in tqdm(grouped, desc="光流模拟LBS轨迹"):
        group = group.sort_values("frame_num").reset_index(drop=True)
        prev_gray = None
        video_lbs = []
        # 用于计算累计位移的变量
        cum_disp = 0.0
        max_cum_disp = 1.0  # 先初始化为1，最后统一归一化
        
        for idx, row in group.iterrows():
            frame_path = Path(row["frame_path"])
            curr_frame = read_frame_safely(frame_path)
            if curr_frame is None:
                video_lbs.append([0.0] * 6)
                continue
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            
            if prev_gray is not None:
                # 计算稠密光流
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                dx = np.mean(flow[..., 0])
                dy = np.mean(flow[..., 1])
            else:
                dx, dy = 0.0, 0.0
            
            # 计算瞬时特征
            speed = np.sqrt(dx**2 + dy**2)
            direction = np.arctan2(dy, dx)
            
            if idx == 0:
                acc = 0.0
                dir_change = 0.0
                curvature = 0.0
                cum_disp_norm = 0.0
            else:
                prev_speed = video_lbs[-1][0]
                acc = speed - prev_speed
                prev_direction = video_lbs[-1][2]
                dir_change = direction - prev_direction
                dir_change = np.arctan2(np.sin(dir_change), np.cos(dir_change))
                curvature = np.abs(dir_change) / (speed + 1e-6)
                cum_disp += speed
                cum_disp_norm = cum_disp / (max_cum_disp + 1e-6)
            
            video_lbs.append([speed, acc, direction, dir_change, curvature, cum_disp_norm])
            prev_gray = curr_gray
        
        # 对整个视频的累计位移做归一化
        video_lbs = np.array(video_lbs)
        if len(video_lbs) > 1:
            max_cum = np.max(video_lbs[:, 5])
            if max_cum > 1e-6:
                video_lbs[:, 5] = video_lbs[:, 5] / max_cum
        lbs_feat_list.extend(video_lbs)
    
    return np.array(lbs_feat_list)
def main():
    print("📂 读取融合特征CSV...")
    df = pd.read_csv(FEAT_CSV_PATH, encoding="utf-8-sig")
    print(f"原始特征维度：{len(df.columns) - 5}（去除元数据列）")
    
    # 提取LBS特征
    lbs_features = extract_lbs_features_by_optical_flow(df)
    lbs_df = pd.DataFrame(lbs_features, columns=[
        "lbs_speed", "lbs_acceleration", "lbs_direction",
        "lbs_dir_change", "lbs_curvature", "lbs_cum_disp_norm"
    ])
    # 保存单独的LBS特征文件
    lbs_df.to_csv(LBS_SAVE_PATH, index=False, encoding="utf-8-sig")
    print(f"✅ 模拟LBS特征已保存至：{LBS_SAVE_PATH}")
    
    # 合并到原始特征
    df_fused = pd.concat([df, lbs_df], axis=1)
    df_fused.to_csv(FUSED_WITH_LBS_PATH, index=False, encoding="utf-8-sig")
    print(f"✅ 融合后特征（3063+6=3069维）已保存至：{FUSED_WITH_LBS_PATH}")

if __name__ == "__main__":
    main()