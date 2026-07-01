# -*- coding: utf-8 -*-
# 03_dynamic_features.py - 动力学特征提取（保留帧号匹配+解决读取失败）
import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# ===================== 1. 核心配置（路径+标注） =====================
# 数据集根路径（和01/02号程序一致）
DATASET_ROOT = {
    "ShanghaiTech": r"D:\毕业设计\dataset\ShanghaiTech_Video_Frame",
    "UCSD": r"D:\毕业设计\dataset\UCSD_Anomaly_Dataset",
    "UMN": r"D:\毕业设计\dataset\UMN",
    "dataset_self": r"D:\毕业设计\dataset\dataset_self"
}
# 标注CSV路径（01号程序生成的标注文件）
FINAL_ANNOTATION_CSV = r"D:\毕业设计\dataset\all_dataset_annotations.csv"
# 特征保存路径
# DYNAMIC_FEATURE_SAVE_PATH = Path("./chapter3_results/dynamic_features.csv")
DYNAMIC_FEATURE_SAVE_PATH = Path("./chapter3_results/dynamic_features_v2.csv")
DYNAMIC_FEATURE_SAVE_PATH.parent.mkdir(exist_ok=True)

# 帧路径规则（精准匹配各数据集的帧命名+路径）
DATASET_FRAME_RULE = {
    "ShanghaiTech": {
        "frame_dir": lambda v: Path(DATASET_ROOT["ShanghaiTech"]) / "Test" / f"{v.split('.')[0]}_frames",
        "suffix": [".jpg"],
        "frame_name": lambda fn: f"frame_{fn:04d}"  # frame_0001.jpg
    },
    "UCSD": {
        "frame_dir": lambda v: Path(DATASET_ROOT["UCSD"]) 
                              / v.split("_")[0]  # UCSDped1/UCSDped2
                              / ("Test" if "Test" in v else "Train") 
                              / v.split("_")[1].replace(".avi", ""),  # Train004.avi → Train004
        "suffix": [".tif"],
        "frame_name": lambda fn: f"{fn:03d}"  # 001.tif（UCSD原始命名）
    },
    "UMN": {
        "frame_dir": lambda v: Path(DATASET_ROOT["UMN"]) / "_".join(v.split("_")[:2]) / "frames",  # Scene_1_1.mp4 → Scene_1/frames
        "suffix": [".jpg"],
        "frame_name": lambda fn: f"frame_{fn:04d}"  # frame_0001.jpg
    },
    "dataset_self": {
        "frame_dir": lambda v: Path(DATASET_ROOT["dataset_self"]) / f"{v.split('.')[0]}_frames",
        "suffix": [".jpg"],
        "frame_name": lambda fn: f"frame_{fn:04d}"  # 02号拆帧命名：frame_0001.jpg
    }
}

# ===================== 2. 根治读取失败：增强版帧读取函数 =====================
def read_frame_safely(frame_path: Path) -> np.ndarray:
    """
    安全读取帧（统一用 PIL，兼容所有格式，对 UCSD .tif 更友好）
    :return: OpenCV格式的BGR帧（None=读取失败）
    """
    try:
        # 1. 用 PIL 读取（解决中文路径+UCSD .tif 兼容性问题）
        img = Image.open(str(frame_path))
        frame = np.array(img)
        
        # 2. 格式统一：适配所有情况
        if len(frame.shape) == 2:  # 灰度图（UCSD）→ 转3通道BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif len(frame.shape) == 3 and frame.shape[2] == 3:  # RGB（PIL）→ BGR（OpenCV）
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif len(frame.shape) == 3 and frame.shape[2] == 4:  # RGBA → BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        
        # 3. 校验帧有效性
        if frame.size == 0:
            print(f"❌ 帧为空：{frame_path}")
            return None
        return frame
    except FileNotFoundError:
        print(f"❌ 帧文件不存在：{frame_path}")
        return None
    except Exception as e:
        print(f"❌ 读取帧失败：{frame_path}，错误：{str(e)}")
        return None

# ===================== 3. 动力学特征提取（保留核心逻辑） =====================
def extract_dynamic_features(frame: np.ndarray) -> dict:
    """提取单帧动力学特征（输入为OpenCV BGR帧）"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = frame.shape[:2]

    # 1. 人群密度（轮廓检测，过滤小轮廓）
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]
    crowd_density = len(valid_contours)

    # 2. 平均运动速度（Sobel梯度）
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    avg_speed = np.mean(np.sqrt(grad_x**2 + grad_y**2))

    # 3. 重心y轴占比
    centroid_y_ratio = 0.0
    if len(valid_contours) > 0:
        centroids = []
        for cnt in valid_contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cy = M["m01"] / M["m00"]
                centroids.append(cy)
        if centroids:
            avg_centroid_y = np.mean(centroids)
            centroid_y_ratio = avg_centroid_y / h

    return {
        "crowd_density": crowd_density,
        "avg_speed": round(avg_speed, 4),
        "centroid_y_ratio": round(centroid_y_ratio, 4)
    }

# ===================== 4. 精准帧路径匹配（保留帧号逻辑） =====================
def get_frame_path(dataset: str, video_name: str, frame_num: int) -> Path:
    """根据数据集+视频名+帧号，精准匹配帧路径"""
    rule = DATASET_FRAME_RULE[dataset]
    # 1. 获取帧文件夹
    frame_dir = rule["frame_dir"](video_name)
    if not frame_dir.exists():
        print(f"⚠️ 帧文件夹不存在：{frame_dir}")
        return None
    
    # 2. 生成帧文件名（按数据集规则）
    frame_base = rule["frame_name"](frame_num)
    
    # 3. 匹配后缀
    for suffix in rule["suffix"]:
        frame_path = frame_dir / f"{frame_base}{suffix}"
        if frame_path.exists():
            return frame_path
    
    # 所有后缀都匹配失败
    print(f"⚠️ 未找到帧：{frame_dir}/{frame_base}[{','.join(rule['suffix'])}]")
    return None

# ===================== 5. 核心：按标注帧号提取特征 =====================
def main():
    # 1. 读取标注CSV（保证异常类型精准）
    if not Path(FINAL_ANNOTATION_CSV).exists():
        print(f"❌ 标注文件不存在：{FINAL_ANNOTATION_CSV}")
        return
    annot_df = pd.read_csv(FINAL_ANNOTATION_CSV, encoding="utf-8-sig")
    print(f"✅ 读取标注完成，共 {len(annot_df)} 条标注")

    # 2. 按视频分组提取（保留帧号匹配）
    feature_list = []
    video_groups = annot_df.groupby(["dataset", "video_name"])
    
    for (dataset, video_name), group in tqdm(video_groups, desc="提取动力学特征"):
        # 获取该视频的标注信息（异常类型+帧号范围）
        anomaly_type = group["anomaly_type"].iloc[0]
        start_frame = int(group["start_frame"].iloc[0])
        end_frame = int(group["end_frame"].iloc[0])
        
        # 校验帧号范围
        if start_frame > end_frame:
            print(f"⚠️ 帧号范围错误：{video_name}（start={start_frame}, end={end_frame}）")
            continue
        
        # 遍历标注的异常帧号
        for frame_num in range(start_frame, end_frame + 1):
            # 精准匹配帧路径
            frame_path = get_frame_path(dataset, video_name, frame_num)
            if frame_path is None:
                continue
            
            # 安全读取帧（解决读取失败）
            frame = read_frame_safely(frame_path)
            if frame is None:
                continue
            
            # 提取特征
            dynamic_feat = extract_dynamic_features(frame)
            
            # 拼接最终特征（保留标注的异常类型）
            feature_list.append({
                "dataset": dataset,
                "video_name": video_name,
                "frame_num": frame_num,
                "anomaly_type": anomaly_type,  # 按标注赋值，而非文件夹
                "frame_path": str(frame_path),
                **dynamic_feat
            })

    # 3. 保存结果
    if len(feature_list) == 0:
        print("\n❌ 未提取到任何特征！请检查：")
        print("  1. 标注CSV的帧号范围是否正确")
        print("  2. 帧文件夹路径是否匹配")
        print("  3. 帧文件是否存在且未损坏")
    else:
        feat_df = pd.DataFrame(feature_list)
        feat_df.to_csv(DYNAMIC_FEATURE_SAVE_PATH, index=False, encoding="utf-8-sig")
        print(f"\n🎉 特征提取完成！共 {len(feat_df)} 条特征（按标注帧号提取）")
        print(f"📁 特征文件保存至：{DYNAMIC_FEATURE_SAVE_PATH}")
        
        # 按异常类型统计（验证结果准确性）
        print("\n📊 特征统计（按标注的异常类型分组）：")
        stat_df = feat_df.groupby("anomaly_type")[["crowd_density", "avg_speed", "centroid_y_ratio"]].mean()
        print(stat_df.round(4))

if __name__ == "__main__":
    main()