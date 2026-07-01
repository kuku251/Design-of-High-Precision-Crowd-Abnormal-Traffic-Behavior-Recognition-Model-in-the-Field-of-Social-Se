import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm  # 进度条
import warnings
warnings.filterwarnings('ignore')

# ===================== 配置参数（无需修改，适配你的路径）=====================
ANNOTATION_CSV = Path("./chapter3_results/dynamic_features_v2.csv")  # 03号结果CSV
SAVE_PATH = Path("./chapter3_results/optical_mobilenet_features.csv")  # 04号结果保存路径
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 优先用GPU，无则CPU
FRAME_SIZE = (224, 224)  # MobileNet输入尺寸
MOBILENET_FEATURE_DIM = 1000  # MobileNet输出特征维度

# ===================== 初始化模型和工具 =====================
# 1. MobileNet预训练模型（提取深度学习特征）
mobilenet = models.mobilenet_v2(pretrained=True).to(DEVICE)
mobilenet.eval()  # 评估模式，不训练
# 图像预处理（适配MobileNet输入要求）
transform = transforms.Compose([
    transforms.Resize(FRAME_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. 光流提取（Farneback算法，捕捉运动特征）
def extract_optical_flow(prev_frame, curr_frame):
    """提取两帧间的光流特征（强制统一尺寸+通道，兼容所有帧格式）"""
    # 1. 统一目标尺寸（和MobileNet输入保持一致，避免冗余resize）
    target_size = (224, 224)
    
    # 2. 转灰度图（不管输入是彩色/灰度，都转成单通道）
    if len(prev_frame.shape) == 3:  # 彩色帧（BGR）
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    else:  # 已经是灰度帧
        prev_gray = prev_frame
        
    if len(curr_frame.shape) == 3:  # 彩色帧（BGR）
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    else:  # 已经是灰度帧
        curr_gray = curr_frame
    
    # 3. 强制resize到相同尺寸（解决分辨率不一致问题）
    prev_gray = cv2.resize(prev_gray, target_size)
    curr_gray = cv2.resize(curr_gray, target_size)
    
    # 4. 计算光流（现在输入完全符合Farneback要求）
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    
    # 5. 提取统计特征（和之前逻辑一致，不变）
    flow_x = flow[..., 0]
    flow_y = flow[..., 1]
    magnitude = np.sqrt(flow_x**2 + flow_y**2)
    angle = np.arctan2(flow_y, flow_x) * 180 / np.pi  # 角度转度
    
    optical_features = np.concatenate([
        [np.mean(flow_x), np.std(flow_x), np.max(flow_x), np.min(flow_x)],
        [np.mean(flow_y), np.std(flow_y), np.max(flow_y), np.min(flow_y)],
        [np.mean(magnitude), np.std(magnitude), np.mean(angle), np.std(angle)]
    ])
    return optical_features

# 3. 安全读取帧（兼容.tif/.jpg，解决UCSD读取问题）
def read_frame_safely(frame_path: Path) -> np.ndarray:
    try:
        # 用PIL读取（兼容.tif和中文路径）
        img = Image.open(str(frame_path))
        frame = np.array(img)
        # 格式统一：灰度图转3通道，RGB转BGR
        if len(frame.shape) == 2:  # 灰度图（UCSD）
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif len(frame.shape) == 3 and frame.shape[2] == 3:  # RGB（PIL）→ BGR（OpenCV）
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif len(frame.shape) == 3 and frame.shape[2] == 4:  # RGBA → BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        return frame
    except Exception as e:
        # 读取失败不报错，仅跳过（不影响整体流程）
        return None

# 4. 提取单帧MobileNet特征
def extract_mobilenet_feature(frame: np.ndarray) -> np.ndarray:
    """将BGR帧转为MobileNet特征（1000维）"""
    # 转PIL图像（适配transform）
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)  # 增加batch维度
    # 提取特征（禁用梯度计算，提速）
    with torch.no_grad():
        feature = mobilenet(img_tensor).cpu().numpy().flatten()
    return feature

# ===================== 核心流程：批量提取特征 =====================
def main():
    # 1. 读取标注CSV（只取唯一帧，避免重复提取）
    df = pd.read_csv(ANNOTATION_CSV)
    df_unique = df.drop_duplicates(subset=["frame_path"]).reset_index(drop=True)  # 去重
    print(f"📊 共需处理 {len(df_unique)} 帧，开始提取光流+MobileNet特征...")

    # 2. 初始化结果列表
    feature_list = []
    prev_frame = None  # 光流需要前一帧，初始化None

    # 3. 遍历所有帧（带进度条）
    for idx, row in tqdm(df_unique.iterrows(), total=len(df_unique)):
        frame_path = Path(row["frame_path"])
        anomaly_type = row["anomaly_type"]
        dataset = row["dataset"]
        video_name = row["video_name"]
        frame_num = row["frame_num"]

        # 安全读取当前帧
        curr_frame = read_frame_safely(frame_path)
        if curr_frame is None:
            continue  # 跳过读取失败的帧

        # 提取MobileNet特征（单帧独立，无需前帧）
        mobilenet_feat = extract_mobilenet_feature(curr_frame)

        # 提取光流特征（第一帧无前置帧，用当前帧自计算）
        if prev_frame is None:
            optical_feat = extract_optical_flow(curr_frame, curr_frame)
        else:
            optical_feat = extract_optical_flow(prev_frame, curr_frame)
        prev_frame = curr_frame  # 更新前帧

        # 合并特征（光流12维 + MobileNet1000维 = 1012维）
        combined_feat = np.concatenate([optical_feat, mobilenet_feat])

        # 构造结果字典（和03号CSV格式对齐，方便后续融合）
        result = {
            "dataset": dataset,
            "video_name": video_name,
            "frame_num": frame_num,
            "anomaly_type": anomaly_type,
            "frame_path": str(frame_path),
            # 光流特征（12维）
            "optical_flow_x_mean": combined_feat[0],
            "optical_flow_x_std": combined_feat[1],
            "optical_flow_x_max": combined_feat[2],
            "optical_flow_x_min": combined_feat[3],
            "optical_flow_y_mean": combined_feat[4],
            "optical_flow_y_std": combined_feat[5],
            "optical_flow_y_max": combined_feat[6],
            "optical_flow_y_min": combined_feat[7],
            "flow_magnitude_mean": combined_feat[8],
            "flow_magnitude_std": combined_feat[9],
            "flow_angle_mean": combined_feat[10],
            "flow_angle_std": combined_feat[11],
            # MobileNet特征（1000维，命名为mobilenet_0 ~ mobilenet_999）
            **{f"mobilenet_{i}": combined_feat[12 + i] for i in range(MOBILENET_FEATURE_DIM)}
        }
        feature_list.append(result)

    # 4. 保存结果CSV
    feat_df = pd.DataFrame(feature_list)
    feat_df.to_csv(SAVE_PATH, index=False, encoding="utf-8-sig")  # 兼容中文
    print(f"✅ 特征提取完成！共提取 {len(feat_df)} 帧特征，保存至：{SAVE_PATH}")
    print(f"📌 特征维度：光流12维 + MobileNet1000维 = 1012维")

if __name__ == "__main__":
    main()