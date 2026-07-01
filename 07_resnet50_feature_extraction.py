# -*- coding: utf-8 -*-
# 07_resnet50_feature_extraction.py - ResNet50视觉特征提取（适配所有数据集+兼容现有特征流程）
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ===================== 1. 全局配置（和你的现有代码完全对齐） =====================
# 输入路径：基于03号程序的动力学特征CSV（获取所有帧路径）
INPUT_CSV_PATH = Path("./chapter3_results/dynamic_features_v2.csv")
# 输出路径：ResNet50特征保存路径
RESNET50_FEAT_SAVE_PATH = Path("./chapter3_results/resnet50_features.csv")
RESNET50_FEAT_SAVE_PATH.parent.mkdir(exist_ok=True)

# 设备配置：优先GPU，无则CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ResNet50输入尺寸（固定224x224）
IMG_SIZE = 224
# ResNet50特征维度（去掉fc层后为2048维）
RESNET50_FEAT_DIM = 2048

# ===================== 2. 初始化ResNet50模型（特征提取模式） =====================
def init_resnet50_model():
    """初始化预训练ResNet50，移除分类头，仅保留特征提取部分"""
    # 加载预训练模型（适配torchvision新版本）
    resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    # 移除最后一层全连接层（保留avgpool前的特征）
    feature_extractor = torch.nn.Sequential(*list(resnet50.children())[:-1])
    # 设为评估模式（关闭Dropout/BatchNorm训练模式）
    feature_extractor = feature_extractor.to(DEVICE).eval()
    return feature_extractor

# ===================== 3. 图像预处理（严格匹配ImageNet预训练参数） =====================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),  # 缩放至224x224
    transforms.ToTensor(),  # 转为张量 [H,W,C]→[C,H,W]，值范围0-1
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet均值
        std=[0.229, 0.224, 0.225]    # ImageNet方差
    )
])

# ===================== 4. 安全读取帧（复用你现有代码的兼容逻辑） =====================
def read_frame_safely(frame_path: Path) -> Image.Image:
    """
    安全读取帧（兼容.tif/.jpg、中文路径、灰度/彩色帧）
    :return: PIL Image对象（RGB模式），读取失败返回None
    """
    try:
        # 读取帧（兼容UCSD的.tif和其他数据集的.jpg）
        img = Image.open(str(frame_path))
        # 灰度图转RGB（保证输入通道数一致）
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except FileNotFoundError:
        print(f"❌ 帧文件不存在：{frame_path}")
        return None
    except Exception as e:
        print(f"❌ 读取帧失败：{frame_path}，错误：{str(e)}")
        return None

# ===================== 5. 提取单帧ResNet50特征 =====================
def extract_single_frame_feature(frame_path: Path, model) -> np.ndarray:
    """
    提取单帧的ResNet50特征
    :param frame_path: 帧路径
    :param model: 初始化后的ResNet50特征提取器
    :return: 2048维特征向量（numpy数组），失败返回None
    """
    # 安全读取帧
    img = read_frame_safely(frame_path)
    if img is None:
        return None
    
    # 预处理+增加batch维度
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)  # [1,3,224,224]
    
    # 前向传播（关闭梯度计算，提速+减少显存占用）
    with torch.no_grad():
        feature = model(img_tensor)  # [1,2048,1,1]
    
    # 展平为2048维向量并转为numpy
    feature_np = feature.squeeze().cpu().numpy()  # (2048,)
    return feature_np

# ===================== 6. 批量提取所有帧的ResNet50特征 =====================
def batch_extract_resnet50_features():
    """批量处理所有帧，提取ResNet50特征并保存"""
    # 1. 加载输入CSV（获取帧路径和元信息）
    df = pd.read_csv(INPUT_CSV_PATH, encoding="utf-8-sig")
    # 去重：避免重复提取同一帧
    df_unique = df.drop_duplicates(subset=["frame_path"]).reset_index(drop=True)
    print(f"📊 待处理帧总数（去重后）：{len(df_unique)}")
    print(f"🔧 使用设备：{DEVICE}")
    
    # 2. 初始化ResNet50模型
    model = init_resnet50_model()
    print("✅ ResNet50模型初始化完成（预训练权重加载成功）")
    
    # 3. 初始化结果列表
    result_list = []
    fail_count = 0
    
    # 4. 批量提取特征（带进度条）
    for idx, row in tqdm(df_unique.iterrows(), total=len(df_unique), desc="ResNet50特征提取进度"):
        frame_path = Path(row["frame_path"])
        # 提取特征
        feature = extract_single_frame_feature(frame_path, model)
        if feature is None:
            fail_count += 1
            continue
        
        # 构造特征列（resnet50_0 ~ resnet50_2047）
        feature_dict = {f"resnet50_{i}": round(val, 6) for i, val in enumerate(feature)}
        # 合并元信息（dataset/video_name等）+ 特征
        row_dict = row.to_dict()
        row_dict.update(feature_dict)
        result_list.append(row_dict)
    
    # 5. 保存结果CSV
    if result_list:
        df_result = pd.DataFrame(result_list)
        df_result.to_csv(RESNET50_FEAT_SAVE_PATH, index=False, encoding="utf-8-sig")
        print(f"\n✅ ResNet50特征提取完成！")
        print(f"   📁 保存路径：{RESNET50_FEAT_SAVE_PATH}")
        print(f"   ✅ 成功提取帧数：{len(result_list)}")
        print(f"   ❌ 提取失败帧数：{fail_count}")
        print(f"   📏 特征维度：{RESNET50_FEAT_DIM}（resnet50_0 ~ resnet50_2047）")
    else:
        print("❌ 未提取到任何有效特征！请检查帧路径和模型初始化")

# ===================== 7. 主函数 =====================
if __name__ == "__main__":
    # 检查输入文件是否存在
    if not INPUT_CSV_PATH.exists():
        print(f"❌ 输入CSV文件不存在：{INPUT_CSV_PATH}")
        print("   请先运行03_dynamic_features.py生成动力学特征CSV")
    else:
        # 执行批量提取
        batch_extract_resnet50_features()
        
        # 可选：验证特征文件
        if RESNET50_FEAT_SAVE_PATH.exists():
            df_verify = pd.read_csv(RESNET50_FEAT_SAVE_PATH, encoding="utf-8-sig")
            feat_cols = [col for col in df_verify.columns if col.startswith("resnet50_")]
            print(f"\n🔍 特征文件验证：")
            print(f"   特征列数量：{len(feat_cols)}（预期2048）")
            print(f"   总数据行数：{len(df_verify)}")
            print(f"   示例特征值（第一帧前5维）：{df_verify[feat_cols[:5]].iloc[0].tolist()}")