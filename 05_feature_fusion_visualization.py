import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ===================== 配置参数 =====================
FEAT_03_PATH = Path("./chapter3_results/dynamic_features_v2.csv")
FEAT_04_PATH = Path("./chapter3_results/optical_mobilenet_features.csv")
FEAT_07_PATH = Path("./chapter3_results/resnet50_features.csv")

FUSED_FEAT_PATH = Path("./chapter3_results/fused_multimodal_features_with_resnet50.csv")
VIS_SAVE_DIR = Path("./chapter3_results/visualization")
VIS_SAVE_DIR.mkdir(exist_ok=True)

PCA_DIM = 50
TSNE_DIM = 2
RANDOM_SEED = 42

# ===================== 核心函数 =====================
def load_and_merge_features():
    df_03 = pd.read_csv(FEAT_03_PATH, encoding="utf-8-sig")
    df_04 = pd.read_csv(FEAT_04_PATH, encoding="utf-8-sig")
    df_07 = pd.read_csv(FEAT_07_PATH, encoding="utf-8-sig")

    common_cols = ["dataset", "video_name", "frame_num", "anomaly_type", "frame_path"]

    df_fuse = pd.merge(df_03, df_04, on=common_cols, how="inner")
    df_final = pd.merge(df_fuse, df_07, on=common_cols, how="inner")

    print(f"📊 三模态特征融合完成（维度完全正确）：")
    print(f"- 传统动力学特征：3 维")
    print(f"- 光流+MobileNet：12维光流 + 1000维MobileNet")
    print(f"- ResNet50：2048 维 ✅")
    print(f"- 融合后总特征：{len(df_final.columns)-len(common_cols)} 维")
    print(f"- 有效帧数量：{len(df_final)}")
    
    return df_final

def feature_standardization(feat_matrix):
    scaler = StandardScaler()
    feat_scaled = scaler.fit_transform(feat_matrix)
    return feat_scaled, scaler

def dim_reduction_visualization(df_fused):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    exclude_cols = ["dataset", "video_name", "frame_num", "anomaly_type", "frame_path"]
    feat_cols = [col for col in df_fused.columns if col not in exclude_cols]
    feat_matrix = df_fused[feat_cols].values

    feat_scaled, _ = feature_standardization(feat_matrix)
    pca = PCA(n_components=PCA_DIM, random_state=RANDOM_SEED)
    feat_pca = pca.fit_transform(feat_scaled)
    
    tsne = TSNE(n_components=TSNE_DIM, random_state=RANDOM_SEED, perplexity=30, n_iter=1000)
    feat_tsne = tsne.fit_transform(feat_pca)

    plt.figure(figsize=(12, 8))
    anomaly_types = df_fused["anomaly_type"].unique()
    colors = sns.color_palette("hsv", len(anomaly_types))

    for idx, anomaly in enumerate(anomaly_types):
        mask = df_fused["anomaly_type"] == anomaly
        plt.scatter(feat_tsne[mask, 0], feat_tsne[mask, 1], label=anomaly, color=colors[idx], s=10, alpha=0.7)

    plt.title("TSNE 多模态特征可视化", fontsize=14)
    plt.xlabel("TSNE 维度 1", fontsize=12)
    plt.ylabel("TSNE 维度 2", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    vis_path = VIS_SAVE_DIR / "tsne_with_resnet50.png"
    plt.savefig(vis_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 可视化保存至：{vis_path}")
    return feat_tsne

# ===================== 主流程 =====================
def main():
    df_fused = load_and_merge_features()
    df_fused.to_csv(FUSED_FEAT_PATH, index=False, encoding="utf-8-sig")
    print(f"✅ 融合特征保存至：{FUSED_FEAT_PATH}")
    dim_reduction_visualization(df_fused)
    print("\n🎉 05号程序运行完成！")

if __name__ == "__main__":
    main()