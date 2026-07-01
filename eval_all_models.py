# -*- coding: utf-8 -*-
# 13_eval_all_models.py - 支持多增强等级评估
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve, auc,
    precision_recall_fscore_support
)
import warnings
warnings.filterwarnings('ignore')

# ===================== 中文显示修复 =====================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

# ===================== 导入模块 =====================
from model_architecture import init_model
from data_loader import build_dataloader

# ===================== 全局配置 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MODEL_TYPE = "mobilenet" 
MODEL_TYPE = "resnet50" 

BATCH_SIZE = 32

# 要评估的增强等级列表（可手动修改）
ENHANCE_LEVELS = [0, 1, 2] 

CLASS_NAMES = ['正常行走', '人体异常', '车辆闯入']
NUM_CLASSES = 3
ALL_LABELS = np.arange(NUM_CLASSES)

# 自动创建保存目录
BASE_DIR = Path("./chapter4_results/")
(BASE_DIR / "models").mkdir(parents=True, exist_ok=True)
(BASE_DIR / "figures").mkdir(parents=True, exist_ok=True)
(BASE_DIR / "results").mkdir(parents=True, exist_ok=True)

BASE_SAVE = BASE_DIR / "models"
FIGURE_SAVE = BASE_DIR / "figures"
RESULT_SAVE = BASE_DIR / "results"

# ===================== 模型加载（根据路径） =====================
def load_model_from_path(model_type, weight_path):
    model = init_model(model_type).to(DEVICE)
    if not weight_path.exists():
        raise FileNotFoundError(f"权重文件不存在: {weight_path}")
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    model.eval()
    print(f"✅ 模型加载完成：{weight_path.name}")
    return model

# ===================== 测试集全指标评估 =====================
def evaluate_model(model, test_loader):
    y_true = []
    y_pred = []
    y_score = []
    
    with torch.no_grad():
        for batch_X, y_cls, _ in tqdm(test_loader, desc="测试集评估中"):
            batch_X = batch_X.to(DEVICE)
            outputs, _ = model(batch_X)
            
            pred = torch.argmax(outputs, dim=1)
            y_true.extend(y_cls.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            y_score.extend(torch.softmax(outputs, dim=1).cpu().numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_score = np.array(y_score)

    print("\n📊 测试集真实类别分布：")
    unique_true = np.unique(y_true)
    for label in unique_true:
        cnt = np.sum(y_true == label)
        print(f"  {CLASS_NAMES[label]}: {cnt} 样本")

    acc = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0, labels=ALL_LABELS)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0, labels=ALL_LABELS)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0, labels=ALL_LABELS)
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0, labels=ALL_LABELS)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0, labels=ALL_LABELS)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0, labels=ALL_LABELS)
    
    try:
        valid_labels = unique_true
        y_true_onehot = np.eye(NUM_CLASSES)[y_true]
        auc_list = []
        for i in valid_labels:
            if np.sum(y_true_onehot[:, i]) > 0:
                auc_i = roc_auc_score(y_true_onehot[:, i], y_score[:, i])
                auc_list.append(auc_i)
        roc_auc = np.mean(auc_list) if auc_list else 0.0
    except:
        roc_auc = 0.0

    cm = confusion_matrix(y_true, y_pred, labels=ALL_LABELS)
    
    # 推理速度
    test_sample = next(iter(test_loader))[0].to(DEVICE)
    for _ in range(10):
        _ = model(test_sample)
    start = time.time()
    for _ in range(100):
        _ = model(test_sample)
    avg_time = (time.time() - start) / 100
    fps = 1 / avg_time
    
    metrics = {
        "accuracy": acc, "precision_macro": precision_macro, "recall_macro": recall_macro,
        "f1_macro": f1_macro, "precision_weighted": precision_weighted, "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted, "roc_auc": roc_auc, "fps": fps, "confusion_matrix": cm,
        "y_true": y_true, "y_pred": y_pred, "y_score": y_score
    }
    return metrics

# ===================== 混淆矩阵绘图 =====================
def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('行为分类混淆矩阵', fontsize=16)
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 混淆矩阵已保存: {save_path.name}")

# ===================== ROC曲线绘图 =====================
def plot_roc_curve(y_true, y_score, num_classes, class_names, save_path):
    y_true_onehot = np.eye(num_classes)[y_true]
    valid_labels = np.unique(y_true)
    
    fpr, tpr, roc_auc = {}, {}, {}
    for i in valid_labels:
        fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for idx, i in enumerate(valid_labels):
        plt.plot(fpr[i], tpr[i], lw=2, color=colors[idx], label=f'{class_names[i]} (AUC={roc_auc[i]:.4f})')
    
    plt.plot([0,1], [0,1], 'k--', lw=2)
    plt.xlabel('假阳性率', fontsize=12)
    plt.ylabel('真阳性率', fontsize=12)
    plt.title('多分类ROC曲线', fontsize=16)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ ROC曲线已保存: {save_path.name}")
# ===================== 保存CSV结果（删除 plan_type 参数） =====================
def save_results_to_csv(metrics, model_type, enhance_level):
    global_data = {
        "模型类型": [model_type], 
        "增强等级": [enhance_level],
        "准确率": round(metrics["accuracy"], 4),
        "Macro精确率": round(metrics["precision_macro"], 4), 
        "Macro召回率": round(metrics["recall_macro"], 4),
        "MacroF1": round(metrics["f1_macro"], 4), 
        "WeightedF1": round(metrics["f1_weighted"], 4),
        "AUC": round(metrics["roc_auc"], 4), 
        "FPS": round(metrics["fps"], 2)
    }
    pd.DataFrame(global_data).to_csv(
        RESULT_SAVE / f"{model_type}_enh{enhance_level}_全局指标.csv", 
        index=False, encoding="utf-8-sig"
    )

    p, r, f1, sup = precision_recall_fscore_support(
        metrics["y_true"], metrics["y_pred"], labels=ALL_LABELS, zero_division=0
    )
    class_data = {
        "类别名称": CLASS_NAMES, 
        "精确率": np.round(p, 4), 
        "召回率": np.round(r, 4), 
        "F1分数": np.round(f1, 4), 
        "样本数": sup
    }
    pd.DataFrame(class_data).to_csv(
        RESULT_SAVE / f"{model_type}_enh{enhance_level}_类别指标.csv", 
        index=False, encoding="utf-8-sig"
    )
    print(f"✅ 评估结果已保存为CSV (模型: {model_type}, 等级: {enhance_level})")

# ===================== 主函数（循环评估所有增强等级） =====================
def main():
    _, _, test_loader, _ = build_dataloader(batch_size=BATCH_SIZE)
    
    for enhance_level in ENHANCE_LEVELS:
        print(f"\n{'='*60}")
        print(f"🔹 全模型评估 | 模型: {MODEL_TYPE} | 增强等级: {enhance_level}")
        print(f"{'='*60}")
        
        weight_path = BASE_SAVE / f"{MODEL_TYPE}_enh{enhance_level}_best.pth"
        if not weight_path.exists():
            print(f"⚠️ 权重文件不存在，跳过: {weight_path}")
            continue
        
        model = load_model_from_path(MODEL_TYPE, weight_path)
        metrics = evaluate_model(model, test_loader)
        
        print("\n" + "="*60)
        print(f"📊 测试集核心指标 (等级 {enhance_level})")
        print(f"准确率: {metrics['accuracy']:.4f} | Macro F1: {metrics['f1_macro']:.4f} | AUC: {metrics['roc_auc']:.4f} | FPS: {metrics['fps']:.2f}")
        print("="*60)
        
        print("\n📋 分类报告：")
        print(classification_report(metrics['y_true'], metrics['y_pred'], target_names=CLASS_NAMES, zero_division=0))
        
        # 绘图
        plot_confusion_matrix(metrics['confusion_matrix'], CLASS_NAMES, 
                              FIGURE_SAVE / f"{MODEL_TYPE}_enh{enhance_level}_混淆矩阵.png")
        plot_roc_curve(metrics['y_true'], metrics['y_score'], NUM_CLASSES, CLASS_NAMES, 
                       FIGURE_SAVE / f"{MODEL_TYPE}_enh{enhance_level}_ROC曲线.png")
        # 保存CSV
        save_results_to_csv(metrics,  MODEL_TYPE, enhance_level)
    
    print("\n🎉 所有增强等级评估完成！图表+CSV已保存至 chapter4_results")

if __name__ == "__main__":
    main()