# -*- coding: utf-8 -*-
# eval_lbs_compare.py - 公平对比版本：统一使用原始测试集
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score, f1_score
from data_loader import build_dataloader, BATCH_SIZE
from model_architecture import init_model
from model_architecture_lbs import init_model_lbs

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_TYPE = "mobilenet"
ENHANCE_LEVEL = 2

# 权重路径
WEIGHT_NO_LBS = Path("./chapter4_results/models/mobilenet_enh2_best.pth")
WEIGHT_WITH_LBS = Path("./chapter5_results/models/mobilenet_lbs_enh2_best.pth")

def evaluate(model, test_loader, input_dim=3063):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y, _ in test_loader:
            X = X.to(DEVICE)
            # 自动适配维度：若模型期望3069但数据只有3063，则补零
            if input_dim == 3069 and X.shape[1] == 3063:
                zero_pad = torch.zeros(X.shape[0], 6, device=DEVICE)
                X = torch.cat([X, zero_pad], dim=1)
            elif input_dim == 3063 and X.shape[1] == 3069:
                X = X[:, :3063]
            out, _ = model(X)
            pred = torch.argmax(out, dim=1)
            y_true.extend(y.numpy())
            y_pred.extend(pred.cpu().numpy())
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return acc, f1, y_true, y_pred

def main():
    # 统一使用原始 data_loader 的测试集（确保样本一致）
    _, _, test_loader, _ = build_dataloader(batch_size=BATCH_SIZE, enhance_level=ENHANCE_LEVEL)
    
    # 无LBS模型
    model_no = init_model(MODEL_TYPE, use_cbam=True).to(DEVICE)
    model_no.load_state_dict(torch.load(WEIGHT_NO_LBS, map_location=DEVICE))
    acc_no, f1_no, y_true, y_pred_no = evaluate(model_no, test_loader, input_dim=3063)
    
    # 有LBS模型（测试时输入3063维，自动补零模拟缺失LBS特征）
    model_lbs = init_model_lbs(MODEL_TYPE, use_cbam=True).to(DEVICE)
    model_lbs.load_state_dict(torch.load(WEIGHT_WITH_LBS, map_location=DEVICE))
    acc_lbs, f1_lbs, _, y_pred_lbs = evaluate(model_lbs, test_loader, input_dim=3069)
    
    print("\n📊 有无模拟LBS特征性能对比（相同测试集，样本数一致）：")
    print("="*50)
    print(f"无LBS特征模型: Accuracy={acc_no:.4f}, Macro F1={f1_no:.4f}")
    print(f"有LBS特征模型: Accuracy={acc_lbs:.4f}, Macro F1={f1_lbs:.4f}")
    print("\n📋 无LBS特征分类报告：")
    print(classification_report(y_true, y_pred_no, target_names=['正常行走','人体异常','车辆闯入'], digits=4))
    print("\n📋 有LBS特征分类报告：")
    print(classification_report(y_true, y_pred_lbs, target_names=['正常行走','人体异常','车辆闯入'], digits=4))
    
    compare_df = pd.DataFrame({
        "配置": ["无LBS", "有LBS"],
        "准确率": [acc_no, acc_lbs],
        "Macro F1": [f1_no, f1_lbs]
    })
    compare_df.to_csv(Path("./chapter5_results/lbs_ablation_comparison_fair.csv"), index=False, encoding="utf-8-sig")
    print("\n✅ 公平对比结果已保存至 chapter5_results/lbs_ablation_comparison_fair.csv")

if __name__ == "__main__":
    main()