# -*- coding: utf-8 -*-
# train_lbs.py - 训练包含模拟LBS特征的模型
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report, roc_auc_score
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from model_architecture_lbs import init_model_lbs, INPUT_DIM_LBS
from data_loader_lbs import build_dataloader_lbs, BATCH_SIZE
from model_architecture import MobileNetV2_Classifier, ResNet50_Classifier, NUM_CLASSES, NUM_RISK

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# MODEL_TYPE = "mobilenet"
MODEL_TYPE = "resnet50"
ENHANCE_LEVEL = 0
USE_CBAM = True
NUM_CLASSES = 3

BASE_DIR = Path("./chapter5_results/")
BASE_SAVE = BASE_DIR / "models"
BASE_SAVE.mkdir(parents=True, exist_ok=True)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        p_t = torch.exp(-ce_loss)
        loss = ((1 - p_t) ** self.gamma) * ce_loss
        return loss.mean()

def calculate_metrics(model, loader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch_X, y_cls, _ in loader:
            batch_X = batch_X.to(DEVICE)
            pred_cls, _ = model(batch_X)
            probs = torch.softmax(pred_cls, dim=1)
            _, pred = torch.max(pred_cls, 1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y_cls.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    normal_mask = (np.array(all_labels) == 0)
    normal_recall = np.sum((np.array(all_preds) == 0) & normal_mask) / np.sum(normal_mask) if np.sum(normal_mask) > 0 else 0.0
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    except:
        auc = 0.0
    return acc, normal_recall, macro_f1, auc, all_preds, all_labels

def main():
    print(f"\n🚀 第五章实验：含模拟LBS特征 | 模型: {MODEL_TYPE} | 增强等级: {ENHANCE_LEVEL}")
    train_loader, val_loader, test_loader, class_weights = build_dataloader_lbs(enhance_level=ENHANCE_LEVEL)
    model = init_model_lbs(MODEL_TYPE, use_cbam=USE_CBAM).to(DEVICE)
    
    cls_criterion = FocalLoss(alpha=class_weights, gamma=2.0).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    
    best_acc = 0.0
    patience = 10
    patience_counter = 0
    save_path = BASE_SAVE / f"{MODEL_TYPE}_lbs_enh{ENHANCE_LEVEL}_best.pth"
    
    # 训练循环修正部分
    for epoch in range(100):
        model.train()
        train_loss_sum = 0.0
        train_pbar = tqdm(train_loader, desc=f"训练 Epoch {epoch+1}/100")
        for batch_X, y_cls, _ in train_pbar:
            batch_X, y_cls = batch_X.to(DEVICE), y_cls.to(DEVICE)
            pred_cls, _ = model(batch_X)
            loss = cls_criterion(pred_cls, y_cls)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_sum += loss.item() * batch_X.size(0)
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})   # 修正格式
        scheduler.step()
        
        # calculate_metrics 返回 (acc, normal_recall, macro_f1, auc, preds, labels)
        # 需要接收全部或用 _ 占位
        val_acc, _, val_f1, _, _, _ = calculate_metrics(model, val_loader)
        print(f"Epoch {epoch+1}: val_acc={val_acc:.4f}, val_f1={val_f1:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"✅ 保存最优模型")
        else:
            patience_counter += 1
            if patience_counter >= patience and epoch >= 30:
                print("⏹️ 早停触发")   # 正确缩进
                break
        
        # ... 训练循环代码保持不变 ...
        
    # 测试
    model.load_state_dict(torch.load(save_path))
    test_acc, _, test_f1, _, test_preds, test_labels = calculate_metrics(model, test_loader)
    print("\n🔍 测试集结果（含LBS特征）：")
    print(classification_report(test_labels, test_preds, target_names=['正常行走','人体异常','车辆闯入'], digits=4))
    print(f"准确率: {test_acc:.4f} | Macro F1: {test_f1:.4f}")

if __name__ == "__main__":
    main()