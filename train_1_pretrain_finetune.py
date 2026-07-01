# -*- coding: utf-8 -*-
# 10_train_1_pretrain_finetune.py - 支持 enhance_level 消融实验
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report, roc_auc_score, recall_score
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ===================== 导入模块 =====================
from model_architecture import init_model, INPUT_DIM, HIDDEN_DIM
NUM_CLASSES = 3
NUM_RISK = 3
from data_loader import build_dataloader, BATCH_SIZE

# ===================== 全局配置 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_TYPE = "mobilenet"
# MODEL_TYPE = "resnet50"
BASE_DIR = Path("./chapter4_results/")
BASE_SAVE = BASE_DIR / "models"
BASE_SAVE.mkdir(parents=True, exist_ok=True)
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)

# ===================== 消融实验关键变量 =====================
ENHANCE_LEVEL = 2   # 修改此处以运行不同等级：0/1/2
USE_CBAM = True

# ===================== FocalLoss（用于等级1和2） =====================
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

# ===================== 指标计算 =====================
def calculate_metrics(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
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

# ===================== 训练主函数 =====================
def finetune_main():
    train_losses = []
    val_accs = []

    print(f"\n🚀 启动训练 | 模型: {MODEL_TYPE} | 增强等级: {ENHANCE_LEVEL}")
    
    train_loader, val_loader, test_loader, class_weights = build_dataloader(
        batch_size=BATCH_SIZE, enhance_level=ENHANCE_LEVEL
    )
    
    model = init_model(MODEL_TYPE, use_cbam=USE_CBAM).to(DEVICE)
    assert model.classifier.out_features == NUM_CLASSES, f"模型分类器输出错误！"

    # ========== 根据增强等级选择损失函数 ==========
    if ENHANCE_LEVEL == 0:
        cls_criterion = nn.CrossEntropyLoss().to(DEVICE)
        print("🔧 损失函数: 标准交叉熵 (CrossEntropyLoss)")
    else:
        cls_criterion = FocalLoss(alpha=class_weights, gamma=2.0).to(DEVICE)
        print(f"🔧 损失函数: FocalLoss (gamma=2.0, 类别权重已启用)")
    
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    best_acc = 0.0
    best_normal_recall = 0.0
    patience = 10
    patience_counter = 0
    min_epochs = 30

    # 模型保存路径（包含增强等级）
    cbam_str = "cbam" if USE_CBAM else "noCBAM"
    save_path = BASE_SAVE / f"{MODEL_TYPE}_{cbam_str}_enh{ENHANCE_LEVEL}_best.pth"
    # save_path = BASE_SAVE / f"{MODEL_TYPE}_enh{ENHANCE_LEVEL}_best.pth"

    print(f"\n📁 模型将保存至: {save_path}")
    for epoch in range(100):
        # 训练
        model.train()
        train_correct = train_total = 0
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
            _, pred = torch.max(pred_cls, 1)
            train_correct += (pred == y_cls).sum().item()
            train_total += y_cls.size(0)
            train_pbar.set_postfix({"loss": round(loss.item(),4)})

        train_acc = train_correct / train_total
        train_avg_loss = train_loss_sum / train_total

        # 验证
        val_acc, val_normal_recall, val_macro_f1, val_auc, _, _ = calculate_metrics(model, val_loader)
        scheduler.step()

        print(f"\n📊 Epoch {epoch+1:02d} | 训练损失:{train_avg_loss:.4f} | 训练精度:{train_acc:.4f}")
        print(f"🔍 验证指标：准确率:{val_acc:.4f} | 正常类召回:{val_normal_recall:.4f} | Macro F1:{val_macro_f1:.4f}")
        
        train_losses.append(train_avg_loss)
        val_accs.append(val_acc)
        # 早停
        if epoch >= min_epochs:
            if val_acc > best_acc and val_normal_recall >= 0.8:
                best_acc = val_acc
                best_normal_recall = val_normal_recall
                patience_counter = 0
                torch.save(model.state_dict(), save_path)
                print(f"✅ 保存最优模型 (准确率: {val_acc:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"⏹️ 早停触发，停止训练")
                    np.save(BASE_SAVE / f"{MODEL_TYPE}_enh{ENHANCE_LEVEL}_losses.npy", train_losses)
                    np.save(BASE_SAVE / f"{MODEL_TYPE}_enh{ENHANCE_LEVEL}_accs.npy", val_accs)
                    break

    # 测试评估
    print("\n🔍 测试集最终评估：")
    model.load_state_dict(torch.load(save_path))
    test_acc, test_normal_recall, test_macro_f1, test_auc, test_preds, test_labels = calculate_metrics(model, test_loader)
    print(classification_report(test_labels, test_preds, target_names=['正常行走', '人体异常', '车辆闯入'], digits=4))
    print(f"\n✅ 最终结果 (增强等级{ENHANCE_LEVEL})：准确率:{test_acc:.4f} | 正常类召回:{test_normal_recall:.4f} | Macro F1:{test_macro_f1:.4f}")

if __name__ == "__main__":
    finetune_main()