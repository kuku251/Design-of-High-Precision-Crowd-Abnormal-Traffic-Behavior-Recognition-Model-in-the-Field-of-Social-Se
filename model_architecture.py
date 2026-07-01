# -*- coding: utf-8 -*-
# model_architecture.py - 3分类适配版（支持二阶段分类）
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import os
warnings.filterwarnings('ignore')

# ===================== 全局配置 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 3063
HIDDEN_DIM = 256
NUM_CLASSES = 3
NUM_RISK = 3
SAVE_DIR = "./chapter4_results/models/"

os.makedirs(SAVE_DIR, exist_ok=True)

# ===================== 1. CBAM空间-通道双注意力模块 =====================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, in_planes // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes // ratio, in_planes, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).squeeze(-1)).unsqueeze(-1)
        max_out = self.fc(self.max_pool(x).squeeze(-1)).unsqueeze(-1)
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return x * self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.ca(x)
        x = self.sa(x)
        return x.squeeze(-1)

# ===================== 2. 轻量化模型：MobileNetV2 =====================
class MobileNetV2_Classifier(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES, num_risk=NUM_RISK, use_cbam=True):
        super(MobileNetV2_Classifier, self).__init__()
        self.use_cbam = use_cbam
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Dropout(0.2)
        )
        if use_cbam:
            self.cbam = CBAM(hidden_dim)
        self.depthwise_conv1 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU6(inplace=True)
        )
        self.pointwise_conv1 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim*2, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_dim*2),
            nn.ReLU6(inplace=True)
        )
        self.depthwise_conv2 = nn.Sequential(
            nn.Conv1d(hidden_dim*2, hidden_dim*2, kernel_size=3, padding=1, groups=hidden_dim*2, bias=False),
            nn.BatchNorm1d(hidden_dim*2),
            nn.ReLU6(inplace=True)
        )
        self.pointwise_conv2 = nn.Sequential(
            nn.Conv1d(hidden_dim*2, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU6(inplace=True)
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.risk_classifier = nn.Linear(hidden_dim, num_risk)

    def forward(self, x):
        x = self.input_layer(x)
        if self.use_cbam:
            x = self.cbam(x)
        x = x.unsqueeze(-1)
        x = self.depthwise_conv1(x)
        x = self.pointwise_conv1(x)
        x = self.depthwise_conv2(x)
        x = self.pointwise_conv2(x)
        x = self.avg_pool(x).squeeze(-1)
        
        pred_cls = self.classifier(x)
        pred_risk = self.risk_classifier(x)
        return pred_cls, pred_risk

# ===================== 3. 高精度模型：ResNet50 =====================
class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_dim, out_dim, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_dim)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_dim, out_dim, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_dim)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet50_Classifier(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES, num_risk=NUM_RISK):
        super(ResNet50_Classifier, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.cbam = CBAM(hidden_dim)
        
        # ResNet blocks
        self.layer1 = self._make_layer(hidden_dim, hidden_dim, 3, stride=1)
        self.layer2 = self._make_layer(hidden_dim, hidden_dim*2, 4, stride=2)
        self.layer3 = self._make_layer(hidden_dim*2, hidden_dim, 3, stride=1)
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.risk_classifier = nn.Linear(hidden_dim, num_risk)

    def _make_layer(self, in_dim, out_dim, num_blocks, stride):
        layers = [ResidualBlock(in_dim, out_dim, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_dim, out_dim, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.cbam(x)
        x = x.unsqueeze(-1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x).squeeze(-1)
        
        pred_cls = self.classifier(x)
        pred_risk = self.risk_classifier(x)
        return pred_cls, pred_risk

# ===================== 4. 二阶段分类专用模型（单输出） =====================
class StageClassifier(nn.Module):
    """二阶段分类专用模型，只输出分类结果"""
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_classes=2):
        super(StageClassifier, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Dropout(0.2)
        )
        self.cbam = CBAM(hidden_dim)
        self.depthwise_conv1 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU6(inplace=True)
        )
        self.pointwise_conv1 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim*2, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_dim*2),
            nn.ReLU6(inplace=True)
        )
        self.depthwise_conv2 = nn.Sequential(
            nn.Conv1d(hidden_dim*2, hidden_dim*2, kernel_size=3, padding=1, groups=hidden_dim*2, bias=False),
            nn.BatchNorm1d(hidden_dim*2),
            nn.ReLU6(inplace=True)
        )
        self.pointwise_conv2 = nn.Sequential(
            nn.Conv1d(hidden_dim*2, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU6(inplace=True)
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.cbam(x)
        x = x.unsqueeze(-1)
        x = self.depthwise_conv1(x)
        x = self.pointwise_conv1(x)
        x = self.depthwise_conv2(x)
        x = self.pointwise_conv2(x)
        x = self.avg_pool(x).squeeze(-1)
        pred = self.classifier(x)
        return pred

# ===================== 5. 模型初始化函数 =====================
def init_model(model_type, num_classes=NUM_CLASSES, use_cbam=True):
    if model_type == "mobilenet":
        model = MobileNetV2_Classifier(num_classes=num_classes, num_risk=NUM_RISK, use_cbam=use_cbam)
    elif model_type == "resnet50":
        model = ResNet50_Classifier(num_classes=num_classes, num_risk=NUM_RISK, use_cbam=use_cbam)
    return model

def init_stage_model(model_type, num_classes=2):
    """
    初始化二阶段分类专用模型（单输出）
    
    Args:
        model_type: 模型类型
        num_classes: 分类类别数（默认2）
    
    Returns:
        model: 初始化的模型
    """
    if model_type == "mobilenet":
        model = StageClassifier(num_classes=num_classes)
    elif model_type == "resnet50":
        # ResNet50 也使用 StageClassifier
        model = StageClassifier(num_classes=num_classes)
    else:
        raise ValueError(f"未知模型类型: {model_type}")
    
    return model

# ===================== 测试 =====================
if __name__ == "__main__":
    # 测试三分类模型
    model = init_model("mobilenet", num_classes=3)
    print(f"三分类模型输出维度: {model.classifier.out_features}")
    
    # 测试二阶段模型
    model_s1 = init_stage_model("mobilenet", num_classes=2)
    print(f"二阶段模型输出维度: {model_s1.classifier.out_features}")
    
    # 测试前向传播
    x = torch.randn(1, INPUT_DIM)
    out = model_s1(x)
    print(f"前向传播输出维度: {out.shape}")
