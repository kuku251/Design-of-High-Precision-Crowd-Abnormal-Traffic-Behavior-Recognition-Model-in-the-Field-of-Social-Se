# -*- coding: utf-8 -*-
# model_architecture_lbs.py - 适配3069维输入的模型（其他同原版）
import torch
import torch.nn as nn
from model_architecture import MobileNetV2_Classifier, ResNet50_Classifier, NUM_CLASSES, NUM_RISK

INPUT_DIM_LBS = 3069  # 3063 + 6

def init_model_lbs(model_type, num_classes=NUM_CLASSES, use_cbam=True):
    if model_type == "mobilenet":
        model = MobileNetV2_Classifier(
            input_dim=INPUT_DIM_LBS,
            num_classes=num_classes,
            num_risk=NUM_RISK,
            use_cbam=use_cbam
        )
    elif model_type == "resnet50":
        model = ResNet50_Classifier(
            input_dim=INPUT_DIM_LBS,
            num_classes=num_classes,
            num_risk=NUM_RISK
        )
    else:
        raise ValueError(f"未知模型类型: {model_type}")
    return model