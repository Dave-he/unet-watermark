# -*- coding: utf-8 -*-
# @Time    : 2023/3/23 14:46
import torch.nn as nn
import segmentation_models_pytorch as smp

def get_loss_function(cfg):
    """根据配置获取损失函数"""
    loss_name = cfg.LOSS.NAME
    # 使用正确的配置项名称，如果不存在则使用默认值
    mode = getattr(cfg.LOSS, 'MODE', 'binary')
    smooth = getattr(cfg.LOSS, 'SMOOTH', cfg.LOSS.DICE_SMOOTH)
    
    if loss_name == "DiceLoss":
        return smp.losses.DiceLoss(mode=mode, smooth=smooth)
    elif loss_name == "JaccardLoss":
        return smp.losses.JaccardLoss(mode=mode, smooth=smooth)
    elif loss_name == "BCEWithLogitsLoss":
        return nn.BCEWithLogitsLoss()
    elif loss_name == "FocalLoss":
        return smp.losses.FocalLoss(mode=mode)
    elif loss_name == "TverskyLoss":
        return smp.losses.TverskyLoss(mode=mode, smooth=smooth)
    elif loss_name == "LovaszLoss":
        return smp.losses.LovaszLoss(mode=mode)
    else:
        raise ValueError(f"不支持的损失函数: {loss_name}")

class CombinedLoss(nn.Module):
    """组合损失函数"""
    
    def __init__(self, losses, weights=None):
        """
        初始化组合损失函数
        
        Args:
            losses (list): 损失函数列表
            weights (list): 权重列表
        """
        super(CombinedLoss, self).__init__()
        self.losses = losses
        self.weights = weights if weights else [1.0] * len(losses)
    
    def forward(self, pred, target):
        total_loss = 0
        for loss_fn, weight in zip(self.losses, self.weights):
            total_loss += weight * loss_fn(pred, target)
        return total_loss