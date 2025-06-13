# -*- coding: utf-8 -*-
"""
评估指标模块
提供各种分割任务的评估指标计算函数
"""

import torch
import segmentation_models_pytorch as smp
from typing import Dict, Any

def get_metrics():
    """获取评估指标 - 使用新的函数式API"""
    def compute_metrics(output, target):
        # 计算统计信息
        tp, fp, fn, tn = smp.metrics.get_stats(
            output, target, 
            mode='binary', 
            threshold=0.5
        )
        
        # 计算各种指标
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")
        recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro")
        precision = smp.metrics.precision(tp, fp, fn, tn, reduction="micro")
        
        # 确保所有指标都转移到CPU
        return {
            'iou': iou.cpu().item(),
            'f1': f1.cpu().item(),
            'accuracy': accuracy.cpu().item(),
            'recall': recall.cpu().item(),
            'precision': precision.cpu().item()
        }
    
    return compute_metrics

def dice_coef(pred, target, smooth=1e-5):
    """计算Dice系数"""
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    result = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return result.cpu().item() if hasattr(result, 'cpu') else result

def iou_score(pred, target, smooth=1e-5):
    """计算IoU分数"""
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    result = (intersection + smooth) / (union + smooth)
    return result.cpu().item() if hasattr(result, 'cpu') else result