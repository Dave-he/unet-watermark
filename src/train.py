# -*- coding: utf-8 -*-
"""
训练模块
提供模型训练、验证和早停机制
"""

import os
import torch
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import logging
from datetime import datetime
from typing import Dict, Tuple, Any, Optional

# 设置多进程启动方法为spawn以支持CUDA
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    # 如果已经设置过，忽略错误
    pass

# 导入自定义模块
from configs.config import get_cfg_defaults, update_config
from models.unet_model import create_model_from_config
from utils.dataset import create_datasets
from utils.losses import get_loss_function
from utils.metrics import get_metrics, dice_coef

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

def train_epoch(model, train_loader, criterion, optimizer, device, metrics, cfg, scheduler=None):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    metric_values = {'iou': 0.0, 'f1': 0.0, 'accuracy': 0.0, 'recall': 0.0, 'precision': 0.0}
    
    # 混合精度训练
    scaler = torch.GradScaler() if device.type == 'cuda' else None
    
    # 减少指标计算频率
    metric_calc_interval = max(1, len(train_loader) // 10)
    
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")
    
    for i, (images, masks) in progress_bar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # 前向传播
        if scaler:
            with torch.autocast(device_type="cuda"):
                outputs = model(images)
                if masks.dim() == 3:
                    masks = masks.unsqueeze(1)
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        
        # 只在特定间隔计算指标
        if i % metric_calc_interval == 0:
            with torch.no_grad():
                preds = torch.sigmoid(outputs)
                preds_3d = preds.squeeze(1) if preds.dim() == 4 else preds
                masks_3d = masks.squeeze(1) if masks.dim() == 4 else masks
                batch_metrics = metrics(preds_3d, masks_3d)
                for name, value in batch_metrics.items():
                    metric_values[name] += value
        
        # 更新进度条
        if (i + 1) % cfg.TRAIN.LOG_INTERVAL == 0:
            avg_loss = total_loss / (i + 1)
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    avg_loss = total_loss / len(train_loader)
    avg_metrics = {name: val / len(train_loader) for name, val in metric_values.items()}
    
    return avg_loss, avg_metrics

def validate(model, val_loader, criterion, device, metrics):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    metric_values = {'iou': 0.0, 'f1': 0.0, 'accuracy': 0.0, 'recall': 0.0, 'precision': 0.0}
    
    with torch.no_grad():
        progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validation")
        
        for i, (images, masks) in progress_bar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)  # 输出形状: (N, 1, H, W)
            
            # 确保目标张量也是4D
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)  # (N, H, W) -> (N, 1, H, W)
            
            loss = criterion(outputs, masks)
            
            # 计算指标
            preds = torch.sigmoid(outputs)
            # 为了计算指标，将预测和目标都压缩为3D
            preds_3d = preds.squeeze(1) if preds.dim() == 4 else preds
            masks_3d = masks.squeeze(1) if masks.dim() == 4 else masks
            batch_metrics = metrics(preds_3d, masks_3d)
            for name, value in batch_metrics.items():
                metric_values[name] += value
            
            total_loss += loss.item()
            
            # 更新进度条
            avg_loss = total_loss / (i + 1)
            avg_metrics = {name: val / (i + 1) for name, val in metric_values.items()}
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'iou': f'{avg_metrics["iou"]:.4f}',
                'f1': f'{avg_metrics["f1"]:.4f}'
            })
    
    avg_loss = total_loss / len(val_loader)
    avg_metrics = {name: val / len(val_loader) for name, val in metric_values.items()}
    
    return avg_loss, avg_metrics

def save_training_plots(train_losses, val_losses, train_metrics, val_metrics, output_dir):
    """保存训练过程图表"""
    plt.figure(figsize=(15, 10))
    
    # 损失曲线
    plt.subplot(2, 3, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Val Loss', color='red')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 指标曲线
    metric_names = ['iou', 'f1', 'accuracy', 'recall', 'precision']
    for idx, metric_name in enumerate(metric_names):
        plt.subplot(2, 3, idx + 2)
        train_values = [m[metric_name] for m in train_metrics]
        val_values = [m[metric_name] for m in val_metrics]
        plt.plot(train_values, label=f'Train {metric_name.upper()}', color='blue')
        plt.plot(val_values, label=f'Val {metric_name.upper()}', color='red')
        plt.title(f'{metric_name.upper()} Curves')
        plt.xlabel('Epoch')
        plt.ylabel(metric_name.upper())
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

def train(cfg, resume_from=None, use_blurred_mask=False):
    """训练函数
    
    Args:
        cfg: 配置对象
        resume_from: 要恢复的checkpoint路径，如果为None则从头开始训练
        use_blurred_mask: 是否使用模糊的mask进行训练，默认False生成精确mask
    """
    # 创建输出目录
    os.makedirs(os.path.dirname(cfg.TRAIN.MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(cfg.TRAIN.OUTPUT_DIR, exist_ok=True)
    
    # 创建检查点保存目录
    checkpoint_dir = cfg.TRAIN.CHECKPOINT_DIR
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(cfg.TRAIN.MODEL_SAVE_PATH), exist_ok=True)
    
    # 设置设备
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 创建模型
    model = create_model_from_config(cfg).to(device)
    logger.info(f"创建模型: {cfg.MODEL.NAME} with {cfg.MODEL.ENCODER_NAME} encoder")
    
    # 创建数据集和数据加载器
    train_dataset, val_dataset = create_datasets(cfg, use_blurred_mask=use_blurred_mask)
    
    # 记录mask类型
    mask_type = "模糊mask" if use_blurred_mask else "精确mask"
    logger.info(f"使用{mask_type}进行训练")
    
    # 优化数据加载器配置
    num_workers = min(cfg.DATA.NUM_WORKERS, os.cpu_count())
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.TRAIN.BATCH_SIZE, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else 2
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.TRAIN.BATCH_SIZE * 2,  # 验证时可以用更大的batch size
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else False
    )
    
    # 创建损失函数和优化器
    criterion = get_loss_function(cfg)
    
    if cfg.OPTIMIZER.NAME == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.TRAIN.LR,
            weight_decay=cfg.TRAIN.WEIGHT_DECAY
        )
    elif cfg.OPTIMIZER.NAME == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.TRAIN.LR,
            momentum=0.9,
            weight_decay=cfg.TRAIN.WEIGHT_DECAY
        )
    else:
        raise ValueError(f"不支持的优化器: {cfg.OPTIMIZER.NAME}")
    
    # 学习率调度器
    if cfg.OPTIMIZER.LR_SCHEDULER == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=cfg.OPTIMIZER.SCHEDULER_FACTOR, 
            patience=cfg.OPTIMIZER.SCHEDULER_PATIENCE,

        )
    elif cfg.OPTIMIZER.LR_SCHEDULER == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=cfg.TRAIN.EPOCHS
        )
    else:
        scheduler = None
    
    # 评估指标
    metrics = get_metrics()
    
    # 初始化训练状态
    start_epoch = 0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_metrics_history = []
    val_metrics_history = []
    
    # 从checkpoint恢复训练
    if resume_from and os.path.exists(resume_from):
        logger.info(f"从checkpoint恢复训练: {resume_from}")
        try:
            checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
            
            # 加载模型状态
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("成功加载模型权重")
            else:
                # 兼容旧格式的checkpoint
                model.load_state_dict(checkpoint)
                logger.info("成功加载模型权重（旧格式）")
            
            # 加载优化器状态
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info("成功加载优化器状态")
            
            # 加载调度器状态
            if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                logger.info("成功加载调度器状态")
            
            # 恢复训练状态
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch']
                logger.info(f"从第 {start_epoch} 轮继续训练")
            
            if 'best_val_loss' in checkpoint:
                best_val_loss = checkpoint['best_val_loss']
                logger.info(f"恢复最佳验证损失: {best_val_loss:.4f}")
            
            # 可选：恢复训练历史（如果checkpoint中包含）
            if 'train_losses' in checkpoint:
                train_losses = checkpoint['train_losses']
            if 'val_losses' in checkpoint:
                val_losses = checkpoint['val_losses']
            if 'train_metrics_history' in checkpoint:
                train_metrics_history = checkpoint['train_metrics_history']
            if 'val_metrics_history' in checkpoint:
                val_metrics_history = checkpoint['val_metrics_history']
                
        except Exception as e:
            logger.error(f"加载checkpoint失败: {e}")
            logger.info("将从头开始训练")
            start_epoch = 0
            best_val_loss = float('inf')
    elif resume_from:
        logger.warning(f"指定的checkpoint文件不存在: {resume_from}")
        logger.info("将从头开始训练")
    
    # 早停机制 - 根据配置决定是否启用
    early_stopping = None
    if cfg.TRAIN.USE_EARLY_STOPPING:
        early_stopping = EarlyStopping(
            patience=cfg.TRAIN.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True
        )
        logger.info(f"启用早停机制，耐心值: {cfg.TRAIN.EARLY_STOPPING_PATIENCE}")
    else:
        logger.info("禁用早停机制，将训练完整的epoch数")
    
    logger.info(f"开始训练 {cfg.MODEL.NAME} 模型...")
    logger.info(f"训练集: {len(train_dataset)} 张图像")
    logger.info(f"验证集: {len(val_dataset)} 张图像")
    logger.info(f"训练轮数: {cfg.TRAIN.EPOCHS}")
    
    # 优化检查点保存策略
    save_interval = max(5, cfg.TRAIN.EPOCHS // 10)  # 动态调整保存间隔
    
    # 初始化变量，避免作用域问题
    epoch = start_epoch
    val_loss = float('inf')
    train_loss = 0.0
    val_metrics_epoch = {'iou': 0.0, 'f1': 0.0, 'accuracy': 0.0, 'recall': 0.0, 'precision': 0.0}
    train_metrics_epoch = {'iou': 0.0, 'f1': 0.0, 'accuracy': 0.0, 'recall': 0.0, 'precision': 0.0}
    
    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
        logger.info(f"\nEpoch [{epoch+1}/{cfg.TRAIN.EPOCHS}]")
        
        # 训练
        train_loss, train_metrics_epoch = train_epoch(
            model, train_loader, criterion, optimizer, device, metrics, cfg
        )
        
        # 验证
        val_loss, val_metrics_epoch = validate(
            model, val_loader, criterion, device, metrics
        )
        
        # 记录历史
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_metrics_history.append(train_metrics_epoch)
        val_metrics_history.append(val_metrics_epoch)
        
        # 学习率调度
        if scheduler:
            if cfg.OPTIMIZER.LR_SCHEDULER == "ReduceLROnPlateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # 打印训练信息
        logger.info(
            f"Train Loss: {train_loss:.4f}, Train IoU: {train_metrics_epoch['iou']:.4f}, "
            f"Train F1: {train_metrics_epoch['f1']:.4f}"
        )
        logger.info(
            f"Val Loss: {val_loss:.4f}, Val IoU: {val_metrics_epoch['iou']:.4f}, "
            f"Val F1: {val_metrics_epoch['f1']:.4f}"
        )
        
        # 只保存最佳模型和最后几个检查点
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # 只保存必要信息
            best_model_info = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics_epoch,
                'config': cfg
            }
            torch.save(best_model_info, cfg.TRAIN.MODEL_SAVE_PATH)
        
        # 减少检查点保存频率
        if (epoch + 1) % save_interval == 0 or epoch >= cfg.TRAIN.EPOCHS - 3:
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"checkpoint_epoch_{epoch+1:03d}.pth"
            )
            checkpoint_info = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_loss': val_loss,
                'val_metrics': val_metrics_epoch,
                'train_loss': train_loss,
                'train_metrics': train_metrics_epoch,
                'config': cfg,
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_metrics_history': train_metrics_history,
                'val_metrics_history': val_metrics_history
            }
            torch.save(checkpoint_info, checkpoint_path)
            logger.info(f"保存检查点: {checkpoint_path} (Epoch: {epoch+1})")
        
        # 早停检查 - 只有在启用早停时才进行检查
        if early_stopping and early_stopping(val_loss, model):
            logger.info(f"早停触发，在第 {epoch+1} 轮停止训练")
            break
    
    # 保存最终模型
    final_model_path = os.path.join(
        checkpoint_dir,
        f"final_model_epoch_{epoch+1:03d}.pth"
    )
    final_model_info = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_loss': val_loss,
        'val_metrics': val_metrics_epoch,
        'train_loss': train_loss,
        'train_metrics': train_metrics_epoch,
        'config': cfg,
        'best_val_loss': best_val_loss,
        'is_final': True
    }
    torch.save(final_model_info, final_model_path)
    logger.info(f"保存最终模型: {final_model_path}")
    
        
        
    # 保存训练曲线
    save_training_plots(
        train_losses, val_losses, 
        train_metrics_history, val_metrics_history, 
        cfg.TRAIN.OUTPUT_DIR
    )
    
    logger.info(f"训练完成！最佳验证损失: {best_val_loss:.4f}")
    logger.info(f"模型保存在: {cfg.TRAIN.MODEL_SAVE_PATH}")
    logger.info(f"训练曲线保存在: {cfg.TRAIN.OUTPUT_DIR}/training_curves.png")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='SMP UNet++ 水印检测训练')
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/unet_watermark.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--device', 
        type=str, 
        default=None,
        help='训练设备 (cuda/cpu)'
    )
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=None,
        help='批次大小'
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=None,
        help='训练轮数'
    )
    parser.add_argument(
        '--lr', 
        type=float, 
        default=None,
        help='学习率'
    )
    parser.add_argument(
        '--no-early-stopping', 
        action='store_true',
        help='禁用早停机制'
    )
    parser.add_argument(
        '--early-stopping-patience', 
        type=int, 
        default=None,
        help='早停耐心值'
    )
    parser.add_argument(
        '--resume', 
        type=str, 
        default=None,
        help='从指定的checkpoint文件恢复训练'
    )
    parser.add_argument(
        '--use-blurred-mask', 
        action='store_true',
        help='使用模糊的mask进行训练，默认生成精确的mask'
    )
    
    args = parser.parse_args()
    
    # 加载配置
    cfg = get_cfg_defaults()
    if os.path.exists(args.config):
        update_config(cfg, args.config)
    else:
        logger.warning(f"配置文件不存在: {args.config}，使用默认配置")
    
    # 命令行参数覆盖配置
    if args.device:
        cfg.DEVICE = args.device
    if args.batch_size:
        cfg.TRAIN.BATCH_SIZE = args.batch_size
    if args.epochs:
        cfg.TRAIN.EPOCHS = args.epochs
    if args.lr:
        cfg.TRAIN.LR = args.lr
    if args.no_early_stopping:
        cfg.TRAIN.USE_EARLY_STOPPING = False
    if args.early_stopping_patience:
        cfg.TRAIN.EARLY_STOPPING_PATIENCE = args.early_stopping_patience
    
    # 开始训练
    train(cfg, resume_from=args.resume, use_blurred_mask=getattr(args, 'use_blurred_mask', False))

if __name__ == "__main__":
    main()