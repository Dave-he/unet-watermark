#!/usr/bin/env python3
"""
文字水印检测模型训练脚本
专门针对文字水印优化的UNet模型训练
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# 添加src目录到路径
sys.path.append('src')

from configs.config import get_cfg_defaults, update_config
from models.unet_model import create_model_from_config
from utils.dataset import WatermarkDataset, get_train_transform, get_val_transform
from utils.losses import get_loss_function
from utils.metrics import calculate_metrics
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import logging
from tqdm import tqdm
import cv2
from sklearn.metrics import precision_recall_fscore_support

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextWatermarkTrainer:
    """文字水印检测模型训练器"""
    
    def __init__(self, config_path="src/configs/unet_text_watermark.yaml"):
        # 加载配置
        self.cfg = get_cfg_defaults()
        if os.path.exists(config_path):
            update_config(self.cfg, config_path)
        else:
            logger.warning(f"配置文件不存在: {config_path}，使用默认配置")
        
        # 设置设备
        self.device = torch.device(self.cfg.DEVICE if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 创建输出目录
        os.makedirs(self.cfg.TRAIN.OUTPUT_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(self.cfg.TRAIN.MODEL_SAVE_PATH), exist_ok=True)
        
        # 初始化模型
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # 训练状态
        self.best_val_dice = 0.0
        self.patience_counter = 0
        
    def _create_model(self):
        """创建模型"""
        logger.info("创建文字水印检测模型...")
        self.model = create_model_from_config(self.cfg)
        self.model.to(self.device)
        
        # 打印模型信息
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"模型参数总数: {total_params:,}")
        logger.info(f"可训练参数: {trainable_params:,}")
        
    def _create_optimizer(self):
        """创建优化器和调度器"""
        # 创建优化器
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.cfg.TRAIN.LR,
            weight_decay=self.cfg.TRAIN.WEIGHT_DECAY
        )
        
        # 创建学习率调度器
        if self.cfg.OPTIMIZER.LR_SCHEDULER == "CosineAnnealingWarmRestarts":
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.cfg.OPTIMIZER.SCHEDULER_T_0,
                T_mult=self.cfg.OPTIMIZER.SCHEDULER_T_MULT,
                eta_min=self.cfg.OPTIMIZER.SCHEDULER_ETA_MIN
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                verbose=True
            )
        
        logger.info(f"优化器: {self.cfg.OPTIMIZER.NAME}")
        logger.info(f"学习率调度器: {self.cfg.OPTIMIZER.LR_SCHEDULER}")
        
    def _create_loss_function(self):
        """创建损失函数"""
        self.criterion = get_loss_function(self.cfg)
        logger.info(f"损失函数: {self.cfg.LOSS.NAME}")
        
    def _create_data_loaders(self):
        """创建数据加载器"""
        logger.info("创建数据加载器...")
        
        # 获取数据变换
        train_transform = get_train_transform(self.cfg)
        val_transform = get_val_transform(self.cfg)
        
        # 创建数据集
        train_dataset = WatermarkDataset(
            root_dir=self.cfg.DATA.ROOT_DIR,
            transform=train_transform,
            is_train=True,
            cfg=self.cfg
        )
        
        val_dataset = WatermarkDataset(
            root_dir=self.cfg.DATA.ROOT_DIR,
            transform=val_transform,
            is_train=False,
            cfg=self.cfg
        )
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            shuffle=True,
            num_workers=self.cfg.DATA.NUM_WORKERS,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            shuffle=False,
            num_workers=self.cfg.DATA.NUM_WORKERS,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        logger.info(f"训练集大小: {len(train_dataset)}")
        logger.info(f"验证集大小: {len(val_dataset)}")
        
    def _enhance_text_features_batch(self, images):
        """批量增强文字特征"""
        if not self.cfg.DATA.get('TEXT_ENHANCEMENT', False):
            return images
        
        enhanced_images = []
        for img in images:
            # 转换为numpy格式
            if isinstance(img, torch.Tensor):
                img_np = img.permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img
            
            # 应用文字特征增强
            enhanced_img = self._enhance_single_image(img_np)
            
            # 转换回tensor格式
            enhanced_img = enhanced_img.astype(np.float32) / 255.0
            enhanced_img = torch.from_numpy(enhanced_img.transpose(2, 0, 1))
            enhanced_images.append(enhanced_img)
        
        return torch.stack(enhanced_images)
    
    def _enhance_single_image(self, image_rgb):
        """增强单张图像的文字特征"""
        # 对比度增强
        if self.cfg.DATA.get('CONTRAST_BOOST', 1.0) != 1.0:
            image_rgb = cv2.convertScaleAbs(image_rgb, alpha=self.cfg.DATA.CONTRAST_BOOST, beta=0)
        
        # 边缘增强
        if self.cfg.DATA.get('EDGE_ENHANCEMENT', False):
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edges_3channel = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            image_rgb = cv2.addWeighted(image_rgb, 0.8, edges_3channel, 0.2, 0)
        
        return image_rgb
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.cfg.TRAIN.EPOCHS}")
        
        for batch_idx, (images, masks) in enumerate(progress_bar):
            # 移动到设备
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # 文字特征增强
            if self.cfg.DATA.get('TEXT_ENHANCEMENT', False):
                images = self._enhance_text_features_batch(images)
                images = images.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=self.cfg.TRAIN.USE_AMP):
                outputs = self.model(images)
                if isinstance(outputs, dict):
                    outputs = outputs['out']
                loss = self.criterion(outputs, masks)
            
            # 反向传播
            if self.cfg.TRAIN.USE_AMP:
                self.scaler.scale(loss).backward()
                if self.cfg.TRAIN.GRADIENT_CLIP > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.TRAIN.GRADIENT_CLIP)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.cfg.TRAIN.GRADIENT_CLIP > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.TRAIN.GRADIENT_CLIP)
                self.optimizer.step()
            
            total_loss += loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss/(batch_idx+1):.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            # 日志记录
            if (batch_idx + 1) % self.cfg.TRAIN.LOG_INTERVAL == 0:
                logger.info(f"Epoch {epoch+1}/{self.cfg.TRAIN.EPOCHS}, "
                          f"Batch {batch_idx+1}/{num_batches}, "
                          f"Loss: {loss.item():.4f}, "
                          f"Avg Loss: {total_loss/(batch_idx+1):.4f}")
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, epoch):
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc="Validating"):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # 前向传播
                outputs = self.model(images)
                if isinstance(outputs, dict):
                    outputs = outputs['out']
                
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()
                
                # 收集预测结果
                predictions = torch.sigmoid(outputs) > 0.5
                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_targets.extend(masks.cpu().numpy().flatten())
        
        # 计算指标
        avg_loss = total_loss / len(self.val_loader)
        metrics = calculate_metrics(np.array(all_predictions), np.array(all_targets))
        
        # 计算额外的文字特定指标
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='binary', zero_division=0
        )
        
        logger.info(f"Validation - Epoch {epoch+1}:")
        logger.info(f"  Loss: {avg_loss:.4f}")
        logger.info(f"  Dice: {metrics['dice']:.4f}")
        logger.info(f"  IoU: {metrics['iou']:.4f}")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1: {f1:.4f}")
        
        return avg_loss, metrics['dice']
    
    def save_checkpoint(self, epoch, val_dice, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_dice': val_dice,
            'cfg': self.cfg
        }
        
        # 保存最新检查点
        checkpoint_path = os.path.join(self.cfg.TRAIN.OUTPUT_DIR, 'latest_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_model_path = self.cfg.TRAIN.MODEL_SAVE_PATH
            torch.save(checkpoint, best_model_path)
            logger.info(f"保存最佳模型: {best_model_path}")
        
        # 定期保存
        if (epoch + 1) % self.cfg.TRAIN.SAVE_INTERVAL == 0:
            epoch_checkpoint_path = os.path.join(
                self.cfg.TRAIN.OUTPUT_DIR, f'checkpoint_epoch_{epoch+1}.pth'
            )
            torch.save(checkpoint, epoch_checkpoint_path)
    
    def train(self):
        """主训练循环"""
        logger.info("开始训练文字水印检测模型...")
        
        # 初始化组件
        self._create_model()
        self._create_optimizer()
        self._create_loss_function()
        self._create_data_loaders()
        
        # 初始化AMP scaler
        if self.cfg.TRAIN.USE_AMP:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # 训练循环
        for epoch in range(self.cfg.TRAIN.EPOCHS):
            # 训练
            train_loss = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_dice = self.validate(epoch)
            
            # 更新学习率调度器
            if isinstance(self.scheduler, CosineAnnealingWarmRestarts):
                self.scheduler.step()
            else:
                self.scheduler.step(val_dice)
            
            # 检查是否是最佳模型
            is_best = val_dice > self.best_val_dice
            if is_best:
                self.best_val_dice = val_dice
                self.patience_counter = 0
                logger.info(f"新的最佳验证Dice: {val_dice:.4f}")
            else:
                self.patience_counter += 1
            
            # 保存检查点
            self.save_checkpoint(epoch, val_dice, is_best)
            
            # 早停检查
            if (self.cfg.TRAIN.USE_EARLY_STOPPING and 
                self.patience_counter >= self.cfg.TRAIN.EARLY_STOPPING_PATIENCE):
                logger.info(f"早停触发，在epoch {epoch+1}停止训练")
                break
            
            logger.info(f"Epoch {epoch+1} 完成 - Train Loss: {train_loss:.4f}, "
                       f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
        
        logger.info(f"训练完成！最佳验证Dice: {self.best_val_dice:.4f}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='训练文字水印检测模型')
    parser.add_argument('--config', type=str, 
                       default='src/configs/unet_text_watermark.yaml',
                       help='配置文件路径')
    
    args = parser.parse_args()
    
    # 创建训练器并开始训练
    trainer = TextWatermarkTrainer(args.config)
    trainer.train()

if __name__ == '__main__':
    main()