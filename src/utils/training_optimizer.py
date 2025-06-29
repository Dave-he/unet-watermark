#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练优化器
提供高效的训练循环、学习率调度和性能监控
"""

import os
import time
import torch
import logging
import numpy as np
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
from pathlib import Path

from .enhanced_memory_manager import EnhancedMemoryManager, get_global_memory_manager, memory_optimized
from .adaptive_batch_processor import AdaptiveBatchProcessor
from .optimized_dataloader import OptimizedDataLoader, DataLoaderConfig

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础训练参数
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # 批处理配置
    batch_size: int = 8
    accumulation_steps: int = 1
    adaptive_batch_size: bool = True
    min_batch_size: int = 1
    max_batch_size: int = 32
    
    # 优化器配置
    optimizer_type: str = 'adamw'  # 'adam', 'adamw', 'sgd'
    scheduler_type: str = 'cosine'  # 'cosine', 'step', 'plateau', 'warmup'
    warmup_epochs: int = 5
    
    # 混合精度
    enable_mixed_precision: bool = True
    gradient_clip_norm: float = 1.0
    
    # 内存优化
    memory_threshold: float = 0.8
    cleanup_frequency: int = 10
    enable_gradient_checkpointing: bool = False
    
    # 保存和日志
    save_frequency: int = 10
    log_frequency: int = 1
    checkpoint_dir: str = './checkpoints'
    log_dir: str = './logs'
    
    # 早停
    early_stopping: bool = True
    patience: int = 20
    min_delta: float = 1e-6
    
    # 验证
    validation_frequency: int = 1
    validation_metric: str = 'loss'  # 'loss', 'accuracy', 'iou'
    
    # 数据加载
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    
    # 高级优化
    enable_compile: bool = False  # PyTorch 2.0+
    enable_channels_last: bool = False
    enable_deterministic: bool = False

@dataclass
class TrainingMetrics:
    """训练指标"""
    epoch: int
    step: int
    train_loss: float
    val_loss: Optional[float] = None
    learning_rate: float = 0.0
    batch_size: int = 0
    memory_usage: float = 0.0
    time_per_batch: float = 0.0
    throughput: float = 0.0  # samples/sec
    timestamp: float = field(default_factory=time.time)
    
    # 自定义指标
    custom_metrics: Dict[str, float] = field(default_factory=dict)

class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 20, min_delta: float = 1e-6, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
        self.compare = self._get_compare_fn()
    
    def _get_compare_fn(self):
        if self.mode == 'min':
            return lambda current, best: current < best - self.min_delta
        else:
            return lambda current, best: current > best + self.min_delta
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif self.compare(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def reset(self):
        self.best_score = None
        self.counter = 0
        self.early_stop = False

class LearningRateScheduler:
    """学习率调度器"""
    
    def __init__(self, 
                 optimizer: torch.optim.Optimizer,
                 scheduler_type: str = 'cosine',
                 total_epochs: int = 100,
                 warmup_epochs: int = 5,
                 **kwargs):
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        
        self.scheduler = self._create_scheduler(**kwargs)
        self.warmup_scheduler = self._create_warmup_scheduler() if warmup_epochs > 0 else None
        
        self.current_epoch = 0
    
    def _create_scheduler(self, **kwargs):
        if self.scheduler_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.total_epochs - self.warmup_epochs,
                **kwargs
            )
        elif self.scheduler_type == 'step':
            step_size = kwargs.get('step_size', 30)
            gamma = kwargs.get('gamma', 0.1)
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=step_size, 
                gamma=gamma
            )
        elif self.scheduler_type == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=kwargs.get('factor', 0.5),
                patience=kwargs.get('patience', 10),
                **kwargs
            )
        else:
            return None
    
    def _create_warmup_scheduler(self):
        def warmup_lambda(epoch):
            return (epoch + 1) / self.warmup_epochs
        
        return torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, 
            lr_lambda=warmup_lambda
        )
    
    def step(self, epoch: int, val_loss: Optional[float] = None):
        self.current_epoch = epoch
        
        if epoch < self.warmup_epochs and self.warmup_scheduler:
            self.warmup_scheduler.step()
        else:
            if self.scheduler_type == 'plateau' and val_loss is not None:
                self.scheduler.step(val_loss)
            elif self.scheduler:
                self.scheduler.step()
    
    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']

class TrainingOptimizer:
    """训练优化器"""
    
    def __init__(self, 
                 model: torch.nn.Module,
                 train_loader: torch.utils.data.DataLoader,
                 val_loader: Optional[torch.utils.data.DataLoader] = None,
                 loss_fn: Optional[Callable] = None,
                 config: Optional[TrainingConfig] = None,
                 device: Optional[torch.device] = None):
        """
        初始化训练优化器
        
        Args:
            model: 训练模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            loss_fn: 损失函数
            config: 训练配置
            device: 计算设备
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn or torch.nn.MSELoss()
        self.config = config or TrainingConfig()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 内存管理器
        self.memory_manager = get_global_memory_manager()
        
        # 自适应批处理器
        if self.config.adaptive_batch_size:
            self.batch_processor = AdaptiveBatchProcessor(
                initial_batch_size=self.config.batch_size,
                min_batch_size=self.config.min_batch_size,
                max_batch_size=self.config.max_batch_size,
                memory_threshold=self.config.memory_threshold
            )
        else:
            self.batch_processor = None
        
        # 模型优化
        self._optimize_model()
        
        # 优化器和调度器
        self.optimizer = self._create_optimizer()
        self.scheduler = LearningRateScheduler(
            self.optimizer,
            scheduler_type=self.config.scheduler_type,
            total_epochs=self.config.epochs,
            warmup_epochs=self.config.warmup_epochs
        )
        
        # 混合精度
        if self.config.enable_mixed_precision and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # 早停
        if self.config.early_stopping:
            self.early_stopping = EarlyStopping(
                patience=self.config.patience,
                min_delta=self.config.min_delta
            )
        else:
            self.early_stopping = None
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # 指标历史
        self.metrics_history: List[TrainingMetrics] = []
        self.loss_history = deque(maxlen=100)
        
        # 创建目录
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)
        
        logger.info(f"训练优化器初始化完成: device={self.device}, "
                   f"mixed_precision={self.config.enable_mixed_precision}")
    
    def _optimize_model(self):
        """优化模型"""
        # 移动到设备
        self.model = self.model.to(self.device)
        
        # 梯度检查点
        if self.config.enable_gradient_checkpointing:
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                logger.info("启用梯度检查点")
        
        # Channels Last 内存格式
        if self.config.enable_channels_last and torch.cuda.is_available():
            self.model = self.model.to(memory_format=torch.channels_last)
            logger.info("启用 Channels Last 内存格式")
        
        # PyTorch 编译
        if self.config.enable_compile and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model)
                logger.info("启用 PyTorch 编译优化")
            except Exception as e:
                logger.warning(f"PyTorch 编译失败: {e}")
        
        # 确定性训练
        if self.config.enable_deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logger.info("启用确定性训练")
        else:
            torch.backends.cudnn.benchmark = True
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """创建优化器"""
        if self.config.optimizer_type.lower() == 'adam':
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type.lower() == 'adamw':
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type.lower() == 'sgd':
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"不支持的优化器类型: {self.config.optimizer_type}")
    
    @memory_optimized
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        epoch_metrics = {
            'train_loss': 0.0,
            'batch_count': 0,
            'sample_count': 0,
            'time': 0.0
        }
        
        epoch_start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            batch_start_time = time.time()
            
            # 处理批次数据
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0], batch[1]
            else:
                inputs, targets = batch, batch  # 自监督学习
            
            # 移动到设备
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            # Channels Last 格式
            if self.config.enable_channels_last:
                inputs = inputs.to(memory_format=torch.channels_last)
                if targets.dim() == 4:
                    targets = targets.to(memory_format=torch.channels_last)
            
            # 前向传播
            if self.scaler and self.config.enable_mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, targets)
                    loss = loss / self.config.accumulation_steps
            else:
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss = loss / self.config.accumulation_steps
            
            # 反向传播
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 梯度累积
            if (batch_idx + 1) % self.config.accumulation_steps == 0:
                # 梯度裁剪
                if self.config.gradient_clip_norm > 0:
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.gradient_clip_norm
                    )
                
                # 优化器步骤
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # 更新指标
            batch_loss = loss.item() * self.config.accumulation_steps
            epoch_metrics['train_loss'] += batch_loss
            epoch_metrics['batch_count'] += 1
            epoch_metrics['sample_count'] += inputs.size(0)
            
            batch_time = time.time() - batch_start_time
            
            # 记录批次指标
            if batch_idx % self.config.log_frequency == 0:
                current_lr = self.scheduler.get_lr()
                memory_usage = self.memory_manager.get_memory_snapshot().gpu_allocated_gb
                throughput = inputs.size(0) / batch_time
                
                batch_metrics = TrainingMetrics(
                    epoch=self.current_epoch,
                    step=self.global_step,
                    train_loss=batch_loss,
                    learning_rate=current_lr,
                    batch_size=inputs.size(0),
                    memory_usage=memory_usage,
                    time_per_batch=batch_time,
                    throughput=throughput
                )
                
                self.metrics_history.append(batch_metrics)
                self.loss_history.append(batch_loss)
                
                logger.debug(f"Epoch {self.current_epoch}, Batch {batch_idx}: "
                           f"loss={batch_loss:.6f}, lr={current_lr:.2e}, "
                           f"memory={memory_usage:.2f}GB, throughput={throughput:.1f} samples/s")
            
            # 定期清理内存
            if batch_idx % self.config.cleanup_frequency == 0:
                self.memory_manager.auto_cleanup()
        
        # 计算epoch平均指标
        epoch_time = time.time() - epoch_start_time
        epoch_metrics['time'] = epoch_time
        
        if epoch_metrics['batch_count'] > 0:
            epoch_metrics['train_loss'] /= epoch_metrics['batch_count']
            epoch_metrics['avg_throughput'] = epoch_metrics['sample_count'] / epoch_time
        
        return epoch_metrics
    
    @memory_optimized
    def validate(self) -> Dict[str, float]:
        """验证模型"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        val_metrics = {
            'val_loss': 0.0,
            'batch_count': 0,
            'sample_count': 0
        }
        
        with torch.no_grad():
            for batch in self.val_loader:
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0], batch[1]
                else:
                    inputs, targets = batch, batch
                
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                if self.config.enable_channels_last:
                    inputs = inputs.to(memory_format=torch.channels_last)
                    if targets.dim() == 4:
                        targets = targets.to(memory_format=torch.channels_last)
                
                # 前向传播
                if self.scaler and self.config.enable_mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                        loss = self.loss_fn(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, targets)
                
                val_metrics['val_loss'] += loss.item()
                val_metrics['batch_count'] += 1
                val_metrics['sample_count'] += inputs.size(0)
        
        # 计算平均指标
        if val_metrics['batch_count'] > 0:
            val_metrics['val_loss'] /= val_metrics['batch_count']
        
        return val_metrics
    
    def train(self, 
             progress_callback: Optional[Callable] = None,
             custom_metrics_fn: Optional[Callable] = None) -> Dict[str, Any]:
        """完整训练循环"""
        logger.info(f"开始训练: {self.config.epochs} epochs")
        
        training_start_time = time.time()
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # 训练一个epoch
            train_metrics = self.train_epoch()
            
            # 验证
            val_metrics = {}
            if epoch % self.config.validation_frequency == 0:
                val_metrics = self.validate()
            
            # 学习率调度
            val_loss = val_metrics.get('val_loss')
            self.scheduler.step(epoch, val_loss)
            
            # 合并指标
            epoch_metrics = {**train_metrics, **val_metrics}
            epoch_metrics['epoch'] = epoch
            epoch_metrics['learning_rate'] = self.scheduler.get_lr()
            epoch_metrics['epoch_time'] = time.time() - epoch_start_time
            
            # 自定义指标
            if custom_metrics_fn:
                custom_metrics = custom_metrics_fn(self.model, epoch_metrics)
                epoch_metrics.update(custom_metrics)
            
            # 记录epoch指标
            epoch_summary = TrainingMetrics(
                epoch=epoch,
                step=self.global_step,
                train_loss=epoch_metrics['train_loss'],
                val_loss=val_loss,
                learning_rate=epoch_metrics['learning_rate'],
                custom_metrics=epoch_metrics.get('custom_metrics', {})
            )
            self.metrics_history.append(epoch_summary)
            
            # 日志输出
            log_msg = f"Epoch {epoch + 1}/{self.config.epochs}: "
            log_msg += f"train_loss={epoch_metrics['train_loss']:.6f}"
            if val_loss is not None:
                log_msg += f", val_loss={val_loss:.6f}"
            log_msg += f", lr={epoch_metrics['learning_rate']:.2e}"
            log_msg += f", time={epoch_metrics['epoch_time']:.2f}s"
            
            logger.info(log_msg)
            
            # 保存检查点
            if epoch % self.config.save_frequency == 0:
                self.save_checkpoint(epoch, epoch_metrics)
            
            # 早停检查
            if self.early_stopping and val_loss is not None:
                if self.early_stopping(val_loss):
                    logger.info(f"早停触发，在epoch {epoch + 1}")
                    break
            
            # 更新最佳验证损失
            if val_loss is not None and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, epoch_metrics, is_best=True)
            
            # 进度回调
            if progress_callback:
                progress_callback(epoch, self.config.epochs, epoch_metrics)
        
        total_training_time = time.time() - training_start_time
        
        # 训练总结
        training_summary = {
            'total_epochs': self.current_epoch + 1,
            'total_time': total_training_time,
            'best_val_loss': self.best_val_loss,
            'final_lr': self.scheduler.get_lr(),
            'total_steps': self.global_step
        }
        
        logger.info(f"训练完成: {training_summary}")
        
        # 保存训练历史
        self.save_training_history()
        
        return training_summary
    
    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.scheduler.state_dict() if self.scheduler.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'metrics': metrics,
            'config': self.config,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss
        }
        
        # 保存最新检查点
        checkpoint_path = os.path.join(self.config.checkpoint_dir, 'latest.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳检查点
        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, 'best.pt')
            torch.save(checkpoint, best_path)
            logger.info(f"保存最佳模型: {best_path}")
        
        # 保存epoch检查点
        if epoch % (self.config.save_frequency * 5) == 0:
            epoch_path = os.path.join(self.config.checkpoint_dir, f'epoch_{epoch}.pt')
            torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 加载模型
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载优化器
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 加载调度器
        if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            if self.scheduler.scheduler:
                self.scheduler.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 加载混合精度缩放器
        if self.scaler and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # 恢复训练状态
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        logger.info(f"检查点加载完成: epoch={self.current_epoch}, step={self.global_step}")
    
    def save_training_history(self):
        """保存训练历史"""
        history_path = os.path.join(self.config.log_dir, 'training_history.json')
        
        # 转换为可序列化格式
        history_data = []
        for metrics in self.metrics_history:
            if isinstance(metrics, TrainingMetrics):
                data = {
                    'epoch': metrics.epoch,
                    'step': metrics.step,
                    'train_loss': metrics.train_loss,
                    'val_loss': metrics.val_loss,
                    'learning_rate': metrics.learning_rate,
                    'batch_size': metrics.batch_size,
                    'memory_usage': metrics.memory_usage,
                    'time_per_batch': metrics.time_per_batch,
                    'throughput': metrics.throughput,
                    'timestamp': metrics.timestamp,
                    'custom_metrics': metrics.custom_metrics
                }
                history_data.append(data)
        
        with open(history_path, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        logger.info(f"训练历史已保存: {history_path}")
    
    def get_training_stats(self) -> Dict:
        """获取训练统计"""
        if not self.metrics_history:
            return {}
        
        # 计算统计信息
        train_losses = [m.train_loss for m in self.metrics_history if hasattr(m, 'train_loss') and m.train_loss is not None]
        val_losses = [m.val_loss for m in self.metrics_history if hasattr(m, 'val_loss') and m.val_loss is not None]
        
        stats = {
            'total_epochs': self.current_epoch + 1,
            'total_steps': self.global_step,
            'current_lr': self.scheduler.get_lr(),
            'best_val_loss': self.best_val_loss,
            'avg_train_loss': np.mean(train_losses) if train_losses else 0,
            'min_train_loss': np.min(train_losses) if train_losses else 0,
            'avg_val_loss': np.mean(val_losses) if val_losses else 0,
            'min_val_loss': np.min(val_losses) if val_losses else 0
        }
        
        # 添加内存统计
        memory_stats = self.memory_manager.get_stats()
        stats['memory'] = memory_stats
        
        # 添加批处理统计
        if self.batch_processor:
            batch_stats = self.batch_processor.get_performance_stats()
            stats['batch_processor'] = batch_stats
        
        return stats
    
    def cleanup(self):
        """清理资源"""
        # 清理内存
        self.memory_manager.aggressive_cleanup()
        
        # 保存最终状态
        if self.current_epoch > 0:
            self.save_checkpoint(self.current_epoch, {'final': True})
            self.save_training_history()
        
        logger.info("训练优化器资源清理完成")

if __name__ == "__main__":
    # 测试代码
    from torchvision import models
    import torch.nn as nn
    
    # 创建测试模型
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    
    # 创建虚拟数据
    train_data = [(torch.randn(3, 224, 224), torch.randn(1)) for _ in range(100)]
    val_data = [(torch.randn(3, 224, 224), torch.randn(1)) for _ in range(20)]
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=8)
    
    # 创建配置
    config = TrainingConfig(
        epochs=5,
        learning_rate=1e-3,
        batch_size=8,
        enable_mixed_precision=True,
        save_frequency=2
    )
    
    # 创建训练优化器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = TrainingOptimizer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=nn.MSELoss(),
        config=config,
        device=device
    )
    
    # 开始训练
    training_summary = trainer.train()
    print(f"训练完成: {training_summary}")
    
    # 获取统计信息
    stats = trainer.get_training_stats()
    print(f"训练统计: {stats}")
    
    # 清理
    trainer.cleanup()
    
    print("测试完成")