import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import logging
from datetime import datetime

# 导入自定义模块
from configs.config import get_cfg_defaults, update_config
from models.smp_models import create_model_from_config
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

def train_epoch(model, train_loader, criterion, optimizer, device, metrics, cfg):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    metric_values = {'iou': 0.0, 'f1': 0.0, 'accuracy': 0.0, 'recall': 0.0, 'precision': 0.0}
    
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")
    
    for i, (images, masks) in progress_bar:
        images = images.to(device)
        masks = masks.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        # 在 train_epoch 函数中
        outputs = model(images)
        if outputs.dim() == 4 and outputs.size(1) == 1:
            outputs = outputs.squeeze(1)  # (N, 1, H, W) -> (N, H, W)
        loss = criterion(outputs, masks)  # 现在都是 (N, H, W)
        
        # 在 validate 函数中 - 移除错误的双重调整
        outputs = model(images)
        if outputs.dim() == 4 and outputs.size(1) == 1:
            outputs = outputs.squeeze(1)  # 只调整输出
        # 移除: if masks.dim() == 3: masks = masks.unsqueeze(1)
        loss = criterion(outputs, masks)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 计算指标
        with torch.no_grad():
            preds = torch.sigmoid(outputs)
            # 确保 preds 和 masks 形状一致
            batch_metrics = metrics(preds, masks)
            for name, value in batch_metrics.items():
                metric_values[name] += value
        
        total_loss += loss.item()
        
        # 更新进度条
        if (i + 1) % cfg.TRAIN.LOG_INTERVAL == 0:
            avg_loss = total_loss / (i + 1)
            avg_metrics = {name: val / (i + 1) for name, val in metric_values.items()}
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'iou': f'{avg_metrics["iou"]:.4f}',
                'f1': f'{avg_metrics["f1"]:.4f}'
            })
    
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
            
            outputs = model(images)
            # 在计算损失前添加
            if outputs.dim() == 4 and outputs.size(1) == 1:
                outputs = outputs.squeeze(1)  # 从 (N, 1, H, W) 变为 (N, H, W)
            # 或者调整目标张量
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)  # 从 (N, H, W) 变为 (N, 1, H, W)
            loss = criterion(outputs, masks)
            
            # 计算指标
            preds = torch.sigmoid(outputs)
            batch_metrics = metrics(preds, masks)
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

def train(cfg):
    """训练函数"""
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
    train_dataset, val_dataset = create_datasets(cfg)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.TRAIN.BATCH_SIZE, 
        shuffle=True, 
        num_workers=cfg.DATA.NUM_WORKERS,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg.TRAIN.BATCH_SIZE, 
        shuffle=False, 
        num_workers=cfg.DATA.NUM_WORKERS,
        pin_memory=True if device.type == 'cuda' else False
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
    
    # 早停机制
    early_stopping = EarlyStopping(
        patience=cfg.TRAIN.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True
    )
    
    # 训练历史记录
    train_losses = []
    val_losses = []
    train_metrics_history = []
    val_metrics_history = []
    
    best_val_loss = float('inf')
    
    logger.info(f"开始训练 {cfg.MODEL.NAME} 模型...")
    logger.info(f"训练集: {len(train_dataset)} 张图像")
    logger.info(f"验证集: {len(val_dataset)} 张图像")
    logger.info(f"训练轮数: {cfg.TRAIN.EPOCHS}")
    
    for epoch in range(cfg.TRAIN.EPOCHS):
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
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_info = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_loss': val_loss,
                'val_metrics': val_metrics_epoch,
                'train_loss': train_loss,
                'train_metrics': train_metrics_epoch,
                'config': cfg,
                'best_val_loss': best_val_loss
            }
            torch.save(best_model_info, cfg.TRAIN.MODEL_SAVE_PATH)
            logger.info(f"保存最佳模型: {cfg.TRAIN.MODEL_SAVE_PATH} (Val Loss: {val_loss:.4f})")
        
        # 定期保存检查点（每隔50轮）
        if (epoch + 1) % cfg.TRAIN.SAVE_INTERVAL == 0:
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
                'best_val_loss': best_val_loss
            }
            torch.save(checkpoint_info, checkpoint_path)
            logger.info(f"保存检查点: {checkpoint_path} (Epoch: {epoch+1})")
        
        # 早停检查
        if early_stopping(val_loss, model):
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
    
    # 开始训练
    train(cfg)

if __name__ == "__main__":
    main()