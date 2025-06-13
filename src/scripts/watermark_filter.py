#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
水印检测过滤脚本
使用训练好的模型检测图片中的水印，如果图片中没有水印，则删除该图片

使用方法:
python watermark_filter.py --input_dir /path/to/images --model_path /path/to/model.pth --config_path /path/to/config.yaml
"""

import os
import cv2
import torch
import numpy as np
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import shutil

# 导入自定义模块
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import get_cfg_defaults, update_config
from models.smp_models import create_model_from_config
from utils.dataset import get_val_transform

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WatermarkFilter:
    """水印过滤器类"""
    
    def __init__(self, model_path, config_path=None, device='cpu', watermark_threshold=0.001):
        """
        初始化水印过滤器
        
        Args:
            model_path (str): 模型文件路径
            config_path (str): 配置文件路径
            device (str): 设备类型 ('cpu' 或 'cuda')
            watermark_threshold (float): 水印面积阈值，低于此值认为没有水印
        """
        self.device = torch.device(device)
        self.watermark_threshold = watermark_threshold
        
        # 加载配置
        self.cfg = get_cfg_defaults()
        if config_path and os.path.exists(config_path):
            update_config(self.cfg, config_path)
        
        # 加载模型
        self.model, self.model_info = self._load_model(model_path)
        
        # 数据变换
        self.transform = get_val_transform(self.cfg.DATA.IMG_SIZE)
        
        logger.info(f"水印过滤器初始化完成")
        logger.info(f"设备: {self.device}")
        logger.info(f"水印阈值: {self.watermark_threshold}")
        logger.info(f"模型: {os.path.basename(model_path)}")
    
    def _load_model(self, model_path):
        """加载模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        logger.info(f"正在加载模型: {model_path}")
        
        # 创建模型
        model = create_model_from_config(self.cfg)
        
        # 加载权重
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 处理不同的保存格式
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            model_info = {
                'epoch': checkpoint.get('epoch', 'Unknown'),
                'val_loss': checkpoint.get('val_loss', 'Unknown'),
                'train_loss': checkpoint.get('train_loss', 'Unknown'),
                'val_metrics': checkpoint.get('val_metrics', {})
            }
        else:
            model.load_state_dict(checkpoint)
            model_info = {}
        
        model.to(self.device)
        model.eval()
        
        logger.info("模型加载完成")
        return model, model_info
    
    def predict_mask(self, image_path):
        """
        预测单张图像的水印掩码
        
        Args:
            image_path (str): 图像路径
            
        Returns:
            np.ndarray: 预测的掩码
        """
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")
            
            original_size = (image.shape[1], image.shape[0])  # (width, height)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 应用变换
            augmented = self.transform(image=image_rgb)
            input_tensor = augmented['image'].unsqueeze(0).to(self.device)
            
            # 模型推理
            with torch.no_grad():
                output = self.model(input_tensor)
                prob = torch.sigmoid(output)
                
                # 立即移动到CPU并清理GPU内存
                mask = prob.cpu().numpy()[0, 0]  # 移除批次和通道维度
                
                # 清理GPU内存
                del input_tensor, output, prob
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            # 调整掩码大小
            mask = cv2.resize(mask, original_size)
            
            # 二值化掩码
            binary_mask = (mask > self.cfg.PREDICT.THRESHOLD).astype(np.uint8) * 255
            
            # 后处理
            if self.cfg.PREDICT.POST_PROCESS:
                binary_mask = self._post_process_mask(binary_mask)
            
            return binary_mask
            
        except Exception as e:
            # 确保在异常情况下也清理GPU内存
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            raise e
    
    def _post_process_mask(self, mask):
        """掩码后处理"""
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        return mask
    
    def has_watermark(self, image_path):
        """
        检测图像是否包含水印
        
        Args:
            image_path (str): 图像路径
            
        Returns:
            tuple: (has_watermark, watermark_ratio)
        """
        try:
            # 预测掩码
            mask = self.predict_mask(image_path)
            
            # 计算水印面积比例
            image = cv2.imread(image_path)
            total_pixels = image.shape[0] * image.shape[1]
            watermark_pixels = np.sum(mask > 0)
            watermark_ratio = watermark_pixels / total_pixels
            
            # 判断是否有水印
            has_watermark = watermark_ratio >= self.watermark_threshold
            
            return has_watermark, watermark_ratio
            
        except Exception as e:
            logger.error(f"检测图像 {image_path} 时出错: {str(e)}")
            return False, 0.0
    
    def filter_images(self, input_dir, backup_dir=None, dry_run=False):
        """
        过滤图像文件夹，删除没有水印的图片
        
        Args:
            input_dir (str): 输入图像目录
            backup_dir (str): 备份目录（可选）
            dry_run (bool): 是否为试运行模式
            
        Returns:
            dict: 处理结果统计
        """
        # 获取所有图片文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(input_dir).glob(f'*{ext}'))
            image_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))
        
        if not image_files:
            logger.warning(f"在 {input_dir} 中未找到图像文件")
            return {'total': 0, 'with_watermark': 0, 'without_watermark': 0, 'deleted': 0, 'errors': 0}
        
        logger.info(f"开始处理 {len(image_files)} 张图片")
        
        # 创建备份目录
        if backup_dir and not dry_run:
            os.makedirs(backup_dir, exist_ok=True)
        
        # 统计信息
        stats = {
            'total': len(image_files),
            'with_watermark': 0,
            'without_watermark': 0,
            'deleted': 0,
            'errors': 0
        }
        
        # 处理每张图片
        progress_bar = tqdm(image_files, desc="检测水印", unit="张")
        
        for image_path in progress_bar:
            try:
                # 检测水印
                has_watermark, watermark_ratio = self.has_watermark(str(image_path))
                
                progress_bar.set_postfix({
                    '当前': image_path.name[:15] + '...',
                    '水印比例': f'{watermark_ratio:.4f}'
                })
                
                if has_watermark:
                    stats['with_watermark'] += 1
                    logger.info(f"保留: {image_path.name} (水印比例: {watermark_ratio:.6f})")
                else:
                    stats['without_watermark'] += 1
                    
                    if dry_run:
                        logger.info(f"[试运行] 将删除: {image_path.name} (水印比例: {watermark_ratio:.6f})")
                    else:
                        # 备份文件（如果指定了备份目录）
                        if backup_dir:
                            backup_path = os.path.join(backup_dir, image_path.name)
                            shutil.copy2(str(image_path), backup_path)
                            logger.info(f"备份到: {backup_path}")
                        
                        # 删除文件
                        os.remove(str(image_path))
                        stats['deleted'] += 1
                        logger.info(f"删除: {image_path.name} (水印比例: {watermark_ratio:.6f})")
                
            except Exception as e:
                stats['errors'] += 1
                logger.error(f"处理 {image_path.name} 时出错: {str(e)}")
        
        return stats

def main():
    parser = argparse.ArgumentParser(description='水印检测过滤脚本')
    parser.add_argument('--input_dir', type=str, default='data/train/watermarked', help='输入图像目录')
    parser.add_argument('--model_path', type=str, default='models/unet_watermark.pth', help='模型文件路径')
    parser.add_argument('--config_path', type=str, help='配置文件路径')
    parser.add_argument('--device', type=str, default='cpu', help='设备类型')
    parser.add_argument('--threshold', type=float, default=0.0001, help='水印面积阈值')
    parser.add_argument('--backup_dir', type=str, help='备份目录（可选）')
    parser.add_argument('--dry_run', action='store_true', help='试运行模式，不实际删除文件')
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.exists(args.input_dir):
        logger.error(f"输入目录不存在: {args.input_dir}")
        return
    
    # 检查模型文件
    if not os.path.exists(args.model_path):
        logger.error(f"模型文件不存在: {args.model_path}")
        return
    
    # 检查配置文件
    if args.config_path and not os.path.exists(args.config_path):
        logger.error(f"配置文件不存在: {args.config_path}")
        return
    
    try:
        # 创建水印过滤器
        filter_obj = WatermarkFilter(
            model_path=args.model_path,
            config_path=args.config_path,
            device=args.device,
            watermark_threshold=args.threshold
        )
        
        # 过滤图像
        stats = filter_obj.filter_images(
            input_dir=args.input_dir,
            backup_dir=args.backup_dir,
            dry_run=args.dry_run
        )
        
        # 输出统计结果
        logger.info("\n" + "=" * 50)
        logger.info("处理完成！统计结果:")
        logger.info(f"总图片数: {stats['total']}")
        logger.info(f"有水印: {stats['with_watermark']}")
        logger.info(f"无水印: {stats['without_watermark']}")
        logger.info(f"已删除: {stats['deleted']}")
        logger.info(f"错误数: {stats['errors']}")
        logger.info("=" * 50)
        
        if args.dry_run:
            logger.info("这是试运行模式，没有实际删除文件")
            logger.info("要实际执行删除，请移除 --dry_run 参数")
        
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        raise

if __name__ == '__main__':
    main()