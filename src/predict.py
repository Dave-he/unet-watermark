# -*- coding: utf-8 -*-
"""
预测模块 - 批量分步处理版本
提供4步水印检测和修复功能：
1. 对文件夹内所有图片使用UNet模型预测水印mask区域
2. 对文件夹内所有图片调用IOPaint修复水印区域
3. 对文件夹内所有图片使用OCR提取文字mask区域
4. 对文件夹内所有图片调用IOPaint修复文字区域
"""

import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import subprocess
import tempfile
import logging
from pathlib import Path
import shutil
import random
import time
import gc
from typing import Tuple, Optional, Dict, Any, List
import glob

# 导入自定义模块
from configs.config import get_cfg_defaults, update_config
from models.unet_model import create_model_from_config
from utils.dataset import get_val_transform

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WatermarkPredictor:
    """
    批量4步水印移除器
    Step 1: 对所有图片使用UNet模型预测水印mask
    Step 2: 对所有图片使用IOPaint修复水印区域
    Step 3: 对所有图片使用OCR提取文字mask
    Step 4: 对所有图片使用IOPaint修复文字区域
    """
    
    def __init__(self, model_path, config_path=None, config=None, device='cpu'):
        """初始化批量水印移除器"""
        self.device = torch.device(device)
        
        # 加载配置
        if config is not None:
            self.cfg = config
        else:
            self.cfg = get_cfg_defaults()
            if config_path and os.path.exists(config_path):
                update_config(self.cfg, config_path)
        
        # 加载UNet模型
        self.model, self.model_info = self._load_unet_model(model_path)
        
        # 数据变换
        self.transform = get_val_transform(self.cfg.DATA.IMG_SIZE)
        
        logger.info(f"批量水印移除器初始化完成，使用设备: {self.device}")
        self._print_model_info()
    
    def _load_unet_model(self, model_path):
        """加载UNet模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        try:
            # 创建模型
            model = create_model_from_config(self.cfg).to(self.device)
            
            # 加载权重
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"从检查点加载UNet模型: {model_path}")
                model_info = {
                    'epoch': checkpoint.get('epoch', 'Unknown'),
                    'val_loss': checkpoint.get('val_loss', 'Unknown'),
                    'val_metrics': checkpoint.get('val_metrics', {})
                }
            else:
                model.load_state_dict(checkpoint)
                logger.info(f"加载UNet模型权重: {model_path}")
                model_info = {'epoch': 'Unknown', 'val_loss': 'Unknown'}
            
            model.eval()
            return model, model_info
            
        except Exception as e:
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            raise e
    
    def _print_model_info(self):
        """打印模型信息"""
        logger.info("=" * 50)
        logger.info("UNet模型信息:")
        logger.info(f"训练轮数: {self.model_info.get('epoch', 'Unknown')}")
        logger.info(f"验证损失: {self.model_info.get('val_loss', 'Unknown')}")
        
        val_metrics = self.model_info.get('val_metrics', {})
        if val_metrics:
            logger.info(f"验证IoU: {val_metrics.get('iou', 'Unknown')}")
            logger.info(f"验证F1: {val_metrics.get('f1', 'Unknown')}")
        logger.info("=" * 50)
    
    def _get_image_files(self, input_folder, limit=None):
        """获取文件夹中的所有图像文件
        
        Args:
            input_folder: 输入文件夹路径
            limit: 限制处理的图片数量，如果为None则处理所有图片
        
        Returns:
            list: 图像文件路径列表
        """
        image_files = []
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
        
        for ext in extensions:
            # 添加小写和大写扩展名
            image_files.extend(glob.glob(os.path.join(input_folder, ext)))
            image_files.extend(glob.glob(os.path.join(input_folder, ext.upper())))
        
        image_files = sorted(list(set(image_files)))  # 去重并排序
        
        # 如果指定了limit，随机选择指定数量的图片
        if limit is not None and limit > 0 and len(image_files) > limit:
            total_count = len(image_files)
            random.shuffle(image_files)
            image_files = image_files[:limit]
            logger.info(f"随机选择了 {limit} 张图片进行处理（总共 {total_count} 张）")
        
        return image_files
    
    def _optimize_mask(self, mask):
        """优化预测的mask，使其稍微大一些，保证连通性，抑制噪声"""
        if mask is None:
            return mask
            
        # 确保mask是单通道的
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        # 二值化确保mask只有0和255
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # 形态学操作去噪和连接断开的区域
        # 使用小核进行开运算去噪
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
        
        # 使用中等核进行闭运算连接断开的区域
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_medium, iterations=3)
        
        # 使用更大的核进行强闭运算，确保连通性
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)
        
        # 膨胀操作扩大mask区域
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.dilate(mask, kernel_dilate, iterations=2)
        
        # 连通组件分析，保留最大的连通域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        if num_labels > 1:  # 有连通组件（除了背景）
            # 找到最大的连通组件（排除背景标签0）
            largest_component_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            
            # 创建只包含最大连通组件的mask
            mask = (labels == largest_component_label).astype(np.uint8) * 255
            
            # 如果最大连通组件太小，则保留所有较大的组件
            max_area = stats[largest_component_label, cv2.CC_STAT_AREA]
            if max_area < 500:  # 如果最大组件面积小于阈值
                mask = np.zeros_like(labels, dtype=np.uint8)
                for i in range(1, num_labels):
                    if stats[i, cv2.CC_STAT_AREA] > 200:  # 保留面积大于200的组件
                        mask[labels == i] = 255
        
        # 最后的平滑处理
        mask = cv2.GaussianBlur(mask, (3, 3), 0.5)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        return mask
    
    def step1_batch_predict_watermark_masks(self, input_folder, mask_output_folder, limit=None):
        """步骤1: 批量预测所有图片的水印mask
        
        Args:
            input_folder: 输入文件夹路径
            mask_output_folder: mask输出文件夹路径
            limit: 限制处理的图片数量，如果为None则处理所有图片
        """
        logger.info("=" * 60)
        logger.info("步骤1: 开始批量预测水印mask")
        logger.info("=" * 60)
        
        # 创建mask输出目录
        os.makedirs(mask_output_folder, exist_ok=True)
        
        # 获取所有图像文件
        image_files = self._get_image_files(input_folder, limit=limit)
        if not image_files:
            logger.warning(f"在 {input_folder} 中未找到图像文件")
            return []
        
        logger.info(f"找到 {len(image_files)} 张图片")
        
        processed_files = []
        
        # 批量处理
        progress_bar = tqdm(image_files, desc="预测水印mask", unit="张")
        
        for image_path in progress_bar:
            try:
                # 加载图像
                image = cv2.imread(image_path)
                if image is None:
                    logger.error(f"无法加载图像: {image_path}")
                    continue
                
                # 转换为RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # 应用变换
                if self.transform:
                    transformed = self.transform(image=image_rgb)
                    input_tensor = transformed['image'].unsqueeze(0).to(self.device)
                else:
                    # 手动处理
                    image_resized = cv2.resize(image_rgb, (self.cfg.DATA.IMG_SIZE, self.cfg.DATA.IMG_SIZE))
                    input_tensor = torch.from_numpy(image_resized.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device)
                    input_tensor = input_tensor / 255.0
                
                # 模型推理
                with torch.no_grad():
                    output = self.model(input_tensor)
                    
                    # 移动到CPU
                    if isinstance(output, dict):
                        mask = output['out'].cpu().numpy()[0, 0]
                    else:
                        mask = output.cpu().numpy()[0, 0]
                
                # 调整大小到原图尺寸
                original_height, original_width = image.shape[:2]
                mask_resized = cv2.resize(mask, (original_width, original_height))
                
                # 二值化
                threshold = getattr(self.cfg.PREDICT, 'THRESHOLD', 0.5)
                mask_binary = (mask_resized > threshold).astype(np.uint8) * 255
                
                # 后处理优化mask
                mask_optimized = self._optimize_mask(mask_binary)
                
                # 保存mask
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                mask_path = os.path.join(mask_output_folder, f"{base_name}_mask.png")
                cv2.imwrite(mask_path, mask_optimized)
                
                # 计算水印面积比例
                total_pixels = original_height * original_width
                watermark_pixels = np.sum(mask_optimized > 0)
                watermark_ratio = watermark_pixels / total_pixels
                
                processed_files.append({
                    'image_path': image_path,
                    'mask_path': mask_path,
                    'watermark_ratio': watermark_ratio
                })
                
                progress_bar.set_postfix({'已处理': len(processed_files)})
                
            except Exception as e:
                logger.error(f"处理图像失败 {image_path}: {str(e)}")
                continue
        
        logger.info(f"步骤1完成: 成功处理 {len(processed_files)} 张图片")
        return processed_files
    
    def _batch_iopaint_repair(self, processed_files, output_folder, mask_key, output_suffix, model_name='lama', timeout=300, skip_condition=None, skip_threshold=None):
        """通用的IOPaint批量修复函数
        
        Args:
            processed_files: 待处理的文件列表
            output_folder: 输出文件夹
            mask_key: mask路径在file_info中的键名
            output_suffix: 输出文件后缀
            model_name: IOPaint模型名称
            timeout: 超时时间
            skip_condition: 跳过条件的键名（如'watermark_ratio'或'text_pixels'）
            skip_threshold: 跳过阈值
        """
        # 创建输出目录
        os.makedirs(output_folder, exist_ok=True)
        
        successful_files = []
        
        # 分离需要处理和跳过的文件
        files_to_process = []
        files_to_skip = []
        
        for file_info in processed_files:
            should_skip = False
            if skip_condition and skip_threshold is not None:
                if skip_condition == 'watermark_ratio' and file_info.get('watermark_ratio', 1.0) < skip_threshold:
                    should_skip = True
                elif skip_condition == 'text_pixels' and file_info.get('text_pixels', 1) == 0:
                    should_skip = True
            
            if should_skip:
                files_to_skip.append(file_info)
            else:
                files_to_process.append(file_info)
        
        # 处理跳过的文件：直接复制
        for file_info in files_to_skip:
            base_name = os.path.splitext(os.path.basename(file_info.get('original_path', file_info['image_path'])))[0]
            output_path = os.path.join(output_folder, f"{base_name}_{output_suffix}.png")
            shutil.copy2(file_info['image_path'], output_path)
            
            result_info = {
                'image_path': output_path,
                'original_path': file_info.get('original_path', file_info['image_path']),
                'watermark_ratio': file_info.get('watermark_ratio', 0)
            }
            if 'text_pixels' in file_info:
                result_info['text_pixels'] = file_info['text_pixels']
            
            successful_files.append(result_info)
            
            if skip_condition == 'watermark_ratio':
                logger.info(f"水印面积过小({file_info.get('watermark_ratio', 0):.6f})，跳过修复: {base_name}")
            elif skip_condition == 'text_pixels':
                logger.info(f"无文字区域，跳过修复: {base_name}")
        
        # 如果没有需要处理的文件，直接返回
        if not files_to_process:
            logger.info(f"所有文件都被跳过，无需IOPaint处理")
            return successful_files
        
        # 创建临时目录进行批量处理
        temp_input_dir = None
        temp_mask_dir = None
        temp_output_dir = None
        
        try:
            # 创建临时目录
            temp_input_dir = tempfile.mkdtemp(prefix='iopaint_input_')
            temp_mask_dir = tempfile.mkdtemp(prefix='iopaint_mask_')
            temp_output_dir = tempfile.mkdtemp(prefix='iopaint_output_')
            
            # 复制文件到临时目录
            file_mapping = {}  # 用于映射临时文件名到原始信息
            
            for i, file_info in enumerate(files_to_process):
                # 生成临时文件名
                temp_name = f"img_{i:06d}.png"
                
                # 复制图像文件
                temp_image_path = os.path.join(temp_input_dir, temp_name)
                shutil.copy2(file_info['image_path'], temp_image_path)
                
                # 复制mask文件
                temp_mask_path = os.path.join(temp_mask_dir, temp_name)
                shutil.copy2(file_info[mask_key], temp_mask_path)
                
                # 记录映射关系
                file_mapping[temp_name] = file_info
            
            logger.info(f"开始批量IOPaint处理 {len(files_to_process)} 个文件...")
            
            # 执行批量IOPaint命令
            cmd = [
                'iopaint', 'run',
                '--model', model_name,
                '--device', self.device.type,
                '--image', temp_input_dir,
                '--mask', temp_mask_dir,
                '--output', temp_output_dir
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=timeout)
            logger.info(f"IOPaint批量处理完成")
            
            # 处理输出文件
            for temp_name, file_info in file_mapping.items():
                temp_output_path = os.path.join(temp_output_dir, temp_name)
                
                if os.path.exists(temp_output_path):
                    # 复制到最终输出目录
                    base_name = os.path.splitext(os.path.basename(file_info.get('original_path', file_info['image_path'])))[0]
                    final_output_path = os.path.join(output_folder, f"{base_name}_{output_suffix}.png")
                    shutil.copy2(temp_output_path, final_output_path)
                    
                    result_info = {
                        'image_path': final_output_path,
                        'original_path': file_info.get('original_path', file_info['image_path']),
                        'watermark_ratio': file_info.get('watermark_ratio', 0)
                    }
                    if 'text_pixels' in file_info:
                        result_info['text_pixels'] = file_info['text_pixels']
                    
                    successful_files.append(result_info)
                else:
                    # 如果IOPaint没有生成输出，复制原图作为fallback
                    base_name = os.path.splitext(os.path.basename(file_info.get('original_path', file_info['image_path'])))[0]
                    fallback_output_path = os.path.join(output_folder, f"{base_name}_{output_suffix}.png")
                    shutil.copy2(file_info['image_path'], fallback_output_path)
                    
                    result_info = {
                        'image_path': fallback_output_path,
                        'original_path': file_info.get('original_path', file_info['image_path']),
                        'watermark_ratio': file_info.get('watermark_ratio', 0)
                    }
                    if 'text_pixels' in file_info:
                        result_info['text_pixels'] = file_info['text_pixels']
                    
                    successful_files.append(result_info)
                    logger.error(f"IOPaint未生成输出文件，使用原图: {base_name}")
        
        except subprocess.TimeoutExpired:
            logger.error(f"IOPaint批量处理超时，使用原图作为fallback")
            # 超时时复制所有原图作为fallback
            for file_info in files_to_process:
                base_name = os.path.splitext(os.path.basename(file_info.get('original_path', file_info['image_path'])))[0]
                fallback_output_path = os.path.join(output_folder, f"{base_name}_{output_suffix}.png")
                shutil.copy2(file_info['image_path'], fallback_output_path)
                
                result_info = {
                    'image_path': fallback_output_path,
                    'original_path': file_info.get('original_path', file_info['image_path']),
                    'watermark_ratio': file_info.get('watermark_ratio', 0)
                }
                if 'text_pixels' in file_info:
                    result_info['text_pixels'] = file_info['text_pixels']
                
                successful_files.append(result_info)
        
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else e.stdout if e.stdout else "未知错误"
            logger.error(f"IOPaint批量处理错误: {error_msg}，使用原图作为fallback")
            # 处理错误时复制所有原图作为fallback
            for file_info in files_to_process:
                base_name = os.path.splitext(os.path.basename(file_info.get('original_path', file_info['image_path'])))[0]
                fallback_output_path = os.path.join(output_folder, f"{base_name}_{output_suffix}.png")
                shutil.copy2(file_info['image_path'], fallback_output_path)
                
                result_info = {
                    'image_path': fallback_output_path,
                    'original_path': file_info.get('original_path', file_info['image_path']),
                    'watermark_ratio': file_info.get('watermark_ratio', 0)
                }
                if 'text_pixels' in file_info:
                    result_info['text_pixels'] = file_info['text_pixels']
                
                successful_files.append(result_info)
        
        except Exception as e:
            logger.error(f"IOPaint批量处理异常: {str(e)}，使用原图作为fallback")
            # 异常时复制所有原图作为fallback
            for file_info in files_to_process:
                base_name = os.path.splitext(os.path.basename(file_info.get('original_path', file_info['image_path'])))[0]
                fallback_output_path = os.path.join(output_folder, f"{base_name}_{output_suffix}.png")
                shutil.copy2(file_info['image_path'], fallback_output_path)
                
                result_info = {
                    'image_path': fallback_output_path,
                    'original_path': file_info.get('original_path', file_info['image_path']),
                    'watermark_ratio': file_info.get('watermark_ratio', 0)
                }
                if 'text_pixels' in file_info:
                    result_info['text_pixels'] = file_info['text_pixels']
                
                successful_files.append(result_info)
        
        finally:
            # 清理临时目录
            for temp_dir in [temp_input_dir, temp_mask_dir, temp_output_dir]:
                if temp_dir and os.path.exists(temp_dir):
                    try:
                        shutil.rmtree(temp_dir)
                    except Exception as e:
                        logger.warning(f"清理临时目录失败 {temp_dir}: {str(e)}")
        
        return successful_files
    
    def step2_batch_iopaint_watermark_repair(self, processed_files, step2_output_folder, model_name='lama', timeout=300):
        """步骤2: 批量修复所有图片的水印区域"""
        logger.info("=" * 60)
        logger.info("步骤2: 开始批量修复水印区域")
        logger.info("=" * 60)
        
        successful_files = self._batch_iopaint_repair(
            processed_files=processed_files,
            output_folder=step2_output_folder,
            mask_key='mask_path',
            output_suffix='step2',
            model_name=model_name,
            timeout=timeout,
            skip_condition='watermark_ratio',
            skip_threshold=0.001
        )
        
        logger.info(f"步骤2完成: 成功处理 {len(successful_files)} 张图片")
        return successful_files
    
    def step3_batch_extract_text_masks(self, processed_files, text_mask_output_folder, ocr_languages=None, ocr_engine='easy'):
        """步骤3: 批量提取所有图片的文字mask"""
        logger.info("=" * 60)
        logger.info("步骤3: 开始批量提取文字mask")
        logger.info("=" * 60)
        
        # 创建文字mask输出目录
        os.makedirs(text_mask_output_folder, exist_ok=True)
        
        try:
            if ocr_engine.lower() == 'paddle':
                from ocr.paddle_ocr import PaddleOCRDetector
                ocr_detector = PaddleOCRDetector()
            else:
                from ocr.easy_ocr import EasyOCRDetector
                ocr_detector = EasyOCRDetector()
        except ImportError as e:
            logger.error(f"OCR模块导入错误: {str(e)}")
            return []
        
        successful_files = []
        
        progress_bar = tqdm(processed_files, desc="提取文字mask", unit="张")
        
        for file_info in progress_bar:
            try:
                image_path = file_info['image_path']
                
                # 加载图像
                image = cv2.imread(image_path)
                if image is None:
                    logger.error(f"无法加载图像: {image_path}")
                    continue
                
                # OCR检测文字区域
                if ocr_languages:
                    text_regions = ocr_detector.detect_text_regions(image_path, languages=ocr_languages)
                else:
                    text_regions = ocr_detector.detect_text_regions(image_path)
                
                # 创建文字mask
                height, width = image.shape[:2]
                text_mask = np.zeros((height, width), dtype=np.uint8)
                
                if text_regions:
                    for region in text_regions:
                        # 获取文字区域的边界框
                        if 'bbox' in region:
                            bbox = region['bbox']
                            # 根据不同OCR引擎的bbox格式处理
                            if len(bbox) == 4:  # [x, y, w, h]
                                x, y, w, h = bbox
                                cv2.rectangle(text_mask, (int(x), int(y)), (int(x+w), int(y+h)), 255, -1)
                            elif len(bbox) == 8:  # [x1, y1, x2, y2, x3, y3, x4, y4]
                                points = np.array(bbox).reshape((-1, 2)).astype(np.int32)
                                cv2.fillPoly(text_mask, [points], 255)
                
                # 对文字mask进行形态学处理
                if np.sum(text_mask > 0) > 0:
                    # 膨胀操作扩大文字区域
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    text_mask = cv2.dilate(text_mask, kernel, iterations=2)
                
                # 保存文字mask
                base_name = os.path.splitext(os.path.basename(file_info['original_path']))[0]
                text_mask_path = os.path.join(text_mask_output_folder, f"{base_name}_text_mask.png")
                cv2.imwrite(text_mask_path, text_mask)
                
                # 计算文字区域像素数
                text_pixels = np.sum(text_mask > 0)
                
                successful_files.append({
                    'image_path': image_path,
                    'original_path': file_info['original_path'],
                    'text_mask_path': text_mask_path,
                    'text_pixels': text_pixels,
                    'watermark_ratio': file_info['watermark_ratio']
                })
                
                progress_bar.set_postfix({'已处理': len(successful_files)})
                
            except Exception as e:
                logger.error(f"OCR处理错误: {file_info['image_path']} - {str(e)}")
                continue
        
        logger.info(f"步骤3完成: 成功处理 {len(successful_files)} 张图片")
        return successful_files
    
    def step4_batch_iopaint_text_repair(self, processed_files, final_output_folder, model_name='lama', timeout=300):
        """步骤4: 批量修复所有图片的文字区域"""
        logger.info("=" * 60)
        logger.info("步骤4: 开始批量修复文字区域")
        logger.info("=" * 60)
        
        successful_files = self._batch_iopaint_repair(
            processed_files=processed_files,
            output_folder=final_output_folder,
            mask_key='text_mask_path',
            output_suffix='final',
            model_name=model_name,
            timeout=timeout,
            skip_condition='text_pixels',
            skip_threshold=None
        )
        
        # 转换输出格式以保持向后兼容
        final_results = []
        for file_info in successful_files:
            final_results.append({
                'original_path': file_info['original_path'],
                'final_path': file_info['image_path'],
                'watermark_ratio': file_info['watermark_ratio'],
                'text_pixels': file_info.get('text_pixels', 0)
            })
        
        logger.info(f"步骤4完成: 成功处理 {len(final_results)} 张图片")
        return final_results
     
    def merge_masks_for_video(self, step1_results, step3_results, merged_mask_output_folder):
         """
         合并水印mask和文字mask，生成最终mask用于视频参考
         
         Args:
             step1_results: 步骤1的结果（包含水印mask路径）
             step3_results: 步骤3的结果（包含文字mask路径）
             merged_mask_output_folder: 合并mask的输出文件夹
             
         Returns:
             merged_files: 合并后的文件信息列表
         """
         logger.info("=" * 60)
         logger.info("合并水印mask和文字mask")
         logger.info("=" * 60)
         
         # 创建合并mask输出目录
         os.makedirs(merged_mask_output_folder, exist_ok=True)
         
         # 创建文件映射字典，方便查找对应的文字mask
         text_mask_dict = {}
         if step3_results:
             for file_info in step3_results:
                 original_path = file_info['original_path']
                 base_name = os.path.splitext(os.path.basename(original_path))[0]
                 text_mask_dict[base_name] = file_info['text_mask_path']
         
         merged_files = []
         
         progress_bar = tqdm(step1_results, desc="合并mask", unit="张")
         
         for file_info in progress_bar:
             try:
                 image_path = file_info['image_path']
                 watermark_mask_path = file_info['mask_path']
                 
                 base_name = os.path.splitext(os.path.basename(image_path))[0]
                 merged_mask_path = os.path.join(merged_mask_output_folder, f"{base_name}.png")
                 
                 # 加载水印mask
                 watermark_mask = cv2.imread(watermark_mask_path, cv2.IMREAD_GRAYSCALE)
                 if watermark_mask is None:
                     logger.error(f"无法加载水印mask: {watermark_mask_path}")
                     continue
                 
                 # 初始化合并mask为水印mask
                 merged_mask = watermark_mask.copy()
                 
                 # 查找对应的文字mask
                 text_mask_path = text_mask_dict.get(base_name)
                 if text_mask_path and os.path.exists(text_mask_path):
                     # 加载文字mask
                     text_mask = cv2.imread(text_mask_path, cv2.IMREAD_GRAYSCALE)
                     if text_mask is not None:
                         # 确保两个mask尺寸一致
                         if text_mask.shape != watermark_mask.shape:
                             text_mask = cv2.resize(text_mask, (watermark_mask.shape[1], watermark_mask.shape[0]))
                         
                         # 合并mask：使用逻辑或操作
                         merged_mask = cv2.bitwise_or(watermark_mask, text_mask)
                         
                         logger.debug(f"合并了水印mask和文字mask: {base_name}")
                     else:
                         logger.warning(f"无法加载文字mask: {text_mask_path}")
                 else:
                     logger.debug(f"未找到对应的文字mask，仅使用水印mask: {base_name}")
                 
                 # 对合并后的mask进行优化
                 merged_mask_optimized = self._optimize_mask(merged_mask)
                 
                 # 保存合并后的mask
                 cv2.imwrite(merged_mask_path, merged_mask_optimized)
                 
                 # 计算合并mask的统计信息
                 total_pixels = merged_mask_optimized.shape[0] * merged_mask_optimized.shape[1]
                 mask_pixels = np.sum(merged_mask_optimized > 0)
                 mask_ratio = mask_pixels / total_pixels
                 
                 merged_files.append({
                     'original_path': image_path,
                     'watermark_mask_path': watermark_mask_path,
                     'text_mask_path': text_mask_path,
                     'merged_mask_path': merged_mask_path,
                     'mask_ratio': mask_ratio,
                     'mask_pixels': mask_pixels
                 })
                 
                 progress_bar.set_postfix({'已处理': len(merged_files)})
                 
             except Exception as e:
                 logger.error(f"合并mask失败: {file_info['image_path']} - {str(e)}")
                 continue
         
         logger.info(f"mask合并完成: 成功处理 {len(merged_files)} 张图片")
         return merged_files
     
    def process_folder_batch(self, input_folder, output_folder,
                            watermark_model='lama', text_model='lama',
                            use_ocr=True, ocr_languages=None, ocr_engine='easy',
                            timeout=300, save_intermediate=True, merge_masks=True, limit=None):
        """
        批量处理文件夹：分步骤对所有图片进行处理
        
        Args:
            input_folder: 输入文件夹路径
            output_folder: 输出文件夹路径
            watermark_model: 水印修复模型
            text_model: 文字修复模型
            use_ocr: 是否使用OCR
            ocr_languages: OCR语言列表
            ocr_engine: OCR引擎
            timeout: 每步超时时间
            save_intermediate: 是否保存中间结果
            merge_masks: 是否合并mask用于视频生成
            limit: 限制处理的图片数量，如果为None则处理所有图片
            
        Returns:
            statistics: 处理统计信息
        """
        logger.info("=" * 80)
        logger.info(f"开始批量分步处理文件夹: {input_folder}")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # 创建输出目录结构
        os.makedirs(output_folder, exist_ok=True)
        
        if save_intermediate:
            mask_folder = os.path.join(output_folder, 'step1_masks')
            step2_folder = os.path.join(output_folder, 'step2_watermark_repaired')
            text_mask_folder = os.path.join(output_folder, 'step3_text_masks')
            final_folder = output_folder
        else:
            # 使用临时目录
            temp_dir = tempfile.mkdtemp(prefix="batch_watermark_removal_")
            mask_folder = os.path.join(temp_dir, 'masks')
            step2_folder = os.path.join(temp_dir, 'step2')
            text_mask_folder = os.path.join(temp_dir, 'text_masks')
            final_folder = output_folder
        
        try:
            # 步骤1: 批量预测水印mask
            step1_results = self.step1_batch_predict_watermark_masks(input_folder, mask_folder, limit=limit)
            if not step1_results:
                return {'status': 'error', 'message': '步骤1失败：未找到图像文件或处理失败'}
            
            # 步骤2: 批量修复水印
            step2_results = self.step2_batch_iopaint_watermark_repair(
                step1_results, step2_folder, watermark_model, timeout
            )
            if not step2_results:
                return {'status': 'error', 'message': '步骤2失败：水印修复失败'}
            
            # 步骤3和4: OCR和文字修复（如果启用）
            if use_ocr:
                step3_results = self.step3_batch_extract_text_masks(
                    step2_results, text_mask_folder, ocr_languages, ocr_engine
                )
                if not step3_results:
                    logger.warning("步骤3失败：OCR处理失败，跳过文字修复")
                    # 直接复制步骤2的结果到最终输出
                    for file_info in step2_results:
                        base_name = os.path.splitext(os.path.basename(file_info['original_path']))[0]
                        final_path = os.path.join(final_folder, f"{base_name}_final.png")
                        shutil.copy2(file_info['image_path'], final_path)
                    step4_results = step2_results
                    step3_results = []  # 设置为空列表以便后续合并mask使用
                else:
                    step4_results = self.step4_batch_iopaint_text_repair(
                        step3_results, final_folder, text_model, timeout
                    )
            else:
                logger.info("跳过OCR和文字修复步骤")
                # 直接复制步骤2的结果到最终输出
                for file_info in step2_results:
                    base_name = os.path.splitext(os.path.basename(file_info['original_path']))[0]
                    final_path = os.path.join(final_folder, f"{base_name}_final.png")
                    shutil.copy2(file_info['image_path'], final_path)
                step4_results = step2_results
                step3_results = []  # 设置为空列表以便后续合并mask使用
            
        # 步骤5: 合并水印mask和文字mask，生成最终mask用于视频参考
            if merge_masks and step1_results:
                merged_mask_folder = os.path.join(output_folder, "masks")
                merged_results = self.merge_masks_for_video(
                    step1_results, step3_results, merged_mask_folder
                )
            else:
                merged_results = []
            
            # 计算统计信息
            end_time = time.time()
            processing_time = end_time - start_time
            
            total_images = len(step1_results)
            successful_images = len(step4_results) if 'step4_results' in locals() else len(step2_results)
            
            # 计算平均水印面积比例
            avg_watermark_ratio = 0.0
            avg_text_pixels = 0.0
            if step1_results:
                avg_watermark_ratio = sum(f['watermark_ratio'] for f in step1_results) / len(step1_results)
            if use_ocr and 'step3_results' in locals() and step3_results:
                avg_text_pixels = sum(f['text_pixels'] for f in step3_results) / len(step3_results)
            
            statistics = {
                'status': 'success',
                'total_images': total_images,
                'successful_images': successful_images,
                'success_rate': successful_images / total_images * 100 if total_images > 0 else 0,
                'processing_time': processing_time,
                'avg_processing_time_per_image': processing_time / total_images if total_images > 0 else 0,
                'avg_watermark_ratio': avg_watermark_ratio,
                'avg_text_pixels': avg_text_pixels,
                'steps_completed': {
                    'step1_mask_prediction': len(step1_results),
                    'step2_watermark_repair': len(step2_results),
                    'step3_text_extraction': len(step3_results) if use_ocr and 'step3_results' in locals() else 0,
                    'step4_text_repair': len(step4_results) if 'step4_results' in locals() else 0,
                    'merged_masks': len(merged_results) if 'merged_results' in locals() else 0
                }
            }
            
            # 输出统计信息
            logger.info("=" * 80)
            logger.info("批量处理完成统计:")
            logger.info(f"总图片数: {statistics['total_images']}")
            logger.info(f"成功处理: {statistics['successful_images']}")
            logger.info(f"成功率: {statistics['success_rate']:.2f}%")
            logger.info(f"总处理时间: {statistics['processing_time']:.2f}秒")
            logger.info(f"平均每张图片: {statistics['avg_processing_time_per_image']:.2f}秒")
            logger.info(f"平均水印面积比例: {statistics['avg_watermark_ratio']:.4f}")
            if use_ocr:
                logger.info(f"平均文字区域像素: {statistics['avg_text_pixels']:.0f}")
            logger.info("各步骤完成情况:")
            for step, count in statistics['steps_completed'].items():
                logger.info(f"  {step}: {count}张")
            logger.info("=" * 80)
            
            return statistics
            
        finally:
            # 清理临时目录
            if not save_intermediate and 'temp_dir' in locals() and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='批量4步水印移除工具')
    parser.add_argument('--model', required=True, help='UNet模型路径')
    parser.add_argument('--config', help='配置文件路径')
    parser.add_argument('--input', required=True, help='输入文件夹路径')
    parser.add_argument('--output', required=True, help='输出文件夹路径')
    parser.add_argument('--device', default='cpu', help='设备 (cpu/cuda)')
    parser.add_argument('--watermark-model', default='lama', help='水印修复模型')
    parser.add_argument('--text-model', default='lama', help='文字修复模型')
    parser.add_argument('--use-ocr', action='store_true', help='是否使用OCR')
    parser.add_argument('--ocr-engine', default='easy', choices=['easy', 'paddle'], help='OCR引擎')
    parser.add_argument('--ocr-languages', nargs='+', help='OCR语言列表')
    parser.add_argument('--timeout', type=int, default=300, help='每步超时时间（秒）')
    parser.add_argument('--save-intermediate', action='store_true', help='保存中间结果')
    parser.add_argument('--merge-masks', action='store_true', help='合并水印mask和文字mask用于视频参考')
    parser.add_argument('--limit', type=int, help='限制处理的图片数量，随机选择n张图片进行处理')
    
    args = parser.parse_args()
    
    # 创建批量水印移除器
    remover = WatermarkPredictor(
        model_path=args.model,
        config_path=args.config,
        device=args.device
    )
    
    # 批量处理文件夹
    if os.path.isdir(args.input):
        statistics = remover.process_folder_batch(
            args.input, args.output,
            watermark_model=args.watermark_model,
            text_model=args.text_model,
            use_ocr=args.use_ocr,
            ocr_languages=args.ocr_languages,
            ocr_engine=args.ocr_engine,
            timeout=args.timeout,
            save_intermediate=args.save_intermediate,
            merge_masks=args.merge_masks,
            limit=args.limit
        )
        
        if statistics['status'] == 'success':
            print(f"批量处理完成，成功率: {statistics['success_rate']:.2f}%")
        else:
            print(f"批量处理失败: {statistics['message']}")
    else:
        print(f"输入路径不是文件夹: {args.input}")