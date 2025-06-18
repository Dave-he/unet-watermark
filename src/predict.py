# -*- coding: utf-8 -*-
"""
预测模块
提供水印检测和修复功能
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
from typing import Tuple, Optional, Dict, Any, List

# 导入自定义模块
from configs.config import get_cfg_defaults, update_config
from models.unet_model import create_model_from_config
from utils.dataset import get_val_transform

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WatermarkPredictor:
    
    """水印预测器类 - 简化版，仅支持文件夹修复模式"""
    
    def __init__(self, model_path, config_path=None, config=None, device='cpu'):
        """初始化预测器"""
        self.device = torch.device(device)
        
        # 加载配置
        if config is not None:
            self.cfg = config
        else:
            self.cfg = get_cfg_defaults()
            if config_path and os.path.exists(config_path):
                update_config(self.cfg, config_path)
        
        # 加载模型
        self.model, self.model_info = self._load_model(model_path)
        
        # 数据变换
        self.transform = get_val_transform(self.cfg.DATA.IMG_SIZE)
        
        # 多模型支持
        self.current_model_path = model_path
        self.available_models = self._find_available_models(model_path)
        self.current_model_index = 0
        
        # 迭代级别的模型管理
        self.iteration_model_index = 0
        self.last_iteration_detected = True
        
        logger.info(f"预测器初始化完成，使用设备: {self.device}")
        logger.info(f"发现可用模型: {len(self.available_models)} 个")
        self._print_model_info()
    
    def _find_available_models(self, current_model_path):
        """查找models目录下的所有可用模型"""
        models_dir = os.path.dirname(current_model_path)
        available_models = []
        
        if os.path.exists(models_dir):
            for filename in os.listdir(models_dir):
                if filename.endswith('.pth'):
                    model_path = os.path.join(models_dir, filename)
                    available_models.append(model_path)
        
        # 确保当前模型在列表中且排在第一位
        if current_model_path in available_models:
            available_models.remove(current_model_path)
        available_models.insert(0, current_model_path)
        
        return available_models

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
        
        # 轮廓检测和凸包处理以获得更完整的连通域
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 创建新的mask
            connected_mask = np.zeros_like(mask)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # 最小面积阈值
                    # 计算凸包以获得更完整的连通域
                    hull = cv2.convexHull(contour)
                    
                    # 如果凸包面积与原轮廓面积比值合理，使用凸包
                    hull_area = cv2.contourArea(hull)
                    if hull_area > 0 and area / hull_area > 0.6:  # 凸性比例阈值
                        cv2.fillPoly(connected_mask, [hull], 255)
                    else:
                        # 否则使用多边形近似
                        epsilon = 0.015 * cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, epsilon, True)
                        cv2.fillPoly(connected_mask, [approx], 255)
            
            mask = connected_mask
        
        # 最后的平滑处理
        mask = cv2.GaussianBlur(mask, (3, 3), 0.5)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        return mask

    def select_model_for_iteration(self, iteration):
        """为当前迭代选择最优模型"""
        if iteration == 1 or not self.last_iteration_detected:
            if iteration > 1 and not self.last_iteration_detected:
                if self.iteration_model_index < len(self.available_models) - 1:
                    self.iteration_model_index += 1
                    logger.info(f"上次迭代未检测到水印，切换到模型 {self.iteration_model_index + 1}/{len(self.available_models)}")
                else:
                    logger.info("已尝试所有模型，保持当前模型")
                    return False
            
            # 加载选定的模型
            target_model_path = self.available_models[self.iteration_model_index]
            if target_model_path != self.current_model_path:
                logger.info(f"为第 {iteration} 轮迭代加载模型: {os.path.basename(target_model_path)}")
                self.model, self.model_info = self._load_model(target_model_path)
                self.current_model_path = target_model_path
        
        return True
    
    def predict_mask_single_model(self, image_path):
        """使用当前模型预测水印掩码"""
        current_model_name = os.path.basename(self.current_model_path)
        
        # 预测掩码
        mask = self.predict_mask(image_path)
        
        # 计算水印面积比例
        image = cv2.imread(image_path)
        total_pixels = image.shape[0] * image.shape[1]
        watermark_pixels = np.sum(mask > 0)
        watermark_ratio = watermark_pixels / total_pixels
        
        return mask, current_model_name, watermark_ratio
    
    def _load_model(self, model_path):
        """加载模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        try:
            # 清理之前的模型内存
            if hasattr(self, 'model') and self.model is not None:
                del self.model
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            # 创建模型
            model = create_model_from_config(self.cfg).to(self.device)
            
            # 加载权重
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"从检查点加载模型: {model_path}")
                model_info = {
                    'epoch': checkpoint.get('epoch', 'Unknown'),
                    'val_loss': checkpoint.get('val_loss', 'Unknown'),
                    'val_metrics': checkpoint.get('val_metrics', {}),
                    'train_loss': checkpoint.get('train_loss', 'Unknown'),
                    'train_metrics': checkpoint.get('train_metrics', {}),
                    'best_val_loss': checkpoint.get('best_val_loss', 'Unknown'),
                    'is_final': checkpoint.get('is_final', False)
                }
            else:
                model.load_state_dict(checkpoint)
                logger.info(f"加载模型权重: {model_path}")
                model_info = {'epoch': 'Unknown', 'val_loss': 'Unknown'}
            
            # 清理checkpoint内存
            del checkpoint
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            model.eval()
            return model, model_info
            
        except Exception as e:
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            raise e
    
    def _print_model_info(self):
        """打印模型信息"""
        logger.info("=" * 50)
        logger.info("模型信息:")
        logger.info(f"训练轮数: {self.model_info.get('epoch', 'Unknown')}")
        logger.info(f"验证损失: {self.model_info.get('val_loss', 'Unknown')}")
        logger.info(f"训练损失: {self.model_info.get('train_loss', 'Unknown')}")
        
        val_metrics = self.model_info.get('val_metrics', {})
        if val_metrics:
            logger.info(f"验证IoU: {val_metrics.get('iou', 'Unknown')}")
            logger.info(f"验证F1: {val_metrics.get('f1', 'Unknown')}")
        
        if self.model_info.get('is_final', False):
            logger.info("模型类型: 最终模型")
        elif self.model_info.get('best_val_loss') == self.model_info.get('val_loss'):
            logger.info("模型类型: 最佳模型")
        else:
            logger.info("模型类型: 检查点模型")
        logger.info("=" * 50)
    
    def predict_mask(self, image_path):
        """预测单张图像的水印掩码"""
        # 加载图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法加载图像: {image_path}")
        
        # 转换为RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 应用变换
        if self.transform:
            transformed = self.transform(image=image_rgb)
            input_tensor = transformed['image'].unsqueeze(0).to(self.device)
        else:
            # 如果没有变换，手动处理
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
        
        # 后处理
        mask_processed = self._post_process_mask(mask_binary)
        
        return mask_processed
    
    def _post_process_mask(self, mask):
        """后处理掩码"""
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 移除小的连通组件
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        # 计算最小面积阈值
        total_area = mask.shape[0] * mask.shape[1]
        min_area = total_area * 0.001  # 0.1%的面积阈值
        
        # 创建新的掩码
        new_mask = np.zeros_like(mask)
        for i in range(1, num_labels):  # 跳过背景标签0
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                new_mask[labels == i] = 255
        
        return new_mask
    
    def process_folder_iterative(self, input_folder, output_folder, max_iterations=5,
                            watermark_threshold=0.01, iopaint_model='lama', limit=None):
        """文件夹迭代修复模式 - 核心功能"""
        # 创建输出目录和临时工作目录
        os.makedirs(output_folder, exist_ok=True)
        # 确保data/tmp目录存在
        tmp_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "tmp")
        os.makedirs(tmp_root, exist_ok=True)
        temp_dir = tempfile.mkdtemp(prefix="watermark_removal_", dir=tmp_root)
        
        try:
            # 获取所有图片文件
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(Path(input_folder).glob(ext))
                image_files.extend(Path(input_folder).glob(ext.upper()))
            
            if not image_files:
                logger.warning(f"在 {input_folder} 中未找到图像文件")
                return {'status': 'error', 'message': '未找到图像文件'}
            
            # 如果设置了limit参数，随机选择指定数量的图片
            if limit is not None and limit > 0 and len(image_files) > limit:
                random.shuffle(image_files)
                image_files = image_files[:limit]
                logger.info(f"随机选择了 {len(image_files)} 张图片进行处理（总共 {len(list(Path(input_folder).glob('*')))} 张）")
            
            logger.info(f"开始处理 {len(image_files)} 张图片")
            
            # 初始化图片状态
            image_status = {}
            for img_path in image_files:
                stem = img_path.stem
                current_path = os.path.join(temp_dir, f"{stem}_current.png")
                shutil.copy2(str(img_path), current_path)
                
                image_status[stem] = {
                    'original_path': str(img_path),
                    'current_path': current_path,
                    'completed': False,
                    'iterations': 0,
                    'final_watermark_ratio': 0.0,
                    'detection_model': None,
                    'accumulated_mask': None  # 累积的mask
                }
            
            # 开始迭代处理
            for iteration in range(1, max_iterations + 1):
                logger.info(f"开始第 {iteration}/{max_iterations} 轮处理")
                
                # 为当前迭代选择模型
                if not self.select_model_for_iteration(iteration):
                    logger.info("所有模型都已尝试，停止迭代")
                    break
                
                # 第一阶段：批量掩码预测
                masks_generated, iteration_detected = self._stage1_batch_mask_prediction(
                    image_status, temp_dir, iteration, watermark_threshold
                )
                
                # 更新迭代检测状态
                self.last_iteration_detected = iteration_detected
                
                if not masks_generated:
                    logger.info("所有图片都已完成处理")
                    break
                
                # 第二阶段：批量修复
                self._stage2_batch_repair(
                    image_status, temp_dir, iteration, iopaint_model
                )
                
                # 检查完成状态
                completed_count = sum(1 for info in image_status.values() if info['completed'])
                logger.info(f"第 {iteration} 轮完成，已完成: {completed_count}/{len(image_files)}")
                
                if completed_count == len(image_files):
                    break
            
            # 复制最终结果到输出目录
            self._copy_final_results(image_status, output_folder)
            
            # 生成统计结果
            return self._generate_statistics(image_status, watermark_threshold)
            
        finally:
            # 清理临时目录
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _stage1_batch_mask_prediction(self, image_status, temp_dir, iteration, watermark_threshold):
        """第一阶段：批量掩码预测"""
        logger.info("第一阶段：批量掩码预测")
        
        masks_generated = False
        iteration_detected = False
        
        pending_images = [(name, info) for name, info in image_status.items() 
                         if not info['completed']]
        
        if not pending_images:
            return False, False
        
        current_model_name = os.path.basename(self.current_model_path)
        logger.info(f"本轮迭代使用模型: {current_model_name}")
        
        # 使用进度条显示预测进度
        progress_bar = tqdm(pending_images, desc=f"预测掩码(模型:{current_model_name[:15]})", unit="张")
        
        for image_name, info in progress_bar:
            progress_bar.set_postfix({'当前': image_name[:15] + '...'})
            
            try:
                # 使用当前模型预测
                mask, model_used, watermark_ratio = self.predict_mask_single_model(
                    info['current_path']
                )
                
                # 优化预测的mask
                mask = self._optimize_mask(mask)
                
                info['final_watermark_ratio'] = watermark_ratio
                info['detection_model'] = model_used
                
                # 累积mask：将当前mask的白色区域与之前的mask合并
                if info['accumulated_mask'] is None:
                    # 第一次迭代，直接使用当前mask
                    info['accumulated_mask'] = mask.copy()
                else:
                    # 合并当前mask和累积mask的白色区域
                    info['accumulated_mask'] = cv2.bitwise_or(info['accumulated_mask'], mask)
                
                # 保存累积的最终mask
                final_mask_path = os.path.join(temp_dir, f"{image_name}_final_mask.png")
                cv2.imwrite(final_mask_path, info['accumulated_mask'])
                info['final_mask_path'] = final_mask_path
                
                # 检查是否需要修复
                if watermark_ratio < watermark_threshold:
                    logger.info(f"{image_name}: 无需修复 (水印比例: {watermark_ratio:.6f})")
                    info['completed'] = True
                    info['iterations'] = iteration - 1
                    continue
                
                # 检测到水印
                iteration_detected = True
                
                # 保存当前迭代的掩码
                mask_path = os.path.join(temp_dir, f"{image_name}_mask_iter{iteration}.png")
                cv2.imwrite(mask_path, mask)
                info['mask_path'] = mask_path
                
                logger.info(f"{image_name}: 检测到水印 {watermark_ratio:.6f} (模型: {model_used})")
                masks_generated = True
                
            except Exception as e:
                logger.error(f"{image_name}: 掩码预测失败: {str(e)}")
                info['completed'] = True
                info['iterations'] = iteration
        
        logger.info(f"本轮迭代完成，模型 {current_model_name} 检测状态: {'检测到水印' if iteration_detected else '未检测到水印'}")
        return masks_generated, iteration_detected
    
    def _stage2_batch_repair(self, image_status, temp_dir, iteration, iopaint_model):
        """第二阶段：批量修复"""
        logger.info("第二阶段：批量修复")
        
        # 收集需要修复的图片
        repair_tasks = []
        for image_name, info in image_status.items():
            if not info['completed'] and 'mask_path' in info:
                repair_tasks.append((image_name, info))
        
        if not repair_tasks:
            logger.info("没有需要修复的图片")
            return
        
        logger.info(f"准备批量修复 {len(repair_tasks)} 张图片")
        
        # 创建批量处理的临时目录
        batch_input_dir = os.path.join(temp_dir, f"batch_input_iter{iteration}")
        batch_mask_dir = os.path.join(temp_dir, f"batch_mask_iter{iteration}")
        batch_output_dir = os.path.join(temp_dir, f"batch_output_iter{iteration}")
        
        os.makedirs(batch_input_dir, exist_ok=True)
        os.makedirs(batch_mask_dir, exist_ok=True)
        os.makedirs(batch_output_dir, exist_ok=True)
        
        # 准备批量处理的文件
        batch_mapping = {}
        
        for image_name, info in repair_tasks:
            # 复制图片到批量输入目录
            batch_image_name = f"{image_name}.png"
            batch_image_path = os.path.join(batch_input_dir, batch_image_name)
            shutil.copy2(info['current_path'], batch_image_path)
            
            # 复制mask到批量mask目录
            batch_mask_path = os.path.join(batch_mask_dir, batch_image_name)
            shutil.copy2(info['mask_path'], batch_mask_path)
            
            batch_mapping[batch_image_name] = image_name
        
        try:
            # 执行批量iopaint处理
            success, message = self.batch_remove_watermark_with_iopaint(
                batch_input_dir, batch_mask_dir, batch_output_dir, iopaint_model
            )
            
            if success:
                # 处理批量修复结果
                for batch_image_name, image_name in batch_mapping.items():
                    batch_output_path = os.path.join(batch_output_dir, batch_image_name)
                    
                    if os.path.exists(batch_output_path):
                        # 更新图片状态
                        new_current_path = os.path.join(temp_dir, f"{image_name}_iter{iteration}.png")
                        shutil.copy2(batch_output_path, new_current_path)
                        
                        image_status[image_name]['current_path'] = new_current_path
                        image_status[image_name]['iterations'] = iteration
                        logger.info(f"{image_name}: 批量修复完成")
                    else:
                        logger.error(f"{image_name}: 批量修复输出文件不存在")
                        image_status[image_name]['completed'] = True
                        image_status[image_name]['iterations'] = iteration
            else:
                logger.error(f"批量修复失败: {message}")
                # 标记所有任务为失败
                for image_name, info in repair_tasks:
                    info['completed'] = True
                    info['iterations'] = iteration
        
        finally:
            # 清理批量处理临时目录和mask文件
            for temp_dir_path in [batch_input_dir, batch_mask_dir, batch_output_dir]:
                if os.path.exists(temp_dir_path):
                    shutil.rmtree(temp_dir_path, ignore_errors=True)
            
            # 清理mask文件
            for image_name, info in repair_tasks:
                if 'mask_path' in info:
                    try:
                        os.remove(info['mask_path'])
                    except:
                        pass
                    del info['mask_path']
    
    def batch_remove_watermark_with_iopaint(self, input_dir, mask_dir, output_dir, model_name='lama', timeout=600, repeat_time=3):
        """使用iopaint批量去除水印，支持连续处理多遍"""
        try:
            # 创建临时目录用于中间结果
            temp_dirs = []
            current_input_dir = input_dir
            
            for i in range(repeat_time):
                # 确定当前遍的输出目录
                if i == repeat_time - 1:
                    # 最后一遍，输出到目标目录
                    current_output_dir = output_dir
                else:
                    # 中间遍，输出到临时目录
                    # 确保data/tmp目录存在
                    tmp_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "tmp")
                    os.makedirs(tmp_root, exist_ok=True)
                    temp_dir = tempfile.mkdtemp(prefix=f'iopaint_iter{i+1}_', dir=tmp_root)
                    temp_dirs.append(temp_dir)
                    current_output_dir = temp_dir
                
                # 准备当前遍的iopaint命令
                cmd = [
                    'iopaint', 'run',
                    '--model', model_name,
                    '--device', self.device.type,
                    '--image', current_input_dir,
                    '--mask', mask_dir,
                    '--output', current_output_dir
                ]
                
                # 运行当前遍的iopaint处理
                logger.info(f"执行第{i+1}遍IOPaint处理: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=timeout)
                
                # 检查当前遍是否生成了文件
                output_files = [f for f in os.listdir(current_output_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if not output_files:
                    # 清理临时目录
                    for temp_dir in temp_dirs:
                        if os.path.exists(temp_dir):
                            shutil.rmtree(temp_dir)
                    return False, f"第{i+1}遍IOPaint处理未生成任何输出文件"
                
                logger.info(f"第{i+1}遍IOPaint处理完成，生成 {len(output_files)} 个文件")
                
                # 下一遍的输入目录是当前遍的输出目录
                current_input_dir = current_output_dir
            
            # 清理临时目录
            for temp_dir in temp_dirs:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
            
            # 检查最终输出目录
            final_output_files = [f for f in os.listdir(output_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            logger.info(f"IOPaint连续{repeat_time}遍处理成功，最终生成 {len(final_output_files)} 个文件")
            return True, f"连续{repeat_time}遍处理成功"
            
        except subprocess.TimeoutExpired:
            # 清理临时目录
            for temp_dir in temp_dirs:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
            return False, f"IOPaint处理超时（{timeout}秒）"
        except subprocess.CalledProcessError as e:
            # 清理临时目录
            for temp_dir in temp_dirs:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
            error_msg = e.stderr if e.stderr else e.stdout if e.stdout else "未知错误"
            return False, f"IOPaint处理错误: {error_msg}"
        except Exception as e:
            # 清理临时目录
            for temp_dir in temp_dirs:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
            return False, f"批量处理错误: {str(e)}"
    
    def _copy_final_results(self, image_status, output_folder):
        """复制最终结果到输出目录"""
        logger.info("复制最终结果到输出目录")
        
        # 创建mask子目录
        mask_output_dir = os.path.join(output_folder, 'masks')
        os.makedirs(mask_output_dir, exist_ok=True)
        
        for image_name, info in image_status.items():
            try:
                # 确定输出文件名
                if info['completed'] and info['final_watermark_ratio'] < 0.01:
                    output_filename = f"{image_name}_cleaned.png"
                else:
                    output_filename = f"{image_name}_partial.png"
                
                output_path = os.path.join(output_folder, output_filename)
                shutil.copy2(info['current_path'], output_path)
                
                # 保存最终的mask图片（如果存在）
                if 'final_mask_path' in info and os.path.exists(info['final_mask_path']):
                    mask_filename = f"{image_name}_mask.png"
                    mask_output_path = os.path.join(mask_output_dir, mask_filename)
                    shutil.copy2(info['final_mask_path'], mask_output_path)
                    logger.info(f"{image_name}: mask已保存到 {mask_filename}")
                
                logger.info(f"{image_name}: 已保存到 {output_filename}")
                
            except Exception as e:
                logger.error(f"{image_name}: 复制结果失败: {str(e)}")
    
    def _generate_statistics(self, image_status, watermark_threshold):
        """生成处理统计信息"""
        total_images = len(image_status)
        successful = sum(1 for info in image_status.values() 
                        if info['completed'] and info['final_watermark_ratio'] < watermark_threshold)
        partial = total_images - successful
        
        avg_iterations = np.mean([info['iterations'] for info in image_status.values()])
        
        # 模型使用统计
        model_usage = {}
        for info in image_status.values():
            if info['detection_model']:
                model_usage[info['detection_model']] = model_usage.get(info['detection_model'], 0) + 1
        
        statistics = {
            'status': 'completed',
            'total_images': total_images,
            'successful': successful,
            'partial': partial,
            'success_rate': successful / total_images * 100,
            'average_iterations': round(avg_iterations, 2),
            'model_usage': model_usage
        }
        
        logger.info(f"处理完成: {successful}/{total_images} 成功, 平均迭代 {avg_iterations:.2f} 次")
        logger.info(f"模型使用统计: {model_usage}")
        
        return statistics