#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动循环训练脚本
实现模型选择 -> 训练 -> 预测 -> 数据扩充的循环流程
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
import shutil
import glob
import random
import numpy as np
from typing import Dict, Optional, Any, List

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入各个模块的核心类和函数
from scripts.model_selector import ModelSelector
from train import train, get_cfg_defaults, update_config
from predict import WatermarkPredictor
from scripts.gen_data import load_clean_images, load_watermarks, generate_watermarked_image
from scripts.video_generator import VideoGenerator
from tqdm import tqdm

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/auto_training_loop.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AutoTrainingLoop:
    """自动循环训练类 - 直接调用方法版本"""
    
    def __init__(self, config):
        self.config = config
        self.project_root = config['project_root']
        self.current_cycle = 0
        self.max_cycles = config['max_cycles']
        
        # 数据路径
        self.models_dir = os.path.join(self.project_root, 'models')
        self.checkpoint_dir = os.path.join(self.models_dir, 'checkpoints')
        self.test_data_dir = os.path.join(self.project_root, 'data/test')
        self.train_data_dir = os.path.join(self.project_root, 'data/train')
        
        # 输出路径
        self.output_base_dir = config.get('output_base_dir', 'models/auto')
        os.makedirs(self.output_base_dir, exist_ok=True)
        
        # 验证路径存在
        self._validate_paths()
    
    def _validate_paths(self):
        """验证所需路径是否存在"""
        required_paths = [
            self.test_data_dir,
            self.train_data_dir,
            self.checkpoint_dir
        ]
        
        for path in required_paths:
            if not os.path.exists(path):
                logger.warning(f"路径不存在，将创建: {path}")
                os.makedirs(path, exist_ok=True)
        
        logger.info("路径验证完成")
    
    def _find_best_model(self, model_selection_results):
        """从模型选择结果中找到最佳模型"""
        if not model_selection_results or 'summary' not in model_selection_results:
            logger.warning("模型评估结果为空，使用最新checkpoint")
            return self._find_latest_checkpoint()
        
        summary = model_selection_results.get('summary', {})
        best_model_info = summary.get('best_detection_model')
        
        if best_model_info:
            best_model_name = best_model_info['name']
            model_path = os.path.join(self.checkpoint_dir, best_model_name)
            
            if os.path.exists(model_path):
                logger.info(f"找到最佳模型: {best_model_name} (检测率: {best_model_info['detection_rate']:.2%})")
                return model_path
            else:
                logger.warning(f"最佳模型文件不存在: {model_path}")
        
        return self._find_latest_checkpoint()
    
    def _find_latest_checkpoint(self):
        """查找最新的checkpoint文件"""
        if not os.path.exists(self.checkpoint_dir):
            logger.warning(f"Checkpoint目录不存在: {self.checkpoint_dir}")
            return None
        
        checkpoint_files = glob.glob(os.path.join(self.checkpoint_dir, '*.pth'))
        
        if not checkpoint_files:
            logger.warning("未找到任何checkpoint文件")
            return None
        
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        logger.info(f"使用最新checkpoint: {os.path.basename(latest_checkpoint)}")
        return latest_checkpoint
    
    def step1_model_selection(self, cycle_output_dir):
        """步骤1: 模型选择"""
        logger.info(f"=== 第{self.current_cycle}轮 - 步骤1: 模型选择 ===")
        
        model_selection_dir = os.path.join(cycle_output_dir, 'model_selection')
        os.makedirs(model_selection_dir, exist_ok=True)
        
        try:
            # 直接创建ModelSelector实例并运行
            selector = ModelSelector(
                input_dir=self.test_data_dir,
                model_dir=self.checkpoint_dir,
                output_dir=model_selection_dir,
                num_samples=self.config.get('model_selection_samples', 1000),
                config_path=self.config.get('train_config'),
                device=self.config.get('device', 'cpu')
            )
            
            results = selector.run_evaluation()
            best_model = self._find_best_model(results)
            return best_model
            
        except Exception as e:
            logger.error(f"模型选择失败: {e}")
            return self._find_latest_checkpoint()
    
    def step2_training(self, cycle_output_dir, resume_model=None, use_blurred_mask=False):
        """步骤2: 从检查点继续训练指定轮数"""
        logger.info(f"=== 第{self.current_cycle}轮 - 步骤2: 模型训练 ===")
        
        try:
            # 加载训练配置
            cfg = get_cfg_defaults()
            config_path = self.config.get('train_config', 'src/configs/unet_watermark.yaml')
            
            if os.path.exists(config_path):
                update_config(cfg, config_path)
            else:
                logger.warning(f"配置文件不存在: {config_path}，使用默认配置")
            
            # 解冻配置以允许修改
            cfg.defrost()
            
            # 更新配置参数
            cfg.DEVICE = self.config.get('device', 'cpu')
            
            # 从配置中获取要训练的轮数，默认为10轮
            additional_epochs = self.config.get('epochs', 10)
            
            # 如果有检查点，从检查点继续训练
            if resume_model and os.path.exists(resume_model):
                logger.info(f"从检查点 {resume_model} 继续训练 {additional_epochs} 轮")
                # 加载检查点获取当前epoch
                import torch
                checkpoint = torch.load(resume_model, map_location='cpu', weights_only=False)
                current_epoch = checkpoint.get('epoch', 0)
                target_epochs = current_epoch + additional_epochs
                logger.info(f"当前epoch: {current_epoch}, 目标epoch: {target_epochs}")
            else:
                target_epochs = additional_epochs
                logger.info(f"从头开始训练 {target_epochs} 轮")
            
            cfg.TRAIN.EPOCHS = target_epochs
            cfg.TRAIN.BATCH_SIZE = self.config.get('batch_size', 8)
            cfg.TRAIN.LR = self.config.get('learning_rate', 0.001)
            
            # 保持早停机制启用（所有轮次都使用早停）
            # if self.current_cycle > 1:
            #     cfg.TRAIN.USE_EARLY_STOPPING = False
            
            # 重新冻结配置
            cfg.freeze()
            
            # 直接调用训练函数
            train(cfg, resume_from=resume_model, use_blurred_mask=use_blurred_mask)
            
            # 查找训练产生的最新模型
            latest_model = self._find_latest_checkpoint()
            return latest_model
            
        except Exception as e:
            logger.error(f"训练失败: {e}")
            return None
    
    def step3_prediction(self, cycle_output_dir, model_path):
        """步骤3: 模型预测"""
        logger.info(f"=== 第{self.current_cycle}轮 - 步骤3: 模型预测 ===")
        
        if not model_path or not os.path.exists(model_path):
            logger.error("模型文件不存在，跳过预测步骤")
            return False, None
        
        prediction_dir = os.path.join(cycle_output_dir, 'prediction')
        os.makedirs(prediction_dir, exist_ok=True)
        
        try:
            # 直接创建预测器实例
            predictor = WatermarkPredictor(
                model_path=model_path,
                config_path=self.config.get('train_config'),
                device=self.config.get('device', 'cpu')
            )
            
            # 获取测试图片列表（限制数量以节省时间）
            test_images = []
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                test_images.extend(glob.glob(os.path.join(self.test_data_dir, ext)))
            
            # 限制处理图片数量
            limit = self.config.get('prediction_limit', 100)
            if len(test_images) > limit:
                test_images = random.sample(test_images, limit)
            
            # 执行预测 - 使用正确的方法名
            results = predictor.process_folder_iterative(
                input_folder=self.test_data_dir,
                output_folder=prediction_dir,
                max_iterations=3,
                watermark_threshold=0.1,
                limit=limit,
                use_ocr=False,
                ocr_languages=['en', 'ch_sim'],
                ocr_engine='easy'
            )
            
            logger.info(f"预测完成，处理了 {len(test_images)} 张图片")
            return True, prediction_dir
            
        except Exception as e:
            logger.error(f"预测失败: {e}")
            return False, None
    
    def step4_video_generation(self, cycle_output_dir, prediction_dir):
        """步骤4: 生成对比视频"""
        logger.info(f"=== 第{self.current_cycle}轮 - 步骤4: 生成对比视频 ===")
        
        if not prediction_dir or not os.path.exists(prediction_dir):
            logger.error("预测结果目录不存在，跳过视频生成步骤")
            return False
        
        # 检查是否启用视频生成
        if not self.config.get('generate_video', True):
            logger.info("视频生成功能已禁用，跳过此步骤")
            return True
        
        video_output_dir = os.path.join(cycle_output_dir, 'videos')
        os.makedirs(video_output_dir, exist_ok=True)
        
        try:
            # 创建视频生成器实例
            video_generator = VideoGenerator(
                input_dir=self.test_data_dir,
                repair_dir=prediction_dir,
                output_dir=video_output_dir,
                width=self.config.get('video_width', 1920),
                height=self.config.get('video_height', 1080),
                duration_per_image=self.config.get('video_duration_per_image', 2.0),
                fps=self.config.get('video_fps', 30)
            )
            
            # 生成对比视频
            video_mode = self.config.get('video_mode', 'sidebyside')  # 默认使用并排对比
            
            if video_mode == 'sidebyside':
                video_path = video_generator.create_side_by_side_video()
                logger.info(f"并排对比视频生成完成: {video_path}")
            elif video_mode == 'comparison':
                video_path = video_generator.create_comparison_video()
                logger.info(f"切换对比视频生成完成: {video_path}")
            else:
                logger.warning(f"未知的视频模式: {video_mode}，使用默认的并排对比模式")
                video_path = video_generator.create_side_by_side_video()
                logger.info(f"并排对比视频生成完成: {video_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"视频生成失败: {e}")
            return False
    
    def step5_data_augmentation(self, cycle_output_dir):
        """步骤5: 数据扩充"""
        logger.info(f"=== 第{self.current_cycle}轮 - 步骤5: 数据扩充 ===")
        
        try:
            # 计算当前数据集大小
            watermarked_dir = os.path.join(self.train_data_dir, 'watermarked')
            os.makedirs(watermarked_dir, exist_ok=True)
            
            # 只在需要时创建masks目录
            generate_mask = self.config.get('generate_mask', False)
            if generate_mask:
                masks_dir = os.path.join(self.train_data_dir, 'masks')
                os.makedirs(masks_dir, exist_ok=True)
            
            current_count = len([f for f in os.listdir(watermarked_dir) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            
            if current_count == 0:
                current_count = 1000  # 默认初始数量
            
            # 计算需要增加的数量（20%）
            augment_count = max(100, int(current_count * 0.2))
            target_count = current_count + augment_count
            
            mask_info = "（生成mask）" if generate_mask else "（不生成mask）"
            logger.info(f"当前数据集大小: {current_count}, 目标大小: {target_count} {mask_info}")
            
            # 加载干净图片和水印
            clean_dir = os.path.join(self.train_data_dir, 'clean')
            logos_dir = self.config.get('logos_dir', 'data/WatermarkDataset/logos')
            
            clean_images = load_clean_images(clean_dir)
            watermarks = load_watermarks(logos_dir)
            
            if len(clean_images) == 0 or len(watermarks) == 0:
                logger.error("没有找到干净图片或水印图片")
                return False
            
            # 设置随机种子
            random.seed(42 + self.current_cycle)
            np.random.seed(42 + self.current_cycle)
            
            # 生成新图片
            generated_count = 0
            transparent_ratio = self.config.get('transparent_ratio', 0.6)
            target_transparent = int(augment_count * transparent_ratio)
            transparent_count = 0
            
            pbar = tqdm(total=augment_count, desc="生成带水印图片")
            
            while generated_count < augment_count:
                # 随机选择干净图片和水印
                clean_img_path = random.choice(clean_images)
                watermark_path = random.choice(watermarks)
                
                # 决定是否使用透明水印
                use_transparent = transparent_count < target_transparent
                if use_transparent:
                    transparent_count += 1
                
                # 生成文件名
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_name = f"generated_{timestamp}_{generated_count:06d}.png"
                
                watermarked_path = os.path.join(watermarked_dir, output_name)
                if generate_mask:
                    mask_path = os.path.join(masks_dir, output_name)
                
                try:
                    # 生成带水印图片
                    watermarked_img, mask = generate_watermarked_image(
                        clean_img_path, watermark_path, 
                        enhance_transparent=use_transparent
                    )
                    
                    # 保存图片
                    watermarked_img.save(watermarked_path, quality=95)
                    
                    # 只在需要时保存mask
                    if generate_mask:
                        mask.save(mask_path)
                    
                    generated_count += 1
                    pbar.update(1)
                    
                except Exception as e:
                    logger.warning(f"生成图片失败: {e}")
                    continue
            
            pbar.close()
            if generate_mask:
                logger.info(f"数据扩充完成，生成了 {generated_count} 张新图片和对应的mask")
            else:
                logger.info(f"数据扩充完成，生成了 {generated_count} 张新图片（未生成mask）")
            return True
            
        except Exception as e:
            logger.error(f"数据扩充失败: {e}")
            return False
    
    def run_cycle(self, cycle_num):
        """运行单个训练循环"""
        self.current_cycle = cycle_num
        
        logger.info(f"\n{'='*60}")
        logger.info(f"开始第 {cycle_num}/{self.max_cycles} 轮训练循环")
        logger.info(f"{'='*60}")
        
        cycle_output_dir = os.path.join(self.output_base_dir, f'cycle_{cycle_num:02d}')
        os.makedirs(cycle_output_dir, exist_ok=True)
        
        try:
            # 步骤1: 模型选择
            best_model = self.step1_model_selection(cycle_output_dir)
            
            # 步骤2: 模型训练
            use_blurred_mask = self.config.get('use_blurred_mask', False)
            trained_model = self.step2_training(cycle_output_dir, best_model, use_blurred_mask)
            
            # 步骤3: 模型预测
            prediction_success = False
            prediction_dir = None
            if trained_model:
                prediction_success, prediction_dir = self.step3_prediction(cycle_output_dir, trained_model)
            
            # 步骤4: 生成对比视频
            if prediction_success and prediction_dir:
                self.step4_video_generation(cycle_output_dir, prediction_dir)
            
            # 步骤5: 数据扩充（除了最后一轮）
            if cycle_num < self.max_cycles:
                self.step5_data_augmentation(cycle_output_dir)
            
            # 保存循环信息
            cycle_info = {
                'cycle': cycle_num,
                'timestamp': datetime.now().isoformat(),
                'best_model': os.path.basename(best_model) if best_model else None,
                'trained_model': os.path.basename(trained_model) if trained_model else None,
                'status': 'completed'
            }
            
            with open(os.path.join(cycle_output_dir, 'cycle_info.json'), 'w') as f:
                json.dump(cycle_info, f, indent=2)
            
            logger.info(f"第 {cycle_num} 轮训练循环完成")
            return True
            
        except Exception as e:
            logger.error(f"第 {cycle_num} 轮训练循环失败: {e}")
            
            cycle_info = {
                'cycle': cycle_num,
                'timestamp': datetime.now().isoformat(),
                'status': 'failed',
                'error': str(e)
            }
            
            with open(os.path.join(cycle_output_dir, 'cycle_info.json'), 'w') as f:
                json.dump(cycle_info, f, indent=2)
            
            return False
    
    def run_all_cycles(self):
        """运行所有训练循环"""
        logger.info(f"开始自动循环训练，共 {self.max_cycles} 轮")
        
        successful_cycles = 0
        
        for cycle_num in range(1, self.max_cycles + 1):
            success = self.run_cycle(cycle_num)
            
            if success:
                successful_cycles += 1
            else:
                logger.error(f"第 {cycle_num} 轮失败，继续下一轮")
        
        # 生成最终报告
        final_report = {
            'total_cycles': self.max_cycles,
            'successful_cycles': successful_cycles,
            'failed_cycles': self.max_cycles - successful_cycles,
            'completion_rate': successful_cycles / self.max_cycles,
            'end_time': datetime.now().isoformat(),
            'config': self.config
        }
        
        with open(os.path.join(self.output_base_dir, 'final_report.json'), 'w') as f:
            json.dump(final_report, f, indent=2)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"自动循环训练完成！")
        logger.info(f"成功完成: {successful_cycles}/{self.max_cycles} 轮")
        logger.info(f"完成率: {successful_cycles/self.max_cycles:.1%}")
        logger.info(f"结果保存在: {self.output_base_dir}")
        logger.info(f"{'='*60}")

def auto_main():
    """主函数"""
    parser = argparse.ArgumentParser(description='自动循环训练脚本 - 直接调用方法版本')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--project-root', type=str, default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       help='项目根目录')
    parser.add_argument('--max-cycles', type=int, default=10,
                       help='最大循环次数')
    parser.add_argument('--device', type=str, default='cpu',
                       help='训练设备')
    parser.add_argument('--epochs', type=int, default=50,
                       help='每轮训练的epoch数')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='批次大小')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--output-dir', type=str, default='models/auto',
                       help='输出目录')
    parser.add_argument('--generate-mask', action='store_true',
                       help='是否在数据扩充时生成mask图（默认不生成）')
    parser.add_argument('--generate-video', action='store_true', default=True,
                       help='是否在预测后生成对比视频（默认生成）')
    parser.add_argument('--video-mode', type=str, default='sidebyside', choices=['sidebyside', 'comparison'],
                       help='视频生成模式：sidebyside（并排对比）或comparison（切换对比）')
    parser.add_argument('--video-width', type=int, default=1280,
                       help='视频宽度')
    parser.add_argument('--video-height', type=int, default=720,
                       help='视频高度')
    parser.add_argument('--video-fps', type=int, default=30,
                       help='视频帧率')
    parser.add_argument('--video-duration', type=float, default=2.0,
                       help='每张图片在视频中的展示时长（秒）')
    
    args = parser.parse_args()
    
    # 构建配置
    config = {
        'project_root': args.project_root,
        'max_cycles': args.max_cycles,
        'device': args.device,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'output_base_dir': args.output_dir,
        'generate_mask': getattr(args, 'generate_mask', False),
        'generate_video': getattr(args, 'generate_video', True),
        'video_mode': getattr(args, 'video_mode', 'sidebyside'),
        'video_width': getattr(args, 'video_width', 1280),
        'video_height': getattr(args, 'video_height', 720),
        'video_fps': getattr(args, 'video_fps', 30),
        'video_duration_per_image': getattr(args, 'video_duration', 2.0),
        'train_config': 'src/configs/unet_watermark.yaml',
        'model_selection_samples': 1000,
        'prediction_limit': 100,
        'transparent_ratio': 0.6,
        'logos_dir': 'data/WatermarkDataset/logos'
    }
    
    # 如果提供了配置文件，加载配置
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            file_config = json.load(f)
        config.update(file_config)
    
    # 创建并运行自动训练循环
    trainer = AutoTrainingLoop(config)
    trainer.run_all_cycles()

if __name__ == '__main__':
    auto_main()