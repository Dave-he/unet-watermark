#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型挑选脚本
从输入文件夹随机选择图片，使用多个模型进行预测，输出每个模型的预测结果
"""

import os
import sys
import random
import argparse
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predict import WatermarkPredictor
from configs.config import get_cfg_defaults, update_config

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelSelector:
    """模型选择器类"""
    
    def __init__(self, input_dir, model_dir, output_dir, num_samples=10, config_path=None, device='cpu'):
        """
        初始化模型选择器
        
        Args:
            input_dir (str): 输入图片目录
            model_dir (str): 模型目录
            output_dir (str): 输出目录
            num_samples (int): 随机选择的图片数量
            config_path (str): 配置文件路径
            device (str): 计算设备
        """
        self.input_dir = input_dir
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.num_samples = num_samples
        self.config_path = config_path
        self.device = device
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取图片列表
        self.image_paths = self._get_image_paths()
        
        # 获取模型列表
        self.model_paths = self._get_model_paths()
        
        logger.info(f"找到 {len(self.image_paths)} 张图片")
        logger.info(f"找到 {len(self.model_paths)} 个模型")
    
    def _get_image_paths(self):
        """获取输入目录下的所有图片路径"""
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        image_paths = []
        
        if os.path.isdir(self.input_dir):
            for filename in os.listdir(self.input_dir):
                if any(filename.lower().endswith(ext) for ext in image_extensions):
                    image_paths.append(os.path.join(self.input_dir, filename))
        elif os.path.isfile(self.input_dir):
            if any(self.input_dir.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(self.input_dir)
        
        return sorted(image_paths)
    
    def _get_model_paths(self):
        """获取模型目录下的所有模型路径（包括子目录）"""
        model_paths = []
        
        if os.path.isdir(self.model_dir):
            # 递归搜索所有子目录
            for root, dirs, files in os.walk(self.model_dir):
                for filename in files:
                    if filename.endswith('.pth'):
                        model_paths.append(os.path.join(root, filename))
        elif os.path.isfile(self.model_dir):
            if self.model_dir.endswith('.pth'):
                model_paths.append(self.model_dir)
        
        return sorted(model_paths)
    
    def _select_random_images(self):
        """随机选择指定数量的图片"""
        if len(self.image_paths) <= self.num_samples:
            selected_images = self.image_paths.copy()
        else:
            selected_images = random.sample(self.image_paths, self.num_samples)
        
        logger.info(f"从 {len(self.image_paths)} 张图片中随机选择了 {len(selected_images)} 张")
        return selected_images
    
    def _calculate_watermark_metrics(self, mask, image_shape):
        """计算水印相关指标"""
        total_pixels = image_shape[0] * image_shape[1]
        watermark_pixels = np.sum(mask > 0)
        watermark_ratio = watermark_pixels / total_pixels
        
        # 计算水印区域的连通性
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        num_components = num_labels - 1  # 减去背景
        
        # 计算最大连通区域的面积
        if num_components > 0:
            component_areas = stats[1:, cv2.CC_STAT_AREA]  # 排除背景
            max_component_area = np.max(component_areas)
            max_component_ratio = max_component_area / total_pixels
        else:
            max_component_area = 0
            max_component_ratio = 0
        
        return {
            'watermark_ratio': float(watermark_ratio),
            'watermark_pixels': int(watermark_pixels),
            'total_pixels': int(total_pixels),
            'num_components': int(num_components),
            'max_component_area': int(max_component_area),
            'max_component_ratio': float(max_component_ratio)
        }
    
    def run_evaluation(self):
        """运行模型评估"""
        # 随机选择图片
        selected_images = self._select_random_images()
        
        # 存储所有结果
        all_results = {
            'timestamp': datetime.now().isoformat(),
            'input_dir': self.input_dir,
            'model_dir': self.model_dir,
            'num_samples': len(selected_images),
            'selected_images': [os.path.basename(img) for img in selected_images],
            'models': {},
            'summary': {}
        }
        
        # 为每个模型创建预测器并进行预测
        for model_path in tqdm(self.model_paths, desc="评估模型", unit="模型"):
            model_name = os.path.basename(model_path)
            logger.info(f"\n开始评估模型: {model_name}")
            
            try:
                # 创建预测器
                predictor = WatermarkPredictor(
                    model_path=model_path,
                    config_path=self.config_path,
                    device=self.device
                )
                
                # 存储当前模型的结果
                model_results = {
                    'model_path': model_path,
                    'model_name': model_name,
                    'model_info': predictor.model_info,
                    'predictions': []
                }
                
                # 对每张图片进行预测
                image_progress = tqdm(selected_images, desc=f"预测 {model_name}", unit="图片", leave=False)
                
                for image_path in image_progress:
                    image_name = os.path.basename(image_path)
                    image_progress.set_postfix({'当前': image_name[:20]})
                    
                    try:
                        # 预测水印掩码
                        mask = predictor.predict_mask(image_path)
                        
                        # 读取原图获取尺寸
                        image = cv2.imread(image_path)
                        
                        # 计算水印指标
                        metrics = self._calculate_watermark_metrics(mask, image.shape[:2])
                        
                        # 保存掩码图片
                        mask_filename = f"{Path(image_name).stem}_{model_name.replace('.pth', '')}_mask.png"
                        mask_path = os.path.join(self.output_dir, mask_filename)
                        cv2.imwrite(mask_path, mask)
                        
                        # 存储预测结果
                        prediction_result = {
                            'image_name': image_name,
                            'image_path': image_path,
                            'mask_path': mask_path,
                            'metrics': metrics,
                            'success': True,
                            'error': None
                        }
                        
                        model_results['predictions'].append(prediction_result)
                        
                    except Exception as e:
                        logger.error(f"预测图片 {image_name} 失败: {str(e)}")
                        
                        # 存储错误结果
                        prediction_result = {
                            'image_name': image_name,
                            'image_path': image_path,
                            'mask_path': None,
                            'metrics': None,
                            'success': False,
                            'error': str(e)
                        }
                        
                        model_results['predictions'].append(prediction_result)
                
                # 计算模型统计信息
                successful_predictions = [p for p in model_results['predictions'] if p['success']]
                
                if successful_predictions:
                    watermark_ratios = [p['metrics']['watermark_ratio'] for p in successful_predictions]
                    model_stats = {
                        'total_predictions': len(model_results['predictions']),
                        'successful_predictions': len(successful_predictions),
                        'failed_predictions': len(model_results['predictions']) - len(successful_predictions),
                        'avg_watermark_ratio': float(np.mean(watermark_ratios)),
                        'std_watermark_ratio': float(np.std(watermark_ratios)),
                        'min_watermark_ratio': float(np.min(watermark_ratios)),
                        'max_watermark_ratio': float(np.max(watermark_ratios)),
                        'detection_rate': float(np.mean([1 if ratio > 0.001 else 0 for ratio in watermark_ratios]))
                    }
                else:
                    model_stats = {
                        'total_predictions': len(model_results['predictions']),
                        'successful_predictions': 0,
                        'failed_predictions': len(model_results['predictions']),
                        'avg_watermark_ratio': 0.0,
                        'std_watermark_ratio': 0.0,
                        'min_watermark_ratio': 0.0,
                        'max_watermark_ratio': 0.0,
                        'detection_rate': 0.0
                    }
                
                model_results['statistics'] = model_stats
                all_results['models'][model_name] = model_results
                
                logger.info(f"模型 {model_name} 评估完成:")
                logger.info(f"  成功预测: {model_stats['successful_predictions']}/{model_stats['total_predictions']}")
                logger.info(f"  平均水印比例: {model_stats['avg_watermark_ratio']:.6f}")
                logger.info(f"  检测率: {model_stats['detection_rate']:.2%}")
                
            except Exception as e:
                logger.error(f"加载模型 {model_name} 失败: {str(e)}")
                
                # 存储模型加载失败的结果
                all_results['models'][model_name] = {
                    'model_path': model_path,
                    'model_name': model_name,
                    'model_info': None,
                    'predictions': [],
                    'statistics': None,
                    'load_error': str(e)
                }
        
        # 生成总体统计
        successful_models = [name for name, result in all_results['models'].items() 
                           if 'load_error' not in result and result['statistics'] is not None]
        
        if successful_models:
            # 找出检测率最高的模型
            best_model = max(successful_models, 
                           key=lambda name: all_results['models'][name]['statistics']['detection_rate'])
            
            # 找出平均水印比例最高的模型
            highest_ratio_model = max(successful_models,
                                    key=lambda name: all_results['models'][name]['statistics']['avg_watermark_ratio'])
            
            all_results['summary'] = {
                'total_models': len(self.model_paths),
                'successful_models': len(successful_models),
                'failed_models': len(self.model_paths) - len(successful_models),
                'best_detection_model': {
                    'name': best_model,
                    'detection_rate': all_results['models'][best_model]['statistics']['detection_rate']
                },
                'highest_ratio_model': {
                    'name': highest_ratio_model,
                    'avg_ratio': all_results['models'][highest_ratio_model]['statistics']['avg_watermark_ratio']
                }
            }
        else:
            all_results['summary'] = {
                'total_models': len(self.model_paths),
                'successful_models': 0,
                'failed_models': len(self.model_paths),
                'best_detection_model': None,
                'highest_ratio_model': None
            }
        
        # 保存结果到JSON文件
        results_file = os.path.join(self.output_dir, 'model_evaluation_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n评估结果已保存到: {results_file}")
        
        # 打印总结
        self._print_summary(all_results)
        
        return all_results
    
    def _print_summary(self, results):
        """打印评估总结"""
        print("\n" + "="*60)
        print("模型评估总结")
        print("="*60)
        
        summary = results['summary']
        print(f"总模型数: {summary['total_models']}")
        print(f"成功加载: {summary['successful_models']}")
        print(f"加载失败: {summary['failed_models']}")
        
        if summary['best_detection_model']:
            print(f"\n最佳检测模型: {summary['best_detection_model']['name']}")
            print(f"检测率: {summary['best_detection_model']['detection_rate']:.2%}")
            
            print(f"\n最高水印比例模型: {summary['highest_ratio_model']['name']}")
            print(f"平均水印比例: {summary['highest_ratio_model']['avg_ratio']:.6f}")
        
        print("\n各模型详细统计:")
        print("-" * 60)
        
        for model_name, model_result in results['models'].items():
            if 'load_error' in model_result:
                print(f"{model_name}: 加载失败 - {model_result['load_error']}")
            elif model_result['statistics']:
                stats = model_result['statistics']
                print(f"{model_name}:")
                print(f"  成功预测: {stats['successful_predictions']}/{stats['total_predictions']}")
                print(f"  检测率: {stats['detection_rate']:.2%}")
                print(f"  平均水印比例: {stats['avg_watermark_ratio']:.6f}")
                print(f"  水印比例范围: {stats['min_watermark_ratio']:.6f} - {stats['max_watermark_ratio']:.6f}")
            else:
                print(f"{model_name}: 统计信息不可用")
        
        print("="*60)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='模型挑选脚本')
    parser.add_argument('--input', type=str, required=True,
                       help='输入图片目录或文件路径')
    parser.add_argument('--model', type=str, required=True,
                       help='模型目录或文件路径')
    parser.add_argument('--output', type=str, required=True,
                       help='输出目录')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='随机选择的图片数量 (默认: 10)')
    parser.add_argument('--config', type=str,
                       help='配置文件路径')
    parser.add_argument('--device', type=str, default='cpu',
                       help='计算设备 (默认: cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子 (默认: 42)')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 检查输入路径
    if not os.path.exists(args.input):
        logger.error(f"输入路径不存在: {args.input}")
        return
    
    if not os.path.exists(args.model):
        logger.error(f"模型路径不存在: {args.model}")
        return
    
    # 创建模型选择器
    selector = ModelSelector(
        input_dir=args.input,
        model_dir=args.model,
        output_dir=args.output,
        num_samples=args.num_samples,
        config_path=args.config,
        device=args.device
    )
    
    # 运行评估
    results = selector.run_evaluation()
    
    logger.info(f"模型评估完成！")


if __name__ == '__main__':
    main()