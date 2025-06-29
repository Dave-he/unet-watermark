#!/usr/bin/env python3
"""
文字水印检测测试脚本
快速验证文字水印检测效果
"""

import os
import sys
import torch
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

# 添加src目录到路径
sys.path.append('src')

from predict import WatermarkPredictor
from ocr.easy_ocr import EasyOCRDetector
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextWatermarkTester:
    """文字水印检测测试器"""
    
    def __init__(self, model_path=None, device='auto'):
        """
        初始化测试器
        
        Args:
            model_path: 模型路径，如果为None则使用默认模型
            device: 设备类型
        """
        # 设置设备
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"使用设备: {self.device}")
        
        # 初始化预测器
        self.predictor = WatermarkPredictor(
            model_path=model_path,
            device=self.device
        )
        
        # 初始化OCR检测器
        self.ocr_detector = EasyOCRDetector()
        
        logger.info("文字水印检测器初始化完成")
    
    def test_single_image(self, image_path, output_dir="test_results", show_steps=True):
        """
        测试单张图像的文字水印检测
        
        Args:
            image_path: 输入图像路径
            output_dir: 输出目录
            show_steps: 是否显示中间步骤
        """
        logger.info(f"测试图像: {image_path}")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"无法读取图像: {image_path}")
            return
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 1. 原始UNet预测
        logger.info("1. 执行原始UNet预测...")
        original_mask = self.predictor.predict_mask(image_path)
        
        # 2. 文字特定UNet预测
        logger.info("2. 执行文字特定UNet预测...")
        text_mask = self.predictor.predict_text_watermark_mask(image_path)
        
        # 3. 混合模式预测
        logger.info("3. 执行混合模式预测...")
        mixed_mask = self.predictor.predict_mixed_watermark_mask(image_path)
        
        # 4. OCR文字检测
        logger.info("4. 执行OCR文字检测...")
        ocr_mask = self.ocr_detector.detect_text_mask(image_path)
        
        # 5. 智能类型检测
        logger.info("5. 执行智能类型检测...")
        auto_mask = self.predictor.predict_mask(image_path)  # 这会自动检测类型
        
        # 保存结果
        base_name = Path(image_path).stem
        
        # 保存原始图像
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_original.jpg"), image)
        
        # 保存各种mask
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_original_mask.png"), original_mask * 255)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_text_mask.png"), text_mask * 255)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_mixed_mask.png"), mixed_mask * 255)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_ocr_mask.png"), ocr_mask * 255)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_auto_mask.png"), auto_mask * 255)
        
        # 创建对比图
        if show_steps:
            self._create_comparison_plot(
                image_rgb, original_mask, text_mask, mixed_mask, ocr_mask, auto_mask,
                os.path.join(output_dir, f"{base_name}_comparison.png")
            )
        
        # 计算指标
        metrics = self._calculate_comparison_metrics(
            original_mask, text_mask, mixed_mask, ocr_mask, auto_mask
        )
        
        # 保存指标报告
        self._save_metrics_report(metrics, os.path.join(output_dir, f"{base_name}_metrics.txt"))
        
        logger.info(f"测试完成，结果保存到: {output_dir}")
        return metrics
    
    def _create_comparison_plot(self, image, original_mask, text_mask, mixed_mask, ocr_mask, auto_mask, save_path):
        """
        创建对比图
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 原始图像
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('原始图像')
        axes[0, 0].axis('off')
        
        # 原始UNet预测
        axes[0, 1].imshow(original_mask, cmap='gray')
        axes[0, 1].set_title('原始UNet预测')
        axes[0, 1].axis('off')
        
        # 文字特定预测
        axes[0, 2].imshow(text_mask, cmap='gray')
        axes[0, 2].set_title('文字特定预测')
        axes[0, 2].axis('off')
        
        # 混合模式预测
        axes[1, 0].imshow(mixed_mask, cmap='gray')
        axes[1, 0].set_title('混合模式预测')
        axes[1, 0].axis('off')
        
        # OCR检测
        axes[1, 1].imshow(ocr_mask, cmap='gray')
        axes[1, 1].set_title('OCR检测')
        axes[1, 1].axis('off')
        
        # 智能自动检测
        axes[1, 2].imshow(auto_mask, cmap='gray')
        axes[1, 2].set_title('智能自动检测')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _calculate_comparison_metrics(self, original_mask, text_mask, mixed_mask, ocr_mask, auto_mask):
        """
        计算对比指标
        """
        def calculate_mask_stats(mask):
            total_pixels = mask.size
            positive_pixels = np.sum(mask > 0.5)
            coverage = positive_pixels / total_pixels
            
            # 计算连通组件
            mask_binary = (mask > 0.5).astype(np.uint8)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_binary)
            num_components = num_labels - 1  # 减去背景
            
            # 计算平均组件大小
            if num_components > 0:
                component_sizes = stats[1:, cv2.CC_STAT_AREA]  # 排除背景
                avg_component_size = np.mean(component_sizes)
                max_component_size = np.max(component_sizes)
            else:
                avg_component_size = 0
                max_component_size = 0
            
            return {
                'coverage': coverage,
                'num_components': num_components,
                'avg_component_size': avg_component_size,
                'max_component_size': max_component_size
            }
        
        def calculate_similarity(mask1, mask2):
            """计算两个mask的相似度"""
            intersection = np.sum((mask1 > 0.5) & (mask2 > 0.5))
            union = np.sum((mask1 > 0.5) | (mask2 > 0.5))
            iou = intersection / union if union > 0 else 0
            
            # 计算Dice系数
            dice = 2 * intersection / (np.sum(mask1 > 0.5) + np.sum(mask2 > 0.5)) if (np.sum(mask1 > 0.5) + np.sum(mask2 > 0.5)) > 0 else 0
            
            return {'iou': iou, 'dice': dice}
        
        metrics = {
            'original_stats': calculate_mask_stats(original_mask),
            'text_stats': calculate_mask_stats(text_mask),
            'mixed_stats': calculate_mask_stats(mixed_mask),
            'ocr_stats': calculate_mask_stats(ocr_mask),
            'auto_stats': calculate_mask_stats(auto_mask),
            'similarities': {
                'text_vs_original': calculate_similarity(text_mask, original_mask),
                'mixed_vs_original': calculate_similarity(mixed_mask, original_mask),
                'auto_vs_original': calculate_similarity(auto_mask, original_mask),
                'text_vs_ocr': calculate_similarity(text_mask, ocr_mask),
                'auto_vs_ocr': calculate_similarity(auto_mask, ocr_mask)
            }
        }
        
        return metrics
    
    def _save_metrics_report(self, metrics, save_path):
        """
        保存指标报告
        """
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("文字水印检测对比报告\n")
            f.write("=" * 50 + "\n\n")
            
            # 各方法统计
            f.write("各方法检测统计:\n")
            f.write("-" * 30 + "\n")
            for method, stats in metrics.items():
                if method != 'similarities':
                    f.write(f"{method}:\n")
                    f.write(f"  覆盖率: {stats['coverage']:.4f}\n")
                    f.write(f"  连通组件数: {stats['num_components']}\n")
                    f.write(f"  平均组件大小: {stats['avg_component_size']:.2f}\n")
                    f.write(f"  最大组件大小: {stats['max_component_size']:.2f}\n\n")
            
            # 相似度分析
            f.write("相似度分析:\n")
            f.write("-" * 30 + "\n")
            for comparison, similarity in metrics['similarities'].items():
                f.write(f"{comparison}:\n")
                f.write(f"  IoU: {similarity['iou']:.4f}\n")
                f.write(f"  Dice: {similarity['dice']:.4f}\n\n")
            
            # 推荐
            f.write("推荐分析:\n")
            f.write("-" * 30 + "\n")
            
            # 找出最佳方法
            text_ocr_similarity = metrics['similarities']['text_vs_ocr']['dice']
            auto_ocr_similarity = metrics['similarities']['auto_vs_ocr']['dice']
            
            if text_ocr_similarity > 0.7:
                f.write("✓ 文字特定模式与OCR检测高度一致，推荐用于文字水印\n")
            elif auto_ocr_similarity > 0.7:
                f.write("✓ 智能自动模式与OCR检测高度一致，推荐用于混合场景\n")
            else:
                f.write("⚠ 各方法差异较大，建议进一步调优\n")
    
    def batch_test(self, input_dir, output_dir="batch_test_results"):
        """
        批量测试
        
        Args:
            input_dir: 输入图像目录
            output_dir: 输出目录
        """
        logger.info(f"开始批量测试: {input_dir}")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 支持的图像格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        # 收集所有图像文件
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(input_dir).glob(f"*{ext}"))
            image_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
        
        if not image_files:
            logger.warning(f"在 {input_dir} 中未找到图像文件")
            return
        
        logger.info(f"找到 {len(image_files)} 个图像文件")
        
        # 批量处理
        all_metrics = []
        for i, image_file in enumerate(image_files):
            logger.info(f"处理 {i+1}/{len(image_files)}: {image_file.name}")
            
            try:
                # 为每个图像创建子目录
                image_output_dir = os.path.join(output_dir, image_file.stem)
                metrics = self.test_single_image(
                    str(image_file), 
                    image_output_dir, 
                    show_steps=True
                )
                all_metrics.append({
                    'filename': image_file.name,
                    'metrics': metrics
                })
            except Exception as e:
                logger.error(f"处理 {image_file.name} 时出错: {e}")
        
        # 生成汇总报告
        self._generate_batch_summary(all_metrics, output_dir)
        
        logger.info(f"批量测试完成，结果保存到: {output_dir}")
    
    def _generate_batch_summary(self, all_metrics, output_dir):
        """
        生成批量测试汇总报告
        """
        summary_path = os.path.join(output_dir, "batch_summary.txt")
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("批量文字水印检测汇总报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"总测试图像数: {len(all_metrics)}\n\n")
            
            # 计算平均指标
            if all_metrics:
                avg_similarities = {}
                for comparison in ['text_vs_ocr', 'auto_vs_ocr', 'text_vs_original']:
                    dice_scores = [m['metrics']['similarities'][comparison]['dice'] for m in all_metrics]
                    iou_scores = [m['metrics']['similarities'][comparison]['iou'] for m in all_metrics]
                    
                    avg_similarities[comparison] = {
                        'avg_dice': np.mean(dice_scores),
                        'avg_iou': np.mean(iou_scores),
                        'std_dice': np.std(dice_scores),
                        'std_iou': np.std(iou_scores)
                    }
                
                f.write("平均相似度指标:\n")
                f.write("-" * 40 + "\n")
                for comparison, stats in avg_similarities.items():
                    f.write(f"{comparison}:\n")
                    f.write(f"  平均Dice: {stats['avg_dice']:.4f} ± {stats['std_dice']:.4f}\n")
                    f.write(f"  平均IoU: {stats['avg_iou']:.4f} ± {stats['std_iou']:.4f}\n\n")
                
                # 性能评估
                f.write("性能评估:\n")
                f.write("-" * 40 + "\n")
                
                text_ocr_dice = avg_similarities['text_vs_ocr']['avg_dice']
                auto_ocr_dice = avg_similarities['auto_vs_ocr']['avg_dice']
                
                if text_ocr_dice > 0.7:
                    f.write("✓ 文字特定模式表现优秀，适合文字水印检测\n")
                elif text_ocr_dice > 0.5:
                    f.write("○ 文字特定模式表现良好，可用于文字水印检测\n")
                else:
                    f.write("✗ 文字特定模式需要改进\n")
                
                if auto_ocr_dice > 0.7:
                    f.write("✓ 智能自动模式表现优秀，适合混合场景\n")
                elif auto_ocr_dice > 0.5:
                    f.write("○ 智能自动模式表现良好，可用于混合场景\n")
                else:
                    f.write("✗ 智能自动模式需要改进\n")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='测试文字水印检测效果')
    parser.add_argument('--input', type=str, required=True,
                       help='输入图像路径或目录')
    parser.add_argument('--output', type=str, default='test_results',
                       help='输出目录')
    parser.add_argument('--model', type=str, default=None,
                       help='模型路径')
    parser.add_argument('--device', type=str, default='auto',
                       help='设备类型')
    parser.add_argument('--batch', action='store_true',
                       help='批量测试模式')
    
    args = parser.parse_args()
    
    # 创建测试器
    tester = TextWatermarkTester(model_path=args.model, device=args.device)
    
    # 执行测试
    if args.batch or os.path.isdir(args.input):
        tester.batch_test(args.input, args.output)
    else:
        tester.test_single_image(args.input, args.output)

if __name__ == '__main__':
    main()