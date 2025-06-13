#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量修复优化器
提供更高效的批处理策略，避免CUDA卡住问题
"""

import os
import sys
import time
import psutil
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

# 添加src目录到Python路径
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from predict import WatermarkPredictor
from configs.config import get_cfg_defaults, update_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BatchRepairOptimizer:
    """
    批量修复优化器
    提供更高效的处理策略，避免内存泄漏和CUDA卡住
    """
    
    def __init__(self, model_path: str, config_path: str = None, device: str = 'cuda'):
        self.predictor = WatermarkPredictor(
            model_path=model_path,
            config_path=config_path,
            device=device
        )
        self.device = device
        self.stats = {
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'start_time': time.time()
        }
    
    def monitor_system_resources(self) -> Dict[str, float]:
        """监控系统资源使用情况"""
        import torch
        
        resources = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_available_gb': psutil.virtual_memory().available / 1024**3
        }
        
        if torch.cuda.is_available() and self.device == 'cuda':
            resources.update({
                'gpu_memory_allocated_gb': torch.cuda.memory_allocated() / 1024**3,
                'gpu_memory_reserved_gb': torch.cuda.memory_reserved() / 1024**3,
                'gpu_memory_free_gb': (torch.cuda.get_device_properties(0).total_memory - 
                                     torch.cuda.memory_reserved()) / 1024**3
            })
        
        return resources
    
    def should_pause_processing(self) -> bool:
        """判断是否应该暂停处理以释放资源"""
        resources = self.monitor_system_resources()
        
        # 如果GPU内存使用率过高，暂停处理
        if 'gpu_memory_allocated_gb' in resources:
            if resources['gpu_memory_allocated_gb'] > 18:  # 对于22GB显卡，超过18GB暂停
                return True
        
        # 如果系统内存使用率过高，暂停处理
        if resources['memory_percent'] > 85:
            return True
        
        return False
    
    def cleanup_resources(self):
        """清理系统资源"""
        import torch
        import gc
        
        # 清理Python垃圾回收
        gc.collect()
        
        # 清理CUDA缓存
        if torch.cuda.is_available() and self.device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        logger.info("资源清理完成")
    
    def process_batch_with_optimization(self, 
                                      input_path: str, 
                                      output_dir: str,
                                      max_iterations: int = 5,
                                      watermark_threshold: float = 0.01,
                                      iopaint_model: str = 'lama',
                                      limit: int = None,
                                      batch_size: int = 10,
                                      pause_interval: int = 50) -> List[Dict]:
        """
        优化的批量处理方法
        
        Args:
            input_path: 输入路径
            output_dir: 输出目录
            max_iterations: 最大迭代次数
            watermark_threshold: 水印阈值
            iopaint_model: IOPaint模型
            limit: 处理图片数量限制
            batch_size: 批处理大小
            pause_interval: 暂停间隔（处理多少张图片后暂停清理）
        """
        
        logger.info("开始优化批量处理...")
        logger.info(f"批处理大小: {batch_size}")
        logger.info(f"暂停间隔: {pause_interval}")
        
        # 获取图像列表
        image_paths = self._get_image_paths(input_path, limit)
        if not image_paths:
            logger.warning(f"在 {input_path} 中未找到图像文件")
            return []
        
        logger.info(f"找到 {len(image_paths)} 张图片待处理")
        
        # 分批处理
        all_results = []
        total_batches = (len(image_paths) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(image_paths))
            batch_paths = image_paths[start_idx:end_idx]
            
            logger.info(f"处理批次 {batch_idx + 1}/{total_batches} ({len(batch_paths)} 张图片)")
            
            # 监控资源使用情况
            resources = self.monitor_system_resources()
            logger.info(f"当前资源使用: CPU {resources['cpu_percent']:.1f}%, "
                       f"内存 {resources['memory_percent']:.1f}%")
            
            if 'gpu_memory_allocated_gb' in resources:
                logger.info(f"GPU内存: {resources['gpu_memory_allocated_gb']:.2f}GB / "
                           f"{resources['gpu_memory_reserved_gb']:.2f}GB")
            
            # 检查是否需要暂停
            if self.should_pause_processing():
                logger.warning("资源使用率过高，暂停处理并清理资源...")
                self.cleanup_resources()
                time.sleep(5)  # 等待5秒
            
            # 处理当前批次
            try:
                batch_results = self.predictor.process_batch_iterative(
                    input_path=batch_paths[0] if len(batch_paths) == 1 else os.path.dirname(batch_paths[0]),
                    output_dir=output_dir,
                    max_iterations=max_iterations,
                    watermark_threshold=watermark_threshold,
                    iopaint_model=iopaint_model,
                    limit=len(batch_paths)
                )
                
                all_results.extend(batch_results)
                self.stats['processed'] += len(batch_paths)
                self.stats['successful'] += len([r for r in batch_results if r.get('status') == 'success'])
                self.stats['failed'] += len([r for r in batch_results if r.get('status') == 'error'])
                
            except Exception as e:
                logger.error(f"批次 {batch_idx + 1} 处理失败: {str(e)}")
                self.stats['failed'] += len(batch_paths)
            
            # 定期清理资源
            if (batch_idx + 1) % (pause_interval // batch_size + 1) == 0:
                logger.info("定期清理资源...")
                self.cleanup_resources()
                time.sleep(2)  # 短暂休息
        
        # 最终清理
        self.cleanup_resources()
        
        # 打印统计信息
        elapsed_time = time.time() - self.stats['start_time']
        logger.info(f"\n批量处理完成！")
        logger.info(f"总处理时间: {elapsed_time:.1f}秒")
        logger.info(f"处理图片数: {self.stats['processed']}")
        logger.info(f"成功: {self.stats['successful']}")
        logger.info(f"失败: {self.stats['failed']}")
        logger.info(f"平均处理速度: {self.stats['processed'] / elapsed_time:.2f} 张/秒")
        
        return all_results
    
    def _get_image_paths(self, input_path: str, limit: int = None) -> List[str]:
        """获取图像路径列表"""
        image_paths = []
        
        if os.path.isdir(input_path):
            for filename in os.listdir(input_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    image_paths.append(os.path.join(input_path, filename))
        elif os.path.isfile(input_path):
            if input_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_paths.append(input_path)
        
        # 随机选择指定数量的图片
        if limit is not None and limit > 0 and len(image_paths) > limit:
            import random
            random.shuffle(image_paths)
            image_paths = image_paths[:limit]
        
        return image_paths


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="批量修复优化器")
    parser.add_argument('--input', required=True, help='输入路径')
    parser.add_argument('--output', required=True, help='输出路径')
    parser.add_argument('--model', required=True, help='模型路径')
    parser.add_argument('--config', help='配置文件路径')
    parser.add_argument('--device', default='cuda', help='设备类型')
    parser.add_argument('--max-iterations', type=int, default=5, help='最大迭代次数')
    parser.add_argument('--watermark-threshold', type=float, default=0.01, help='水印阈值')
    parser.add_argument('--iopaint-model', default='lama', help='IOPaint模型')
    parser.add_argument('--limit', type=int, help='处理图片数量限制')
    parser.add_argument('--batch-size', type=int, default=10, help='批处理大小')
    parser.add_argument('--pause-interval', type=int, default=50, help='暂停间隔')
    
    args = parser.parse_args()
    
    # 创建优化器
    optimizer = BatchRepairOptimizer(
        model_path=args.model,
        config_path=args.config,
        device=args.device
    )
    
    # 执行优化处理
    results = optimizer.process_batch_with_optimization(
        input_path=args.input,
        output_dir=args.output,
        max_iterations=args.max_iterations,
        watermark_threshold=args.watermark_threshold,
        iopaint_model=args.iopaint_model,
        limit=args.limit,
        batch_size=args.batch_size,
        pause_interval=args.pause_interval
    )
    
    # 保存结果
    import json
    summary_path = os.path.join(args.output, 'optimized_repair_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"处理结果已保存: {summary_path}")


if __name__ == '__main__':
    main()