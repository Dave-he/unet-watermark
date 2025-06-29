#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化的预测器
整合内存管理、批处理优化和高效推理
"""

import os
import time
import torch
import logging
import numpy as np
from typing import List, Dict, Optional, Callable, Any, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
import cv2

from .enhanced_memory_manager import EnhancedMemoryManager, get_global_memory_manager, memory_optimized
from .adaptive_batch_processor import AdaptiveBatchProcessor, BatchSizeOptimizer
from .optimized_dataloader import OptimizedDataLoader, DataLoaderConfig, create_optimized_dataloader

logger = logging.getLogger(__name__)

@dataclass
class PredictionConfig:
    """预测配置"""
    # 批处理配置
    initial_batch_size: int = 8
    min_batch_size: int = 1
    max_batch_size: int = 32
    
    # 内存管理
    memory_threshold: float = 0.8
    enable_memory_optimization: bool = True
    cleanup_frequency: int = 10  # 每N个批次清理一次内存
    
    # 模型优化
    enable_mixed_precision: bool = True
    enable_torch_compile: bool = False  # PyTorch 2.0+
    enable_tensorrt: bool = False
    
    # 数据加载优化
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    enable_caching: bool = True
    cache_size: int = 1000
    
    # 输出配置
    output_format: str = 'numpy'  # 'numpy', 'tensor', 'pil'
    save_intermediate: bool = False
    intermediate_dir: Optional[str] = None

class OptimizedPredictor:
    """优化的预测器"""
    
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device = None,
                 config: Optional[PredictionConfig] = None):
        """
        初始化优化预测器
        
        Args:
            model: 预测模型
            device: 计算设备
            config: 预测配置
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config or PredictionConfig()
        
        # 内存管理器
        self.memory_manager = get_global_memory_manager()
        
        # 自适应批处理器
        self.batch_processor = AdaptiveBatchProcessor(
            initial_batch_size=self.config.initial_batch_size,
            min_batch_size=self.config.min_batch_size,
            max_batch_size=self.config.max_batch_size,
            memory_threshold=self.config.memory_threshold,
            memory_manager=self.memory_manager
        )
        
        # 批处理大小优化器
        self.batch_optimizer = BatchSizeOptimizer(self.device)
        
        # 模型优化
        self._optimize_model()
        
        # 统计信息
        self.prediction_count = 0
        self.total_time = 0.0
        self.optimization_count = 0
        
        # 创建中间结果目录
        if self.config.save_intermediate and self.config.intermediate_dir:
            os.makedirs(self.config.intermediate_dir, exist_ok=True)
        
        logger.info(f"优化预测器初始化完成: device={self.device}, "
                   f"batch_size=[{self.config.min_batch_size}, {self.config.max_batch_size}]")
    
    def _optimize_model(self):
        """优化模型"""
        # 移动到设备
        self.model = self.model.to(self.device)
        
        # 设置评估模式
        self.model.eval()
        
        # 混合精度
        if self.config.enable_mixed_precision and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("启用混合精度推理")
        else:
            self.scaler = None
        
        # PyTorch 编译优化 (PyTorch 2.0+)
        if self.config.enable_torch_compile and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model)
                logger.info("启用 PyTorch 编译优化")
            except Exception as e:
                logger.warning(f"PyTorch 编译优化失败: {e}")
        
        # TensorRT 优化
        if self.config.enable_tensorrt:
            try:
                import torch_tensorrt
                # 这里需要根据具体模型进行 TensorRT 优化
                logger.info("TensorRT 优化可用但需要具体实现")
            except ImportError:
                logger.warning("TensorRT 不可用")
    
    @memory_optimized
    def predict_single(self, input_data: Union[np.ndarray, torch.Tensor, Image.Image]) -> Any:
        """单个样本预测"""
        start_time = time.time()
        
        # 预处理输入
        processed_input = self._preprocess_input(input_data)
        
        # 推理
        with torch.no_grad():
            if self.scaler and self.config.enable_mixed_precision:
                with torch.cuda.amp.autocast():
                    output = self.model(processed_input)
            else:
                output = self.model(processed_input)
        
        # 后处理输出
        result = self._postprocess_output(output)
        
        # 更新统计
        self.prediction_count += 1
        self.total_time += time.time() - start_time
        
        return result
    
    def predict_batch(self, input_batch: List[Any]) -> List[Any]:
        """批量预测"""
        if not input_batch:
            return []
        
        # 使用自适应批处理器
        results = self.batch_processor.process_batch(
            input_batch, 
            self._process_batch_impl
        )
        
        # 定期清理内存
        if self.prediction_count % self.config.cleanup_frequency == 0:
            self.memory_manager.auto_cleanup()
        
        return results
    
    def predict_all(self, 
                   input_data: List[Any],
                   progress_callback: Optional[Callable] = None,
                   save_results: bool = False,
                   output_dir: Optional[str] = None) -> List[Any]:
        """预测所有数据"""
        logger.info(f"开始批量预测: {len(input_data)} 个样本")
        
        start_time = time.time()
        
        # 使用自适应批处理器处理所有数据
        results = self.batch_processor.process_all(
            input_data,
            self._process_batch_impl,
            progress_callback=progress_callback
        )
        
        total_time = time.time() - start_time
        
        # 保存结果
        if save_results and output_dir:
            self._save_results(results, output_dir)
        
        # 记录统计信息
        logger.info(f"批量预测完成: {len(results)} 个结果, "
                   f"耗时: {total_time:.2f}s, "
                   f"平均速度: {len(results)/total_time:.2f} samples/s")
        
        return results
    
    def predict_from_paths(self,
                          image_paths: List[str],
                          transform: Optional[Callable] = None,
                          progress_callback: Optional[Callable] = None,
                          save_results: bool = False,
                          output_dir: Optional[str] = None) -> List[Any]:
        """从文件路径预测"""
        logger.info(f"从路径预测: {len(image_paths)} 个文件")
        
        # 创建数据加载器配置
        dataloader_config = DataLoaderConfig(
            batch_size=self.batch_processor.current_batch_size,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            prefetch_factor=self.config.prefetch_factor,
            enable_caching=self.config.enable_caching,
            cache_size=self.config.cache_size
        )
        
        # 创建优化数据加载器
        dataloader = create_optimized_dataloader(
            data_paths=image_paths,
            transform=transform,
            device=self.device,
            config=dataloader_config
        )
        
        results = []
        processed_count = 0
        
        try:
            for batch in dataloader:
                # 批量推理
                batch_results = self._process_batch_impl(batch)
                results.extend(batch_results)
                
                # 更新进度
                processed_count += len(batch_results)
                if progress_callback:
                    progress_callback(processed_count, len(image_paths), 
                                    self.batch_processor.current_batch_size)
                
                # 定期清理内存
                if processed_count % (self.config.cleanup_frequency * self.batch_processor.current_batch_size) == 0:
                    self.memory_manager.auto_cleanup()
        
        finally:
            # 清理数据加载器
            dataloader.cleanup()
        
        # 保存结果
        if save_results and output_dir:
            self._save_results(results, output_dir, image_paths)
        
        return results
    
    def _process_batch_impl(self, batch_data: List[Any]) -> List[Any]:
        """批处理实现"""
        if not batch_data:
            return []
        
        # 预处理批次数据
        processed_batch = []
        for item in batch_data:
            processed_item = self._preprocess_input(item)
            processed_batch.append(processed_item)
        
        # 堆叠为批次张量
        if processed_batch:
            batch_tensor = torch.stack(processed_batch)
        else:
            return []
        
        # 批量推理
        with torch.no_grad():
            if self.scaler and self.config.enable_mixed_precision:
                with torch.cuda.amp.autocast():
                    batch_output = self.model(batch_tensor)
            else:
                batch_output = self.model(batch_tensor)
        
        # 后处理批次输出
        results = []
        for i in range(batch_output.shape[0]):
            output_item = batch_output[i:i+1]  # 保持批次维度
            result = self._postprocess_output(output_item)
            results.append(result)
        
        return results
    
    def _preprocess_input(self, input_data: Union[np.ndarray, torch.Tensor, Image.Image, str]) -> torch.Tensor:
        """预处理输入数据"""
        if isinstance(input_data, str):
            # 文件路径
            if os.path.exists(input_data):
                input_data = Image.open(input_data).convert('RGB')
            else:
                raise ValueError(f"文件不存在: {input_data}")
        
        if isinstance(input_data, Image.Image):
            # PIL 图像
            input_data = np.array(input_data)
        
        if isinstance(input_data, np.ndarray):
            # NumPy 数组
            if input_data.dtype != np.float32:
                input_data = input_data.astype(np.float32)
            
            # 归一化到 [0, 1]
            if input_data.max() > 1.0:
                input_data = input_data / 255.0
            
            # 转换为张量
            input_data = torch.from_numpy(input_data)
        
        if isinstance(input_data, torch.Tensor):
            # 确保正确的维度顺序 (C, H, W)
            if input_data.dim() == 3 and input_data.shape[-1] in [1, 3, 4]:
                input_data = input_data.permute(2, 0, 1)
            
            # 添加批次维度
            if input_data.dim() == 3:
                input_data = input_data.unsqueeze(0)
            
            # 移动到设备
            input_data = input_data.to(self.device, non_blocking=True)
        
        return input_data
    
    def _postprocess_output(self, output: torch.Tensor) -> Any:
        """后处理输出"""
        # 移动到 CPU
        if output.is_cuda:
            output = output.cpu()
        
        # 移除批次维度
        if output.dim() == 4 and output.shape[0] == 1:
            output = output.squeeze(0)
        
        # 根据配置转换输出格式
        if self.config.output_format == 'numpy':
            result = output.numpy()
            
            # 如果是图像数据，转换维度顺序
            if result.ndim == 3 and result.shape[0] in [1, 3, 4]:
                result = result.transpose(1, 2, 0)
            
            # 转换到 [0, 255] 范围
            if result.max() <= 1.0:
                result = (result * 255).astype(np.uint8)
            
            return result
            
        elif self.config.output_format == 'pil':
            result = output.numpy()
            
            if result.ndim == 3 and result.shape[0] in [1, 3, 4]:
                result = result.transpose(1, 2, 0)
            
            if result.max() <= 1.0:
                result = (result * 255).astype(np.uint8)
            
            if result.ndim == 3 and result.shape[2] == 1:
                result = result.squeeze(2)
            
            return Image.fromarray(result)
            
        else:  # tensor
            return output
    
    def _save_results(self, 
                     results: List[Any], 
                     output_dir: str,
                     input_paths: Optional[List[str]] = None):
        """保存结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        for i, result in enumerate(results):
            # 确定输出文件名
            if input_paths and i < len(input_paths):
                input_path = Path(input_paths[i])
                output_name = f"{input_path.stem}_result{input_path.suffix}"
            else:
                output_name = f"result_{i:06d}.png"
            
            output_path = os.path.join(output_dir, output_name)
            
            # 保存结果
            if isinstance(result, Image.Image):
                result.save(output_path)
            elif isinstance(result, np.ndarray):
                if result.ndim == 2 or (result.ndim == 3 and result.shape[2] in [1, 3, 4]):
                    cv2.imwrite(output_path, result)
                else:
                    np.save(output_path.replace('.png', '.npy'), result)
            elif isinstance(result, torch.Tensor):
                torch.save(result, output_path.replace('.png', '.pt'))
        
        logger.info(f"结果已保存到: {output_dir}")
    
    def optimize_batch_size(self, test_data: List[Any]) -> int:
        """优化批处理大小"""
        logger.info("开始优化批处理大小")
        
        optimal_size = self.batch_optimizer.find_optimal_batch_size(
            test_func=lambda batch: self._process_batch_impl(batch),
            test_data=test_data,
            max_batch_size=self.config.max_batch_size,
            min_batch_size=self.config.min_batch_size
        )
        
        # 更新批处理器配置
        self.batch_processor.current_batch_size = optimal_size
        self.optimization_count += 1
        
        logger.info(f"批处理大小优化完成: {optimal_size}")
        return optimal_size
    
    def get_performance_stats(self) -> Dict:
        """获取性能统计"""
        stats = {
            'prediction_count': self.prediction_count,
            'total_time': self.total_time,
            'avg_prediction_time': self.total_time / self.prediction_count if self.prediction_count > 0 else 0,
            'predictions_per_second': self.prediction_count / self.total_time if self.total_time > 0 else 0,
            'optimization_count': self.optimization_count,
            'current_batch_size': self.batch_processor.current_batch_size,
            'device': str(self.device),
            'mixed_precision': self.config.enable_mixed_precision
        }
        
        # 添加批处理器统计
        batch_stats = self.batch_processor.get_performance_stats()
        stats['batch_processor'] = batch_stats
        
        # 添加内存统计
        memory_stats = self.memory_manager.get_stats()
        stats['memory'] = memory_stats
        
        return stats
    
    def reset_stats(self):
        """重置统计信息"""
        self.prediction_count = 0
        self.total_time = 0.0
        self.optimization_count = 0
        self.batch_processor.reset_stats()
        self.memory_manager.reset_stats()
        
        logger.info("性能统计已重置")
    
    def cleanup(self):
        """清理资源"""
        # 清理内存
        self.memory_manager.aggressive_cleanup()
        
        # 清理模型
        if hasattr(self.model, 'cpu'):
            self.model.cpu()
        
        logger.info("预测器资源清理完成")

def create_optimized_predictor(model: torch.nn.Module,
                             device: torch.device = None,
                             config: Optional[PredictionConfig] = None) -> OptimizedPredictor:
    """创建优化预测器"""
    return OptimizedPredictor(model, device, config)

if __name__ == "__main__":
    # 测试代码
    import tempfile
    from torchvision import models
    
    # 创建测试模型
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)  # 单输出
    
    # 创建配置
    config = PredictionConfig(
        initial_batch_size=4,
        max_batch_size=16,
        enable_mixed_precision=True,
        output_format='numpy'
    )
    
    # 创建预测器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictor = create_optimized_predictor(model, device, config)
    
    # 创建测试数据
    test_data = []
    for i in range(50):
        # 创建随机图像数据
        img_data = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        test_data.append(img_data)
    
    # 优化批处理大小
    optimal_batch_size = predictor.optimize_batch_size(test_data[:10])
    print(f"最优批处理大小: {optimal_batch_size}")
    
    # 批量预测
    results = predictor.predict_all(test_data)
    print(f"预测完成: {len(results)} 个结果")
    
    # 获取性能统计
    stats = predictor.get_performance_stats()
    print(f"性能统计: {stats}")
    
    # 清理
    predictor.cleanup()
    
    print("测试完成")