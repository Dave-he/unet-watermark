#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化管理器
集成所有优化组件，提供统一的优化接口
"""

import os
import time
import logging
import threading
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Callable, List, Union, Tuple
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass

# 导入优化组件
from .enhanced_memory_manager import EnhancedMemoryManager, get_global_memory_manager
from .adaptive_batch_processor import AdaptiveBatchProcessor, BatchSizeOptimizer
from .optimized_dataloader import OptimizedDataLoader, DataLoaderConfig
from .optimized_predictor import OptimizedPredictor, PredictionConfig
from .training_optimizer import TrainingOptimizer, TrainingConfig
from .performance_analyzer import PerformanceAnalyzer, get_global_performance_analyzer
from .optimization_config import OptimizationConfig, ConfigManager, OptimizationLevel, get_global_config_manager

logger = logging.getLogger(__name__)

@dataclass
class OptimizationResult:
    """优化结果"""
    success: bool
    message: str
    metrics: Dict[str, Any]
    recommendations: List[str]
    execution_time: float
    memory_saved_mb: float = 0.0
    performance_improvement: float = 0.0

class OptimizationManager:
    """优化管理器主类"""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        # 配置管理
        self.config_manager = get_global_config_manager()
        self.config = config or OptimizationConfig()
        
        # 核心组件
        self.memory_manager: Optional[EnhancedMemoryManager] = None
        self.batch_processor: Optional[AdaptiveBatchProcessor] = None
        self.performance_analyzer: Optional[PerformanceAnalyzer] = None
        self.training_optimizer: Optional[TrainingOptimizer] = None
        
        # 状态管理
        self.is_initialized = False
        self.is_monitoring = False
        self.optimization_history: List[OptimizationResult] = []
        
        # 线程安全
        self.lock = threading.Lock()
        
        # 自动初始化
        if self.config.enable_optimization:
            self.initialize()
    
    def initialize(self) -> OptimizationResult:
        """初始化优化管理器"""
        start_time = time.time()
        
        try:
            with self.lock:
                if self.is_initialized:
                    return OptimizationResult(
                        success=True,
                        message="优化管理器已经初始化",
                        metrics={},
                        recommendations=[],
                        execution_time=0.0
                    )
                
                logger.info("初始化优化管理器...")
                
                # 初始化内存管理器
                if self.config.memory.enable_monitoring:
                    self.memory_manager = get_global_memory_manager()
                    self.memory_manager.configure(
                        cleanup_threshold=self.config.memory.cleanup_threshold,
                        aggressive_threshold=self.config.memory.aggressive_cleanup_threshold,
                        monitoring_interval=self.config.memory.monitoring_interval,
                        enable_oom_prevention=self.config.memory.enable_oom_prevention
                    )
                    
                    if self.config.memory.auto_cleanup:
                        self.memory_manager.start_monitoring()
                
                # 初始化批处理处理器
                if self.config.batch.enable_adaptive:
                    self.batch_processor = AdaptiveBatchProcessor(
                        initial_batch_size=self.config.batch.initial_batch_size,
                        min_batch_size=self.config.batch.min_batch_size,
                        max_batch_size=self.config.batch.max_batch_size,
                        memory_threshold=self.config.batch.memory_threshold,
                        adaptation_factor=self.config.batch.adaptation_factor
                    )
                
                # 初始化性能分析器
                if self.config.monitoring.enable_performance_monitoring:
                    self.performance_analyzer = get_global_performance_analyzer()
                    self.performance_analyzer.start_monitoring()
                    self.is_monitoring = True
                
                # 设置 PyTorch 优化
                self._setup_pytorch_optimizations()
                
                # 设置 CUDA 优化
                if torch.cuda.is_available():
                    self._setup_cuda_optimizations()
                
                self.is_initialized = True
                
                execution_time = time.time() - start_time
                
                result = OptimizationResult(
                    success=True,
                    message="优化管理器初始化成功",
                    metrics={
                        "initialization_time": execution_time,
                        "memory_monitoring": self.config.memory.enable_monitoring,
                        "batch_optimization": self.config.batch.enable_adaptive,
                        "performance_monitoring": self.config.monitoring.enable_performance_monitoring
                    },
                    recommendations=self._get_initialization_recommendations(),
                    execution_time=execution_time
                )
                
                self.optimization_history.append(result)
                logger.info(f"优化管理器初始化完成，耗时 {execution_time:.2f}s")
                
                return result
                
        except Exception as e:
            error_msg = f"优化管理器初始化失败: {str(e)}"
            logger.error(error_msg)
            
            return OptimizationResult(
                success=False,
                message=error_msg,
                metrics={},
                recommendations=["检查系统配置和依赖"],
                execution_time=time.time() - start_time
            )
    
    def _setup_pytorch_optimizations(self):
        """设置 PyTorch 优化"""
        # 设置确定性训练
        if self.config.training.deterministic:
            torch.manual_seed(self.config.training.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True
        
        # 设置内存格式
        if self.config.model.channels_last:
            torch.backends.cudnn.allow_tf32 = True
        
        # 设置混合精度
        if self.config.model.mixed_precision and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
    def _setup_cuda_optimizations(self):
        """设置 CUDA 优化"""
        # 设置 CUDA 内存分配
        if self.config.memory.cuda_memory_fraction:
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_per_process_memory_fraction(
                    self.config.memory.cuda_memory_fraction, i
                )
        
        # 启用 CUDA 缓存分配器优化
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    def _get_initialization_recommendations(self) -> List[str]:
        """获取初始化建议"""
        recommendations = []
        
        # 检查硬件配置
        device_info = self.config.get_device_info()
        
        if device_info['memory_gb'] < 8:
            recommendations.append("系统内存较少，建议减少批处理大小和工作进程数")
        
        if device_info['cuda_available']:
            for i, gpu in enumerate(device_info['cuda_devices']):
                if gpu['memory_gb'] < 6:
                    recommendations.append(f"GPU {i} 内存较少，建议启用梯度检查点和混合精度")
        
        if self.config.dataloader.num_workers > device_info['cpu_count']:
            recommendations.append("数据加载器工作进程数超过 CPU 核心数，可能影响性能")
        
        return recommendations
    
    def optimize_model(self, model: nn.Module) -> OptimizationResult:
        """优化模型"""
        start_time = time.time()
        
        try:
            logger.info("开始模型优化...")
            
            original_params = sum(p.numel() for p in model.parameters())
            
            # 应用内存格式优化
            if self.config.model.channels_last:
                model = model.to(memory_format=torch.channels_last)
            
            # 应用梯度检查点
            if self.config.model.gradient_checkpointing:
                if hasattr(model, 'gradient_checkpointing_enable'):
                    model.gradient_checkpointing_enable()
            
            # 应用 PyTorch 编译
            if self.config.model.torch_compile:
                try:
                    model = torch.compile(
                        model,
                        mode=self.config.model.compile_mode,
                        dynamic=self.config.model.compile_dynamic
                    )
                    logger.info("PyTorch 编译优化已应用")
                except Exception as e:
                    logger.warning(f"PyTorch 编译失败: {e}")
            
            # 移动到指定设备
            if self.config.model.device != "cpu":
                model = model.to(self.config.model.device)
            
            # 应用数据并行
            if self.config.model.data_parallel and torch.cuda.device_count() > 1:
                if self.config.model.device_ids:
                    model = nn.DataParallel(model, device_ids=self.config.model.device_ids)
                else:
                    model = nn.DataParallel(model)
                logger.info(f"数据并行已应用，使用 {torch.cuda.device_count()} 个 GPU")
            
            execution_time = time.time() - start_time
            
            result = OptimizationResult(
                success=True,
                message="模型优化完成",
                metrics={
                    "original_parameters": original_params,
                    "optimization_time": execution_time,
                    "channels_last": self.config.model.channels_last,
                    "gradient_checkpointing": self.config.model.gradient_checkpointing,
                    "torch_compile": self.config.model.torch_compile,
                    "data_parallel": self.config.model.data_parallel
                },
                recommendations=self._get_model_recommendations(model),
                execution_time=execution_time
            )
            
            self.optimization_history.append(result)
            logger.info(f"模型优化完成，耗时 {execution_time:.2f}s")
            
            return result
            
        except Exception as e:
            error_msg = f"模型优化失败: {str(e)}"
            logger.error(error_msg)
            
            return OptimizationResult(
                success=False,
                message=error_msg,
                metrics={},
                recommendations=["检查模型兼容性和设备配置"],
                execution_time=time.time() - start_time
            )
    
    def _get_model_recommendations(self, model: nn.Module) -> List[str]:
        """获取模型优化建议"""
        recommendations = []
        
        param_count = sum(p.numel() for p in model.parameters())
        param_size_mb = param_count * 4 / (1024 * 1024)  # 假设 float32
        
        if param_size_mb > 1000:  # 大于 1GB
            recommendations.append("模型较大，建议启用梯度检查点和混合精度训练")
        
        if not self.config.model.mixed_precision and torch.cuda.is_available():
            recommendations.append("建议启用混合精度训练以节省内存和提升速度")
        
        if not self.config.model.torch_compile:
            recommendations.append("建议启用 PyTorch 编译以提升推理速度")
        
        return recommendations
    
    def create_optimized_dataloader(self, dataset, **kwargs) -> OptimizedDataLoader:
        """创建优化的数据加载器"""
        # 合并配置
        dataloader_config = DataLoaderConfig(
            batch_size=kwargs.get('batch_size', self.config.batch.initial_batch_size),
            num_workers=kwargs.get('num_workers', self.config.dataloader.num_workers),
            pin_memory=kwargs.get('pin_memory', self.config.dataloader.pin_memory),
            persistent_workers=kwargs.get('persistent_workers', self.config.dataloader.persistent_workers),
            prefetch_factor=kwargs.get('prefetch_factor', self.config.dataloader.prefetch_factor),
            enable_caching=kwargs.get('enable_caching', self.config.dataloader.enable_caching),
            cache_size=kwargs.get('cache_size', self.config.dataloader.cache_size),
            enable_async_transform=kwargs.get('enable_async_transform', self.config.dataloader.enable_async_transform),
            enable_prefetch=kwargs.get('enable_prefetch', self.config.dataloader.enable_prefetch),
            auto_optimize=kwargs.get('auto_optimize', self.config.dataloader.auto_optimize)
        )
        
        return OptimizedDataLoader(dataset, dataloader_config)
    
    def create_optimized_predictor(self, model: nn.Module, **kwargs) -> OptimizedPredictor:
        """创建优化的预测器"""
        prediction_config = PredictionConfig(
            batch_size=kwargs.get('batch_size', self.config.batch.initial_batch_size),
            device=kwargs.get('device', self.config.model.device),
            mixed_precision=kwargs.get('mixed_precision', self.config.model.mixed_precision),
            torch_compile=kwargs.get('torch_compile', self.config.model.torch_compile),
            enable_memory_optimization=kwargs.get('enable_memory_optimization', True),
            enable_batch_optimization=kwargs.get('enable_batch_optimization', self.config.batch.enable_adaptive)
        )
        
        return OptimizedPredictor(model, prediction_config)
    
    def create_training_optimizer(self, model: nn.Module, **kwargs) -> TrainingOptimizer:
        """创建训练优化器"""
        if self.training_optimizer is None:
            training_config = TrainingConfig(
                epochs=kwargs.get('epochs', self.config.training.epochs),
                learning_rate=kwargs.get('learning_rate', self.config.training.learning_rate),
                batch_size=kwargs.get('batch_size', self.config.batch.initial_batch_size),
                optimizer=kwargs.get('optimizer', self.config.training.optimizer),
                scheduler=kwargs.get('scheduler', self.config.training.scheduler),
                mixed_precision=kwargs.get('mixed_precision', self.config.model.mixed_precision),
                gradient_checkpointing=kwargs.get('gradient_checkpointing', self.config.model.gradient_checkpointing),
                early_stopping=kwargs.get('early_stopping', self.config.training.early_stopping),
                deterministic=kwargs.get('deterministic', self.config.training.deterministic)
            )
            
            self.training_optimizer = TrainingOptimizer(model, training_config)
        
        return self.training_optimizer
    
    @contextmanager
    def optimization_context(self, operation_name: str = "operation", **kwargs):
        """优化上下文管理器"""
        # 开始性能分析
        operation_id = None
        if self.performance_analyzer:
            operation_id = self.performance_analyzer.profiler.start_operation(
                operation_name, 
                kwargs.get('items_count', 1),
                **kwargs
            )
        
        # 内存管理
        memory_context = None
        if self.memory_manager:
            memory_context = self.memory_manager.memory_context()
            memory_context.__enter__()
        
        try:
            yield
            
            # 成功完成
            if operation_id and self.performance_analyzer:
                self.performance_analyzer.profiler.end_operation(operation_id, success=True)
                
        except Exception as e:
            # 处理异常
            if operation_id and self.performance_analyzer:
                self.performance_analyzer.profiler.end_operation(
                    operation_id, success=False, error_message=str(e)
                )
            raise
            
        finally:
            # 清理内存
            if memory_context:
                memory_context.__exit__(None, None, None)
    
    def optimize_batch_size(self, model: nn.Module, sample_input, target_metric: str = "throughput") -> int:
        """优化批处理大小"""
        if not self.batch_processor:
            logger.warning("批处理处理器未初始化")
            return self.config.batch.initial_batch_size
        
        optimizer = BatchSizeOptimizer(
            min_batch_size=self.config.batch.min_batch_size,
            max_batch_size=self.config.batch.max_batch_size,
            memory_threshold=self.config.batch.memory_threshold
        )
        
        def test_function(batch_size):
            # 创建测试批次
            if isinstance(sample_input, torch.Tensor):
                test_input = sample_input[:batch_size]
            else:
                test_input = [sample_input[0]] * batch_size
            
            # 测试推理
            start_time = time.time()
            with torch.no_grad():
                _ = model(test_input)
            end_time = time.time()
            
            return {
                "throughput": batch_size / (end_time - start_time),
                "latency": end_time - start_time,
                "memory_usage": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            }
        
        optimal_batch_size = optimizer.find_optimal_batch_size(test_function, target_metric)
        
        logger.info(f"优化批处理大小: {optimal_batch_size}")
        return optimal_batch_size
    
    def generate_performance_report(self, output_dir: Optional[str] = None) -> str:
        """生成性能报告"""
        if not self.performance_analyzer:
            logger.warning("性能分析器未初始化")
            return ""
        
        if output_dir is None:
            output_dir = self.config.output_dir
        
        return self.performance_analyzer.export_performance_report(output_dir, include_plots=True)
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """获取优化摘要"""
        summary = {
            "config": {
                "optimization_level": self.config.optimization_level.value,
                "device": self.config.model.device,
                "mixed_precision": self.config.model.mixed_precision,
                "batch_size_range": f"{self.config.batch.min_batch_size}-{self.config.batch.max_batch_size}",
                "num_workers": self.config.dataloader.num_workers
            },
            "status": {
                "initialized": self.is_initialized,
                "monitoring": self.is_monitoring,
                "memory_manager_active": self.memory_manager is not None and self.memory_manager.is_monitoring,
                "optimization_count": len(self.optimization_history)
            },
            "hardware": self.config.get_device_info()
        }
        
        # 添加性能统计
        if self.performance_analyzer:
            system_analysis = self.performance_analyzer.analyze_system_performance(5)
            if 'error' not in system_analysis:
                summary['performance'] = {
                    "avg_cpu_usage": system_analysis['cpu']['avg_usage'],
                    "avg_memory_usage": system_analysis['memory']['avg_usage_percent'],
                    "process_memory_mb": system_analysis['process']['avg_memory_mb']
                }
                
                if 'gpu' in system_analysis:
                    summary['performance']['avg_gpu_usage'] = system_analysis['gpu']['avg_utilization']
                    summary['performance']['avg_gpu_memory_usage'] = system_analysis['gpu']['avg_memory_usage_percent']
        
        # 添加内存统计
        if self.memory_manager:
            memory_stats = self.memory_manager.get_stats()
            summary['memory'] = {
                "cleanup_count": memory_stats.get('cleanup_count', 0),
                "total_freed_mb": memory_stats.get('total_freed_mb', 0),
                "avg_cleanup_time": memory_stats.get('avg_cleanup_time', 0)
            }
        
        return summary
    
    def get_recommendations(self) -> List[str]:
        """获取优化建议"""
        recommendations = []
        
        # 从性能分析器获取建议
        if self.performance_analyzer:
            recommendations.extend(self.performance_analyzer.generate_optimization_recommendations())
        
        # 从历史记录获取建议
        for result in self.optimization_history:
            recommendations.extend(result.recommendations)
        
        # 去重并返回
        return list(set(recommendations))
    
    def cleanup(self):
        """清理资源"""
        logger.info("清理优化管理器资源...")
        
        # 停止监控
        if self.memory_manager:
            self.memory_manager.stop_monitoring()
        
        if self.performance_analyzer:
            self.performance_analyzer.stop_monitoring()
        
        # 清理训练优化器
        if self.training_optimizer:
            self.training_optimizer.cleanup()
        
        # 重置状态
        self.is_initialized = False
        self.is_monitoring = False
        
        logger.info("优化管理器资源清理完成")
    
    def __enter__(self):
        """上下文管理器入口"""
        if not self.is_initialized:
            self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.cleanup()

# 全局优化管理器
_global_optimization_manager = None

def get_global_optimization_manager(config: Optional[OptimizationConfig] = None) -> OptimizationManager:
    """获取全局优化管理器"""
    global _global_optimization_manager
    if _global_optimization_manager is None:
        _global_optimization_manager = OptimizationManager(config)
    return _global_optimization_manager

def create_optimization_manager(optimization_level: OptimizationLevel = OptimizationLevel.BALANCED) -> OptimizationManager:
    """创建优化管理器"""
    config = OptimizationConfig(optimization_level=optimization_level)
    return OptimizationManager(config)

# 便捷函数
def optimize_model(model: nn.Module, optimization_level: OptimizationLevel = OptimizationLevel.BALANCED) -> Tuple[nn.Module, OptimizationManager]:
    """快速优化模型"""
    manager = create_optimization_manager(optimization_level)
    result = manager.optimize_model(model)
    
    if result.success:
        logger.info(f"模型优化成功: {result.message}")
    else:
        logger.error(f"模型优化失败: {result.message}")
    
    return model, manager

def optimize_training(model: nn.Module, train_dataset, val_dataset=None, optimization_level: OptimizationLevel = OptimizationLevel.BALANCED, **kwargs) -> TrainingOptimizer:
    """快速优化训练"""
    manager = create_optimization_manager(optimization_level)
    
    # 优化模型
    manager.optimize_model(model)
    
    # 创建优化的数据加载器
    train_dataloader = manager.create_optimized_dataloader(train_dataset)
    val_dataloader = manager.create_optimized_dataloader(val_dataset) if val_dataset else None
    
    # 创建训练优化器
    training_optimizer = manager.create_training_optimizer(model, **kwargs)
    
    return training_optimizer

if __name__ == "__main__":
    # 测试代码
    import torch.nn as nn
    
    print("创建测试模型...")
    model = nn.Sequential(
        nn.Linear(100, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    
    print("创建优化管理器...")
    manager = create_optimization_manager(OptimizationLevel.BALANCED)
    
    print("\n优化前配置:")
    manager.config.print_summary()
    
    print("\n优化模型...")
    result = manager.optimize_model(model)
    print(f"优化结果: {result.message}")
    print(f"执行时间: {result.execution_time:.2f}s")
    
    print("\n优化摘要:")
    summary = manager.get_optimization_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print("\n优化建议:")
    recommendations = manager.get_recommendations()
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    
    print("\n生成性能报告...")
    report_path = manager.generate_performance_report("./test_optimization_report")
    if report_path:
        print(f"报告已生成: {report_path}")
    
    print("\n清理资源...")
    manager.cleanup()
    
    print("测试完成")