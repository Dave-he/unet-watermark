#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自适应批处理优化器
根据系统资源和处理性能动态调整批处理大小
"""

import time
import torch
import logging
import numpy as np
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass
from collections import deque

from .enhanced_memory_manager import EnhancedMemoryManager, get_global_memory_manager

logger = logging.getLogger(__name__)

@dataclass
class BatchMetrics:
    """批处理指标"""
    batch_size: int
    processing_time: float
    memory_usage_gb: float
    success: bool
    throughput: float  # 每秒处理的样本数
    timestamp: float

class AdaptiveBatchProcessor:
    """自适应批处理器"""
    
    def __init__(self,
                 initial_batch_size: int = 8,
                 min_batch_size: int = 1,
                 max_batch_size: int = 32,
                 memory_threshold: float = 0.8,
                 adaptation_window: int = 10,
                 memory_manager: Optional[EnhancedMemoryManager] = None):
        """
        初始化自适应批处理器
        
        Args:
            initial_batch_size: 初始批处理大小
            min_batch_size: 最小批处理大小
            max_batch_size: 最大批处理大小
            memory_threshold: 内存使用阈值
            adaptation_window: 适应窗口大小
            memory_manager: 内存管理器
        """
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.memory_threshold = memory_threshold
        self.adaptation_window = adaptation_window
        
        # 内存管理器
        self.memory_manager = memory_manager or get_global_memory_manager()
        
        # 性能历史记录
        self.metrics_history: deque = deque(maxlen=adaptation_window)
        self.failure_count = 0
        self.success_count = 0
        
        # 适应策略参数
        self.increase_threshold = 0.9  # 成功率阈值，超过此值可以增加批处理大小
        self.decrease_threshold = 0.7  # 成功率阈值，低于此值需要减少批处理大小
        self.memory_safety_margin = 0.1  # 内存安全边际
        
        # 性能统计
        self.total_processed = 0
        self.total_time = 0.0
        self.adaptation_count = 0
        
        logger.info(f"自适应批处理器初始化: batch_size={initial_batch_size}, "
                   f"range=[{min_batch_size}, {max_batch_size}]")
    
    def process_batch(self, 
                     data: List[Any], 
                     process_func: Callable,
                     **kwargs) -> List[Any]:
        """处理批次数据"""
        start_time = time.time()
        batch_size = min(self.current_batch_size, len(data))
        
        try:
            # 检查内存是否足够
            if not self._check_memory_availability(batch_size):
                # 内存不足，减少批处理大小
                batch_size = max(self.min_batch_size, batch_size // 2)
                self.current_batch_size = batch_size
                logger.warning(f"内存不足，调整批处理大小为 {batch_size}")
            
            # 处理数据
            batch_data = data[:batch_size]
            results = process_func(batch_data, **kwargs)
            
            # 记录成功指标
            processing_time = time.time() - start_time
            memory_usage = self._get_current_memory_usage()
            throughput = batch_size / processing_time if processing_time > 0 else 0
            
            metrics = BatchMetrics(
                batch_size=batch_size,
                processing_time=processing_time,
                memory_usage_gb=memory_usage,
                success=True,
                throughput=throughput,
                timestamp=time.time()
            )
            
            self._record_metrics(metrics)
            self.success_count += 1
            self.total_processed += batch_size
            self.total_time += processing_time
            
            # 适应性调整
            self._adapt_batch_size()
            
            return results
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # OOM错误处理
                self._handle_oom_error()
                
                # 递归重试，使用更小的批处理大小
                if batch_size > self.min_batch_size:
                    logger.warning(f"OOM错误，重试使用批处理大小 {self.current_batch_size}")
                    return self.process_batch(data, process_func, **kwargs)
                else:
                    raise RuntimeError(f"即使使用最小批处理大小 {self.min_batch_size} 仍然OOM")
            else:
                raise e
        
        except Exception as e:
            # 其他错误
            processing_time = time.time() - start_time
            memory_usage = self._get_current_memory_usage()
            
            metrics = BatchMetrics(
                batch_size=batch_size,
                processing_time=processing_time,
                memory_usage_gb=memory_usage,
                success=False,
                throughput=0,
                timestamp=time.time()
            )
            
            self._record_metrics(metrics)
            self.failure_count += 1
            
            raise e
    
    def process_all(self, 
                   data: List[Any], 
                   process_func: Callable,
                   progress_callback: Optional[Callable] = None,
                   **kwargs) -> List[Any]:
        """处理所有数据"""
        all_results = []
        processed_count = 0
        total_count = len(data)
        
        logger.info(f"开始处理 {total_count} 个样本，初始批处理大小: {self.current_batch_size}")
        
        while processed_count < total_count:
            # 获取当前批次数据
            remaining_data = data[processed_count:]
            
            # 处理当前批次
            batch_results = self.process_batch(remaining_data, process_func, **kwargs)
            all_results.extend(batch_results)
            
            # 更新进度
            processed_count += len(batch_results)
            
            # 调用进度回调
            if progress_callback:
                progress_callback(processed_count, total_count, self.current_batch_size)
            
            # 定期清理内存
            if processed_count % (self.current_batch_size * 4) == 0:
                self.memory_manager.auto_cleanup()
        
        logger.info(f"处理完成: {processed_count}/{total_count} 样本")
        return all_results
    
    def _check_memory_availability(self, batch_size: int) -> bool:
        """检查内存可用性"""
        snapshot = self.memory_manager.get_memory_snapshot()
        
        # 估算所需内存（基于历史数据）
        estimated_memory = self._estimate_memory_requirement(batch_size)
        
        # 检查GPU内存
        if torch.cuda.is_available():
            available_gpu_memory = snapshot.gpu_free_gb - self.memory_safety_margin
            if estimated_memory > available_gpu_memory:
                return False
        
        # 检查RAM
        available_ram = snapshot.ram_available_gb - self.memory_safety_margin
        if estimated_memory > available_ram:
            return False
        
        return True
    
    def _estimate_memory_requirement(self, batch_size: int) -> float:
        """估算内存需求"""
        if not self.metrics_history:
            # 没有历史数据，使用经验值
            return batch_size * 0.1  # 假设每个样本需要100MB
        
        # 基于历史数据估算
        recent_metrics = list(self.metrics_history)[-5:]  # 最近5次记录
        memory_per_sample = []
        
        for metrics in recent_metrics:
            if metrics.batch_size > 0:
                memory_per_sample.append(metrics.memory_usage_gb / metrics.batch_size)
        
        if memory_per_sample:
            avg_memory_per_sample = np.mean(memory_per_sample)
            return batch_size * avg_memory_per_sample * 1.2  # 增加20%安全边际
        else:
            return batch_size * 0.1
    
    def _get_current_memory_usage(self) -> float:
        """获取当前内存使用量"""
        snapshot = self.memory_manager.get_memory_snapshot()
        if torch.cuda.is_available():
            return snapshot.gpu_allocated_gb
        else:
            return snapshot.ram_used_gb
    
    def _record_metrics(self, metrics: BatchMetrics):
        """记录性能指标"""
        self.metrics_history.append(metrics)
        
        # 记录详细日志
        logger.debug(f"批处理指标: size={metrics.batch_size}, "
                    f"time={metrics.processing_time:.3f}s, "
                    f"memory={metrics.memory_usage_gb:.2f}GB, "
                    f"throughput={metrics.throughput:.1f} samples/s, "
                    f"success={metrics.success}")
    
    def _adapt_batch_size(self):
        """适应性调整批处理大小"""
        if len(self.metrics_history) < 3:  # 需要足够的历史数据
            return
        
        recent_metrics = list(self.metrics_history)[-5:]  # 最近5次记录
        success_rate = sum(1 for m in recent_metrics if m.success) / len(recent_metrics)
        avg_memory_usage = np.mean([m.memory_usage_gb for m in recent_metrics])
        avg_throughput = np.mean([m.throughput for m in recent_metrics if m.success])
        
        old_batch_size = self.current_batch_size
        
        # 决策逻辑
        if success_rate >= self.increase_threshold:
            # 成功率高，可以尝试增加批处理大小
            snapshot = self.memory_manager.get_memory_snapshot()
            memory_usage_ratio = avg_memory_usage / (snapshot.gpu_total_gb if torch.cuda.is_available() else snapshot.ram_used_gb + snapshot.ram_available_gb)
            
            if memory_usage_ratio < self.memory_threshold and self.current_batch_size < self.max_batch_size:
                # 内存使用率不高，可以增加
                self.current_batch_size = min(self.max_batch_size, int(self.current_batch_size * 1.2))
                
        elif success_rate < self.decrease_threshold:
            # 成功率低，需要减少批处理大小
            self.current_batch_size = max(self.min_batch_size, int(self.current_batch_size * 0.8))
        
        # 基于内存压力调整
        pressure = self.memory_manager.check_memory_pressure()
        if pressure['gpu_high'] or pressure['ram_high']:
            self.current_batch_size = max(self.min_batch_size, int(self.current_batch_size * 0.7))
        
        # 记录调整
        if self.current_batch_size != old_batch_size:
            self.adaptation_count += 1
            logger.info(f"批处理大小调整: {old_batch_size} -> {self.current_batch_size} "
                       f"(成功率: {success_rate:.2f}, 内存使用: {avg_memory_usage:.2f}GB, "
                       f"吞吐量: {avg_throughput:.1f} samples/s)")
    
    def _handle_oom_error(self):
        """处理OOM错误"""
        logger.warning("检测到OOM错误，执行紧急内存清理")
        
        # 激进清理内存
        self.memory_manager.aggressive_cleanup()
        
        # 大幅减少批处理大小
        self.current_batch_size = max(self.min_batch_size, self.current_batch_size // 2)
        
        # 记录失败
        self.failure_count += 1
        
        logger.info(f"OOM恢复: 批处理大小调整为 {self.current_batch_size}")
    
    def get_optimal_batch_size(self) -> int:
        """获取最优批处理大小"""
        if len(self.metrics_history) < 3:
            return self.current_batch_size
        
        # 分析历史数据找到最优批处理大小
        successful_metrics = [m for m in self.metrics_history if m.success]
        
        if not successful_metrics:
            return self.min_batch_size
        
        # 按吞吐量排序
        successful_metrics.sort(key=lambda x: x.throughput, reverse=True)
        
        # 返回吞吐量最高的批处理大小
        return successful_metrics[0].batch_size
    
    def get_performance_stats(self) -> Dict:
        """获取性能统计"""
        if not self.metrics_history:
            return {}
        
        successful_metrics = [m for m in self.metrics_history if m.success]
        
        stats = {
            'current_batch_size': self.current_batch_size,
            'optimal_batch_size': self.get_optimal_batch_size(),
            'total_processed': self.total_processed,
            'total_time': self.total_time,
            'avg_throughput': self.total_processed / self.total_time if self.total_time > 0 else 0,
            'success_rate': self.success_count / (self.success_count + self.failure_count) if (self.success_count + self.failure_count) > 0 else 0,
            'adaptation_count': self.adaptation_count,
            'failure_count': self.failure_count
        }
        
        if successful_metrics:
            stats.update({
                'avg_processing_time': np.mean([m.processing_time for m in successful_metrics]),
                'avg_memory_usage': np.mean([m.memory_usage_gb for m in successful_metrics]),
                'max_throughput': max(m.throughput for m in successful_metrics),
                'avg_batch_throughput': np.mean([m.throughput for m in successful_metrics])
            })
        
        return stats
    
    def reset_stats(self):
        """重置统计信息"""
        self.metrics_history.clear()
        self.failure_count = 0
        self.success_count = 0
        self.total_processed = 0
        self.total_time = 0.0
        self.adaptation_count = 0
        logger.info("性能统计已重置")
    
    def set_batch_size_range(self, min_size: int, max_size: int):
        """设置批处理大小范围"""
        self.min_batch_size = max(1, min_size)
        self.max_batch_size = max(self.min_batch_size, max_size)
        self.current_batch_size = max(self.min_batch_size, 
                                    min(self.max_batch_size, self.current_batch_size))
        
        logger.info(f"批处理大小范围更新: [{self.min_batch_size}, {self.max_batch_size}], "
                   f"当前: {self.current_batch_size}")

class BatchSizeOptimizer:
    """批处理大小优化器"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.memory_manager = get_global_memory_manager()
    
    def find_optimal_batch_size(self, 
                               test_func: Callable,
                               test_data: List[Any],
                               max_batch_size: int = 64,
                               min_batch_size: int = 1) -> int:
        """寻找最优批处理大小"""
        logger.info(f"开始寻找最优批处理大小，范围: [{min_batch_size}, {max_batch_size}]")
        
        best_batch_size = min_batch_size
        best_throughput = 0
        
        # 二分搜索最大可用批处理大小
        left, right = min_batch_size, max_batch_size
        max_working_size = min_batch_size
        
        while left <= right:
            mid = (left + right) // 2
            
            try:
                # 测试当前批处理大小
                test_batch = test_data[:mid]
                start_time = time.time()
                
                _ = test_func(test_batch)
                
                processing_time = time.time() - start_time
                throughput = mid / processing_time if processing_time > 0 else 0
                
                logger.debug(f"批处理大小 {mid}: 吞吐量 {throughput:.2f} samples/s")
                
                # 更新最佳结果
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_batch_size = mid
                
                max_working_size = mid
                left = mid + 1
                
                # 清理内存
                self.memory_manager.auto_cleanup()
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.debug(f"批处理大小 {mid}: OOM")
                    right = mid - 1
                    self.memory_manager.aggressive_cleanup()
                else:
                    raise e
        
        # 在可工作范围内寻找最优吞吐量
        for batch_size in range(min_batch_size, max_working_size + 1, max(1, max_working_size // 10)):
            try:
                test_batch = test_data[:batch_size]
                start_time = time.time()
                
                _ = test_func(test_batch)
                
                processing_time = time.time() - start_time
                throughput = batch_size / processing_time if processing_time > 0 else 0
                
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_batch_size = batch_size
                
                self.memory_manager.auto_cleanup()
                
            except Exception as e:
                logger.debug(f"批处理大小 {batch_size} 测试失败: {e}")
                continue
        
        logger.info(f"找到最优批处理大小: {best_batch_size} (吞吐量: {best_throughput:.2f} samples/s)")
        return best_batch_size

if __name__ == "__main__":
    # 测试代码
    def dummy_process_func(data):
        """虚拟处理函数"""
        time.sleep(0.01 * len(data))  # 模拟处理时间
        return [f"processed_{i}" for i in range(len(data))]
    
    # 创建测试数据
    test_data = list(range(100))
    
    # 创建自适应批处理器
    processor = AdaptiveBatchProcessor(
        initial_batch_size=8,
        min_batch_size=1,
        max_batch_size=32
    )
    
    # 处理数据
    results = processor.process_all(test_data, dummy_process_func)
    
    # 获取性能统计
    stats = processor.get_performance_stats()
    print(f"性能统计: {stats}")
    
    print(f"处理完成: {len(results)} 个结果")