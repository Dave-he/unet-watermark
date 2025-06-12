#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUDA内存监控和管理工具
用于诊断和解决CUDA卡顿问题
"""

import torch
import psutil
import time
import logging
import threading
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MemorySnapshot:
    """内存快照数据类"""
    timestamp: float
    gpu_allocated: float  # GB
    gpu_reserved: float   # GB
    gpu_free: float      # GB
    cpu_percent: float
    ram_percent: float
    ram_available: float  # GB

class CUDAMonitor:
    """CUDA内存监控器"""
    
    def __init__(self, device: str = 'cuda', log_interval: float = 5.0):
        self.device = device
        self.log_interval = log_interval
        self.monitoring = False
        self.snapshots: List[MemorySnapshot] = []
        self.monitor_thread: Optional[threading.Thread] = None
        self.callbacks: List[Callable[[MemorySnapshot], None]] = []
        
        # 阈值设置
        self.gpu_memory_warning_threshold = 0.8  # 80%
        self.gpu_memory_critical_threshold = 0.9  # 90%
        self.ram_warning_threshold = 0.8  # 80%
        
    def get_memory_info(self) -> MemorySnapshot:
        """获取当前内存信息"""
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            gpu_allocated=0.0,
            gpu_reserved=0.0,
            gpu_free=0.0,
            cpu_percent=psutil.cpu_percent(),
            ram_percent=psutil.virtual_memory().percent,
            ram_available=psutil.virtual_memory().available / 1024**3
        )
        
        if torch.cuda.is_available() and self.device == 'cuda':
            device_props = torch.cuda.get_device_properties(0)
            total_memory = device_props.total_memory
            
            snapshot.gpu_allocated = torch.cuda.memory_allocated() / 1024**3
            snapshot.gpu_reserved = torch.cuda.memory_reserved() / 1024**3
            snapshot.gpu_free = (total_memory - torch.cuda.memory_reserved()) / 1024**3
        
        return snapshot
    
    def start_monitoring(self):
        """开始监控"""
        if self.monitoring:
            logger.warning("监控已经在运行中")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info(f"CUDA监控已启动，间隔: {self.log_interval}秒")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=self.log_interval + 1)
        logger.info("CUDA监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                snapshot = self.get_memory_info()
                self.snapshots.append(snapshot)
                
                # 检查阈值并发出警告
                self._check_thresholds(snapshot)
                
                # 调用回调函数
                for callback in self.callbacks:
                    try:
                        callback(snapshot)
                    except Exception as e:
                        logger.error(f"监控回调函数执行失败: {e}")
                
                # 限制快照数量，避免内存泄漏
                if len(self.snapshots) > 1000:
                    self.snapshots = self.snapshots[-500:]
                
                time.sleep(self.log_interval)
                
            except Exception as e:
                logger.error(f"监控循环出错: {e}")
                time.sleep(self.log_interval)
    
    def _check_thresholds(self, snapshot: MemorySnapshot):
        """检查阈值并发出警告"""
        if torch.cuda.is_available() and self.device == 'cuda':
            device_props = torch.cuda.get_device_properties(0)
            total_gpu_memory = device_props.total_memory / 1024**3
            gpu_usage_ratio = snapshot.gpu_reserved / total_gpu_memory
            
            if gpu_usage_ratio >= self.gpu_memory_critical_threshold:
                logger.critical(f"GPU内存使用率达到临界值: {gpu_usage_ratio:.1%} "
                              f"({snapshot.gpu_reserved:.2f}GB / {total_gpu_memory:.2f}GB)")
            elif gpu_usage_ratio >= self.gpu_memory_warning_threshold:
                logger.warning(f"GPU内存使用率较高: {gpu_usage_ratio:.1%} "
                             f"({snapshot.gpu_reserved:.2f}GB / {total_gpu_memory:.2f}GB)")
        
        if snapshot.ram_percent >= self.ram_warning_threshold * 100:
            logger.warning(f"系统内存使用率较高: {snapshot.ram_percent:.1f}% "
                         f"(可用: {snapshot.ram_available:.2f}GB)")
    
    def add_callback(self, callback: Callable[[MemorySnapshot], None]):
        """添加监控回调函数"""
        self.callbacks.append(callback)
    
    def get_memory_summary(self) -> Dict:
        """获取内存使用摘要"""
        if not self.snapshots:
            return {}
        
        latest = self.snapshots[-1]
        
        summary = {
            'current': {
                'gpu_allocated_gb': latest.gpu_allocated,
                'gpu_reserved_gb': latest.gpu_reserved,
                'gpu_free_gb': latest.gpu_free,
                'cpu_percent': latest.cpu_percent,
                'ram_percent': latest.ram_percent,
                'ram_available_gb': latest.ram_available
            }
        }
        
        if len(self.snapshots) > 1:
            gpu_allocated_values = [s.gpu_allocated for s in self.snapshots]
            gpu_reserved_values = [s.gpu_reserved for s in self.snapshots]
            
            summary['statistics'] = {
                'gpu_allocated_max_gb': max(gpu_allocated_values),
                'gpu_allocated_avg_gb': sum(gpu_allocated_values) / len(gpu_allocated_values),
                'gpu_reserved_max_gb': max(gpu_reserved_values),
                'gpu_reserved_avg_gb': sum(gpu_reserved_values) / len(gpu_reserved_values),
                'snapshots_count': len(self.snapshots)
            }
        
        return summary
    
    def clear_snapshots(self):
        """清空快照历史"""
        self.snapshots.clear()
        logger.info("监控快照历史已清空")

class CUDAMemoryManager:
    """CUDA内存管理器"""
    
    @staticmethod
    def aggressive_cleanup():
        """激进的内存清理"""
        import gc
        
        # Python垃圾回收
        collected = gc.collect()
        logger.info(f"Python垃圾回收清理了 {collected} 个对象")
        
        if torch.cuda.is_available():
            # 清空CUDA缓存
            torch.cuda.empty_cache()
            
            # 同步CUDA操作
            torch.cuda.synchronize()
            
            # 重置峰值内存统计
            torch.cuda.reset_peak_memory_stats()
            
            logger.info("CUDA内存清理完成")
    
    @staticmethod
    def get_cuda_info() -> Dict:
        """获取CUDA设备信息"""
        if not torch.cuda.is_available():
            return {'available': False}
        
        device_props = torch.cuda.get_device_properties(0)
        
        return {
            'available': True,
            'device_name': device_props.name,
            'total_memory_gb': device_props.total_memory / 1024**3,
            'major': device_props.major,
            'minor': device_props.minor,
            'multi_processor_count': device_props.multi_processor_count,
            'current_allocated_gb': torch.cuda.memory_allocated() / 1024**3,
            'current_reserved_gb': torch.cuda.memory_reserved() / 1024**3,
            'max_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3,
            'max_reserved_gb': torch.cuda.max_memory_reserved() / 1024**3
        }
    
    @staticmethod
    def set_memory_fraction(fraction: float):
        """设置CUDA内存使用比例"""
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(fraction)
            logger.info(f"CUDA内存使用比例设置为: {fraction:.1%}")

@contextmanager
def cuda_memory_context(cleanup_on_exit: bool = True, monitor: bool = False):
    """CUDA内存管理上下文管理器"""
    monitor_instance = None
    
    try:
        if monitor:
            monitor_instance = CUDAMonitor()
            monitor_instance.start_monitoring()
        
        # 进入时清理一次
        CUDAMemoryManager.aggressive_cleanup()
        
        yield monitor_instance
        
    finally:
        if monitor_instance:
            monitor_instance.stop_monitoring()
        
        if cleanup_on_exit:
            CUDAMemoryManager.aggressive_cleanup()

def log_memory_usage(prefix: str = ""):
    """记录当前内存使用情况"""
    monitor = CUDAMonitor()
    snapshot = monitor.get_memory_info()
    
    log_msg = f"{prefix}内存使用情况: "
    
    if torch.cuda.is_available():
        log_msg += f"GPU {snapshot.gpu_allocated:.2f}GB/{snapshot.gpu_reserved:.2f}GB, "
    
    log_msg += f"CPU {snapshot.cpu_percent:.1f}%, RAM {snapshot.ram_percent:.1f}%"
    
    logger.info(log_msg)

# 使用示例
if __name__ == '__main__':
    # 打印CUDA信息
    cuda_info = CUDAMemoryManager.get_cuda_info()
    print("CUDA设备信息:")
    for key, value in cuda_info.items():
        print(f"  {key}: {value}")
    
    # 使用上下文管理器
    with cuda_memory_context(monitor=True) as monitor:
        print("\n在上下文中执行一些操作...")
        time.sleep(3)
        
        if monitor:
            summary = monitor.get_memory_summary()
            print("\n内存使用摘要:")
            for key, value in summary.items():
                print(f"  {key}: {value}")