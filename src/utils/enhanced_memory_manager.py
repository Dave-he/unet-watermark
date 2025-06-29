#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强的内存管理器
提供智能内存监控、自动清理和性能优化功能
"""

import gc
import time
import psutil
import torch
import logging
import threading
from typing import Dict, List, Callable, Optional
from dataclasses import dataclass
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@dataclass
class MemorySnapshot:
    """内存快照数据类"""
    timestamp: float
    cpu_percent: float
    ram_used_gb: float
    ram_available_gb: float
    gpu_allocated_gb: float = 0.0
    gpu_reserved_gb: float = 0.0
    gpu_free_gb: float = 0.0
    gpu_total_gb: float = 0.0

class EnhancedMemoryManager:
    """增强的内存管理器"""
    
    def __init__(self, 
                 gpu_memory_threshold: float = 0.8,
                 cpu_memory_threshold: float = 0.85,
                 auto_cleanup_interval: float = 30.0,
                 enable_monitoring: bool = True):
        """
        初始化内存管理器
        
        Args:
            gpu_memory_threshold: GPU内存使用阈值
            cpu_memory_threshold: CPU内存使用阈值
            auto_cleanup_interval: 自动清理间隔（秒）
            enable_monitoring: 是否启用监控
        """
        self.gpu_memory_threshold = gpu_memory_threshold
        self.cpu_memory_threshold = cpu_memory_threshold
        self.auto_cleanup_interval = auto_cleanup_interval
        self.enable_monitoring = enable_monitoring
        
        # 监控数据
        self.snapshots: List[MemorySnapshot] = []
        self.cleanup_callbacks: List[Callable] = []
        
        # 统计信息
        self.cleanup_count = 0
        self.oom_prevention_count = 0
        
        # 监控线程
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        
        # 启动监控
        if self.enable_monitoring:
            self.start_monitoring()
    
    def get_memory_snapshot(self) -> MemorySnapshot:
        """获取当前内存快照"""
        # CPU和RAM信息
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_info = psutil.virtual_memory()
        ram_used_gb = memory_info.used / 1024**3
        ram_available_gb = memory_info.available / 1024**3
        
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            ram_used_gb=ram_used_gb,
            ram_available_gb=ram_available_gb
        )
        
        # GPU信息（如果可用）
        if torch.cuda.is_available():
            device_props = torch.cuda.get_device_properties(0)
            total_memory = device_props.total_memory / 1024**3
            
            snapshot.gpu_allocated_gb = torch.cuda.memory_allocated() / 1024**3
            snapshot.gpu_reserved_gb = torch.cuda.memory_reserved() / 1024**3
            snapshot.gpu_free_gb = total_memory - snapshot.gpu_reserved_gb
            snapshot.gpu_total_gb = total_memory
        
        return snapshot
    
    def check_memory_pressure(self) -> Dict[str, bool]:
        """检查内存压力"""
        snapshot = self.get_memory_snapshot()
        
        pressure = {
            'cpu_high': snapshot.cpu_percent > 90,
            'ram_high': (snapshot.ram_used_gb / (snapshot.ram_used_gb + snapshot.ram_available_gb)) > self.cpu_memory_threshold,
            'gpu_high': False,
            'gpu_critical': False
        }
        
        if torch.cuda.is_available() and snapshot.gpu_total_gb > 0:
            gpu_usage_ratio = snapshot.gpu_reserved_gb / snapshot.gpu_total_gb
            pressure['gpu_high'] = gpu_usage_ratio > self.gpu_memory_threshold
            pressure['gpu_critical'] = gpu_usage_ratio > 0.95
        
        return pressure
    
    def auto_cleanup(self, force: bool = False) -> bool:
        """自动内存清理"""
        pressure = self.check_memory_pressure()
        
        # 检查是否需要清理
        need_cleanup = force or any([
            pressure['cpu_high'],
            pressure['ram_high'],
            pressure['gpu_high']
        ])
        
        if need_cleanup:
            return self.aggressive_cleanup()
        
        return False
    
    def aggressive_cleanup(self) -> bool:
        """激进的内存清理"""
        try:
            logger.info("执行激进内存清理...")
            
            # 执行自定义清理回调
            for callback in self.cleanup_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.warning(f"清理回调执行失败: {e}")
            
            # Python垃圾回收
            collected = gc.collect()
            logger.debug(f"Python GC回收了 {collected} 个对象")
            
            # GPU内存清理
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.debug("GPU内存缓存已清理")
            
            self.cleanup_count += 1
            return True
            
        except Exception as e:
            logger.error(f"内存清理失败: {e}")
            return False
    
    def prevent_oom(self, required_memory_gb: float = 1.0) -> bool:
        """OOM预防"""
        snapshot = self.get_memory_snapshot()
        
        # 检查GPU内存是否足够
        if torch.cuda.is_available():
            if snapshot.gpu_free_gb < required_memory_gb:
                logger.warning(f"GPU内存不足，需要 {required_memory_gb:.2f}GB，可用 {snapshot.gpu_free_gb:.2f}GB")
                
                # 尝试清理内存
                if self.aggressive_cleanup():
                    # 重新检查
                    new_snapshot = self.get_memory_snapshot()
                    if new_snapshot.gpu_free_gb >= required_memory_gb:
                        self.oom_prevention_count += 1
                        return True
                
                return False
        
        # 检查RAM是否足够
        if snapshot.ram_available_gb < required_memory_gb:
            logger.warning(f"RAM不足，需要 {required_memory_gb:.2f}GB，可用 {snapshot.ram_available_gb:.2f}GB")
            
            if self.aggressive_cleanup():
                new_snapshot = self.get_memory_snapshot()
                if new_snapshot.ram_available_gb >= required_memory_gb:
                    self.oom_prevention_count += 1
                    return True
            
            return False
        
        return True
    
    def add_cleanup_callback(self, callback: Callable):
        """添加清理回调函数"""
        self.cleanup_callbacks.append(callback)
    
    def start_monitoring(self):
        """启动内存监控"""
        if self._monitoring_thread is not None:
            return
        
        def monitor_loop():
            while not self._stop_monitoring.wait(self.auto_cleanup_interval):
                try:
                    snapshot = self.get_memory_snapshot()
                    self.snapshots.append(snapshot)
                    
                    # 保持最近100个快照
                    if len(self.snapshots) > 100:
                        self.snapshots = self.snapshots[-100:]
                    
                    # 检查是否需要自动清理
                    pressure = self.check_memory_pressure()
                    if pressure['gpu_critical'] or pressure['ram_high']:
                        logger.warning("检测到内存压力，执行自动清理")
                        self.auto_cleanup()
                    
                except Exception as e:
                    logger.error(f"内存监控错误: {e}")
        
        self._monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitoring_thread.start()
        logger.info("内存监控已启动")
    
    def stop_monitoring(self):
        """停止内存监控"""
        if self._monitoring_thread is not None:
            self._stop_monitoring.set()
            self._monitoring_thread.join(timeout=5)
            self._monitoring_thread = None
            logger.info("内存监控已停止")
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        if not self.snapshots:
            return {}
        
        recent_snapshots = self.snapshots[-10:]  # 最近10个快照
        
        stats = {
            'cleanup_count': self.cleanup_count,
            'oom_prevention_count': self.oom_prevention_count,
            'avg_cpu_percent': sum(s.cpu_percent for s in recent_snapshots) / len(recent_snapshots),
            'avg_ram_used_gb': sum(s.ram_used_gb for s in recent_snapshots) / len(recent_snapshots),
        }
        
        if torch.cuda.is_available() and recent_snapshots[0].gpu_total_gb > 0:
            stats.update({
                'avg_gpu_allocated_gb': sum(s.gpu_allocated_gb for s in recent_snapshots) / len(recent_snapshots),
                'max_gpu_allocated_gb': max(s.gpu_allocated_gb for s in recent_snapshots),
                'avg_gpu_usage_percent': sum(s.gpu_reserved_gb / s.gpu_total_gb for s in recent_snapshots) / len(recent_snapshots) * 100
            })
        
        return stats
    
    def __del__(self):
        """析构函数"""
        self.stop_monitoring()

@contextmanager
def memory_context(cleanup_on_exit: bool = True, 
                  required_memory_gb: float = 1.0,
                  manager: Optional[EnhancedMemoryManager] = None):
    """内存管理上下文管理器"""
    if manager is None:
        manager = EnhancedMemoryManager()
    
    # 入口检查
    if not manager.prevent_oom(required_memory_gb):
        raise RuntimeError(f"内存不足，无法分配 {required_memory_gb:.2f}GB")
    
    try:
        yield manager
    finally:
        if cleanup_on_exit:
            manager.auto_cleanup(force=True)

def optimize_dataloader_params(device: torch.device, 
                             available_memory_gb: float = None) -> Dict:
    """根据可用内存优化DataLoader参数"""
    if available_memory_gb is None:
        if torch.cuda.is_available():
            available_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        else:
            available_memory_gb = psutil.virtual_memory().available / 1024**3
    
    # 根据内存大小调整参数
    if available_memory_gb >= 16:
        num_workers = 8
        prefetch_factor = 4
        pin_memory = device.type == 'cuda'
    elif available_memory_gb >= 8:
        num_workers = 4
        prefetch_factor = 2
        pin_memory = device.type == 'cuda'
    else:
        num_workers = 2
        prefetch_factor = 1
        pin_memory = False
    
    return {
        'num_workers': num_workers,
        'prefetch_factor': prefetch_factor,
        'pin_memory': pin_memory,
        'persistent_workers': num_workers > 0
    }

# 全局内存管理器实例
_global_memory_manager = None

def get_global_memory_manager() -> EnhancedMemoryManager:
    """获取全局内存管理器实例"""
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = EnhancedMemoryManager()
    return _global_memory_manager

def cleanup_global_memory():
    """清理全局内存"""
    manager = get_global_memory_manager()
    manager.aggressive_cleanup()

# 装饰器
def memory_optimized(required_memory_gb: float = 1.0, cleanup_after: bool = True):
    """内存优化装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with memory_context(cleanup_after, required_memory_gb) as manager:
                return func(*args, **kwargs)
        return wrapper
    return decorator

if __name__ == "__main__":
    # 测试代码
    manager = EnhancedMemoryManager()
    
    print("内存管理器测试")
    snapshot = manager.get_memory_snapshot()
    print(f"当前内存使用: RAM {snapshot.ram_used_gb:.2f}GB")
    
    if torch.cuda.is_available():
        print(f"GPU内存: {snapshot.gpu_allocated_gb:.2f}GB / {snapshot.gpu_total_gb:.2f}GB")
    
    # 测试清理
    manager.aggressive_cleanup()
    print("内存清理完成")
    
    # 测试统计
    time.sleep(1)
    stats = manager.get_statistics()
    print(f"统计信息: {stats}")
    
    manager.stop_monitoring()