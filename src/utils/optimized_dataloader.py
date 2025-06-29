#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化的数据加载器
提供高效的数据加载、预处理和缓存功能
"""

import os
import time
import torch
import logging
import threading
import multiprocessing as mp
from typing import List, Dict, Optional, Callable, Any, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from queue import Queue, Empty
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image

from .enhanced_memory_manager import get_global_memory_manager

logger = logging.getLogger(__name__)

@dataclass
class DataLoaderConfig:
    """数据加载器配置"""
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True
    drop_last: bool = False
    shuffle: bool = False
    
    # 优化参数
    enable_caching: bool = True
    cache_size: int = 1000
    enable_prefetch: bool = True
    prefetch_queue_size: int = 10
    enable_async_transform: bool = True
    max_transform_workers: int = 2

class OptimizedDataset(Dataset):
    """优化的数据集类"""
    
    def __init__(self, 
                 data_paths: List[str],
                 transform: Optional[Callable] = None,
                 cache_size: int = 1000,
                 enable_caching: bool = True):
        """
        初始化优化数据集
        
        Args:
            data_paths: 数据文件路径列表
            transform: 数据变换函数
            cache_size: 缓存大小
            enable_caching: 是否启用缓存
        """
        self.data_paths = data_paths
        self.transform = transform
        self.enable_caching = enable_caching
        
        # 缓存系统
        if enable_caching:
            from functools import lru_cache
            self._load_data = lru_cache(maxsize=cache_size)(self._load_data_impl)
        else:
            self._load_data = self._load_data_impl
        
        # 预加载统计
        self.cache_hits = 0
        self.cache_misses = 0
        
        logger.info(f"优化数据集初始化: {len(data_paths)} 个样本, "
                   f"缓存: {'启用' if enable_caching else '禁用'}")
    
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        # 加载数据
        data = self._load_data(idx)
        
        # 应用变换
        if self.transform:
            data = self.transform(data)
        
        return data
    
    def _load_data_impl(self, idx: int):
        """实际的数据加载实现"""
        path = self.data_paths[idx]
        
        try:
            # 根据文件类型加载数据
            if path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                # 图像文件
                with Image.open(path) as img:
                    data = img.convert('RGB')
                    return np.array(data)
            else:
                # 其他文件类型
                with open(path, 'rb') as f:
                    return f.read()
                    
        except Exception as e:
            logger.error(f"加载数据失败: {path}, 错误: {e}")
            # 返回默认数据
            return np.zeros((224, 224, 3), dtype=np.uint8)
    
    def get_cache_stats(self) -> Dict:
        """获取缓存统计"""
        if hasattr(self._load_data, 'cache_info'):
            cache_info = self._load_data.cache_info()
            return {
                'hits': cache_info.hits,
                'misses': cache_info.misses,
                'hit_rate': cache_info.hits / (cache_info.hits + cache_info.misses) if (cache_info.hits + cache_info.misses) > 0 else 0,
                'current_size': cache_info.currsize,
                'max_size': cache_info.maxsize
            }
        return {}

class PrefetchDataLoader:
    """预取数据加载器"""
    
    def __init__(self, 
                 dataloader: DataLoader,
                 device: torch.device,
                 queue_size: int = 10):
        """
        初始化预取数据加载器
        
        Args:
            dataloader: 原始数据加载器
            device: 目标设备
            queue_size: 预取队列大小
        """
        self.dataloader = dataloader
        self.device = device
        self.queue_size = queue_size
        
        # 预取队列
        self.queue = Queue(maxsize=queue_size)
        self.thread = None
        self.stop_event = threading.Event()
        
        # 统计信息
        self.prefetch_count = 0
        self.wait_count = 0
        
        logger.info(f"预取数据加载器初始化: 队列大小 {queue_size}")
    
    def _prefetch_worker(self):
        """预取工作线程"""
        try:
            for batch in self.dataloader:
                if self.stop_event.is_set():
                    break
                
                # 将数据移动到目标设备
                if isinstance(batch, (list, tuple)):
                    batch = [item.to(self.device, non_blocking=True) if torch.is_tensor(item) else item for item in batch]
                elif torch.is_tensor(batch):
                    batch = batch.to(self.device, non_blocking=True)
                
                # 放入队列
                self.queue.put(batch)
                self.prefetch_count += 1
            
            # 放入结束标记
            self.queue.put(None)
            
        except Exception as e:
            logger.error(f"预取工作线程错误: {e}")
            self.queue.put(None)
    
    def __iter__(self):
        # 启动预取线程
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._prefetch_worker)
        self.thread.start()
        
        return self
    
    def __next__(self):
        try:
            # 从队列获取数据
            batch = self.queue.get(timeout=30)  # 30秒超时
            
            if batch is None:
                # 结束标记
                raise StopIteration
            
            return batch
            
        except Empty:
            logger.warning("预取队列超时")
            self.wait_count += 1
            raise StopIteration
    
    def __len__(self):
        return len(self.dataloader)
    
    def stop(self):
        """停止预取"""
        if self.thread and self.thread.is_alive():
            self.stop_event.set()
            self.thread.join(timeout=5)
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'prefetch_count': self.prefetch_count,
            'wait_count': self.wait_count,
            'queue_size': self.queue.qsize()
        }

class AsyncTransformDataLoader:
    """异步变换数据加载器"""
    
    def __init__(self, 
                 dataset: Dataset,
                 config: DataLoaderConfig,
                 device: torch.device):
        """
        初始化异步变换数据加载器
        
        Args:
            dataset: 数据集
            config: 配置
            device: 目标设备
        """
        self.dataset = dataset
        self.config = config
        self.device = device
        
        # 创建基础数据加载器（不包含变换）
        self.base_loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            prefetch_factor=config.prefetch_factor,
            persistent_workers=config.persistent_workers,
            drop_last=config.drop_last,
            shuffle=config.shuffle
        )
        
        # 异步变换执行器
        if config.enable_async_transform:
            self.transform_executor = ThreadPoolExecutor(
                max_workers=config.max_transform_workers
            )
        else:
            self.transform_executor = None
        
        # 统计信息
        self.transform_time = 0.0
        self.load_time = 0.0
        
        logger.info(f"异步变换数据加载器初始化: batch_size={config.batch_size}, "
                   f"num_workers={config.num_workers}")
    
    def __iter__(self):
        for batch in self.base_loader:
            start_time = time.time()
            
            # 异步应用变换
            if self.transform_executor and hasattr(self.dataset, 'transform') and self.dataset.transform:
                # 提交变换任务
                future = self.transform_executor.submit(self._apply_transform, batch)
                batch = future.result()
            
            # 移动到设备
            if torch.is_tensor(batch):
                batch = batch.to(self.device, non_blocking=True)
            elif isinstance(batch, (list, tuple)):
                batch = [item.to(self.device, non_blocking=True) if torch.is_tensor(item) else item for item in batch]
            
            self.load_time += time.time() - start_time
            yield batch
    
    def _apply_transform(self, batch):
        """应用变换"""
        start_time = time.time()
        
        if hasattr(self.dataset, 'transform') and self.dataset.transform:
            if isinstance(batch, (list, tuple)):
                batch = [self.dataset.transform(item) for item in batch]
            else:
                batch = self.dataset.transform(batch)
        
        self.transform_time += time.time() - start_time
        return batch
    
    def __len__(self):
        return len(self.base_loader)
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'transform_time': self.transform_time,
            'load_time': self.load_time,
            'avg_transform_time': self.transform_time / len(self) if len(self) > 0 else 0,
            'avg_load_time': self.load_time / len(self) if len(self) > 0 else 0
        }

class OptimizedDataLoader:
    """优化的数据加载器"""
    
    def __init__(self, 
                 dataset: Dataset,
                 config: DataLoaderConfig,
                 device: torch.device):
        """
        初始化优化数据加载器
        
        Args:
            dataset: 数据集
            config: 配置
            device: 目标设备
        """
        self.dataset = dataset
        self.config = config
        self.device = device
        self.memory_manager = get_global_memory_manager()
        
        # 自动优化配置
        self._optimize_config()
        
        # 创建数据加载器
        if config.enable_async_transform:
            self.loader = AsyncTransformDataLoader(dataset, config, device)
        else:
            self.loader = DataLoader(
                dataset,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory,
                prefetch_factor=config.prefetch_factor,
                persistent_workers=config.persistent_workers,
                drop_last=config.drop_last,
                shuffle=config.shuffle
            )
        
        # 预取包装
        if config.enable_prefetch:
            self.loader = PrefetchDataLoader(
                self.loader, 
                device, 
                config.prefetch_queue_size
            )
        
        # 统计信息
        self.total_batches = 0
        self.total_time = 0.0
        
        logger.info(f"优化数据加载器创建完成: {self._get_config_summary()}")
    
    def _optimize_config(self):
        """自动优化配置"""
        # 获取系统信息
        cpu_count = mp.cpu_count()
        memory_info = self.memory_manager.get_memory_snapshot()
        
        # 优化 num_workers
        if self.config.num_workers == 0:
            # 自动设置 workers 数量
            if torch.cuda.is_available():
                # GPU 环境，减少 workers 避免 CPU-GPU 瓶颈
                self.config.num_workers = min(4, cpu_count // 2)
            else:
                # CPU 环境，使用更多 workers
                self.config.num_workers = min(8, cpu_count)
        
        # 优化 batch_size
        if memory_info.ram_available_gb < 4:
            # 内存不足，减少批处理大小
            self.config.batch_size = max(1, self.config.batch_size // 2)
            logger.warning(f"内存不足，调整批处理大小为 {self.config.batch_size}")
        
        # 优化 prefetch_factor
        if self.config.prefetch_factor == 2 and self.config.num_workers > 4:
            # 多 worker 时增加预取
            self.config.prefetch_factor = 4
        
        # 优化 pin_memory
        if not torch.cuda.is_available():
            # CPU 环境不需要 pin_memory
            self.config.pin_memory = False
        
        logger.info(f"配置自动优化完成: workers={self.config.num_workers}, "
                   f"batch_size={self.config.batch_size}, "
                   f"prefetch_factor={self.config.prefetch_factor}")
    
    def _get_config_summary(self) -> str:
        """获取配置摘要"""
        return (f"batch_size={self.config.batch_size}, "
                f"num_workers={self.config.num_workers}, "
                f"pin_memory={self.config.pin_memory}, "
                f"prefetch={self.config.enable_prefetch}, "
                f"async_transform={self.config.enable_async_transform}")
    
    def __iter__(self):
        start_time = time.time()
        
        for batch in self.loader:
            self.total_batches += 1
            yield batch
        
        self.total_time += time.time() - start_time
    
    def __len__(self):
        return len(self.loader)
    
    def get_performance_stats(self) -> Dict:
        """获取性能统计"""
        stats = {
            'total_batches': self.total_batches,
            'total_time': self.total_time,
            'avg_batch_time': self.total_time / self.total_batches if self.total_batches > 0 else 0,
            'config': self._get_config_summary()
        }
        
        # 添加数据集缓存统计
        if hasattr(self.dataset, 'get_cache_stats'):
            stats['cache_stats'] = self.dataset.get_cache_stats()
        
        # 添加加载器统计
        if hasattr(self.loader, 'get_stats'):
            stats['loader_stats'] = self.loader.get_stats()
        
        return stats
    
    def cleanup(self):
        """清理资源"""
        if hasattr(self.loader, 'stop'):
            self.loader.stop()
        
        if hasattr(self.loader, 'transform_executor') and self.loader.transform_executor:
            self.loader.transform_executor.shutdown(wait=True)
        
        logger.info("数据加载器资源清理完成")

def create_optimized_dataloader(data_paths: List[str],
                              transform: Optional[Callable] = None,
                              device: torch.device = None,
                              config: Optional[DataLoaderConfig] = None) -> OptimizedDataLoader:
    """创建优化的数据加载器"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if config is None:
        config = DataLoaderConfig()
    
    # 创建优化数据集
    dataset = OptimizedDataset(
        data_paths=data_paths,
        transform=transform,
        cache_size=config.cache_size,
        enable_caching=config.enable_caching
    )
    
    # 创建优化数据加载器
    dataloader = OptimizedDataLoader(dataset, config, device)
    
    return dataloader

def benchmark_dataloader(dataloader: OptimizedDataLoader, 
                        num_epochs: int = 1) -> Dict:
    """基准测试数据加载器"""
    logger.info(f"开始数据加载器基准测试: {num_epochs} 个epoch")
    
    total_start_time = time.time()
    total_batches = 0
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_batches = 0
        
        for batch in dataloader:
            epoch_batches += 1
            total_batches += 1
            
            # 模拟处理时间
            time.sleep(0.001)
        
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch + 1}: {epoch_batches} batches, {epoch_time:.2f}s")
    
    total_time = time.time() - total_start_time
    
    # 获取性能统计
    stats = dataloader.get_performance_stats()
    stats.update({
        'benchmark_total_time': total_time,
        'benchmark_total_batches': total_batches,
        'benchmark_avg_batch_time': total_time / total_batches if total_batches > 0 else 0,
        'benchmark_throughput': total_batches / total_time if total_time > 0 else 0
    })
    
    logger.info(f"基准测试完成: {total_batches} batches, {total_time:.2f}s, "
               f"吞吐量: {stats['benchmark_throughput']:.2f} batches/s")
    
    return stats

if __name__ == "__main__":
    # 测试代码
    import tempfile
    
    # 创建测试数据
    test_dir = tempfile.mkdtemp()
    test_paths = []
    
    for i in range(100):
        # 创建虚拟图像文件
        img = Image.new('RGB', (224, 224), color=(i % 255, (i * 2) % 255, (i * 3) % 255))
        path = os.path.join(test_dir, f"test_{i}.png")
        img.save(path)
        test_paths.append(path)
    
    # 创建配置
    config = DataLoaderConfig(
        batch_size=8,
        num_workers=2,
        enable_caching=True,
        enable_prefetch=True,
        enable_async_transform=True
    )
    
    # 创建数据加载器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = create_optimized_dataloader(test_paths, device=device, config=config)
    
    # 运行基准测试
    stats = benchmark_dataloader(dataloader, num_epochs=2)
    print(f"基准测试结果: {stats}")
    
    # 清理
    dataloader.cleanup()
    
    # 删除测试文件
    import shutil
    shutil.rmtree(test_dir)
    
    print("测试完成")