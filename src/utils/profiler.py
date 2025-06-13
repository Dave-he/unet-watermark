# -*- coding: utf-8 -*-
"""
性能分析工具
用于监控和分析代码执行性能
"""

import time
import torch
from contextlib import contextmanager
from typing import Dict, List, Any, Generator

@contextmanager
def timer(name: str) -> Generator[None, None, None]:
    """简单的计时器"""
    start = time.time()
    yield
    end = time.time()
    print(f"{name}: {end - start:.4f}s")

class PerformanceMonitor:
    def __init__(self) -> None:
        self.times: Dict[str, List[float]] = {}
        self.gpu_memory: List[float] = []
    
    def log_time(self, name: str, duration: float) -> None:
        if name not in self.times:
            self.times[name] = []
        self.times[name].append(duration)
    
    def log_gpu_memory(self) -> None:
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
            self.gpu_memory.append(memory_used)
    
    def report(self) -> None:
        print("\n=== 性能报告 ===")
        for name, times in self.times.items():
            avg_time = sum(times) / len(times)
            print(f"{name}: 平均 {avg_time:.4f}s")
        
        if self.gpu_memory:
            avg_memory = sum(self.gpu_memory) / len(self.gpu_memory)
            max_memory = max(self.gpu_memory)
            print(f"GPU内存: 平均 {avg_memory:.2f}GB, 峰值 {max_memory:.2f}GB")