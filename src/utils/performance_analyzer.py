#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能分析器
提供全面的性能监控、分析和优化建议
"""

import os
import time
import psutil
import torch
import logging
import threading
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta

from .enhanced_memory_manager import get_global_memory_manager

logger = logging.getLogger(__name__)

@dataclass
class PerformanceSnapshot:
    """性能快照"""
    timestamp: float
    
    # CPU 信息
    cpu_percent: float
    cpu_count: int
    cpu_freq: float
    
    # 内存信息
    ram_total_gb: float
    ram_used_gb: float
    ram_available_gb: float
    ram_percent: float
    
    # GPU 信息
    gpu_available: bool
    gpu_count: int = 0
    gpu_memory_total_gb: float = 0.0
    gpu_memory_used_gb: float = 0.0
    gpu_memory_free_gb: float = 0.0
    gpu_utilization: float = 0.0
    gpu_temperature: float = 0.0
    
    # 进程信息
    process_cpu_percent: float = 0.0
    process_memory_mb: float = 0.0
    process_threads: int = 0
    
    # 自定义指标
    custom_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class PerformanceMetrics:
    """性能指标"""
    operation_name: str
    start_time: float
    end_time: float
    duration: float
    
    # 资源使用
    cpu_usage: float
    memory_usage_mb: float
    gpu_memory_usage_gb: float
    
    # 吞吐量
    items_processed: int
    throughput: float  # items/second
    
    # 状态
    success: bool
    error_message: Optional[str] = None
    
    # 自定义数据
    metadata: Dict[str, Any] = field(default_factory=dict)

class SystemMonitor:
    """系统监控器"""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.is_monitoring = False
        self.monitor_thread = None
        self.snapshots: deque = deque(maxlen=1000)
        self.lock = threading.Lock()
        
        # GPU 监控
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            try:
                import pynvml
                pynvml.nvmlInit()
                self.nvml_available = True
                self.gpu_handles = []
                for i in range(torch.cuda.device_count()):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    self.gpu_handles.append(handle)
            except ImportError:
                self.nvml_available = False
                logger.warning("pynvml 不可用，GPU 详细监控功能受限")
        else:
            self.nvml_available = False
    
    def start_monitoring(self):
        """开始监控"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("系统监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("系统监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                snapshot = self._take_snapshot()
                with self.lock:
                    self.snapshots.append(snapshot)
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"监控错误: {e}")
                time.sleep(self.monitoring_interval)
    
    def _take_snapshot(self) -> PerformanceSnapshot:
        """获取性能快照"""
        # CPU 信息
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq().current if psutil.cpu_freq() else 0
        
        # 内存信息
        memory = psutil.virtual_memory()
        ram_total_gb = memory.total / (1024**3)
        ram_used_gb = memory.used / (1024**3)
        ram_available_gb = memory.available / (1024**3)
        ram_percent = memory.percent
        
        # 进程信息
        process = psutil.Process()
        process_cpu_percent = process.cpu_percent()
        process_memory_mb = process.memory_info().rss / (1024**2)
        process_threads = process.num_threads()
        
        # GPU 信息
        gpu_memory_total_gb = 0.0
        gpu_memory_used_gb = 0.0
        gpu_memory_free_gb = 0.0
        gpu_utilization = 0.0
        gpu_temperature = 0.0
        gpu_count = 0
        
        if self.gpu_available:
            gpu_count = torch.cuda.device_count()
            
            if self.nvml_available:
                try:
                    import pynvml
                    total_memory = 0
                    used_memory = 0
                    total_utilization = 0
                    total_temperature = 0
                    
                    for handle in self.gpu_handles:
                        # 内存信息
                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        total_memory += mem_info.total
                        used_memory += mem_info.used
                        
                        # 利用率
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        total_utilization += util.gpu
                        
                        # 温度
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        total_temperature += temp
                    
                    gpu_memory_total_gb = total_memory / (1024**3)
                    gpu_memory_used_gb = used_memory / (1024**3)
                    gpu_memory_free_gb = (total_memory - used_memory) / (1024**3)
                    gpu_utilization = total_utilization / len(self.gpu_handles)
                    gpu_temperature = total_temperature / len(self.gpu_handles)
                    
                except Exception as e:
                    logger.debug(f"NVML 监控错误: {e}")
            else:
                # 使用 PyTorch 获取基本 GPU 信息
                for i in range(gpu_count):
                    gpu_memory_total_gb += torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    gpu_memory_used_gb += torch.cuda.memory_allocated(i) / (1024**3)
                
                gpu_memory_free_gb = gpu_memory_total_gb - gpu_memory_used_gb
        
        return PerformanceSnapshot(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            cpu_count=cpu_count,
            cpu_freq=cpu_freq,
            ram_total_gb=ram_total_gb,
            ram_used_gb=ram_used_gb,
            ram_available_gb=ram_available_gb,
            ram_percent=ram_percent,
            gpu_available=self.gpu_available,
            gpu_count=gpu_count,
            gpu_memory_total_gb=gpu_memory_total_gb,
            gpu_memory_used_gb=gpu_memory_used_gb,
            gpu_memory_free_gb=gpu_memory_free_gb,
            gpu_utilization=gpu_utilization,
            gpu_temperature=gpu_temperature,
            process_cpu_percent=process_cpu_percent,
            process_memory_mb=process_memory_mb,
            process_threads=process_threads
        )
    
    def get_current_snapshot(self) -> PerformanceSnapshot:
        """获取当前快照"""
        return self._take_snapshot()
    
    def get_snapshots(self, duration_seconds: Optional[float] = None) -> List[PerformanceSnapshot]:
        """获取快照历史"""
        with self.lock:
            snapshots = list(self.snapshots)
        
        if duration_seconds is not None:
            cutoff_time = time.time() - duration_seconds
            snapshots = [s for s in snapshots if s.timestamp >= cutoff_time]
        
        return snapshots

class PerformanceProfiler:
    """性能分析器"""
    
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.active_operations: Dict[str, Dict] = {}
        self.lock = threading.Lock()
    
    def start_operation(self, operation_name: str, items_count: int = 1, **metadata) -> str:
        """开始操作计时"""
        operation_id = f"{operation_name}_{int(time.time() * 1000000)}"
        
        start_snapshot = self._get_resource_snapshot()
        
        operation_data = {
            'operation_name': operation_name,
            'start_time': time.time(),
            'start_snapshot': start_snapshot,
            'items_count': items_count,
            'metadata': metadata
        }
        
        with self.lock:
            self.active_operations[operation_id] = operation_data
        
        return operation_id
    
    def end_operation(self, operation_id: str, success: bool = True, error_message: Optional[str] = None):
        """结束操作计时"""
        end_time = time.time()
        end_snapshot = self._get_resource_snapshot()
        
        with self.lock:
            if operation_id not in self.active_operations:
                logger.warning(f"操作 {operation_id} 不存在")
                return
            
            operation_data = self.active_operations.pop(operation_id)
        
        # 计算指标
        duration = end_time - operation_data['start_time']
        items_processed = operation_data['items_count']
        throughput = items_processed / duration if duration > 0 else 0
        
        # 计算资源使用变化
        start_snapshot = operation_data['start_snapshot']
        cpu_usage = end_snapshot['cpu_percent'] - start_snapshot['cpu_percent']
        memory_usage_mb = end_snapshot['memory_mb'] - start_snapshot['memory_mb']
        gpu_memory_usage_gb = end_snapshot['gpu_memory_gb'] - start_snapshot['gpu_memory_gb']
        
        metrics = PerformanceMetrics(
            operation_name=operation_data['operation_name'],
            start_time=operation_data['start_time'],
            end_time=end_time,
            duration=duration,
            cpu_usage=cpu_usage,
            memory_usage_mb=memory_usage_mb,
            gpu_memory_usage_gb=gpu_memory_usage_gb,
            items_processed=items_processed,
            throughput=throughput,
            success=success,
            error_message=error_message,
            metadata=operation_data['metadata']
        )
        
        with self.lock:
            self.metrics.append(metrics)
    
    def _get_resource_snapshot(self) -> Dict:
        """获取资源快照"""
        # CPU 和内存
        process = psutil.Process()
        cpu_percent = process.cpu_percent()
        memory_mb = process.memory_info().rss / (1024**2)
        
        # GPU 内存
        gpu_memory_gb = 0.0
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_memory_gb += torch.cuda.memory_allocated(i) / (1024**3)
        
        return {
            'cpu_percent': cpu_percent,
            'memory_mb': memory_mb,
            'gpu_memory_gb': gpu_memory_gb
        }
    
    def get_metrics(self, operation_name: Optional[str] = None) -> List[PerformanceMetrics]:
        """获取性能指标"""
        with self.lock:
            metrics = list(self.metrics)
        
        if operation_name:
            metrics = [m for m in metrics if m.operation_name == operation_name]
        
        return metrics
    
    def clear_metrics(self):
        """清除指标"""
        with self.lock:
            self.metrics.clear()
            self.active_operations.clear()

class PerformanceAnalyzer:
    """性能分析器主类"""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.system_monitor = SystemMonitor(monitoring_interval)
        self.profiler = PerformanceProfiler()
        self.memory_manager = get_global_memory_manager()
        
        # 分析结果缓存
        self.analysis_cache = {}
        self.cache_timeout = 60  # 缓存60秒
    
    def start_monitoring(self):
        """开始性能监控"""
        self.system_monitor.start_monitoring()
        logger.info("性能分析器监控已启动")
    
    def stop_monitoring(self):
        """停止性能监控"""
        self.system_monitor.stop_monitoring()
        logger.info("性能分析器监控已停止")
    
    def profile_operation(self, operation_name: str, items_count: int = 1, **metadata):
        """操作性能分析装饰器"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                operation_id = self.profiler.start_operation(operation_name, items_count, **metadata)
                try:
                    result = func(*args, **kwargs)
                    self.profiler.end_operation(operation_id, success=True)
                    return result
                except Exception as e:
                    self.profiler.end_operation(operation_id, success=False, error_message=str(e))
                    raise
            return wrapper
        return decorator
    
    def analyze_system_performance(self, duration_minutes: int = 10) -> Dict:
        """分析系统性能"""
        cache_key = f"system_analysis_{duration_minutes}"
        
        # 检查缓存
        if cache_key in self.analysis_cache:
            cached_time, cached_result = self.analysis_cache[cache_key]
            if time.time() - cached_time < self.cache_timeout:
                return cached_result
        
        snapshots = self.system_monitor.get_snapshots(duration_minutes * 60)
        
        if not snapshots:
            return {'error': '没有可用的监控数据'}
        
        # 计算统计信息
        cpu_usage = [s.cpu_percent for s in snapshots]
        ram_usage = [s.ram_percent for s in snapshots]
        process_cpu = [s.process_cpu_percent for s in snapshots]
        process_memory = [s.process_memory_mb for s in snapshots]
        
        analysis = {
            'monitoring_duration_minutes': duration_minutes,
            'sample_count': len(snapshots),
            'timestamp': datetime.now().isoformat(),
            
            # CPU 分析
            'cpu': {
                'avg_usage': np.mean(cpu_usage),
                'max_usage': np.max(cpu_usage),
                'min_usage': np.min(cpu_usage),
                'std_usage': np.std(cpu_usage),
                'cpu_count': snapshots[0].cpu_count,
                'avg_frequency': np.mean([s.cpu_freq for s in snapshots if s.cpu_freq > 0])
            },
            
            # 内存分析
            'memory': {
                'total_gb': snapshots[0].ram_total_gb,
                'avg_usage_percent': np.mean(ram_usage),
                'max_usage_percent': np.max(ram_usage),
                'avg_available_gb': np.mean([s.ram_available_gb for s in snapshots]),
                'min_available_gb': np.min([s.ram_available_gb for s in snapshots])
            },
            
            # 进程分析
            'process': {
                'avg_cpu_percent': np.mean(process_cpu),
                'max_cpu_percent': np.max(process_cpu),
                'avg_memory_mb': np.mean(process_memory),
                'max_memory_mb': np.max(process_memory),
                'avg_threads': np.mean([s.process_threads for s in snapshots])
            }
        }
        
        # GPU 分析
        if snapshots[0].gpu_available:
            gpu_memory_usage = [s.gpu_memory_used_gb for s in snapshots]
            gpu_utilization = [s.gpu_utilization for s in snapshots if s.gpu_utilization > 0]
            gpu_temperature = [s.gpu_temperature for s in snapshots if s.gpu_temperature > 0]
            
            analysis['gpu'] = {
                'gpu_count': snapshots[0].gpu_count,
                'total_memory_gb': snapshots[0].gpu_memory_total_gb,
                'avg_memory_used_gb': np.mean(gpu_memory_usage),
                'max_memory_used_gb': np.max(gpu_memory_usage),
                'avg_memory_usage_percent': np.mean(gpu_memory_usage) / snapshots[0].gpu_memory_total_gb * 100 if snapshots[0].gpu_memory_total_gb > 0 else 0,
                'avg_utilization': np.mean(gpu_utilization) if gpu_utilization else 0,
                'max_utilization': np.max(gpu_utilization) if gpu_utilization else 0,
                'avg_temperature': np.mean(gpu_temperature) if gpu_temperature else 0,
                'max_temperature': np.max(gpu_temperature) if gpu_temperature else 0
            }
        
        # 缓存结果
        self.analysis_cache[cache_key] = (time.time(), analysis)
        
        return analysis
    
    def analyze_operation_performance(self, operation_name: Optional[str] = None) -> Dict:
        """分析操作性能"""
        metrics = self.profiler.get_metrics(operation_name)
        
        if not metrics:
            return {'error': '没有可用的操作指标'}
        
        # 按操作名称分组
        operations = defaultdict(list)
        for metric in metrics:
            operations[metric.operation_name].append(metric)
        
        analysis = {
            'total_operations': len(metrics),
            'operation_types': len(operations),
            'analysis_timestamp': datetime.now().isoformat(),
            'operations': {}
        }
        
        for op_name, op_metrics in operations.items():
            durations = [m.duration for m in op_metrics]
            throughputs = [m.throughput for m in op_metrics if m.throughput > 0]
            success_rate = sum(1 for m in op_metrics if m.success) / len(op_metrics)
            
            cpu_usage = [m.cpu_usage for m in op_metrics]
            memory_usage = [m.memory_usage_mb for m in op_metrics]
            gpu_memory_usage = [m.gpu_memory_usage_gb for m in op_metrics]
            
            op_analysis = {
                'count': len(op_metrics),
                'success_rate': success_rate,
                'duration': {
                    'avg': np.mean(durations),
                    'min': np.min(durations),
                    'max': np.max(durations),
                    'std': np.std(durations),
                    'p50': np.percentile(durations, 50),
                    'p95': np.percentile(durations, 95),
                    'p99': np.percentile(durations, 99)
                },
                'throughput': {
                    'avg': np.mean(throughputs) if throughputs else 0,
                    'max': np.max(throughputs) if throughputs else 0,
                    'min': np.min(throughputs) if throughputs else 0
                },
                'resource_usage': {
                    'avg_cpu_usage': np.mean(cpu_usage),
                    'avg_memory_mb': np.mean(memory_usage),
                    'avg_gpu_memory_gb': np.mean(gpu_memory_usage)
                }
            }
            
            analysis['operations'][op_name] = op_analysis
        
        return analysis
    
    def generate_optimization_recommendations(self) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        # 系统性能分析
        system_analysis = self.analyze_system_performance()
        
        if 'error' not in system_analysis:
            # CPU 优化建议
            if system_analysis['cpu']['avg_usage'] > 80:
                recommendations.append("CPU 使用率过高，考虑减少并行工作进程数或优化算法")
            elif system_analysis['cpu']['avg_usage'] < 30:
                recommendations.append("CPU 使用率较低，可以增加并行工作进程数以提高吞吐量")
            
            # 内存优化建议
            if system_analysis['memory']['avg_usage_percent'] > 85:
                recommendations.append("内存使用率过高，建议启用内存优化或减少批处理大小")
            elif system_analysis['memory']['min_available_gb'] < 2:
                recommendations.append("可用内存不足，建议增加内存清理频率")
            
            # GPU 优化建议
            if 'gpu' in system_analysis:
                gpu_analysis = system_analysis['gpu']
                if gpu_analysis['avg_memory_usage_percent'] > 90:
                    recommendations.append("GPU 内存使用率过高，建议减少批处理大小或启用梯度检查点")
                elif gpu_analysis['avg_utilization'] < 50:
                    recommendations.append("GPU 利用率较低，可以增加批处理大小或启用混合精度训练")
                
                if gpu_analysis['max_temperature'] > 80:
                    recommendations.append("GPU 温度过高，建议检查散热或降低工作负载")
        
        # 操作性能分析
        operation_analysis = self.analyze_operation_performance()
        
        if 'error' not in operation_analysis:
            for op_name, op_data in operation_analysis['operations'].items():
                if op_data['success_rate'] < 0.95:
                    recommendations.append(f"操作 '{op_name}' 成功率较低 ({op_data['success_rate']:.2%})，需要检查错误处理")
                
                if op_data['duration']['std'] > op_data['duration']['avg']:
                    recommendations.append(f"操作 '{op_name}' 执行时间不稳定，建议优化或增加预热")
        
        # 内存管理建议
        memory_stats = self.memory_manager.get_stats()
        if memory_stats.get('cleanup_count', 0) > 0:
            cleanup_frequency = memory_stats.get('cleanup_count', 0) / (time.time() - memory_stats.get('start_time', time.time()))
            if cleanup_frequency > 0.1:  # 每10秒清理一次以上
                recommendations.append("内存清理过于频繁，建议优化内存使用模式")
        
        if not recommendations:
            recommendations.append("系统性能良好，暂无优化建议")
        
        return recommendations
    
    def export_performance_report(self, output_dir: str, include_plots: bool = True) -> str:
        """导出性能报告"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成报告数据
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'system_analysis': self.analyze_system_performance(),
            'operation_analysis': self.analyze_operation_performance(),
            'optimization_recommendations': self.generate_optimization_recommendations(),
            'memory_stats': self.memory_manager.get_stats()
        }
        
        # 保存 JSON 报告
        json_path = os.path.join(output_dir, 'performance_report.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        # 生成图表
        if include_plots:
            self._generate_performance_plots(output_dir)
        
        # 生成 Markdown 报告
        markdown_path = self._generate_markdown_report(output_dir, report_data)
        
        logger.info(f"性能报告已导出到: {output_dir}")
        return markdown_path
    
    def _generate_performance_plots(self, output_dir: str):
        """生成性能图表"""
        try:
            snapshots = self.system_monitor.get_snapshots(600)  # 最近10分钟
            
            if not snapshots:
                return
            
            # 时间轴
            timestamps = [datetime.fromtimestamp(s.timestamp) for s in snapshots]
            
            # 创建子图
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('系统性能监控', fontsize=16)
            
            # CPU 使用率
            axes[0, 0].plot(timestamps, [s.cpu_percent for s in snapshots], 'b-', label='系统 CPU')
            axes[0, 0].plot(timestamps, [s.process_cpu_percent for s in snapshots], 'r-', label='进程 CPU')
            axes[0, 0].set_title('CPU 使用率 (%)')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # 内存使用率
            axes[0, 1].plot(timestamps, [s.ram_percent for s in snapshots], 'g-', label='RAM 使用率')
            axes[0, 1].plot(timestamps, [s.process_memory_mb for s in snapshots], 'orange', label='进程内存 (MB)')
            axes[0, 1].set_title('内存使用')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # GPU 内存
            if snapshots[0].gpu_available:
                axes[1, 0].plot(timestamps, [s.gpu_memory_used_gb for s in snapshots], 'm-', label='GPU 内存使用')
                axes[1, 0].plot(timestamps, [s.gpu_utilization for s in snapshots], 'c-', label='GPU 利用率')
                axes[1, 0].set_title('GPU 性能')
                axes[1, 0].legend()
                axes[1, 0].grid(True)
            else:
                axes[1, 0].text(0.5, 0.5, 'GPU 不可用', ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('GPU 性能')
            
            # 操作性能
            metrics = self.profiler.get_metrics()
            if metrics:
                operation_names = list(set(m.operation_name for m in metrics))
                throughputs = []
                for op_name in operation_names:
                    op_metrics = [m for m in metrics if m.operation_name == op_name]
                    avg_throughput = np.mean([m.throughput for m in op_metrics if m.throughput > 0])
                    throughputs.append(avg_throughput)
                
                axes[1, 1].bar(operation_names, throughputs)
                axes[1, 1].set_title('操作吞吐量 (items/s)')
                axes[1, 1].tick_params(axis='x', rotation=45)
            else:
                axes[1, 1].text(0.5, 0.5, '无操作数据', ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('操作吞吐量')
            
            plt.tight_layout()
            plot_path = os.path.join(output_dir, 'performance_plots.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"性能图表已保存: {plot_path}")
            
        except Exception as e:
            logger.error(f"生成性能图表失败: {e}")
    
    def _generate_markdown_report(self, output_dir: str, report_data: Dict) -> str:
        """生成 Markdown 报告"""
        markdown_path = os.path.join(output_dir, 'performance_report.md')
        
        with open(markdown_path, 'w', encoding='utf-8') as f:
            f.write("# 性能分析报告\n\n")
            f.write(f"**生成时间**: {report_data['timestamp']}\n\n")
            
            # 系统性能摘要
            if 'error' not in report_data['system_analysis']:
                sys_analysis = report_data['system_analysis']
                f.write("## 系统性能摘要\n\n")
                f.write(f"- **监控时长**: {sys_analysis['monitoring_duration_minutes']} 分钟\n")
                f.write(f"- **采样数量**: {sys_analysis['sample_count']}\n")
                f.write(f"- **CPU 平均使用率**: {sys_analysis['cpu']['avg_usage']:.1f}%\n")
                f.write(f"- **内存平均使用率**: {sys_analysis['memory']['avg_usage_percent']:.1f}%\n")
                
                if 'gpu' in sys_analysis:
                    f.write(f"- **GPU 平均使用率**: {sys_analysis['gpu']['avg_utilization']:.1f}%\n")
                    f.write(f"- **GPU 内存使用率**: {sys_analysis['gpu']['avg_memory_usage_percent']:.1f}%\n")
                
                f.write("\n")
            
            # 优化建议
            f.write("## 优化建议\n\n")
            for i, recommendation in enumerate(report_data['optimization_recommendations'], 1):
                f.write(f"{i}. {recommendation}\n")
            f.write("\n")
            
            # 操作性能详情
            if 'error' not in report_data['operation_analysis']:
                op_analysis = report_data['operation_analysis']
                f.write("## 操作性能详情\n\n")
                f.write(f"- **总操作数**: {op_analysis['total_operations']}\n")
                f.write(f"- **操作类型数**: {op_analysis['operation_types']}\n\n")
                
                for op_name, op_data in op_analysis['operations'].items():
                    f.write(f"### {op_name}\n\n")
                    f.write(f"- **执行次数**: {op_data['count']}\n")
                    f.write(f"- **成功率**: {op_data['success_rate']:.2%}\n")
                    f.write(f"- **平均耗时**: {op_data['duration']['avg']:.3f}s\n")
                    f.write(f"- **平均吞吐量**: {op_data['throughput']['avg']:.1f} items/s\n")
                    f.write("\n")
            
            # 性能图表
            plot_path = os.path.join(output_dir, 'performance_plots.png')
            if os.path.exists(plot_path):
                f.write("## 性能图表\n\n")
                f.write("![性能监控图表](performance_plots.png)\n\n")
        
        return markdown_path
    
    def cleanup(self):
        """清理资源"""
        self.stop_monitoring()
        self.profiler.clear_metrics()
        self.analysis_cache.clear()
        logger.info("性能分析器资源清理完成")

# 全局性能分析器实例
_global_analyzer = None

def get_global_performance_analyzer() -> PerformanceAnalyzer:
    """获取全局性能分析器"""
    global _global_analyzer
    if _global_analyzer is None:
        _global_analyzer = PerformanceAnalyzer()
    return _global_analyzer

def performance_profile(operation_name: str, items_count: int = 1, **metadata):
    """性能分析装饰器"""
    analyzer = get_global_performance_analyzer()
    return analyzer.profile_operation(operation_name, items_count, **metadata)

if __name__ == "__main__":
    # 测试代码
    analyzer = PerformanceAnalyzer()
    analyzer.start_monitoring()
    
    # 模拟一些操作
    @analyzer.profile_operation("test_operation", items_count=100)
    def test_operation():
        time.sleep(0.1)
        return "completed"
    
    # 执行测试操作
    for i in range(5):
        test_operation()
        time.sleep(0.5)
    
    # 等待监控数据
    time.sleep(2)
    
    # 分析性能
    system_analysis = analyzer.analyze_system_performance(1)
    operation_analysis = analyzer.analyze_operation_performance()
    recommendations = analyzer.generate_optimization_recommendations()
    
    print("系统分析:", system_analysis)
    print("操作分析:", operation_analysis)
    print("优化建议:", recommendations)
    
    # 导出报告
    report_path = analyzer.export_performance_report('./performance_report')
    print(f"报告已导出: {report_path}")
    
    # 清理
    analyzer.cleanup()
    
    print("测试完成")