#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化配置管理器
统一管理所有性能优化相关的配置
"""

import os
import yaml
import json
import logging
import torch
import psutil
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)

class OptimizationLevel(Enum):
    """优化级别"""
    CONSERVATIVE = "conservative"  # 保守优化，优先稳定性
    BALANCED = "balanced"         # 平衡优化，性能与稳定性并重
    AGGRESSIVE = "aggressive"     # 激进优化，优先性能
    CUSTOM = "custom"             # 自定义配置

class DeviceType(Enum):
    """设备类型"""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon
    AUTO = "auto"

@dataclass
class MemoryConfig:
    """内存配置"""
    # 内存监控
    enable_monitoring: bool = True
    monitoring_interval: float = 1.0
    
    # 内存清理
    auto_cleanup: bool = True
    cleanup_threshold: float = 0.8  # 内存使用率阈值
    aggressive_cleanup_threshold: float = 0.9
    cleanup_interval: float = 30.0  # 清理间隔（秒）
    
    # OOM 预防
    enable_oom_prevention: bool = True
    oom_threshold: float = 0.95
    emergency_cleanup: bool = True
    
    # CUDA 内存
    cuda_empty_cache: bool = True
    cuda_memory_fraction: Optional[float] = None  # 限制 CUDA 内存使用
    
    # Python GC
    enable_gc: bool = True
    gc_threshold: tuple = (700, 10, 10)

@dataclass
class BatchConfig:
    """批处理配置"""
    # 基础配置
    initial_batch_size: int = 8
    min_batch_size: int = 1
    max_batch_size: int = 64
    
    # 自适应调整
    enable_adaptive: bool = True
    adaptation_factor: float = 1.2
    success_threshold: float = 0.95
    memory_threshold: float = 0.8
    
    # 优化策略
    enable_gradient_accumulation: bool = False
    accumulation_steps: int = 4
    
    # 批处理大小优化
    enable_batch_size_finder: bool = True
    finder_max_trials: int = 10
    finder_growth_factor: float = 2.0

@dataclass
class DataLoaderConfig:
    """数据加载配置"""
    # 基础配置
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    
    # 缓存配置
    enable_caching: bool = True
    cache_size: int = 1000
    cache_policy: str = "lru"  # lru, fifo, random
    
    # 预处理优化
    enable_async_transform: bool = True
    transform_workers: int = 2
    
    # 预取配置
    enable_prefetch: bool = True
    prefetch_buffer_size: int = 10
    
    # 自动优化
    auto_optimize: bool = True
    benchmark_duration: float = 10.0

@dataclass
class ModelConfig:
    """模型配置"""
    # 设备配置
    device: str = "auto"
    device_ids: Optional[List[int]] = None
    
    # 精度配置
    mixed_precision: bool = True
    amp_enabled: bool = True
    fp16: bool = False
    bf16: bool = False
    
    # 编译优化
    torch_compile: bool = False
    compile_mode: str = "default"  # default, reduce-overhead, max-autotune
    compile_dynamic: bool = False
    
    # 内存优化
    gradient_checkpointing: bool = False
    channels_last: bool = True
    
    # TensorRT 优化
    enable_tensorrt: bool = False
    tensorrt_precision: str = "fp16"  # fp32, fp16, int8
    
    # 模型并行
    data_parallel: bool = False
    distributed: bool = False

@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础配置
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # 优化器配置
    optimizer: str = "adamw"  # adam, adamw, sgd, rmsprop
    optimizer_params: Dict[str, Any] = field(default_factory=dict)
    
    # 学习率调度
    scheduler: str = "cosine"  # cosine, step, plateau, linear
    scheduler_params: Dict[str, Any] = field(default_factory=dict)
    warmup_steps: int = 0
    warmup_ratio: float = 0.1
    
    # 早停配置
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 1e-4
    
    # 验证配置
    validation_interval: int = 1
    validation_ratio: float = 0.2
    
    # 检查点配置
    save_interval: int = 5
    keep_checkpoints: int = 3
    
    # 确定性训练
    deterministic: bool = False
    seed: int = 42

@dataclass
class MonitoringConfig:
    """监控配置"""
    # 性能监控
    enable_performance_monitoring: bool = True
    monitoring_interval: float = 1.0
    
    # 指标收集
    collect_system_metrics: bool = True
    collect_gpu_metrics: bool = True
    collect_process_metrics: bool = True
    
    # 日志配置
    log_level: str = "INFO"
    log_to_file: bool = True
    log_file: str = "optimization.log"
    
    # 报告配置
    generate_reports: bool = True
    report_interval: int = 100  # 每100个epoch生成一次报告
    export_plots: bool = True
    
    # 警告配置
    enable_warnings: bool = True
    memory_warning_threshold: float = 0.9
    temperature_warning_threshold: float = 80.0

@dataclass
class OptimizationConfig:
    """完整的优化配置"""
    # 优化级别
    optimization_level: OptimizationLevel = OptimizationLevel.BALANCED
    
    # 子配置
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # 全局配置
    enable_optimization: bool = True
    profile_operations: bool = True
    
    # 实验配置
    experiment_name: str = "optimization_experiment"
    output_dir: str = "./optimization_output"
    
    def __post_init__(self):
        """初始化后处理"""
        self._apply_optimization_level()
        self._auto_detect_hardware()
        self._validate_config()
    
    def _apply_optimization_level(self):
        """应用优化级别"""
        if self.optimization_level == OptimizationLevel.CONSERVATIVE:
            self._apply_conservative_settings()
        elif self.optimization_level == OptimizationLevel.BALANCED:
            self._apply_balanced_settings()
        elif self.optimization_level == OptimizationLevel.AGGRESSIVE:
            self._apply_aggressive_settings()
        # CUSTOM 级别不做任何修改
    
    def _apply_conservative_settings(self):
        """应用保守设置"""
        # 内存配置
        self.memory.cleanup_threshold = 0.7
        self.memory.aggressive_cleanup_threshold = 0.8
        self.memory.enable_oom_prevention = True
        
        # 批处理配置
        self.batch.initial_batch_size = 4
        self.batch.max_batch_size = 16
        self.batch.adaptation_factor = 1.1
        
        # 数据加载配置
        self.dataloader.num_workers = min(4, psutil.cpu_count())
        self.dataloader.prefetch_factor = 1
        
        # 模型配置
        self.model.mixed_precision = False
        self.model.torch_compile = False
        self.model.gradient_checkpointing = True
        
        # 训练配置
        self.training.deterministic = True
    
    def _apply_balanced_settings(self):
        """应用平衡设置"""
        # 内存配置
        self.memory.cleanup_threshold = 0.8
        self.memory.aggressive_cleanup_threshold = 0.9
        
        # 批处理配置
        self.batch.initial_batch_size = 8
        self.batch.max_batch_size = 32
        
        # 数据加载配置
        self.dataloader.num_workers = min(8, psutil.cpu_count())
        
        # 模型配置
        self.model.mixed_precision = True
        self.model.channels_last = True
    
    def _apply_aggressive_settings(self):
        """应用激进设置"""
        # 内存配置
        self.memory.cleanup_threshold = 0.85
        self.memory.aggressive_cleanup_threshold = 0.95
        
        # 批处理配置
        self.batch.initial_batch_size = 16
        self.batch.max_batch_size = 64
        self.batch.adaptation_factor = 1.5
        
        # 数据加载配置
        self.dataloader.num_workers = psutil.cpu_count()
        self.dataloader.prefetch_factor = 4
        
        # 模型配置
        self.model.mixed_precision = True
        self.model.torch_compile = True
        self.model.channels_last = True
        
        # 训练配置
        self.training.gradient_checkpointing = False
    
    def _auto_detect_hardware(self):
        """自动检测硬件配置"""
        if self.model.device == "auto":
            if torch.cuda.is_available():
                self.model.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.model.device = "mps"
            else:
                self.model.device = "cpu"
        
        # 根据设备调整配置
        if self.model.device == "cpu":
            self.model.mixed_precision = False
            self.model.amp_enabled = False
            self.dataloader.pin_memory = False
        
        # 检测 GPU 数量
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count > 1 and self.model.device_ids is None:
                self.model.device_ids = list(range(gpu_count))
                self.model.data_parallel = True
        
        # 根据内存大小调整批处理
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        if available_memory_gb < 8:
            self.batch.max_batch_size = min(self.batch.max_batch_size, 16)
            self.dataloader.num_workers = min(self.dataloader.num_workers, 2)
        elif available_memory_gb > 32:
            self.batch.max_batch_size = max(self.batch.max_batch_size, 32)
    
    def _validate_config(self):
        """验证配置"""
        # 验证批处理大小
        if self.batch.min_batch_size > self.batch.max_batch_size:
            raise ValueError("min_batch_size 不能大于 max_batch_size")
        
        if self.batch.initial_batch_size < self.batch.min_batch_size:
            self.batch.initial_batch_size = self.batch.min_batch_size
        
        if self.batch.initial_batch_size > self.batch.max_batch_size:
            self.batch.initial_batch_size = self.batch.max_batch_size
        
        # 验证数据加载器配置
        if self.dataloader.num_workers < 0:
            self.dataloader.num_workers = 0
        
        # 验证内存阈值
        if not 0 < self.memory.cleanup_threshold < 1:
            raise ValueError("cleanup_threshold 必须在 0 和 1 之间")
        
        if self.memory.aggressive_cleanup_threshold <= self.memory.cleanup_threshold:
            self.memory.aggressive_cleanup_threshold = self.memory.cleanup_threshold + 0.1
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    def save(self, file_path: str):
        """保存配置到文件"""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self.to_dict()
        
        # 处理枚举类型
        config_dict['optimization_level'] = self.optimization_level.value
        
        if file_path.suffix.lower() == '.yaml' or file_path.suffix.lower() == '.yml':
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        else:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"配置已保存到: {file_path}")
    
    @classmethod
    def load(cls, file_path: str) -> 'OptimizationConfig':
        """从文件加载配置"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {file_path}")
        
        if file_path.suffix.lower() in ['.yaml', '.yml']:
            with open(file_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
        
        # 处理枚举类型
        if 'optimization_level' in config_dict:
            config_dict['optimization_level'] = OptimizationLevel(config_dict['optimization_level'])
        
        # 递归创建配置对象
        return cls._from_dict(config_dict)
    
    @classmethod
    def _from_dict(cls, config_dict: Dict[str, Any]) -> 'OptimizationConfig':
        """从字典创建配置对象"""
        # 创建子配置对象
        if 'memory' in config_dict:
            config_dict['memory'] = MemoryConfig(**config_dict['memory'])
        
        if 'batch' in config_dict:
            config_dict['batch'] = BatchConfig(**config_dict['batch'])
        
        if 'dataloader' in config_dict:
            config_dict['dataloader'] = DataLoaderConfig(**config_dict['dataloader'])
        
        if 'model' in config_dict:
            config_dict['model'] = ModelConfig(**config_dict['model'])
        
        if 'training' in config_dict:
            config_dict['training'] = TrainingConfig(**config_dict['training'])
        
        if 'monitoring' in config_dict:
            config_dict['monitoring'] = MonitoringConfig(**config_dict['monitoring'])
        
        return cls(**config_dict)
    
    def update(self, **kwargs):
        """更新配置"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"未知配置项: {key}")
        
        # 重新验证配置
        self._validate_config()
    
    def get_device_info(self) -> Dict[str, Any]:
        """获取设备信息"""
        info = {
            'device': self.model.device,
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'cuda_available': torch.cuda.is_available()
        }
        
        if torch.cuda.is_available():
            info.update({
                'cuda_device_count': torch.cuda.device_count(),
                'cuda_devices': []
            })
            
            for i in range(torch.cuda.device_count()):
                device_props = torch.cuda.get_device_properties(i)
                info['cuda_devices'].append({
                    'name': device_props.name,
                    'memory_gb': device_props.total_memory / (1024**3),
                    'compute_capability': f"{device_props.major}.{device_props.minor}"
                })
        
        if hasattr(torch.backends, 'mps'):
            info['mps_available'] = torch.backends.mps.is_available()
        
        return info
    
    def print_summary(self):
        """打印配置摘要"""
        print("\n" + "="*50)
        print("优化配置摘要")
        print("="*50)
        
        print(f"优化级别: {self.optimization_level.value}")
        print(f"设备: {self.model.device}")
        print(f"混合精度: {self.model.mixed_precision}")
        print(f"批处理大小: {self.batch.initial_batch_size} (范围: {self.batch.min_batch_size}-{self.batch.max_batch_size})")
        print(f"数据加载器工作进程: {self.dataloader.num_workers}")
        print(f"内存监控: {self.memory.enable_monitoring}")
        print(f"性能监控: {self.monitoring.enable_performance_monitoring}")
        
        device_info = self.get_device_info()
        print(f"\n硬件信息:")
        print(f"  CPU 核心数: {device_info['cpu_count']}")
        print(f"  系统内存: {device_info['memory_gb']:.1f} GB")
        
        if device_info['cuda_available']:
            print(f"  CUDA 设备数: {device_info['cuda_device_count']}")
            for i, gpu in enumerate(device_info['cuda_devices']):
                print(f"    GPU {i}: {gpu['name']} ({gpu['memory_gb']:.1f} GB)")
        
        print("="*50 + "\n")

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_dir: str = "./configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self._current_config: Optional[OptimizationConfig] = None
        self._config_history: List[OptimizationConfig] = []
    
    def create_config(self, optimization_level: OptimizationLevel = OptimizationLevel.BALANCED) -> OptimizationConfig:
        """创建新配置"""
        config = OptimizationConfig(optimization_level=optimization_level)
        self._current_config = config
        return config
    
    def load_config(self, config_name: str) -> OptimizationConfig:
        """加载配置"""
        config_path = self.config_dir / f"{config_name}.yaml"
        if not config_path.exists():
            config_path = self.config_dir / f"{config_name}.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_name}")
        
        config = OptimizationConfig.load(config_path)
        self._current_config = config
        return config
    
    def save_config(self, config: OptimizationConfig, config_name: str, format: str = "yaml"):
        """保存配置"""
        if format.lower() == "yaml":
            config_path = self.config_dir / f"{config_name}.yaml"
        else:
            config_path = self.config_dir / f"{config_name}.json"
        
        config.save(config_path)
        
        # 添加到历史记录
        self._config_history.append(config)
    
    def get_current_config(self) -> Optional[OptimizationConfig]:
        """获取当前配置"""
        return self._current_config
    
    def list_configs(self) -> List[str]:
        """列出所有配置"""
        configs = []
        for file_path in self.config_dir.glob("*.yaml"):
            configs.append(file_path.stem)
        for file_path in self.config_dir.glob("*.json"):
            if file_path.stem not in configs:
                configs.append(file_path.stem)
        return sorted(configs)
    
    def create_preset_configs(self):
        """创建预设配置"""
        presets = {
            "conservative": OptimizationLevel.CONSERVATIVE,
            "balanced": OptimizationLevel.BALANCED,
            "aggressive": OptimizationLevel.AGGRESSIVE
        }
        
        for name, level in presets.items():
            config = OptimizationConfig(optimization_level=level)
            self.save_config(config, name)
        
        logger.info("预设配置已创建")
    
    def compare_configs(self, config1_name: str, config2_name: str) -> Dict[str, Any]:
        """比较两个配置"""
        config1 = self.load_config(config1_name)
        config2 = self.load_config(config2_name)
        
        dict1 = config1.to_dict()
        dict2 = config2.to_dict()
        
        differences = {}
        
        def compare_dicts(d1, d2, path=""):
            for key in set(d1.keys()) | set(d2.keys()):
                current_path = f"{path}.{key}" if path else key
                
                if key not in d1:
                    differences[current_path] = {"config1": None, "config2": d2[key]}
                elif key not in d2:
                    differences[current_path] = {"config1": d1[key], "config2": None}
                elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
                    compare_dicts(d1[key], d2[key], current_path)
                elif d1[key] != d2[key]:
                    differences[current_path] = {"config1": d1[key], "config2": d2[key]}
        
        compare_dicts(dict1, dict2)
        
        return {
            "config1": config1_name,
            "config2": config2_name,
            "differences": differences
        }

# 全局配置管理器
_global_config_manager = None

def get_global_config_manager() -> ConfigManager:
    """获取全局配置管理器"""
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = ConfigManager()
    return _global_config_manager

def create_optimization_config(optimization_level: OptimizationLevel = OptimizationLevel.BALANCED) -> OptimizationConfig:
    """创建优化配置"""
    return OptimizationConfig(optimization_level=optimization_level)

if __name__ == "__main__":
    # 测试代码
    print("创建配置管理器...")
    manager = ConfigManager("./test_configs")
    
    # 创建预设配置
    print("创建预设配置...")
    manager.create_preset_configs()
    
    # 加载并显示配置
    print("\n加载平衡配置:")
    config = manager.load_config("balanced")
    config.print_summary()
    
    # 创建自定义配置
    print("创建自定义配置...")
    custom_config = OptimizationConfig(optimization_level=OptimizationLevel.CUSTOM)
    custom_config.batch.initial_batch_size = 16
    custom_config.model.mixed_precision = True
    custom_config.training.learning_rate = 2e-4
    
    manager.save_config(custom_config, "custom")
    
    # 比较配置
    print("\n比较配置:")
    comparison = manager.compare_configs("balanced", "custom")
    print(f"配置差异数量: {len(comparison['differences'])}")
    
    for path, diff in comparison['differences'].items():
        print(f"  {path}: {diff['config1']} -> {diff['config2']}")
    
    print("\n测试完成")