#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化工具集成脚本
将优化工具集成到现有的预测和训练代码中
"""

import os
import sys
import argparse
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Any

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OptimizationIntegrator:
    """优化工具集成器"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.src_dir = self.project_root / "src"
        self.utils_dir = self.src_dir / "utils"
        
        # 检查项目结构
        self._validate_project_structure()
    
    def _validate_project_structure(self):
        """验证项目结构"""
        required_dirs = [self.src_dir, self.utils_dir]
        
        for dir_path in required_dirs:
            if not dir_path.exists():
                logger.warning(f"目录不存在: {dir_path}")
    
    def create_optimized_predict_wrapper(self) -> str:
        """创建优化的预测包装器"""
        wrapper_path = self.src_dir / "optimized_predict.py"
        
        wrapper_code = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化的预测包装器
在原有预测功能基础上添加优化功能
"""

import os
import sys
import time
import torch
import logging
from typing import List, Union, Optional, Any
from pathlib import Path

# 导入原有预测模块
try:
    from .predict import WatermarkPredictor, predict_mask
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    try:
        from predict import WatermarkPredictor, predict_mask
    except ImportError:
        logger.warning("无法导入原有预测模块，请检查路径")
        WatermarkPredictor = None
        predict_mask = None

# 导入优化工具
from .utils.optimization_manager import (
    OptimizationManager,
    create_optimization_manager,
    OptimizationLevel
)
from .utils.optimization_config import OptimizationConfig
from .utils.performance_analyzer import performance_profile
from .utils.enhanced_memory_manager import memory_optimized

logger = logging.getLogger(__name__)

class OptimizedWatermarkPredictor:
    """优化的水印预测器"""
    
    def __init__(self, model_path: str, optimization_level: OptimizationLevel = OptimizationLevel.BALANCED):
        self.model_path = model_path
        self.optimization_level = optimization_level
        
        # 创建优化管理器
        self.optimization_manager = create_optimization_manager(optimization_level)
        
        # 初始化原有预测器
        if WatermarkPredictor:
            self.predictor = WatermarkPredictor(model_path)
            
            # 优化模型
            if hasattr(self.predictor, 'model'):
                self.optimization_manager.optimize_model(self.predictor.model)
        else:
            self.predictor = None
            logger.warning("原有预测器不可用")
        
        # 创建优化的预测器
        if self.predictor and hasattr(self.predictor, 'model'):
            self.optimized_predictor = self.optimization_manager.create_optimized_predictor(
                self.predictor.model
            )
        else:
            self.optimized_predictor = None
    
    @memory_optimized
    @performance_profile("single_prediction")
    def predict(self, image_path: str, **kwargs) -> Any:
        """预测单张图片"""
        if self.optimized_predictor:
            return self.optimized_predictor.predict_from_path(image_path, **kwargs)
        elif self.predictor:
            return self.predictor.predict(image_path, **kwargs)
        else:
            raise RuntimeError("预测器未正确初始化")
    
    @memory_optimized
    @performance_profile("batch_prediction")
    def predict_batch(self, image_paths: List[str], **kwargs) -> List[Any]:
        """批量预测"""
        if self.optimized_predictor:
            return self.optimized_predictor.predict_batch_from_paths(image_paths, **kwargs)
        elif self.predictor:
            # 使用原有预测器的批量预测或循环预测
            results = []
            for image_path in image_paths:
                result = self.predictor.predict(image_path, **kwargs)
                results.append(result)
            return results
        else:
            raise RuntimeError("预测器未正确初始化")
    
    def get_optimization_summary(self) -> dict:
        """获取优化摘要"""
        return self.optimization_manager.get_optimization_summary()
    
    def generate_performance_report(self, output_dir: str) -> str:
        """生成性能报告"""
        return self.optimization_manager.generate_performance_report(output_dir)
    
    def cleanup(self):
        """清理资源"""
        if self.optimization_manager:
            self.optimization_manager.cleanup()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

@memory_optimized
@performance_profile("optimized_predict_mask")
def optimized_predict_mask(image_path: str, model_path: str, **kwargs) -> Any:
    """优化的预测掩码函数"""
    with OptimizedWatermarkPredictor(model_path) as predictor:
        return predictor.predict(image_path, **kwargs)

@memory_optimized
@performance_profile("optimized_batch_predict_mask")
def optimized_batch_predict_mask(image_paths: List[str], model_path: str, **kwargs) -> List[Any]:
    """优化的批量预测掩码函数"""
    with OptimizedWatermarkPredictor(model_path) as predictor:
        return predictor.predict_batch(image_paths, **kwargs)

# 向后兼容的函数
def create_optimized_predictor(model_path: str, optimization_level: str = "balanced") -> OptimizedWatermarkPredictor:
    """创建优化的预测器"""
    level_map = {
        "conservative": OptimizationLevel.CONSERVATIVE,
        "balanced": OptimizationLevel.BALANCED,
        "aggressive": OptimizationLevel.AGGRESSIVE,
        "custom": OptimizationLevel.CUSTOM
    }
    
    level = level_map.get(optimization_level.lower(), OptimizationLevel.BALANCED)
    return OptimizedWatermarkPredictor(model_path, level)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="优化的水印预测")
    parser.add_argument("--model", required=True, help="模型路径")
    parser.add_argument("--input", required=True, help="输入图片路径或目录")
    parser.add_argument("--output", help="输出目录")
    parser.add_argument("--optimization", default="balanced", 
                       choices=["conservative", "balanced", "aggressive"],
                       help="优化级别")
    parser.add_argument("--batch-size", type=int, default=8, help="批处理大小")
    parser.add_argument("--report", help="性能报告输出目录")
    
    args = parser.parse_args()
    
    # 创建优化预测器
    predictor = create_optimized_predictor(args.model, args.optimization)
    
    try:
        input_path = Path(args.input)
        
        if input_path.is_file():
            # 单文件预测
            print(f"预测单张图片: {input_path}")
            result = predictor.predict(str(input_path))
            print(f"预测结果: {result}")
            
        elif input_path.is_dir():
            # 批量预测
            image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
            image_paths = [
                str(p) for p in input_path.rglob("*")
                if p.suffix.lower() in image_extensions
            ]
            
            print(f"找到 {len(image_paths)} 张图片")
            
            if image_paths:
                print("开始批量预测...")
                results = predictor.predict_batch(image_paths)
                print(f"预测完成，处理了 {len(results)} 张图片")
        
        # 生成性能报告
        if args.report:
            print(f"生成性能报告: {args.report}")
            report_path = predictor.generate_performance_report(args.report)
            print(f"报告已保存: {report_path}")
        
        # 显示优化摘要
        summary = predictor.get_optimization_summary()
        print("\n优化摘要:")
        for key, value in summary.get('config', {}).items():
            print(f"  {key}: {value}")
    
    finally:
        predictor.cleanup()
'''
        
        with open(wrapper_path, 'w', encoding='utf-8') as f:
            f.write(wrapper_code)
        
        logger.info(f"优化预测包装器已创建: {wrapper_path}")
        return str(wrapper_path)
    
    def create_optimized_training_wrapper(self) -> str:
        """创建优化的训练包装器"""
        wrapper_path = self.src_dir / "optimized_training.py"
        
        wrapper_code = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化的训练包装器
在原有训练功能基础上添加优化功能
"""

import os
import sys
import time
import torch
import torch.nn as nn
import logging
from typing import Optional, Dict, Any
from pathlib import Path

# 导入优化工具
from .utils.optimization_manager import (
    OptimizationManager,
    create_optimization_manager,
    OptimizationLevel
)
from .utils.optimization_config import OptimizationConfig
from .utils.performance_analyzer import performance_profile
from .utils.enhanced_memory_manager import memory_optimized

logger = logging.getLogger(__name__)

class OptimizedTrainer:
    """优化的训练器"""
    
    def __init__(self, 
                 model: nn.Module,
                 train_dataset,
                 val_dataset=None,
                 optimization_level: OptimizationLevel = OptimizationLevel.BALANCED,
                 **training_kwargs):
        
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        # 创建优化管理器
        self.optimization_manager = create_optimization_manager(optimization_level)
        
        # 优化模型
        self.optimization_manager.optimize_model(model)
        
        # 创建优化的数据加载器
        self.train_dataloader = self.optimization_manager.create_optimized_dataloader(
            train_dataset,
            batch_size=training_kwargs.get('batch_size', 8),
            shuffle=True
        )
        
        if val_dataset:
            self.val_dataloader = self.optimization_manager.create_optimized_dataloader(
                val_dataset,
                batch_size=training_kwargs.get('batch_size', 8),
                shuffle=False
            )
        else:
            self.val_dataloader = None
        
        # 创建训练优化器
        self.training_optimizer = self.optimization_manager.create_training_optimizer(
            model, **training_kwargs
        )
    
    @memory_optimized
    @performance_profile("training_epoch")
    def train_epoch(self, epoch: int, criterion) -> Dict[str, Any]:
        """训练一个epoch"""
        return self.training_optimizer.train_epoch(
            self.train_dataset, criterion, epoch=epoch
        )
    
    @memory_optimized
    @performance_profile("validation")
    def validate(self, criterion) -> Dict[str, Any]:
        """验证"""
        if self.val_dataset:
            return self.training_optimizer.validate(self.val_dataset, criterion)
        else:
            return {}
    
    def train(self, 
              criterion,
              epochs: int,
              save_dir: Optional[str] = None,
              save_interval: int = 5) -> Dict[str, Any]:
        """完整训练流程"""
        
        logger.info(f"开始训练，共 {epochs} 个epoch")
        
        best_val_loss = float('inf')
        training_history = {
            'train_losses': [],
            'val_losses': [],
            'train_accuracies': [],
            'val_accuracies': []
        }
        
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            
            # 训练
            train_metrics = self.train_epoch(epoch, criterion)
            training_history['train_losses'].append(train_metrics.avg_loss)
            training_history['train_accuracies'].append(train_metrics.accuracy)
            
            logger.info(f"训练损失: {train_metrics.avg_loss:.4f}, "
                       f"训练准确率: {train_metrics.accuracy:.2%}, "
                       f"吞吐量: {train_metrics.throughput:.1f} samples/s")
            
            # 验证
            if self.val_dataset:
                val_metrics = self.validate(criterion)
                training_history['val_losses'].append(val_metrics.avg_loss)
                training_history['val_accuracies'].append(val_metrics.accuracy)
                
                logger.info(f"验证损失: {val_metrics.avg_loss:.4f}, "
                           f"验证准确率: {val_metrics.accuracy:.2%}")
                
                # 保存最佳模型
                if val_metrics.avg_loss < best_val_loss:
                    best_val_loss = val_metrics.avg_loss
                    if save_dir:
                        self.save_checkpoint(save_dir, epoch, "best")
            
            # 定期保存检查点
            if save_dir and (epoch + 1) % save_interval == 0:
                self.save_checkpoint(save_dir, epoch, "latest")
        
        # 获取训练统计
        training_stats = self.training_optimizer.get_training_stats()
        training_history.update(training_stats)
        
        logger.info("训练完成")
        return training_history
    
    def save_checkpoint(self, save_dir: str, epoch: int, checkpoint_type: str = "latest"):
        """保存检查点"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.training_optimizer.optimizer.state_dict(),
            'training_config': self.training_optimizer.config.__dict__
        }
        
        if hasattr(self.training_optimizer, 'scheduler') and self.training_optimizer.scheduler:
            checkpoint['scheduler_state_dict'] = self.training_optimizer.scheduler.state_dict()
        
        checkpoint_path = save_path / f"checkpoint_{checkpoint_type}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        logger.info(f"检查点已保存: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.model.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.training_optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.training_optimizer.scheduler:
            self.training_optimizer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        epoch = checkpoint.get('epoch', 0)
        logger.info(f"检查点已加载: {checkpoint_path}, epoch: {epoch}")
        
        return epoch
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """获取优化摘要"""
        return self.optimization_manager.get_optimization_summary()
    
    def generate_performance_report(self, output_dir: str) -> str:
        """生成性能报告"""
        return self.optimization_manager.generate_performance_report(output_dir)
    
    def cleanup(self):
        """清理资源"""
        if self.training_optimizer:
            self.training_optimizer.cleanup()
        if self.optimization_manager:
            self.optimization_manager.cleanup()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

def create_optimized_trainer(model: nn.Module,
                           train_dataset,
                           val_dataset=None,
                           optimization_level: str = "balanced",
                           **kwargs) -> OptimizedTrainer:
    """创建优化的训练器"""
    level_map = {
        "conservative": OptimizationLevel.CONSERVATIVE,
        "balanced": OptimizationLevel.BALANCED,
        "aggressive": OptimizationLevel.AGGRESSIVE,
        "custom": OptimizationLevel.CUSTOM
    }
    
    level = level_map.get(optimization_level.lower(), OptimizationLevel.BALANCED)
    return OptimizedTrainer(model, train_dataset, val_dataset, level, **kwargs)

if __name__ == "__main__":
    # 示例使用
    print("优化训练包装器")
    print("请在您的训练脚本中导入并使用 OptimizedTrainer 或 create_optimized_trainer")
'''
        
        with open(wrapper_path, 'w', encoding='utf-8') as f:
            f.write(wrapper_code)
        
        logger.info(f"优化训练包装器已创建: {wrapper_path}")
        return str(wrapper_path)
    
    def create_integration_guide(self) -> str:
        """创建集成指南"""
        guide_path = self.project_root / "OPTIMIZATION_INTEGRATION_GUIDE.md"
        
        guide_content = '''# 优化工具集成指南

本指南介绍如何将优化工具集成到现有的水印检测项目中。

## 概述

优化工具包含以下主要组件：

1. **内存管理器** - 智能内存监控和清理
2. **批处理优化器** - 自适应批处理大小调整
3. **数据加载器优化** - 高效的数据加载和预处理
4. **模型优化器** - PyTorch 模型性能优化
5. **训练优化器** - 训练过程优化
6. **性能分析器** - 全面的性能监控和分析
7. **配置管理器** - 统一的配置管理
8. **优化管理器** - 集成所有优化组件

## 快速开始

### 1. 优化现有预测代码

```python
# 原有代码
from src.predict import WatermarkPredictor

predictor = WatermarkPredictor("model.pth")
result = predictor.predict("image.jpg")

# 优化后代码
from src.optimized_predict import OptimizedWatermarkPredictor

with OptimizedWatermarkPredictor("model.pth") as predictor:
    result = predictor.predict("image.jpg")
    # 自动应用内存优化、性能监控等
```

### 2. 优化现有训练代码

```python
# 原有代码
model = create_model()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(epochs):
    for batch in dataloader:
        # 训练逻辑
        pass

# 优化后代码
from src.optimized_training import OptimizedTrainer

with OptimizedTrainer(model, train_dataset, val_dataset) as trainer:
    history = trainer.train(criterion, epochs=100, save_dir="./checkpoints")
    # 自动应用所有优化
```

### 3. 使用优化管理器

```python
from src.utils.optimization_manager import create_optimization_manager, OptimizationLevel

# 创建优化管理器
manager = create_optimization_manager(OptimizationLevel.BALANCED)

# 优化模型
manager.optimize_model(model)

# 创建优化的数据加载器
dataloader = manager.create_optimized_dataloader(dataset)

# 创建优化的预测器
predictor = manager.create_optimized_predictor(model)

# 生成性能报告
report_path = manager.generate_performance_report("./reports")
```

## 配置选项

### 优化级别

- **CONSERVATIVE**: 保守优化，优先稳定性
- **BALANCED**: 平衡优化，性能与稳定性并重（推荐）
- **AGGRESSIVE**: 激进优化，优先性能
- **CUSTOM**: 自定义配置

### 自定义配置

```python
from src.utils.optimization_config import OptimizationConfig, OptimizationLevel

config = OptimizationConfig(optimization_level=OptimizationLevel.CUSTOM)
config.batch.initial_batch_size = 16
config.model.mixed_precision = True
config.memory.cleanup_threshold = 0.8

manager = OptimizationManager(config)
```

## 性能监控

### 使用装饰器

```python
from src.utils.performance_analyzer import performance_profile
from src.utils.enhanced_memory_manager import memory_optimized

@memory_optimized
@performance_profile("my_operation", items_count=100)
def my_function():
    # 您的代码
    pass
```

### 手动监控

```python
from src.utils.performance_analyzer import get_global_performance_analyzer

analyzer = get_global_performance_analyzer()
analyzer.start_monitoring()

# 您的代码

# 分析性能
system_analysis = analyzer.analyze_system_performance()
operation_analysis = analyzer.analyze_operation_performance()
recommendations = analyzer.generate_optimization_recommendations()
```

## 内存管理

### 自动内存管理

```python
from src.utils.enhanced_memory_manager import get_global_memory_manager

memory_manager = get_global_memory_manager()
memory_manager.start_monitoring()  # 自动清理内存
```

### 手动内存管理

```python
# 使用上下文管理器
with memory_manager.memory_context():
    # 内存密集型操作
    large_tensor = torch.randn(10000, 10000)
    # 自动清理

# 手动清理
memory_manager.cleanup()
memory_manager.aggressive_cleanup()
```

## 批处理优化

### 自适应批处理

```python
from src.utils.adaptive_batch_processor import AdaptiveBatchProcessor

batch_processor = AdaptiveBatchProcessor(
    initial_batch_size=8,
    min_batch_size=1,
    max_batch_size=64
)

# 自动调整批处理大小
for data in dataset:
    batch_size = batch_processor.get_batch_size()
    # 使用调整后的批处理大小
```

### 批处理大小优化

```python
from src.utils.adaptive_batch_processor import BatchSizeOptimizer

optimizer = BatchSizeOptimizer()
optimal_batch_size = optimizer.find_optimal_batch_size(test_function)
```

## 数据加载优化

```python
from src.utils.optimized_dataloader import OptimizedDataLoader, DataLoaderConfig

config = DataLoaderConfig(
    batch_size=32,
    num_workers=4,
    enable_caching=True,
    enable_prefetch=True
)

dataloader = OptimizedDataLoader(dataset, config)
```

## 集成到现有代码

### 1. 最小侵入式集成

只需要在现有代码中添加几行：

```python
# 在文件开头添加
from src.utils.optimization_manager import get_global_optimization_manager

# 在主函数开头添加
manager = get_global_optimization_manager()
manager.initialize()

# 在程序结束时添加
manager.cleanup()
```

### 2. 渐进式集成

逐步替换现有组件：

1. 首先替换数据加载器
2. 然后优化模型
3. 最后集成训练优化

### 3. 完全集成

使用优化包装器替换现有的预测和训练代码。

## 性能报告

优化工具会自动生成详细的性能报告，包括：

- 系统资源使用情况
- 操作性能统计
- 优化建议
- 性能图表

```python
# 生成报告
report_path = manager.generate_performance_report("./performance_reports")
print(f"报告已生成: {report_path}")
```

## 故障排除

### 常见问题

1. **导入错误**: 确保所有优化工具文件都在正确的位置
2. **内存不足**: 降低批处理大小或启用更激进的内存清理
3. **性能下降**: 检查优化配置，可能需要调整优化级别
4. **CUDA 错误**: 确保 CUDA 版本兼容，检查 GPU 内存使用

### 调试技巧

1. 启用详细日志：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

2. 检查优化摘要：
```python
summary = manager.get_optimization_summary()
print(summary)
```

3. 查看优化建议：
```python
recommendations = manager.get_recommendations()
for rec in recommendations:
    print(rec)
```

## 最佳实践

1. **选择合适的优化级别**: 开发时使用 CONSERVATIVE，生产时使用 BALANCED
2. **监控内存使用**: 定期检查内存使用情况，避免内存泄漏
3. **定期生成性能报告**: 跟踪性能变化，及时发现问题
4. **渐进式优化**: 逐步应用优化，避免一次性改动过大
5. **测试验证**: 在应用优化后验证结果的正确性

## 示例项目

查看 `examples/optimization_example.py` 获取完整的使用示例。

## 支持

如果遇到问题，请：

1. 检查日志输出
2. 查看性能报告
3. 参考故障排除部分
4. 查看示例代码

---

通过遵循本指南，您可以轻松地将优化工具集成到现有项目中，显著提升性能和资源利用率。
'''
        
        with open(guide_path, 'w', encoding='utf-8') as f:
            f.write(guide_content)
        
        logger.info(f"集成指南已创建: {guide_path}")
        return str(guide_path)
    
    def create_requirements_update(self) -> str:
        """创建依赖更新文件"""
        requirements_path = self.project_root / "optimization_requirements.txt"
        
        requirements_content = '''# 优化工具额外依赖
# 这些依赖是优化工具正常运行所需的

# 性能监控
psutil>=5.8.0
numpy>=1.21.0
matplotlib>=3.5.0
pandas>=1.3.0

# GPU 监控（可选）
pynvml>=11.0.0

# 配置管理
PyYAML>=6.0

# 数据处理
scipy>=1.7.0

# 可视化（可选）
seaborn>=0.11.0
plotly>=5.0.0

# 日志和调试
tqdm>=4.62.0

# 注意：
# 1. 这些依赖大部分是可选的，核心功能不依赖它们
# 2. 如果某些依赖不可用，相关功能会被禁用但不会影响基本功能
# 3. 建议根据实际需求选择性安装
'''
        
        with open(requirements_path, 'w', encoding='utf-8') as f:
            f.write(requirements_content)
        
        logger.info(f"依赖更新文件已创建: {requirements_path}")
        return str(requirements_path)
    
    def integrate_all(self) -> Dict[str, str]:
        """执行完整集成"""
        logger.info("开始集成优化工具...")
        
        results = {}
        
        try:
            # 创建优化预测包装器
            results['predict_wrapper'] = self.create_optimized_predict_wrapper()
            
            # 创建优化训练包装器
            results['training_wrapper'] = self.create_optimized_training_wrapper()
            
            # 创建集成指南
            results['integration_guide'] = self.create_integration_guide()
            
            # 创建依赖更新文件
            results['requirements'] = self.create_requirements_update()
            
            logger.info("优化工具集成完成！")
            
        except Exception as e:
            logger.error(f"集成过程中出现错误: {e}")
            raise
        
        return results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="优化工具集成脚本")
    parser.add_argument("--project-root", default=".", help="项目根目录")
    parser.add_argument("--component", choices=["predict", "training", "guide", "requirements", "all"],
                       default="all", help="要集成的组件")
    parser.add_argument("--force", action="store_true", help="强制覆盖现有文件")
    
    args = parser.parse_args()
    
    # 创建集成器
    integrator = OptimizationIntegrator(args.project_root)
    
    try:
        if args.component == "all":
            results = integrator.integrate_all()
            print("\n集成完成！创建的文件:")
            for component, path in results.items():
                print(f"  {component}: {path}")
        
        elif args.component == "predict":
            path = integrator.create_optimized_predict_wrapper()
            print(f"优化预测包装器已创建: {path}")
        
        elif args.component == "training":
            path = integrator.create_optimized_training_wrapper()
            print(f"优化训练包装器已创建: {path}")
        
        elif args.component == "guide":
            path = integrator.create_integration_guide()
            print(f"集成指南已创建: {path}")
        
        elif args.component == "requirements":
            path = integrator.create_requirements_update()
            print(f"依赖更新文件已创建: {path}")
        
        print("\n下一步:")
        print("1. 查看 OPTIMIZATION_INTEGRATION_GUIDE.md 了解如何使用")
        print("2. 安装额外依赖: pip install -r optimization_requirements.txt")
        print("3. 运行示例: python examples/optimization_example.py")
        print("4. 在您的代码中使用优化包装器")
        
    except Exception as e:
        logger.error(f"集成失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()