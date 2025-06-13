# 水印修复CUDA优化方案

## 问题描述

在使用CUDA进行水印修复时，经常遇到以下问题：
- CUDA内存不足导致程序卡住
- GPU内存泄漏，处理速度逐渐变慢
- 批处理时内存管理不当
- 长时间运行后系统资源耗尽

## 优化方案

### 1. 新增优化组件

#### 批量修复优化器 (`src/scripts/batch_repair_optimizer.py`)
- **智能批处理**: 根据系统资源动态调整批处理大小
- **资源监控**: 实时监控CPU、GPU内存使用情况
- **自动暂停**: 当资源使用率过高时自动暂停并清理
- **错误恢复**: 单张图片处理失败不影响整个批次

#### CUDA内存监控器 (`src/utils/cuda_monitor.py`)
- **实时监控**: 持续监控GPU内存分配和使用情况
- **阈值警告**: 内存使用率达到警告线时自动提醒
- **内存清理**: 提供激进的内存清理策略
- **上下文管理**: 自动管理CUDA内存生命周期

#### 优化的CLI接口
- **优化模式**: 新增 `--optimize` 参数启用优化处理
- **批处理控制**: `--batch-size` 和 `--pause-interval` 参数
- **内存监控**: 集成实时内存使用情况显示

### 2. 核心优化策略

#### 内存管理优化
```python
# 在关键位置添加内存清理
torch.cuda.empty_cache()  # 清空CUDA缓存
torch.cuda.synchronize()  # 同步CUDA操作
gc.collect()              # Python垃圾回收
```

#### 批处理优化
- **小批量处理**: 默认批处理大小从100降至5-10
- **定期清理**: 每处理25张图片自动清理内存
- **资源检查**: 处理前检查可用内存，不足时暂停

#### 错误处理优化
- **超时机制**: IOPaint调用添加超时限制
- **异常恢复**: 单个图片失败不中断整个流程
- **资源清理**: 异常时确保临时文件和内存被正确清理

### 3. 代码质量优化

#### 类型提示和文档注释
- **全面类型提示**: 为所有函数和方法添加类型提示，提高代码可读性和IDE支持
- **文档字符串**: 为所有模块、类和函数添加详细的文档注释
- **导入优化**: 清理重复导入，添加必要的类型导入

#### 代码结构优化
- **模块化设计**: 改进代码组织结构，提高可维护性
- **错误处理**: 统一错误处理模式，提高程序健壮性
- **性能监控**: 集成性能分析工具，便于性能调优

#### 依赖管理优化
- **requirements.txt**: 清理和更新依赖列表
- **版本控制**: 明确指定关键依赖的版本范围
- **可选依赖**: 区分核心依赖和可选依赖

## 使用方法

### 1. 快速启动（推荐）

使用优化启动脚本：
```bash
# 基本用法
python repair_optimized.py \
  --input data/train/watermarked \
  --output data/result \
  --model models/checkpoints/checkpoint_epoch_030.pth

# 快速模式（更激进的优化）
python repair_optimized.py \
  --input data/train/watermarked \
  --output data/result \
  --model models/checkpoints/checkpoint_epoch_030.pth \
  --fast

# 自定义参数
python repair_optimized.py \
  --input data/train/watermarked \
  --output data/result \
  --model models/checkpoints/checkpoint_epoch_030.pth \
  --batch-size 3 \
  --limit 50
```

### 2. 使用原始CLI（启用优化）

```bash
# 启用优化模式
python main.py repair \
  --input data/train/watermarked \
  --output data/result \
  --model models/checkpoints/checkpoint_epoch_030.pth \
  --optimize \
  --batch-size 5 \
  --pause-interval 25
```

### 3. 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--optimize` | False | 启用优化批处理模式 |
| `--batch-size` | 5 | 批处理大小（建议1-10） |
| `--pause-interval` | 25 | 处理多少张图片后暂停清理 |
| `--max-iterations` | 3 | 最大迭代次数（减少以提高速度） |
| `--threshold` | 0.005 | 水印检测阈值（稍微提高以减少误检） |
| `--fast` | - | 快速模式预设 |
| `--conservative` | - | 保守模式预设 |

## 性能对比

### 优化前
- **内存使用**: 持续增长，容易OOM
- **处理速度**: 随时间递减
- **稳定性**: 容易卡住或崩溃
- **资源利用**: 不均衡，峰值过高

### 优化后
- **内存使用**: 稳定控制在安全范围内
- **处理速度**: 保持稳定
- **稳定性**: 显著提升，自动错误恢复
- **资源利用**: 均衡，避免峰值

## 监控和调试

### 内存使用监控
```python
from src.utils.cuda_monitor import CUDAMonitor, log_memory_usage

# 记录内存使用情况
log_memory_usage("处理前 - ")

# 启动实时监控
monitor = CUDAMonitor()
monitor.start_monitoring()
```

### 资源使用情况
优化模式会自动显示：
- GPU内存分配和保留情况
- CPU和系统内存使用率
- 处理速度和完成统计
- 峰值内存使用情况

## 故障排除

### 常见问题

1. **仍然出现OOM错误**
   - 减小 `--batch-size` 到 1-3
   - 降低 `--pause-interval` 到 10-15
   - 使用 `--fast` 模式

2. **处理速度太慢**
   - 增加 `--batch-size` 到 8-10
   - 提高 `--threshold` 到 0.01
   - 减少 `--max-iterations` 到 2

3. **内存清理过于频繁**
   - 增加 `--pause-interval` 到 50-100
   - 使用 `--conservative` 模式

### 调试模式

启用详细日志：
```bash
export CUDA_LAUNCH_BLOCKING=1
python repair_optimized.py --input ... --output ... --model ...
```

## 技术细节

### 内存管理策略
1. **预防性清理**: 在关键操作前后清理内存
2. **阈值监控**: 内存使用率超过80%时暂停处理
3. **分批处理**: 将大批量拆分为小批量处理
4. **异常安全**: 确保异常情况下内存被正确释放

### 性能优化技巧
1. **模型复用**: 避免重复加载模型
2. **批量推理**: 合理利用GPU并行能力
3. **内存池**: 复用已分配的内存空间
4. **异步处理**: 重叠CPU和GPU操作

## 更新日志

### v1.0 (当前版本)
- 新增批量修复优化器
- 新增CUDA内存监控器
- 优化CLI接口
- 添加快速启动脚本
- 完善错误处理和恢复机制

## 贡献

如果您发现问题或有改进建议，请：
1. 检查现有的issue
2. 提供详细的错误信息和系统配置
3. 包含复现步骤
4. 建议具体的解决方案

---

**注意**: 这些优化主要针对CUDA环境。如果使用CPU，某些优化可能不适用，但不会影响正常功能。