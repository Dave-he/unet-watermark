# 从Checkpoint继续训练指南

本指南说明如何从预训练的checkpoint文件继续训练水印分割模型。

## 快速开始

### 方法1: 使用CLI接口（推荐）

```bash
# 从checkpoint_epoch_080.pth继续训练
python main.py train --resume models/checkpoints/checkpoint_epoch_080.pth

# 继续训练并调整参数
python main.py train --resume models/checkpoints/checkpoint_epoch_080.pth --lr 0.00005 --epochs 200
```

### 方法2: 直接使用训练脚本

```bash
# 使用train_smp.py直接恢复训练
python src/train_smp.py --resume models/checkpoints/checkpoint_epoch_080.pth

# 指定配置文件和其他参数
python src/train_smp.py --config src/configs/unet_watermark.yaml --resume models/checkpoints/checkpoint_epoch_080.pth --lr 0.00005
```

### 方法3: 使用示例脚本

```bash
# 运行示例脚本
python resume_training_example.py

# 只查看checkpoint信息
python resume_training_example.py --info-only

# 显示更多使用示例
python resume_training_example.py --show-examples
```

## 详细说明

### 1. 查看可用的Checkpoint文件

```bash
# 列出checkpoints目录中的所有checkpoint文件
python main.py predict --list-checkpoints models/checkpoints
```

### 2. 检查Checkpoint信息

```bash
# 查看特定checkpoint的详细信息
python resume_training_example.py --checkpoint models/checkpoints/checkpoint_epoch_080.pth --info-only
```

### 3. 从Checkpoint恢复训练

当从checkpoint恢复训练时，系统会自动恢复：

- ✅ **模型权重**: 完整的模型参数
- ✅ **优化器状态**: 包括动量、学习率历史等
- ✅ **调度器状态**: 学习率调度器的当前状态
- ✅ **训练进度**: 当前epoch、最佳验证损失等
- ✅ **训练历史**: 损失曲线和指标历史（可选）

### 4. 参数调整

恢复训练时可以调整以下参数：

```bash
python main.py train --resume models/checkpoints/checkpoint_epoch_080.pth \
    --lr 0.00005 \          # 调整学习率
    --epochs 200 \          # 设置总训练轮数
    --batch-size 8 \        # 调整批次大小
    --device cuda           # 指定设备
```

## 常见使用场景

### 场景1: 训练中断后恢复

如果训练过程中断（如服务器重启、程序崩溃等），可以从最新的checkpoint恢复：

```bash
# 找到最新的checkpoint
ls -la models/checkpoints/checkpoint_epoch_*.pth

# 从最新checkpoint恢复
python main.py train --resume models/checkpoints/checkpoint_epoch_080.pth
```

### 场景2: 添加新数据集继续训练

当有新的训练数据时，可以从之前的checkpoint继续训练：

```bash
# 更新数据目录并从checkpoint继续训练
python main.py train --resume models/checkpoints/checkpoint_epoch_080.pth \
    --data-dir data/new_dataset \
    --lr 0.00001  # 使用较小的学习率进行微调
```

### 场景3: 超参数调优

基于已有的训练结果，调整超参数继续训练：

```bash
# 降低学习率继续训练
python main.py train --resume models/checkpoints/checkpoint_epoch_080.pth \
    --lr 0.00005 \
    --epochs 150
```

### 场景4: 不同设备间迁移

在不同设备（CPU/GPU）间迁移训练：

```bash
# 从GPU checkpoint在CPU上继续训练
python main.py train --resume models/checkpoints/checkpoint_epoch_080.pth \
    --device cpu

# 从CPU checkpoint在GPU上继续训练
python main.py train --resume models/checkpoints/checkpoint_epoch_080.pth \
    --device cuda
```

## 注意事项

### ⚠️ 重要提醒

1. **Checkpoint兼容性**: 确保checkpoint文件与当前代码版本兼容
2. **配置文件**: 建议使用与原训练相同的配置文件
3. **数据集路径**: 确保数据集路径正确，特别是在不同机器间迁移时
4. **设备兼容**: checkpoint会自动适配不同设备（CPU/GPU）

### 📁 文件结构

```
models/
├── checkpoints/
│   ├── checkpoint_epoch_010.pth
│   ├── checkpoint_epoch_020.pth
│   ├── ...
│   ├── checkpoint_epoch_080.pth  # 要恢复的checkpoint
│   └── final_model_epoch_100.pth
└── unet_watermark.pth  # 最佳模型
```

### 🔍 故障排除

**问题1**: `FileNotFoundError: checkpoint file not found`
```bash
# 检查文件是否存在
ls -la models/checkpoints/checkpoint_epoch_080.pth

# 列出所有可用的checkpoint
ls -la models/checkpoints/
```

**问题2**: `RuntimeError: Error loading checkpoint`
```bash
# 检查checkpoint文件完整性
python resume_training_example.py --checkpoint models/checkpoints/checkpoint_epoch_080.pth --info-only
```

**问题3**: 内存不足
```bash
# 减小批次大小
python main.py train --resume models/checkpoints/checkpoint_epoch_080.pth --batch-size 4
```

## 编程接口

如果需要在代码中使用恢复训练功能：

```python
from src.configs.config import get_cfg_defaults, update_config
from src.train_smp import train

# 加载配置
cfg = get_cfg_defaults()
update_config(cfg, "src/configs/unet_watermark.yaml")

# 从checkpoint恢复训练
train(cfg, resume_from="models/checkpoints/checkpoint_epoch_080.pth")
```

## 性能优化建议

1. **学习率调整**: 恢复训练时建议使用较小的学习率
2. **早停机制**: 可以调整早停的耐心值
3. **批次大小**: 根据可用内存调整批次大小
4. **数据加载**: 确保数据加载器配置合适

```bash
# 优化配置示例
python main.py train --resume models/checkpoints/checkpoint_epoch_080.pth \
    --lr 0.00005 \                    # 较小的学习率
    --batch-size 8 \                  # 适中的批次大小
    --early-stopping-patience 15 \   # 增加早停耐心值
    --epochs 200                      # 设置合理的总轮数
```

---

更多信息请参考：
- [训练配置说明](src/configs/unet_watermark.yaml)
- [模型架构文档](README.md)