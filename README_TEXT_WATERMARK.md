# UNet文字水印检测优化指南

本文档介绍如何使用优化后的UNet模型更好地处理文字水印。

## 🚀 快速开始

### 1. 环境准备

确保已安装所需依赖：
```bash
pip install torch torchvision opencv-python easyocr matplotlib scikit-learn tqdm
```

### 2. 使用优化后的CLI命令

现在可以为水印和文字分别指定不同的IOPaint模型：

```bash
# 使用不同模型处理水印和文字
python src/cli.py repair \
    --input-dir ./input_images \
    --output-dir ./output_images \
    --watermark-model lama \
    --text-model mat

# 使用相同模型（向后兼容）
python src/cli.py repair \
    --input-dir ./input_images \
    --output-dir ./output_images \
    --iopaint-model lama
```

### 3. 自动训练模式

```bash
# 使用不同模型进行自动训练
python src/auto_train.py \
    --max-cycles 5 \
    --watermark-model lama \
    --text-model mat \
    --output-dir ./auto_train_results
```

## 🎯 文字水印专用功能

### 智能类型检测

系统现在可以自动检测水印类型并应用相应的优化策略：

- **文字水印**: 使用文字特定的形态学操作和阈值
- **图像水印**: 使用传统的水印检测优化
- **混合水印**: 使用平衡的处理策略

### 新增的预测方法

```python
from src.predict import WatermarkPredictor

predictor = WatermarkPredictor()

# 专门针对文字水印优化
text_mask = predictor.predict_text_watermark_mask(image_path)

# 混合模式（文字+图像水印）
mixed_mask = predictor.predict_mixed_watermark_mask(image_path)

# 智能自动检测（推荐）
auto_mask = predictor.predict_mask(image_path)  # 自动检测类型
```

## 🔧 专用训练

### 训练文字水印专用模型

使用专门的配置文件训练针对文字水印优化的模型：

```bash
# 使用专用训练脚本
python train_text_watermark.py --config src/configs/unet_text_watermark.yaml
```

### 文字水印配置特点

`src/configs/unet_text_watermark.yaml` 包含以下优化：

- **模型架构**: UnetPlusPlus + EfficientNet-B3（更好的特征提取）
- **图像分辨率**: 512x512（更高分辨率保留文字细节）
- **损失函数**: 组合损失（BCE + Dice + Focal Loss）
- **数据增强**: 文字特定的增强策略
- **训练参数**: 更多epoch和更低学习率

## 🧪 测试和验证

### 单图像测试

```bash
# 测试单张图像
python test_text_watermark.py \
    --input ./test_image.jpg \
    --output ./test_results \
    --model ./models/text_watermark_model.pth
```

### 批量测试

```bash
# 批量测试目录中的所有图像
python test_text_watermark.py \
    --input ./test_images/ \
    --output ./batch_results \
    --batch
```

测试脚本会生成：
- 各种方法的预测结果对比
- 详细的指标分析报告
- 可视化对比图
- 性能评估和推荐

## 📊 性能优化策略

### 1. 文字特征增强

系统在检测到文字类型时会自动应用：
- 对比度增强
- 边缘检测增强
- 形态学操作优化
- 锐化处理

### 2. 智能mask优化

根据检测到的水印类型应用不同的后处理：

```python
# 文字水印优化
- 水平/垂直连接增强
- 文字特定的面积阈值
- 长宽比过滤

# 图像水印优化
- 传统形态学操作
- 连通组件分析
- 标准面积阈值

# 混合优化
- 平衡的处理策略
- 适中的参数设置
```

### 3. OCR集成

结合EasyOCR进行文字区域检测，提供额外的验证和对比：
- 文字区域精确定位
- 多语言支持
- 与UNet预测结果对比分析

## 🔍 配置参数说明

### 文字检测相关参数

```yaml
# 文字特征增强
DATA:
  TEXT_ENHANCEMENT: true
  CONTRAST_BOOST: 1.2
  EDGE_ENHANCEMENT: true

# 文字特定损失权重
LOSS:
  BCE_WEIGHT: 0.3
  DICE_WEIGHT: 0.5
  FOCAL_WEIGHT: 0.2

# 文字检测阈值
PREDICTION:
  TEXT_AREA_THRESHOLD: 50
  TEXT_ASPECT_RATIO_MIN: 0.1
  TEXT_ASPECT_RATIO_MAX: 10.0
```

### 智能检测参数

```yaml
VALIDATION:
  TEXT_SCORE_THRESHOLD: 0.6
  MIXED_SCORE_THRESHOLD: 0.4
  EDGE_DENSITY_THRESHOLD: 0.3
```

## 📈 性能监控

### 关键指标

- **Dice系数**: 整体分割质量
- **IoU**: 交并比
- **精确率/召回率**: 文字检测准确性
- **F1分数**: 综合性能指标

### 对比分析

测试脚本会自动计算：
- 不同方法间的相似度
- 与OCR检测的一致性
- 各方法的优劣势分析

## 🛠️ 故障排除

### 常见问题

1. **文字检测效果不佳**
   - 检查图像分辨率是否足够
   - 调整对比度增强参数
   - 尝试不同的阈值设置

2. **模型加载失败**
   - 确认模型路径正确
   - 检查模型文件完整性
   - 验证配置文件格式

3. **内存不足**
   - 减小batch_size
   - 降低图像分辨率
   - 使用CPU模式

### 调试技巧

```bash
# 启用详细日志
export PYTHONPATH=$PYTHONPATH:./src
python -u test_text_watermark.py --input test.jpg --output debug/ 2>&1 | tee debug.log

# 检查中间结果
# 测试脚本会保存所有中间步骤的结果
```

## 🎯 最佳实践

### 1. 数据准备
- 确保训练数据包含足够的文字水印样本
- 标注要精确，特别是文字边界
- 包含不同字体、大小、颜色的文字样本

### 2. 模型选择
- 对于纯文字水印：使用 `predict_text_watermark_mask`
- 对于混合场景：使用 `predict_mixed_watermark_mask`
- 对于未知类型：使用 `predict_mask`（自动检测）

### 3. 参数调优
- 根据具体数据集调整阈值参数
- 监控训练过程中的各项指标
- 使用测试脚本进行效果验证

### 4. 生产部署
- 使用批量处理提高效率
- 设置合适的设备和内存限制
- 定期评估和更新模型

## 📚 API参考

### WatermarkPredictor新增方法

```python
class WatermarkPredictor:
    def predict_text_watermark_mask(self, image_path):
        """专门用于文字水印检测"""
        
    def predict_mixed_watermark_mask(self, image_path):
        """用于混合水印检测"""
        
    def _detect_watermark_type(self, image_path, mask):
        """智能检测水印类型"""
        
    def _enhance_text_features(self, image):
        """增强文字特征"""
        
    def _optimize_mask(self, mask, mask_type='auto'):
        """根据类型优化mask"""
```

### 配置选项

```python
# CLI新增参数
--watermark-model: 水印修复模型
--text-model: 文字修复模型

# 自动训练新增参数
--watermark-model: 水印修复模型
--text-model: 文字修复模型
```

---

## 🤝 贡献

欢迎提交问题和改进建议！请确保：
- 提供详细的问题描述
- 包含复现步骤
- 附上相关的日志和截图

## 📄 许可证

本项目遵循原项目的许可证条款。