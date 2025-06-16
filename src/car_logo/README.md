# Logo自动贴合工具使用说明

这个工具可以将logo自动贴合到图片上，支持单张图片处理和批量处理两种模式。

## 功能特性

- **单张图片模式**: 将指定logo贴合到指定图片上
- **批量处理模式**: 从无水印图片目录中随机选择图片，与logo进行批量贴合
- **多种特征检测算法**: 支持SIFT、SURF、ORB算法
- **随机选择**: 支持限制处理图片数量，随机选择指定数量的图片

## 安装依赖

```bash
pip install opencv-python numpy
```

## 使用方法

### 1. 单张图片模式

将单个logo贴合到单张图片上：

```bash
python logo_placement.py single --logo path/to/logo.png --part path/to/image.jpg --output result.png
```

参数说明：
- `--logo`: logo图像路径
- `--part`: 零件图像路径
- `--output`: 输出图像路径（默认：output.png）
- `--feature_method`: 特征检测方法，可选SIFT/SURF/ORB（默认：SIFT）
- `--show`: 显示中间结果

### 2. 批量处理模式

从data/train/clean目录中的无水印图片批量处理：

```bash
python logo_placement.py batch --limit 10
```

参数说明：
- `--clean_dir`: 无水印图片目录路径（默认：data/train/clean）
- `--logo_dir`: logo文件目录路径（默认：data/car_logo）
- `--output_dir`: 输出目录路径（默认：output）
- `--limit`: 随机选择的图片数量限制（可选）
- `--feature_method`: 特征检测方法，可选SIFT/SURF/ORB（默认：SIFT）

## 使用示例

### 示例1：处理10张随机图片

```bash
python logo_placement.py batch --limit 10 --output_dir results
```

这将从`data/train/clean`目录中随机选择10张图片，与`data/car_logo`目录中的logo进行贴合，结果保存到`results`目录。

### 示例2：处理所有图片

```bash
python logo_placement.py batch --clean_dir data/train/clean --logo_dir data/car_logo --output_dir output
```

### 示例3：使用不同的特征检测算法

```bash
python logo_placement.py batch --limit 5 --feature_method ORB
```

### 示例4：单张图片处理并显示结果

```bash
python logo_placement.py single --logo data/car_logo/A_A_001.png --part data/train/clean/00b7c525585a553.jpg --output single_result.png --show
```

## 输出说明

### 批量处理模式输出

- 输出文件命名格式：`{原图名}_with_{logo名}.png`
- 例如：`00b7c525585a553_with_A_A_001.png`

### 单张图片模式输出

- 主要结果：指定的输出文件
- 附加文件：
  - `matches.png`: 特征匹配可视化
  - `result.png`: 带边框的结果
  - `combined.png`: 原图和结果的对比

## 注意事项

1. **图片格式支持**: 支持jpg、jpeg、png、bmp格式
2. **logo透明度**: 支持带透明通道的PNG logo
3. **特征匹配**: 如果特征匹配失败，该图片会被跳过
4. **内存使用**: 批量处理大量图片时注意内存使用
5. **输出目录**: 输出目录会自动创建

## 故障排除

### 常见错误

1. **找不到图像文件**: 检查路径是否正确
2. **特征匹配失败**: 尝试使用不同的特征检测算法
3. **内存不足**: 减少批量处理的图片数量

### 性能优化

- 对于大批量处理，建议使用ORB算法（速度更快）
- 可以通过`--limit`参数控制处理数量
- SIFT算法精度最高但速度较慢