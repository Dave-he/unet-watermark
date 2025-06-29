# 水印提取工具

这个工具用于从原图和水印图的对比中提取透明背景的水印素材。

## 功能特点

- 自动对比原图和水印图，提取水印区域
- 生成透明背景的水印素材
- 智能检测水印位置，如果水印在图像中位置较远则自动分成多个素材图片
- 增强水印清晰度（对比度和锐化处理）
- 支持批量处理

## 安装依赖

```bash
pip install -r requirements_extract.txt
```

## 使用方法

### 基本用法

```bash
python extract_watermarks.py
```

这将使用默认参数：
- 原图文件夹: `data/train/clean`
- 水印图文件夹: `data/train/watermarked`
- 输出文件夹: `data/extracted_watermarks`

### 自定义参数

```bash
python extract_watermarks.py \
    --clean_dir /path/to/clean/images \
    --watermarked_dir /path/to/watermarked/images \
    --output_dir /path/to/output \
    --threshold 30 \
    --min_area 100 \
    --limit 50
```

### 参数说明

- `--clean_dir`: 原图文件夹路径
- `--watermarked_dir`: 水印图文件夹路径
- `--output_dir`: 输出文件夹路径
- `--threshold`: 差异检测阈值（0-255，默认30）
- `--min_area`: 最小水印区域面积（像素，默认100）
- `--limit`: 随机处理的图片数量限制（可选，不设置则处理所有图片）

## 工作原理

1. **图片对比**: 计算原图和水印图的像素差异
2. **区域检测**: 使用阈值和形态学操作识别水印区域
3. **智能分组**: 使用DBSCAN聚类算法判断水印区域是否需要分离
4. **透明处理**: 创建透明背景的RGBA图像
5. **清晰度增强**: 应用对比度和锐化滤镜提高水印清晰度

## 输出文件

- 单个水印区域: `{原文件名}_watermark.png`
- 多个分离区域: `{原文件名}_watermark_part1.png`, `{原文件名}_watermark_part2.png`, ...

## 注意事项

1. 确保原图和水印图文件名相同（不包括扩展名）
2. 支持的图片格式: JPG, PNG
3. 输出格式为PNG（支持透明背景）
4. 如果阈值设置过低，可能会检测到过多噪声
5. 如果阈值设置过高，可能会遗漏较淡的水印

## 调优建议

- **threshold**: 根据水印的明显程度调整，水印越淡需要越低的阈值
- **min_area**: 根据水印大小调整，过小的值可能包含噪声，过大的值可能遗漏小水印
- **limit**: 用于快速测试或处理大数据集的子集，随机选择指定数量的图片进行处理
- 如果水印检测效果不佳，可以尝试调整这些参数

### 使用场景

- **快速测试**: 使用 `--limit 10` 先处理少量图片验证效果
- **分批处理**: 对于大数据集，可以分批处理避免内存不足
- **样本抽取**: 随机选择部分图片进行水印提取分析

## 示例

假设你有以下文件结构：
```
data/
├── train/
│   ├── clean/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── watermarked/
│       ├── image1.jpg
│       └── image2.jpg
```

运行工具后，将在 `data/extracted_watermarks/` 目录下生成：
```
data/
└── extracted_watermarks/
    ├── image1_watermark.png
    └── image2_watermark.png
```

如果某个图片的水印分布在不同位置，可能会生成：
```
data/
└── extracted_watermarks/
    ├── image1_watermark_part1.png
    ├── image1_watermark_part2.png
    └── image2_watermark.png
```