# 图像分类使用指南

本指南介绍如何使用DINOv2模型对`data/train/watermarked`文件夹下的图片进行分类。

## 概述

我们提供了两个分类脚本：

1. **dinov2_classification.py** - 使用DINOv2模型进行高质量特征提取和分类
2. **simple_image_classification.py** - 简化版本，支持DINOv2和ResNet50备选方案

## 环境准备

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 额外依赖（如果需要）

如果遇到DINOv2相关问题，可能需要安装：

```bash
pip install timm
pip install transformers
```

## 使用方法

### 方法一：使用DINOv2分类脚本

```bash
python dinov2_classification.py
```

### 方法二：使用简化版分类脚本（推荐）

```bash
python simple_image_classification.py
```

## 脚本功能

### 主要功能

1. **特征提取**：使用深度学习模型提取图像特征
2. **聚类分析**：使用K-means算法对图像进行无监督聚类
3. **类型分析**：基于文件名模式分析图片类型
4. **可视化**：生成2D散点图展示聚类结果
5. **结果保存**：生成详细的分类报告和结果文件

### 支持的图片类型

根据文件名模式，脚本会自动识别以下类型：

- **mixed_trans_**: 混合变换水印图片
- **norm_**: 标准图片
- **text_trans_**: 文本变换水印图片
- **trans_**: 变换图片
- **other**: 其他类型图片

## 输出结果

运行完成后，会在指定目录（默认为`image_classification_results`）生成以下文件：

### 文件结构
```
image_classification_results/
├── classification_results.json    # 详细分类结果
├── clusters.json                 # 聚类分组信息
├── classification_report.md      # 可读性强的分类报告
├── cluster_visualization.png     # 聚类可视化图片
└── cluster_0/                    # 各聚类文件夹
    ├── image_list.txt           # 该聚类的图片列表
    └── ...
```

### 结果解读

1. **classification_report.md**：包含完整的分析报告
   - 模型信息和统计数据
   - 图片类型分布
   - 每个聚类的详细组成

2. **cluster_visualization.png**：2D可视化图
   - 使用PCA降维到2D空间
   - 不同颜色代表不同聚类
   - 可以直观看出聚类效果

3. **clusters.json**：机器可读的聚类结果
   - 每个聚类包含的图片列表
   - 便于后续处理和分析

## 参数调整

### 主要参数

在脚本的`main()`函数中可以调整以下参数：

```python
# 配置参数
image_dir = 'data/train/watermarked'  # 图片目录
n_clusters = 6                        # 聚类数量
batch_size = 16                       # 批处理大小
output_dir = 'image_classification_results'  # 输出目录
```

### 聚类数量建议

- **4-6个聚类**：适合基本分类（推荐开始值）
- **8-10个聚类**：更细粒度的分类
- **根据图片类型调整**：如果发现某种类型图片很多，可以增加聚类数

### 批处理大小

- **GPU内存充足**：可以设置为32或更大
- **GPU内存有限**：建议设置为8或16
- **仅使用CPU**：建议设置为4或8

## 故障排除

### 常见问题

1. **DINOv2加载失败**
   - 脚本会自动切换到ResNet50备选方案
   - 确保网络连接正常（首次使用需要下载模型）

2. **内存不足**
   - 减小`batch_size`参数
   - 减少聚类数量
   - 使用CPU而非GPU

3. **图片加载错误**
   - 检查图片文件是否损坏
   - 确保图片格式为JPG
   - 脚本会自动跳过损坏的图片

### 性能优化

1. **使用GPU**：确保PyTorch能够使用CUDA
2. **调整workers**：在DataLoader中调整`num_workers`参数
3. **批处理大小**：根据硬件配置调整批处理大小

## 高级用法

### 自定义聚类算法

可以修改`cluster_images`方法使用其他聚类算法：

```python
from sklearn.cluster import DBSCAN, AgglomerativeClustering

# 使用DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
cluster_labels = dbscan.fit_predict(features_for_clustering)
```

### 特征降维

可以调整PCA参数或使用其他降维方法：

```python
from sklearn.manifold import TSNE

# 使用t-SNE降维
tsne = TSNE(n_components=2, random_state=42)
features_2d = tsne.fit_transform(features_for_clustering)
```

## 结果分析建议

1. **查看分类报告**：首先阅读`classification_report.md`了解整体情况
2. **检查聚类组成**：看每个聚类的图片类型分布是否合理
3. **可视化分析**：通过散点图判断聚类效果
4. **调整参数**：根据结果调整聚类数量或其他参数
5. **人工验证**：随机检查几个聚类中的图片是否确实相似

## 扩展应用

分类结果可以用于：

1. **数据清洗**：识别异常或错误标注的图片
2. **数据增强**：为不同类型的图片设计特定的增强策略
3. **模型训练**：为不同类型的图片使用不同的训练策略
4. **质量评估**：评估水印添加的效果和一致性

## 技术细节

### DINOv2模型

- **架构**：Vision Transformer (ViT)
- **预训练**：自监督学习
- **特征维度**：768维（ViT-B/14）
- **优势**：无需标注数据，特征表达能力强

### ResNet50备选方案

- **架构**：残差神经网络
- **预训练**：ImageNet监督学习
- **特征维度**：2048维
- **优势**：稳定可靠，兼容性好

---

如有问题或需要进一步定制，请参考脚本中的注释或联系开发者。