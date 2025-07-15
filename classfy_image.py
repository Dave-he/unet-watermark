#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
稳定图像分类脚本 v2.0
集成修复版视频生成功能
解决了绿色视频问题
"""

import os
import json
import random
import hashlib
import pickle
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import shutil
import datetime

# 设置警告过滤
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# 设置所有随机种子
def set_all_seeds(seed=42):
    """设置所有随机种子以确保可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 设置确定性算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 设置环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # 启用确定性算法
    torch.use_deterministic_algorithms(True, warn_only=True)

# 在脚本开始时设置种子
set_all_seeds(42)

class WatermarkDataset(Dataset):
    """水印图片数据集"""
    
    def __init__(self, image_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        
        # 获取所有JPG图片，排序确保一致性
        self.image_paths = sorted(list(self.image_dir.glob('*.jpg')))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"在目录 {image_dir} 中没有找到JPG图片")
        
        print(f"找到 {len(self.image_paths)} 张图片")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, img_path.name
        except Exception as e:
            print(f"加载图片失败 {img_path}: {e}")
            # 返回默认的黑色图片
            default_image = Image.new('RGB', (224, 224), (0, 0, 0))
            if self.transform:
                default_image = self.transform(default_image)
            return default_image, img_path.name

class StableImageClassifier:
    """稳定图像分类器"""
    
    def __init__(self):
        self.model = None
        self.model_type = "unknown"
        self.features = []
        self.image_names = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"使用设备: {self.device}")
        
        # 创建缓存目录
        self.cache_dir = Path('cache')
        self.cache_dir.mkdir(exist_ok=True)
        
        # 加载模型
        self._load_model()
    
    def _load_model(self):
        """加载DINOv2或备选模型"""
        print("正在加载模型...")
        
        # 尝试加载DINOv2
        if self._try_load_dinov2():
            return
        
        # 备选方案：加载其他预训练模型
        print("DINOv2加载失败，尝试备选模型...")
        self._load_backup_model()
    
    def _try_load_dinov2(self):
        """尝试加载DINOv2模型"""
        try:
            # 方法1: torch.hub
            print("尝试从torch.hub加载DINOv2...")
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            model.eval()
            model = model.to(self.device)
            
            self.model = model
            self.model_type = "dinov2_vits14_hub"
            print("✓ DINOv2模型加载成功 (torch.hub)")
            return True
            
        except Exception as e:
            print(f"torch.hub加载失败: {e}")
        
        try:
            # 方法2: transformers
            print("尝试从transformers加载DINOv2...")
            from transformers import AutoImageProcessor, AutoModel
            
            processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
            model = AutoModel.from_pretrained('facebook/dinov2-small')
            model.eval()
            model = model.to(self.device)
            
            self.model = model
            self.processor = processor
            self.model_type = "dinov2_small_transformers"
            print("✓ DINOv2模型加载成功 (transformers)")
            return True
            
        except Exception as e:
            print(f"transformers加载失败: {e}")
        
        try:
            # 方法3: timm
            print("尝试从timm加载DINOv2...")
            import timm
            
            model = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True)
            model.eval()
            model = model.to(self.device)
            
            self.model = model
            self.model_type = "dinov2_vits14_timm"
            print("✓ DINOv2模型加载成功 (timm)")
            return True
            
        except Exception as e:
            print(f"timm加载失败: {e}")
        
        return False
    
    def _load_backup_model(self):
        """加载备选模型"""
        try:
            # 尝试ResNet50
            print("尝试加载ResNet50...")
            import torchvision.models as models
            import torchvision.transforms as transforms
            
            model = models.resnet50(pretrained=True)
            model.eval()
            model = model.to(self.device)
            
            # 移除最后的分类层
            self.model = nn.Sequential(*list(model.children())[:-1])
            self.model_type = "resnet50_backup"
            
            # 设置预处理
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            print("✓ ResNet50备选模型加载成功")
            return
            
        except Exception as e:
            print(f"ResNet50加载失败: {e}")
        
        try:
            # 尝试EfficientNet
            print("尝试加载EfficientNet...")
            import timm
            
            model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
            model.eval()
            model = model.to(self.device)
            
            self.model = model
            self.model_type = "efficientnet_b0_backup"
            print("✓ EfficientNet备选模型加载成功")
            return
            
        except Exception as e:
            print(f"EfficientNet加载失败: {e}")
        
        # 最后的备选方案
        print("使用简单的ViT模型作为最终备选...")
        try:
            import torchvision.models as models
            model = models.vit_b_16(pretrained=True)
            model.eval()
            model = model.to(self.device)
            
            # 移除分类头
            self.model = nn.Sequential(*list(model.children())[:-1])
            self.model_type = "vit_b16_backup"
            print("✓ ViT备选模型加载成功")
            
        except Exception as e:
            print(f"所有模型加载失败: {e}")
            raise RuntimeError("无法加载任何可用的模型")
    
    def _get_cache_key(self, image_dir, batch_size):
        """生成缓存键"""
        # 基于图片目录内容和参数生成唯一键
        image_dir = Path(image_dir)
        
        # 获取目录中所有图片的修改时间
        image_files = sorted(list(image_dir.glob('*.jpg')))
        if not image_files:
            return None
        
        # 创建基于文件列表和参数的哈希
        content = f"{len(image_files)}_{batch_size}_{self.model_type}"
        for img_file in image_files[:10]:  # 只取前10个文件避免过长
            stat = img_file.stat()
            content += f"_{img_file.name}_{stat.st_size}_{stat.st_mtime}"
        
        return hashlib.md5(content.encode()).hexdigest()
    
    def extract_features(self, image_dir, batch_size=8, use_cache=True):
        """提取图片特征（支持缓存）"""
        print(f"\n开始提取特征...")
        print(f"图片目录: {image_dir}")
        print(f"批处理大小: {batch_size}")
        print(f"使用缓存: {use_cache}")
        
        # 检查缓存
        cache_key = self._get_cache_key(image_dir, batch_size)
        cache_file = self.cache_dir / f'features_{cache_key}.pkl' if cache_key else None
        
        if use_cache and cache_file and cache_file.exists():
            print(f"从缓存加载特征: {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.features = cached_data['features']
                    self.image_names = cached_data['image_names']
                    print(f"✓ 成功加载 {len(self.features)} 个特征向量")
                    return
            except Exception as e:
                print(f"缓存加载失败: {e}，重新提取特征")
        
        # 设置数据变换
        if not hasattr(self, 'transform'):
            if 'dinov2' in self.model_type:
                # DINOv2的预处理
                import torchvision.transforms as transforms
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            # 其他模型的transform在_load_backup_model中已设置
        
        # 创建数据集和数据加载器
        dataset = WatermarkDataset(image_dir, transform=self.transform)
        
        # 确保数据加载器的确定性（使用单进程避免序列化问题）
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False,  # 不打乱顺序
            num_workers=0,  # 使用单进程避免序列化问题
            pin_memory=True
        )
        
        self.features = []
        self.image_names = []
        
        print(f"开始处理 {len(dataset)} 张图片...")
        
        with torch.no_grad():
            for batch_images, batch_names in tqdm(dataloader, desc="提取特征"):
                batch_images = batch_images.to(self.device)
                
                # 根据模型类型提取特征
                if 'transformers' in self.model_type:
                    # transformers版本的DINOv2
                    outputs = self.model(batch_images)
                    batch_features = outputs.last_hidden_state.mean(dim=1)  # 全局平均池化
                elif 'dinov2' in self.model_type:
                    # torch.hub或timm版本的DINOv2
                    batch_features = self.model(batch_images)
                else:
                    # 备选模型
                    batch_features = self.model(batch_images)
                    if len(batch_features.shape) > 2:
                        batch_features = batch_features.view(batch_features.size(0), -1)
                
                # 转换为numpy并添加到列表
                batch_features = batch_features.cpu().numpy()
                
                for i, feature in enumerate(batch_features):
                    self.features.append(feature)
                    self.image_names.append(batch_names[i])
        
        self.features = np.array(self.features)
        
        print(f"✓ 特征提取完成")
        print(f"特征形状: {self.features.shape}")
        print(f"图片数量: {len(self.image_names)}")
        
        # 保存到缓存
        if use_cache and cache_file:
            try:
                cache_data = {
                    'features': self.features,
                    'image_names': self.image_names,
                    'model_type': self.model_type
                }
                with open(cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)
                print(f"✓ 特征已缓存到: {cache_file}")
            except Exception as e:
                print(f"缓存保存失败: {e}")
    
    def analyze_image_types(self):
        """分析图片类型"""
        type_mapping = {}
        type_counts = {
            'watermarked': 0,
            'original': 0,
            'processed': 0,
            'other': 0
        }
        
        for img_name in self.image_names:
            # 基于文件名前缀分析类型
            if img_name.startswith('watermarked_'):
                img_type = 'watermarked'
            elif img_name.startswith('original_'):
                img_type = 'original'
            elif img_name.startswith('processed_'):
                img_type = 'processed'
            else:
                img_type = 'other'
            
            type_mapping[img_name] = img_type
            type_counts[img_type] += 1
        
        print(f"\n图片类型统计:")
        for img_type, count in type_counts.items():
            if count > 0:
                percentage = (count / len(self.image_names)) * 100
                print(f"  {img_type}: {count} 张 ({percentage:.1f}%)")
        
        return type_mapping, type_counts
    
    def stable_cluster_images(self, n_clusters=6, use_pca=True, pca_components=50):
        """稳定的图片聚类"""
        print(f"\n开始稳定聚类...")
        print(f"聚类数量: {n_clusters}")
        print(f"使用PCA: {use_pca}")
        
        if len(self.features) == 0:
            raise ValueError("没有可用的特征，请先运行extract_features")
        
        features = self.features.copy()
        
        # 可选的PCA降维
        if use_pca and features.shape[1] > pca_components:
            print(f"应用PCA降维: {features.shape[1]} -> {pca_components}")
            pca = PCA(n_components=pca_components, random_state=42)
            features = pca.fit_transform(features)
            print(f"PCA解释方差比: {pca.explained_variance_ratio_.sum():.3f}")
        
        # 稳定的K-means聚类
        print("执行K-means聚类...")
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,  # 固定随机种子
            n_init=20,        # 增加初始化次数提高稳定性
            max_iter=500,     # 增加最大迭代次数
            tol=1e-6,         # 更严格的收敛条件
            algorithm='lloyd'  # 使用确定性算法
        )
        
        cluster_labels = kmeans.fit_predict(features)
        
        # 创建聚类字典
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(self.image_names[i])
        
        # 按聚类大小排序
        clusters = dict(sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True))
        
        print(f"✓ 聚类完成")
        for cluster_id, images in clusters.items():
            print(f"  聚类 {cluster_id}: {len(images)} 张图片")
        
        # 返回结果
        results = {
            'model_type': self.model_type,
            'feature_dim': features.shape[1],
            'n_clusters': n_clusters,
            'inertia': kmeans.inertia_,
            'use_pca': use_pca
        }
        
        return results, clusters
    
    def analyze_cluster_composition(self, clusters):
        """分析聚类组成"""
        print(f"\n=== 聚类组成分析 ===")
        
        type_mapping, _ = self.analyze_image_types()
        
        for cluster_id, images in clusters.items():
            print(f"\n聚类 {cluster_id} ({len(images)} 张图片):")
            
            # 统计类型分布
            type_counts = {}
            for img_name in images:
                img_type = type_mapping.get(img_name, 'other')
                type_counts[img_type] = type_counts.get(img_type, 0) + 1
            
            # 显示类型分布
            for img_type, count in sorted(type_counts.items()):
                if count > 0:
                    percentage = (count / len(images)) * 100
                    print(f"  {img_type}: {count} 张 ({percentage:.1f}%)")
    
    def visualize_clusters(self, clusters, output_dir='stable_classification_results'):
        """可视化聚类结果"""
        print(f"\n生成聚类可视化...")
        
        if len(self.features) == 0:
            print("没有特征数据，跳过可视化")
            return
        
        # 使用PCA降维到2D进行可视化
        pca_2d = PCA(n_components=2, random_state=42)
        features_2d = pca_2d.fit_transform(self.features)
        
        # 创建聚类标签数组
        labels = np.zeros(len(self.image_names))
        for cluster_id, images in clusters.items():
            for img_name in images:
                if img_name in self.image_names:
                    idx = self.image_names.index(img_name)
                    labels[idx] = cluster_id
        
        # 创建图表
        plt.figure(figsize=(12, 8))
        
        # 使用不同颜色绘制每个聚类
        colors = plt.cm.Set3(np.linspace(0, 1, len(clusters)))
        
        for i, (cluster_id, color) in enumerate(zip(clusters.keys(), colors)):
            cluster_mask = labels == cluster_id
            plt.scatter(
                features_2d[cluster_mask, 0], 
                features_2d[cluster_mask, 1],
                c=[color], 
                label=f'聚类 {cluster_id} ({np.sum(cluster_mask)} 张)',
                alpha=0.7,
                s=50
            )
        
        plt.title(f'稳定图像聚类可视化\n模型: {self.model_type}', fontsize=14)
        plt.xlabel(f'PCA 第一主成分 (解释方差: {pca_2d.explained_variance_ratio_[0]:.1%})')
        plt.ylabel(f'PCA 第二主成分 (解释方差: {pca_2d.explained_variance_ratio_[1]:.1%})')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存图片
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        plt.savefig(output_path / 'stable_cluster_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 可视化已保存: {output_path / 'stable_cluster_visualization.png'}")
    
    def save_results(self, results, clusters, output_dir='stable_classification_results'):
        """保存分类结果"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\n保存结果到: {output_path}")
        
        # 保存详细结果
        detailed_results = {
            **results,
            'total_images': len(self.image_names),
            'clusters': {str(k): v for k, v in clusters.items()},
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        with open(output_path / 'stable_classification_results.json', 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)
        
        # 保存聚类分组
        with open(output_path / 'stable_clusters.json', 'w', encoding='utf-8') as f:
            serializable_clusters = {str(k): v for k, v in clusters.items()}
            json.dump(serializable_clusters, f, ensure_ascii=False, indent=2)
        
        # 分析图片类型
        type_mapping, type_counts = self.analyze_image_types()
        
        # 创建详细报告
        report_lines = ["# 稳定图像分类报告\n\n"]
        report_lines.append(f"**使用模型**: {results['model_type']}\n")
        report_lines.append(f"**总图片数量**: {len(self.image_names)}\n")
        report_lines.append(f"**特征维度**: {results['feature_dim']}\n")
        report_lines.append(f"**聚类数量**: {len(clusters)}\n")
        report_lines.append(f"**聚类质量指标**: {results.get('inertia', 'N/A')}\n")
        report_lines.append(f"**随机种子**: 42 (固定)\n\n")
        
        # 添加稳定性改进说明
        report_lines.append("## 稳定性改进\n\n")
        report_lines.append("- ✓ 固定所有随机种子确保可重复性\n")
        report_lines.append("- ✓ 确保数据加载顺序一致\n")
        report_lines.append("- ✓ 使用确定性聚类算法\n")
        report_lines.append("- ✓ 添加特征缓存机制\n")
        report_lines.append("- ✓ 增加聚类初始化次数提高稳定性\n\n")
        
        # 添加模型加载信息
        if 'dinov2' in self.model_type:
            report_lines.append("**模型状态**: ✓ DINOv2模型加载成功\n\n")
        else:
            report_lines.append("**模型状态**: ⚠️ DINOv2加载失败，使用备选模型\n\n")
        
        # 添加图片类型统计
        report_lines.append("## 图片类型统计\n\n")
        for img_type, count in type_counts.items():
            if count > 0:
                percentage = (count / len(self.image_names)) * 100
                report_lines.append(f"- **{img_type}**: {count} 张 ({percentage:.1f}%)\n")
        
        report_lines.append("\n## 聚类详情\n\n")
        
        for cluster_id, images in clusters.items():
            cluster_dir = output_path / f'cluster_{cluster_id}'
            cluster_dir.mkdir(exist_ok=True)
            
            # 分析该聚类的类型组成
            cluster_type_counts = {}
            for img_name in images:
                img_type = type_mapping.get(img_name, 'other')
                cluster_type_counts[img_type] = cluster_type_counts.get(img_type, 0) + 1
            
            report_lines.append(f"### 聚类 {cluster_id}\n\n")
            report_lines.append(f"**图片数量**: {len(images)}\n\n")
            report_lines.append("**类型组成**:\n")
            for img_type, count in sorted(cluster_type_counts.items()):
                if count > 0:
                    percentage = (count / len(images)) * 100
                    report_lines.append(f"- {img_type}: {count} 张 ({percentage:.1f}%)\n")
            
            # 将图片名称写入文件（排序确保一致性）
            with open(cluster_dir / 'image_list.txt', 'w', encoding='utf-8') as f:
                for img_name in sorted(images):
                    f.write(f"{img_name}\n")
            
            report_lines.append("\n")
        
        # 保存报告
        with open(output_path / 'stable_classification_report.md', 'w', encoding='utf-8') as f:
            f.writelines(report_lines)
        
        print(f"\n稳定分类结果已保存到: {output_path}")
        print(f"- 详细结果: stable_classification_results.json")
        print(f"- 聚类分组: stable_clusters.json")
        print(f"- 分类报告: stable_classification_report.md")
        print(f"- 可视化图片: stable_cluster_visualization.png")

class FixedClusterVideoGenerator:
    """修复版聚类视频生成器（集成到主脚本）"""
    
    def __init__(self, classification_results_dir='stable_classification_results', 
                 image_source_dir='data/train/watermarked'):
        self.results_dir = Path(classification_results_dir)
        self.image_source_dir = Path(image_source_dir)
        self.clusters_file = self.results_dir / 'stable_clusters.json'
        
        # 视频参数
        self.video_fps = 2
        self.video_size = (512, 512)
        self.images_per_video = 100
    
    def load_clusters(self):
        """加载聚类结果"""
        if not self.clusters_file.exists():
            raise FileNotFoundError(f"聚类结果文件不存在: {self.clusters_file}")
        
        with open(self.clusters_file, 'r', encoding='utf-8') as f:
            clusters = json.load(f)
        
        print(f"加载了 {len(clusters)} 个聚类")
        return clusters
    
    def select_images_for_cluster(self, cluster_images, max_images=100):
        """为聚类选择指定数量的图片（确定性选择）"""
        if len(cluster_images) <= max_images:
            return sorted(cluster_images)
        else:
            # 使用固定种子进行确定性选择
            sorted_images = sorted(cluster_images)
            random.seed(42)
            selected = random.sample(sorted_images, max_images)
            return sorted(selected)
    
    def resize_image_to_video_size_fixed(self, image_path):
        """修复版图片尺寸调整函数"""
        try:
            # 直接使用OpenCV处理，避免颜色空间转换问题
            img = cv2.imread(str(image_path))
            if img is None:
                return self._create_fallback_image()
            
            # 获取原始尺寸
            h, w = img.shape[:2]
            target_w, target_h = self.video_size
            
            # 计算缩放比例，保持宽高比
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # 缩放图片
            img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # 创建黑色背景
            background = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            
            # 计算居中位置
            x = (target_w - new_w) // 2
            y = (target_h - new_h) // 2
            
            # 将缩放后的图片放置到背景上
            background[y:y+new_h, x:x+new_w] = img_resized
            
            return background
            
        except Exception as e:
            print(f"处理图片 {image_path} 时出错: {e}")
            return self._create_fallback_image()
    
    def _create_fallback_image(self):
        """创建备用图片（黑色）"""
        return np.zeros((self.video_size[1], self.video_size[0], 3), dtype=np.uint8)
    

def copy_images_to_clusters(image_dir, clusters, target_dir):
    """将图片复制到目标目录的对应类别文件夹中"""
    print(f"\n开始复制图片到目标目录: {target_dir}")
    
    target_path = Path(target_dir)
    target_path.mkdir(exist_ok=True)
    
    source_path = Path(image_dir)
    
    for cluster_id, image_names in clusters.items():
        # 创建类别文件夹
        cluster_dir = target_path / f"cluster_{cluster_id}"
        cluster_dir.mkdir(exist_ok=True)
        
        print(f"复制聚类 {cluster_id} 的 {len(image_names)} 张图片...")
        
        for image_name in image_names:
            source_file = source_path / image_name
            target_file = cluster_dir / image_name
            
            try:
                if source_file.exists():
                    shutil.copy2(source_file, target_file)
                else:
                    print(f"警告: 源文件不存在 {source_file}")
            except Exception as e:
                print(f"复制文件失败 {source_file} -> {target_file}: {e}")
    
    print(f"✓ 图片复制完成到: {target_dir}")

def main():
    """主函数"""
    import argparse
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='稳定图像分类工具 v2.0')
    parser.add_argument('--image_dir', '-i', type=str, default='data/train/watermarked',
                       help='输入图片目录路径 (默认: data/train/watermarked)')
    parser.add_argument('--target_dir', '-t', type=str, default='data/classfy/',
                       help='目标输出目录路径 (必需)')
    parser.add_argument('--n_clusters', '-n', type=int, default=6,
                       help='聚类数量 (默认: 6)')
    parser.add_argument('--batch_size', '-b', type=int, default=8,
                       help='批处理大小 (默认: 8)')
    parser.add_argument('--output_dir', '-o', type=str, default='data/stable_classification',
                       help='分析结果输出目录')
    parser.add_argument('--no_cache', action='store_true',
                       help='禁用特征缓存')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 配置参数
    image_dir = args.image_dir
    target_dir = args.target_dir
    n_clusters = args.n_clusters
    batch_size = args.batch_size
    output_dir = args.output_dir
    use_cache = not args.no_cache
    
    print("=== 稳定图像分类 v2.0 开始 ===")
    print(f"图片目录: {image_dir}")
    print(f"目标目录: {target_dir}")
    print(f"聚类数量: {n_clusters}")
    print(f"批处理大小: {batch_size}")
    print(f"分析结果目录: {output_dir}")
    print(f"使用缓存: {use_cache}")
    print(f"随机种子: 42 (固定)")
    
    # 检查图片目录是否存在
    if not os.path.exists(image_dir):
        print(f"错误: 图片目录 {image_dir} 不存在!")
        return
    
    try:
        # 创建稳定分类器
        classifier = StableImageClassifier()
        
        # 提取特征（支持缓存）
        classifier.extract_features(image_dir, batch_size=batch_size, use_cache=use_cache)
        
        if len(classifier.features) == 0:
            print("错误: 没有成功提取到任何特征!")
            return
        
        # 分析图片类型
        classifier.analyze_image_types()
        
        # 进行稳定聚类
        results, clusters = classifier.stable_cluster_images(n_clusters=n_clusters)
        
        # 分析聚类组成
        classifier.analyze_cluster_composition(clusters)
        
        # 生成可视化
        classifier.visualize_clusters(clusters, output_dir)
        
        # 保存分析结果
        classifier.save_results(results, clusters, output_dir)
        
        # 复制图片到目标目录的对应类别文件夹
        copy_images_to_clusters(image_dir, clusters, target_dir)
        
        print(f"\n=== 分类完成 ===")
        print(f"分析结果已保存到: {output_dir}")
        print(f"分类图片已复制到: {target_dir}")
        
    except Exception as e:
        print(f"分类过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()