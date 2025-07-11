#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
鲁棒性图像分类脚本
专门处理DINOv2加载失败的情况，提供多种备选方案
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 检查是否有GPU可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
from generate_cluster_videos import gen_main

class WatermarkDataset(Dataset):
    """水印图片数据集"""
    
    def __init__(self, image_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        
        # 获取所有jpg图片文件
        self.image_paths = list(self.image_dir.glob('*.jpg'))
        print(f"找到 {len(self.image_paths)} 张图片")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            # 加载图片
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, str(img_path.name)
        except Exception as e:
            print(f"加载图片 {img_path} 时出错: {e}")
            # 返回一个默认的黑色图片
            if self.transform:
                default_img = self.transform(Image.new('RGB', (224, 224), (0, 0, 0)))
            else:
                default_img = Image.new('RGB', (224, 224), (0, 0, 0))
            return default_img, str(img_path.name)

class RobustImageClassifier:
    """鲁棒性图像分类器"""
    
    def __init__(self):
        self.model = None
        self.model_type = None
        self.features = []
        self.image_names = []
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def try_load_dinov2(self):
        """尝试加载DINOv2模型"""
        try:
            raise ValueError("参数无效")
            print("正在尝试加载DINOv2模型...")
            # 方法1: 从torch hub加载
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=True)
            self.model_type = 'dinov2_hub'
            print("✓ DINOv2模型 (torch hub) 加载成功!")
            return True
        except Exception as e1:
            print(f"✗ torch hub加载失败: {e1}")
            
            try:
                print("尝试使用transformers库加载DINOv2...")
                from transformers import AutoModel, AutoImageProcessor
                self.model = AutoModel.from_pretrained('facebook/dinov2-base')
                self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
                self.model_type = 'dinov2_transformers'
                print("✓ DINOv2模型 (transformers) 加载成功!")
                return True
            except Exception as e2:
                print(f"✗ transformers加载失败: {e2}")
                
                try:
                    print("尝试使用timm库加载DINOv2...")
                    import timm
                    self.model = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True)
                    self.model_type = 'dinov2_timm'
                    print("✓ DINOv2模型 (timm) 加载成功!")
                    return True
                except Exception as e3:
                    print(f"✗ timm加载失败: {e3}")
                    return False
    
    def load_backup_model(self):
        """加载备选模型"""
        print("\n所有DINOv2加载方法都失败，使用备选模型...")
        
        # 尝试多个备选模型
        backup_models = [
            ('resnet50', lambda: models.resnet50(pretrained=True)),
            ('resnet101', lambda: models.resnet101(pretrained=True)),
            ('efficientnet_b0', lambda: models.efficientnet_b0(pretrained=True)),
            ('vit_b_16', lambda: models.vit_b_16(pretrained=True))
        ]
        
        for model_name, model_loader in backup_models:
            try:
                print(f"尝试加载 {model_name}...")
                self.model = model_loader()
                
                # 移除分类层，只保留特征提取
                if 'resnet' in model_name:
                    self.model = nn.Sequential(*list(self.model.children())[:-1])
                elif 'efficientnet' in model_name:
                    self.model = nn.Sequential(*list(self.model.children())[:-1])
                elif 'vit' in model_name:
                    # ViT模型需要特殊处理
                    pass
                
                self.model_type = model_name
                print(f"✓ {model_name} 模型加载成功!")
                return True
            except Exception as e:
                print(f"✗ {model_name} 加载失败: {e}")
        
        raise Exception("所有模型加载都失败了!")
    
    def load_model(self):
        """加载模型（优先DINOv2，失败则使用备选）"""
        # 首先尝试DINOv2
        if not self.try_load_dinov2():
            # DINOv2失败，使用备选模型
            self.load_backup_model()
        
        self.model.to(device)
        self.model.eval()
        print(f"\n最终使用模型: {self.model_type}")
    
    def extract_features(self, image_dir, batch_size=16):
        """提取图片特征"""
        if self.model is None:
            self.load_model()
        
        # 创建数据集和数据加载器
        dataset = WatermarkDataset(image_dir, transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        print(f"\n开始使用{self.model_type}模型提取特征...")
        self.features = []
        self.image_names = []
        
        with torch.no_grad():
            for batch_images, batch_names in tqdm(dataloader, desc="提取特征"):
                batch_images = batch_images.to(device)
                
                try:
                    # 根据模型类型提取特征
                    if 'dinov2' in self.model_type:
                        if self.model_type == 'dinov2_transformers':
                            outputs = self.model(batch_images)
                            batch_features = outputs.last_hidden_state[:, 0]  # CLS token
                        else:
                            batch_features = self.model(batch_images)
                    elif 'vit' in self.model_type and 'dinov2' not in self.model_type:
                        # 标准ViT模型
                        batch_features = self.model(batch_images)
                        if len(batch_features.shape) > 2:
                            batch_features = batch_features.mean(dim=1)  # 全局平均池化
                    else:
                        # ResNet, EfficientNet等
                        batch_features = self.model(batch_images)
                        batch_features = batch_features.view(batch_features.size(0), -1)  # 展平
                    
                    # 转换为numpy数组
                    batch_features = batch_features.cpu().numpy()
                    
                    self.features.extend(batch_features)
                    self.image_names.extend(batch_names)
                    
                except Exception as e:
                    print(f"处理批次时出错: {e}")
                    # 跳过这个批次
                    continue
        
        self.features = np.array(self.features)
        print(f"特征提取完成! 特征维度: {self.features.shape}")
    
    def analyze_image_types(self):
        """分析图片类型（基于文件名模式）"""
        type_counts = {
            'mixed_trans': 0,
            'norm': 0, 
            'text_trans': 0,
            'trans': 0,
            'other': 0
        }
        
        type_mapping = {}
        
        for img_name in self.image_names:
            if img_name.startswith('mixed_trans_'):
                img_type = 'mixed_trans'
            elif img_name.startswith('norm_'):
                img_type = 'norm'
            elif img_name.startswith('text_trans_'):
                img_type = 'text_trans'
            elif img_name.startswith('trans_'):
                img_type = 'trans'
            else:
                img_type = 'other'
            
            type_counts[img_type] += 1
            type_mapping[img_name] = img_type
        
        print("\n基于文件名的图片类型分析:")
        for img_type, count in type_counts.items():
            if count > 0:
                print(f"{img_type}: {count} 张图片")
        
        return type_mapping, type_counts
    
    def cluster_images(self, n_clusters=6, use_pca=True, pca_components=50):
        """对图片进行聚类分析"""
        if len(self.features) == 0:
            raise ValueError("请先提取特征!")
        
        print(f"\n开始聚类分析，聚类数量: {n_clusters}")
        
        # 可选：使用PCA降维
        features_for_clustering = self.features
        if use_pca and self.features.shape[1] > pca_components:
            print(f"使用PCA降维到 {pca_components} 维...")
            pca = PCA(n_components=pca_components)
            features_for_clustering = pca.fit_transform(self.features)
            print(f"PCA解释方差比: {pca.explained_variance_ratio_.sum():.3f}")
        
        # K-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_for_clustering)
        
        # 整理结果
        results = {
            'cluster_labels': cluster_labels.tolist(),
            'image_names': self.image_names,
            'n_clusters': n_clusters,
            'model_type': self.model_type,
            'feature_dim': self.features.shape[1]
        }
        
        # 按聚类分组
        clusters = {}
        for i, (img_name, label) in enumerate(zip(self.image_names, cluster_labels)):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(img_name)
        
        # 打印聚类结果统计
        print("\n聚类结果统计:")
        for cluster_id, images in clusters.items():
            print(f"聚类 {cluster_id}: {len(images)} 张图片")
        
        return results, clusters
    
    def analyze_cluster_composition(self, clusters):
        """分析每个聚类的组成"""
        type_mapping, _ = self.analyze_image_types()
        
        print("\n聚类组成分析:")
        for cluster_id, images in clusters.items():
            type_counts = {}
            for img_name in images:
                img_type = type_mapping.get(img_name, 'other')
                type_counts[img_type] = type_counts.get(img_type, 0) + 1
            
            print(f"\n聚类 {cluster_id} ({len(images)} 张图片):")
            for img_type, count in sorted(type_counts.items()):
                if count > 0:
                    percentage = (count / len(images)) * 100
                    print(f"  {img_type}: {count} 张 ({percentage:.1f}%)")
    
    def visualize_clusters(self, clusters, output_dir='robust_classification_results'):
        """可视化聚类结果"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 使用PCA降维到2D进行可视化
        if self.features.shape[1] > 2:
            pca_2d = PCA(n_components=2)
            features_2d = pca_2d.fit_transform(self.features)
        else:
            features_2d = self.features
        
        # 创建聚类标签
        cluster_labels = []
        for img_name in self.image_names:
            for cluster_id, images in clusters.items():
                if img_name in images:
                    cluster_labels.append(cluster_id)
                    break
        
        # 绘制散点图
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                            c=cluster_labels, cmap='tab10', alpha=0.7)
        plt.colorbar(scatter)
        plt.title(f'{self.model_type.upper()}特征聚类可视化 (PCA降维到2D)')
        plt.xlabel('第一主成分')
        plt.ylabel('第二主成分')
        
        # 保存图片
        plt.savefig(output_path / 'cluster_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"可视化结果已保存到: {output_path / 'cluster_visualization.png'}")
    
    def save_results(self, results, clusters, output_dir='robust_classification_results'):
        """保存分类结果"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 保存详细结果
        with open(output_path / 'classification_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 保存聚类分组
        with open(output_path / 'clusters.json', 'w', encoding='utf-8') as f:
            serializable_clusters = {str(k): v for k, v in clusters.items()}
            json.dump(serializable_clusters, f, ensure_ascii=False, indent=2)
        
        # 分析图片类型
        type_mapping, type_counts = self.analyze_image_types()
        
        # 创建详细报告
        report_lines = ["# 鲁棒性图像分类报告\n\n"]
        report_lines.append(f"**使用模型**: {results['model_type']}\n")
        report_lines.append(f"**总图片数量**: {len(self.image_names)}\n")
        report_lines.append(f"**特征维度**: {results['feature_dim']}\n")
        report_lines.append(f"**聚类数量**: {len(clusters)}\n\n")
        
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
            
            # 将图片名称写入文件
            with open(cluster_dir / 'image_list.txt', 'w', encoding='utf-8') as f:
                for img_name in sorted(images):
                    f.write(f"{img_name}\n")
            
            report_lines.append("\n")
        
        # 保存报告
        with open(output_path / 'classification_report.md', 'w', encoding='utf-8') as f:
            f.writelines(report_lines)
        
        print(f"\n分类结果已保存到: {output_path}")
        print(f"- 详细结果: classification_results.json")
        print(f"- 聚类分组: clusters.json")
        print(f"- 分类报告: classification_report.md")
        print(f"- 可视化图片: cluster_visualization.png")

def main():
    """主函数"""
    # 配置参数
    image_dir = 'data/train/watermarked'
    n_clusters = 6  # 根据观察到的图片类型调整
    batch_size = 8   # 保守的批处理大小
    output_dir = 'data/classfy'
    
    print("=== 鲁棒性图像分类开始 ===")
    print(f"图片目录: {image_dir}")
    print(f"聚类数量: {n_clusters}")
    print(f"批处理大小: {batch_size}")
    print(f"输出目录: {output_dir}")
    
    # 检查图片目录是否存在
    if not os.path.exists(image_dir):
        print(f"错误: 图片目录 {image_dir} 不存在!")
        return
    
    try:
        # 创建鲁棒性分类器
        classifier = RobustImageClassifier()
        
        # 提取特征
        classifier.extract_features(image_dir, batch_size=batch_size)
        
        if len(classifier.features) == 0:
            print("错误: 没有成功提取到任何特征!")
            return
        
        # 分析图片类型
        classifier.analyze_image_types()
        
        # 进行聚类
        results, clusters = classifier.cluster_images(n_clusters=n_clusters)
        
        # 分析聚类组成
        classifier.analyze_cluster_composition(clusters)
        
        # 可视化结果
        classifier.visualize_clusters(clusters, output_dir)
        
        # 保存结果
        classifier.save_results(results, clusters, output_dir)
        
        print("\n=== 分类完成! ===")
        
        # 提供DINOv2故障排除建议
        if 'dinov2' not in classifier.model_type:
            print("\n=== DINOv2故障排除建议 ===")
            print("1. 检查网络连接，确保可以访问GitHub和Hugging Face")
            print("2. 尝试安装额外依赖: pip install timm transformers")
            print("3. 如果在公司网络，可能需要配置代理")
            print("4. 可以手动下载模型文件到本地")
            print("5. 当前使用的备选模型也能提供良好的分类效果")
        
    except Exception as e:
        print(f"分类过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
    gen_main()