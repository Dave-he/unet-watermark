#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DINOv2图像分类脚本
使用DINOv2模型对data/train/watermarked文件夹下的图片进行分类
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from pathlib import Path

# 检查是否有GPU可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

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

class DINOv2Classifier:
    """DINOv2分类器"""
    
    def __init__(self, model_name='dinov2_vitb14'):
        self.model_name = model_name
        self.model = None
        self.features = []
        self.image_names = []
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def load_model(self):
        """加载DINOv2模型"""
        try:
            print(f"正在加载 {self.model_name} 模型...")
            # 从torch hub加载DINOv2模型
            self.model = torch.hub.load('facebookresearch/dinov2', self.model_name)
            self.model.to(device)
            self.model.eval()
            print("模型加载成功!")
        except Exception as e:
            print(f"模型加载失败: {e}")
            print("尝试使用备用方法...")
            # 备用方法：直接从transformers库加载
            try:
                from transformers import AutoModel, AutoImageProcessor
                self.model = AutoModel.from_pretrained('facebook/dinov2-base')
                self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
                self.model.to(device)
                self.model.eval()
                print("使用transformers库加载模型成功!")
            except Exception as e2:
                print(f"备用方法也失败: {e2}")
                raise e2
    
    def extract_features(self, image_dir, batch_size=32):
        """提取图片特征"""
        if self.model is None:
            self.load_model()
        
        # 创建数据集和数据加载器
        dataset = WatermarkDataset(image_dir, transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        print("开始提取特征...")
        self.features = []
        self.image_names = []
        
        with torch.no_grad():
            for batch_images, batch_names in tqdm(dataloader, desc="提取特征"):
                batch_images = batch_images.to(device)
                
                # 提取特征
                if hasattr(self, 'processor'):
                    # 使用transformers版本
                    outputs = self.model(batch_images)
                    batch_features = outputs.last_hidden_state[:, 0]  # CLS token
                else:
                    # 使用torch hub版本
                    batch_features = self.model(batch_images)
                
                # 转换为numpy数组
                batch_features = batch_features.cpu().numpy()
                
                self.features.extend(batch_features)
                self.image_names.extend(batch_names)
        
        self.features = np.array(self.features)
        print(f"特征提取完成! 特征维度: {self.features.shape}")
    
    def cluster_images(self, n_clusters=5, use_pca=True, pca_components=50):
        """对图片进行聚类分析"""
        if len(self.features) == 0:
            raise ValueError("请先提取特征!")
        
        print(f"开始聚类分析，聚类数量: {n_clusters}")
        
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
            'cluster_centers': kmeans.cluster_centers_.tolist() if not use_pca else None
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
    
    def visualize_clusters(self, clusters, output_dir='classification_results'):
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
        plt.title('DINOv2特征聚类可视化 (PCA降维到2D)')
        plt.xlabel('第一主成分')
        plt.ylabel('第二主成分')
        
        # 保存图片
        plt.savefig(output_path / 'cluster_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"可视化结果已保存到: {output_path / 'cluster_visualization.png'}")
    
    def save_results(self, results, clusters, output_dir='classification_results'):
        """保存分类结果"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 保存详细结果
        with open(output_path / 'classification_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 保存聚类分组
        with open(output_path / 'clusters.json', 'w', encoding='utf-8') as f:
            # 转换为可序列化的格式
            serializable_clusters = {str(k): v for k, v in clusters.items()}
            json.dump(serializable_clusters, f, ensure_ascii=False, indent=2)
        
        # 创建每个聚类的文件夹并生成报告
        report_lines = ["# DINOv2图像分类报告\n"]
        report_lines.append(f"总图片数量: {len(self.image_names)}\n")
        report_lines.append(f"聚类数量: {len(clusters)}\n\n")
        
        for cluster_id, images in clusters.items():
            cluster_dir = output_path / f'cluster_{cluster_id}'
            cluster_dir.mkdir(exist_ok=True)
            
            report_lines.append(f"## 聚类 {cluster_id}\n")
            report_lines.append(f"图片数量: {len(images)}\n")
            report_lines.append("图片列表:\n")
            
            # 将图片名称写入文件
            with open(cluster_dir / 'image_list.txt', 'w', encoding='utf-8') as f:
                for img_name in images:
                    f.write(f"{img_name}\n")
                    report_lines.append(f"- {img_name}\n")
            
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
    n_clusters = 8  # 可以根据需要调整聚类数量
    batch_size = 16  # 根据GPU内存调整
    output_dir = 'dinov2_classification_results'
    
    print("=== DINOv2图像分类开始 ===")
    print(f"图片目录: {image_dir}")
    print(f"聚类数量: {n_clusters}")
    print(f"批处理大小: {batch_size}")
    print(f"输出目录: {output_dir}")
    
    # 检查图片目录是否存在
    if not os.path.exists(image_dir):
        print(f"错误: 图片目录 {image_dir} 不存在!")
        return
    
    try:
        # 创建分类器
        classifier = DINOv2Classifier()
        
        # 提取特征
        classifier.extract_features(image_dir, batch_size=batch_size)
        
        # 进行聚类
        results, clusters = classifier.cluster_images(n_clusters=n_clusters)
        
        # 可视化结果
        classifier.visualize_clusters(clusters, output_dir)
        
        # 保存结果
        classifier.save_results(results, clusters, output_dir)
        
        print("\n=== 分类完成! ===")
        
    except Exception as e:
        print(f"分类过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()