#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为每个聚类生成视频的脚本
从分类结果中为每个类别挑选100张图片生成视频
"""

import os
import json
import random
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import shutil
import zipfile
import datetime

class ClusterVideoGenerator:
    """聚类视频生成器"""
    
    def __init__(self, classification_results_dir='robust_classification_results', 
                 image_source_dir='data/train/watermarked'):
        self.results_dir = Path(classification_results_dir)
        self.image_source_dir = Path(image_source_dir)
        self.clusters_file = self.results_dir / 'clusters.json'
        
        # 视频参数
        self.video_fps = 2  # 每秒2帧，可以清楚看到每张图片
        self.video_size = (512, 512)  # 视频尺寸
        self.images_per_video = 100  # 每个视频包含的图片数量
        
    def load_clusters(self):
        """加载聚类结果"""
        if not self.clusters_file.exists():
            raise FileNotFoundError(f"聚类结果文件不存在: {self.clusters_file}")
        
        with open(self.clusters_file, 'r', encoding='utf-8') as f:
            clusters = json.load(f)
        
        print(f"加载了 {len(clusters)} 个聚类")
        for cluster_id, images in clusters.items():
            print(f"聚类 {cluster_id}: {len(images)} 张图片")
        
        return clusters
    
    def select_images_for_cluster(self, cluster_images, max_images=100):
        """为聚类选择指定数量的图片"""
        if len(cluster_images) <= max_images:
            return cluster_images
        else:
            # 随机选择指定数量的图片
            return random.sample(cluster_images, max_images)
    
    def resize_image_to_video_size(self, image_path):
        """将图片调整到视频尺寸"""
        try:
            # 使用PIL加载图片
            img = Image.open(image_path).convert('RGB')
            
            # 计算缩放比例，保持宽高比
            img_w, img_h = img.size
            target_w, target_h = self.video_size
            
            # 计算缩放比例
            scale = min(target_w / img_w, target_h / img_h)
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
            
            # 缩放图片
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # 创建黑色背景
            background = Image.new('RGB', self.video_size, (0, 0, 0))
            
            # 将缩放后的图片居中放置
            x = (target_w - new_w) // 2
            y = (target_h - new_h) // 2
            background.paste(img, (x, y))
            
            # 转换为numpy数组（BGR格式用于OpenCV）
            img_array = np.array(background)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            return img_bgr
            
        except Exception as e:
            print(f"处理图片 {image_path} 时出错: {e}")
            # 返回黑色图片
            return np.zeros((self.video_size[1], self.video_size[0], 3), dtype=np.uint8)
    
    def create_video_for_cluster(self, cluster_id, selected_images, output_dir):
        """为指定聚类创建视频"""
        cluster_dir = output_dir / f'cluster_{cluster_id}'
        cluster_dir.mkdir(exist_ok=True)
        
        video_path = cluster_dir / f'cluster_{cluster_id}_video.mp4'
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            str(video_path), 
            fourcc, 
            self.video_fps, 
            self.video_size
        )
        
        print(f"\n为聚类 {cluster_id} 创建视频 ({len(selected_images)} 张图片)...")
        
        # 复制选中的图片到聚类目录
        selected_images_dir = cluster_dir / 'selected_images'
        selected_images_dir.mkdir(exist_ok=True)
        
        valid_images = []
        
        for i, img_name in enumerate(tqdm(selected_images, desc=f"处理聚类 {cluster_id}")):
            img_path = self.image_source_dir / img_name
            
            if not img_path.exists():
                print(f"警告: 图片文件不存在: {img_path}")
                continue
            
            # 复制图片到聚类目录
            dest_path = selected_images_dir / img_name
            try:
                shutil.copy2(img_path, dest_path)
            except Exception as e:
                print(f"复制图片失败 {img_path}: {e}")
                continue
            
            # 处理图片并添加到视频
            processed_img = self.resize_image_to_video_size(img_path)
            
            # 每张图片显示多帧（让观看更清楚）
            frames_per_image = max(1, int(self.video_fps * 0.5))  # 每张图片显示0.5秒
            for _ in range(frames_per_image):
                video_writer.write(processed_img)
            
            valid_images.append(img_name)
        
        video_writer.release()
        
        if len(valid_images) > 0:
            print(f"✓ 聚类 {cluster_id} 视频创建完成: {video_path}")
            print(f"  包含 {len(valid_images)} 张有效图片")
            
            # 保存选中图片列表
            with open(cluster_dir / 'selected_images_list.txt', 'w', encoding='utf-8') as f:
                for img_name in valid_images:
                    f.write(f"{img_name}\n")
        else:
            print(f"✗ 聚类 {cluster_id} 没有有效图片，删除空视频文件")
            if video_path.exists():
                video_path.unlink()
        
        return len(valid_images)
    
    def generate_all_videos(self, output_dir='cluster_videos'):
        """为所有聚类生成视频"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 加载聚类结果
        clusters = self.load_clusters()
        
        print(f"\n开始为 {len(clusters)} 个聚类生成视频...")
        print(f"每个视频最多包含 {self.images_per_video} 张图片")
        print(f"视频尺寸: {self.video_size}")
        print(f"视频帧率: {self.video_fps} FPS")
        print(f"输出目录: {output_path}")
        
        total_videos = 0
        total_images = 0
        
        for cluster_id, cluster_images in clusters.items():
            # 为每个聚类选择图片
            selected_images = self.select_images_for_cluster(
                cluster_images, 
                self.images_per_video
            )
            
            # 创建视频
            valid_count = self.create_video_for_cluster(
                cluster_id, 
                selected_images, 
                output_path
            )
            
            if valid_count > 0:
                total_videos += 1
                total_images += valid_count
        
        # 生成总结报告
        self.generate_summary_report(output_path, clusters, total_videos, total_images)
        
        print(f"\n=== 视频生成完成! ===")
        print(f"成功生成 {total_videos} 个视频")
        print(f"总共处理 {total_images} 张图片")
        print(f"输出目录: {output_path}")
    
    def generate_summary_report(self, output_path, clusters, total_videos, total_images):
        """生成总结报告"""
        report_lines = ["# 聚类视频生成报告\n\n"]
        
        report_lines.append(f"**生成时间**: {Path().cwd()}\n")
        report_lines.append(f"**总聚类数**: {len(clusters)}\n")
        report_lines.append(f"**成功生成视频数**: {total_videos}\n")
        report_lines.append(f"**总处理图片数**: {total_images}\n")
        report_lines.append(f"**每个视频最大图片数**: {self.images_per_video}\n")
        report_lines.append(f"**视频参数**: {self.video_size[0]}x{self.video_size[1]}, {self.video_fps} FPS\n\n")
        
        report_lines.append("## 各聚类详情\n\n")
        
        for cluster_id, cluster_images in clusters.items():
            cluster_dir = output_path / f'cluster_{cluster_id}'
            video_file = cluster_dir / f'cluster_{cluster_id}_video.mp4'
            
            report_lines.append(f"### 聚类 {cluster_id}\n\n")
            report_lines.append(f"- **原始图片数**: {len(cluster_images)}\n")
            
            if video_file.exists():
                selected_count = min(len(cluster_images), self.images_per_video)
                report_lines.append(f"- **选中图片数**: {selected_count}\n")
                report_lines.append(f"- **视频文件**: `cluster_{cluster_id}_video.mp4`\n")
                report_lines.append(f"- **状态**: ✓ 成功生成\n")
            else:
                report_lines.append(f"- **状态**: ✗ 生成失败\n")
            
            report_lines.append("\n")
        
        report_lines.append("## 文件结构\n\n")
        report_lines.append("```\n")
        report_lines.append(f"{output_path.name}/\n")
        for cluster_id in clusters.keys():
            report_lines.append(f"├── cluster_{cluster_id}/\n")
            report_lines.append(f"│   ├── cluster_{cluster_id}_video.mp4\n")
            report_lines.append(f"│   ├── selected_images_list.txt\n")
            report_lines.append(f"│   └── selected_images/\n")
            report_lines.append(f"│       └── (选中的图片文件)\n")
        report_lines.append("└── video_generation_report.md\n")
        report_lines.append("```\n")
        
        # 保存报告
        with open(output_path / 'video_generation_report.md', 'w', encoding='utf-8') as f:
            f.writelines(report_lines)
        
        print(f"\n总结报告已保存: {output_path / 'video_generation_report.md'}")

def gen_main():
    # 设置随机种子以确保可重现性
    random.seed(42)
    """主函数"""
    print("=== 聚类视频生成器 ===")
    
    # 检查必要文件是否存在
    results_dir = 'data/classfy'
    image_dir = 'data/train/watermarked'
    
    if not os.path.exists(results_dir):
        print(f"错误: 分类结果目录不存在: {results_dir}")
        print("请先运行 robust_image_classification.py 进行图片分类")
        return
    
    if not os.path.exists(image_dir):
        print(f"错误: 图片源目录不存在: {image_dir}")
        return
    
    try:
        # 创建视频生成器
        generator = ClusterVideoGenerator(
            classification_results_dir=results_dir,
            image_source_dir=image_dir
        )
        
        # 生成所有视频
        output_dir = 'data/classfy'
        generator.generate_all_videos(output_dir=output_dir)
        
        # 压缩生成的目录
        if os.path.exists(output_dir):
            print(f"\n开始压缩目录: {output_dir}")
            
            # 生成带时间戳的压缩文件名
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_filename = f"cluster_videos_{timestamp}.zip"
            
            # 创建压缩文件
            with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(output_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        # 在压缩文件中保持相对路径结构
                        arcname = os.path.relpath(file_path, os.path.dirname(output_dir))
                        zipf.write(file_path, arcname)
            
            print(f"✓ 压缩完成: {zip_filename}")
            
            # 删除原目录
            print(f"删除原目录: {output_dir}")
            shutil.rmtree(output_dir)
            print(f"✓ 原目录已删除")
            
            print(f"\n=== 处理完成! ===")
            print(f"压缩文件: {zip_filename}")
            
            # 显示压缩文件大小
            zip_size = os.path.getsize(zip_filename) / (1024 * 1024)  # MB
            print(f"压缩文件大小: {zip_size:.2f} MB")
        
    except Exception as e:
        print(f"视频生成过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    gen_main()