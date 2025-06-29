#!/usr/bin/env python3
"""
水印提取工具
对比clean和watermarked文件夹中的图片，生成透明背景的水印素材
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import argparse
from pathlib import Path
import logging
from sklearn.cluster import DBSCAN
from scipy import ndimage
import random

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WatermarkExtractor:
    def __init__(self, clean_dir, watermarked_dir, output_dir, threshold=30, min_area=100):
        self.clean_dir = Path(clean_dir)
        self.watermarked_dir = Path(watermarked_dir)
        self.output_dir = Path(output_dir)
        self.threshold = threshold
        self.min_area = min_area
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_watermark_from_pair(self, clean_path, watermarked_path):
        """从一对图片中提取水印"""
        try:
            # 读取图片
            clean_img = cv2.imread(str(clean_path))
            watermarked_img = cv2.imread(str(watermarked_path))
            
            if clean_img is None or watermarked_img is None:
                logger.warning(f"无法读取图片: {clean_path} 或 {watermarked_path}")
                return None
                
            # 确保两张图片尺寸相同
            if clean_img.shape != watermarked_img.shape:
                # 调整到相同尺寸
                h, w = min(clean_img.shape[0], watermarked_img.shape[0]), min(clean_img.shape[1], watermarked_img.shape[1])
                clean_img = cv2.resize(clean_img, (w, h))
                watermarked_img = cv2.resize(watermarked_img, (w, h))
            
            # 计算差异
            diff = cv2.absdiff(clean_img, watermarked_img)
            
            # 转换为灰度图
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            
            # 应用阈值
            _, binary_mask = cv2.threshold(gray_diff, self.threshold, 255, cv2.THRESH_BINARY)
            
            # 形态学操作去除噪声
            kernel = np.ones((3,3), np.uint8)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
            
            # 查找连通区域
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                logger.warning(f"未找到水印区域: {clean_path.name}")
                return None
                
            # 过滤小区域
            valid_contours = [c for c in contours if cv2.contourArea(c) > self.min_area]
            
            if not valid_contours:
                logger.warning(f"未找到足够大的水印区域: {clean_path.name}")
                return None
                
            return self._process_watermark_regions(watermarked_img, valid_contours, clean_path.stem)
            
        except Exception as e:
            logger.error(f"处理图片对时出错 {clean_path.name}: {str(e)}")
            return None
    
    def _process_watermark_regions(self, watermarked_img, contours, base_name):
        """处理水印区域，如果距离较远则分成多个素材"""
        if len(contours) == 1:
            # 只有一个区域，直接提取
            return self._extract_single_watermark(watermarked_img, contours[0], f"{base_name}_watermark.png")
        
        # 多个区域，计算中心点
        centers = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centers.append((cx, cy))
        
        if len(centers) < 2:
            # 如果中心点少于2个，合并处理
            return self._extract_combined_watermark(watermarked_img, contours, f"{base_name}_watermark.png")
        
        # 使用DBSCAN聚类判断是否需要分离
        centers_array = np.array(centers)
        
        # 计算图片对角线长度作为距离参考
        h, w = watermarked_img.shape[:2]
        diagonal = np.sqrt(h*h + w*w)
        eps = diagonal * 0.25  # 如果距离超过对角线的25%，则认为是分离的，更容易分离远距离水印
        
        clustering = DBSCAN(eps=eps, min_samples=1).fit(centers_array)
        labels = clustering.labels_
        
        unique_labels = set(labels)
        
        if len(unique_labels) == 1:
            # 所有区域都在一个聚类中，合并处理
            return self._extract_combined_watermark(watermarked_img, contours, f"{base_name}_watermark.png")
        else:
            # 分成多个聚类，分别处理
            results = []
            for i, label in enumerate(unique_labels):
                cluster_contours = [contours[j] for j, l in enumerate(labels) if l == label]
                result = self._extract_combined_watermark(
                    watermarked_img, 
                    cluster_contours, 
                    f"{base_name}_watermark_part{i+1}.png"
                )
                if result:
                    results.append(result)
            return results if results else None
    
    def _extract_single_watermark(self, img, contour, filename):
        """提取单个水印区域"""
        # 获取边界框
        x, y, w, h = cv2.boundingRect(contour)
        
        # 适当放大边距以保证水印区域完整性
        margin_x = max(20, int(w * 0.15))  # 至少20像素或宽度的15%
        margin_y = max(20, int(h * 0.15))  # 至少20像素或高度的15%
        
        x = max(0, x - margin_x)
        y = max(0, y - margin_y)
        w = min(img.shape[1] - x, w + 2 * margin_x)
        h = min(img.shape[0] - y, h + 2 * margin_y)
        
        # 提取区域
        watermark_region = img[y:y+h, x:x+w]
        
        # 创建精确的mask
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        cv2.fillPoly(mask, [contour], 255)
        region_mask = mask[y:y+h, x:x+w]
        
        # 创建RGBA图像
        watermark_rgba = self._create_transparent_watermark(watermark_region, region_mask)
        
        # 保存
        output_path = self.output_dir / filename
        cv2.imwrite(str(output_path), watermark_rgba)
        logger.info(f"保存水印: {output_path}")
        
        return str(output_path)
    
    def _extract_combined_watermark(self, img, contours, filename):
        """提取合并的水印区域"""
        # 获取所有轮廓的边界框
        all_points = np.vstack([contour.reshape(-1, 2) for contour in contours])
        x, y, w, h = cv2.boundingRect(all_points)
        
        # 适当放大边距以保证水印区域完整性
        margin_x = max(25, int(w * 0.12))  # 至少25像素或宽度的12%
        margin_y = max(25, int(h * 0.12))  # 至少25像素或高度的12%
        
        x = max(0, x - margin_x)
        y = max(0, y - margin_y)
        w = min(img.shape[1] - x, w + 2 * margin_x)
        h = min(img.shape[0] - y, h + 2 * margin_y)
        
        # 提取区域
        watermark_region = img[y:y+h, x:x+w]
        
        # 创建合并的mask
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        for contour in contours:
            cv2.fillPoly(mask, [contour], 255)
        region_mask = mask[y:y+h, x:x+w]
        
        # 创建RGBA图像
        watermark_rgba = self._create_transparent_watermark(watermark_region, region_mask)
        
        # 保存
        output_path = self.output_dir / filename
        cv2.imwrite(str(output_path), watermark_rgba)
        logger.info(f"保存合并水印: {output_path}")
        
        return str(output_path)
    
    def _create_transparent_watermark(self, watermark_region, mask):
        """创建透明背景的水印图像"""
        # 转换为RGBA
        if len(watermark_region.shape) == 3:
            watermark_rgba = cv2.cvtColor(watermark_region, cv2.COLOR_BGR2BGRA)
        else:
            watermark_rgba = cv2.cvtColor(watermark_region, cv2.COLOR_GRAY2BGRA)
        
        # 设置alpha通道
        watermark_rgba[:, :, 3] = mask
        
        # 增强对比度以提高清晰度
        watermark_pil = Image.fromarray(cv2.cvtColor(watermark_rgba, cv2.COLOR_BGRA2RGBA))
        enhancer = ImageEnhance.Contrast(watermark_pil)
        watermark_pil = enhancer.enhance(1.3)  # 增强对比度
        
        # 锐化
        enhancer = ImageEnhance.Sharpness(watermark_pil)
        watermark_pil = enhancer.enhance(1.4)  # 增强锐度
        
        # 增强亮度以提高可见性
        enhancer = ImageEnhance.Brightness(watermark_pil)
        watermark_pil = enhancer.enhance(1.1)  # 轻微增强亮度
        
        # 转换回OpenCV格式
        watermark_rgba = cv2.cvtColor(np.array(watermark_pil), cv2.COLOR_RGBA2BGRA)
        
        return watermark_rgba
    
    def process_all_images(self, limit=None):
        """处理所有图片对"""
        clean_files = {f.stem: f for f in self.clean_dir.glob('*.jpg')}
        watermarked_files = {f.stem: f for f in self.watermarked_dir.glob('*.jpg')}
        
        # 找到匹配的文件对
        common_files = set(clean_files.keys()) & set(watermarked_files.keys())
        common_files = list(common_files)  # 转换为列表以支持随机选择
        
        # 如果设置了limit参数，随机选择指定数量的文件
        if limit and limit < len(common_files):
            common_files = random.sample(common_files, limit)
            logger.info(f"随机选择 {limit} 对图片进行处理")
        
        logger.info(f"找到 {len(common_files)} 对匹配的图片")
        
        processed_count = 0
        success_count = 0
        
        for filename in common_files:
            clean_path = clean_files[filename]
            watermarked_path = watermarked_files[filename]
            
            logger.info(f"处理: {filename}")
            
            result = self.extract_watermark_from_pair(clean_path, watermarked_path)
            
            if result:
                success_count += 1
                if isinstance(result, list):
                    logger.info(f"成功提取 {len(result)} 个水印部分")
                else:
                    logger.info(f"成功提取水印")
            
            processed_count += 1
            
            if processed_count % 10 == 0:
                logger.info(f"已处理 {processed_count}/{len(common_files)} 个文件")
        
        logger.info(f"处理完成: {processed_count} 个文件，成功提取 {success_count} 个水印")

def main():
    parser = argparse.ArgumentParser(description='提取水印素材')
    parser.add_argument('--clean_dir', default='data/train/clean', help='原图文件夹路径')
    parser.add_argument('--watermarked_dir', default='data/train/watermarked', help='水印图文件夹路径')
    parser.add_argument('--output_dir', default='data/extracted_watermarks', help='输出文件夹路径')
    parser.add_argument('--threshold', type=int, default=30, help='差异阈值')
    parser.add_argument('--min_area', type=int, default=100, help='最小水印区域面积')
    parser.add_argument('--limit', type=int, default=None, help='随机处理的图片数量限制')
    
    args = parser.parse_args()
    
    extractor = WatermarkExtractor(
        clean_dir=args.clean_dir,
        watermarked_dir=args.watermarked_dir,
        output_dir=args.output_dir,
        threshold=args.threshold,
        min_area=args.min_area
    )
    
    extractor.process_all_images(limit=args.limit)

if __name__ == '__main__':
    main()