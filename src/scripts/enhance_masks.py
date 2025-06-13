# -*- coding: utf-8 -*-
"""
Mask增强工具
用于对分割mask进行边框圆滑模糊放大处理
"""

import cv2
import numpy as np
import os
from pathlib import Path
from scipy import ndimage
from tqdm import tqdm
from shutil import copy
from typing import Optional

def enhance_mask(mask_path: str, output_path: str, expand_pixels: int = 10, 
                blur_kernel: int = 15, smooth_iterations: int = 2) -> None:
    """
    对mask进行边框圆滑模糊放大处理
    
    Args:
        mask_path: 输入mask路径
        output_path: 输出mask路径
        expand_pixels: 膨胀像素数，用于放大mask区域
        blur_kernel: 高斯模糊核大小
        smooth_iterations: 形态学平滑迭代次数
    """
    # 读取mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"无法读取mask文件: {mask_path}")
        return
    
    # 1. 形态学操作 - 先闭运算填补小洞
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    
    # 2. 膨胀操作 - 放大mask区域
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand_pixels*2+1, expand_pixels*2+1))
    mask_expanded = cv2.dilate(mask, kernel_dilate, iterations=1)
    
    # 3. 高斯模糊 - 使边缘平滑
    if blur_kernel > 0:
        # 确保核大小为奇数
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        mask_blurred = cv2.GaussianBlur(mask_expanded.astype(np.float32), (blur_kernel, blur_kernel), 0)
    else:
        mask_blurred = mask_expanded.astype(np.float32)
    
    # 4. 形态学平滑 - 进一步平滑边缘
    kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for _ in range(smooth_iterations):
        mask_blurred = cv2.morphologyEx(mask_blurred, cv2.MORPH_OPEN, kernel_smooth)
        mask_blurred = cv2.morphologyEx(mask_blurred, cv2.MORPH_CLOSE, kernel_smooth)
    
    # 5. 二值化处理 - 保持mask的二值特性
    _, mask_final = cv2.threshold(mask_blurred, 127, 255, cv2.THRESH_BINARY)
    
    # 6. 边缘平滑处理 - 使用双边滤波进一步平滑
    mask_final = cv2.bilateralFilter(mask_final.astype(np.uint8), 9, 75, 75)
    
    # 最终二值化
    _, mask_final = cv2.threshold(mask_final, 127, 255, cv2.THRESH_BINARY)
    
    # 保存处理后的mask
    cv2.imwrite(output_path, mask_final)

def convert_yolo_to_enhanced_mask(image_path, label_path, output_path, expand_pixels=10, blur_kernel=15):
    """
    从YOLO标注直接生成增强的mask
    """
    # 读取图片获取尺寸
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return
    
    h, w = img.shape[:2]
    
    # 创建空白掩码
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # 读取YOLO标注
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    # YOLO格式: class_id center_x center_y width height
                    center_x, center_y, width, height = map(float, parts[1:5])
                    
                    # 转换为像素坐标，并适当扩大边界框
                    expand_ratio = 0.1  # 扩大10%
                    width_expanded = width * (1 + expand_ratio)
                    height_expanded = height * (1 + expand_ratio)
                    
                    x1 = int((center_x - width_expanded/2) * w)
                    y1 = int((center_y - height_expanded/2) * h)
                    x2 = int((center_x + width_expanded/2) * w)
                    y2 = int((center_y + height_expanded/2) * h)
                    
                    # 确保坐标在图像范围内
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)
                    
                    # 绘制椭圆而不是矩形，使边缘更平滑
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    axes = ((x2 - x1) // 2, (y2 - y1) // 2)
                    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    
    # 应用增强处理
    # 1. 形态学闭运算
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    
    # 2. 膨胀操作
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand_pixels*2+1, expand_pixels*2+1))
    mask = cv2.dilate(mask, kernel_dilate, iterations=1)
    
    # 3. 高斯模糊
    if blur_kernel > 0:
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        mask = cv2.GaussianBlur(mask.astype(np.float32), (blur_kernel, blur_kernel), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # 4. 双边滤波平滑
    mask = cv2.bilateralFilter(mask.astype(np.uint8), 9, 75, 75)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # 保存掩码
    cv2.imwrite(output_path, mask)

def process_existing_masks(mask_dir, output_dir, expand_pixels=10, blur_kernel=15):
    """
    处理已存在的mask文件
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有mask文件
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for mask_file in tqdm(mask_files, desc="处理mask文件", unit="个"):
        mask_path = os.path.join(mask_dir, mask_file)
        output_path = os.path.join(output_dir, mask_file)
        enhance_mask(mask_path, output_path, expand_pixels, blur_kernel)

def main():
    # 配置参数
    expand_pixels = 15  # 膨胀像素数
    blur_kernel = 21   # 高斯模糊核大小
    
    # 方法1: 处理已存在的mask文件
    print("处理训练集mask...")
    train_mask_dir = "data/train/masks"
    train_enhanced_dir = "data/train/masks"
    if os.path.exists(train_mask_dir):
        process_existing_masks(train_mask_dir, train_enhanced_dir, expand_pixels, blur_kernel)
    
    print("处理验证集mask...")
    val_mask_dir = "data/val/masks"
    val_enhanced_dir = "data/val/masks"
    if os.path.exists(val_mask_dir):
        process_existing_masks(val_mask_dir, val_enhanced_dir, expand_pixels, blur_kernel)
    
    # 方法2: 从YOLO标注直接生成增强mask
    print("从YOLO标注生成增强mask...")
    
    # 训练集
    watermark_img_dir = "data/train/watermarked"
    train_img_dir = "data/WatermarkDataset/images/train"
    train_label_dir = "data/WatermarkDataset/labels/train"
    train_enhanced_dir = "data/train/masks"
    os.makedirs(train_enhanced_dir, exist_ok=True)
    os.makedirs(watermark_img_dir, exist_ok=True)
    
    if os.path.exists(train_img_dir) and os.path.exists(train_label_dir):
        # 获取所有图片文件
        train_img_files = [f for f in os.listdir(train_img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        for img_file in tqdm(train_img_files, desc="训练集YOLO转换", unit="张"):
            img_path = os.path.join(train_img_dir, img_file)
            label_path = os.path.join(train_label_dir, img_file.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt'))
            mask_path = os.path.join(train_enhanced_dir, img_file.replace('.jpg', '.png').replace('.jpeg', '.png'))
            convert_yolo_to_enhanced_mask(img_path, label_path, mask_path, expand_pixels, blur_kernel)
            copy(Path(train_img_dir) / img_file, Path(watermark_img_dir) / img_file)
    
    # 验证集
    val_img_dir = "data/WatermarkDataset/images/val"
    val_label_dir = "data/WatermarkDataset/labels/val"
    val_enhanced_dir = "data/train/masks"
    os.makedirs(val_enhanced_dir, exist_ok=True)
    
    if os.path.exists(val_img_dir) and os.path.exists(val_label_dir):
        # 获取所有图片文件
        val_img_files = [f for f in os.listdir(val_img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        for img_file in tqdm(val_img_files, desc="验证集YOLO转换", unit="张"):
            img_path = os.path.join(val_img_dir, img_file)
            label_path = os.path.join(val_label_dir, img_file.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt'))
            mask_path = os.path.join(val_enhanced_dir, img_file.replace('.jpg', '.png').replace('.jpeg', '.png'))
            convert_yolo_to_enhanced_mask(img_path, label_path, mask_path, expand_pixels, blur_kernel)
    
    print("\n所有mask增强处理完成！")
    print(f"训练集处理完成，共 {len(os.listdir(train_enhanced_dir)) if os.path.exists(train_enhanced_dir) else 0} 个文件")
    print(f"验证集处理完成，共 {len(os.listdir(val_enhanced_dir)) if os.path.exists(val_enhanced_dir) else 0} 个文件")

if __name__ == "__main__":
    main()