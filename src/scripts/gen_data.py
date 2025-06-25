# -*- coding: utf-8 -*-
"""
数据生成模块
用于生成带水印的训练数据和对应的掩码
"""

import os
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
from tqdm import tqdm
import argparse
import hashlib
from typing import List, Tuple, Optional

def load_watermarks(logos_dir):
    """加载所有水印图片"""
    watermarks = []
    
    # 加载combined目录下的水印
    combined_dir = os.path.join(logos_dir, 'combined')
    if os.path.exists(combined_dir):
        for file in os.listdir(combined_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                watermarks.append(os.path.join(combined_dir, file))
    
    # 加载independent目录下的水印
    independent_dir = os.path.join(logos_dir, 'independent')
    if os.path.exists(independent_dir):
        for file in os.listdir(independent_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                watermarks.append(os.path.join(independent_dir, file))
    
      # 加载independent目录下的水印
    independent_dir = os.path.join(logos_dir, 'car_logo')
    if os.path.exists(independent_dir):
        for file in os.listdir(independent_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                watermarks.append(os.path.join(independent_dir, file))

    return watermarks

def load_clean_images(clean_dir):
    """加载所有干净图片路径"""
    clean_images = []
    for file in os.listdir(clean_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            clean_images.append(os.path.join(clean_dir, file))
    return clean_images

def resize_watermark(watermark, target_size, min_scale=0.05, max_scale=0.3):
    """调整水印大小"""
    # 随机缩放比例
    scale = random.uniform(min_scale, max_scale)
    
    # 计算新尺寸
    new_width = int(target_size[0] * scale)
    new_height = int(watermark.height * new_width / watermark.width)
    
    # 确保水印不会太大
    if new_height > target_size[1] * max_scale:
        new_height = int(target_size[1] * max_scale)
        new_width = int(watermark.width * new_height / watermark.height)
    
    return watermark.resize((new_width, new_height), Image.Resampling.LANCZOS)

def apply_watermark_effects(watermark, enhance_transparent=True, target_size=None):
    """对水印应用随机效果，包括大小变换、形变、任意角度旋转、模糊和残缺效果"""
    # 1. 大小变换：调整为原图的3%-35%（扩大范围）
    if target_size is not None:
        scale = random.uniform(0.03, 0.35)
        new_width = int(target_size[0] * scale)
        new_height = int(watermark.height * new_width / watermark.width)
        
        # 确保高度不超过限制
        if new_height > target_size[1] * 0.35:
            new_height = int(target_size[1] * 0.35)
            new_width = int(watermark.width * new_height / watermark.height)
        
        watermark = watermark.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # 2. 任意角度旋转（0-360度）
    if random.random() < 0.8:  # 80%概率应用旋转
        angle = random.uniform(0, 360)
        watermark = watermark.rotate(angle, expand=True, fillcolor=(0, 0, 0, 0))
    
    # 3. 形变效果：更强的拉升变形
    if random.random() < 0.75:  # 75%概率应用形变
        # 水平拉升（更大范围）
        h_stretch = random.uniform(0.6, 1.6)
        # 垂直拉升（更大范围）
        v_stretch = random.uniform(0.6, 1.6)
        
        new_w = int(watermark.width * h_stretch)
        new_h = int(watermark.height * v_stretch)
        watermark = watermark.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # 随机倾斜变换（增强效果）
        if random.random() < 0.6:
            watermark_array = np.array(watermark)
            h, w = watermark_array.shape[:2]
            
            # 创建更强的倾斜变换矩阵
            shear_x = random.uniform(-0.4, 0.4)
            shear_y = random.uniform(-0.4, 0.4)
            
            M = np.array([[1, shear_x, 0],
                         [shear_y, 1, 0]], dtype=np.float32)
            
            watermark_array = cv2.warpAffine(watermark_array, M, (w, h), 
                                           borderMode=cv2.BORDER_CONSTANT, 
                                           borderValue=(0, 0, 0, 0))
            watermark = Image.fromarray(watermark_array)
    
    # 4. 模糊效果
    if random.random() < 0.4:  # 40%概率应用模糊
        blur_radius = random.uniform(0.5, 2.5)
        watermark = watermark.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    # 5. 残缺效果（随机擦除部分区域）
    if random.random() < 0.3:  # 30%概率应用残缺
        watermark_array = np.array(watermark)
        h, w = watermark_array.shape[:2]
        
        # 随机生成1-3个残缺区域
        num_defects = random.randint(1, 3)
        for _ in range(num_defects):
            # 残缺区域大小（5%-20%的水印面积）
            defect_w = random.randint(int(w * 0.05), int(w * 0.2))
            defect_h = random.randint(int(h * 0.05), int(h * 0.2))
            
            # 随机位置
            x = random.randint(0, max(0, w - defect_w))
            y = random.randint(0, max(0, h - defect_h))
            
            # 将该区域设为透明
            if watermark_array.shape[2] == 4:  # RGBA
                watermark_array[y:y+defect_h, x:x+defect_w, 3] = 0
            else:  # RGB，转换为RGBA
                rgba_array = np.zeros((h, w, 4), dtype=np.uint8)
                rgba_array[:, :, :3] = watermark_array
                rgba_array[:, :, 3] = 255
                rgba_array[y:y+defect_h, x:x+defect_w, 3] = 0
                watermark_array = rgba_array
        
        watermark = Image.fromarray(watermark_array)
    
    # 6. 透明度设置
    if enhance_transparent:
        # 为透明水印使用更低的透明度范围，增加难度
        alpha = random.uniform(0.08, 0.45)  # 进一步降低透明度
    else:
        # 普通水印透明度
        alpha = random.uniform(0.25, 0.85)
    
    # 7. 颜色效果调整（更激进）
    # 更激进的亮度调整
    brightness = random.uniform(0.4, 1.8)
    enhancer = ImageEnhance.Brightness(watermark)
    watermark = enhancer.enhance(brightness)
    
    # 更大范围的对比度调整
    contrast = random.uniform(0.5, 1.6)
    enhancer = ImageEnhance.Contrast(watermark)
    watermark = enhancer.enhance(contrast)
    
    # 色彩饱和度调整
    saturation = random.uniform(0.5, 1.5)
    enhancer = ImageEnhance.Color(watermark)
    watermark = enhancer.enhance(saturation)
    
    # 8. 应用透明度
    if watermark.mode != 'RGBA':
        watermark = watermark.convert('RGBA')
    
    # 调整alpha通道
    watermark_array = np.array(watermark)
    watermark_array[:, :, 3] = (watermark_array[:, :, 3] * alpha).astype(np.uint8)
    
    return Image.fromarray(watermark_array)

def generate_multiple_watermarks_image(clean_image_path, watermark_paths, enhance_transparent=True, max_watermarks=3):
    """在同一底图上生成多个水印的图片和对应的mask"""
    # 加载底图
    clean_img = Image.open(clean_image_path).convert('RGB')
    
    # 随机决定水印数量（1-max_watermarks个）
    num_watermarks = random.randint(1, max_watermarks)
    
    # 如果只有一个水印路径，重复使用
    if len(watermark_paths) == 1:
        selected_watermarks = watermark_paths * num_watermarks
    else:
        # 随机选择不同的水印
        selected_watermarks = random.sample(watermark_paths, min(num_watermarks, len(watermark_paths)))
        # 如果需要的水印数量大于可用水印数量，随机补充
        while len(selected_watermarks) < num_watermarks:
            selected_watermarks.append(random.choice(watermark_paths))
    
    # 创建总的mask
    combined_mask = Image.new('L', clean_img.size, 0)
    watermarked_img = clean_img.copy()
    
    # 记录已放置的水印位置，避免重叠过多
    placed_regions = []
    
    for i, watermark_path in enumerate(selected_watermarks):
        try:
            # 加载水印
            watermark = Image.open(watermark_path).convert('RGBA')
            
            # 应用水印效果
            watermark_processed = apply_watermark_effects(watermark, enhance_transparent, target_size=clean_img.size)
            
            # 尝试找到合适的位置（避免过度重叠）
            max_attempts = 20
            placed = False
            
            for attempt in range(max_attempts):
                # 计算可放置的范围
                max_x = clean_img.width - watermark_processed.width
                max_y = clean_img.height - watermark_processed.height
                
                if max_x <= 0 or max_y <= 0:
                    # 水印太大，重新调整
                    watermark_processed = resize_watermark(watermark, clean_img.size, 0.02, 0.12)
                    max_x = clean_img.width - watermark_processed.width
                    max_y = clean_img.height - watermark_processed.height
                
                if max_x <= 0 or max_y <= 0:
                    break  # 无法放置，跳过这个水印
                
                pos_x = random.randint(0, max_x)
                pos_y = random.randint(0, max_y)
                
                # 检查与已放置水印的重叠程度
                current_region = (pos_x, pos_y, pos_x + watermark_processed.width, pos_y + watermark_processed.height)
                
                # 计算重叠面积
                overlap_ratio = 0
                for placed_region in placed_regions:
                    overlap_area = calculate_overlap_area(current_region, placed_region)
                    current_area = watermark_processed.width * watermark_processed.height
                    if current_area > 0:
                        overlap_ratio = max(overlap_ratio, overlap_area / current_area)
                
                # 如果重叠不超过30%或者是第一个水印，则放置
                if overlap_ratio < 0.3 or len(placed_regions) == 0:
                    # 创建当前水印的mask
                    watermark_mask = Image.new('L', watermark_processed.size, 255)
                    if watermark_processed.mode == 'RGBA':
                        alpha_channel = watermark_processed.split()[3]
                        watermark_mask = alpha_channel
                    
                    # 将mask添加到总mask中
                    combined_mask.paste(watermark_mask, (pos_x, pos_y), watermark_mask)
                    
                    # 合成水印到图片
                    watermarked_img.paste(watermark_processed, (pos_x, pos_y), watermark_processed)
                    
                    # 记录放置的区域
                    placed_regions.append(current_region)
                    placed = True
                    break
            
            if not placed and len(placed_regions) == 0:
                # 如果是第一个水印且无法放置，强制放置
                pos_x = random.randint(0, max(0, max_x))
                pos_y = random.randint(0, max(0, max_y))
                
                watermark_mask = Image.new('L', watermark_processed.size, 255)
                if watermark_processed.mode == 'RGBA':
                    alpha_channel = watermark_processed.split()[3]
                    watermark_mask = alpha_channel
                
                combined_mask.paste(watermark_mask, (pos_x, pos_y), watermark_mask)
                watermarked_img.paste(watermark_processed, (pos_x, pos_y), watermark_processed)
                
        except Exception as e:
            print(f"处理水印 {watermark_path} 时出错: {e}")
            continue
    
    return watermarked_img, combined_mask

def calculate_overlap_area(rect1, rect2):
    """计算两个矩形的重叠面积"""
    x1_max = max(rect1[0], rect2[0])
    y1_max = max(rect1[1], rect2[1])
    x2_min = min(rect1[2], rect2[2])
    y2_min = min(rect1[3], rect2[3])
    
    if x2_min > x1_max and y2_min > y1_max:
        return (x2_min - x1_max) * (y2_min - y1_max)
    return 0

def generate_watermarked_image(clean_image_path, watermark_path, enhance_transparent=True):
    """生成带水印的图片和对应的mask（兼容原接口）"""
    return generate_multiple_watermarks_image(clean_image_path, [watermark_path], enhance_transparent, max_watermarks=1)

def generate_filename(clean_path, watermark_path, index):
    """生成唯一的文件名"""
    clean_name = os.path.splitext(os.path.basename(clean_path))[0]
    watermark_name = os.path.splitext(os.path.basename(watermark_path))[0]
    
    # 创建唯一标识
    unique_str = f"{clean_name}_{watermark_name}_{index}"
    hash_obj = hashlib.md5(unique_str.encode())
    return hash_obj.hexdigest()[:16]

def main():
    parser = argparse.ArgumentParser(description='生成带水印的训练数据')
    parser.add_argument('--clean_dir', default='data/train/clean', help='干净图片目录')
    parser.add_argument('--logos_dir', default='data/WatermarkDataset/logos', help='水印图片目录')
    parser.add_argument('--output_dir', default='data/train', help='输出目录')
    parser.add_argument('--target_count', type=int, default=10000, help='目标图片数量')
    parser.add_argument('--transparent_ratio', type=float, default=0.6, help='透明水印样本比例')
    parser.add_argument('--generate_mask', action='store_true', help='是否生成mask图（默认不生成）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--max_watermarks', type=int, default=3, help='每张图片最大水印数量')
    parser.add_argument('--multi_watermark_ratio', type=float, default=0.4, help='多水印样本比例')
    parser.add_argument('--enable_advanced_effects', action='store_true', default=True, help='启用高级效果（任意角度旋转、模糊、残缺等）')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 创建输出目录
    watermarked_dir = os.path.join(args.output_dir, 'watermarked')
    os.makedirs(watermarked_dir, exist_ok=True)
    
    if args.generate_mask:
        masks_dir = os.path.join(args.output_dir, 'masks')
        os.makedirs(masks_dir, exist_ok=True)
    
    # 加载图片路径
    clean_images = load_clean_images(args.clean_dir)
    watermarks = load_watermarks(args.logos_dir)
    
    print(f"找到 {len(clean_images)} 张干净图片")
    print(f"找到 {len(watermarks)} 张水印图片")
    
    if len(clean_images) == 0 or len(watermarks) == 0:
        print("错误：没有找到图片文件")
        return
    
    # 计算需要生成的图片数量（减去已有的图片）
    existing_watermarked = len([f for f in os.listdir(watermarked_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if args.generate_mask:
        existing_masks = len([f for f in os.listdir(masks_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        remaining_count = args.target_count - min(existing_watermarked, existing_masks)
        
        if remaining_count <= 0:
            print(f"已经有足够的图片（{min(existing_watermarked, existing_masks)}张），无需生成更多")
            return
    else:
        remaining_count = args.target_count - existing_watermarked
        
        if remaining_count <= 0:
            print(f"已经有足够的图片（{existing_watermarked}张），无需生成更多")
            return
    
    print(f"需要生成 {remaining_count} 张新图片")
    
    # 生成图片
    generated_count = 0
    transparent_count = 0
    multi_watermark_count = 0
    target_transparent = int(remaining_count * args.transparent_ratio)
    
    pbar = tqdm(total=remaining_count, desc="生成带水印图片")
    
    while generated_count < remaining_count:
        # 决定是否生成透明水印
        should_generate_transparent = (transparent_count < target_transparent) or \
                                    (random.random() < args.transparent_ratio)
        
        # 随机选择干净图片和水印
        clean_path = random.choice(clean_images)
        watermark_path = random.choice(watermarks)
        
        try:
            # 决定是否生成多水印图片
            should_generate_multi = random.random() < args.multi_watermark_ratio
            
            if should_generate_multi and len(watermarks) > 1:
                # 生成多水印图片
                # 随机选择2-max_watermarks个不同的水印
                num_watermarks = random.randint(2, args.max_watermarks)
                selected_watermarks = random.sample(watermarks, min(num_watermarks, len(watermarks)))
                
                watermarked_img, mask = generate_multiple_watermarks_image(
                    clean_path, selected_watermarks,
                    enhance_transparent=should_generate_transparent,
                    max_watermarks=args.max_watermarks
                )
                
                # 生成文件名（多水印）
                prefix = "multi_trans_" if should_generate_transparent else "multi_norm_"
                watermark_names = "_".join([os.path.splitext(os.path.basename(w))[0][:8] for w in selected_watermarks[:2]])
                clean_name = os.path.splitext(os.path.basename(clean_path))[0]
                unique_str = f"{clean_name}_{watermark_names}_{generated_count}"
                hash_obj = hashlib.md5(unique_str.encode())
                filename = prefix + hash_obj.hexdigest()[:16]
                
            else:
                # 生成单水印图片
                watermarked_img, mask = generate_watermarked_image(
                    clean_path, watermark_path, 
                    enhance_transparent=should_generate_transparent
                )
                
                # 生成文件名（单水印）
                prefix = "trans_" if should_generate_transparent else "norm_"
                filename = prefix + generate_filename(clean_path, watermark_path, generated_count)
            
            # 保存图片
            watermarked_img.save(os.path.join(watermarked_dir, f"{filename}.jpg"), quality=95)
            
            # 只在需要时保存mask
            if args.generate_mask:
                mask.save(os.path.join(masks_dir, f"{filename}.png"))
            
            if should_generate_transparent:
                transparent_count += 1
            
            if should_generate_multi and len(watermarks) > 1:
                multi_watermark_count += 1
            
            generated_count += 1
            pbar.update(1)
            
        except Exception as e:
            print(f"\n生成图片时出错: {e}")
            continue
    
    pbar.close()
    print(f"\n成功生成 {generated_count} 张带水印图片")
    print(f"其中透明水印: {transparent_count} 张 ({transparent_count/generated_count*100:.1f}%)")
    print(f"其中多水印图片: {multi_watermark_count} 张 ({multi_watermark_count/generated_count*100:.1f}%)")
    print(f"单水印图片: {generated_count - multi_watermark_count} 张 ({(generated_count - multi_watermark_count)/generated_count*100:.1f}%)")
    print(f"\n优化特性:")
    print(f"- 支持任意角度旋转 (0-360度)")
    print(f"- 支持水印形变和倾斜")
    print(f"- 支持模糊效果 (40%概率)")
    print(f"- 支持残缺效果 (30%概率)")
    print(f"- 支持多水印叠加 (最多{args.max_watermarks}个)")
    print(f"\n带水印图片保存在: {watermarked_dir}")
    
    if args.generate_mask:
        print(f"Mask图片保存在: {masks_dir}")
    else:
        print("未生成mask图片（使用 --generate_mask 参数启用）")

if __name__ == '__main__':
    main()