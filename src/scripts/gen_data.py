import os
import random
import numpy as np
from PIL import Image, ImageEnhance
import cv2
from tqdm import tqdm
import argparse
import hashlib

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

def apply_watermark_effects(watermark, enhance_transparent=True):
    """对水印应用随机效果，特别针对透明水印优化"""
    if enhance_transparent:
        # 为透明水印使用更低的透明度范围，增加难度
        alpha = random.uniform(0.1, 0.5)  # 降低最小透明度
    else:
        # 普通水印透明度
        alpha = random.uniform(0.3, 0.9)
    
    # 更激进的亮度调整，模拟不同光照条件
    brightness = random.uniform(0.5, 1.5)  # 扩大范围
    enhancer = ImageEnhance.Brightness(watermark)
    watermark = enhancer.enhance(brightness)
    
    # 更大范围的对比度调整
    contrast = random.uniform(0.6, 1.4)  # 扩大范围
    enhancer = ImageEnhance.Contrast(watermark)
    watermark = enhancer.enhance(contrast)
    
    # 添加色彩饱和度调整
    saturation = random.uniform(0.7, 1.3)
    enhancer = ImageEnhance.Color(watermark)
    watermark = enhancer.enhance(saturation)
    
    # 应用透明度
    if watermark.mode != 'RGBA':
        watermark = watermark.convert('RGBA')
    
    # 调整alpha通道
    watermark_array = np.array(watermark)
    watermark_array[:, :, 3] = (watermark_array[:, :, 3] * alpha).astype(np.uint8)
    
    return Image.fromarray(watermark_array)

def generate_watermarked_image(clean_image_path, watermark_path, enhance_transparent=True):
    """生成带水印的图片和对应的mask"""
    # 加载图片
    clean_img = Image.open(clean_image_path).convert('RGB')
    watermark = Image.open(watermark_path).convert('RGBA')
    
    # 调整水印大小
    watermark_resized = resize_watermark(watermark, clean_img.size)
    
    # 应用水印效果，传递 enhance_transparent 参数
    watermark_resized = apply_watermark_effects(watermark_resized, enhance_transparent)
    
    # 随机位置放置水印
    max_x = clean_img.width - watermark_resized.width
    max_y = clean_img.height - watermark_resized.height
    
    if max_x <= 0 or max_y <= 0:
        # 如果水印太大，重新调整
        watermark_resized = resize_watermark(watermark, clean_img.size, 0.02, 0.15)
        max_x = clean_img.width - watermark_resized.width
        max_y = clean_img.height - watermark_resized.height
    
    pos_x = random.randint(0, max(0, max_x))
    pos_y = random.randint(0, max(0, max_y))
    
    # 创建mask
    mask = Image.new('L', clean_img.size, 0)
    watermark_mask = Image.new('L', watermark_resized.size, 255)
    
    # 如果水印有透明通道，使用透明通道作为mask
    if watermark_resized.mode == 'RGBA':
        alpha_channel = watermark_resized.split()[3]
        watermark_mask = alpha_channel
    
    mask.paste(watermark_mask, (pos_x, pos_y))
    
    # 合成带水印的图片
    watermarked_img = clean_img.copy()
    watermarked_img.paste(watermark_resized, (pos_x, pos_y), watermark_resized)
    
    return watermarked_img, mask

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
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 创建输出目录
    watermarked_dir = os.path.join(args.output_dir, 'watermarked')
    masks_dir = os.path.join(args.output_dir, 'masks')
    os.makedirs(watermarked_dir, exist_ok=True)
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
    existing_masks = len([f for f in os.listdir(masks_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    remaining_count = args.target_count - min(existing_watermarked, existing_masks)
    
    if remaining_count <= 0:
        print(f"已经有足够的图片（{min(existing_watermarked, existing_masks)}张），无需生成更多")
        return
    
    print(f"需要生成 {remaining_count} 张新图片")
    
    # 生成图片
    generated_count = 0
    transparent_count = 0
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
            # 生成带水印图片和mask
            watermarked_img, mask = generate_watermarked_image(
                clean_path, watermark_path, 
                enhance_transparent=should_generate_transparent
            )
            
            # 生成文件名
            prefix = "trans_" if should_generate_transparent else "norm_"
            filename = prefix + generate_filename(clean_path, watermark_path, generated_count)
            
            # 保存图片
            watermarked_img.save(os.path.join(watermarked_dir, f"{filename}.jpg"), quality=95)
            mask.save(os.path.join(masks_dir, f"{filename}.png"))
            
            if should_generate_transparent:
                transparent_count += 1
            
            generated_count += 1
            pbar.update(1)
            
        except Exception as e:
            print(f"\n生成图片时出错: {e}")
            continue
    
    pbar.close()
    print(f"\n成功生成 {generated_count} 张带水印图片")
    print(f"其中透明水印: {transparent_count} 张 ({transparent_count/generated_count*100:.1f}%)")
    print(f"带水印图片保存在: {watermarked_dir}")
    print(f"Mask图片保存在: {masks_dir}")

if __name__ == '__main__':
    main()