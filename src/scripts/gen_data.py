# -*- coding: utf-8 -*-
"""
数据生成模块
用于生成带水印的训练数据和对应的掩码
"""

import os
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont
import cv2
from tqdm import tqdm
import argparse
import hashlib
from typing import List, Tuple, Optional
import string
import sys

# 添加OCR模块路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ocr.easy_ocr import EasyOCRDetector

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

def load_system_fonts():
    """加载系统字体，优先支持中文的字体"""
    fonts = []
    
    # 优先使用支持中文的字体
    chinese_fonts = [
        '/System/Library/Fonts/STHeiti Light.ttc',  # 黑体，支持中文
        '/System/Library/Fonts/Hiragino Sans GB.ttc',  # 冬青黑体，支持中文
        '/System/Library/Fonts/PingFang.ttc',  # 苹方，支持中文
        '/System/Library/Fonts/Helvetica.ttc',  # 支持部分中文
        '/System/Library/Fonts/Arial Unicode MS.ttf',  # Unicode字体
    ]
    
    # 英文字体（作为备选）
    english_fonts = [
        '/System/Library/Fonts/Arial.ttf',
        '/System/Library/Fonts/Times.ttc',
        '/System/Library/Fonts/Courier.ttc',
        '/System/Library/Fonts/Georgia.ttf',
        '/System/Library/Fonts/Verdana.ttf',
        '/System/Library/Fonts/Impact.ttf',
    ]
    
    # 首先添加中文字体
    for font_path in chinese_fonts:
        if os.path.exists(font_path):
            fonts.append(font_path)
    
    # 然后添加英文字体
    for font_path in english_fonts:
        if os.path.exists(font_path):
            fonts.append(font_path)
    
    # 扫描字体目录
    font_dirs = ['/System/Library/Fonts/', '/Library/Fonts/']
    additional_fonts = ['STSong.ttc', 'STFangsong.ttc', 'STKaiti.ttc']  # 更多中文字体
    
    for font_dir in font_dirs:
        if os.path.exists(font_dir):
            for font_file in additional_fonts:
                font_path = os.path.join(font_dir, font_file)
                if os.path.exists(font_path) and font_path not in fonts:
                    fonts.append(font_path)
    
    # 如果没有找到字体，使用默认字体
    if not fonts:
        fonts = [None]  # PIL会使用默认字体
    
    return fonts

def test_font_text_compatibility(font_path, text):
    """测试字体是否能正确渲染文字"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        
        # 创建测试图像
        test_img = Image.new('RGB', (200, 100), color='white')
        draw = ImageDraw.Draw(test_img)
        
        # 加载字体
        if font_path:
            font = ImageFont.truetype(font_path, 24)
        else:
            font = ImageFont.load_default()
        
        # 绘制文字
        draw.text((10, 30), text, fill='black', font=font)
        
        # 检查是否有实际内容
        img_array = np.array(test_img)
        has_content = np.any(img_array < 255)
        
        if not has_content:
            return False
        
        # 检查是否是方框（简单检测）
        gray = np.mean(img_array, axis=2)
        variance = np.var(gray)
        
        # 如果方差太小，可能是方框
        return variance > 50
        
    except Exception:
        return False

def select_compatible_font(fonts, text):
    """选择与文字兼容的字体"""
    # 检测文字类型
    has_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)
    has_english = any(char.isalpha() and ord(char) < 128 for char in text)
    
    # 如果包含中文，优先选择中文字体
    if has_chinese:
        chinese_fonts = [
            '/System/Library/Fonts/STHeiti Light.ttc',
            '/System/Library/Fonts/Hiragino Sans GB.ttc',
            '/System/Library/Fonts/PingFang.ttc',
        ]
        
        for font_path in chinese_fonts:
            if font_path in fonts and test_font_text_compatibility(font_path, text):
                return font_path
    
    # 测试所有可用字体
    for font_path in fonts:
        if test_font_text_compatibility(font_path, text):
            return font_path
    
    # 如果都不兼容，返回第一个字体
    return fonts[0] if fonts else None

def generate_text_content():
    """生成随机文字内容，增加多样性"""
    text_types = [
        # 英文文本 - 扩展内容
        ['SAMPLE', 'DEMO', 'PREVIEW', 'WATERMARK', 'COPYRIGHT', 'DRAFT', 'CONFIDENTIAL', 'PROTOTYPE', 'BETA', 'ALPHA'],
        ['Sample Text', 'Demo Version', 'Preview Only', 'Not for Sale', 'Copyright Protected', 'Test Version', 'Internal Use'],
        ['www.example.com', 'example.com', 'demo.site.com', 'preview.example.org', 'test.domain.net'],
        ['PROPERTY OF', 'RESTRICTED', 'EVALUATION COPY', 'FOR REVIEW ONLY', 'TRIAL VERSION'],
        
        # 中文文本 - 扩展内容
        ['样本', '演示', '预览', '水印', '版权', '草稿', '机密', '测试', '试用', '内部'],
        ['样本文字', '演示版本', '仅供预览', '版权保护', '请勿商用', '测试版本', '内部使用'],
        ['示例网站', '演示站点', '测试文本', '预览内容', '版权所有'],
        ['仅供参考', '禁止转载', '内部资料', '测试专用', '评估版本'],
        
        # 数字和符号
        ['2024', '©2024', '®', '™', '№123', 'V1.0', 'BETA', 'V2.1', '©2023-2024'],
        ['#001', '#DEMO', 'ID:12345', 'REF:ABC', 'CODE:XYZ'],
        
        # 混合文本 - 增加多样性
        ['Sample 样本', 'Demo 演示', 'Preview 预览', 'Copyright ©2024', 'Test 测试'],
        ['DEMO 演示版', 'SAMPLE 样本', 'BETA 测试版', 'TRIAL 试用版'],
        ['版权 ©2024', '测试 TEST', '样本 SAMPLE', '演示 DEMO']
    ]
    
    text_category = random.choice(text_types)
    return random.choice(text_category)

def apply_text_effects(text_img, enhance_transparent=True):
    """对文字图像应用各种效果"""
    # 1. 旋转效果
    if random.random() < 0.7:  # 70%概率应用旋转
        angle = random.uniform(-45, 45)  # 文字通常不需要360度旋转
        text_img = text_img.rotate(angle, expand=True, fillcolor=(0, 0, 0, 0))
    
    # 2. 缩放效果
    if random.random() < 0.6:
        scale_x = random.uniform(0.8, 1.4)
        scale_y = random.uniform(0.8, 1.4)
        new_w = int(text_img.width * scale_x)
        new_h = int(text_img.height * scale_y)
        text_img = text_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # 3. 模糊效果
    if random.random() < 0.3:  # 30%概率应用模糊
        blur_radius = random.uniform(0.5, 1.5)
        text_img = text_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    # 4. 透明度设置
    if enhance_transparent:
        alpha = random.uniform(0.1, 0.5)  # 文字水印通常更透明
    else:
        alpha = random.uniform(0.3, 0.8)
    
    # 5. 颜色效果调整
    if random.random() < 0.5:
        brightness = random.uniform(0.6, 1.4)
        enhancer = ImageEnhance.Brightness(text_img)
        text_img = enhancer.enhance(brightness)
    
    if random.random() < 0.5:
        contrast = random.uniform(0.7, 1.3)
        enhancer = ImageEnhance.Contrast(text_img)
        text_img = enhancer.enhance(contrast)
    
    # 应用透明度
    if text_img.mode != 'RGBA':
        text_img = text_img.convert('RGBA')
    
    text_array = np.array(text_img)
    text_array[:, :, 3] = (text_array[:, :, 3] * alpha).astype(np.uint8)
    
    return Image.fromarray(text_array)

def generate_text_watermark(clean_image_path, enhance_transparent=True, use_ocr_mask=True):
    """生成文字水印图片和对应的mask"""
    # 加载底图
    clean_img = Image.open(clean_image_path).convert('RGB')
    
    # 生成文字内容
    text_content = generate_text_content()
    
    # 加载字体
    fonts = load_system_fonts()
    
    # 随机字体大小（相对于图片尺寸）
    min_font_size = max(20, int(min(clean_img.size) * 0.03))
    max_font_size = max(60, int(min(clean_img.size) * 0.15))
    font_size = random.randint(min_font_size, max_font_size)
    
    # 选择与文字兼容的字体
    if fonts:
        compatible_font_path = select_compatible_font(fonts, text_content)
        if compatible_font_path:
            try:
                font = ImageFont.truetype(compatible_font_path, font_size)
            except:
                font = ImageFont.load_default()
        else:
            font = ImageFont.load_default()
    else:
        font = ImageFont.load_default()
    
    # 创建文字图像
    # 先创建一个临时图像来测量文字尺寸
    temp_img = Image.new('RGBA', (1, 1))
    temp_draw = ImageDraw.Draw(temp_img)
    
    try:
        bbox = temp_draw.textbbox((0, 0), text_content, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except Exception as e:
        # 如果textbbox失败，使用textlength作为备选方案
        try:
            text_width = temp_draw.textlength(text_content, font=font)
            # 估算文字高度（基于字体大小）
            text_height = font.size if hasattr(font, 'size') else 20
        except:
            # 最后的备选方案：使用固定尺寸
            text_width = len(text_content) * 15
            text_height = 20
    
    # 创建文字图像（留一些边距）
    margin = 20
    text_img = Image.new('RGBA', (text_width + margin*2, text_height + margin*2), (0, 0, 0, 0))
    text_draw = ImageDraw.Draw(text_img)
    
    # 随机文字颜色
    colors = [
        (255, 255, 255, 255),  # 白色
        (0, 0, 0, 255),        # 黑色
        (255, 0, 0, 255),      # 红色
        (0, 255, 0, 255),      # 绿色
        (0, 0, 255, 255),      # 蓝色
        (255, 255, 0, 255),    # 黄色
        (255, 0, 255, 255),    # 紫色
        (0, 255, 255, 255),    # 青色
        (128, 128, 128, 255),  # 灰色
    ]
    text_color = random.choice(colors)
    
    # 绘制文字
    text_draw.text((margin, margin), text_content, font=font, fill=text_color)
    
    # 应用文字效果
    text_img = apply_text_effects(text_img, enhance_transparent)
    
    # 确保文字不会超出图片边界
    if text_img.width > clean_img.width * 0.8:
        scale = (clean_img.width * 0.8) / text_img.width
        new_w = int(text_img.width * scale)
        new_h = int(text_img.height * scale)
        text_img = text_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    if text_img.height > clean_img.height * 0.8:
        scale = (clean_img.height * 0.8) / text_img.height
        new_w = int(text_img.width * scale)
        new_h = int(text_img.height * scale)
        text_img = text_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # 随机位置放置文字
    max_x = clean_img.width - text_img.width
    max_y = clean_img.height - text_img.height
    
    if max_x <= 0 or max_y <= 0:
        # 文字太大，缩小后重新计算
        scale = min(clean_img.width / text_img.width, clean_img.height / text_img.height) * 0.8
        new_w = int(text_img.width * scale)
        new_h = int(text_img.height * scale)
        text_img = text_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        max_x = clean_img.width - text_img.width
        max_y = clean_img.height - text_img.height
    
    pos_x = random.randint(0, max(0, max_x))
    pos_y = random.randint(0, max(0, max_y))
    
    # 合成图像
    watermarked_img = clean_img.copy()
    watermarked_img.paste(text_img, (pos_x, pos_y), text_img)
    
    # 生成mask
    if use_ocr_mask:
        # 使用OCR生成更精确的mask
        try:
            ocr_detector = EasyOCRDetector(verbose=False)  # 关闭详细输出
            # 直接传递PIL图像对象
            mask_array = ocr_detector.generate_text_mask(watermarked_img)
            # 转换为PIL图像
            if isinstance(mask_array, np.ndarray):
                mask = Image.fromarray(mask_array)
            else:
                mask = mask_array
        except Exception as e:
            print(f"OCR mask生成失败，使用传统方法: {e}")
            # 回退到传统mask生成方法
            mask = Image.new('L', clean_img.size, 0)
            text_mask = text_img.split()[3] if text_img.mode == 'RGBA' else Image.new('L', text_img.size, 255)
            mask.paste(text_mask, (pos_x, pos_y), text_mask)
    else:
        # 传统mask生成方法
        mask = Image.new('L', clean_img.size, 0)
        text_mask = text_img.split()[3] if text_img.mode == 'RGBA' else Image.new('L', text_img.size, 255)
        mask.paste(text_mask, (pos_x, pos_y), text_mask)
    
    return watermarked_img, mask

def generate_mixed_watermark(clean_image_path, watermark_paths, enhance_transparent=True, use_ocr_mask=True, max_watermarks=2):
    """生成混合水印（图像水印+文字水印）"""
    # 加载底图
    clean_img = Image.open(clean_image_path).convert('RGB')
    
    # 先添加图像水印
    if len(watermark_paths) > 0:
        # 随机选择1-2个图像水印
        num_image_watermarks = random.randint(1, min(max_watermarks, len(watermark_paths)))
        selected_watermarks = random.sample(watermark_paths, min(num_image_watermarks, len(watermark_paths)))
        
        watermarked_img, image_mask = generate_multiple_watermarks_image(
            clean_image_path, selected_watermarks,
            enhance_transparent=enhance_transparent,
            max_watermarks=num_image_watermarks
        )
    else:
        watermarked_img = clean_img.copy()
        image_mask = Image.new('L', clean_img.size, 0)
    
    # 再添加文字水印
    # 将当前图像保存为临时文件，然后生成文字水印
    import tempfile
    import uuid
    
    # 使用UUID生成安全的临时文件名，避免编码问题
    temp_filename = f"temp_mixed_{uuid.uuid4().hex[:8]}.jpg"
    temp_path = os.path.join(tempfile.gettempdir(), temp_filename)
    
    try:
        watermarked_img.save(temp_path, quality=95)
    except Exception as e:
        print(f"临时文件保存失败: {e}")
        # 如果保存失败，直接返回原图和空mask
        return watermarked_img, image_mask
    
    try:
        # 生成文字水印
        final_img, text_mask = generate_text_watermark(
            temp_path, enhance_transparent=enhance_transparent, use_ocr_mask=use_ocr_mask
        )
        
        # 合并mask
        combined_mask = Image.new('L', clean_img.size, 0)
        
        # 添加图像水印mask
        if image_mask:
            combined_mask = Image.blend(combined_mask.convert('RGB'), image_mask.convert('RGB'), 0.5).convert('L')
        
        # 添加文字水印mask
        if text_mask:
            # 使用最大值合并
            combined_array = np.maximum(np.array(combined_mask), np.array(text_mask))
            combined_mask = Image.fromarray(combined_array)
        
    finally:
        # 清理临时文件
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    return final_img, combined_mask

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
    # 安全处理文件名，避免中文字符编码问题
    try:
        clean_name = os.path.splitext(os.path.basename(clean_path))[0]
        watermark_name = os.path.splitext(os.path.basename(watermark_path))[0]
        unique_str = f"{clean_name}_{watermark_name}_{index}"
        # 确保字符串可以安全编码
        unique_str.encode('ascii')
    except UnicodeEncodeError:
        # 如果包含非ASCII字符，使用简化的文件名
        clean_name = "clean"
        watermark_name = "watermark"
        unique_str = f"{clean_name}_{watermark_name}_{index}"
    
    # 创建唯一标识
    hash_obj = hashlib.md5(unique_str.encode('utf-8'))
    return hash_obj.hexdigest()[:16]

def main():
    parser = argparse.ArgumentParser(description='生成带水印的训练数据')
    parser.add_argument('--clean_dir', default='data/train/clean', help='干净图片目录')
    parser.add_argument('--logos_dir', default='data/WatermarkDataset/logos', help='水印图片目录')
    parser.add_argument('--output_dir', default='data/train', help='输出目录')
    parser.add_argument('--count', type=int, default=10000, help='目标图片数量')
    parser.add_argument('--transparent_ratio', type=float, default=0.6, help='透明水印样本比例')
    parser.add_argument('--generate_mask', action='store_true', help='是否生成mask图（默认不生成）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--max_watermarks', type=int, default=3, help='每张图片最大水印数量')
    parser.add_argument('--multi_watermark_ratio', type=float, default=0.4, help='多水印样本比例')
    parser.add_argument('--enable_advanced_effects', action='store_true', default=True, help='启用高级效果（任意角度旋转、模糊、残缺等）')
    
    # 文字水印相关参数
    parser.add_argument('--text_watermark_ratio', type=float, default=0.3, help='文字水印样本比例')
    parser.add_argument('--use_ocr_mask', action='store_true', default=True, help='使用OCR生成更精确的文字区域mask')
    parser.add_argument('--mixed_watermark_ratio', type=float, default=0.2, help='混合水印（图像+文字）样本比例')
    
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
        remaining_count = args.count - min(existing_watermarked, existing_masks)
        
        if remaining_count <= 0:
            print(f"已经有足够的图片（{min(existing_watermarked, existing_masks)}张），无需生成更多")
            return
    else:
        remaining_count = args.count - existing_watermarked
        
        if remaining_count <= 0:
            print(f"已经有足够的图片（{existing_watermarked}张），无需生成更多")
            return
    
    print(f"需要生成 {remaining_count} 张新图片")
    
    # 生成图片
    generated_count = 0
    transparent_count = 0
    multi_watermark_count = 0
    text_watermark_count = 0
    mixed_watermark_count = 0
    
    target_transparent = int(remaining_count * args.transparent_ratio)
    target_text_watermark = int(remaining_count * args.text_watermark_ratio)
    target_mixed_watermark = int(remaining_count * args.mixed_watermark_ratio)
    
    pbar = tqdm(total=remaining_count, desc="生成带水印图片")
    
    while generated_count < remaining_count:
        # 决定是否生成透明水印
        should_generate_transparent = (transparent_count < target_transparent) or \
                                    (random.random() < args.transparent_ratio)
        
        # 随机选择干净图片
        clean_path = random.choice(clean_images)
        
        try:
            # 决定水印类型：文字水印、混合水印、还是图像水印
            watermark_type_rand = random.random()
            
            if watermark_type_rand < args.text_watermark_ratio and text_watermark_count < target_text_watermark:
                # 生成文字水印
                watermarked_img, mask = generate_text_watermark(
                    clean_path, 
                    enhance_transparent=should_generate_transparent,
                    use_ocr_mask=args.use_ocr_mask
                )
                
                # 生成文件名（文字水印）
                prefix = "text_trans_" if should_generate_transparent else "text_norm_"
                # 安全处理文件名，避免中文字符编码问题
                try:
                    clean_name = os.path.splitext(os.path.basename(clean_path))[0]
                    unique_str = f"{clean_name}_text_{generated_count}"
                    # 确保字符串可以安全编码
                    unique_str.encode('ascii')
                except UnicodeEncodeError:
                    # 如果包含非ASCII字符，使用简化的文件名
                    clean_name = "clean"
                    unique_str = f"{clean_name}_text_{generated_count}"
                
                hash_obj = hashlib.md5(unique_str.encode('utf-8'))
                filename = prefix + hash_obj.hexdigest()[:16]
                
                text_watermark_count += 1
                
            elif watermark_type_rand < (args.text_watermark_ratio + args.mixed_watermark_ratio) and \
                 mixed_watermark_count < target_mixed_watermark and len(watermarks) > 0:
                # 生成混合水印
                watermarked_img, mask = generate_mixed_watermark(
                    clean_path, watermarks,
                    enhance_transparent=should_generate_transparent,
                    use_ocr_mask=args.use_ocr_mask,
                    max_watermarks=2
                )
                
                # 生成文件名（混合水印）
                prefix = "mixed_trans_" if should_generate_transparent else "mixed_norm_"
                # 安全处理文件名，避免中文字符编码问题
                try:
                    clean_name = os.path.splitext(os.path.basename(clean_path))[0]
                    unique_str = f"{clean_name}_mixed_{generated_count}"
                    # 确保字符串可以安全编码
                    unique_str.encode('ascii')
                except UnicodeEncodeError:
                    # 如果包含非ASCII字符，使用简化的文件名
                    clean_name = "clean"
                    unique_str = f"{clean_name}_mixed_{generated_count}"
                
                hash_obj = hashlib.md5(unique_str.encode('utf-8'))
                filename = prefix + hash_obj.hexdigest()[:16]
                
                mixed_watermark_count += 1
                
            else:
                # 生成图像水印（单个或多个）
                if len(watermarks) == 0:
                    print("警告：没有找到图像水印，跳过")
                    continue
                    
                watermark_path = random.choice(watermarks)
                
                # 决定是否生成多水印图片
                should_generate_multi = random.random() < args.multi_watermark_ratio
                
                if should_generate_multi and len(watermarks) > 1:
                    # 生成多水印图片
                    num_watermarks = random.randint(2, args.max_watermarks)
                    selected_watermarks = random.sample(watermarks, min(num_watermarks, len(watermarks)))
                    
                    watermarked_img, mask = generate_multiple_watermarks_image(
                        clean_path, selected_watermarks,
                        enhance_transparent=should_generate_transparent,
                        max_watermarks=args.max_watermarks
                    )
                    
                    # 生成文件名（多水印）
                    prefix = "multi_trans_" if should_generate_transparent else "multi_norm_"
                    # 安全处理文件名，避免中文字符编码问题
                    try:
                        watermark_names = "_".join([os.path.splitext(os.path.basename(w))[0][:8] for w in selected_watermarks[:2]])
                        clean_name = os.path.splitext(os.path.basename(clean_path))[0]
                        unique_str = f"{clean_name}_{watermark_names}_{generated_count}"
                        # 确保字符串可以安全编码
                        unique_str.encode('ascii')
                    except UnicodeEncodeError:
                        # 如果包含非ASCII字符，使用简化的文件名
                        clean_name = "clean"
                        watermark_names = "watermarks"
                        unique_str = f"{clean_name}_{watermark_names}_{generated_count}"
                    
                    hash_obj = hashlib.md5(unique_str.encode('utf-8'))
                    filename = prefix + hash_obj.hexdigest()[:16]
                    
                    multi_watermark_count += 1
                    
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
            
            generated_count += 1
            pbar.update(1)
            
        except Exception as e:
            print(f"\n生成图片时出错: {e}")
            continue
    
    pbar.close()
    
    # 计算各类型水印数量
    single_watermark_count = generated_count - multi_watermark_count - text_watermark_count - mixed_watermark_count
    
    print(f"\n成功生成 {generated_count} 张带水印图片")
    print(f"\n=== 水印类型统计 ===")
    print(f"文字水印: {text_watermark_count} 张 ({text_watermark_count/generated_count*100:.1f}%)")
    print(f"混合水印: {mixed_watermark_count} 张 ({mixed_watermark_count/generated_count*100:.1f}%)")
    print(f"多图像水印: {multi_watermark_count} 张 ({multi_watermark_count/generated_count*100:.1f}%)")
    print(f"单图像水印: {single_watermark_count} 张 ({single_watermark_count/generated_count*100:.1f}%)")
    print(f"\n=== 透明度统计 ===")
    print(f"透明水印: {transparent_count} 张 ({transparent_count/generated_count*100:.1f}%)")
    print(f"普通水印: {generated_count - transparent_count} 张 ({(generated_count - transparent_count)/generated_count*100:.1f}%)")
    
    print(f"\n=== 优化特性 ===")
    print(f"- 支持文字水印生成（多种字体、颜色、效果）")
    print(f"- 支持混合水印（图像+文字组合）")
    if args.use_ocr_mask:
        print(f"- 使用OCR生成精确文字区域mask")
    print(f"- 支持任意角度旋转 (0-360度)")
    print(f"- 支持水印形变和倾斜")
    print(f"- 支持模糊效果 (40%概率)")
    print(f"- 支持残缺效果 (30%概率)")
    print(f"- 支持多水印叠加 (最多{args.max_watermarks}个)")
    
    print(f"\n=== 输出路径 ===")
    print(f"带水印图片保存在: {watermarked_dir}")
    
    if args.generate_mask:
        print(f"Mask图片保存在: {masks_dir}")
    else:
        print("未生成mask图片（使用 --generate_mask 参数启用）")
    
    print(f"\n=== 使用建议 ===")
    print(f"- 文字水印比例: {args.text_watermark_ratio*100:.1f}% (可用 --text_watermark_ratio 调整)")
    print(f"- 混合水印比例: {args.mixed_watermark_ratio*100:.1f}% (可用 --mixed_watermark_ratio 调整)")
    print(f"- 透明水印比例: {args.transparent_ratio*100:.1f}% (可用 --transparent_ratio 调整)")
    if args.use_ocr_mask:
        print(f"- OCR mask已启用，文字区域标注更精确")
    else:
        print(f"- 使用 --use_ocr_mask 启用OCR精确标注")

if __name__ == '__main__':
    main()