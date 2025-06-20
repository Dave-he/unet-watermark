#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频生成器模块
用于生成原图与修复图的对比视频
"""

import os
import cv2
import numpy as np
from pathlib import Path
from moviepy import ImageSequenceClip, concatenate_videoclips
from PIL import Image, ImageDraw, ImageFont
import logging
import argparse
import sys
import locale
from tqdm import tqdm
from typing import Optional, List, Tuple, Dict, Any


# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoGenerator:
    """视频生成器类"""
    
    def __init__(self, input_dir: str, repair_dir: str, output_dir: str, mask_dir: Optional[str] = None, 
                 width: int = 640, height: int = 480, duration_per_image: float = 2.0, fps: int = 30) -> None:
        """
        初始化视频生成器
        
        Args:
            input_dir (str): 原图目录
            repair_dir (str): 修复图目录
            output_dir (str): 输出目录
            mask_dir (str, optional): mask图目录
            width (int): 视频宽度
            height (int): 视频高度
            duration_per_image (float): 每张图片展示时长(秒)
            fps (int): 视频帧率
        """
        self.input_dir = input_dir
        self.repair_dir = repair_dir
        self.output_dir = output_dir
        self.mask_dir = mask_dir
        self.width = width
        self.height = height
        self.duration_per_image = duration_per_image
        self.fps = fps
        
        # 支持的图片格式
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        logger.info(f"视频生成器初始化完成")
        logger.info(f"视频尺寸: {width}x{height}, 帧率: {fps}fps")
        logger.info(f"每张图片展示时长: {duration_per_image}秒")
        if mask_dir:
            logger.info(f"包含mask图片: {mask_dir}")
    
    def find_image_triplets(self):
        """
        查找原图、修复图和mask图的三元组配对
        
        Returns:
            list: 配对的图片路径列表 [(original_path, repaired_path, mask_path), ...]
        """
        triplets = []
        
        # 获取修复图目录中的所有图片
        repair_files = {}
        repair_file_list = list(Path(self.repair_dir).iterdir())
        
        # 添加进度条显示修复图扫描过程
        for file_path in tqdm(repair_file_list, desc="扫描修复图", unit="个"):
            if file_path.suffix.lower() in self.image_extensions:
                # 提取文件名（不含扩展名）
                base_name = file_path.stem
                # 移除可能的后缀（如_cleaned, _repaired等）
                clean_name = base_name.replace('_cleaned', '').replace('_repaired', '').replace('_fixed', '').replace('_partial_cleaned', '')
                repair_files[clean_name] = str(file_path)
        
        # 获取mask图目录中的所有图片（如果提供）
        mask_files = {}
        if self.mask_dir and os.path.exists(self.mask_dir):
            mask_file_list = list(Path(self.mask_dir).iterdir())
            for file_path in tqdm(mask_file_list, desc="扫描mask图", unit="个"):
                if file_path.suffix.lower() in self.image_extensions:
                    base_name = file_path.stem
                    # 移除可能的后缀
                    clean_name = base_name.replace('_mask', '').replace('_final_mask', '')
                    mask_files[clean_name] = str(file_path)
        
        # 在原图目录中查找对应的原图
        input_file_list = list(Path(self.input_dir).iterdir())
        
        # 添加进度条显示原图匹配过程
        for file_path in tqdm(input_file_list, desc="匹配图片组", unit="个"):
            if file_path.suffix.lower() in self.image_extensions:
                base_name = file_path.stem
                
                # 尝试匹配修复图
                if base_name in repair_files:
                    mask_path = mask_files.get(base_name, None)
                    triplets.append((str(file_path), repair_files[base_name], mask_path))
                    logger.debug(f"找到配对: {file_path.name} <-> {Path(repair_files[base_name]).name} <-> {Path(mask_path).name if mask_path else 'None'}")
        
        logger.info(f"共找到 {len(triplets)} 组图片")
        return triplets
    
    def find_image_pairs(self):
        """
        查找原图和修复图的配对
        
        Returns:
            list: 配对的图片路径列表 [(original_path, repaired_path), ...]
        """
        pairs = []
        
        # 获取修复图目录中的所有图片
        repair_files = {}
        repair_file_list = list(Path(self.repair_dir).iterdir())
        
        # 添加进度条显示修复图扫描过程
        for file_path in tqdm(repair_file_list, desc="扫描修复图", unit="个"):
            if file_path.suffix.lower() in self.image_extensions:
                # 提取文件名（不含扩展名）
                base_name = file_path.stem
                # 移除可能的后缀（如_cleaned, _repaired等）
                clean_name = base_name.replace('_cleaned', '').replace('_repaired', '').replace('_fixed', '').replace('_partial_cleaned', '')
                repair_files[clean_name] = str(file_path)
        
        # 在原图目录中查找对应的原图
        input_file_list = list(Path(self.input_dir).iterdir())
        
        # 添加进度条显示原图匹配过程
        for file_path in tqdm(input_file_list, desc="匹配图片对", unit="个"):
            if file_path.suffix.lower() in self.image_extensions:
                base_name = file_path.stem
                
                # 尝试匹配修复图
                if base_name in repair_files:
                    pairs.append((str(file_path), repair_files[base_name]))
                    logger.debug(f"找到配对: {file_path.name} <-> {Path(repair_files[base_name]).name}")
        
        logger.info(f"共找到 {len(pairs)} 对图片")
        return pairs
    
    def resize_image_with_padding(self, image_path, target_width, target_height):
        """
        调整图片尺寸，保持宽高比并添加黑边
        
        Args:
            image_path (str): 图片路径
            target_width (int): 目标宽度
            target_height (int): 目标高度
            
        Returns:
            np.ndarray: 处理后的图片数组
        """
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        h, w = image.shape[:2]
        
        # 计算缩放比例
        scale = min(target_width / w, target_height / h)
        
        # 计算新尺寸
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 缩放图片
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # 创建黑色背景
        result = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        # 计算居中位置
        y_offset = (target_height - new_h) // 2
        x_offset = (target_width - new_w) // 2
        
        # 将缩放后的图片放置在中心
        result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        return result
    
    def add_text_overlay(self, image, text, position='top'):
        """
        在图片上添加文字标签
        
        Args:
            image (np.ndarray): 图片数组
            text (str): 要添加的文字
            position (str): 文字位置 ('top' 或 'bottom')
            
        Returns:
            np.ndarray: 添加文字后的图片
        """
        # 转换为PIL图片
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        # 尝试加载字体
        try:
            # 在macOS上尝试使用系统字体
            font_size = max(24, self.height // 40)
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
        except:
            try:
                # 备用字体
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
            except:
                # 使用默认字体
                font = ImageFont.load_default()
        
        # 获取文字尺寸
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # 计算文字位置
        x = (self.width - text_width) // 2
        if position == 'top':
            y = 20
        else:  # bottom
            y = self.height - text_height - 20
        
        # 添加半透明背景
        overlay = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        padding = 10
        overlay_draw.rectangle([
            x - padding, y - padding,
            x + text_width + padding, y + text_height + padding
        ], fill=(0, 0, 0, 128))
        
        # 合并背景
        pil_image = Image.alpha_composite(pil_image.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(pil_image)
        
        # 绘制文字
        draw.text((x, y), text, font=font, fill=(255, 255, 255))
        
        # 转换回OpenCV格式
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    def create_comparison_video(self):
        """
        创建对比视频
        
        Returns:
            str: 生成的视频文件路径
        """
        # 查找图片配对
        image_pairs = self.find_image_pairs()
        
        if not image_pairs:
            raise ValueError("未找到任何图片配对")
        
        # 准备视频帧序列
        all_frames = []
        
        # 添加总体进度条
        pair_progress = tqdm(image_pairs, desc="处理图片对", unit="对")
        
        for i, (original_path, repaired_path) in enumerate(pair_progress):
            # 更新进度条描述
            pair_progress.set_postfix({
                '当前': Path(original_path).name[:15] + '...' if len(Path(original_path).name) > 15 else Path(original_path).name,
                '进度': f"{i+1}/{len(image_pairs)}"
            })
            
            try:
                # 处理原图
                original_frame = self.resize_image_with_padding(original_path, self.width, self.height)
                original_frame = self.add_text_overlay(original_frame, f"原图 - {Path(original_path).name}", 'top')
                
                # 处理修复图
                repaired_frame = self.resize_image_with_padding(repaired_path, self.width, self.height)
                repaired_frame = self.add_text_overlay(repaired_frame, f"修复后 - {Path(repaired_path).name}", 'top')
                
                # 计算每张图片需要的帧数
                frames_per_image = int(self.duration_per_image * self.fps)
                
                # 添加原图帧
                for _ in range(frames_per_image):
                    all_frames.append(original_frame)
                
                # 添加修复图帧
                for _ in range(frames_per_image):
                    all_frames.append(repaired_frame)
                
            except Exception as e:
                logger.error(f"处理图片对失败 {original_path} <-> {repaired_path}: {str(e)}")
                continue
        
        pair_progress.close()
        
        if not all_frames:
            raise ValueError("没有成功处理任何图片对")
        
        logger.info(f"准备生成视频，总帧数: {len(all_frames)}")
        
        # 生成视频文件名
        video_filename = f"watermark_comparison_{len(image_pairs)}pairs.mp4"
        video_path = os.path.join(self.output_dir, video_filename)
        
        # 创建临时图片序列目录
        temp_dir = os.path.join(self.output_dir, "temp_frames")
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # 保存所有帧为临时图片（添加进度条）
            frame_paths = []
            frame_progress = tqdm(all_frames, desc="保存帧图片", unit="帧")
            
            for i, frame in enumerate(frame_progress):
                frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
                
                # 更新进度条信息
                if i % 100 == 0:  # 每100帧更新一次显示
                    frame_progress.set_postfix({
                        '已保存': f"{i+1}/{len(all_frames)}",
                        '大小': f"{len(frame_paths)}"
                    })
            
            frame_progress.close()
            
            # 使用moviepy创建视频
            logger.info("开始生成视频...")
            
            # 创建视频进度条
            video_progress = tqdm(total=len(frame_paths), desc="生成视频", unit="帧")
            
            try:
                clip = ImageSequenceClip(frame_paths, fps=self.fps)
                clip.write_videofile(
                    video_path,
                    codec='libx264',
                    audio=False,
                    logger=None
                )
            finally:
                video_progress.close()
            
            logger.info(f"视频生成完成: {video_path}")
            
        finally:
            # 清理临时文件
            import shutil
            if os.path.exists(temp_dir):
                cleanup_progress = tqdm(desc="清理临时文件", total=1, unit="目录")
                shutil.rmtree(temp_dir)
                cleanup_progress.update(1)
                cleanup_progress.close()
                logger.info("临时文件已清理")
        
        return video_path
    
    def create_side_by_side_video(self):
        """
        创建左右对比视频（原图和修复图并排显示）
        
        Returns:
            str: 生成的视频文件路径
        """
        # 查找图片配对
        image_pairs = self.find_image_pairs()
        
        if not image_pairs:
            raise ValueError("未找到任何图片配对")
        
        # 准备视频帧序列
        all_frames = []
        
        # 计算单张图片的尺寸（左右各占一半）
        single_width = self.width // 2
        single_height = self.height
        
        # 添加总体进度条
        pair_progress = tqdm(image_pairs, desc="处理并排对比", unit="对")
        
        for i, (original_path, repaired_path) in enumerate(pair_progress):
            # 更新进度条描述
            pair_progress.set_postfix({
                '当前': Path(original_path).name[:15] + '...' if len(Path(original_path).name) > 15 else Path(original_path).name,
                '模式': '并排对比'
            })
            
            try:
                # 处理原图
                original_frame = self.resize_image_with_padding(original_path, single_width, single_height)
                original_frame = self.add_text_overlay(original_frame, f"Original - {Path(original_path).name}", 'top')
                
                # 处理修复图 - 添加这行缺失的代码
                repaired_frame = self.resize_image_with_padding(repaired_path, single_width, single_height)
                repaired_frame = self.add_text_overlay(repaired_frame, f"Repaired - {Path(repaired_path).name}", 'top')
                
                # 简化文字标签
                original_frame = self.add_text_overlay(original_frame, f"Original", 'top')
                repaired_frame = self.add_text_overlay(repaired_frame, f"Repaired", 'top')
                
                # 合并左右图片
                combined_frame = np.hstack([original_frame, repaired_frame])
                
                # 添加图片名称标签
                combined_frame = self.add_text_overlay(
                    combined_frame, 
                    f"{Path(original_path).name}", 
                    'bottom'
                )
                
                # 计算每张图片需要的帧数
                frames_per_image = int(self.duration_per_image * self.fps)
                
                # 添加帧
                for _ in range(frames_per_image):
                    all_frames.append(combined_frame)
                
            except Exception as e:
                logger.error(f"处理图片对失败 {original_path} <-> {repaired_path}: {str(e)}")
                continue
        
        pair_progress.close()
        
        if not all_frames:
            raise ValueError("没有成功处理任何图片对")
        
        logger.info(f"准备生成并排对比视频，总帧数: {len(all_frames)}")
        
        # 生成视频文件名
        video_filename = f"watermark_sidebyside_{len(image_pairs)}pairs.mp4"
        video_path = os.path.join(self.output_dir, video_filename)
        
        # 创建临时图片序列目录
        temp_dir = os.path.join(self.output_dir, "temp_frames_sidebyside")
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # 保存所有帧为临时图片（添加进度条）
            frame_paths = []
            frame_progress = tqdm(all_frames, desc="保存并排帧", unit="帧")
            
            for i, frame in enumerate(frame_progress):
                frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
                
                # 更新进度条信息
                if i % 50 == 0:  # 每50帧更新一次显示
                    frame_progress.set_postfix({
                        '已保存': f"{i+1}/{len(all_frames)}"
                    })
            
            frame_progress.close()
            
            # 使用moviepy创建视频
            logger.info("开始生成并排对比视频...")
            
            # 创建视频进度条
            video_progress = tqdm(total=len(frame_paths), desc="生成并排视频", unit="帧")
            
            try:
                clip = ImageSequenceClip(frame_paths, fps=self.fps)
                clip.write_videofile(
                    video_path,
                    codec='libx264',
                    audio=False,
                    logger=None
                )
            finally:
                video_progress.close()
            
            logger.info(f"并排对比视频生成完成: {video_path}")
            
        finally:
            # 清理临时文件
            import shutil
            if os.path.exists(temp_dir):
                cleanup_progress = tqdm(desc="清理临时文件", total=1, unit="目录")
                shutil.rmtree(temp_dir)
                cleanup_progress.update(1)
                cleanup_progress.close()
                logger.info("临时文件已清理")
        
        return video_path

    def create_three_way_comparison_video(self):
        """
        创建三路对比视频（原图、修复图、mask图）
        
        Returns:
            str: 生成的视频文件路径
        """
        # 查找图片三元组
        image_triplets = self.find_image_triplets()
        
        if not image_triplets:
            raise ValueError("未找到任何图片配对")
        
        # 准备视频帧序列
        all_frames = []
        
        # 计算单张图片的尺寸（三等分宽度）
        single_width = self.width // 3
        single_height = self.height
        
        # 添加总体进度条
        triplet_progress = tqdm(image_triplets, desc="处理三路对比", unit="组")
        
        for i, (original_path, repaired_path, mask_path) in enumerate(triplet_progress):
            # 更新进度条描述
            triplet_progress.set_postfix({
                '当前': Path(original_path).name[:15] + '...' if len(Path(original_path).name) > 15 else Path(original_path).name,
                '模式': '三路对比'
            })
            
            try:
                # 处理原图
                original_frame = self.resize_image_with_padding(original_path, single_width, single_height)
                original_frame = self.add_text_overlay(original_frame, "Original", 'top')
                
                # 处理修复图
                repaired_frame = self.resize_image_with_padding(repaired_path, single_width, single_height)
                repaired_frame = self.add_text_overlay(repaired_frame, "Repaired", 'top')
                
                # 处理mask图
                if mask_path and os.path.exists(mask_path):
                    # 读取mask并转换为彩色图像
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_HOT)  # 使用热力图颜色
                    
                    # 保存为临时文件以便resize_image_with_padding处理
                    temp_mask_path = os.path.join(self.output_dir, "temp_mask.png")
                    cv2.imwrite(temp_mask_path, mask_colored)
                    
                    mask_frame = self.resize_image_with_padding(temp_mask_path, single_width, single_height)
                    mask_frame = self.add_text_overlay(mask_frame, "Mask", 'top')
                    
                    # 清理临时文件
                    os.remove(temp_mask_path)
                else:
                    # 如果没有mask，创建黑色占位图
                    mask_frame = np.zeros((single_height, single_width, 3), dtype=np.uint8)
                    mask_frame = self.add_text_overlay(mask_frame, "No Mask", 'top')
                
                # 合并三张图片
                combined_frame = np.hstack([original_frame, repaired_frame, mask_frame])
                
                # 添加图片名称标签
                combined_frame = self.add_text_overlay(
                    combined_frame, 
                    f"{Path(original_path).name}", 
                    'bottom'
                )
                
                # 计算每张图片需要的帧数
                frames_per_image = int(self.duration_per_image * self.fps)
                
                # 添加帧
                for _ in range(frames_per_image):
                    all_frames.append(combined_frame)
                
            except Exception as e:
                logger.error(f"处理图片组失败 {original_path} <-> {repaired_path} <-> {mask_path}: {str(e)}")
                continue
        
        triplet_progress.close()
        
        if not all_frames:
            raise ValueError("没有成功处理任何图片组")
        
        logger.info(f"准备生成三路对比视频，总帧数: {len(all_frames)}")
        
        # 生成视频文件名
        video_filename = f"watermark_threeway_{len(image_triplets)}groups.mp4"
        video_path = os.path.join(self.output_dir, video_filename)
        
        # 创建临时图片序列目录
        temp_dir = os.path.join(self.output_dir, "temp_frames_threeway")
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # 保存所有帧为临时图片（添加进度条）
            frame_paths = []
            frame_progress = tqdm(all_frames, desc="保存三路帧", unit="帧")
            
            for i, frame in enumerate(frame_progress):
                frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
                
                # 更新进度条信息
                if i % 50 == 0:  # 每50帧更新一次显示
                    frame_progress.set_postfix({
                        '已保存': f"{i+1}/{len(all_frames)}"
                    })
            
            frame_progress.close()
            
            # 使用moviepy创建视频
            logger.info("开始生成三路对比视频...")
            
            # 创建视频进度条
            video_progress = tqdm(total=len(frame_paths), desc="生成三路视频", unit="帧")
            
            try:
                clip = ImageSequenceClip(frame_paths, fps=self.fps)
                clip.write_videofile(
                    video_path,
                    codec='libx264',
                    audio=False,
                    logger=None
                )
            finally:
                video_progress.close()
            
            logger.info(f"三路对比视频生成完成: {video_path}")
            
        finally:
            # 清理临时文件
            import shutil
            if os.path.exists(temp_dir):
                cleanup_progress = tqdm(desc="清理临时文件", total=1, unit="目录")
                shutil.rmtree(temp_dir)
                cleanup_progress.update(1)
                cleanup_progress.close()
                logger.info("临时文件已清理")
        
        return video_path

def main():
    """主函数 - 命令行入口"""
    parser = argparse.ArgumentParser(
        description="水印修复对比视频生成器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  生成切换对比视频:
    python video_generator.py --input data/original --repair data/repaired --output videos
    
  生成并排对比视频:
    python video_generator.py --input data/original --repair data/repaired --output videos --mode sidebyside
    
  自定义参数:
    python video_generator.py --input data/original --repair data/repaired --output videos \
                              --width 1920 --height 1080 --duration 3 --fps 24
        """
    )
    
    # 必需参数
    parser.add_argument('--input', '-i', type=str, default='data/test',
                       help='原图目录路径')
    parser.add_argument('--repair', '-r', type=str, default='data/result',
                       help='修复图目录路径')
    parser.add_argument('--output', '-o', type=str, default='data/video',
                       help='视频输出目录')
    parser.add_argument('--mask', type=str, default='data/result/mask',
                       help='修复图mask输出目录')
    
    # 可选参数
    parser.add_argument('--mode', '-m', type=str, choices=['switch', 'sidebyside'], 
                       default='sidebyside', help='视频模式：switch(切换对比) 或 sidebyside(并排对比) (默认: switch)')
    parser.add_argument('--width', '-w', type=int, default=640,
                       help='视频宽度 (默认: 640)')
    parser.add_argument('--height', type=int, default=480,
                       help='视频高度 (默认: 480)')
    parser.add_argument('--duration', '-d', type=float, default=2.0,
                       help='每张图片展示时长(秒) (默认: 2.0)')
    parser.add_argument('--fps', '-f', type=int, default=30,
                       help='视频帧率 (默认: 30)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='显示详细日志')
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 验证输入参数
    if not os.path.exists(args.input):
        logger.error(f"原图目录不存在: {args.input}")
        sys.exit(1)
    
    if not os.path.exists(args.repair):
        logger.error(f"修复图目录不存在: {args.repair}")
        sys.exit(1)
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 打印配置信息
    logger.info("=" * 60)
    logger.info("水印修复对比视频生成器")
    logger.info("=" * 60)
    logger.info(f"原图目录: {args.input}")
    logger.info(f"修复图目录: {args.repair}")
    logger.info(f"输出目录: {args.output}")
    logger.info(f"视频模式: {args.mode}")
    logger.info(f"视频尺寸: {args.width}x{args.height}")
    logger.info(f"每张图片展示时长: {args.duration}秒")
    logger.info(f"视频帧率: {args.fps}fps")
    logger.info("=" * 60)
    
    try:
        # 创建视频生成器
        generator = VideoGenerator(
            input_dir=args.input,
            repair_dir=args.repair,
            output_dir=args.output,
            mask_dir=args.mask,
            width=args.width,
            height=args.height,
            duration_per_image=args.duration,
            fps=args.fps
        )
        
        # 根据模式生成视频
        if args.mode == 'switch':
            video_path = generator.create_comparison_video()
        elif args.mode == 'sidebyside':
            video_path = generator.create_side_by_side_video()
        
        logger.info("\n" + "=" * 60)
        logger.info("视频生成完成！")
        logger.info(f"输出路径: {video_path}")
        logger.info("=" * 60)
        
    except KeyboardInterrupt:
        logger.info("\n用户中断操作")
        sys.exit(1)
    except Exception as e:
        logger.error(f"视频生成过程中出现错误: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

