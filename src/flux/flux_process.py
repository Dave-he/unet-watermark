#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLUX Kontext 批量图像处理脚本
支持水印去除、图像编辑等功能
"""

import torch
import numpy as np
from PIL import Image
from diffusers import FluxKontextPipeline
import argparse
import os
import sys
import random
import logging
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.video_generator import VideoGenerator

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 全局模型变量
model = None

# ===== 1. 环境初始化与模型加载 =====
def init_model():
    """加载4位量化版 Flux-Kontext 模型，启用 Nunchaku 加速"""
    global model
    
    logger.info("正在加载 FLUX Kontext 模型...")
    
    try:
        pipeline = FluxKontextPipeline.from_pretrained(
            "mit-han-lab/nunchaku-flux.1-kontext-dev",  # INT4量化模型
            torch_dtype=torch.float16,                   # A10兼容FP16
            variant="int4_r32",                           # 量化配置
            device_map="auto"
        )
        pipeline.to("cuda")
        
        # 启用硬件加速优化
        pipeline.enable_nunchaku_optimizations(
            attention_mode="nunchaku-fp16",    # A10适配模式
            use_cpu_offload=True,               # 显存<18GB时启用CPU卸载
            cache_threshold=0.12                # 首层特征缓存平衡点
        )
        
        model = pipeline
        logger.info("模型加载完成")
        return pipeline
        
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        # 尝试加载标准版本
        logger.info("尝试加载标准版本...")
        try:
            pipeline = FluxKontextPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-Kontext-dev",
                torch_dtype=torch.float16,
                device_map="auto"
            )
            pipeline.to("cuda")
            model = pipeline
            logger.info("标准版本模型加载完成")
            return pipeline
        except Exception as e2:
            logger.error(f"标准版本加载也失败: {str(e2)}")
            raise e2

# ===== 2. 核心推理函数 =====
def remove_watermark(image: Image.Image, prompt: str = "Remove watermark") -> Image.Image:
    """图像水印擦除（优化参数配置）"""
    global model
    
    if model is None:
        raise ValueError("模型未初始化，请先调用 init_model()")
    
    try:
        output = model(
            image=image,
            prompt=f"{prompt} | Keep original details, remove watermark only",
            guidance_scale=2.5,                # 控制编辑强度
            num_inference_steps=20,             # 加速推理步数
            generator=torch.Generator(device="cuda").manual_seed(42)
        ).images[0]
        return output
    except Exception as e:
        logger.error(f"图像处理失败: {str(e)}")
        raise e

def edit_image(image: Image.Image, prompt: str) -> Image.Image:
    """通用图像编辑功能"""
    global model
    
    if model is None:
        raise ValueError("模型未初始化，请先调用 init_model()")
    
    try:
        output = model(
            image=image,
            prompt=prompt,
            guidance_scale=3.0,                # 编辑任务使用稍高的引导强度
            num_inference_steps=25,             # 编辑任务使用更多步数
            generator=torch.Generator(device="cuda").manual_seed(42)
        ).images[0]
        return output
    except Exception as e:
        logger.error(f"图像编辑失败: {str(e)}")
        raise e

# ===== 3. 批量处理功能 =====
def get_image_files(input_dir: str, limit: Optional[int] = None, random_select: bool = False) -> List[str]:
    """获取图像文件列表"""
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    image_files = []
    for file_path in Path(input_dir).iterdir():
        if file_path.suffix.lower() in supported_formats:
            image_files.append(str(file_path))
    
    logger.info(f"找到 {len(image_files)} 张图片")
    
    # 随机选择
    if random_select and limit and len(image_files) > limit:
        image_files = random.sample(image_files, limit)
        logger.info(f"随机选择了 {len(image_files)} 张图片")
    elif limit and len(image_files) > limit:
        image_files = image_files[:limit]
        logger.info(f"选择前 {len(image_files)} 张图片")
    
    return image_files

def process_batch(input_dir: str, output_dir: str, prompt: str = "Remove watermark", 
                 limit: Optional[int] = None, random_select: bool = False, 
                 task_type: str = "watermark") -> List[tuple]:
    """批量处理图像"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取图像文件
    image_files = get_image_files(input_dir, limit, random_select)
    
    if not image_files:
        logger.warning("未找到任何图像文件")
        return []
    
    # 初始化模型
    if model is None:
        init_model()
    
    processed_pairs = []
    failed_count = 0
    
    # 处理图像
    progress_bar = tqdm(image_files, desc="处理图像", unit="张")
    
    for image_path in progress_bar:
        try:
            # 更新进度条描述
            filename = Path(image_path).name
            progress_bar.set_postfix({"当前": filename[:20] + "..." if len(filename) > 20 else filename})
            
            # 加载图像
            image = Image.open(image_path).convert('RGB')
            
            # 处理图像
            if task_type == "watermark":
                processed_image = remove_watermark(image, prompt)
            else:
                processed_image = edit_image(image, prompt)
            
            # 保存处理后的图像
            output_filename = f"{Path(image_path).stem}_processed{Path(image_path).suffix}"
            output_path = os.path.join(output_dir, output_filename)
            processed_image.save(output_path, quality=95)
            
            # 记录成功处理的图像对
            processed_pairs.append((image_path, output_path))
            
            logger.debug(f"处理完成: {filename} -> {output_filename}")
            
        except Exception as e:
            failed_count += 1
            logger.error(f"处理失败 {Path(image_path).name}: {str(e)}")
            continue
    
    progress_bar.close()
    
    logger.info(f"批量处理完成: 成功 {len(processed_pairs)} 张, 失败 {failed_count} 张")
    return processed_pairs

# ===== 4. 视频生成功能 =====
def generate_comparison_video(input_dir: str, output_dir: str, video_output_dir: str,
                            video_type: str = "sidebyside", duration: float = 2.0, 
                            fps: int = 30, resolution: tuple = (1280, 720)):
    """生成对比视频"""
    try:
        logger.info("开始生成对比视频...")
        
        # 创建视频输出目录
        os.makedirs(video_output_dir, exist_ok=True)
        
        # 初始化视频生成器
        video_generator = VideoGenerator(
            input_dir=input_dir,
            repair_dir=output_dir,
            output_dir=video_output_dir,
            width=resolution[0],
            height=resolution[1],
            duration_per_image=duration,
            fps=fps
        )
        
        # 生成视频
        if video_type == "sidebyside":
            video_path = video_generator.create_side_by_side_video()
        else:
            video_path = video_generator.create_comparison_video()
        
        logger.info(f"对比视频生成完成: {video_path}")
        return video_path
        
    except Exception as e:
        logger.error(f"视频生成失败: {str(e)}")
        return None

# ===== 5. 命令行接口 =====
def main():
    parser = argparse.ArgumentParser(description="FLUX Kontext 批量图像处理工具")
    
    # 基本参数
    parser.add_argument("--input", "-i", required=True, help="输入图像目录")
    parser.add_argument("--output", "-o", required=True, help="输出图像目录")
    parser.add_argument("--prompt", "-p", default="Remove watermark", help="处理提示词")
    
    # 处理选项
    parser.add_argument("--limit", "-l", type=int, help="限制处理图像数量")
    parser.add_argument("--random", "-r", action="store_true", help="随机选择图像")
    parser.add_argument("--task", "-t", choices=["watermark", "edit"], default="watermark", 
                       help="任务类型: watermark(去水印) 或 edit(通用编辑)")
    
    # 视频生成选项
    parser.add_argument("--video", "-v", action="store_true", help="生成对比视频")
    parser.add_argument("--video-output", help="视频输出目录 (默认为输出目录下的videos子目录)")
    parser.add_argument("--video-type", choices=["sidebyside", "sequence"], default="sidebyside",
                       help="视频类型: sidebyside(并排对比) 或 sequence(序列对比)")
    parser.add_argument("--duration", type=float, default=2.0, help="每张图片展示时长(秒)")
    parser.add_argument("--fps", type=int, default=30, help="视频帧率")
    parser.add_argument("--resolution", nargs=2, type=int, default=[1280, 720], 
                       help="视频分辨率 (宽度 高度)")
    
    # 其他选项
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 检查输入目录
    if not os.path.exists(args.input):
        logger.error(f"输入目录不存在: {args.input}")
        return 1
    
    # 设置视频输出目录
    if args.video and not args.video_output:
        args.video_output = os.path.join(args.output, "videos")
    
    try:
        # 批量处理图像
        logger.info(f"开始批量处理: {args.input} -> {args.output}")
        logger.info(f"任务类型: {args.task}, 提示词: {args.prompt}")
        
        processed_pairs = process_batch(
            input_dir=args.input,
            output_dir=args.output,
            prompt=args.prompt,
            limit=args.limit,
            random_select=args.random,
            task_type=args.task
        )
        
        if not processed_pairs:
            logger.warning("没有成功处理任何图像")
            return 1
        
        # 生成对比视频
        if args.video:
            video_path = generate_comparison_video(
                input_dir=args.input,
                output_dir=args.output,
                video_output_dir=args.video_output,
                video_type=args.video_type,
                duration=args.duration,
                fps=args.fps,
                resolution=tuple(args.resolution)
            )
            
            if video_path:
                logger.info(f"处理完成! 对比视频: {video_path}")
            else:
                logger.warning("视频生成失败，但图像处理已完成")
        
        logger.info("所有任务完成!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("用户中断操作")
        return 1
    except Exception as e:
        logger.error(f"处理过程中发生错误: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())