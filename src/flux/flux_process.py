#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLUX Kontext 简化版批量图像处理脚本
直接加载标准版本模型，简化配置
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
import cv2

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.video_generator import VideoGenerator
from ocr.easy_ocr import EasyOCRDetector

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 全局模型变量
model = None
ocr_detector = None

# ===== 1. 简化模型加载 =====
def init_model(model_path: Optional[str] = None):
    """简化版模型加载，直接使用标准FLUX模型"""
    global model
    
    logger.info("正在加载 FLUX Kontext 模型...")
    
    # 确定模型路径
    if model_path and os.path.exists(model_path):
        logger.info(f"使用本地模型: {model_path}")
        model_name_or_path = model_path
    else:
        logger.info("使用标准在线模型: black-forest-labs/FLUX.1-Kontext-dev")
        model_name_or_path = "black-forest-labs/FLUX.1-Kontext-dev"
    
    try:
        # 直接加载标准版本，使用简化配置
        pipeline = FluxKontextPipeline.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="balanced"  # 自动设备映射
        )
        
        model = pipeline
        logger.info("模型加载完成")
        return pipeline
        
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        raise e

def init_ocr_detector():
    """初始化OCR检测器"""
    global ocr_detector
    
    if ocr_detector is None:
        logger.info("正在初始化EasyOCR检测器...")
        try:
            ocr_detector = EasyOCRDetector(languages=['en', 'ch_sim'], gpu=torch.cuda.is_available(), verbose=False)
            logger.info("OCR检测器初始化完成")
        except Exception as e:
            logger.error(f"OCR检测器初始化失败: {str(e)}")
            ocr_detector = None
    
    return ocr_detector

# ===== 2. 核心推理函数 =====
def remove_watermark(image: Image.Image, prompt: str = "Remove watermark") -> Image.Image:
    """图像水印擦除"""
    global model
    
    if model is None:
        raise ValueError("模型未初始化，请先调用 init_model()")
    
    try:
        # 自动检测设备类型
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        output = model(
            image=image,
            prompt=f"{prompt} | Keep original details, remove watermark only",
            guidance_scale=2.5,
            num_inference_steps=20,
            generator=torch.Generator(device=device).manual_seed(42)
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
        # 自动检测设备类型
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        output = model(
            image=image,
            prompt=prompt,
            guidance_scale=3.0,
            num_inference_steps=25,
            generator=torch.Generator(device=device).manual_seed(42)
        ).images[0]
        return output
    except Exception as e:
        logger.error(f"图像编辑失败: {str(e)}")
        raise e

def detect_and_remove_text(image: Image.Image, text_removal_method: str = "flux") -> Image.Image:
    """检测并移除图像中的文字区域
    
    Args:
        image: 输入图像
        text_removal_method: 文字移除方法 ("flux", "iopaint", "lama")
    """
    global model, ocr_detector
    
    if ocr_detector is None:
        init_ocr_detector()
    
    if ocr_detector is None:
        logger.warning("OCR检测器未初始化，跳过文字检测")
        return image
    
    try:
        # 使用OCR检测文字区域
        text_mask = ocr_detector.generate_text_mask(image)
        
        # 检查是否检测到文字
        if text_mask is None or np.sum(text_mask) == 0:
            logger.debug("未检测到文字区域")
            return image
        
        # 计算文字区域占比
        text_ratio = np.sum(text_mask > 0) / (text_mask.shape[0] * text_mask.shape[1])
        logger.debug(f"检测到文字区域占比: {text_ratio:.3f}")
        
        # 如果文字区域占比太小，跳过处理
        if text_ratio < 0.001 or text_ratio > 0.5:  # 0.1%
            logger.debug("文字区域太小或者太大，跳过处理")
            return image
        
        # 根据选择的方法处理文字区域
        if text_removal_method.lower() == "flux":
            # 使用FLUX模型移除文字
            if model is None:
                raise ValueError("FLUX模型未初始化，请先调用 init_model()")
            
            logger.debug("正在使用FLUX模型移除检测到的文字区域...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            processed_image = model(
                image=image,
                prompt="Remove all text, characters, letters, numbers, and symbols from the image. Keep all other details intact.",
                guidance_scale=3.0,
                num_inference_steps=25,
                generator=torch.Generator(device=device).manual_seed(42)
            ).images[0]
            
            logger.debug("FLUX文字移除完成")
            return processed_image
            
        elif text_removal_method.lower() in ["mat", "lama"]:
            # 使用IOPaint+LAMA模型移除文字
            logger.debug(f"正在使用{text_removal_method.upper()}模型移除检测到的文字区域...")
            
            # 导入IOPaint相关模块
            try:
                from iopaint.batch_processing import batch_inpaint
                import tempfile
                import shutil
                from pathlib import Path
            except ImportError as e:
                logger.error(f"无法导入IOPaint模块: {str(e)}，回退到FLUX模型")
                return detect_and_remove_text(image, "flux")
            
            # 创建临时目录
            temp_dir = tempfile.mkdtemp(prefix='flux_text_removal_')
            try:
                # 保存输入图像
                temp_image_path = os.path.join(temp_dir, "input.png")
                image.save(temp_image_path)
                
                # 保存文字mask
                temp_mask_path = os.path.join(temp_dir, "mask.png")
                # 将numpy数组转换为PIL图像
                mask_image = Image.fromarray((text_mask * 255).astype(np.uint8))
                mask_image.save(temp_mask_path)
                
                # 创建输出目录
                temp_output_dir = os.path.join(temp_dir, "output")
                os.makedirs(temp_output_dir, exist_ok=True)
                
                # 使用IOPaint处理
                model_name = "lama" if text_removal_method.lower() == "lama" else "mat"
                batch_inpaint(
                    model=model_name,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    image=Path(temp_dir),
                    mask=Path(temp_dir),
                    output=Path(temp_output_dir)
                )
                
                # 读取处理结果
                output_image_path = os.path.join(temp_output_dir, "input.png")
                if os.path.exists(output_image_path):
                    processed_image = Image.open(output_image_path).convert('RGB')
                    logger.debug(f"{text_removal_method.upper()}文字移除完成")
                    return processed_image
                else:
                    logger.warning(f"{text_removal_method.upper()}处理失败，未生成输出文件，返回原图")
                    return image
                    
            finally:
                # 清理临时目录
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    logger.warning(f"清理临时目录失败: {str(e)}")
        
        else:
            logger.warning(f"不支持的文字移除方法: {text_removal_method}，使用FLUX模型")
            return detect_and_remove_text(image, "flux")
        
    except Exception as e:
        logger.error(f"文字检测和移除失败: {str(e)}")
        return image

# ===== 3. 简化图像尺寸处理 =====
def resize_image_for_flux(image: Image.Image) -> Image.Image:
    """简化的图像尺寸调整"""
    width, height = image.size
    
    # 简单的尺寸调整逻辑
    max_size = 1024
    min_size = 512
    
    # 如果图像太大，按比例缩小
    if max(width, height) > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * max_size / width)
        else:
            new_height = max_size
            new_width = int(width * max_size / height)
    # 如果图像太小，按比例放大
    elif max(width, height) < min_size:
        if width > height:
            new_width = min_size
            new_height = int(height * min_size / width)
        else:
            new_height = min_size
            new_width = int(width * min_size / height)
    else:
        new_width, new_height = width, height
    
    # 确保尺寸是8的倍数
    new_width = ((new_width + 7) // 8) * 8
    new_height = ((new_height + 7) // 8) * 8
    
    if new_width != width or new_height != height:
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        logger.debug(f"图像尺寸调整: {width}x{height} -> {new_width}x{new_height}")
    
    return image

# ===== 4. 批量处理功能 =====
def get_image_files(input_dir: str, output_dir: str = None, limit: Optional[int] = None, random_select: bool = False) -> List[str]:
    """获取图像文件列表，跳过已处理的文件"""
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    image_files = []
    for file_path in Path(input_dir).iterdir():
        if file_path.suffix.lower() in supported_formats:
            image_files.append(str(file_path))
    
    logger.info(f"找到 {len(image_files)} 张图片")
    
    # 如果指定了输出目录，过滤掉已处理的文件
    if output_dir and os.path.exists(output_dir):
        unprocessed_files = []
        for image_path in image_files:
            filename = Path(image_path).name
            output_path = os.path.join(output_dir, filename)
            if not os.path.exists(output_path):
                unprocessed_files.append(image_path)
        
        processed_count = len(image_files) - len(unprocessed_files)
        if processed_count > 0:
            logger.info(f"跳过 {processed_count} 张已处理的图片")
        
        image_files = unprocessed_files
        logger.info(f"剩余 {len(image_files)} 张未处理的图片")
    
    # 随机选择
    if random_select and limit and len(image_files) > limit:
        image_files = random.sample(image_files, limit)
        logger.info(f"从未处理的图片中随机选择了 {len(image_files)} 张")
    elif limit and len(image_files) > limit:
        image_files = image_files[:limit]
        logger.info(f"选择前 {len(image_files)} 张未处理的图片")
    
    return image_files

def process_batch(input_dir: str, output_dir: str, prompt: str = "Remove watermark", 
                 limit: Optional[int] = None, random_select: bool = False, 
                 task_type: str = "watermark", model_path: Optional[str] = None,
                 enable_text_detection: bool = True, text_removal_method: str = "flux") -> List[tuple]:
    """批量处理图像"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取图像文件
    image_files = get_image_files(input_dir, output_dir, limit, random_select)
    
    if not image_files:
        logger.warning("未找到任何图像文件")
        return []
    
    # 初始化模型
    if model is None:
        init_model(model_path)
    
    processed_pairs = []
    failed_count = 0
    
    # 处理图像
    progress_bar = tqdm(image_files, desc="处理图像", unit="张")
    
    for image_path in progress_bar:
        try:
            # 更新进度条描述
            filename = Path(image_path).name
            progress_bar.set_postfix({"当前": filename[:20] + "..." if len(filename) > 20 else filename})
            
            # 加载图像并调整尺寸
            image = Image.open(image_path).convert('RGB')
            image = resize_image_for_flux(image)
            
            # 处理图像
            if task_type == "watermark":
                processed_image = remove_watermark(image, prompt)
            else:
                processed_image = edit_image(image, prompt)
            
            # 检测并移除文字区域（如果启用）
            if enable_text_detection:
                processed_image = detect_and_remove_text(processed_image, text_removal_method)
            
            # 保存处理后的图像
            output_filename = f"{Path(image_path).stem}{Path(image_path).suffix}"
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

# ===== 5. 视频生成功能 =====
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

# ===== 6. 命令行接口 =====
def main():
    parser = argparse.ArgumentParser(description="FLUX Kontext 简化版批量图像处理工具")
    
    # 基本参数
    parser.add_argument("--input", "-i", default='data/test', help="输入图像目录")
    parser.add_argument("--output", "-o", default='data/res', help="输出图像目录")
    parser.add_argument("--prompt", "-p", default="Remove watermarks, logos, and labels from the image. Be careful not to modify the details of the parts themselves, and do not lose small parts. If there are small labels on the parts themselves, erase them and keep their screen printing. If there is a car in the image, please cover the car logo.", help="处理提示词")
    parser.add_argument("--model-path", "-m", help="本地FLUX模型目录路径 (可选，不指定则使用在线模型)")
    
    # 处理选项
    parser.add_argument("--limit", "-l", type=int, help="限制处理图像数量")
    parser.add_argument("--random", "-r", action="store_true", help="随机选择图像")
    parser.add_argument("--task", "-t", choices=["watermark", "edit"], default="watermark", 
                       help="任务类型: watermark(去水印) 或 edit(通用编辑)")
    parser.add_argument("--enable-text-detection", action="store_true", default=True, 
                       help="启用文字检测和移除功能")
    parser.add_argument("--text-removal-method", choices=["flux", "iopaint", "lama"], default="lama",
                       help="文字移除方法: flux(FLUX模型), iopaint(IOPaint), lama(LAMA模型)")
    
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
    
    # 处理文字检测参数
    enable_text_detection = args.enable_text_detection
    text_removal_method = args.text_removal_method
    
    try:
        # 批量处理图像
        logger.info(f"开始批量处理: {args.input} -> {args.output}")
        logger.info(f"任务类型: {args.task}, 提示词: {args.prompt}")
        logger.info(f"文字检测功能: {'启用' if enable_text_detection else '禁用'}")
        if enable_text_detection:
            logger.info(f"文字移除方法: {text_removal_method.upper()}")
        
        processed_pairs = process_batch(
            input_dir=args.input,
            output_dir=args.output,
            prompt=args.prompt,
            limit=args.limit,
            random_select=args.random,
            task_type=args.task,
            model_path=args.model_path,
            enable_text_detection=enable_text_detection,
            text_removal_method=text_removal_method
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











































































































































































































































































































































































































