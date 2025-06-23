#!/usr/bin/env python3
"""
Stable Diffusion 3 Medium Inpainting Script
使用 Stable Diffusion 3 Medium 模型批量去除图片中的水印和文字
"""

import os
import argparse
from pathlib import Path
import torch
from PIL import Image, ImageDraw
import numpy as np
from diffusers import StableDiffusion3InpaintPipeline
import cv2
import random
from typing import List, Tuple

class WatermarkRemover:
    def __init__(self, model_name: str = "stabilityai/stable-diffusion-3-medium-diffusers", device: str = "auto"):
        """
        初始化水印去除器
        
        Args:
            model_name: 模型名称
            device: 设备类型 (auto, cuda, mps, cpu)
        """
        self.device = self._get_device(device)
        print(f"使用设备: {self.device}")
        
        # 加载 Stable Diffusion 3 Medium Inpainting Pipeline
        print("正在加载 Stable Diffusion 3 Medium 模型...")
        try:
            # 首先尝试使用 balanced device_map
            if self.device.type == "cuda":
                try:
                    self.pipe = StableDiffusion3InpaintPipeline.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16,
                        device_map="balanced"
                    )
                except Exception as device_map_error:
                    print(f"使用 device_map='balanced' 失败: {device_map_error}")
                    print("尝试不使用 device_map...")
                    # 回退到不使用 device_map
                    self.pipe = StableDiffusion3InpaintPipeline.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16
                    ).to(self.device)
            else:
                self.pipe = StableDiffusion3InpaintPipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32
                ).to(self.device)
            
            # 启用内存优化
            if self.device.type == "cuda":
                try:
                    self.pipe.enable_model_cpu_offload()
                    self.pipe.enable_attention_slicing()
                except Exception as opt_error:
                    print(f"内存优化启用失败: {opt_error}")
            
            print("模型加载完成!")
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            print("尝试使用 CPU 模式...")
            self.device = torch.device("cpu")
            self.pipe = StableDiffusion3InpaintPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float32
            ).to(self.device)
    
    def _get_device(self, device: str) -> torch.device:
        """获取可用设备"""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device)
    
    def detect_text_regions(self, image: Image.Image, threshold: float = 0.5, 
                           max_mask_ratio: float = 0.3, min_area: int = 200, 
                           max_area_ratio: float = 0.1) -> Image.Image:
        """
        使用优化的文字检测方法创建遮罩，避免大面积遮罩
        
        Args:
            image: 输入图像
            threshold: 检测阈值
            max_mask_ratio: 最大遮罩面积占图像面积的比例
            min_area: 最小检测区域面积
            max_area_ratio: 单个区域最大面积占图像面积的比例
            
        Returns:
            遮罩图像
        """
        # 转换为 OpenCV 格式
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # 图像总面积
        total_area = gray.shape[0] * gray.shape[1]
        max_single_area = int(total_area * max_area_ratio)
        
        # 使用更保守的形态学操作检测文字区域
        # 创建更小的结构元素
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        
        # 应用形态学梯度
        gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        
        # 提高二值化阈值，减少误检测
        _, binary = cv2.threshold(gradient, 80, 255, cv2.THRESH_BINARY)
        
        # 使用更小的核进行连接操作
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 1))
        connected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel2)
        
        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 6))
        connected = cv2.morphologyEx(connected, cv2.MORPH_CLOSE, kernel3)
        
        # 查找轮廓
        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 创建遮罩
        mask = np.zeros_like(gray)
        valid_regions = []
        total_mask_area = 0
        
        # 按面积排序轮廓，优先处理小的区域
        contours = sorted(contours, key=cv2.contourArea)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # 过滤过小和过大的轮廓
            if area < min_area or area > max_single_area:
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            
            # 检查长宽比，过滤非文字形状
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio > 10 or aspect_ratio < 0.1:  # 过于细长或过于扁平
                continue
            
            # 减少边界框扩展，避免遮罩过大
            padding = min(3, min(w, h) // 4)  # 动态调整padding
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(mask.shape[1] - x, w + 2 * padding)
            h = min(mask.shape[0] - y, h + 2 * padding)
            
            region_area = w * h
            
            # 检查总遮罩面积是否超过限制
            if total_mask_area + region_area > total_area * max_mask_ratio:
                print(f"警告: 遮罩面积过大，跳过剩余区域。当前遮罩占比: {total_mask_area/total_area:.2%}")
                break
                
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
            valid_regions.append((x, y, w, h))
            total_mask_area += region_area
        
        # 输出统计信息
        mask_ratio = total_mask_area / total_area
        print(f"检测到 {len(valid_regions)} 个文字区域，遮罩面积占比: {mask_ratio:.2%}")
        
        if mask_ratio > 0.2:
            print(f"警告: 遮罩面积较大 ({mask_ratio:.2%})，可能影响生成质量")
        
        # 转换回 PIL 图像
        mask_image = Image.fromarray(mask).convert("RGB")
        return mask_image
    
    def create_manual_mask(self, image: Image.Image, regions: List[Tuple[int, int, int, int]]) -> Image.Image:
        """
        手动创建遮罩
        
        Args:
            image: 输入图像
            regions: 区域列表 [(x1, y1, x2, y2), ...]
            
        Returns:
            遮罩图像
        """
        mask = Image.new("RGB", image.size, (0, 0, 0))
        draw = ImageDraw.Draw(mask)
        
        for x1, y1, x2, y2 in regions:
            draw.rectangle([x1, y1, x2, y2], fill=(255, 255, 255))
        
        return mask
    
    def remove_watermark_with_mask(self, image: Image.Image, mask: Image.Image, prompt: str, 
                                  negative_prompt: str = "", num_inference_steps: int = 25, 
                                  guidance_scale: float = 6.0, strength: float = 0.6) -> Image.Image:
        """
        使用给定遮罩进行 inpainting 去除水印，优化参数以减少重新创造内容
        
        Args:
            image: 输入图像
            mask: 遮罩图像
            prompt: 提示词
            negative_prompt: 负面提示词
            num_inference_steps: 推理步数，增加以提高质量
            guidance_scale: 引导强度，降低以减少过度生成
            strength: 强度，降低以保持原图内容
            
        Returns:
            处理后的图像
        """
        try:
            # 检查遮罩是否为空
            mask_array = np.array(mask.convert('L'))
            if np.sum(mask_array) == 0:
                print("未检测到文字区域，返回原图")
                return image
            
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                mask_image=mask,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength,
                generator=torch.Generator(device=self.device).manual_seed(42)
            ).images[0]
            
            return result
            
        except Exception as e:
            print(f"图像处理失败: {e}")
            return image
    
    def remove_watermark_auto(self, image: Image.Image, prompt: str = "clean image without watermark", 
                             strength: float = 0.6, guidance_scale: float = 6.0, 
                             num_inference_steps: int = 25, max_mask_ratio: float = 0.25,
                             min_area: int = 200, max_area_ratio: float = 0.08,
                             negative_prompt: str = "blurry, low quality, distorted") -> Image.Image:
        """
        自动检测并去除图像中的水印，优化参数以减少重新创造内容
        
        Args:
            image: 输入图像
            prompt: 生成提示词
            strength: 生成强度 (0.0-1.0)，降低以保持原图内容
            guidance_scale: 引导尺度，降低以减少过度生成
            num_inference_steps: 推理步数，增加以提高质量
            max_mask_ratio: 最大遮罩面积占图像面积的比例
            min_area: 最小检测区域面积
            max_area_ratio: 单个区域最大面积占图像面积的比例
            negative_prompt: 负面提示词
            
        Returns:
            处理后的图像
        """
        try:
            # 检测文字区域并创建遮罩，使用优化参数
            mask = self.detect_text_regions(image, max_mask_ratio=max_mask_ratio,
                                           min_area=min_area, max_area_ratio=max_area_ratio)
            
            # 使用遮罩进行inpainting
            return self.remove_watermark_with_mask(
                image=image,
                mask=mask,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength
            )
            
        except Exception as e:
            print(f"处理图像时出错: {e}")
            return image
    
    def process_single_image(self, input_path: str, output_path: str, prompt: str,
                           auto_detect: bool = True, manual_regions: List[Tuple[int, int, int, int]] = None,
                           strength: float = 0.6, guidance_scale: float = 6.0,
                           num_inference_steps: int = 25, max_mask_ratio: float = 0.25,
                           min_area: int = 200, max_area_ratio: float = 0.08,
                           **kwargs) -> bool:
        """
        处理单张图像
        
        Args:
            input_path: 输入图像路径
            output_path: 输出图像路径
            prompt: 提示词
            auto_detect: 是否自动检测文字区域
            manual_regions: 手动指定的区域
            **kwargs: 其他参数
            
        Returns:
            是否成功
        """
        try:
            # 加载图像
            image = Image.open(input_path).convert("RGB")
            print(f"处理图像: {input_path} (尺寸: {image.size})")
            
            # 去除水印
            if auto_detect:
                # 使用自动检测方法，包含优化的遮罩控制
                result = self.remove_watermark_auto(
                    image=image,
                    prompt=prompt,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    max_mask_ratio=max_mask_ratio,
                    min_area=min_area,
                    max_area_ratio=max_area_ratio,
                    **kwargs
                )
                
                # 保存调试遮罩
                mask = self.detect_text_regions(image, max_mask_ratio=max_mask_ratio,
                                               min_area=min_area, max_area_ratio=max_area_ratio)
                mask_path = output_path.replace('.', '_mask.')
                mask.save(mask_path)
                print(f"调试遮罩已保存: {mask_path}")
                
            elif manual_regions:
                # 使用手动指定区域
                mask = self.create_manual_mask(image, manual_regions)
                result = self.remove_watermark_with_mask(
                    image=image,
                    mask=mask,
                    prompt=prompt,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    **kwargs
                )
                
                # 保存遮罩用于调试
                mask_path = output_path.replace('.', '_mask.')
                mask.save(mask_path)
                print(f"手动遮罩已保存: {mask_path}")
                
            else:
                print("警告: 未指定遮罩创建方法，跳过此图像")
                return False
            
            # 保存结果
            result.save(output_path, quality=95)
            print(f"结果已保存: {output_path}")
            
            return True
            
        except Exception as e:
            print(f"处理图像 {input_path} 时出错: {e}")
            return False
    
    def process_folder(self, input_folder: str, output_folder: str, prompt: str,
                      extensions: List[str] = None, limit: int = None, **kwargs) -> None:
        """
        批量处理文件夹中的图像
        
        Args:
            input_folder: 输入文件夹路径
            output_folder: 输出文件夹路径
            prompt: 提示词
            extensions: 支持的文件扩展名
            limit: 限制处理的图像数量，随机选择
            **kwargs: 其他参数
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        
        # 创建输出文件夹
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 获取所有图像文件
        image_files = []
        for ext in extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"在 {input_folder} 中未找到图像文件")
            return
        
        print(f"找到 {len(image_files)} 个图像文件")
        
        # 如果设置了 limit 参数，随机选择指定数量的图像
        if limit is not None and limit > 0:
            if limit < len(image_files):
                image_files = random.sample(image_files, limit)
                print(f"随机选择了 {len(image_files)} 个图像进行处理")
            else:
                print(f"limit 参数 ({limit}) 大于等于总图像数量，将处理所有图像")
        
        # 处理每个图像
        success_count = 0
        for i, img_file in enumerate(image_files, 1):
            print(f"\n进度: {i}/{len(image_files)}")
            
            output_file = output_path / f"cleaned_{img_file.name}"
            
            if self.process_single_image(str(img_file), str(output_file), prompt, **kwargs):
                success_count += 1
        
        print(f"\n批量处理完成! 成功处理 {success_count}/{len(image_files)} 个图像")

def main():
    parser = argparse.ArgumentParser(description="使用 Stable Diffusion 3 Medium 去除图片水印")
    parser.add_argument("--input", "-i", type=str, default="test", help="输入文件夹路径")
    parser.add_argument("--output", "-o", type=str, default="output", help="输出文件夹路径")
    parser.add_argument("--prompt", type=str, 
                       default="clean image without watermarks, text, or logos, high quality, natural",
                       help="修复提示词")
    parser.add_argument("--negative_prompt", type=str,
                       default="watermark, text, logo, signature, blurry, low quality, artifacts",
                       help="负面提示词")
    parser.add_argument("--model", type=str, 
                       default="stabilityai/stable-diffusion-3-medium-diffusers",
                       help="模型名称")
    parser.add_argument("--device", type=str, default="auto", 
                       choices=["auto", "cuda", "mps", "cpu"],
                       help="设备类型")
    parser.add_argument("--steps", type=int, default=25, help="推理步数")
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="引导尺度")
    parser.add_argument("--strength", type=float, default=0.6, help="生成强度 (0.0-1.0)")
    parser.add_argument("--max_mask_ratio", type=float, default=0.25, 
                       help="最大遮罩面积占图像面积的比例")
    parser.add_argument("--min_area", type=int, default=200, 
                       help="最小检测区域面积")
    parser.add_argument("--max_area_ratio", type=float, default=0.08, 
                       help="单个区域最大面积占图像面积的比例")
    parser.add_argument("--auto_detect", action="store_true", default=True,
                       help="自动检测文字区域")
    parser.add_argument("--limit", type=int, default=10,
                       help="限制处理的图像数量，从输入文件夹中随机选择指定数量的图像")
    
    args = parser.parse_args()
    
    # 创建水印去除器
    remover = WatermarkRemover(model_name=args.model, device=args.device)
    
    # 批量处理
    remover.process_folder(
        input_folder=args.input,
        output_folder=args.output,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        strength=args.strength,
        max_mask_ratio=args.max_mask_ratio,
        min_area=args.min_area,
        max_area_ratio=args.max_area_ratio,
        auto_detect=args.auto_detect,
        limit=args.limit
    )

if __name__ == "__main__":
    main()