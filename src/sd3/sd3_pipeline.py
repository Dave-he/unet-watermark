# -*- coding: utf-8 -*-
"""
三步水印修复脚本：1. UNet掩码预测 2. Lama修复 3. Stable Diffusion 3.5 Medium文生图修复
"""
import os
import sys
import shutil
import tempfile
import torch
import random
import glob
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import subprocess
import logging

# 第三步依赖将在函数内部导入

from configs.config import get_cfg_defaults, update_config
from models.unet_model import create_model_from_config
from utils.dataset import get_val_transform

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def predict_mask(model, transform, device, image_path, img_size, threshold=0.5):
    import cv2
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transformed = transform(image=image_rgb)
    input_tensor = transformed['image'].unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        if isinstance(output, dict):
            mask = output['out'].cpu().numpy()[0, 0]
        else:
            mask = output.cpu().numpy()[0, 0]
    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
    mask_binary = (mask_resized > threshold).astype('uint8') * 255
    return mask_binary

def run_lama(input_path, mask_path, output_path, model_name='lama', device='cpu'):
    cmd = [
        'iopaint', 'run',
        '--model', model_name,
        '--device', device,
        '--image', input_path,
        '--mask', mask_path,
        '--output', output_path
    ]
    logger.info(f"运行Lama修复: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def run_sd3(image_path, prompt, output_path, device='cuda', strength=0.3):
    # 检查输入路径是否存在
    if not os.path.exists(image_path):
        logger.error(f"输入路径不存在: {image_path}")
        return
    
    # 如果是文件夹，处理文件夹中的所有图片
    if os.path.isdir(image_path):
        logger.info(f"检测到文件夹，处理其中的所有图片: {image_path}")
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
            image_files.extend(glob.glob(os.path.join(image_path, ext)))
            image_files.extend(glob.glob(os.path.join(image_path, ext.upper())))
        
        if not image_files:
            logger.warning(f"文件夹中没有找到图片文件: {image_path}")
            return
            
        # 为文件夹处理创建输出目录
        os.makedirs(output_path, exist_ok=True)
        
        for img_file in image_files:
            img_name = os.path.splitext(os.path.basename(img_file))[0]
            single_output = os.path.join(output_path, f"{img_name}_sd3.png")
            run_sd3(img_file, prompt, single_output, device, strength)
        return
    
    # 处理单个文件
    if not os.path.isfile(image_path):
        logger.error(f"输入路径不是有效文件: {image_path}")
        return
        
    try:
        # 尝试导入SD3 Pipeline (使用text-to-image，因为img2img版本可能不可用)
        from diffusers import StableDiffusion3Pipeline
        logger.info(f"加载Stable Diffusion 3 Medium模型...")
        
        # 根据设备类型决定是否使用float16
        if device == 'cpu':
            pipe = StableDiffusion3Pipeline.from_pretrained(
                "stabilityai/stable-diffusion-3-medium-diffusers"
            )
        else:
            pipe = StableDiffusion3Pipeline.from_pretrained(
                "stabilityai/stable-diffusion-3-medium-diffusers", 
                torch_dtype=torch.float16
            )
        pipe = pipe.to(device)
        
        logger.info(f"基于prompt进行SD3生成: {image_path}")
        
        # 使用text-to-image生成
        result = pipe(
            prompt=prompt, 
            num_inference_steps=28, 
            guidance_scale=7.0
        )
        out_img = result.images[0]
        out_img.save(output_path)
        logger.info(f"SD3修复完成: {output_path}")
        
    except ImportError:
        logger.warning("StableDiffusion3Pipeline不可用，降级使用SD1.5 Img2Img")
        # 降级使用SD1.5 Img2Img Pipeline
        from diffusers import StableDiffusionImg2ImgPipeline
        logger.info(f"加载Stable Diffusion 1.5 Img2Img模型...")
        
        # 根据设备类型决定是否使用float16
        if device == 'cpu':
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5"
            )
        else:
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5", 
                torch_dtype=torch.float16
            )
        pipe = pipe.to(device)
        
        # 加载Lama修复后的图像作为输入
        init_image = Image.open(image_path).convert("RGB")
        logger.info(f"基于Lama修复结果进行SD1.5优化: {image_path}")
        
        # 使用可调节的strength保持原图内容，只进行细节优化
        result = pipe(
            prompt=prompt, 
            image=init_image,
            strength=strength,  # 可调节的strength参数
            num_inference_steps=50, 
            guidance_scale=7.5
        )
        out_img = result.images[0]
        out_img.save(output_path)
        logger.info(f"SD1.5 Img2Img修复完成: {output_path}")

def main(unet_model_path, input_folder, output_folder, prompt, config_path=None, device='cuda', limit=None, strength=0.3):
    os.makedirs(output_folder, exist_ok=True)
    
    # 创建固定的临时目录用于保存中间结果
    temp_dir = os.path.join(output_folder, "tmp")
    os.makedirs(temp_dir, exist_ok=True)
    logger.info(f"中间结果将保存到: {temp_dir}")
    
    cfg = get_cfg_defaults()
    if config_path and os.path.exists(config_path):
        update_config(cfg, config_path)
    
    # 加载自主训练的UNet模型
    logger.info(f"加载自主训练的UNet模型: {unet_model_path}")
    model = create_model_from_config(cfg).to(device)
    checkpoint = torch.load(unet_model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    transform = get_val_transform(cfg.DATA.IMG_SIZE)
    
    image_files = list(Path(input_folder).glob("*.jpg")) + list(Path(input_folder).glob("*.png"))
    
    # 随机选择指定数量的图片
    if limit and limit < len(image_files):
        image_files = random.sample(image_files, limit)
        logger.info(f"随机选择了 {limit} 张图片进行处理")
    
    for img_path in tqdm(image_files, desc="处理图片"):
        stem = img_path.stem
        mask_path = os.path.join(temp_dir, f"{stem}_mask.png")
        lama_out_path = os.path.join(temp_dir, f"{stem}_lama.png")
        sd3_out_path = os.path.join(output_folder, f"{stem}_sd3.png")
        
        # Step 1: UNet掩码预测（使用自主训练的模型）
        logger.info(f"Step 1: 使用自主训练的UNet模型预测掩码: {img_path.name}")
        mask = predict_mask(model, transform, device, str(img_path), cfg.DATA.IMG_SIZE)
        Image.fromarray(mask).save(mask_path)
        logger.info(f"掩码已保存: {mask_path}")
        
        # Step 2: Lama修复
        logger.info(f"Step 2: Lama修复: {img_path.name}")
        run_lama(str(img_path), mask_path, lama_out_path, device=device)
        logger.info(f"Lama修复结果已保存: {lama_out_path}")
        
        # Step 3: Stable Diffusion修复
        logger.info(f"Step 3: Stable Diffusion修复: {img_path.name}")
        run_sd3(lama_out_path, prompt, sd3_out_path, device=device, strength=strength)
        
    logger.info(f"所有处理完成！中间结果保存在: {temp_dir}，最终结果保存在: {output_folder}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="三步水印修复脚本：UNet+Lama+StableDiffusion3.5")
    parser.add_argument('--unet_model', type=str, default='models/unet_watermark.pth', help='UNet模型路径')
    parser.add_argument('--input', type=str, default='data/test', help='输入图片文件夹')
    parser.add_argument('--output', type=str, default='data/result', help='输出文件夹')
    parser.add_argument('--prompt', type=str, default='high quality, detailed, clean image, enhance details, remove artifacts', help='Stable Diffusion修复prompt')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--device', type=str, default='cpu', help='推理设备')
    parser.add_argument('--limit', type=int, default=3, help='随机选择处理的图片数量，不指定则处理所有图片')
    parser.add_argument('--strength', type=float, default=0.3, help='SD修复强度(0.1-0.9)，越低越保持原图，越高修改越大')
    args = parser.parse_args()
    main(args.unet_model, args.input, args.output, args.prompt, args.config, args.device, args.limit, args.strength)