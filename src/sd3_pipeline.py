# -*- coding: utf-8 -*-
"""
三步水印修复脚本：1. UNet掩码预测 2. Lama修复 3. Stable Diffusion 3.5 Medium文生图修复
"""
import os
import sys
import shutil
import tempfile
import torch
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import subprocess
import logging

# 第三步依赖
from diffusers import StableDiffusionPipeline

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

def run_sd3(image_path, prompt, output_path, device='cuda'):
    logger.info(f"加载Stable Diffusion 3.5 Medium模型...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16
    )
    pipe = pipe.to(device)
    logger.info(f"调用SD3.5生成修复图像: {image_path}")
    image = Image.open(image_path).convert('RGB')
    result = pipe(prompt=prompt, image=image, num_inference_steps=40, guidance_scale=4.5)
    out_img = result.images[0]
    out_img.save(output_path)
    logger.info(f"SD3.5修复完成: {output_path}")

def main(unet_model_path, input_folder, output_folder, prompt, config_path=None, device='cuda'):
    os.makedirs(output_folder, exist_ok=True)
    cfg = get_cfg_defaults()
    if config_path and os.path.exists(config_path):
        update_config(cfg, config_path)
    model = create_model_from_config(cfg).to(device)
    checkpoint = torch.load(unet_model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    transform = get_val_transform(cfg.DATA.IMG_SIZE)
    temp_dir = tempfile.mkdtemp(prefix="sd3_pipeline_")
    try:
        image_files = list(Path(input_folder).glob("*.jpg")) + list(Path(input_folder).glob("*.png"))
        for img_path in tqdm(image_files, desc="处理图片"):
            stem = img_path.stem
            mask_path = os.path.join(temp_dir, f"{stem}_mask.png")
            lama_out_path = os.path.join(temp_dir, f"{stem}_lama.png")
            sd3_out_path = os.path.join(output_folder, f"{stem}_sd3.png")
            # Step 1: 掩码预测
            mask = predict_mask(model, transform, device, str(img_path), cfg.DATA.IMG_SIZE)
            Image.fromarray(mask).save(mask_path)
            # Step 2: Lama修复
            run_lama(str(img_path), mask_path, lama_out_path, device=device)
            # Step 3: Stable Diffusion修复
            run_sd3(lama_out_path, prompt, sd3_out_path, device=device)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="三步水印修复脚本：UNet+Lama+StableDiffusion3.5")
    parser.add_argument('--unet_model', type=str, default='models/unet_watermark.pth', help='UNet模型路径')
    parser.add_argument('--input', type=str, default='data/test', help='输入图片文件夹')
    parser.add_argument('--output', type=str, default='data/result', help='输出文件夹')
    parser.add_argument('--prompt', type=str, default='Remove watermarks and textual information from images without altering the subject', help='Stable Diffusion修复prompt')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--device', type=str, default='cpu', help='推理设备')
    args = parser.parse_args()
    main(args.unet_model, args.input, args.output, args.prompt, args.config, args.device)