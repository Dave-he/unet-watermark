#!/usr/bin/env python3
import sys
sys.path.append('./src')

from configs.config import get_cfg_defaults
from models.unet_model import SMPModelFactory
import torch

def get_model_size(model):
    """计算模型大小"""
    total_params = sum(p.numel() for p in model.parameters())
    size_mb = total_params * 4 / 1024 / 1024
    return total_params, size_mb

print("UNet模型大小对比分析\n" + "="*50)

# 当前配置
print("\n1. 当前配置 (UNet++ + ResNet34):")
cfg = get_cfg_defaults()
cfg.merge_from_file('./src/configs/unet_watermark.yaml')
model = SMPModelFactory.create_model(
    model_name=cfg.MODEL.NAME,
    encoder_name=cfg.MODEL.ENCODER_NAME,
    encoder_weights=cfg.MODEL.ENCODER_WEIGHTS,
    in_channels=cfg.MODEL.IN_CHANNELS,
    classes=cfg.MODEL.CLASSES,
    decoder_channels=cfg.MODEL.DECODER_CHANNELS
)
params, size = get_model_size(model)
print(f"   参数数量: {params:,}")
print(f"   模型大小: {size:.2f} MB")

# 不同编码器对比
print("\n2. 不同编码器对比 (UNet++ 架构):")
encoders = [
    ('resnet18', '更轻量'),
    ('resnet34', '当前使用'),
    ('resnet50', '更重'),
    ('efficientnet-b0', '高效网络'),
    ('efficientnet-b3', '更大的高效网络')
]

for encoder, desc in encoders:
    try:
        model = SMPModelFactory.create_model(
            model_name='UnetPlusPlus',
            encoder_name=encoder,
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
            decoder_channels=[256, 128, 64, 32, 16]
        )
        params, size = get_model_size(model)
        print(f"   {encoder:20} ({desc:10}): {params:>10,} 参数, {size:>6.2f} MB")
    except Exception as e:
        print(f"   {encoder:20} ({desc:10}): 不支持")

# 不同架构对比
print("\n3. 不同架构对比 (ResNet34 编码器):")
architectures = [
    ('Unet', '标准UNet'),
    ('UnetPlusPlus', '当前使用'),
    ('MAnet', 'Multi-Attention'),
    ('Linknet', '轻量级'),
    ('FPN', 'Feature Pyramid'),
    ('DeepLabV3Plus', 'DeepLab系列')
]

for arch, desc in architectures:
    try:
        model = SMPModelFactory.create_model(
            model_name=arch,
            encoder_name='resnet34',
            encoder_weights='imagenet',
            in_channels=3,
            classes=1
        )
        params, size = get_model_size(model)
        print(f"   {arch:15} ({desc:15}): {params:>10,} 参数, {size:>6.2f} MB")
    except Exception as e:
        print(f"   {arch:15} ({desc:15}): 创建失败")

# 解码器通道数影响
print("\n4. 解码器通道数影响 (UNet++ + ResNet34):")
decoder_configs = [
    ([128, 64, 32, 16, 8], '更轻量'),
    ([256, 128, 64, 32, 16], '当前配置'),
    ([512, 256, 128, 64, 32], '更重'),
    ([1024, 512, 256, 128, 64], '最重')
]

for channels, desc in decoder_configs:
    try:
        model = SMPModelFactory.create_model(
            model_name='UnetPlusPlus',
            encoder_name='resnet34',
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
            decoder_channels=channels
        )
        params, size = get_model_size(model)
        print(f"   {str(channels):30} ({desc:8}): {params:>10,} 参数, {size:>6.2f} MB")
    except Exception as e:
        print(f"   {str(channels):30} ({desc:8}): 创建失败")

print("\n" + "="*50)
print("总结:")
print("- 当前模型约100MB是合理的，因为使用了相对轻量的配置")
print("- ResNet34编码器比ResNet50等更轻量")
print("- UNet++比标准UNet稍重，但提供更好的特征融合")
print("- 解码器通道数[256,128,64,32,16]是平衡性能和大小的选择")
print("- 如需更小模型，可考虑ResNet18编码器或Linknet架构")
print("- 如需更大模型，可考虑ResNet50编码器或增加解码器通道数")