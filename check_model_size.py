#!/usr/bin/env python3
import sys
sys.path.append('./src')

from configs.config import get_cfg_defaults
from models.unet_model import create_model_from_config
import torch

# 加载配置
cfg = get_cfg_defaults()
cfg.merge_from_file('./src/configs/unet_watermark.yaml')

# 创建模型
model = create_model_from_config(cfg)

# 计算参数数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# 输出模型信息
print(f'模型架构: {cfg.MODEL.NAME}')
print(f'编码器: {cfg.MODEL.ENCODER_NAME}')
print(f'编码器权重: {cfg.MODEL.ENCODER_WEIGHTS}')
print(f'编码器深度: {cfg.MODEL.ENCODER_DEPTH}')
print(f'解码器通道: {cfg.MODEL.DECODER_CHANNELS}')
print(f'输入通道: {cfg.MODEL.IN_CHANNELS}')
print(f'输出类别: {cfg.MODEL.CLASSES}')
print(f'\n参数统计:')
print(f'总参数数量: {total_params:,}')
print(f'可训练参数: {trainable_params:,}')
print(f'模型大小 (MB): {total_params * 4 / 1024 / 1024:.2f}')
print(f'\n分析:')
print(f'- 使用的是 {cfg.MODEL.NAME} 架构')
print(f'- 编码器是 {cfg.MODEL.ENCODER_NAME}，这是一个相对轻量的骨干网络')
print(f'- 解码器通道数设置为 {cfg.MODEL.DECODER_CHANNELS}，通道数较少')
print(f'- 这解释了为什么模型只有约110MB的原因')