#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试模糊mask功能的简单脚本
"""

import sys
import os
sys.path.append('src')

from utils.dataset import WatermarkDataset, create_datasets
from configs.config import get_cfg_defaults
import cv2
import numpy as np

def test_blurred_mask():
    """测试模糊mask功能"""
    print("测试模糊mask功能...")
    
    # 创建测试配置
    cfg = get_cfg_defaults()
    cfg.defrost()
    cfg.DATA.ROOT_DIR = "data/train"
    cfg.DATA.IMG_SIZE = 512
    cfg.DATA.GENERATE_MASK_THRESHOLD = 30
    cfg.freeze()
    
    print("\n1. 测试精确mask（默认）:")
    try:
        train_dataset, val_dataset = create_datasets(cfg, use_blurred_mask=False)
        print("✓ 精确mask数据集创建成功")
    except Exception as e:
        print(f"✗ 精确mask数据集创建失败: {e}")
    
    print("\n2. 测试模糊mask:")
    try:
        train_dataset_blur, val_dataset_blur = create_datasets(cfg, use_blurred_mask=True)
        print("✓ 模糊mask数据集创建成功")
    except Exception as e:
        print(f"✗ 模糊mask数据集创建失败: {e}")
    
    print("\n3. 测试WatermarkDataset直接初始化:")
    try:
        # 测试精确mask
        dataset_precise = WatermarkDataset(
            watermarked_dirs=["data/train/watermarked"],
            clean_dirs=["data/train/clean"],
            mask_dirs=["data/train/masks"],
            use_blurred_mask=False
        )
        print(f"✓ 精确mask数据集: {len(dataset_precise)} 张图像")
        
        # 测试模糊mask
        dataset_blurred = WatermarkDataset(
            watermarked_dirs=["data/train/watermarked"],
            clean_dirs=["data/train/clean"],
            mask_dirs=["data/train/masks"],
            use_blurred_mask=True
        )
        print(f"✓ 模糊mask数据集: {len(dataset_blurred)} 张图像")
        
    except Exception as e:
        print(f"✗ WatermarkDataset初始化失败: {e}")
    
    print("\n测试完成！")

if __name__ == "__main__":
    test_blurred_mask()