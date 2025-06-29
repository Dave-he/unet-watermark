#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文字水印集成测试脚本
测试gen_data.py和auto_train.py中的文字水印功能
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
import logging

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_gen_data_text_watermark():
    """测试gen_data.py的文字水印生成功能"""
    logger.info("测试gen_data.py文字水印生成功能...")
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # 创建测试目录结构
        clean_dir = temp_path / "clean"
        watermarked_dir = temp_path / "watermarked"
        masks_dir = temp_path / "masks"
        
        clean_dir.mkdir()
        watermarked_dir.mkdir()
        masks_dir.mkdir()
        
        # 创建一个简单的测试图片
        from PIL import Image
        test_img = Image.new('RGB', (512, 512), color='white')
        test_img_path = clean_dir / "test.jpg"
        test_img.save(test_img_path)
        
        # 测试文字水印生成
        try:
            from scripts.gen_data import generate_text_watermark
            
            # 生成文字水印
            watermarked_img, mask = generate_text_watermark(
                str(test_img_path),
                enhance_transparent=True,
                use_ocr_mask=True
            )
            
            # 保存结果
            watermarked_path = watermarked_dir / "test_text_watermark.png"
            mask_path = masks_dir / "test_text_watermark.png"
            
            watermarked_img.save(watermarked_path)
            mask.save(mask_path)
            
            logger.info(f"✓ 文字水印生成成功: {watermarked_path}")
            logger.info(f"✓ 文字mask生成成功: {mask_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ 文字水印生成失败: {e}")
            return False

def test_gen_data_mixed_watermark():
    """测试gen_data.py的混合水印生成功能"""
    logger.info("测试gen_data.py混合水印生成功能...")
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # 创建测试目录结构
        clean_dir = temp_path / "clean"
        watermark_dir = temp_path / "watermarks"
        watermarked_dir = temp_path / "watermarked"
        masks_dir = temp_path / "masks"
        
        clean_dir.mkdir()
        watermark_dir.mkdir()
        watermarked_dir.mkdir()
        masks_dir.mkdir()
        
        # 创建测试图片
        from PIL import Image
        test_img = Image.new('RGB', (512, 512), color='white')
        test_img_path = clean_dir / "test.jpg"
        test_img.save(test_img_path)
        
        # 创建测试水印
        watermark_img = Image.new('RGBA', (100, 50), color=(255, 0, 0, 128))
        watermark_path = watermark_dir / "test_watermark.png"
        watermark_img.save(watermark_path)
        
        # 测试混合水印生成
        try:
            from scripts.gen_data import generate_mixed_watermark
            
            # 生成混合水印
            watermarked_img, mask = generate_mixed_watermark(
                str(test_img_path),
                [str(watermark_path)],
                enhance_transparent=True,
                use_ocr_mask=True,
                max_watermarks=1
            )
            
            # 保存结果
            watermarked_path = watermarked_dir / "test_mixed_watermark.png"
            mask_path = masks_dir / "test_mixed_watermark.png"
            
            watermarked_img.save(watermarked_path)
            mask.save(mask_path)
            
            logger.info(f"✓ 混合水印生成成功: {watermarked_path}")
            logger.info(f"✓ 混合mask生成成功: {mask_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ 混合水印生成失败: {e}")
            return False

def test_auto_train_integration():
    """测试auto_train.py的文字水印集成"""
    logger.info("测试auto_train.py文字水印集成...")
    
    try:
        from auto_train import AutoTrainingLoop
        
        # 创建测试配置
        test_config = {
            'text_watermark_ratio': 0.3,
            'mixed_watermark_ratio': 0.2,
            'use_ocr_mask': True,
            'transparent_ratio': 0.6,
            'multi_watermark_ratio': 0.4,
            'max_watermarks': 3
        }
        
        # 创建AutoTrainingLoop实例
        auto_trainer = AutoTrainingLoop(test_config)
        
        logger.info("✓ AutoTrainingLoop初始化成功")
        logger.info(f"✓ 文字水印比例: {test_config['text_watermark_ratio']}")
        logger.info(f"✓ 混合水印比例: {test_config['mixed_watermark_ratio']}")
        logger.info(f"✓ OCR mask启用: {test_config['use_ocr_mask']}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ AutoTrainingLoop集成测试失败: {e}")
        return False

def test_ocr_module():
    """测试OCR模块是否可用"""
    logger.info("测试OCR模块...")
    
    try:
        from utils.ocr_detector import EasyOCRDetector
        
        # 创建OCR检测器
        ocr_detector = EasyOCRDetector()
        
        # 创建测试图片
        from PIL import Image, ImageDraw, ImageFont
        test_img = Image.new('RGB', (300, 100), color='white')
        draw = ImageDraw.Draw(test_img)
        
        # 绘制测试文字
        try:
            # 尝试使用系统字体
            font = ImageFont.load_default()
        except:
            font = None
            
        draw.text((50, 30), "Test Watermark", fill='black', font=font)
        
        # 检测文字区域
        text_regions = ocr_detector.detect_text_regions(test_img)
        
        logger.info(f"✓ OCR模块工作正常，检测到 {len(text_regions)} 个文字区域")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ OCR模块测试失败: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("开始文字水印集成测试...")
    
    tests = [
        ("OCR模块测试", test_ocr_module),
        ("文字水印生成测试", test_gen_data_text_watermark),
        ("混合水印生成测试", test_gen_data_mixed_watermark),
        ("AutoTrain集成测试", test_auto_train_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"运行测试: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"测试 {test_name} 发生异常: {e}")
            results.append((test_name, False))
    
    # 输出测试结果
    logger.info(f"\n{'='*50}")
    logger.info("测试结果汇总")
    logger.info(f"{'='*50}")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\n总计: {passed}/{total} 个测试通过")
    
    if passed == total:
        logger.info("🎉 所有测试通过！文字水印功能集成成功！")
        return True
    else:
        logger.warning(f"⚠️  有 {total - passed} 个测试失败，请检查相关功能")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)