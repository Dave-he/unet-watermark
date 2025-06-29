#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试OCR修复是否成功
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def test_ocr_with_pil_image():
    """测试OCR是否能正确处理PIL图像"""
    try:
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        from ocr.easy_ocr import EasyOCRDetector
        
        print("创建测试图像...")
        # 创建测试图像
        test_img = Image.new('RGB', (400, 200), color='white')
        draw = ImageDraw.Draw(test_img)
        
        # 绘制测试文字
        try:
            font = ImageFont.load_default()
        except:
            font = None
            
        draw.text((50, 50), "Test Watermark 测试水印", fill='black', font=font)
        draw.text((50, 100), "Hello World 你好世界", fill='red', font=font)
        
        print("初始化OCR检测器...")
        # 测试OCR
        ocr_detector = EasyOCRDetector(verbose=False)
        
        print("生成文字mask...")
        # 直接传递PIL图像
        mask_array = ocr_detector.generate_text_mask(test_img)
        
        print(f"✓ OCR处理成功！mask形状: {mask_array.shape}")
        
        # 转换为PIL图像
        if isinstance(mask_array, np.ndarray):
            import numpy as np
            mask_img = Image.fromarray(mask_array)
            print(f"✓ mask转换成功！PIL图像尺寸: {mask_img.size}")
        
        return True
        
    except Exception as e:
        print(f"✗ OCR测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_text_watermark_generation():
    """测试文字水印生成功能"""
    try:
        from scripts.gen_data import generate_text_watermark
        from PIL import Image
        import tempfile
        import os
        
        print("创建临时测试图像...")
        # 创建临时测试图像
        test_img = Image.new('RGB', (512, 512), color='white')
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            test_img.save(temp_file.name)
            temp_path = temp_file.name
        
        try:
            print("生成文字水印...")
            # 测试文字水印生成
            watermarked_img, mask = generate_text_watermark(
                temp_path,
                enhance_transparent=True,
                use_ocr_mask=True
            )
            
            print(f"✓ 文字水印生成成功！")
            print(f"  - 水印图像尺寸: {watermarked_img.size}")
            print(f"  - mask图像尺寸: {mask.size}")
            
            return True
            
        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
    except Exception as e:
        print(f"✗ 文字水印生成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("开始OCR修复验证测试...\n")
    
    tests = [
        ("OCR PIL图像处理测试", test_ocr_with_pil_image),
        ("文字水印生成测试", test_text_watermark_generation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"{'='*50}")
        print(f"运行测试: {test_name}")
        print(f"{'='*50}")
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"测试 {test_name} 发生异常: {e}")
            results.append((test_name, False))
        
        print()
    
    # 输出测试结果
    print(f"{'='*50}")
    print("测试结果汇总")
    print(f"{'='*50}")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！OCR修复成功！")
        return True
    else:
        print(f"⚠️  有 {total - passed} 个测试失败，请检查相关功能")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)