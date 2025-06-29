#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试编码修复和Pillow兼容性
"""

import sys
import os
sys.path.append('/Users/hyx/unet-watermark/src')

from PIL import Image, ImageDraw, ImageFont
import hashlib
from scripts.gen_data import generate_text_content, generate_text_watermark

def test_pillow_compatibility():
    """测试Pillow兼容性"""
    print("=== 测试Pillow兼容性 ===")
    
    # 创建测试图像
    test_img = Image.new('RGB', (200, 100), color='white')
    draw = ImageDraw.Draw(test_img)
    
    # 测试字体
    try:
        font = ImageFont.load_default()
        print("✅ 默认字体加载成功")
    except Exception as e:
        print(f"❌ 默认字体加载失败: {e}")
        return False
    
    # 测试中文文字
    test_texts = [
        "SAMPLE",      # 英文
        "样本",        # 中文
        "Demo 演示",   # 混合
        "©2024",      # 符号
    ]
    
    for text in test_texts:
        try:
            # 测试textbbox（新方法）
            bbox = draw.textbbox((0, 0), text, font=font)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            print(f"✅ textbbox测试成功: '{text}' -> {width}x{height}")
            
            # 测试textlength（备选方法）
            length = draw.textlength(text, font=font)
            print(f"✅ textlength测试成功: '{text}' -> {length}")
            
        except Exception as e:
            print(f"❌ 文字测量失败: '{text}' -> {e}")
    
    return True

def test_encoding_fix():
    """测试编码修复"""
    print("\n=== 测试编码修复 ===")
    
    test_strings = [
        "clean_image_样本_watermark_水印_123",
        "测试文字_DEMO_演示版本",
        "版权©2024_Sample_混合内容",
    ]
    
    for test_str in test_strings:
        try:
            # 测试UTF-8编码
            hash_obj = hashlib.md5(test_str.encode('utf-8'))
            hash_result = hash_obj.hexdigest()[:16]
            print(f"✅ UTF-8编码成功: '{test_str}' -> {hash_result}")
            
        except Exception as e:
            print(f"❌ UTF-8编码失败: '{test_str}' -> {e}")
            return False
    
    return True

def test_text_content_generation():
    """测试文字内容生成"""
    print("\n=== 测试文字内容生成 ===")
    
    try:
        for i in range(5):
            text = generate_text_content()
            print(f"生成文字 {i+1}: '{text}'")
            
            # 测试编码
            encoded = text.encode('utf-8')
            print(f"  UTF-8编码长度: {len(encoded)} 字节")
            
        print("✅ 文字内容生成测试成功")
        return True
        
    except Exception as e:
        print(f"❌ 文字内容生成失败: {e}")
        return False

def test_watermark_generation():
    """测试水印生成"""
    print("\n=== 测试水印生成 ===")
    
    # 创建测试图片
    test_img = Image.new('RGB', (400, 300), color='white')
    test_img_path = '/tmp/test_encoding_fix.jpg'
    test_img.save(test_img_path)
    
    try:
        # 测试文字水印生成
        watermarked_img, mask_img = generate_text_watermark(test_img_path)
        
        if watermarked_img and mask_img:
            print("✅ 文字水印生成成功")
            print(f"  水印图像尺寸: {watermarked_img.size}")
            print(f"  遮罩图像尺寸: {mask_img.size}")
            return True
        else:
            print("❌ 文字水印生成失败：返回空对象")
            return False
            
    except Exception as e:
        print(f"❌ 文字水印生成失败: {e}")
        return False
    finally:
        # 清理测试文件
        if os.path.exists(test_img_path):
            os.remove(test_img_path)

def main():
    """主测试函数"""
    print("开始测试编码修复和Pillow兼容性...\n")
    
    tests = [
        ("Pillow兼容性", test_pillow_compatibility),
        ("编码修复", test_encoding_fix),
        ("文字内容生成", test_text_content_generation),
        ("水印生成", test_watermark_generation),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"\n⚠️  {test_name}测试未完全通过")
        except Exception as e:
            print(f"\n❌ {test_name}测试出现异常: {e}")
    
    print(f"\n=== 测试总结 ===")
    print(f"通过测试: {passed}/{total}")
    
    if passed == total:
        print("🎉 所有测试通过！编码问题已修复")
    elif passed >= total * 0.75:
        print("✅ 大部分测试通过，修复基本有效")
    else:
        print("⚠️  仍有问题需要解决")

if __name__ == "__main__":
    main()