#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试改进后的文字水印生成
"""

import sys
import os
sys.path.append('/Users/hyx/unet-watermark/src')

from PIL import Image
import numpy as np
from scripts.gen_data import generate_text_watermark, generate_text_content, load_system_fonts, select_compatible_font

def test_text_diversity():
    """测试文字内容多样性"""
    print("=== 测试文字内容多样性 ===")
    texts = set()
    for i in range(20):
        text = generate_text_content()
        texts.add(text)
        print(f"{i+1:2d}. {text}")
    
    print(f"\n生成了 {len(texts)} 种不同的文字内容（共20次生成）")
    return len(texts)

def test_font_compatibility():
    """测试字体兼容性"""
    print("\n=== 测试字体兼容性 ===")
    fonts = load_system_fonts()
    print(f"找到 {len(fonts)} 个系统字体")
    
    test_texts = [
        "SAMPLE",  # 英文
        "样本",     # 中文
        "Demo 演示", # 混合
        "©2024",   # 符号
    ]
    
    for text in test_texts:
        compatible_font = select_compatible_font(fonts, text)
        if compatible_font:
            font_name = os.path.basename(compatible_font)
            print(f"文字 '{text}' -> 兼容字体: {font_name}")
        else:
            print(f"文字 '{text}' -> 未找到兼容字体")

def test_watermark_generation():
    """测试水印生成效果"""
    print("\n=== 测试水印生成效果 ===")
    
    # 创建测试图片
    test_img = Image.new('RGB', (800, 600), color='white')
    test_img_path = '/tmp/test_clean_image.jpg'
    test_img.save(test_img_path)
    
    success_count = 0
    total_tests = 10
    
    for i in range(total_tests):
        try:
            watermarked_img, mask_img = generate_text_watermark(test_img_path)
            
            # 检查是否成功生成
            if watermarked_img and mask_img:
                # 检查水印是否有实际内容
                mask_array = np.array(mask_img)
                has_watermark = np.any(mask_array > 0)
                
                if has_watermark:
                    success_count += 1
                    print(f"测试 {i+1}: 成功生成水印")
                else:
                    print(f"测试 {i+1}: 生成的水印为空")
            else:
                print(f"测试 {i+1}: 生成失败")
                
        except Exception as e:
            print(f"测试 {i+1}: 出现错误 - {e}")
    
    print(f"\n成功率: {success_count}/{total_tests} ({success_count/total_tests*100:.1f}%)")
    
    # 清理测试文件
    if os.path.exists(test_img_path):
        os.remove(test_img_path)
    
    return success_count / total_tests

def main():
    """主测试函数"""
    print("开始测试改进后的文字水印生成...\n")
    
    # 测试文字多样性
    diversity_score = test_text_diversity()
    
    # 测试字体兼容性
    test_font_compatibility()
    
    # 测试水印生成
    success_rate = test_watermark_generation()
    
    print("\n=== 测试总结 ===")
    print(f"文字多样性: {diversity_score}/20 种不同文字")
    print(f"水印生成成功率: {success_rate*100:.1f}%")
    
    if diversity_score >= 15 and success_rate >= 0.8:
        print("✅ 改进效果良好！")
    elif diversity_score >= 10 and success_rate >= 0.6:
        print("⚠️  改进有效果，但仍有优化空间")
    else:
        print("❌ 需要进一步改进")

if __name__ == "__main__":
    main()