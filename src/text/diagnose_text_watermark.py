#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文字水印显示问题诊断脚本
检查字体支持、文字渲染和多样性问题
"""

import sys
import os
from pathlib import Path
import random
import tempfile

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def test_font_support():
    """测试字体支持情况"""
    print("=== 字体支持测试 ===")
    
    try:
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        
        # 测试不同字体对中英文的支持
        test_texts = [
            "SAMPLE",
            "样本", 
            "Demo 演示",
            "Copyright ©2024",
            "WATERMARK",
            "水印",
            "Preview 预览"
        ]
        
        # 测试系统字体
        font_paths = [
            '/System/Library/Fonts/Arial.ttf',
            '/System/Library/Fonts/Helvetica.ttc', 
            '/System/Library/Fonts/PingFang.ttc',  # 支持中文
            '/System/Library/Fonts/STHeiti Light.ttc',  # 支持中文
            '/System/Library/Fonts/Hiragino Sans GB.ttc',  # 支持中文
            None  # 默认字体
        ]
        
        results = []
        
        for font_path in font_paths:
            font_name = os.path.basename(font_path) if font_path else "默认字体"
            print(f"\n测试字体: {font_name}")
            
            try:
                if font_path and os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, 32)
                else:
                    font = ImageFont.load_default()
                
                font_results = []
                
                for text in test_texts:
                    # 创建测试图像
                    img = Image.new('RGB', (300, 100), color='white')
                    draw = ImageDraw.Draw(img)
                    
                    try:
                        # 尝试绘制文字
                        draw.text((10, 30), text, fill='black', font=font)
                        
                        # 检查是否有实际内容（不是空白或方框）
                        img_array = np.array(img)
                        # 检查是否有非白色像素
                        has_content = np.any(img_array < 255)
                        
                        if has_content:
                            # 进一步检查是否是方框（简单检测）
                            gray = np.mean(img_array, axis=2)
                            # 如果图像变化很小，可能是方框
                            variance = np.var(gray)
                            is_likely_box = variance < 100  # 阈值可调整
                            
                            status = "方框" if is_likely_box else "正常"
                        else:
                            status = "空白"
                            
                        font_results.append((text, status))
                        print(f"  '{text}': {status}")
                        
                    except Exception as e:
                        font_results.append((text, f"错误: {e}"))
                        print(f"  '{text}': 错误 - {e}")
                
                results.append((font_name, font_results))
                
            except Exception as e:
                print(f"  字体加载失败: {e}")
                results.append((font_name, [("加载失败", str(e))]))
        
        return results
        
    except Exception as e:
        print(f"字体测试失败: {e}")
        return []

def test_text_content_diversity():
    """测试文字内容多样性"""
    print("\n=== 文字内容多样性测试 ===")
    
    try:
        # 导入生成函数
        from scripts.gen_data import generate_text_content
        
        # 生成多个文字样本
        samples = []
        for i in range(20):
            text = generate_text_content()
            samples.append(text)
        
        print(f"生成了 {len(samples)} 个文字样本:")
        for i, text in enumerate(samples, 1):
            print(f"{i:2d}. '{text}'")
        
        # 分析多样性
        unique_samples = set(samples)
        print(f"\n多样性分析:")
        print(f"- 总样本数: {len(samples)}")
        print(f"- 唯一样本数: {len(unique_samples)}")
        print(f"- 重复率: {(len(samples) - len(unique_samples)) / len(samples) * 100:.1f}%")
        
        # 分析字符类型
        has_chinese = sum(1 for text in unique_samples if any('\u4e00' <= char <= '\u9fff' for char in text))
        has_english = sum(1 for text in unique_samples if any(char.isalpha() and ord(char) < 128 for char in text))
        has_numbers = sum(1 for text in unique_samples if any(char.isdigit() for char in text))
        has_symbols = sum(1 for text in unique_samples if any(not char.isalnum() and not char.isspace() for char in text))
        
        print(f"\n字符类型分布:")
        print(f"- 包含中文: {has_chinese} 个 ({has_chinese/len(unique_samples)*100:.1f}%)")
        print(f"- 包含英文: {has_english} 个 ({has_english/len(unique_samples)*100:.1f}%)")
        print(f"- 包含数字: {has_numbers} 个 ({has_numbers/len(unique_samples)*100:.1f}%)")
        print(f"- 包含符号: {has_symbols} 个 ({has_symbols/len(unique_samples)*100:.1f}%)")
        
        return samples
        
    except Exception as e:
        print(f"文字内容测试失败: {e}")
        import traceback
        traceback.print_exc()
        return []

def test_actual_text_watermark_generation():
    """测试实际文字水印生成"""
    print("\n=== 实际文字水印生成测试 ===")
    
    try:
        from PIL import Image
        from scripts.gen_data import generate_text_watermark
        
        # 创建测试图像
        test_img = Image.new('RGB', (512, 512), color='white')
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            test_img.save(temp_file.name)
            temp_path = temp_file.name
        
        try:
            print("生成文字水印样本...")
            
            # 生成多个样本
            for i in range(5):
                print(f"\n样本 {i+1}:")
                
                watermarked_img, mask = generate_text_watermark(
                    temp_path,
                    enhance_transparent=True,
                    use_ocr_mask=False  # 先不使用OCR，简化测试
                )
                
                print(f"  ✓ 生成成功")
                print(f"  - 水印图像尺寸: {watermarked_img.size}")
                print(f"  - mask图像尺寸: {mask.size}")
                
                # 保存样本用于检查
                sample_dir = Path("debug_samples")
                sample_dir.mkdir(exist_ok=True)
                
                watermarked_img.save(sample_dir / f"text_watermark_{i+1}.png")
                mask.save(sample_dir / f"text_mask_{i+1}.png")
                
                print(f"  - 已保存到: debug_samples/text_watermark_{i+1}.png")
            
            print(f"\n✓ 所有样本已保存到 debug_samples/ 目录")
            print(f"请检查生成的图像是否包含正确的文字而不是方框")
            
        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
        return True
        
    except Exception as e:
        print(f"文字水印生成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def suggest_solutions(font_results, text_samples):
    """根据测试结果提供解决方案"""
    print("\n=== 问题诊断和解决方案 ===")
    
    # 分析字体问题
    problematic_fonts = []
    good_fonts = []
    
    for font_name, results in font_results:
        has_boxes = any("方框" in status for text, status in results)
        has_errors = any("错误" in status for text, status in results)
        
        if has_boxes or has_errors:
            problematic_fonts.append(font_name)
        else:
            good_fonts.append(font_name)
    
    print("\n字体问题分析:")
    if problematic_fonts:
        print(f"问题字体: {', '.join(problematic_fonts)}")
    if good_fonts:
        print(f"正常字体: {', '.join(good_fonts)}")
    
    # 提供解决方案
    print("\n建议的解决方案:")
    
    if problematic_fonts:
        print("\n1. 字体问题解决:")
        print("   - 优先使用支持中文的字体（如 PingFang.ttc, STHeiti Light.ttc）")
        print("   - 避免使用只支持英文的字体渲染中文")
        print("   - 在字体选择时添加中文字体检测")
        
        print("\n   代码修改建议:")
        print("   ```python")
        print("   # 在 load_system_fonts() 中优先添加中文字体")
        print("   chinese_fonts = [")
        print("       '/System/Library/Fonts/PingFang.ttc',")
        print("       '/System/Library/Fonts/STHeiti Light.ttc',")
        print("       '/System/Library/Fonts/Hiragino Sans GB.ttc'")
        print("   ]")
        print("   ```")
    
    if len(set(text_samples)) < len(text_samples) * 0.8:
        print("\n2. 文字多样性问题解决:")
        print("   - 扩展文字内容库")
        print("   - 添加更多文字类型和组合")
        print("   - 实现动态文字生成")
    
    print("\n3. 通用改进建议:")
    print("   - 添加字体-文字兼容性检测")
    print("   - 实现文字渲染质量验证")
    print("   - 增加文字效果的多样性")
    print("   - 添加文字大小和位置的随机性")

def main():
    """主函数"""
    print("文字水印显示问题诊断工具")
    print("=" * 50)
    
    # 运行测试
    font_results = test_font_support()
    text_samples = test_text_content_diversity()
    generation_success = test_actual_text_watermark_generation()
    
    # 提供解决方案
    suggest_solutions(font_results, text_samples)
    
    print("\n=== 诊断完成 ===")
    print("请查看 debug_samples/ 目录中的生成样本")
    print("如果仍然看到方框，请按照上述建议修改代码")

if __name__ == "__main__":
    main()