import os
from PIL import Image, ImageOps
import numpy as np
from pathlib import Path

def remove_background_and_resize(input_path, output_path, target_size=(256, 256), threshold=240):
    """
    移除图片背景并调整到指定尺寸
    
    Args:
        input_path: 输入图片路径
        output_path: 输出图片路径
        target_size: 目标尺寸 (width, height)
        threshold: 背景移除阈值，越高移除越多白色背景
    """
    try:
        # 打开图片
        img = Image.open(input_path)
        
        # 转换为RGBA模式以支持透明度
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # 获取图片数据
        data = np.array(img)
        
        # 创建透明蒙版：将接近白色的像素设为透明
        # 计算每个像素的亮度
        rgb = data[:, :, :3]
        brightness = np.mean(rgb, axis=2)
        
        # 创建alpha通道：亮度高于阈值的像素设为透明
        alpha = np.where(brightness > threshold, 0, 255)
        data[:, :, 3] = alpha
        
        # 创建新图片
        processed_img = Image.fromarray(data, 'RGBA')
        
        # 调整尺寸，保持宽高比
        processed_img.thumbnail(target_size, Image.Resampling.LANCZOS)
        
        # 创建目标尺寸的透明背景
        final_img = Image.new('RGBA', target_size, (255, 255, 255, 0))
        
        # 计算居中位置
        x = (target_size[0] - processed_img.width) // 2
        y = (target_size[1] - processed_img.height) // 2
        
        # 粘贴处理后的图片到中心
        final_img.paste(processed_img, (x, y), processed_img)
        
        # 保存为PNG格式以保持透明度
        final_img.save(output_path, 'PNG')
        
        return True
        
    except Exception as e:
        print(f"处理图片失败 {input_path}: {e}")
        return False

def process_all_logos(input_dir, output_dir, target_size=(256, 256)):
    """
    处理所有车标图片
    
    Args:
        input_dir: 输入目录（car_logos文件夹）
        output_dir: 输出目录
        target_size: 目标尺寸
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 创建输出目录
    output_path.mkdir(exist_ok=True)
    
    total_processed = 0
    total_success = 0
    
    # 遍历所有字母文件夹
    for letter_folder in input_path.iterdir():
        if letter_folder.is_dir():
            letter = letter_folder.name
            print(f"\n处理字母文件夹: {letter}")
            
            # 处理该文件夹下的所有图片
            for img_file in letter_folder.glob('*.jpg'):
                total_processed += 1
                
                # 构建输出文件名（改为PNG格式），直接保存到输出目录根目录
                # 使用字母前缀避免文件名冲突
                output_file = output_path / f"{letter}_{img_file.stem}.png"
                
                # 处理图片
                if remove_background_and_resize(img_file, output_file, target_size):
                    total_success += 1
                    print(f"✓ 处理成功: {img_file.name} -> {output_file.name}")
                else:
                    print(f"✗ 处理失败: {img_file.name}")
    
    print(f"\n处理完成！")
    print(f"总计处理: {total_processed} 张图片")
    print(f"成功处理: {total_success} 张图片")
    print(f"失败: {total_processed - total_success} 张图片")

def main():
    """
    主函数
    """
    input_dir = "car_logos"  # 原始图片目录
    output_dir = "car_logos_processed"  # 处理后图片目录
    target_size = (256, 256)  # 目标尺寸
    
    print(f"开始处理车标图片...")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"目标尺寸: {target_size[0]}x{target_size[1]}")
    print(f"输出格式: PNG (支持透明背景)")
    
    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        print(f"错误：输入目录 {input_dir} 不存在！")
        return
    
    # 开始处理
    process_all_logos(input_dir, output_dir, target_size)

if __name__ == "__main__":
    main()