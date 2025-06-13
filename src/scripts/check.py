# -*- coding: utf-8 -*-
"""
数据集检查工具
用于验证训练数据集的完整性和一致性
"""

import os
import argparse
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
import shutil
from typing import Set, Dict, Optional

def get_filename_without_ext(filepath: str) -> str:
    """获取不带扩展名的文件名"""
    return Path(filepath).stem

def is_black_mask(mask_path: str, threshold: float = 0.01) -> bool:
    """检测mask图片是否为全黑或接近全黑
    
    Args:
        mask_path: mask图片路径
        threshold: 非零像素比例阈值，低于此值认为是全黑
    
    Returns:
        bool: True表示是全黑mask，False表示正常mask
    """
    try:
        # 读取图片
        img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"    [警告] 无法读取mask文件: {mask_path}")
            return True  # 无法读取的文件也认为是无效的
        
        # 计算非零像素的比例
        total_pixels = img.shape[0] * img.shape[1]
        non_zero_pixels = np.count_nonzero(img)
        non_zero_ratio = non_zero_pixels / total_pixels
        
        return non_zero_ratio < threshold
    except Exception as e:
        print(f"    [警告] 检测mask文件时出错 {mask_path}: {e}")
        return True  # 出错的文件也认为是无效的

def get_image_files(directory: str) -> Set[str]:
    """获取目录中的所有图片文件"""
    if not os.path.exists(directory):
        return set()
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    files = set()
    
    for filename in os.listdir(directory):
        if Path(filename).suffix.lower() in image_extensions:
            files.add(get_filename_without_ext(filename))
    
    return files

def validate_dataset(base_dir: str, dry_run: bool = False, move_dir: Optional[str] = None) -> None:
    """
    验证数据集的完整性，检查三个目录之间的对应关系
    
    Args:
        base_dir: 基础目录路径
        dry_run: 如果为True，只检测不删除；如果为False，真正删除文件
    """
    # 构建目录路径
    watermarked_dir = os.path.join(base_dir, "watermarked")
    clean_dir = os.path.join(base_dir, "clean")
    masks_dir = os.path.join(base_dir, "masks")
    
    # 检查目录是否存在
    dirs_info = {
        "watermarked": watermarked_dir,
        "clean": clean_dir,
        "masks": masks_dir
    }
    
    existing_dirs = {}
    for name, path in dirs_info.items():
        if os.path.exists(path):
            existing_dirs[name] = path
            print(f"✓ 找到目录: {path}")
        else:
            print(f"✗ 目录不存在: {path}")
    
    if len(existing_dirs) < 2:
        print("\n错误: 至少需要两个目录存在才能进行验证")
        return
    
    # 获取各目录中的文件列表（不带扩展名）
    files_by_dir = {}
    for name, path in existing_dirs.items():
        files_by_dir[name] = get_image_files(path)
        print(f"{name} 目录: {len(files_by_dir[name])} 个文件")
    
    # 分析数据完整性
    print("\n" + "="*60)
    print("数据集完整性分析")
    print("="*60)
    
    # 找出所有唯一的文件名
    all_files = set()
    for files in files_by_dir.values():
        all_files.update(files)
    
    # 分类文件状态
    valid_files = set()  # 有效文件（至少在两个目录中存在）
    invalid_files = defaultdict(list)  # 无效文件（仅在单个目录中存在）
    black_mask_files = []  # 全黑mask文件
    
    for filename in all_files:
        present_in = []
        for dir_name, files in files_by_dir.items():
            if filename in files:
                present_in.append(dir_name)
        
        if len(present_in) >= 2:
            # 检查是否满足有效条件：
            # 1. watermarked + clean 存在
            # 2. watermarked + masks 存在  
            # 3. clean + masks 存在
            # 4. 三个都存在
            has_watermarked = "watermarked" in present_in
            has_clean = "clean" in present_in
            has_masks = "masks" in present_in
            
            # 如果有mask文件，检查是否为全黑
            is_valid = True
            if has_masks and "masks" in existing_dirs:
                # 查找mask文件的完整路径
                mask_path = None
                for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
                    potential_path = os.path.join(existing_dirs["masks"], filename + ext)
                    if os.path.exists(potential_path):
                        mask_path = potential_path
                        break
                
                if mask_path and is_black_mask(mask_path):
                    is_valid = False
                    black_mask_files.append(filename)
                    # 将所有相关文件标记为无效
                    for dir_name in present_in:
                        invalid_files[dir_name].append(filename)
            
            if is_valid and ((has_watermarked and has_clean) or \
                            (has_watermarked and has_masks) or \
                            (has_clean and has_masks)):
                valid_files.add(filename)
            elif is_valid:
                # 理论上不会到这里，但为了安全起见
                for dir_name in present_in:
                    invalid_files[dir_name].append(filename)
        else:
            # 仅在单个目录中存在
            for dir_name in present_in:
                invalid_files[dir_name].append(filename)
    
    # 显示结果
    print(f"\n✓ 有效文件: {len(valid_files)} 个")
    if valid_files:
        print("  这些文件在至少两个相关目录中都存在对应文件，且mask不为全黑")
    
    # 显示全黑mask文件信息
    if black_mask_files:
        print(f"\n⚠ 全黑mask文件: {len(black_mask_files)} 个")
        print("  这些文件的mask图片为全黑或接近全黑，已标记为无效")
        if len(black_mask_files) <= 10:
            for filename in sorted(black_mask_files):
                print(f"    - {filename}")
        else:
            for filename in sorted(black_mask_files)[:10]:
                print(f"    - {filename}")
            print(f"    ... 还有 {len(black_mask_files) - 10} 个文件")
    
    # 显示无效文件
    total_invalid = sum(len(files) for files in invalid_files.values())
    if total_invalid > 0:
        print(f"\n✗ 无效文件: {total_invalid} 个")
        
        for dir_name, files in invalid_files.items():
            if files:
                print(f"\n  {dir_name} 目录中的孤立文件 ({len(files)} 个):")
                # 显示前10个文件信息，但删除所有文件
                for i, filename in enumerate(sorted(files)):
                    if i < 10:  # 只显示前10个的详细信息
                        show_details = True
                    else:
                        show_details = False
                    file_path = None
                    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
                        potential_path = os.path.join(existing_dirs[dir_name], filename + ext)
                        if os.path.exists(potential_path):
                            file_path = potential_path
                            break
                    
                    if file_path:
                        if dry_run:
                            if show_details:
                                action_text = "移动" if move_dir else "删除"
                                print(f"    [检测] 将{action_text}: {file_path}")
                        else:
                            if move_dir:
                                # 移动文件到指定目录
                                try:
                                    # 创建目标目录结构
                                    rel_path = os.path.relpath(file_path, base_dir)
                                    target_path = os.path.join(move_dir, rel_path)
                                    target_dir = os.path.dirname(target_path)
                                    os.makedirs(target_dir, exist_ok=True)
                                    
                                    # 移动文件
                                    shutil.move(file_path, target_path)
                                    if show_details:
                                        print(f"    [移动] {file_path} -> {target_path}")
                                except Exception as e:
                                    if show_details:
                                        print(f"    [错误] 移动失败: {e}")
                            else:
                                # 删除文件
                                if show_details:
                                    print(f"    [删除] {file_path}")
                                try:
                                    os.remove(file_path)
                                    if not show_details:
                                        # 对于不显示详细信息的文件，静默删除
                                        pass
                                except Exception as e:
                                    if show_details:
                                        print(f"    [错误] 删除失败: {e}")
                
                if len(files) > 10:
                    print(f"    ... 还有 {len(files) - 10} 个文件")
    else:
        print("\n✓ 未发现无效文件")
    
    # 显示统计信息
    print("\n" + "="*60)
    print("统计信息")
    print("="*60)
    
    for dir_name, files in files_by_dir.items():
        valid_count = len([f for f in files if f in valid_files])
        invalid_count = len(invalid_files.get(dir_name, []))
        print(f"{dir_name:12}: 总计 {len(files):4} | 有效 {valid_count:4} | 无效 {invalid_count:4}")
    
    # 显示操作结果
    print("\n" + "="*60)
    if dry_run:
        mode_text = "检测模式"
    else:
        mode_text = "移动模式" if move_dir else "删除模式"
    print(f"运行模式: {mode_text}")
    
    if move_dir and not dry_run:
        print(f"目标目录: {move_dir}")
    
    if dry_run:
        if total_invalid > 0:
            action_text = "移动" if move_dir else "删除"
            print(f"检测结果: 发现 {total_invalid} 个无效文件")
            print(f"提示: 使用 --delete 参数可真正执行{action_text}操作")
        else:
            print("检测结果: 数据集完整，无需清理")
    else:
        if total_invalid > 0:
            action_text = "移动" if move_dir else "删除"
            print(f"清理完成: 尝试{action_text} {total_invalid} 个无效文件")
        else:
            action_text = "移动" if move_dir else "删除"
            print(f"清理完成: 数据集完整，无文件被{action_text}")
        print(f"最终有效文件数: {len(valid_files)}")

def main():
    parser = argparse.ArgumentParser(
        description="验证数据集完整性，检查watermarked、clean、masks目录之间的对应关系",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""使用示例:
  python check.py                              # 检测模式，只显示无效文件
  python check.py --delete                     # 清理模式，删除无效文件
  python check.py --delete --move-dir backup   # 清理模式，移动无效文件到backup目录
  python check.py --base-dir custom            # 指定自定义基础目录
  
数据有效性规则:
  - 文件在以下任一组合中存在对应文件即为有效:
    1. watermarked + clean (训练对)
    2. watermarked + masks (带标注的水印图)
    3. clean + masks (干净图+标注)
    4. 三个目录都有 (完整数据)
  - 仅在单个目录中存在的文件将被标记为无效
  - mask图片为全黑或接近全黑的文件将被标记为无效
        """
    )
    
    parser.add_argument(
        "--delete", 
        action="store_true", 
        help="真正删除无效文件（默认为检测模式）"
    )
    
    parser.add_argument(
        "--base-dir", 
        default="data/train",
        help="基础目录路径（默认: data/train）"
    )
    
    parser.add_argument(
        "--move-dir",
        type=str,
        default="data/backup",
        help="移动无效文件到指定目录（如不指定则删除文件）"
    )
    
    args = parser.parse_args()
    
    print(f"基础目录: {args.base_dir}")
    print(f"运行模式: {'清理模式' if args.delete else '检测模式'}")
    print("-" * 60)
    
    # 执行数据集验证
    validate_dataset(args.base_dir, dry_run=not args.delete, move_dir=getattr(args, 'move_dir', None))

if __name__ == "__main__":
    main()
