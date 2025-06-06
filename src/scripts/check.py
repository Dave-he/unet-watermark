import os
import argparse
from pathlib import Path
from collections import defaultdict

def get_filename_without_ext(filepath):
    """获取不带扩展名的文件名"""
    return Path(filepath).stem

def get_image_files(directory):
    """获取目录中的所有图片文件"""
    if not os.path.exists(directory):
        return set()
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    files = set()
    
    for filename in os.listdir(directory):
        if Path(filename).suffix.lower() in image_extensions:
            files.add(get_filename_without_ext(filename))
    
    return files

def validate_dataset(base_dir, dry_run=False):
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
            
            if (has_watermarked and has_clean) or \
               (has_watermarked and has_masks) or \
               (has_clean and has_masks):
                valid_files.add(filename)
            else:
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
        print("  这些文件在至少两个相关目录中都存在对应文件")
    
    # 显示无效文件
    total_invalid = sum(len(files) for files in invalid_files.values())
    if total_invalid > 0:
        print(f"\n✗ 无效文件: {total_invalid} 个")
        
        for dir_name, files in invalid_files.items():
            if files:
                print(f"\n  {dir_name} 目录中的孤立文件 ({len(files)} 个):")
                for filename in sorted(files)[:10]:  # 只显示前10个
                    file_path = None
                    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
                        potential_path = os.path.join(existing_dirs[dir_name], filename + ext)
                        if os.path.exists(potential_path):
                            file_path = potential_path
                            break
                    
                    if file_path:
                        if dry_run:
                            print(f"    [检测] 将删除: {file_path}")
                        else:
                            print(f"    [删除] {file_path}")
                            try:
                                os.remove(file_path)
                            except Exception as e:
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
    mode_text = "检测模式" if dry_run else "清理模式"
    print(f"运行模式: {mode_text}")
    
    if dry_run:
        if total_invalid > 0:
            print(f"检测结果: 发现 {total_invalid} 个无效文件")
            print("提示: 使用 --delete 参数可真正执行删除操作")
        else:
            print("检测结果: 数据集完整，无需清理")
    else:
        if total_invalid > 0:
            print(f"清理完成: 尝试删除 {total_invalid} 个无效文件")
        else:
            print("清理完成: 数据集完整，无文件被删除")
        print(f"最终有效文件数: {len(valid_files)}")

def main():
    parser = argparse.ArgumentParser(
        description="验证数据集完整性，检查watermarked、clean、masks目录之间的对应关系",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""使用示例:
  python check.py                    # 检测模式，只显示无效文件
  python check.py --delete           # 清理模式，删除无效文件
  python check.py --base-dir custom  # 指定自定义基础目录
  
数据有效性规则:
  - 文件在以下任一组合中存在对应文件即为有效:
    1. watermarked + clean (训练对)
    2. watermarked + masks (带标注的水印图)
    3. clean + masks (干净图+标注)
    4. 三个目录都有 (完整数据)
  - 仅在单个目录中存在的文件将被标记为无效
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
    
    args = parser.parse_args()
    
    print(f"基础目录: {args.base_dir}")
    print(f"运行模式: {'清理模式' if args.delete else '检测模式'}")
    print("-" * 60)
    
    # 执行数据集验证
    validate_dataset(args.base_dir, dry_run=not args.delete)

if __name__ == "__main__":
    main()
