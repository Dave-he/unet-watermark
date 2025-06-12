import os
import torch
import argparse
from pathlib import Path
from tabulate import tabulate

def list_checkpoints(checkpoint_dir):
    """列出所有检查点"""
    if not os.path.exists(checkpoint_dir):
        print(f"检查点目录不存在: {checkpoint_dir}")
        return
    
    checkpoints = []
    for file in sorted(Path(checkpoint_dir).glob("*.pth")):
        try:
            checkpoint = torch.load(file, map_location='cpu', weights_only=False)
            info = {
                '文件名': file.name,
                '轮数': checkpoint.get('epoch', 'Unknown'),
                '验证损失': f"{checkpoint.get('val_loss', 0):.4f}" if isinstance(checkpoint.get('val_loss'), (int, float)) else 'Unknown',
                '验证IoU': f"{checkpoint.get('val_metrics', {}).get('iou', 0):.4f}" if checkpoint.get('val_metrics') else 'Unknown',
                '验证F1': f"{checkpoint.get('val_metrics', {}).get('f1', 0):.4f}" if checkpoint.get('val_metrics') else 'Unknown',
                '文件大小': f"{file.stat().st_size / 1024 / 1024:.1f}MB"
            }
            checkpoints.append(info)
        except Exception as e:
            print(f"无法读取检查点 {file}: {e}")
    
    if checkpoints:
        print("\n可用的模型检查点:")
        print(tabulate(checkpoints, headers='keys', tablefmt='grid'))
    else:
        print("未找到有效的检查点文件")

def compare_models(model_paths):
    """比较多个模型的性能"""
    models_info = []
    
    for path in model_paths:
        if not os.path.exists(path):
            print(f"模型文件不存在: {path}")
            continue
        
        try:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            info = {
                '模型路径': Path(path).name,
                '轮数': checkpoint.get('epoch', 'Unknown'),
                '验证损失': f"{checkpoint.get('val_loss', 0):.4f}" if isinstance(checkpoint.get('val_loss'), (int, float)) else 'Unknown',
                '训练损失': f"{checkpoint.get('train_loss', 0):.4f}" if isinstance(checkpoint.get('train_loss'), (int, float)) else 'Unknown',
                '验证IoU': f"{checkpoint.get('val_metrics', {}).get('iou', 0):.4f}" if checkpoint.get('val_metrics') else 'Unknown',
                '验证F1': f"{checkpoint.get('val_metrics', {}).get('f1', 0):.4f}" if checkpoint.get('val_metrics') else 'Unknown'
            }
            models_info.append(info)
        except Exception as e:
            print(f"无法读取模型 {path}: {e}")
    
    if models_info:
        print("\n模型性能比较:")
        print(tabulate(models_info, headers='keys', tablefmt='grid'))
    else:
        print("未找到有效的模型文件")

def main():
    parser = argparse.ArgumentParser(description='模型管理工具')
    parser.add_argument('--list', type=str, help='列出指定目录下的所有检查点')
    parser.add_argument('--compare', nargs='+', help='比较多个模型的性能')
    
    args = parser.parse_args()
    
    if args.list:
        list_checkpoints(args.list)
    elif args.compare:
        compare_models(args.compare)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()