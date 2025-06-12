#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化的水印修复启动脚本
解决CUDA卡顿问题，提供更快的处理速度
"""

import os
import sys
import argparse
from pathlib import Path

# 添加src目录到Python路径
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from cli import main as cli_main

def create_optimized_args(input_path, output_path, model_path, **kwargs):
    """
    创建优化的参数配置
    
    Args:
        input_path: 输入路径
        output_path: 输出路径
        model_path: 模型路径
        **kwargs: 其他参数
    """
    
    # 默认优化参数
    default_params = {
        'command': 'repair',
        'input': input_path,
        'output': output_path,
        'model': model_path,
        'device': 'cuda',
        'optimize': True,  # 启用优化模式
        'batch_size': 5,   # 较小的批处理大小，避免内存溢出
        'pause_interval': 25,  # 更频繁的清理
        'max_iterations': 3,   # 减少迭代次数
        'threshold': 0.005,    # 稍微提高阈值
        'iopaint_model': 'ldm',  # 使用LDM模型
        'limit': None
    }
    
    # 更新参数
    default_params.update(kwargs)
    
    # 创建命名空间对象
    class Args:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key.replace('-', '_'), value)
    
    return Args(**default_params)

def main():
    parser = argparse.ArgumentParser(
        description="优化的水印修复工具 - 解决CUDA卡顿问题",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本用法
  python repair_optimized.py --input data/train/watermarked --output data/result --model models/checkpoints/checkpoint_epoch_030.pth
  
  # 自定义参数
  python repair_optimized.py --input data/train/watermarked --output data/result --model models/checkpoints/checkpoint_epoch_030.pth --batch-size 3 --limit 50
  
  # 快速模式（更激进的优化）
  python repair_optimized.py --input data/train/watermarked --output data/result --model models/checkpoints/checkpoint_epoch_030.pth --fast
        """
    )
    
    # 必需参数
    parser.add_argument('--input', required=True, help='输入图像路径或目录')
    parser.add_argument('--output', required=True, help='输出目录')
    parser.add_argument('--model', required=True, help='模型文件路径')
    
    # 可选参数
    parser.add_argument('--config', help='配置文件路径')
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda', 'auto'], help='设备类型')
    parser.add_argument('--batch-size', type=int, default=5, help='批处理大小（建议1-10）')
    parser.add_argument('--pause-interval', type=int, default=25, help='暂停清理间隔')
    parser.add_argument('--max-iterations', type=int, default=3, help='最大迭代次数')
    parser.add_argument('--threshold', type=float, default=0.005, help='水印检测阈值')
    parser.add_argument('--iopaint-model', default='ldm', choices=['lama', 'ldm', 'zits', 'mat', 'fcf'], help='IOPaint模型')
    parser.add_argument('--limit', type=int, help='处理图片数量限制')
    
    # 预设模式
    parser.add_argument('--fast', action='store_true', help='快速模式（更激进的优化）')
    parser.add_argument('--conservative', action='store_true', help='保守模式（更稳定但较慢）')
    
    args = parser.parse_args()
    
    # 验证输入
    if not os.path.exists(args.input):
        print(f"错误: 输入路径不存在: {args.input}")
        return 1
    
    if not os.path.exists(args.model):
        print(f"错误: 模型文件不存在: {args.model}")
        return 1
    
    # 应用预设模式
    if args.fast:
        print("使用快速模式...")
        args.batch_size = 3
        args.pause_interval = 15
        args.max_iterations = 2
        args.threshold = 0.01
    elif args.conservative:
        print("使用保守模式...")
        args.batch_size = 8
        args.pause_interval = 40
        args.max_iterations = 5
        args.threshold = 0.001
    
    # 创建优化参数
    optimized_args = create_optimized_args(
        input_path=args.input,
        output_path=args.output,
        model_path=args.model,
        config=args.config,
        device=args.device,
        batch_size=args.batch_size,
        pause_interval=args.pause_interval,
        max_iterations=args.max_iterations,
        threshold=args.threshold,
        iopaint_model=args.iopaint_model,
        limit=args.limit
    )
    
    print("="*60)
    print("优化水印修复工具")
    print("="*60)
    print(f"输入路径: {args.input}")
    print(f"输出路径: {args.output}")
    print(f"模型路径: {args.model}")
    print(f"设备: {args.device}")
    print(f"批处理大小: {args.batch_size}")
    print(f"暂停间隔: {args.pause_interval}")
    print(f"最大迭代次数: {args.max_iterations}")
    print(f"水印阈值: {args.threshold}")
    print(f"IOPaint模型: {args.iopaint_model}")
    if args.limit:
        print(f"处理限制: {args.limit} 张图片")
    print("="*60)
    print()
    
    # 执行修复
    try:
        # 临时替换sys.argv以传递给cli_main
        original_argv = sys.argv.copy()
        sys.argv = ['cli.py'] + [
            'repair',
            '--input', args.input,
            '--output', args.output,
            '--model', args.model,
            '--device', args.device,
            '--optimize',
            '--batch-size', str(args.batch_size),
            '--pause-interval', str(args.pause_interval),
            '--max-iterations', str(args.max_iterations),
            '--threshold', str(args.threshold),
            '--iopaint-model', args.iopaint_model
        ]
        
        if args.config:
            sys.argv.extend(['--config', args.config])
        
        if args.limit:
            sys.argv.extend(['--limit', str(args.limit)])
        
        # 调用CLI主函数
        cli_main()
        
        print("\n" + "="*60)
        print("修复完成！")
        print("="*60)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n用户中断操作")
        return 1
    except Exception as e:
        print(f"\n错误: {str(e)}")
        return 1
    finally:
        # 恢复原始argv
        sys.argv = original_argv

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)