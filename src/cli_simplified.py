#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版命令行接口 - 仅支持文件夹修复模式
"""

import os
import argparse
import subprocess
import torch
import json

from configs.config import get_cfg_defaults, update_config
from predict_smp_simplified import WatermarkPredictor


def setup_device(device_str):
    """设置计算设备"""
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    
    print(f"使用设备: {device}")
    if device.type == 'cuda':
        print(f"GPU名称: {torch.cuda.get_device_name(device)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    
    return device


def repair_command(args):
    """文件夹修复命令"""
    print("=" * 60)
    print("开始文件夹水印修复")
    print("=" * 60)
    
    # 验证参数
    if not os.path.exists(args.input):
        print(f"错误: 输入路径不存在: {args.input}")
        return
    
    if not os.path.exists(args.model):
        print(f"错误: 模型文件不存在: {args.model}")
        return
    
    if not os.path.isdir(args.input):
        print(f"错误: 输入路径必须是文件夹: {args.input}")
        return
    
    # 设置设备
    device = setup_device(args.device)
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 检查iopaint是否安装
    try:
        subprocess.run(['iopaint', '--help'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("错误: iopaint未安装或不在PATH中")
        print("请运行: pip install iopaint")
        return
    
    print(f"输入文件夹: {args.input}")
    print(f"输出文件夹: {args.output}")
    print(f"模型路径: {args.model}")
    print(f"最大迭代次数: {args.max_iterations}")
    print(f"水印阈值: {args.watermark_threshold}")
    print(f"IOPaint模型: {args.iopaint_model}")
    if hasattr(args, 'limit') and args.limit:
        print(f"处理图片限制: {args.limit} 张")
    print()
    
    try:
        # 加载配置
        config_path = getattr(args, 'config', None)
        if config_path and not os.path.exists(config_path):
            print(f"警告: 配置文件不存在: {config_path}，使用默认配置")
            config_path = None
        
        # 创建预测器
        predictor = WatermarkPredictor(
            model_path=args.model,
            config_path=config_path,
            device=device
        )
        
        # 执行文件夹迭代修复
        results = predictor.process_folder_iterative(
            input_folder=args.input,
            output_folder=args.output,
            max_iterations=args.max_iterations,
            watermark_threshold=args.watermark_threshold,
            iopaint_model=args.iopaint_model,
            limit=getattr(args, 'limit', None)
        )
        
        print("\n文件夹修复完成！")
        print(f"总图片数: {results.get('total_images', 0)}")
        print(f"成功处理: {results.get('successful', 0)}")
        print(f"部分处理: {results.get('partial', 0)}")
        print(f"成功率: {results.get('success_rate', 0):.1f}%")
        print(f"平均迭代次数: {results.get('average_iterations', 0)}")
        
        # 保存结果摘要
        summary_path = os.path.join(args.output, 'repair_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n结果摘要已保存: {summary_path}")
        
    except Exception as e:
        print(f"修复过程中出现错误: {str(e)}")
        raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="水印修复系统 - 简化版（仅支持文件夹修复模式）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python cli_simplified.py --input /path/to/images --output /path/to/output --model /path/to/model.pth
  python cli_simplified.py --input /path/to/images --output /path/to/output --model /path/to/model.pth --max-iterations 10 --watermark-threshold 0.005
        """
    )
    
    # 文件夹修复参数
    parser.add_argument('--input', type=str, required=True,
                       help='输入图像文件夹路径')
    parser.add_argument('--output', type=str, required=True,
                       help='输出文件夹路径')
    parser.add_argument('--model', type=str, required=True,
                       help='模型文件路径')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--device', type=str, default='auto',
                       help='计算设备 (默认: auto)')
    parser.add_argument('--max-iterations', type=int, default=10,
                       help='最大迭代次数 (默认: 10)')
    parser.add_argument('--watermark-threshold', type=float, default=0.0001,
                       help='水印面积阈值，低于此值认为修复完成 (默认: 0.0001)')
    parser.add_argument('--iopaint-model', type=str, default='lama',
                       help='IOPaint修复模型 (默认: lama)')
    parser.add_argument('--limit', type=int, default=None,
                       help='随机处理的图片数量限制 (默认: 处理所有图片)')
    
    args = parser.parse_args()
    
    try:
        repair_command(args)
    except KeyboardInterrupt:
        print("\n用户中断操作")
    except Exception as e:
        print(f"\n程序执行出错: {str(e)}")
        raise


if __name__ == '__main__':
    main()