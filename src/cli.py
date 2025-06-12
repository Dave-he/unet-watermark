#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
命令行接口模块
处理命令行参数解析和命令执行
"""

import os
import sys
import argparse
import subprocess
import torch

from configs.config import get_cfg_defaults, update_config
from models.smp_models import WatermarkSegmentationModel
from train_smp import train
from predict_smp import WatermarkPredictor


def setup_device(device_str):
    """
    设置计算设备
    
    Args:
        device_str (str): 设备字符串，如 'cuda', 'cpu', 'cuda:0'
        
    Returns:
        torch.device: 配置好的设备
    """
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    
    print(f"使用设备: {device}")
    if device.type == 'cuda':
        print(f"GPU名称: {torch.cuda.get_device_name(device)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    
    return device


def train_command(args):
    """
    执行训练命令
    
    Args:
        args: 命令行参数
    """
    print("=" * 60)
    print("开始训练水印分割模型")
    print("=" * 60)
    
    # 加载配置
    cfg = get_cfg_defaults()
    if args.config:
        update_config(cfg, args.config)
    
    # 解冻配置以允许修改
    cfg.defrost()
    
    # 覆盖命令行参数
    if args.device:
        cfg.DEVICE = args.device
    if args.data_dir:
        cfg.DATA.ROOT_DIR = args.data_dir
    if args.output_dir:
        cfg.TRAIN.OUTPUT_DIR = args.output_dir
    if args.model_save_path:
        cfg.TRAIN.MODEL_SAVE_PATH = args.model_save_path
    if args.batch_size:
        cfg.TRAIN.BATCH_SIZE = args.batch_size
    if args.epochs:
        cfg.TRAIN.EPOCHS = args.epochs
    if args.lr:
        cfg.TRAIN.LR = args.lr
    
    # 处理早停参数
    if args.no_early_stopping:
        cfg.TRAIN.USE_EARLY_STOPPING = False
    if args.early_stopping_patience:
        cfg.TRAIN.EARLY_STOPPING_PATIENCE = args.early_stopping_patience
    
    # 设置设备
    device = setup_device(cfg.DEVICE)
    cfg.DEVICE = str(device)
    
    # 重新冻结配置
    cfg.freeze()
    
    # 创建输出目录
    os.makedirs(cfg.TRAIN.OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(cfg.TRAIN.MODEL_SAVE_PATH), exist_ok=True)
    
    # 打印配置信息
    print(f"配置文件: {args.config or 'default'}")
    print(f"数据目录: {cfg.DATA.ROOT_DIR}")
    print(f"输出目录: {cfg.TRAIN.OUTPUT_DIR}")
    print(f"模型保存路径: {cfg.TRAIN.MODEL_SAVE_PATH}")
    print(f"批次大小: {cfg.TRAIN.BATCH_SIZE}")
    print(f"训练轮数: {cfg.TRAIN.EPOCHS}")
    print(f"学习率: {cfg.TRAIN.LR}")
    print(f"模型: {cfg.MODEL.NAME} + {cfg.MODEL.ENCODER_NAME}")
    if getattr(args, 'resume', None):
        print(f"恢复训练: {args.resume}")
    print()
    
    try:
        # 创建模型
        model = WatermarkSegmentationModel(cfg)
        print(f"模型信息: {model.get_model_info()}")
        
        # 开始训练 - 传递cfg和resume参数
        train(cfg, resume_from=getattr(args, 'resume', None))
        
        print("\n训练完成！")
        print(f"最佳模型已保存到: {cfg.TRAIN.MODEL_SAVE_PATH}")
        
    except Exception as e:
        print(f"训练过程中出现错误: {str(e)}")
        raise


def predict_command(args):
    """执行预测命令"""
    print("=" * 60)
    print("开始水印分割预测")
    print("=" * 60)
    
    # 检查必要参数
    if not args.input:
        raise ValueError("预测模式需要指定输入路径 --input")
    if not args.output:
        raise ValueError("预测模式需要指定输出路径 --output")
    if not args.model:
        raise ValueError("预测模式需要指定模型路径 --model")
    
    # 检查文件/目录是否存在
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"输入路径不存在: {args.input}")
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"模型文件不存在: {args.model}")
    
    # 设置设备
    device = setup_device(args.device or 'auto')
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 打印预测信息
    print(f"输入路径: {args.input}")
    print(f"输出路径: {args.output}")
    print(f"模型路径: {args.model}")
    print(f"批次大小: {args.batch_size or 8}")
    print(f"阈值: {args.threshold or 0.5}")
    print()
    
    try:
        # 加载配置（如果提供）
        cfg = get_cfg_defaults()
        if args.config:
            update_config(cfg, args.config)
        
        # 创建预测器
        predictor = WatermarkPredictor(
            model_path=args.model,
            config_path=args.config,
            device=str(device)
        )
        
        # 覆盖阈值配置
        if args.threshold is not None:
            predictor.cfg.PREDICT.THRESHOLD = args.threshold
        
        # 执行预测
        results = predictor.process_batch(
            input_path=args.input,
            output_dir=args.output,
            save_mask=args.save_mask or False,
            remove_watermark=args.save_overlay or False,
            iopaint_model=getattr(args, 'iopaint_model', 'lama'),
            limit=args.limit
        )
        
        print("\n预测完成！")
        print(f"处理图像数量: {len(results)}")
        print(f"成功: {sum(1 for r in results if r['status'] == 'success')}")
        print(f"失败: {sum(1 for r in results if r['status'] == 'failed')}")
        
    except Exception as e:
        print(f"预测过程中出现错误: {str(e)}")
        raise


def repair_command(args):
    """修复命令"""
    # 验证参数
    if not os.path.exists(args.input):
        print(f"错误: 输入路径不存在: {args.input}")
        return
    
    if not os.path.exists(args.model):
        print(f"错误: 模型文件不存在: {args.model}")
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
    
    # 检查是否使用优化模式
    use_optimizer = getattr(args, 'optimize', False)
    
    if use_optimizer:
        print("使用优化批处理模式...")
        from scripts.batch_repair_optimizer import BatchRepairOptimizer
        from utils.cuda_monitor import cuda_memory_context, log_memory_usage
        
        # 记录初始内存状态
        log_memory_usage("开始处理前 - ")
        
        # 使用CUDA内存管理上下文
        with cuda_memory_context(monitor=True) as monitor:
            # 创建优化器
            optimizer = BatchRepairOptimizer(
                model_path=args.model,
                config_path=getattr(args, 'config', None),
                device=device
            )
            
            # 执行优化处理
            results = optimizer.process_batch_with_optimization(
                input_path=args.input,
                output_dir=args.output,
                max_iterations=getattr(args, 'max_iterations', 5),
                watermark_threshold=getattr(args, 'threshold', 0.01),
                iopaint_model=getattr(args, 'iopaint_model', 'lama'),
                limit=getattr(args, 'limit', None),
                batch_size=getattr(args, 'batch_size', 10),
                pause_interval=getattr(args, 'pause_interval', 50)
            )
            
            # 记录内存使用摘要
            if monitor:
                memory_summary = monitor.get_memory_summary()
                print("\n内存使用摘要:")
                if 'current' in memory_summary:
                    current = memory_summary['current']
                    print(f"  当前GPU内存: {current.get('gpu_allocated_gb', 0):.2f}GB (分配) / "
                          f"{current.get('gpu_reserved_gb', 0):.2f}GB (保留)")
                    print(f"  当前系统内存: {current.get('ram_percent', 0):.1f}%")
                
                if 'statistics' in memory_summary:
                    stats = memory_summary['statistics']
                    print(f"  峰值GPU内存: {stats.get('gpu_allocated_max_gb', 0):.2f}GB (分配) / "
                          f"{stats.get('gpu_reserved_max_gb', 0):.2f}GB (保留)")
        
        # 保存结果摘要
        summary_path = os.path.join(args.output, 'optimized_repair_summary.json')
        
    else:
        print("使用标准处理模式...")
        
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
        
        # 覆盖阈值（如果指定）
        if hasattr(args, 'threshold') and args.threshold is not None:
            predictor.watermark_threshold = args.threshold
        
        # 执行修复
        results = predictor.process_batch_iterative(
            input_path=args.input,
            output_dir=args.output,
            max_iterations=getattr(args, 'max_iterations', 10),
            watermark_threshold=getattr(args, 'threshold', 1e-6),
            iopaint_model=getattr(args, 'iopaint_model', 'lama'),
            limit=getattr(args, 'limit', None)
        )
        
        # 保存结果摘要
        summary_path = os.path.join(args.output, 'repair_summary.json')
    
    # 保存结果
    import json
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n修复完成！结果摘要已保存: {summary_path}")
    
    # 记录最终内存状态
    if device == 'cuda':
        from utils.cuda_monitor import log_memory_usage
        log_memory_usage("处理完成后 - ")


def main():
    """主函数 - 解析命令行参数并执行相应操作"""
    parser = argparse.ArgumentParser(
        description="水印分割系统 - 基于SMP库的UNet++模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  训练模式:
    python main.py train --config src/configs/unet_watermark.yaml --device cuda
    python main.py train --data-dir data/train --epochs 100 --batch-size 16
    
  预测模式:
    python main.py predict --input data/test --output results --model models/best_model.pth
    python main.py predict --input single_image.jpg --output results --model models/best_model.pth --save-mask
    
  循环修复模式:
    python main.py repair --input data/test --output results --model models/best_model.pth
    python main.py repair --input single_image.jpg --output results --model models/best_model.pth --max-iterations 10
        """
    )
    
    # 添加子命令
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--config', type=str, help='配置文件路径',
                             default="src/configs/unet_watermark.yaml")
    train_parser.add_argument('--device', type=str, 
                             default='auto', help='计算设备 (默认: auto)')
    train_parser.add_argument('--data-dir', type=str, help='数据集根目录')
    train_parser.add_argument('--output-dir', type=str, help='输出目录')
    train_parser.add_argument('--model-save-path', type=str, help='模型保存路径')
    train_parser.add_argument('--batch-size', type=int, help='批次大小')
    train_parser.add_argument('--epochs', type=int, help='训练轮数')
    train_parser.add_argument('--lr', type=float, help='学习率')
    
    # 添加早停控制参数
    train_parser.add_argument('--no-early-stopping', action='store_true',
                             help='禁用早停机制')
    train_parser.add_argument('--early-stopping-patience', type=int,
                             help='早停耐心值（等待验证损失改善的轮数）')
    
    # 添加恢复训练参数
    train_parser.add_argument('--resume', type=str,
                             help='从指定的checkpoint文件恢复训练')
    
    # 预测命令
    predict_parser = subparsers.add_parser('predict', help='预测/推理')
    predict_parser.add_argument('--input', type=str, default= 'data/test',
                               help='输入图像路径或目录')
    predict_parser.add_argument('--output', type=str, default='data/result',
                               help='输出目录')
    predict_parser.add_argument('--model', type=str, default='models/',
                               help='模型文件路径')
    predict_parser.add_argument('--config', type=str, help='配置文件路径')
    predict_parser.add_argument('--device', type=str, 
                               default='auto', help='计算设备 (默认: auto)')
    predict_parser.add_argument('--batch-size', type=int, default=8, 
                               help='批次大小 (默认: 8)')
    predict_parser.add_argument('--threshold', type=float, default=0.3, 
                               help='二值化阈值 (默认: 0.5)')
    predict_parser.add_argument('--save-mask', action='store_true', default=True,
                               help='保存预测掩码')
    predict_parser.add_argument('--save-overlay', action='store_true', 
                               help='保存叠加可视化图像')
    predict_parser.add_argument('--limit', type=int, 
                               help='随机选择的图片数量限制')
    predict_parser.add_argument('--iopaint-model', type=str, default='lama',
                               help='IOPaint修复模型 (默认: lama)')
    
    # 循环修复命令
    repair_parser = subparsers.add_parser('repair', help='循环检测和修复水印')
    repair_parser.add_argument('--input', type=str, default='/Users/hyx/Pictures/image',
                              help='输入图像路径或目录')
    repair_parser.add_argument('--output', type=str, default='data/result',
                              help='输出目录')
    repair_parser.add_argument('--model', type=str, default='models/checkpoint_epoch_080.pth',
                              help='模型文件路径')
    repair_parser.add_argument('--config', type=str, help='配置文件路径')
    repair_parser.add_argument('--device', type=str, 
                              default='auto', help='计算设备 (默认: auto)')
    repair_parser.add_argument('--threshold', type=float, default=0.3, 
                              help='二值化阈值 (默认: 0.5)')
    repair_parser.add_argument('--max-iterations', type=int, default=10,
                              help='最大迭代次数 (默认: 10)')
    repair_parser.add_argument('--save-mask', action='store_true', default=True,
                            help='保存预测掩码')
    repair_parser.add_argument('--watermark-threshold', type=float, default=0.000001,
                              help='水印面积阈值，低于此值认为修复完成 (默认: 0.000001)')
    repair_parser.add_argument('--min-detection-threshold', type=float, default=0.01,
                              help='最小检测阈值，低于此值认为模型未检测到水印 (默认: 0.01)')
    repair_parser.add_argument('--iopaint-model', type=str, default='lama',
                              help='IOPaint修复模型 (默认: lama)')
    repair_parser.add_argument('--limit', type=int, default=10,
                              help='随机选择的图片数量限制')
    
    # 优化相关参数
    repair_parser.add_argument('--optimize', action='store_true', help='启用优化批处理模式')
    repair_parser.add_argument('--batch-size', type=int, default=10, help='批处理大小（优化模式）')
    repair_parser.add_argument('--pause-interval', type=int, default=50, help='暂停清理间隔（优化模式）')
    
    # 解析参数
    args = parser.parse_args()
    
    # 检查是否提供了命令
    if not args.command:
        parser.print_help()
        return
    
    # 执行相应命令
    try:
        if args.command == 'train':
            train_command(args)
        elif args.command == 'predict':
            predict_command(args)
        elif args.command == 'repair':
            repair_command(args)
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        print("\n用户中断操作")
    except Exception as e:
        print(f"执行命令时出现错误: {str(e)}")
        sys.exit(1)