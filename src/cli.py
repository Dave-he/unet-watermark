#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
命令行接口模块
处理命令行参数解析和命令执行
"""

import os
import sys
import argparse
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
    print()
    
    try:
        # 创建模型
        model = WatermarkSegmentationModel(cfg)
        print(f"模型信息: {model.get_model_info()}")
        
        # 开始训练 - 只传递cfg参数
        train(cfg)
        
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
            iopaint_model='lama',
            limit=args.limit
        )
        
        print("\n预测完成！")
        print(f"处理图像数量: {len(results)}")
        print(f"成功: {sum(1 for r in results if r['status'] == 'success')}")
        print(f"失败: {sum(1 for r in results if r['status'] == 'failed')}")
        
    except Exception as e:
        print(f"预测过程中出现错误: {str(e)}")
        raise


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
    
    # 预测命令
    predict_parser = subparsers.add_parser('predict', help='预测/推理')
    predict_parser.add_argument('--input', type=str, required=True, 
                               help='输入图像路径或目录')
    predict_parser.add_argument('--output', type=str, required=True, 
                               help='输出目录')
    predict_parser.add_argument('--model', type=str, required=True, 
                               help='模型文件路径')
    predict_parser.add_argument('--config', type=str, help='配置文件路径')
    predict_parser.add_argument('--device', type=str, 
                               default='auto', help='计算设备 (默认: auto)')
    predict_parser.add_argument('--batch-size', type=int, default=8, 
                               help='批次大小 (默认: 8)')
    predict_parser.add_argument('--threshold', type=float, default=0.5, 
                               help='二值化阈值 (默认: 0.5)')
    predict_parser.add_argument('--save-mask', action='store_true',
                               help='保存预测掩码')
    predict_parser.add_argument('--save-overlay', action='store_true', 
                               help='保存叠加可视化图像')
    predict_parser.add_argument('--limit', type=int, 
                               help='随机选择的图片数量限制')
    
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
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        print("\n用户中断操作")
    except Exception as e:
        print(f"\n错误: {str(e)}")
        sys.exit(1)