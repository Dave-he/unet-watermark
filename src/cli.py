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
import json
from typing import Dict, Any, Optional

from configs.config import get_cfg_defaults, update_config
from models.unet_model import WatermarkSegmentationModel
from train import train
from predict import WatermarkPredictor
from auto_train import AutoTrainingLoop


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
        
        # 检查是否使用文件夹迭代模式
        if getattr(args, 'folder_mode', False) and os.path.isdir(args.input):
            # 使用新的文件夹迭代处理方法
            results = predictor.process_folder_iterative(
                input_folder=args.input,
                output_folder=args.output,
                max_iterations=getattr(args, 'max_iterations', 5),
                watermark_threshold=getattr(args, 'watermark_threshold', 0.01),
                iopaint_model=getattr(args, 'iopaint_model', 'lama')
            )
            
            print("\n文件夹迭代处理完成！")
            print(f"总图片数: {results.get('total_images', 0)}")
            print(f"成功处理: {results.get('successful', 0)}")
            print(f"部分处理: {results.get('partial', 0)}")
            print(f"成功率: {results.get('success_rate', 0):.1f}%")
            print(f"平均迭代次数: {results.get('average_iterations', 0)}")
        else:
            # 使用原有的批处理方法
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
    if getattr(args, 'generate_video', False):
        print(f"生成对比视频: 是 ({args.video_width}x{args.video_height}, {args.fps}fps, {args.duration}s/图)")
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
        
        # 检查是否需要生成视频
        if getattr(args, 'generate_video', False):
            print("开始生成对比视频...")
            try:
                from scripts.video_generator import VideoGenerator
                
                # 检查mask目录
                mask_dir = os.path.join(args.output, 'masks')
                if not os.path.exists(mask_dir):
                    mask_dir = None
                
                # 创建视频生成器
                generator = VideoGenerator(
                    input_dir=args.input,
                    repair_dir=args.output,
                    output_dir=args.output,
                    mask_dir=mask_dir,
                    width=getattr(args, 'video_width', 1920),
                    height=getattr(args, 'video_height', 1080),
                    duration_per_image=getattr(args, 'duration', 2.0),
                    fps=getattr(args, 'fps', 30)
                )
                
                # 根据是否有mask选择视频类型
                if mask_dir:
                    video_path = generator.create_three_way_comparison_video()
                    print(f"三路对比视频已生成: {video_path}")
                else:
                    video_path = generator.create_side_by_side_video()
                    print(f"并排对比视频已生成: {video_path}")
                    
            except Exception as e:
                print(f"视频生成失败: {str(e)}")
        
    except Exception as e:
        print(f"修复过程中出现错误: {str(e)}")
        raise


def auto_train_command(args):
    """执行自动循环训练命令"""
    print("=" * 60)
    print("开始自动循环训练")
    print("=" * 60)
    
    # 设置设备
    device = setup_device(args.device or 'auto')
    
    # 构建配置
    config = {
        'project_root': args.project_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'max_cycles': args.max_cycles,
        'device': str(device),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'output_base_dir': args.output_dir,
        'train_config': args.config or 'src/configs/unet_watermark.yaml',
        'model_selection_samples': args.model_selection_samples,
        'prediction_limit': args.prediction_limit,
        'transparent_ratio': args.transparent_ratio,
        'logos_dir': args.logos_dir
    }
    
    # 如果提供了配置文件，加载配置
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, 'r') as f:
            file_config = json.load(f)
        config.update(file_config)
    
    # 打印配置信息
    print(f"项目根目录: {config['project_root']}")
    print(f"最大循环次数: {config['max_cycles']}")
    print(f"训练设备: {config['device']}")
    print(f"每轮训练轮数: {config['epochs']}")
    print(f"批次大小: {config['batch_size']}")
    print(f"学习率: {config['learning_rate']}")
    print(f"输出目录: {config['output_base_dir']}")
    print(f"训练配置文件: {config['train_config']}")
    print()
    
    try:
        # 创建并运行自动训练循环
        trainer = AutoTrainingLoop(config)
        trainer.run_all_cycles()
        
        print("\n自动循环训练完成！")
        print(f"结果保存在: {config['output_base_dir']}")
        
    except Exception as e:
        print(f"自动训练过程中出现错误: {str(e)}")
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
    
  循环修复模式:
    python main.py repair --input data/test --output results --model models/best_model.pth
    python main.py repair --input single_image.jpg --output results --model models/best_model.pth --max-iterations 10
    
  自动循环训练模式:
    python main.py auto-train --max-cycles 10 --device cuda --epochs 50
    python main.py auto-train --config-file training_config.json --output-dir models/auto
    python main.py auto-train --batch-size 16 --learning-rate 0.0001 --transparent-ratio 0.7
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
    
    # 循环修复命令
    repair_parser = subparsers.add_parser('repair', help='循环检测和修复水印')
    
    repair_parser.add_argument('--input', type=str, default='data/test',
                              help='输入图像路径或目录')
    repair_parser.add_argument('--output', type=str, default='data/result',
                              help='输出目录')
    repair_parser.add_argument('--model', type=str, default='models/checkpoint_epoch_020.pth',
                              help='模型文件路径')
    repair_parser.add_argument('--config', type=str, help='配置文件路径')
    repair_parser.add_argument('--device', type=str, 
                              default='auto', help='计算设备 (默认: auto)')
    repair_parser.add_argument('--threshold', type=float, default=0.3, 
                              help='二值化阈值 (默认: 0.3)')
    repair_parser.add_argument('--max-iterations', type=int, default=10,
                              help='最大迭代次数 (默认: 10)')
    repair_parser.add_argument('--save-mask', default=True,
                            help='保存预测掩码')
    repair_parser.add_argument('--watermark-threshold', type=float, default=0.01,
                              help='水印面积阈值，低于此值认为修复完成 (默认: 0.01)')
    repair_parser.add_argument('--min-detection-threshold', type=float, default=0.001,
                              help='最小检测阈值，低于此值认为模型未检测到水印 (默认: 0.001)')
    repair_parser.add_argument('--iopaint-model', type=str, default='lama',
                              help='IOPaint修复模型 (默认: lama)')
    repair_parser.add_argument('--limit', type=int, default=10,
                              help='随机选择的图片数量限制')
    
    # 添加视频生成参数
    repair_parser.add_argument('--generate-video', action='store_true', help='修复完成后自动生成对比视频')
    repair_parser.add_argument('--video-width', type=int, default=1920, help='视频宽度 (默认: 1920)')
    repair_parser.add_argument('--video-height', type=int, default=1080, help='视频高度 (默认: 1080)')
    repair_parser.add_argument('--duration', type=float, default=2.0, help='每张图片展示时长(秒) (默认: 2.0)')
    repair_parser.add_argument('--fps', type=int, default=30, help='视频帧率 (默认: 30)')
    
    # 处理模式选择
    repair_parser.add_argument('--optimize', action='store_true', help='启用优化批处理模式')
    repair_parser.add_argument('--folder-mode', action='store_true', help='启用文件夹迭代处理模式')
    repair_parser.add_argument('--batch-size', type=int, default=10, help='批处理大小（优化模式）')
    repair_parser.add_argument('--pause-interval', type=int, default=50, help='暂停清理间隔（优化模式）')
    
    # 自动循环训练命令
    auto_train_parser = subparsers.add_parser('auto', help='自动循环训练')
    auto_train_parser.add_argument('--config-file', type=str, help='JSON配置文件路径')
    auto_train_parser.add_argument('--config', type=str, help='训练配置文件路径',
                                  default='src/configs/unet_watermark.yaml')
    auto_train_parser.add_argument('--project-root', type=str, help='项目根目录')
    auto_train_parser.add_argument('--max-cycles', type=int, default=100,
                                  help='最大循环次数 (默认: 100)')
    auto_train_parser.add_argument('--device', type=str, default='auto',
                                  help='训练设备 (默认: auto)')
    auto_train_parser.add_argument('--epochs', type=int, default=50,
                                  help='每轮训练的epoch数 (默认: 50)')
    auto_train_parser.add_argument('--batch-size', type=int, default=8,
                                  help='批次大小 (默认: 8)')
    auto_train_parser.add_argument('--learning-rate', type=float, default=0.001,
                                  help='学习率 (默认: 0.001)')
    auto_train_parser.add_argument('--output-dir', type=str, default='models/auto',
                                  help='输出目录 (默认: models/auto)')
    auto_train_parser.add_argument('--model-selection-samples', type=int, default=1000,
                                  help='模型选择时的样本数量 (默认: 1000)')
    auto_train_parser.add_argument('--prediction-limit', type=int, default=100,
                                  help='预测时的图片数量限制 (默认: 100)')
    auto_train_parser.add_argument('--transparent-ratio', type=float, default=0.6,
                                  help='透明水印比例 (默认: 0.6)')
    auto_train_parser.add_argument('--logos-dir', type=str, default='data/WatermarkDataset/logos',
                                  help='水印图片目录 (默认: data/WatermarkDataset/logos)')
    
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
        elif args.command == 'repair':
            repair_command(args)
        elif args.command == 'auto':
            auto_train_command(args)
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        print("\n用户中断操作")
    except Exception as e:
        print(f"执行命令时出现错误: {str(e)}")
        sys.exit(1)