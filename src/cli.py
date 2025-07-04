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
        
        # 开始训练 - 传递cfg、resume和use_blurred_mask参数
        train(cfg, resume_from=getattr(args, 'resume', None), 
              use_blurred_mask=getattr(args, 'use_blurred_mask', True))
        
        print("\n训练完成！")
        print(f"最佳模型已保存到: {cfg.TRAIN.MODEL_SAVE_PATH}")
        
    except Exception as e:
        print(f"训练过程中出现错误: {str(e)}")
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
    # 处理模型参数的向后兼容性
    watermark_model = getattr(args, 'watermark_model', None) or args.iopaint_model
    text_model = getattr(args, 'text_model', None) or args.iopaint_model
    
    print(f"水印修复模型: {watermark_model}")
    print(f"文字修复模型: {text_model}")
    print(f"处理超时时间: 300秒")
    
    # 处理流程控制参数
    use_unet = getattr(args, 'use_unet', True) and not getattr(args, 'no_unet', False)
    use_ocr = getattr(args, 'use_ocr', True) and not getattr(args, 'no_ocr', False)
    
    print(f"启用UNet水印检测: {'是' if use_unet else '否'}")
    if use_ocr:
        print(f"启用OCR文字检测: 是 (语言: {', '.join(args.ocr_languages)})")
    else:
        print(f"启用OCR文字检测: 否")
    if getattr(args, 'video', False):
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
        
        # 执行文件夹批量修复（新的4步批量处理模式）
        results = predictor.process_folder_batch(
            input_folder=args.input,
            output_folder=args.output,
            watermark_model=watermark_model,
            text_model=text_model,
            use_unet=use_unet,
            use_ocr=use_ocr,
            ocr_languages=getattr(args, 'ocr_languages', ['en', 'ch_sim']),
            ocr_engine=getattr(args, 'ocr_engine', 'easy'),
            timeout=getattr(args, 'timeout', 300),
            save_intermediate=getattr(args, 'save_intermediate', True),
            merge_masks=getattr(args, 'merge_masks', True),
            limit=getattr(args, 'limit', None),
            steps=getattr(args, 'steps', 3)
        )
        
        print("\n文件夹修复完成！")
        if results['status'] == 'success':
            print(f"总图片数: {results.get('total_images', 0)}")
            print(f"成功处理: {results.get('successful_images', 0)}")
            print(f"成功率: {results.get('success_rate', 0):.1f}%")
            print(f"总处理时间: {results.get('total_time', 0):.2f}秒")
            
            # 显示各步骤统计
            if 'step1_watermark_mask' in results:
                print(f"步骤1 - 水印mask预测: {results['step1_watermark_mask'].get('successful', 0)} 张成功")
            if 'step2_watermark_repair' in results:
                print(f"步骤2 - 水印修复: {results['step2_watermark_repair'].get('successful', 0)} 张成功")
            if 'step3_text_mask' in results:
                print(f"步骤3 - 文字mask提取: {results['step3_text_mask'].get('successful', 0)} 张成功")
            if 'step4_text_repair' in results:
                print(f"步骤4 - 文字修复: {results['step4_text_repair'].get('successful', 0)} 张成功")
            if 'step5_merge_masks' in results:
                print(f"步骤5 - mask合并: {results['step5_merge_masks'].get('successful', 0)} 张成功")
        else:
            print(f"批量处理失败: {results.get('message', '未知错误')}")
        
        # 保存结果摘要
        summary_path = os.path.join(args.output, 'repair_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n结果摘要已保存: {summary_path}")
        
        # 检查是否需要生成视频
        if getattr(args, 'video', False):
            print("\n开始生成对比视频...")
            try:
                from scripts.video_generator import VideoGenerator
                
                # 检查mask目录
                mask_dir = os.path.join(args.output, 'masks')
                if not os.path.exists(mask_dir):
                    mask_dir = None
                
                # 创建视频生成器
                generator = VideoGenerator(
                    input_dir=args.video_input,
                    repair_dir=args.output,  # 使用最终修复结果
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
        'model_selection_samples': args.samples,
        'prediction_limit': args.prediction_limit,
        'transparent_ratio': args.transparent_ratio,
            'text_watermark_ratio': getattr(args, 'text_watermark_ratio', 0.3),
            'mixed_watermark_ratio': getattr(args, 'mixed_watermark_ratio', 0.2),
            'use_ocr_mask': getattr(args, 'use_ocr_mask', True),
        'logos_dir': args.logos_dir,
        'use_blurred_mask': getattr(args, 'use_blurred_mask', True)
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
                             default="src/configs/unet_watermark_large.yaml")
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
    
    # 添加模糊mask参数
    train_parser.add_argument('--use-blurred-mask', action='store_true',
                             help='使用模糊的mask进行训练，默认生成精确的mask')
    
    # 循环修复命令
    repair_parser = subparsers.add_parser('repair', help='循环检测和修复水印')
    
    repair_parser.add_argument('--input', type=str, default='data/test',
                              help='输入图像路径或目录')
    repair_parser.add_argument('--output', type=str, default='data/result',
                              help='输出目录')
    repair_parser.add_argument('--model', type=str, default='models/unet_watermark.pth',
                              help='模型文件路径')
    repair_parser.add_argument('--config', type=str, help='配置文件路径',
                              default="src/configs/unet_watermark_large.yaml")
    repair_parser.add_argument('--device', type=str, 
                              default='auto', help='计算设备 (默认: auto)')
    repair_parser.add_argument('--watermark-model', type=str, default='lama',
                              help='水印修复IOPaint模型 (默认: lama)')
    repair_parser.add_argument('--text-model', type=str, default='mat',
                              help='文字修复IOPaint模型 (默认: mat)')
    repair_parser.add_argument('--iopaint-model', type=str, default='lama',
                              help='IOPaint修复模型 (默认: lama, 兼容性参数)')
    repair_parser.add_argument('--timeout', type=int, default=300,
                              help='每步处理超时时间（秒） (默认: 300)')
    repair_parser.add_argument('--steps', type=int, default=3,
                              help='IOPaint迭代修复次数 (默认: 3)')
    repair_parser.add_argument('--save-intermediate', action='store_true', default=True,
                              help='保存中间处理结果 (默认: True)')
    repair_parser.add_argument('--merge-masks', action='store_true', default=True,
                              help='合并水印和文字mask用于视频生成 (默认: True)')
    repair_parser.add_argument('--limit', type=int, help='限制处理的图片数量，随机选择n张图片进行处理')
    
    # 添加流程控制参数
    repair_parser.add_argument('--use-unet', action='store_true', default=True,
                              help='启用UNet模型进行水印检测和擦除 (默认: True)')
    repair_parser.add_argument('--no-unet', action='store_true',
                              help='禁用UNet模型，跳过水印检测步骤')
    repair_parser.add_argument('--use-ocr', action='store_true', default=True,
                              help='启用OCR文字检测，将文字掩码与水印掩码合并 (默认: True)')
    repair_parser.add_argument('--no-ocr', action='store_true',
                              help='禁用OCR文字检测，跳过文字处理步骤')
    repair_parser.add_argument('--ocr-engine', type=str, choices=['paddle', 'easy'], default='easy',
                              help='选择OCR引擎: paddle (PaddleOCR) 或 easy (EasyOCR)')
    repair_parser.add_argument('--ocr-languages', type=str, nargs='+', default=['en', 'ch_sim'],
                              help='OCR支持的语言列表 (默认: en ch_sim)')
    
    # 添加Stable Diffusion修复参数
    repair_parser.add_argument('--use-sd', action='store_true', help='在IOPaint修复后使用Stable Diffusion进行额外修复')
    repair_parser.add_argument('--sd-model', type=str, default='stabilityai/stable-diffusion-3-medium-diffusers',
                              help='Stable Diffusion模型名称 (默认: stabilityai/stable-diffusion-3-medium-diffusers)')
    repair_parser.add_argument('--sd-prompt', type=str, 
                              default='clean image without watermarks, text, or logos, high quality, natural',
                              help='SD修复的正向提示词')
    repair_parser.add_argument('--sd-negative-prompt', type=str,
                              default='watermark, text, logo, signature, blurry, low quality, artifacts',
                              help='SD修复的负向提示词')
    repair_parser.add_argument('--sd-steps', type=int, default=25, help='SD推理步数 (默认: 25)')
    repair_parser.add_argument('--sd-guidance-scale', type=float, default=6.0, help='SD引导强度 (默认: 6.0)')
    repair_parser.add_argument('--sd-strength', type=float, default=0.6, help='SD修复强度 (默认: 0.6)')
    repair_parser.add_argument('--sd-max-mask-ratio', type=float, default=0.25, help='SD最大mask比例 (默认: 0.25)')
    repair_parser.add_argument('--sd-min-area', type=int, default=200, help='SD最小区域面积 (默认: 200)')
    repair_parser.add_argument('--sd-max-area-ratio', type=float, default=0.08, help='SD最大区域比例 (默认: 0.08)')
    
    # 添加视频生成参数
    repair_parser.add_argument('--video', action='store_true', help='修复完成后自动生成对比视频')
    repair_parser.add_argument('--video-input',default='data/test', help='video对比目录')
    repair_parser.add_argument('--video-width', type=int, default=1920, help='视频宽度 (默认: 1920)')
    repair_parser.add_argument('--video-height', type=int, default=1080, help='视频高度 (默认: 1080)')
    repair_parser.add_argument('--duration', type=float, default=2.0, help='每张图片展示时长(秒) (默认: 2.0)')
    repair_parser.add_argument('--fps', type=int, default=30, help='视频帧率 (默认: 30)')
    
    # 新的批量处理模式已内置，不需要额外的模式选择参数
    
    # 自动循环训练命令
    auto_train_parser = subparsers.add_parser('auto', help='自动循环训练')
    auto_train_parser.add_argument('--config-file', type=str, help='JSON配置文件路径')
    auto_train_parser.add_argument('--config', type=str, help='训练配置文件路径',
                                  default='src/configs/unet_watermark_large.yaml')
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
    auto_train_parser.add_argument('--samples', type=int, default=1000,
                                  help='模型选择时的样本数量 (默认: 1000)')
    auto_train_parser.add_argument('--prediction-limit', type=int, default=100,
                                  help='预测时的图片数量限制 (默认: 100)')
    auto_train_parser.add_argument('--transparent-ratio', type=float, default=0.6,
                                  help='透明水印比例 (默认: 0.6)')
    auto_train_parser.add_argument('--text-watermark-ratio', type=float, default=0.5,
                                  help='文字水印样本比例')
    auto_train_parser.add_argument('--mixed-watermark-ratio', type=float, default=0.2,
                                  help='混合水印样本比例')
    auto_train_parser.add_argument('--use-ocr-mask', action='store_true', default=True,
                                  help='使用OCR生成精确文字区域mask')
    auto_train_parser.add_argument('--logos-dir', type=str, default='data/WatermarkDataset/logos',
                                  help='水印图片目录 (默认: data/WatermarkDataset/logos)')
    auto_train_parser.add_argument('--use-blurred-mask', action='store_true',
                                  help='使用模糊的mask进行训练，默认生成精确的mask')
    
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