#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
水印分割系统主启动入口
支持训练和预测模式，提供统一的命令行接口

Usage:
    # 训练模式
    python main.py train --config src/configs/unet_watermark.yaml --device cuda
    
    # 预测模式
    python main.py predict --input data/test --output results --model models/epoch_300.pth
    
    # 查看帮助
    python main.py --help
"""

import sys
from pathlib import Path

# 添加src目录到Python路径
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from cli import main
from auto_train import auto_main

if __name__ == '__main__':
    # 自动训练模式
    if '--auto' in sys.argv:
        auto_main()
    else:
        main()
