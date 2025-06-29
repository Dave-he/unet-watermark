#!/usr/bin/env python3
"""
文字水印检测快速设置脚本
自动配置环境并提供快速测试功能
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import logging
import argparse

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextWatermarkSetup:
    """文字水印检测设置器"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.src_dir = self.project_root / "src"
        self.models_dir = self.project_root / "models"
        self.configs_dir = self.src_dir / "configs"
        
    def check_environment(self):
        """检查环境依赖"""
        logger.info("检查环境依赖...")
        
        # 检查Python版本
        python_version = sys.version_info
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
            logger.error("需要Python 3.8或更高版本")
            return False
        
        logger.info(f"Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # 检查必需的包
        required_packages = [
            'torch', 'torchvision', 'opencv-python', 'easyocr', 
            'matplotlib', 'scikit-learn', 'tqdm', 'numpy', 'pillow'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                logger.info(f"✓ {package} 已安装")
            except ImportError:
                missing_packages.append(package)
                logger.warning(f"✗ {package} 未安装")
        
        if missing_packages:
            logger.info("安装缺失的包...")
            self.install_packages(missing_packages)
        
        return True
    
    def install_packages(self, packages):
        """安装缺失的包"""
        for package in packages:
            try:
                logger.info(f"安装 {package}...")
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                logger.info(f"✓ {package} 安装成功")
            except subprocess.CalledProcessError as e:
                logger.error(f"✗ {package} 安装失败: {e}")
    
    def check_project_structure(self):
        """检查项目结构"""
        logger.info("检查项目结构...")
        
        required_dirs = [
            self.src_dir,
            self.configs_dir,
            self.models_dir
        ]
        
        for dir_path in required_dirs:
            if not dir_path.exists():
                logger.info(f"创建目录: {dir_path}")
                dir_path.mkdir(parents=True, exist_ok=True)
            else:
                logger.info(f"✓ 目录存在: {dir_path}")
        
        # 检查关键文件
        key_files = [
            self.src_dir / "predict.py",
            self.src_dir / "cli.py",
            self.configs_dir / "unet_text_watermark.yaml",
            "train_text_watermark.py",
            "test_text_watermark.py"
        ]
        
        missing_files = []
        for file_path in key_files:
            if not file_path.exists():
                missing_files.append(file_path)
                logger.warning(f"✗ 文件缺失: {file_path}")
            else:
                logger.info(f"✓ 文件存在: {file_path}")
        
        if missing_files:
            logger.warning("部分关键文件缺失，请确保已正确下载所有文件")
            return False
        
        return True
    
    def create_sample_data(self):
        """创建示例数据"""
        logger.info("创建示例数据目录...")
        
        sample_dirs = [
            "sample_data/images",
            "sample_data/masks",
            "sample_data/test_images",
            "results/watermark_repair",
            "results/text_repair",
            "results/test_results"
        ]
        
        for dir_name in sample_dirs:
            dir_path = self.project_root / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"✓ 创建目录: {dir_path}")
        
        # 创建README文件
        sample_readme = self.project_root / "sample_data" / "README.md"
        if not sample_readme.exists():
            with open(sample_readme, 'w', encoding='utf-8') as f:
                f.write("# 示例数据说明\n\n")
                f.write("## 目录结构\n\n")
                f.write("- `images/`: 训练用的原始图像\n")
                f.write("- `masks/`: 对应的标注mask\n")
                f.write("- `test_images/`: 测试图像\n\n")
                f.write("## 数据格式\n\n")
                f.write("- 图像格式: JPG, PNG\n")
                f.write("- Mask格式: PNG (二值图像)\n")
                f.write("- 命名规则: 图像和mask文件名应该对应\n")
    
    def create_quick_test_script(self):
        """创建快速测试脚本"""
        logger.info("创建快速测试脚本...")
        
        test_script_content = '''#!/bin/bash

# 文字水印检测快速测试脚本

echo "=== 文字水印检测快速测试 ==="

# 检查测试图像
if [ ! -d "sample_data/test_images" ] || [ -z "$(ls -A sample_data/test_images)" ]; then
    echo "错误: sample_data/test_images 目录为空"
    echo "请添加一些测试图像到该目录"
    exit 1
fi

# 创建结果目录
mkdir -p results/quick_test

echo "1. 测试基本水印检测..."
python test_text_watermark.py \
    --input sample_data/test_images \
    --output results/quick_test \
    --batch

echo "2. 测试CLI修复功能..."
python src/cli.py repair \
    --input-dir sample_data/test_images \
    --output-dir results/quick_test/repaired \
    --watermark-model lama \
    --text-model lama

echo "3. 生成测试报告..."
python -c "
import os
from pathlib import Path

results_dir = Path('results/quick_test')
if results_dir.exists():
    print('\\n=== 测试结果 ===')
    for item in results_dir.iterdir():
        if item.is_dir():
            print(f'目录: {item.name}')
            for subitem in item.iterdir():
                print(f'  - {subitem.name}')
        else:
            print(f'文件: {item.name}')
    print('\\n测试完成！请查看 results/quick_test 目录中的结果。')
else:
    print('测试结果目录不存在，可能测试失败。')
"

echo "快速测试完成！"
'''
        
        test_script_path = self.project_root / "quick_test.sh"
        with open(test_script_path, 'w', encoding='utf-8') as f:
            f.write(test_script_content)
        
        # 设置执行权限
        os.chmod(test_script_path, 0o755)
        logger.info(f"✓ 创建快速测试脚本: {test_script_path}")
    
    def create_config_templates(self):
        """创建配置模板"""
        logger.info("检查配置文件...")
        
        # 检查是否已存在配置文件
        config_file = self.configs_dir / "unet_text_watermark.yaml"
        if config_file.exists():
            logger.info(f"✓ 配置文件已存在: {config_file}")
            return
        
        # 如果不存在，创建基本配置模板
        logger.info("创建基本配置模板...")
        
        basic_config = '''
# UNet文字水印检测配置
# 这是一个基本模板，请根据实际需求调整

DEVICE: "cuda"  # 或 "cpu"

MODEL:
  NAME: "UnetPlusPlus"
  ENCODER_NAME: "efficientnet-b3"
  ENCODER_WEIGHTS: "imagenet"
  IN_CHANNELS: 3
  CLASSES: 1

DATA:
  ROOT_DIR: "./sample_data"
  IMAGE_SIZE: 512
  NUM_WORKERS: 4
  AUGMENTATION_TYPE: "text_specific"
  TEXT_ENHANCEMENT: true
  CONTRAST_BOOST: 1.2
  EDGE_ENHANCEMENT: true

TRAIN:
  BATCH_SIZE: 8
  EPOCHS: 1500
  LR: 0.003
  WEIGHT_DECAY: 0.0001
  USE_AMP: true
  GRADIENT_CLIP: 1.0
  OUTPUT_DIR: "./results/training"
  MODEL_SAVE_PATH: "./models/text_watermark_model.pth"
  LOG_INTERVAL: 10
  SAVE_INTERVAL: 100
  USE_EARLY_STOPPING: true
  EARLY_STOPPING_PATIENCE: 50

LOSS:
  NAME: "CombinedLoss"
  BCE_WEIGHT: 0.3
  DICE_WEIGHT: 0.5
  FOCAL_WEIGHT: 0.2

OPTIMIZER:
  NAME: "AdamW"
  LR_SCHEDULER: "CosineAnnealingWarmRestarts"
  SCHEDULER_T_0: 50
  SCHEDULER_T_MULT: 2
  SCHEDULER_ETA_MIN: 0.00001

PREDICTION:
  CONFIDENCE_THRESHOLD: 0.5
  MIN_AREA_THRESHOLD: 100
  TEXT_AREA_THRESHOLD: 50
  TEXT_ASPECT_RATIO_MIN: 0.1
  TEXT_ASPECT_RATIO_MAX: 10.0

VALIDATION:
  TEXT_SCORE_THRESHOLD: 0.6
  MIXED_SCORE_THRESHOLD: 0.4
  EDGE_DENSITY_THRESHOLD: 0.3
'''
        
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(basic_config)
        
        logger.info(f"✓ 创建配置文件: {config_file}")
    
    def run_basic_test(self):
        """运行基本测试"""
        logger.info("运行基本功能测试...")
        
        try:
            # 测试导入
            sys.path.append(str(self.src_dir))
            
            # 测试基本模块导入
            logger.info("测试模块导入...")
            
            try:
                from predict import WatermarkPredictor
                logger.info("✓ WatermarkPredictor 导入成功")
            except ImportError as e:
                logger.error(f"✗ WatermarkPredictor 导入失败: {e}")
                return False
            
            try:
                from ocr.easy_ocr import EasyOCRDetector
                logger.info("✓ EasyOCRDetector 导入成功")
            except ImportError as e:
                logger.error(f"✗ EasyOCRDetector 导入失败: {e}")
                return False
            
            # 测试配置加载
            try:
                from configs.config import get_cfg_defaults
                cfg = get_cfg_defaults()
                logger.info("✓ 配置系统正常")
            except Exception as e:
                logger.error(f"✗ 配置系统错误: {e}")
                return False
            
            logger.info("✓ 基本功能测试通过")
            return True
            
        except Exception as e:
            logger.error(f"基本测试失败: {e}")
            return False
    
    def print_usage_guide(self):
        """打印使用指南"""
        print("\n" + "="*60)
        print("🎉 文字水印检测环境设置完成！")
        print("="*60)
        
        print("\n📋 快速开始:")
        print("1. 添加测试图像到 sample_data/test_images/ 目录")
        print("2. 运行快速测试: ./quick_test.sh")
        print("3. 查看结果: results/quick_test/")
        
        print("\n🔧 主要功能:")
        print("• 智能文字水印检测")
        print("• 分离的水印和文字修复")
        print("• 批量处理和测试")
        print("• 专用模型训练")
        
        print("\n📖 详细文档:")
        print("• README_TEXT_WATERMARK.md - 完整使用指南")
        print("• src/configs/unet_text_watermark.yaml - 配置说明")
        
        print("\n🚀 常用命令:")
        print("# 单图测试")
        print("python test_text_watermark.py --input image.jpg --output results/")
        
        print("\n# 批量修复")
        print("python src/cli.py repair --input-dir images/ --output-dir results/ \\")
        print("    --watermark-model lama --text-model mat")
        
        print("\n# 训练专用模型")
        print("python train_text_watermark.py --config src/configs/unet_text_watermark.yaml")
        
        print("\n" + "="*60)
    
    def setup(self, run_test=True):
        """执行完整设置"""
        logger.info("开始文字水印检测环境设置...")
        
        # 1. 检查环境
        if not self.check_environment():
            logger.error("环境检查失败")
            return False
        
        # 2. 检查项目结构
        if not self.check_project_structure():
            logger.error("项目结构检查失败")
            return False
        
        # 3. 创建示例数据目录
        self.create_sample_data()
        
        # 4. 创建配置模板
        self.create_config_templates()
        
        # 5. 创建快速测试脚本
        self.create_quick_test_script()
        
        # 6. 运行基本测试
        if run_test:
            if not self.run_basic_test():
                logger.warning("基本测试失败，但设置已完成")
        
        # 7. 打印使用指南
        self.print_usage_guide()
        
        logger.info("设置完成！")
        return True

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='文字水印检测快速设置')
    parser.add_argument('--no-test', action='store_true',
                       help='跳过基本功能测试')
    parser.add_argument('--install-deps', action='store_true',
                       help='强制重新安装依赖')
    
    args = parser.parse_args()
    
    setup = TextWatermarkSetup()
    
    # 强制安装依赖
    if args.install_deps:
        required_packages = [
            'torch', 'torchvision', 'opencv-python', 'easyocr', 
            'matplotlib', 'scikit-learn', 'tqdm', 'numpy', 'pillow'
        ]
        setup.install_packages(required_packages)
    
    # 执行设置
    success = setup.setup(run_test=not args.no_test)
    
    if success:
        print("\n✅ 设置成功完成！")
        sys.exit(0)
    else:
        print("\n❌ 设置过程中出现问题，请检查日志")
        sys.exit(1)

if __name__ == '__main__':
    main()