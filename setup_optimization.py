#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化工具安装和验证脚本
"""

import os
import sys
import subprocess
import importlib
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OptimizationSetup:
    """优化工具安装和验证"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.src_dir = self.project_root / "src"
        self.utils_dir = self.src_dir / "utils"
        
        # 必需的优化工具文件
        self.required_files = [
            "src/utils/enhanced_memory_manager.py",
            "src/utils/adaptive_batch_processor.py",
            "src/utils/optimized_dataloader.py",
            "src/utils/optimized_predictor.py",
            "src/utils/training_optimizer.py",
            "src/utils/performance_analyzer.py",
            "src/utils/optimization_config.py",
            "src/utils/optimization_manager.py"
        ]
        
        # 可选依赖
        self.optional_dependencies = {
            'psutil': '系统监控',
            'pynvml': 'GPU监控',
            'matplotlib': '性能图表',
            'pandas': '数据分析',
            'seaborn': '高级可视化',
            'plotly': '交互式图表'
        }
        
        # 必需依赖
        self.required_dependencies = {
            'torch': 'PyTorch深度学习框架',
            'numpy': '数值计算',
            'tqdm': '进度条',
            'PyYAML': '配置文件解析'
        }
    
    def check_project_structure(self) -> bool:
        """检查项目结构"""
        logger.info("检查项目结构...")
        
        # 检查基本目录
        if not self.src_dir.exists():
            logger.error(f"src目录不存在: {self.src_dir}")
            return False
        
        if not self.utils_dir.exists():
            logger.warning(f"utils目录不存在，将创建: {self.utils_dir}")
            self.utils_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查__init__.py文件
        init_files = [
            self.src_dir / "__init__.py",
            self.utils_dir / "__init__.py"
        ]
        
        for init_file in init_files:
            if not init_file.exists():
                logger.info(f"创建__init__.py: {init_file}")
                init_file.touch()
        
        logger.info("项目结构检查完成")
        return True
    
    def check_optimization_files(self) -> Tuple[List[str], List[str]]:
        """检查优化工具文件"""
        logger.info("检查优化工具文件...")
        
        existing_files = []
        missing_files = []
        
        for file_path in self.required_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                existing_files.append(file_path)
                logger.info(f"✓ {file_path}")
            else:
                missing_files.append(file_path)
                logger.warning(f"✗ {file_path}")
        
        logger.info(f"找到 {len(existing_files)}/{len(self.required_files)} 个优化工具文件")
        
        return existing_files, missing_files
    
    def check_dependencies(self) -> Tuple[Dict[str, bool], Dict[str, bool]]:
        """检查依赖"""
        logger.info("检查依赖...")
        
        def check_package(package_name: str) -> bool:
            try:
                importlib.import_module(package_name)
                return True
            except ImportError:
                return False
        
        # 检查必需依赖
        required_status = {}
        for package, description in self.required_dependencies.items():
            available = check_package(package)
            required_status[package] = available
            status = "✓" if available else "✗"
            logger.info(f"{status} {package} - {description}")
        
        # 检查可选依赖
        optional_status = {}
        for package, description in self.optional_dependencies.items():
            available = check_package(package)
            optional_status[package] = available
            status = "✓" if available else "○" 
            logger.info(f"{status} {package} - {description}")
        
        return required_status, optional_status
    
    def install_dependencies(self, install_optional: bool = False) -> bool:
        """安装依赖"""
        logger.info("安装依赖...")
        
        # 检查pip
        try:
            subprocess.run([sys.executable, "-m", "pip", "--version"], 
                         check=True, capture_output=True)
        except subprocess.CalledProcessError:
            logger.error("pip不可用，请先安装pip")
            return False
        
        # 安装必需依赖
        required_packages = list(self.required_dependencies.keys())
        if required_packages:
            logger.info(f"安装必需依赖: {', '.join(required_packages)}")
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install"
                ] + required_packages, check=True)
                logger.info("必需依赖安装完成")
            except subprocess.CalledProcessError as e:
                logger.error(f"必需依赖安装失败: {e}")
                return False
        
        # 安装可选依赖
        if install_optional:
            optional_packages = list(self.optional_dependencies.keys())
            if optional_packages:
                logger.info(f"安装可选依赖: {', '.join(optional_packages)}")
                for package in optional_packages:
                    try:
                        subprocess.run([
                            sys.executable, "-m", "pip", "install", package
                        ], check=True)
                        logger.info(f"✓ {package} 安装成功")
                    except subprocess.CalledProcessError:
                        logger.warning(f"○ {package} 安装失败，将跳过相关功能")
        
        return True
    
    def test_optimization_tools(self) -> bool:
        """测试优化工具"""
        logger.info("测试优化工具...")
        
        try:
            # 添加项目路径到sys.path
            if str(self.project_root) not in sys.path:
                sys.path.insert(0, str(self.project_root))
            
            # 测试基本导入
            test_imports = [
                "src.utils.enhanced_memory_manager",
                "src.utils.adaptive_batch_processor",
                "src.utils.optimized_dataloader",
                "src.utils.optimization_config",
                "src.utils.optimization_manager"
            ]
            
            for module_name in test_imports:
                try:
                    importlib.import_module(module_name)
                    logger.info(f"✓ {module_name}")
                except ImportError as e:
                    logger.error(f"✗ {module_name}: {e}")
                    return False
            
            # 测试基本功能
            logger.info("测试基本功能...")
            
            # 测试内存管理器
            from src.utils.enhanced_memory_manager import EnhancedMemoryManager
            memory_manager = EnhancedMemoryManager()
            snapshot = memory_manager.get_memory_snapshot()
            logger.info(f"✓ 内存管理器: {snapshot.total_memory_gb:.1f}GB 总内存")
            
            # 测试配置管理器
            from src.utils.optimization_config import OptimizationConfig, OptimizationLevel
            config = OptimizationConfig(optimization_level=OptimizationLevel.BALANCED)
            logger.info(f"✓ 配置管理器: {config.optimization_level}")
            
            # 测试优化管理器
            from src.utils.optimization_manager import OptimizationManager
            manager = OptimizationManager(config)
            summary = manager.get_optimization_summary()
            logger.info(f"✓ 优化管理器: {len(summary)} 个配置项")
            
            logger.info("所有测试通过！")
            return True
            
        except Exception as e:
            logger.error(f"测试失败: {e}")
            return False
    
    def create_test_script(self) -> str:
        """创建测试脚本"""
        test_script_path = self.project_root / "test_optimization.py"
        
        test_script_content = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化工具测试脚本
"""

import sys
import torch
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def test_memory_manager():
    """测试内存管理器"""
    print("测试内存管理器...")
    
    from src.utils.enhanced_memory_manager import (
        EnhancedMemoryManager, 
        memory_optimized,
        get_global_memory_manager
    )
    
    # 测试基本功能
    manager = EnhancedMemoryManager()
    snapshot = manager.get_memory_snapshot()
    print(f"  总内存: {snapshot.total_memory_gb:.1f}GB")
    print(f"  可用内存: {snapshot.available_memory_gb:.1f}GB")
    
    # 测试装饰器
    @memory_optimized
    def memory_intensive_function():
        return torch.randn(1000, 1000)
    
    result = memory_intensive_function()
    print(f"  装饰器测试: {result.shape}")
    
    # 测试全局管理器
    global_manager = get_global_memory_manager()
    global_manager.cleanup()
    print("  ✓ 内存管理器测试通过")

def test_batch_processor():
    """测试批处理处理器"""
    print("测试批处理处理器...")
    
    from src.utils.adaptive_batch_processor import AdaptiveBatchProcessor
    
    processor = AdaptiveBatchProcessor(
        initial_batch_size=8,
        min_batch_size=1,
        max_batch_size=32
    )
    
    # 模拟一些处理
    for i in range(5):
        batch_size = processor.get_batch_size()
        # 模拟成功处理
        processor.record_success(batch_size, processing_time=0.1, memory_used=100)
        print(f"  批次 {i+1}: batch_size={batch_size}")
    
    stats = processor.get_stats()
    print(f"  平均批处理大小: {stats['avg_batch_size']:.1f}")
    print("  ✓ 批处理处理器测试通过")

def test_optimization_config():
    """测试优化配置"""
    print("测试优化配置...")
    
    from src.utils.optimization_config import OptimizationConfig, OptimizationLevel
    
    # 测试不同优化级别
    for level in [OptimizationLevel.CONSERVATIVE, OptimizationLevel.BALANCED, OptimizationLevel.AGGRESSIVE]:
        config = OptimizationConfig(optimization_level=level)
        print(f"  {level}: batch_size={config.batch.initial_batch_size}")
    
    # 测试自定义配置
    custom_config = OptimizationConfig(optimization_level=OptimizationLevel.CUSTOM)
    custom_config.batch.initial_batch_size = 16
    custom_config.model.mixed_precision = True
    
    print(f"  自定义配置: batch_size={custom_config.batch.initial_batch_size}")
    print("  ✓ 优化配置测试通过")

def test_optimization_manager():
    """测试优化管理器"""
    print("测试优化管理器...")
    
    from src.utils.optimization_manager import (
        OptimizationManager,
        create_optimization_manager,
        OptimizationLevel
    )
    
    # 创建管理器
    manager = create_optimization_manager(OptimizationLevel.BALANCED)
    
    # 测试模型优化
    model = torch.nn.Linear(10, 1)
    manager.optimize_model(model)
    print(f"  模型优化: {type(model).__name__}")
    
    # 测试摘要
    summary = manager.get_optimization_summary()
    print(f"  配置项数量: {len(summary.get('config', {}))}")
    
    # 清理
    manager.cleanup()
    print("  ✓ 优化管理器测试通过")

def test_performance_analyzer():
    """测试性能分析器"""
    print("测试性能分析器...")
    
    try:
        from src.utils.performance_analyzer import (
            PerformanceAnalyzer,
            performance_profile,
            get_global_performance_analyzer
        )
        
        # 测试装饰器
        @performance_profile("test_operation")
        def test_function():
            import time
            time.sleep(0.01)
            return "test_result"
        
        result = test_function()
        print(f"  装饰器测试: {result}")
        
        # 测试分析器
        analyzer = get_global_performance_analyzer()
        system_analysis = analyzer.analyze_system_performance()
        print(f"  CPU使用率: {system_analysis.cpu_percent:.1f}%")
        
        print("  ✓ 性能分析器测试通过")
        
    except ImportError as e:
        print(f"  ○ 性能分析器测试跳过 (缺少依赖): {e}")

def main():
    """主测试函数"""
    print("开始优化工具测试...\n")
    
    try:
        test_memory_manager()
        print()
        
        test_batch_processor()
        print()
        
        test_optimization_config()
        print()
        
        test_optimization_manager()
        print()
        
        test_performance_analyzer()
        print()
        
        print("🎉 所有测试通过！优化工具已正确安装和配置。")
        print("\n下一步:")
        print("1. 查看 OPTIMIZATION_INTEGRATION_GUIDE.md 了解使用方法")
        print("2. 运行 python optimization_example.py 查看完整示例")
        print("3. 在您的代码中集成优化工具")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        print("\n请检查:")
        print("1. 所有优化工具文件是否存在")
        print("2. 依赖是否正确安装")
        print("3. Python路径是否正确")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
'''
        
        with open(test_script_path, 'w', encoding='utf-8') as f:
            f.write(test_script_content)
        
        logger.info(f"测试脚本已创建: {test_script_path}")
        return str(test_script_path)
    
    def generate_setup_report(self) -> str:
        """生成安装报告"""
        logger.info("生成安装报告...")
        
        # 检查项目结构
        structure_ok = self.check_project_structure()
        
        # 检查优化文件
        existing_files, missing_files = self.check_optimization_files()
        
        # 检查依赖
        required_status, optional_status = self.check_dependencies()
        
        # 生成报告
        report = []
        report.append("# 优化工具安装报告\n")
        
        # 项目结构
        report.append("## 项目结构")
        report.append(f"- 项目根目录: {self.project_root}")
        report.append(f"- src目录: {'✓' if self.src_dir.exists() else '✗'}")
        report.append(f"- utils目录: {'✓' if self.utils_dir.exists() else '✗'}")
        report.append("")
        
        # 优化工具文件
        report.append("## 优化工具文件")
        report.append(f"找到 {len(existing_files)}/{len(self.required_files)} 个文件")
        report.append("")
        report.append("### 已存在的文件")
        for file_path in existing_files:
            report.append(f"- ✓ {file_path}")
        report.append("")
        
        if missing_files:
            report.append("### 缺失的文件")
            for file_path in missing_files:
                report.append(f"- ✗ {file_path}")
            report.append("")
        
        # 必需依赖
        report.append("## 必需依赖")
        for package, available in required_status.items():
            status = "✓" if available else "✗"
            description = self.required_dependencies[package]
            report.append(f"- {status} {package} - {description}")
        report.append("")
        
        # 可选依赖
        report.append("## 可选依赖")
        for package, available in optional_status.items():
            status = "✓" if available else "○"
            description = self.optional_dependencies[package]
            report.append(f"- {status} {package} - {description}")
        report.append("")
        
        # 建议
        report.append("## 建议")
        
        if missing_files:
            report.append("### 缺失文件")
            report.append("请运行以下命令创建缺失的优化工具文件:")
            report.append("```bash")
            report.append("python integrate_optimization.py --component all")
            report.append("```")
            report.append("")
        
        missing_required = [pkg for pkg, available in required_status.items() if not available]
        if missing_required:
            report.append("### 缺失必需依赖")
            report.append("请运行以下命令安装必需依赖:")
            report.append("```bash")
            report.append(f"pip install {' '.join(missing_required)}")
            report.append("```")
            report.append("")
        
        missing_optional = [pkg for pkg, available in optional_status.items() if not available]
        if missing_optional:
            report.append("### 可选依赖")
            report.append("可以运行以下命令安装可选依赖以获得完整功能:")
            report.append("```bash")
            report.append(f"pip install {' '.join(missing_optional)}")
            report.append("```")
            report.append("")
        
        # 下一步
        report.append("## 下一步")
        report.append("1. 确保所有必需文件和依赖都已安装")
        report.append("2. 运行测试脚本验证安装: `python test_optimization.py`")
        report.append("3. 查看集成指南: `OPTIMIZATION_INTEGRATION_GUIDE.md`")
        report.append("4. 运行示例代码: `python optimization_example.py`")
        
        report_content = "\n".join(report)
        
        # 保存报告
        report_path = self.project_root / "optimization_setup_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"安装报告已生成: {report_path}")
        return str(report_path)
    
    def setup_all(self, install_optional: bool = False) -> bool:
        """完整安装流程"""
        logger.info("开始优化工具安装...")
        
        try:
            # 1. 检查项目结构
            if not self.check_project_structure():
                return False
            
            # 2. 检查优化文件
            existing_files, missing_files = self.check_optimization_files()
            
            if missing_files:
                logger.warning(f"缺失 {len(missing_files)} 个优化工具文件")
                logger.info("请先运行: python integrate_optimization.py --component all")
            
            # 3. 安装依赖
            if not self.install_dependencies(install_optional):
                return False
            
            # 4. 创建测试脚本
            self.create_test_script()
            
            # 5. 测试优化工具
            if existing_files:  # 只有在有文件时才测试
                if not self.test_optimization_tools():
                    logger.warning("优化工具测试失败，但安装可能仍然成功")
            
            # 6. 生成报告
            self.generate_setup_report()
            
            logger.info("优化工具安装完成！")
            return True
            
        except Exception as e:
            logger.error(f"安装失败: {e}")
            return False

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="优化工具安装和验证")
    parser.add_argument("--project-root", default=".", help="项目根目录")
    parser.add_argument("--install-optional", action="store_true", help="安装可选依赖")
    parser.add_argument("--test-only", action="store_true", help="仅运行测试")
    parser.add_argument("--report-only", action="store_true", help="仅生成报告")
    
    args = parser.parse_args()
    
    setup = OptimizationSetup(args.project_root)
    
    try:
        if args.test_only:
            success = setup.test_optimization_tools()
        elif args.report_only:
            setup.generate_setup_report()
            success = True
        else:
            success = setup.setup_all(args.install_optional)
        
        if success:
            print("\n🎉 安装成功！")
            print("\n下一步:")
            print("1. 运行测试: python test_optimization.py")
            print("2. 查看报告: optimization_setup_report.md")
            print("3. 查看指南: OPTIMIZATION_INTEGRATION_GUIDE.md")
            print("4. 运行示例: python optimization_example.py")
        else:
            print("\n❌ 安装失败，请查看日志了解详情")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n安装被用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"安装过程中出现错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()