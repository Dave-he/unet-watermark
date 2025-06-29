#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化工具使用示例
展示如何在现有代码中集成和使用优化工具
"""

import sys
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 导入优化工具
from src.utils.optimization_manager import (
    OptimizationManager, 
    create_optimization_manager,
    optimize_model,
    optimize_training
)
from src.utils.optimization_config import OptimizationConfig, OptimizationLevel
from src.utils.performance_analyzer import get_global_performance_analyzer, performance_profile
from src.utils.enhanced_memory_manager import get_global_memory_manager, memory_optimized

def create_sample_model():
    """创建示例模型"""
    return nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((8, 8)),
        nn.Flatten(),
        nn.Linear(128 * 8 * 8, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )

def create_sample_dataset(num_samples=1000, image_size=(3, 32, 32)):
    """创建示例数据集"""
    # 生成随机数据
    data = torch.randn(num_samples, *image_size)
    targets = torch.randint(0, 10, (num_samples,))
    
    return TensorDataset(data, targets)

def example_1_basic_optimization():
    """示例1: 基础模型优化"""
    print("\n" + "="*60)
    print("示例1: 基础模型优化")
    print("="*60)
    
    # 创建模型
    model = create_sample_model()
    print(f"原始模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 使用快速优化函数
    optimized_model, manager = optimize_model(model, OptimizationLevel.BALANCED)
    
    # 查看优化结果
    summary = manager.get_optimization_summary()
    print("\n优化摘要:")
    for key, value in summary['config'].items():
        print(f"  {key}: {value}")
    
    # 获取建议
    recommendations = manager.get_recommendations()
    print("\n优化建议:")
    for i, rec in enumerate(recommendations[:3], 1):
        print(f"  {i}. {rec}")
    
    manager.cleanup()
    return optimized_model

def example_2_custom_configuration():
    """示例2: 自定义配置优化"""
    print("\n" + "="*60)
    print("示例2: 自定义配置优化")
    print("="*60)
    
    # 创建自定义配置
    config = OptimizationConfig(optimization_level=OptimizationLevel.CUSTOM)
    
    # 自定义设置
    config.batch.initial_batch_size = 16
    config.batch.max_batch_size = 64
    config.model.mixed_precision = True
    config.model.torch_compile = False  # 避免编译问题
    config.memory.cleanup_threshold = 0.7
    config.dataloader.num_workers = 4
    
    print("自定义配置:")
    config.print_summary()
    
    # 创建优化管理器
    manager = OptimizationManager(config)
    
    # 创建和优化模型
    model = create_sample_model()
    result = manager.optimize_model(model)
    
    print(f"\n优化结果: {result.message}")
    print(f"执行时间: {result.execution_time:.3f}s")
    
    manager.cleanup()
    return model

def example_3_dataloader_optimization():
    """示例3: 数据加载器优化"""
    print("\n" + "="*60)
    print("示例3: 数据加载器优化")
    print("="*60)
    
    # 创建数据集
    dataset = create_sample_dataset(1000)
    
    # 创建优化管理器
    manager = create_optimization_manager(OptimizationLevel.BALANCED)
    
    # 创建优化的数据加载器
    print("创建优化的数据加载器...")
    dataloader = manager.create_optimized_dataloader(
        dataset,
        batch_size=32,
        shuffle=True
    )
    
    # 测试数据加载性能
    print("测试数据加载性能...")
    start_time = time.time()
    
    for i, (data, targets) in enumerate(dataloader):
        if i >= 10:  # 只测试前10个批次
            break
        # 模拟处理时间
        time.sleep(0.01)
    
    end_time = time.time()
    print(f"加载10个批次耗时: {end_time - start_time:.3f}s")
    
    # 获取数据加载器统计
    stats = dataloader.get_stats()
    print("\n数据加载器统计:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    manager.cleanup()

def example_4_memory_optimization():
    """示例4: 内存优化"""
    print("\n" + "="*60)
    print("示例4: 内存优化")
    print("="*60)
    
    # 获取内存管理器
    memory_manager = get_global_memory_manager()
    memory_manager.start_monitoring()
    
    print("初始内存状态:")
    initial_snapshot = memory_manager.get_memory_snapshot()
    print(f"  Python 内存: {initial_snapshot.python_memory_mb:.1f} MB")
    print(f"  进程内存: {initial_snapshot.process_memory_mb:.1f} MB")
    if initial_snapshot.gpu_memory_gb > 0:
        print(f"  GPU 内存: {initial_snapshot.gpu_memory_gb:.1f} GB")
    
    # 使用内存优化装饰器
    @memory_optimized
    def memory_intensive_operation():
        """内存密集型操作"""
        # 创建大量数据
        large_tensors = []
        for i in range(100):
            tensor = torch.randn(1000, 1000)
            large_tensors.append(tensor)
        
        # 模拟处理
        result = sum(t.sum() for t in large_tensors)
        return result
    
    print("\n执行内存密集型操作...")
    result = memory_intensive_operation()
    print(f"操作结果: {result:.2f}")
    
    # 等待内存清理
    time.sleep(2)
    
    print("\n操作后内存状态:")
    final_snapshot = memory_manager.get_memory_snapshot()
    print(f"  Python 内存: {final_snapshot.python_memory_mb:.1f} MB")
    print(f"  进程内存: {final_snapshot.process_memory_mb:.1f} MB")
    if final_snapshot.gpu_memory_gb > 0:
        print(f"  GPU 内存: {final_snapshot.gpu_memory_gb:.1f} GB")
    
    # 获取内存统计
    stats = memory_manager.get_stats()
    print("\n内存管理统计:")
    print(f"  清理次数: {stats.get('cleanup_count', 0)}")
    print(f"  总释放内存: {stats.get('total_freed_mb', 0):.1f} MB")
    
    memory_manager.stop_monitoring()

def example_5_performance_profiling():
    """示例5: 性能分析"""
    print("\n" + "="*60)
    print("示例5: 性能分析")
    print("="*60)
    
    # 获取性能分析器
    analyzer = get_global_performance_analyzer()
    analyzer.start_monitoring()
    
    # 使用性能分析装饰器
    @performance_profile("model_inference", items_count=32)
    def run_inference(model, data):
        """运行推理"""
        with torch.no_grad():
            return model(data)
    
    @performance_profile("data_processing", items_count=100)
    def process_data():
        """数据处理"""
        data = torch.randn(100, 3, 32, 32)
        # 模拟数据预处理
        processed = data * 2.0 + 1.0
        return processed
    
    # 创建模型和数据
    model = create_sample_model()
    
    print("执行性能测试...")
    
    # 执行多次操作
    for i in range(5):
        data = process_data()
        result = run_inference(model, data[:32])
        time.sleep(0.1)
    
    # 等待监控数据收集
    time.sleep(2)
    
    # 分析性能
    print("\n系统性能分析:")
    system_analysis = analyzer.analyze_system_performance(1)
    if 'error' not in system_analysis:
        print(f"  平均 CPU 使用率: {system_analysis['cpu']['avg_usage']:.1f}%")
        print(f"  平均内存使用率: {system_analysis['memory']['avg_usage_percent']:.1f}%")
        print(f"  进程平均内存: {system_analysis['process']['avg_memory_mb']:.1f} MB")
    
    print("\n操作性能分析:")
    operation_analysis = analyzer.analyze_operation_performance()
    if 'error' not in operation_analysis:
        for op_name, op_data in operation_analysis['operations'].items():
            print(f"  {op_name}:")
            print(f"    执行次数: {op_data['count']}")
            print(f"    平均耗时: {op_data['duration']['avg']:.3f}s")
            print(f"    平均吞吐量: {op_data['throughput']['avg']:.1f} items/s")
    
    # 生成优化建议
    recommendations = analyzer.generate_optimization_recommendations()
    print("\n优化建议:")
    for i, rec in enumerate(recommendations[:3], 1):
        print(f"  {i}. {rec}")
    
    analyzer.stop_monitoring()

def example_6_training_optimization():
    """示例6: 训练优化"""
    print("\n" + "="*60)
    print("示例6: 训练优化")
    print("="*60)
    
    # 创建数据集
    train_dataset = create_sample_dataset(800)
    val_dataset = create_sample_dataset(200)
    
    # 创建模型
    model = create_sample_model()
    
    print("创建优化的训练环境...")
    
    # 使用快速训练优化
    training_optimizer = optimize_training(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        optimization_level=OptimizationLevel.BALANCED,
        epochs=3,  # 少量epoch用于演示
        learning_rate=1e-3
    )
    
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    
    print("\n开始训练...")
    
    # 模拟训练循环
    for epoch in range(3):
        print(f"\nEpoch {epoch + 1}/3")
        
        # 训练一个epoch
        train_metrics = training_optimizer.train_epoch(
            train_dataset, 
            criterion,
            epoch=epoch
        )
        
        print(f"  训练损失: {train_metrics.avg_loss:.4f}")
        print(f"  训练吞吐量: {train_metrics.throughput:.1f} samples/s")
        
        # 验证
        if val_dataset:
            val_metrics = training_optimizer.validate(val_dataset, criterion)
            print(f"  验证损失: {val_metrics.avg_loss:.4f}")
            print(f"  验证准确率: {val_metrics.accuracy:.2%}")
    
    # 获取训练统计
    stats = training_optimizer.get_training_stats()
    print("\n训练统计:")
    print(f"  总训练时间: {stats.get('total_training_time', 0):.2f}s")
    print(f"  平均epoch时间: {stats.get('avg_epoch_time', 0):.2f}s")
    print(f"  最佳验证分数: {stats.get('best_val_score', 0):.4f}")
    
    training_optimizer.cleanup()

def example_7_comprehensive_optimization():
    """示例7: 综合优化示例"""
    print("\n" + "="*60)
    print("示例7: 综合优化示例")
    print("="*60)
    
    # 创建优化管理器
    manager = create_optimization_manager(OptimizationLevel.BALANCED)
    
    try:
        with manager.optimization_context("comprehensive_example", items_count=1000):
            # 创建模型和数据
            model = create_sample_model()
            dataset = create_sample_dataset(1000)
            
            print("1. 优化模型...")
            model_result = manager.optimize_model(model)
            print(f"   {model_result.message}")
            
            print("2. 创建优化的数据加载器...")
            dataloader = manager.create_optimized_dataloader(dataset, batch_size=32)
            
            print("3. 创建优化的预测器...")
            predictor = manager.create_optimized_predictor(model)
            
            print("4. 执行批量预测...")
            sample_data = [torch.randn(3, 32, 32) for _ in range(100)]
            
            start_time = time.time()
            predictions = predictor.predict_batch(sample_data)
            end_time = time.time()
            
            print(f"   预测100个样本耗时: {end_time - start_time:.3f}s")
            print(f"   平均吞吐量: {len(sample_data) / (end_time - start_time):.1f} samples/s")
            
            print("5. 优化批处理大小...")
            sample_input = torch.randn(1, 3, 32, 32)
            optimal_batch_size = manager.optimize_batch_size(model, sample_input)
            print(f"   最优批处理大小: {optimal_batch_size}")
            
            print("6. 生成性能报告...")
            report_path = manager.generate_performance_report("./optimization_report")
            if report_path:
                print(f"   报告已生成: {report_path}")
            
            print("7. 获取优化摘要...")
            summary = manager.get_optimization_summary()
            print("   优化状态:")
            for key, value in summary['status'].items():
                print(f"     {key}: {value}")
            
            if 'performance' in summary:
                print("   性能指标:")
                for key, value in summary['performance'].items():
                    if isinstance(value, float):
                        print(f"     {key}: {value:.2f}")
                    else:
                        print(f"     {key}: {value}")
    
    except Exception as e:
        print(f"优化过程中出现错误: {e}")
    
    finally:
        manager.cleanup()

def main():
    """主函数"""
    print("优化工具使用示例")
    print("="*60)
    
    # 设置日志级别
    import logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    try:
        # 运行所有示例
        example_1_basic_optimization()
        example_2_custom_configuration()
        example_3_dataloader_optimization()
        example_4_memory_optimization()
        example_5_performance_profiling()
        example_6_training_optimization()
        example_7_comprehensive_optimization()
        
        print("\n" + "="*60)
        print("所有示例执行完成！")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n用户中断执行")
    except Exception as e:
        print(f"\n执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()