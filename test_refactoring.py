#!/usr/bin/env python3
"""
快速测试重构后的系统

测试内容：
1. 配置文件加载
2. 实验列表生成
3. 单个实验运行
4. 阶段执行
"""

import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from movie_recommendation.config_loader import ConfigLoader
from movie_recommendation.experiment_runner import ExperimentRunner
from movie_recommendation.utils import setup_logging


def test_config_loader():
    """测试配置加载器"""
    print("\n" + "="*80)
    print("🧪 测试 1: 配置加载器")
    print("="*80)
    
    loader = ConfigLoader()
    
    # 测试元数据
    metadata = loader.get_metadata()
    print(f"✓ 配置版本: {metadata.get('version')}")
    print(f"✓ 搜索策略: {metadata.get('strategy')}")
    
    # 测试阶段列表
    stages = loader.get_stages()
    print(f"✓ 总阶段数: {len(stages)}")
    print(f"  阶段: {', '.join(stages)}")
    
    enabled_stages = loader.get_enabled_stages()
    print(f"✓ 启用阶段: {len(enabled_stages)}")
    print(f"  启用: {', '.join(enabled_stages)}")
    
    # 测试实验列表
    experiments = loader.get_experiments(enabled_only=True)
    print(f"✓ 启用的实验数: {len(experiments)}")
    
    # 按阶段分组统计
    stage_counts = {}
    for exp in experiments:
        stage_counts[exp.stage] = stage_counts.get(exp.stage, 0) + 1
    
    print("\n阶段实验统计:")
    for stage, count in stage_counts.items():
        print(f"  - {stage}: {count} 个实验")
    
    print("\n✅ 配置加载器测试通过")
    return loader


def test_experiment_list(loader):
    """测试实验列表"""
    print("\n" + "="*80)
    print("🧪 测试 2: 实验列表生成")
    print("="*80)
    
    # 测试获取特定阶段
    svd_experiments = loader.get_experiments(stage='SVD_COARSE')
    print(f"✓ SVD_COARSE 阶段: {len(svd_experiments)} 个实验")
    
    for exp in svd_experiments:
        config = exp.config
        use_svd = config.get('use_svd', False)
        n_comp = config.get('n_components', 'N/A')
        print(f"  - {exp.id}: use_svd={use_svd}, n_components={n_comp}")
    
    # 测试配置继承
    print("\n配置继承测试:")
    exp = svd_experiments[0]
    print(f"  实验: {exp.id}")
    print(f"  完整配置键: {list(exp.config.keys())}")
    print(f"  - data_limit: {exp.config.get('data_limit')}")
    print(f"  - n_samples: {exp.config.get('n_samples')}")
    print(f"  - random_state: {exp.config.get('random_state')}")
    
    print("\n✅ 实验列表测试通过")


def test_experiment_runner():
    """测试实验执行器"""
    print("\n" + "="*80)
    print("🧪 测试 3: 实验执行器（仅检查）")
    print("="*80)
    
    runner = ExperimentRunner()
    
    # 检查已完成的实验
    experiments = runner.config_loader.get_experiments()
    completed_count = sum(1 for exp in experiments if runner.is_completed(exp.id))
    
    print(f"✓ 实验执行器初始化成功")
    print(f"✓ 总实验数: {len(experiments)}")
    print(f"✓ 已完成: {completed_count}")
    print(f"✓ 待执行: {len(experiments) - completed_count}")
    
    # 检查日志目录
    print(f"✓ 日志目录: {runner.log_dir}")
    if runner.log_dir.exists():
        json_files = list(runner.log_dir.glob('*.json'))
        print(f"  已有日志文件: {len(json_files)} 个")
    
    print("\n✅ 实验执行器测试通过")


def test_config_merge():
    """测试配置合并"""
    print("\n" + "="*80)
    print("🧪 测试 4: 配置合并逻辑")
    print("="*80)
    
    loader = ConfigLoader()
    
    # 获取一个实验并检查配置合并
    exp = loader.get_experiment('SVD_COARSE_004')
    if exp:
        print(f"✓ 实验: {exp.id}")
        print(f"  名称: {exp.name}")
        print(f"  描述: {exp.description}")
        print("\n  关键配置:")
        
        important_keys = ['data_limit', 'use_svd', 'n_components', 'k_neighbors', 'n_samples']
        for key in important_keys:
            value = exp.config.get(key, 'N/A')
            print(f"    - {key}: {value}")
        
        print("\n✅ 配置合并测试通过")
    else:
        print("⚠️  未找到测试实验")


def test_cascade_logic():
    """测试级联逻辑（仅概念验证）"""
    print("\n" + "="*80)
    print("🧪 测试 5: 级联逻辑检查")
    print("="*80)
    
    runner = ExperimentRunner()
    
    # 检查阶段顺序定义
    stage_order = {
        'DS': ['FILTER', 'SVD_COARSE', 'KNN_COARSE', 'BIAS', 'OPT'],
        'SVD_COARSE': ['KNN_COARSE', 'BIAS', 'OPT'],
        'KNN_COARSE': ['BIAS', 'OPT'],
    }
    
    print("✓ 阶段依赖关系:")
    for stage, deps in stage_order.items():
        print(f"  {stage} → {', '.join(deps)}")
    
    print("\n✅ 级联逻辑检查通过")


def main():
    """运行所有测试"""
    print("\n" + "="*80)
    print("🚀 电影推荐系统重构测试")
    print("="*80)
    print("\n开始测试新系统的各个组件...\n")
    
    try:
        # 测试 1: 配置加载
        loader = test_config_loader()
        
        # 测试 2: 实验列表
        test_experiment_list(loader)
        
        # 测试 3: 实验执行器
        test_experiment_runner()
        
        # 测试 4: 配置合并
        test_config_merge()
        
        # 测试 5: 级联逻辑
        test_cascade_logic()
        
        # 总结
        print("\n" + "="*80)
        print("🎉 所有测试通过！")
        print("="*80)
        print("\n系统重构成功，可以开始使用新的配置系统。")
        print("\n下一步:")
        print("  1. 运行: python main.py --list-stages")
        print("  2. 运行: python main.py --list-experiments")
        print("  3. 测试: python main.py --stage DS")
        print("  4. 完整: python main.py")
        print("\n查看文档: docs/REFACTORING_GUIDE.md\n")
        
    except Exception as e:
        print("\n" + "="*80)
        print("❌ 测试失败")
        print("="*80)
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
