#!/usr/bin/env python3
"""
電影推薦系統實驗執行器 v2.0

使用 JSON 配置檔案驅動的自動化實驗執行系統。

主要功能：
- 從 JSON 配置檔案載入實驗定義
- 自動執行實驗並追蹤進度
- 支援最佳配置自動級聯到後續階段
- 生成完整的分析報告和可視化

新特性：
- 靈活的 JSON 配置格式
- 支援階段級別的配置管理
- 自動檢測已完成的實驗
- 智慧的最佳配置級聯機制

Usage:
  python main.py                           # 執行所有啟用的實驗階段（自動級聯最佳配置）
  python main.py --stage SVD_KNN_GRID      # 只執行特定階段
  python main.py --stage DS FILTER         # 執行多個階段
  python main.py --list-stages             # 列出所有可用階段
  python main.py --list-experiments        # 列出所有實驗
  python main.py --force                   # 強制重新運行所有實驗
  python main.py --report-only             # 只生成報告，不運行實驗
  python main.py --config custom.json      # 使用自訂配置檔案
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Optional, List

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from movie_recommendation.experiment_runner import ExperimentRunner
from movie_recommendation.config_loader import ConfigLoader
from movie_recommendation.utils import setup_logging
from movie_recommendation.hybrid_engine import GenomeHybridModel


def list_stages(config_path: Optional[Path] = None):
    """列出所有階段"""
    loader = ConfigLoader(config_path)
    
    print("\n" + "="*80)
    print("📋 可用的實驗階段")
    print("="*80)
    
    stages = loader._raw_config.get('stages', {})
    
    for stage_key, stage_data in stages.items():
        enabled = stage_data.get('enabled', True)
        name = stage_data.get('name', stage_key)
        desc = stage_data.get('description', '')
        exp_count = len(stage_data.get('experiments', []))
        
        status = "✅" if enabled else "❌"
        print(f"\n{status} {stage_key}: {name}")
        print(f"   描述: {desc}")
        print(f"   實驗數: {exp_count}")
    
    print("\n" + "="*80)


def list_experiments(config_path: Optional[Path] = None, stage: Optional[str] = None):
    """列出所有實驗"""
    loader = ConfigLoader(config_path)
    
    experiments = loader.get_experiments(stage=stage, enabled_only=False)
    
    print("\n" + "="*80)
    if stage:
        print(f"📋 {stage} 階段的實驗")
    else:
        print("📋 所有實驗")
    print("="*80)
    
    current_stage = None
    for exp in experiments:
        if exp.stage != current_stage:
            current_stage = exp.stage
            print(f"\n【{current_stage}】")
        
        status = "✅" if exp.enabled else "❌"
        print(f"  {status} {exp.id}: {exp.name}")
        if exp.description:
            print(f"      {exp.description}")
    
    print(f"\n總計: {len(experiments)} 個實驗")
    print("="*80 + "\n")


def generate_reports():
    """生成分析報告"""
    print("\n" + "="*80)
    print("📊 生成實驗分析報告")
    print("="*80 + "\n")
    
    try:
        from movie_recommendation.report_generator import generate_report
        
        # 生成可視化報告
        print("📊 生成可視化報告...")
        generate_report(include_dataset_analysis=False)
        
        # 檢查是否需要生成完整資料集報告
        print("\n" + "="*80)
        print("📊 檢查完整資料集報告")
        print("="*80)
        
        reports_dir = Path('reports')
        full_dataset_files = [
            reports_dir / 'figures' / 'data_rating_distribution_full.png',
            reports_dir / 'figures' / 'data_user_activity_long_tail_full.png',
            reports_dir / 'figures' / 'data_movie_popularity_long_tail_full.png',
            reports_dir / 'dataset_statistics_full.json'
        ]
        
        all_exist = all(f.exists() for f in full_dataset_files)
        
        if all_exist:
            print("✅ 完整資料集報告已存在")
        else:
            print("📊 生成完整資料集報告（20M 評分）...")
            print("⚠️  這可能需要 1-2 分鐘")
            generate_report(
                include_dataset_analysis=True,
                use_full_dataset=True,
                sample_size=None
            )
    
    except ImportError as e:
        print(f"⚠️  無法匯入報告生成模組: {e}")
    except Exception as e:
        print(f"⚠️  報告生成失敗: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description='電影推薦系統實驗執行器 v2.0',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  python main.py                          # 執行所有實驗（自動級聯）
  python main.py --stage SVD_KNN_GRID     # 只執行網格搜索階段
  python main.py --list-stages            # 列出所有階段
  python main.py --force                  # 強制重新運行所有實驗
  python main.py --report-only            # 只生成報告
        """
    )
    
    parser.add_argument(
        '--config',
        type=Path,
        help='配置檔案路徑（預設: configs/experiments.json）'
    )
    
    parser.add_argument(
        '--stage',
        type=str,
        nargs='+',
        help='只執行指定的階段（可指定多個）'
    )
    
    parser.add_argument(
        '--list-stages',
        action='store_true',
        help='列出所有可用的實驗階段'
    )
    
    parser.add_argument(
        '--list-experiments',
        action='store_true',
        help='列出所有實驗'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='強制重新運行已完成的實驗'
    )
    
    parser.add_argument(
        '--report-only',
        action='store_true',
        help='只生成報告，不運行實驗'
    )
    
    parser.add_argument(
        '--no-report',
        action='store_true',
        help='不生成報告，只運行實驗'
    )
    
    args = parser.parse_args()
    
    # 處理列出階段/實驗的命令
    if args.list_stages:
        list_stages(args.config)
        return
    
    if args.list_experiments:
        list_experiments(args.config)
        return
    
    # 設置日誌
    setup_logging("main", log_dir="log")
    
    # 只生成報告
    if args.report_only:
        generate_reports()
        return
    
    # 創建實驗運行器
    runner = ExperimentRunner(config_path=args.config)
    
    print("\n" + "="*80)
    print("🎬 電影推薦系統實驗執行器 v2.0")
    print("="*80)
    
    # 顯示配置資訊
    metadata = runner.config_loader.get_metadata()
    print(f"\n📋 配置資訊:")
    print(f"   版本: {metadata.get('version', 'N/A')}")
    print(f"   策略: {metadata.get('strategy', 'N/A')}")
    print(f"   描述: {metadata.get('description', 'N/A')}")
    
    if args.config:
        print(f"   配置檔案: {args.config}")
    
    # 執行實驗（強制啟用級聯模式）
    cascade_best = True  # 必須使用級聯模式以確保最佳配置傳遞
    stages = args.stage if args.stage else None
    
    if stages:
        print(f"\n📌 執行階段: {', '.join(stages)}")
    else:
        enabled_stages = runner.config_loader.get_enabled_stages()
        print(f"\n📌 執行所有啟用的階段: {', '.join(enabled_stages)}")
    
    print(f"🔄 最佳配置級聯: ✅ 啟用（必須）")
    print(f"♻️  強制重新運行: {'是' if args.force else '否'}")
    
    # 獲取預設樣本數
    defaults = runner.config_loader._raw_config.get('defaults', {})
    n_samples = defaults.get('n_samples', 500)
    print(f"📊 驗證樣本數: {n_samples:,}")
    print("="*80 + "\n")
    
    # 運行實驗
    result = runner.run_all(
        force=args.force,
        cascade_best=cascade_best,
        stages=stages
    )
    
    # 生成報告
    if not args.no_report:
        generate_reports()
    
    print("\n" + "="*80)
    print("✨ 完成！")
    print("="*80)
    
    # 顯示最佳配置彙總
    if result.get('best_configs'):
        print("\n🏆 最佳配置彙總:")
        for stage, config in result['best_configs'].items():
            print(f"   {stage}:")
            for key, value in config.items():
                print(f"      {key}: {value}")
    
    print()


if __name__ == '__main__':
    main()

