"""
命令行工具模塊

提供各種命令行工具功能，用於分析、報告生成和配置管理。
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple, Optional
import json


def analyze_experiments(
    command: str,
    log_dir: str = 'log',
    run_dir: str = 'run',
    sample_size: int = 100000
):
    """
    分析實驗結果
    
    Args:
        command: 分析命令 (progress, svd, knn, dataset, all)
        log_dir: 日誌目錄
        run_dir: 運行目錄
        sample_size: 資料集分析樣本大小
    """
    from ..analysis.analyzer import ExperimentAnalyzer, DatasetAnalyzer
    from ..analysis.analyzer import (
        print_progress_report,
        print_svd_analysis,
        print_knn_analysis,
        print_dataset_analysis
    )
    
    if command == 'progress':
        analyzer = ExperimentAnalyzer(log_dir=log_dir, run_dir=run_dir)
        print_progress_report(analyzer)
        
    elif command == 'svd':
        analyzer = ExperimentAnalyzer(log_dir=log_dir, run_dir=run_dir)
        print_svd_analysis(analyzer)
        
    elif command == 'knn':
        analyzer = ExperimentAnalyzer(log_dir=log_dir, run_dir=run_dir)
        print_knn_analysis(analyzer)
        
    elif command == 'dataset':
        dataset_analyzer = DatasetAnalyzer()
        print_dataset_analysis(dataset_analyzer, sample_size=sample_size)
        
    elif command == 'all':
        # 實驗結果分析
        analyzer = ExperimentAnalyzer(log_dir=log_dir, run_dir=run_dir)
        print_progress_report(analyzer)
        print()
        print_svd_analysis(analyzer)
        print()
        print_knn_analysis(analyzer)
        print()
        
        # 資料集分析
        dataset_analyzer = DatasetAnalyzer()
        print_dataset_analysis(dataset_analyzer, sample_size=sample_size)


def generate_report_cli(
    log_dir: str = 'log',
    output_dir: str = 'reports',
    include_dataset: bool = True,
    use_full_dataset: bool = False,
    sample_size: Optional[int] = None
):
    """
    生成實驗報告
    
    Args:
        log_dir: 日誌目錄
        output_dir: 輸出目錄
        include_dataset: 是否包含資料集分析
        use_full_dataset: 是否使用完整資料集
        sample_size: 資料集分析樣本大小
    """
    from ..analysis.report_generator import generate_report
    
    print("\n" + "="*80)
    print("📊 生成實驗報告")
    print("="*80)
    
    generate_report(
        log_dir=log_dir,
        output_dir=output_dir,
        include_dataset_analysis=include_dataset,
        use_full_dataset=use_full_dataset,
        sample_size=sample_size
    )
    
    print("\n✅ 報告生成完成")


def get_existing_experiments(stage_config: Dict[str, Any]) -> Set[Tuple[int, int]]:
    """
    獲取現有階段中已存在的實驗配置 (SVD, KNN) 組合
    
    Args:
        stage_config: 階段配置
    
    Returns:
        已存在的 (n_components, k_neighbors) 組合集合
    """
    existing = set()
    for exp in stage_config.get('experiments', []):
        config = exp.get('config', {})
        n_comp = config.get('n_components')
        k_neigh = config.get('k_neighbors')
        if n_comp is not None and k_neigh is not None:
            existing.add((n_comp, k_neigh))
    return existing


def get_next_experiment_id(stage_config: Dict[str, Any], stage_id: str) -> int:
    """
    獲取下一個可用的實驗ID編號
    
    Args:
        stage_config: 階段配置
        stage_id: 階段ID
    
    Returns:
        下一個可用的編號
    """
    existing_ids = []
    for exp in stage_config.get('experiments', []):
        exp_id = exp.get('id', '')
        if exp_id.startswith(f"{stage_id}_"):
            try:
                num = int(exp_id.split('_')[-1])
                existing_ids.append(num)
            except ValueError:
                pass
    
    return max(existing_ids, default=0) + 1


def generate_grid_experiments(
    svd_values: List[int],
    knn_values: List[int],
    stage_id: str = "SVD_KNN_GRID",
    base_config: Optional[Dict[str, Any]] = None,
    existing_stage_config: Optional[Dict[str, Any]] = None,
    skip_existing: bool = True
) -> Dict[str, Any]:
    """
    生成網格搜索實驗配置
    
    Args:
        svd_values: SVD 維度列表
        knn_values: KNN 鄰居數列表
        stage_id: 階段 ID
        base_config: 基礎配置
        existing_stage_config: 現有的階段配置（用於檢測重複）
        skip_existing: 是否跳過已存在的實驗
    
    Returns:
        完整的階段配置
    """
    if base_config is None:
        base_config = {
            "data_limit": None,
            "min_item_ratings": 0,
            "use_svd": True
        }
    
    # 獲取已存在的實驗
    existing_experiments = set()
    exp_counter = 1
    
    if existing_stage_config:
        existing_experiments = get_existing_experiments(existing_stage_config)
        exp_counter = get_next_experiment_id(existing_stage_config, stage_id)
        print(f"\nℹ️  檢測到 {len(existing_experiments)} 個已存在的實驗")
        print(f"ℹ️  下一個實驗ID將從 {stage_id}_{exp_counter:03d} 開始")
    
    experiments = []
    skipped_count = 0
    
    # 生成所有 SVD × KNN 組合
    for svd in svd_values:
        for knn in knn_values:
            # 檢查是否已存在
            if skip_existing and (svd, knn) in existing_experiments:
                skipped_count += 1
                continue
            exp_id = f"{stage_id}_{exp_counter:03d}"
            exp_name = f"SVD={svd}×KNN={knn}"
            
            # 計算 2 的冪次表示
            svd_power = svd.bit_length() - 1 if svd > 0 else 0
            knn_power = knn.bit_length() - 1 if knn > 0 else 0
            description = f"2^{svd_power}維度 × 2^{knn_power}鄰居"
            
            experiment = {
                "id": exp_id,
                "name": exp_name,
                "description": description,
                "config": {
                    "n_components": svd,
                    "k_neighbors": knn
                }
            }
            
            experiments.append(experiment)
            exp_counter += 1
    
    if skip_existing and skipped_count > 0:
        print(f"✅ 跳過 {skipped_count} 個已存在的實驗")
        print(f"➕ 將添加 {len(experiments)} 個新實驗")
    
    stage_config = {
        "name": "SVD×KNN 網格搜索",
        "description": "同時測試所有 SVD 和 KNN 組合，找出最佳配對",
        "enabled": True,
        "base_config": base_config,
        "experiments": experiments
    }
    
    return stage_config


def update_config_with_grid(
    config_path: Path,
    svd_values: Optional[List[int]] = None,
    knn_values: Optional[List[int]] = None,
    stage_id: str = "SVD_KNN_GRID",
    skip_existing: bool = True,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    更新配置文件，添加或更新網格搜索階段
    
    Args:
        config_path: 配置文件路徑
        svd_values: SVD 維度列表
        knn_values: KNN 鄰居數列表
        stage_id: 階段 ID
        skip_existing: 是否跳過已存在的實驗
        dry_run: 只生成預覽，不實際保存
    
    Returns:
        更新後的配置
    """
    if svd_values is None:
        svd_values = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    
    if knn_values is None:
        knn_values = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    
    # 讀取現有配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 檢查階段是否已存在
    existing_stage = config.get('stages', {}).get(stage_id)
    
    # 生成新的階段配置
    new_stage = generate_grid_experiments(
        svd_values=svd_values,
        knn_values=knn_values,
        stage_id=stage_id,
        existing_stage_config=existing_stage,
        skip_existing=skip_existing
    )
    
    # 更新配置
    if 'stages' not in config:
        config['stages'] = {}
    
    if existing_stage and skip_existing:
        # 合併實驗
        existing_experiments = existing_stage.get('experiments', [])
        new_experiments = new_stage['experiments']
        config['stages'][stage_id]['experiments'] = existing_experiments + new_experiments
        print(f"\n✅ 將添加 {len(new_experiments)} 個新實驗到現有階段")
    else:
        config['stages'][stage_id] = new_stage
        print(f"\n✅ 創建新階段，共 {len(new_stage['experiments'])} 個實驗")
    
    # 保存配置（如果不是 dry run）
    if not dry_run:
        backup_path = config_path.with_suffix('.json.backup')
        config_path.rename(backup_path)
        print(f"💾 備份原配置到: {backup_path}")
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"💾 配置已保存: {config_path}")
    else:
        print("\n🔍 Dry run 模式 - 不保存配置")
        print(f"預覽: 將有 {len(config['stages'][stage_id]['experiments'])} 個實驗")
    
    return config


# CLI 主函數
def main_analyze():
    """分析工具主函數"""
    parser = argparse.ArgumentParser(
        description='電影推薦系統分析工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m movie_recommendation.utils.cli analyze progress   # 顯示執行進度
  python -m movie_recommendation.utils.cli analyze svd       # 分析 SVD 階段結果
  python -m movie_recommendation.utils.cli analyze dataset   # 分析資料集統計
        """
    )
    
    parser.add_argument(
        'command',
        choices=['progress', 'svd', 'knn', 'dataset', 'all'],
        help='要執行的分析命令'
    )
    
    parser.add_argument('--log-dir', default='log', help='日誌目錄路徑')
    parser.add_argument('--run-dir', default='run', help='配置文件目錄路徑')
    parser.add_argument('--sample-size', type=int, default=100000, help='資料集分析樣本大小')
    
    args = parser.parse_args()
    analyze_experiments(args.command, args.log_dir, args.run_dir, args.sample_size)


def main_report():
    """報告生成工具主函數"""
    parser = argparse.ArgumentParser(
        description='實驗報告生成工具',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--log-dir', default='log', help='日誌目錄路徑')
    parser.add_argument('--output-dir', default='reports', help='輸出目錄路徑')
    parser.add_argument('--no-dataset', action='store_true', help='不包含資料集分析')
    parser.add_argument('--full-dataset', action='store_true', help='使用完整資料集')
    parser.add_argument('--sample-size', type=int, help='資料集分析樣本大小')
    
    args = parser.parse_args()
    generate_report_cli(
        log_dir=args.log_dir,
        output_dir=args.output_dir,
        include_dataset=not args.no_dataset,
        use_full_dataset=args.full_dataset,
        sample_size=args.sample_size
    )


def main_grid_config():
    """網格配置生成工具主函數"""
    parser = argparse.ArgumentParser(
        description='網格搜索配置生成器',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--config', type=Path, default='configs/experiments.json',
                        help='配置文件路徑')
    parser.add_argument('--stage-id', default='SVD_KNN_GRID', help='階段ID')
    parser.add_argument('--svd-values', type=int, nargs='+', 
                        help='SVD 維度列表（例如：2 4 8 16）')
    parser.add_argument('--knn-values', type=int, nargs='+',
                        help='KNN 鄰居數列表（例如：5 10 15 20）')
    parser.add_argument('--no-skip-existing', action='store_true',
                        help='不跳過已存在的實驗')
    parser.add_argument('--dry-run', action='store_true',
                        help='只生成預覽，不保存')
    
    args = parser.parse_args()
    update_config_with_grid(
        config_path=args.config,
        svd_values=args.svd_values,
        knn_values=args.knn_values,
        stage_id=args.stage_id,
        skip_existing=not args.no_skip_existing,
        dry_run=args.dry_run
    )


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == 'analyze':
            sys.argv.pop(1)
            main_analyze()
        elif sys.argv[1] == 'report':
            sys.argv.pop(1)
            main_report()
        elif sys.argv[1] == 'grid':
            sys.argv.pop(1)
            main_grid_config()
        else:
            print("使用方法:")
            print("  python -m movie_recommendation.utils.cli analyze <command>")
            print("  python -m movie_recommendation.utils.cli report [options]")
            print("  python -m movie_recommendation.utils.cli grid [options]")
    else:
        print("使用方法:")
        print("  python -m movie_recommendation.utils.cli analyze <command>")
        print("  python -m movie_recommendation.utils.cli report [options]")
        print("  python -m movie_recommendation.utils.cli grid [options]")
