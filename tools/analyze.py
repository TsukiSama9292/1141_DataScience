#!/usr/bin/env python3
"""
電影推薦系統分析工具

提供命令行接口來分析實驗結果和資料集統計
"""

import sys
from pathlib import Path

# 添加項目根目錄到 Python 路徑
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.movie_recommendation.analysis import (
    ExperimentAnalyzer,
    DatasetAnalyzer,
    print_progress_report,
    print_svd_analysis,
    print_knn_analysis,
    print_dataset_analysis
)


def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='電影推薦系統分析工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s progress          # 顯示執行進度
  %(prog)s svd              # 分析 SVD 階段結果
  %(prog)s knn              # 分析 KNN 階段結果
  %(prog)s dataset          # 分析資料集統計
  %(prog)s all              # 執行所有分析
        """
    )
    
    parser.add_argument(
        'command',
        choices=['progress', 'svd', 'knn', 'dataset', 'all'],
        help='要執行的分析命令'
    )
    
    parser.add_argument(
        '--log-dir',
        default='log',
        help='日誌目錄路徑 (default: log)'
    )
    
    parser.add_argument(
        '--run-dir',
        default='run',
        help='配置文件目錄路徑 (default: run)'
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=100000,
        help='資料集分析的樣本大小 (default: 100000)'
    )
    
    args = parser.parse_args()
    
    # 執行對應的分析
    if args.command == 'progress':
        analyzer = ExperimentAnalyzer(log_dir=args.log_dir, run_dir=args.run_dir)
        print_progress_report(analyzer)
        
    elif args.command == 'svd':
        analyzer = ExperimentAnalyzer(log_dir=args.log_dir, run_dir=args.run_dir)
        print_svd_analysis(analyzer)
        
    elif args.command == 'knn':
        analyzer = ExperimentAnalyzer(log_dir=args.log_dir, run_dir=args.run_dir)
        print_knn_analysis(analyzer)
        
    elif args.command == 'dataset':
        dataset_analyzer = DatasetAnalyzer()
        print_dataset_analysis(dataset_analyzer, sample_size=args.sample_size)
        
    elif args.command == 'all':
        # 實驗結果分析
        analyzer = ExperimentAnalyzer(log_dir=args.log_dir, run_dir=args.run_dir)
        print_progress_report(analyzer)
        print()
        print_svd_analysis(analyzer)
        print()
        print_knn_analysis(analyzer)
        print()
        
        # 資料集分析
        dataset_analyzer = DatasetAnalyzer()
        print_dataset_analysis(dataset_analyzer, sample_size=args.sample_size)


if __name__ == "__main__":
    main()
