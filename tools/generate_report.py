#!/usr/bin/env python3
"""
實驗報告生成工具

生成完整的實驗分析報告，包括圖表、統計資料和摘要
"""

import sys
from pathlib import Path

# 添加項目根目錄到 Python 路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.movie_recommendation.report_generator import generate_report


def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='實驗報告生成工具',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--log-dir',
        default='log',
        help='日誌目錄路徑 (default: log)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='reports',
        help='輸出目錄路徑 (default: reports)'
    )
    
    parser.add_argument(
        '--no-dataset',
        action='store_true',
        help='不包含資料集分析'
    )
    
    parser.add_argument(
        '--full-dataset',
        action='store_true',
        help='使用完整資料集（20M 評分）進行分析（分批處理避免記憶體溢出）'
    )
    
    args = parser.parse_args()
    
    # 生成報告
    generate_report(
        log_dir=args.log_dir,
        output_dir=args.output_dir,
        include_dataset_analysis=not args.no_dataset,
        use_full_dataset=args.full_dataset
    )


if __name__ == "__main__":
    main()
