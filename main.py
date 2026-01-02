#!/usr/bin/env python3
"""
電影推薦系統實驗執行器

自動執行 run/ 目錄下的所有實驗配置，並在完成後生成完整的分析報告。

功能：
- 順序執行所有實驗配置（自動跳過已完成）
- 分析實驗結果（SVD、KNN 等）
- 生成可視化圖表
- 產生完整報告

Usage:
  uv run main.py              # 執行所有配置並生成報告
  uv run main.py --no-report  # 只執行配置，不生成報告
"""

import subprocess
import sys
import re
from pathlib import Path


def find_run_scripts(run_dir: Path):
    return sorted([p for p in run_dir.glob('*.py') if p.name != 'test_refactoring.sh'])


def _extract_experiment_name(script_path: Path):
    """Try to extract the Experiment `name` from a run script.

    Falls back to the script stem when no `name=` is found.
    """
    try:
        text = script_path.read_text(encoding='utf-8')
    except Exception:
        return script_path.stem

    # look for name="實驗..." or name='...'
    m = re.search(r"name\s*=\s*[\"']([^\"']+)[\"']", text)
    if m:
        return m.group(1)
    return script_path.stem


def run_scripts(scripts):
    results = []
    py = sys.executable or 'python3'
    log_dir = Path('log')
    log_dir.mkdir(parents=True, exist_ok=True)

    for script in scripts:
        exp_name = _extract_experiment_name(script)
        json_path = log_dir / f"{exp_name}.json"

        # Skip if a JSON summary exists
        if json_path.exists():
            print(f"--- Skipping: {script.name} (already completed: {json_path.name})")
            results.append((script.name, 'skipped'))
            continue

        # Fallback: check text log for a completion marker
        log_file = log_dir / f"{exp_name}.log"
        if log_file.exists():
            try:
                text = log_file.read_text(encoding='utf-8')
                if '實驗完成' in text or '完成' in text:
                    print(f"--- Skipping: {script.name} (log indicates completed: {log_file.name})")
                    results.append((script.name, 'skipped'))
                    continue
            except Exception:
                # if reading fails, fall back to running the script
                pass

        print(f"--- Running: {script.name}")
        proc = subprocess.run([py, str(script)])
        results.append((script.name, proc.returncode))
        print(f"--- Finished: {script.name} (code={proc.returncode})\n")

    return results


def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description='電影推薦系統實驗執行器')
    parser.add_argument('--no-report', action='store_true', 
                       help='不生成報告，只執行實驗配置')
    args = parser.parse_args()
    
    root = Path.cwd()
    run_dir = root / 'run'
    
    if not run_dir.exists():
        print('❌ run/ 目錄不存在')
        return
    
    scripts = find_run_scripts(run_dir)
    if not scripts:
        print('❌ run/ 目錄中沒有找到實驗腳本')
        return
    
    print("=" * 80)
    print("🚀 開始執行實驗配置")
    print("=" * 80)
    print(f"📁 找到 {len(scripts)} 個配置文件")
    print()
    
    # 執行所有配置
    results = run_scripts(scripts)
    
    # 顯示執行摘要
    print("\n" + "=" * 80)
    print("📊 執行摘要")
    print("=" * 80)
    
    completed = sum(1 for _, code in results if code == 0)
    skipped = sum(1 for _, code in results if code == 'skipped')
    failed = sum(1 for _, code in results if code not in [0, 'skipped'])
    
    print(f"✅ 成功: {completed}")
    print(f"⏭️  跳過: {skipped}")
    if failed > 0:
        print(f"❌ 失敗: {failed}")
    print()
    
    # 生成報告（除非指定 --no-report）
    # 只有在所有配置都完成後才生成報告
    if not args.no_report:
        print("=" * 80)
        print("📊 生成實驗報告")
        print("=" * 80)
        print()
        
        # 檢查是否還有新執行的配置
        newly_run = completed > 0
        
        if newly_run:
            print("⚠️ 本次執行了新的配置，需要等待所有配置完成後才生成報告")
            print("提示: 再次運行 'uv run main.py' 以生成完整報告")
        else:
            # 所有配置都已完成（全部被跳過），可以生成報告
            try:
                # 導入報告生成模組
                from src.movie_recommendation.report_generator import generate_report
                from src.movie_recommendation.analysis import (
                    ExperimentAnalyzer, 
                    print_progress_report,
                    print_svd_analysis,
                    print_knn_analysis
                )
                
                # 顯示進度
                print("📈 檢查執行進度...")
                analyzer = ExperimentAnalyzer()
                print_progress_report(analyzer)
                
                # 分析結果
                progress = analyzer.check_progress()
                
                # SVD 分析
                svd_completed = progress['stages'].get('SVD', {}).get('rate', 0) == 100
                if svd_completed:
                    print("🔍 分析 SVD 階段結果...")
                    print_svd_analysis(analyzer)
                
                # KNN 分析
                knn_completed = progress['stages'].get('KNN', {}).get('rate', 0) == 100
                if knn_completed:
                    print("🔍 分析 KNN 階段結果...")
                    print_knn_analysis(analyzer)
                
                # 生成完整報告（圖表、摘要等）
                if progress['total_completed'] > 0:
                    print("📊 生成可視化報告（實驗結果）...")
                    generate_report(include_dataset_analysis=False)  # 先只生成實驗結果
                    
                    # 檢查是否需要生成完整資料集報告
                    print()
                    print("=" * 80)
                    print("📊 檢查完整資料集報告")
                    print("=" * 80)
                    
                    reports_dir = Path('reports')
                    full_dataset_files = [
                        reports_dir / 'figures' / 'data_rating_distribution_full.png',
                        reports_dir / 'figures' / 'data_user_activity_long_tail_full.png',
                        reports_dir / 'figures' / 'data_movie_popularity_long_tail_full.png',
                        reports_dir / 'dataset_statistics_full.json'
                    ]
                    
                    all_exist = all(f.exists() for f in full_dataset_files)
                    
                    if all_exist:
                        print("✅ 完整資料集報告已存在，跳過生成")
                        for f in full_dataset_files:
                            print(f"   - {f.relative_to(reports_dir)}")
                    else:
                        print("📊 開始生成完整資料集報告（20M 評分）...")
                        print("⚠️  這可能需要 1-2 分鐘時間")
                        print()
                        generate_report(use_full_dataset=True)
                else:
                    print("⚠️ 尚無完成的配置，跳過報告生成")
                
            except ImportError as e:
                print(f"⚠️ 無法導入報告生成模組: {e}")
                print("提示: 確保已安裝所需套件（matplotlib 等）")
            except Exception as e:
                print(f"⚠️ 報告生成失敗: {e}")
    else:
        print("⏭️  跳過報告生成（使用 --no-report）")
    
    print("\n" + "=" * 80)
    print("✨ 完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()
