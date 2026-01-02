"""
實驗報告生成器

整合分析、可視化和文檔生成功能，產生完整的實驗報告
"""

import json
from pathlib import Path
from typing import Dict, Any
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from .analysis import ExperimentAnalyzer, DatasetAnalyzer

# 設置中文字體
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False


class ReportGenerator:
    """實驗報告生成器"""
    
    def __init__(self, analyzer: ExperimentAnalyzer = None, 
                 dataset_analyzer: DatasetAnalyzer = None,
                 output_dir: str = 'reports',
                 dataset_size: str = None):
        """
        Args:
            analyzer: 實驗分析器
            dataset_analyzer: 資料集分析器
            output_dir: 輸出目錄
            dataset_size: 資料集大小標識 ('full' 或數字如 '100000')
        """
        self.analyzer = analyzer or ExperimentAnalyzer()
        self.dataset_analyzer = dataset_analyzer or DatasetAnalyzer()
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / 'figures'
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_size = dataset_size  # 用於文件命名
    
    def generate_svd_plots(self) -> bool:
        """生成 SVD 維度分析圖"""
        svd_analysis = self.analyzer.analyze_svd()
        
        if not svd_analysis or not svd_analysis['results']:
            print("⚠️ SVD 結果不足，跳過圖表生成")
            return False
        
        data = sorted(svd_analysis['results'], key=lambda x: x['n_components'])
        
        dims = [d['n_components'] for d in data]
        hit_rates = [d['hit_rate'] for d in data]
        ndcgs = [d['ndcg'] for d in data]
        rmses = [d['rmse'] for d in data]
        times = [d['total_time'] for d in data]
        
        # 創建圖表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('SVD Dimension Analysis', fontsize=16, fontweight='bold')
        
        # 1. Hit Rate vs Dimension
        ax1.plot(dims, hit_rates, 'o-', linewidth=2, markersize=8, color='#2E86AB')
        ax1.set_xlabel('SVD Dimension', fontsize=12)
        ax1.set_ylabel('Hit Rate@10', fontsize=12)
        ax1.set_title('Hit Rate@10 vs SVD Dimension', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        if hit_rates:
            best_hit_rate = max(hit_rates)
            best_idx = hit_rates.index(best_hit_rate)
            ax1.axhline(y=best_hit_rate, color='r', linestyle='--', alpha=0.5, 
                       label=f'Best: {best_hit_rate:.4f}')
            ax1.plot(dims[best_idx], hit_rates[best_idx], 'r*', markersize=20, 
                    label=f'Best Dim: {dims[best_idx]}')
            ax1.legend()
        
        # 2. NDCG vs Dimension
        ax2.plot(dims, ndcgs, 'o-', linewidth=2, markersize=8, color='#F18F01')
        ax2.set_xlabel('SVD Dimension', fontsize=12)
        ax2.set_ylabel('NDCG@10', fontsize=12)
        ax2.set_title('NDCG@10 vs SVD Dimension', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. RMSE vs Dimension
        ax3.plot(dims, rmses, 'o-', linewidth=2, markersize=8, color='#C73E1D')
        ax3.set_xlabel('SVD Dimension', fontsize=12)
        ax3.set_ylabel('RMSE', fontsize=12)
        ax3.set_title('RMSE vs SVD Dimension', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Execution Time vs Dimension
        ax4.plot(dims, times, 'o-', linewidth=2, markersize=8, color='#6A994E')
        ax4.set_xlabel('SVD Dimension', fontsize=12)
        ax4.set_ylabel('Execution Time (seconds)', fontsize=12)
        ax4.set_title('Execution Time vs SVD Dimension', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.figures_dir / 'svd_dimension_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ SVD 分析圖已保存: {output_path}")
        return True
    
    def generate_knn_plots(self) -> bool:
        """生成 KNN K值分析圖"""
        knn_analysis = self.analyzer.analyze_knn()
        
        if not knn_analysis or not knn_analysis['results']:
            print("⚠️ KNN 結果不足，跳過圖表生成")
            return False
        
        data = sorted(knn_analysis['results'], key=lambda x: x['k_neighbors'])
        
        ks = [d['k_neighbors'] for d in data]
        hit_rates = [d['hit_rate'] for d in data]
        ndcgs = [d['ndcg'] for d in data]
        rmses = [d['rmse'] for d in data]
        times = [d['total_time'] for d in data]
        
        # 創建圖表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('KNN K-value Analysis', fontsize=16, fontweight='bold')
        
        # 1. Hit Rate vs K
        ax1.plot(ks, hit_rates, 'o-', linewidth=2, markersize=8, color='#2E86AB')
        ax1.set_xlabel('K (Number of Neighbors)', fontsize=12)
        ax1.set_ylabel('Hit Rate@10', fontsize=12)
        ax1.set_title('Hit Rate@10 vs K', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        if hit_rates:
            best_hit_rate = max(hit_rates)
            best_idx = hit_rates.index(best_hit_rate)
            ax1.axhline(y=best_hit_rate, color='r', linestyle='--', alpha=0.5, 
                       label=f'Best: {best_hit_rate:.4f}')
            ax1.plot(ks[best_idx], hit_rates[best_idx], 'r*', markersize=20, 
                    label=f'Best K: {ks[best_idx]}')
            ax1.legend()
        
        # 2. NDCG vs K
        ax2.plot(ks, ndcgs, 'o-', linewidth=2, markersize=8, color='#F18F01')
        ax2.set_xlabel('K (Number of Neighbors)', fontsize=12)
        ax2.set_ylabel('NDCG@10', fontsize=12)
        ax2.set_title('NDCG@10 vs K', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. RMSE vs K
        ax3.plot(ks, rmses, 'o-', linewidth=2, markersize=8, color='#C73E1D')
        ax3.set_xlabel('K (Number of Neighbors)', fontsize=12)
        ax3.set_ylabel('RMSE', fontsize=12)
        ax3.set_title('RMSE vs K', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Execution Time vs K
        ax4.plot(ks, times, 'o-', linewidth=2, markersize=8, color='#6A994E')
        ax4.set_xlabel('K (Number of Neighbors)', fontsize=12)
        ax4.set_ylabel('Execution Time (seconds)', fontsize=12)
        ax4.set_title('Execution Time vs K', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.figures_dir / 'knn_k_value_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ KNN 分析圖已保存: {output_path}")
        return True
    
    def generate_comparison_plot(self) -> bool:
        """生成階段對比圖"""
        stages = {
            'DS': 'DS_004',
            'FILTER': 'FILTER_001',
            'SVD': 'SVD_008',
            'KNN': 'KNN_004',
        }
        
        stage_data = {}
        for stage, config in stages.items():
            data = self.analyzer.load_result(config)
            if data:
                metrics = data.get('metrics', {})
                stage_data[stage] = {
                    'hit_rate': metrics.get('hit_rate', 0),
                    'ndcg': metrics.get('ndcg', 0),
                }
        
        if not stage_data:
            print("⚠️ 階段數據不足，跳過對比圖生成")
            return False
        
        # 創建圖表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Best Configuration Comparison Across Stages', 
                     fontsize=16, fontweight='bold')
        
        stages_list = list(stage_data.keys())
        hit_rates = [stage_data[s]['hit_rate'] for s in stages_list]
        ndcgs = [stage_data[s]['ndcg'] for s in stages_list]
        
        x = np.arange(len(stages_list))
        width = 0.6
        
        # Hit Rate 比較
        bars1 = ax1.bar(x, hit_rates, width, color='#2E86AB')
        ax1.set_xlabel('Stage', fontsize=12)
        ax1.set_ylabel('Hit Rate@10', fontsize=12)
        ax1.set_title('Hit Rate@10 Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(stages_list)
        ax1.grid(True, alpha=0.3, axis='y')
        
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=10)
        
        # NDCG 比較
        bars2 = ax2.bar(x, ndcgs, width, color='#F18F01')
        ax2.set_xlabel('Stage', fontsize=12)
        ax2.set_ylabel('NDCG@10', fontsize=12)
        ax2.set_title('NDCG@10 Comparison', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(stages_list)
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        output_path = self.figures_dir / 'stage_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 階段對比圖已保存: {output_path}")
        return True
    
    def generate_dataset_plots(self, use_full_dataset: bool = False) -> bool:
        """生成資料集分析圖表
        
        Args:
            use_full_dataset: 是否使用完整資料集（20M 評分），使用分批處理避免記憶體溢出
        """
        # 確定文件名後綴
        suffix = f"_{self.dataset_size}" if self.dataset_size else ""
        
        # 檢查所有資料集圖表是否已存在
        required_plots = [
            f'data_rating_distribution{suffix}.png',
            f'data_user_activity_long_tail{suffix}.png',
            f'data_movie_popularity_long_tail{suffix}.png'
        ]
        
        all_exist = all((self.figures_dir / plot).exists() for plot in required_plots)
        if all_exist:
            print(f"✅ 資料集圖表已存在（{self.dataset_size or 'sample'}），跳過生成")
            return True
        
        # 1. 評分分布圖 - 使用分批處理
        print("📊 生成評分分布圖...")
        rating_stats = self.dataset_analyzer.analyze_rating_distribution(use_full_dataset=use_full_dataset)
        
        if rating_stats is None:
            print("⚠️ 無法載入資料，跳過資料集圖表生成")
            return False
        
        plt.figure(figsize=(12, 6))
        
        # 確保顯示所有可能的評分值 (0.5 遞增)
        all_ratings = [0.5 * i for i in range(1, 11)]  # 0.5, 1.0, 1.5, ..., 5.0
        rating_dist = rating_stats['distribution']
        counts = [rating_dist.get(r, 0) for r in all_ratings]
        
        bars = plt.bar(all_ratings, counts, width=0.4,
                      color='#2E86AB', alpha=0.8, edgecolor='black')
        plt.xlabel('Rating (Stars)', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        
        # 標題顯示總評分數
        total = rating_stats['total_ratings']
        plt.title(f'Rating Distribution (Total: {total:,} ratings)', 
                 fontsize=14, fontweight='bold')
        plt.xticks(all_ratings)
        plt.grid(True, alpha=0.3, axis='y')
        plt.ylim(bottom=0)
        
        # 標註數值（只標註非零的）
        for bar, count in zip(bars, counts):
            if count > 0:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(count):,}',
                        ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        output_path = self.figures_dir / f'data_rating_distribution{suffix}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 評分分布圖已保存: {output_path}")
        
        # 2. 使用者活躍度長尾圖 - 使用分批處理
        print("📊 生成使用者活躍度長尾圖...")
        user_stats = self.dataset_analyzer.analyze_user_activity(use_full_dataset=use_full_dataset, top_n=10000)
        
        if user_stats is None:
            return False
        
        plt.figure(figsize=(12, 6))
        user_counts = user_stats.get('_plot_data', [])
        
        if len(user_counts) > 0:
            plt.plot(range(len(user_counts)), user_counts, color='#2E86AB', linewidth=1.5)
            plt.fill_between(range(len(user_counts)), user_counts, color='#2E86AB', alpha=0.2)
        plt.xlabel('User (sorted by activity)', fontsize=12)
        plt.ylabel('Number of Ratings', fontsize=12)
        
        # 標題顯示總用戶數
        total_users = user_stats['total_users']
        plt.title(f'User Activity Long Tail (Total: {total_users:,} users)', 
                 fontsize=14, fontweight='bold')
        plt.xlim(0, min(len(user_counts), 5000))  # 顯示前 5000 個用戶
        plt.ylim(bottom=0)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.figures_dir / f'data_user_activity_long_tail{suffix}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 使用者活躍度圖已保存: {output_path}")
        
        # 3. 電影流行度長尾圖 - 使用分批處理
        print("📊 生成電影流行度長尾圖...")
        item_stats = self.dataset_analyzer.analyze_item_popularity(use_full_dataset=use_full_dataset, top_n=10000)
        
        if item_stats is None:
            return False
        
        plt.figure(figsize=(12, 6))
        item_counts = item_stats.get('_plot_data', [])
        
        if len(item_counts) > 0:
            plt.plot(range(len(item_counts)), item_counts, color='#F18F01', linewidth=1.5)
            plt.fill_between(range(len(item_counts)), item_counts, color='#F18F01', alpha=0.2)
        plt.xlabel('Movie (sorted by popularity)', fontsize=12)
        plt.ylabel('Number of Ratings', fontsize=12)
        
        # 標題顯示總電影數
        total_items = item_stats['total_items']
        plt.title(f'Movie Popularity Long Tail (Total: {total_items:,} movies)', 
                 fontsize=14, fontweight='bold')
        plt.xlim(0, min(len(item_counts), 5000))  # 顯示前 5000 部電影
        plt.ylim(bottom=0)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.figures_dir / f'data_movie_popularity_long_tail{suffix}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 電影流行度圖已保存: {output_path}")
        
        return True
    
    def generate_summary_table(self) -> str:
        """生成摘要表格（Markdown 格式）"""
        summary = self.analyzer.generate_summary_table()
        
        if not summary:
            return "No results available."
        
        lines = [
            "# 實驗結果摘要\n",
            "| 階段 | 最佳配置 | Hit Rate@10 | NDCG@10 | RMSE | 配置數 |",
            "|------|---------|------------|---------|------|--------|"
        ]
        
        for row in summary:
            lines.append(
                f"| {row['stage']} | {row['best_config']} | "
                f"{row['hit_rate']:.4f} | {row['ndcg']:.4f} | "
                f"{row['rmse']:.4f} | {row['count']} |"
            )
        
        return "\n".join(lines)
    
    def generate_full_report(self, include_dataset_analysis: bool = True, use_full_dataset: bool = False) -> Dict[str, Any]:
        """生成完整報告
        
        Args:
            include_dataset_analysis: 是否包含資料集分析
            use_full_dataset: 是否使用完整資料集（20M 評分）進行分析
        """
        print("=" * 80)
        print("📊 生成完整實驗報告")
        if use_full_dataset:
            print("⚠️  使用完整資料集（20M 評分）- 這可能需要幾分鐘時間")
        print("=" * 80)
        print()
        
        results = {
            'progress': self.analyzer.check_progress(),
            'plots': {},
            'summary_table': None,
            'dataset_analysis': None
        }
        
        # 資料集分析（如果啟用）
        if include_dataset_analysis:
            # 確定統計文件名
            suffix = f"_{self.dataset_size}" if self.dataset_size else ""
            stats_file = self.output_dir / f'dataset_statistics{suffix}.json'
            
            # 檢查統計文件是否已存在
            if stats_file.exists():
                print(f"✅ 資料集統計已存在（{self.dataset_size or 'sample'}），跳過分析")
                results['dataset_analysis'] = str(stats_file)
            else:
                if use_full_dataset:
                    print("📊 分析完整資料集統計特徵（使用分批處理）...")
                else:
                    print("📊 分析資料集統計特徵（使用樣本）...")
                dataset_stats = self.dataset_analyzer.generate_full_analysis()
                if dataset_stats:
                    # 過濾掉不能序列化的字段（以 _ 開頭的內部數據）
                    def filter_internal_fields(obj):
                        if isinstance(obj, dict):
                            return {k: filter_internal_fields(v) 
                                   for k, v in obj.items() 
                                   if isinstance(k, str) and not k.startswith('_')}
                        elif isinstance(obj, list):
                            return [filter_internal_fields(item) for item in obj]
                        else:
                            return obj
                    
                    clean_stats = filter_internal_fields(dataset_stats)
                    
                    # 保存統計資料
                    with open(stats_file, 'w', encoding='utf-8') as f:
                        json.dump(clean_stats, f, indent=2, ensure_ascii=False)
                    results['dataset_analysis'] = str(stats_file)
                    print(f"✅ 資料集統計已保存: {stats_file}")
            print()
            
            # 生成資料集圖表
            print("📈 生成資料集可視化圖表...")
            results['plots']['dataset'] = self.generate_dataset_plots(use_full_dataset=use_full_dataset)
            print()
        
        # 生成實驗結果可視化圖表
        print("📈 生成實驗結果可視化圖表...")
        results['plots']['svd'] = self.generate_svd_plots()
        results['plots']['knn'] = self.generate_knn_plots()
        results['plots']['comparison'] = self.generate_comparison_plot()
        print()
        
        # 生成摘要表格
        print("📋 生成摘要表格...")
        summary_md = self.generate_summary_table()
        summary_file = self.output_dir / 'summary.md'
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary_md)
        results['summary_table'] = str(summary_file)
        print(f"✅ 摘要表格已保存: {summary_file}")
        print()
        
        # 保存最佳配置
        print("🏆 提取最佳配置...")
        best_configs = self.analyzer.get_best_configs()
        if best_configs:
            best_file = self.output_dir / 'best_configs.json'
            with open(best_file, 'w', encoding='utf-8') as f:
                json.dump(best_configs, f, indent=2, ensure_ascii=False)
            print(f"✅ 最佳配置已保存: {best_file}")
        print()
        
        print("=" * 80)
        print("✨ 報告生成完成！")
        print("=" * 80)
        print()
        print(f"📁 輸出目錄: {self.output_dir}/")
        
        # 列出生成的文件
        plot_files = [
            "figures/svd_dimension_analysis.png",
            "figures/knn_k_value_analysis.png",
            "figures/stage_comparison.png"
        ]
        
        if include_dataset_analysis:
            plot_files.extend([
                "figures/data_rating_distribution.png",
                "figures/data_user_activity_long_tail.png",
                "figures/data_movie_popularity_long_tail.png"
            ])
        
        for plot_file in plot_files:
            if (self.output_dir / plot_file.replace('figures/', 'figures/')).exists():
                print(f"   - {plot_file}")
        
        print(f"   - summary.md")
        print(f"   - best_configs.json")
        if include_dataset_analysis:
            print(f"   - dataset_statistics.json")
        print()
        
        return results


def generate_report(log_dir: str = 'log', output_dir: str = 'reports', 
                   include_dataset_analysis: bool = True, use_full_dataset: bool = False,
                   sample_size: int = None):
    """生成完整報告（便捷函數）
    
    Args:
        log_dir: 實驗日誌目錄
        output_dir: 報告輸出目錄
        include_dataset_analysis: 是否包含資料集分析
        use_full_dataset: 是否使用完整資料集（20M 評分）
        sample_size: 樣本大小（如果不使用完整資料集）
    """
    # 確定資料集大小標識
    if use_full_dataset:
        dataset_size = 'full'
    elif sample_size:
        dataset_size = str(sample_size)
    else:
        dataset_size = None
    
    analyzer = ExperimentAnalyzer(log_dir=log_dir)
    dataset_analyzer = DatasetAnalyzer()
    generator = ReportGenerator(
        analyzer=analyzer,
        dataset_analyzer=dataset_analyzer,
        output_dir=output_dir,
        dataset_size=dataset_size
    )
    return generator.generate_full_report(
        include_dataset_analysis=include_dataset_analysis,
        use_full_dataset=use_full_dataset
    )
    analyzer = ExperimentAnalyzer(log_dir=log_dir)
    dataset_analyzer = DatasetAnalyzer()
    generator = ReportGenerator(analyzer, dataset_analyzer, output_dir=output_dir)
    return generator.generate_full_report(include_dataset_analysis=include_dataset_analysis)


# 命令行接口
if __name__ == "__main__":
    generate_report()
