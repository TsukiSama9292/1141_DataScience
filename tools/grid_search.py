#!/usr/bin/env python3
"""
網格搜尋工具 - 自動化超參數優化
支援全面的參數組合搜尋並生成完整報告
"""

import sys
import json
import time
import argparse
import itertools
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# 添加 src 到路徑
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from movie_recommendation.experiment import Experiment, ExperimentConfig
from movie_recommendation.analysis import ExperimentAnalyzer
from movie_recommendation.report_generator import ReportGenerator


class GridSearch:
    """網格搜尋引擎"""
    
    def __init__(self, output_dir: str = "grid_search_results"):
        """
        初始化網格搜尋
        
        Args:
            output_dir: 結果輸出目錄
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
        self.best_config = None
        self.best_score = -float('inf')
        
    def define_search_space(
        self,
        n_components: Optional[List[int]] = None,
        k_neighbors: Optional[List[int]] = None,
        min_item_ratings: Optional[List[int]] = None,
        use_svd: Optional[List[bool]] = None,
        use_item_bias: Optional[List[bool]] = None,
        use_time_decay: Optional[List[bool]] = None,
        use_tfidf: Optional[List[bool]] = None,
        n_samples: int = 500,
        top_n: int = 10,
        random_state: int = 42
    ) -> List[Dict[str, Any]]:
        """
        定義搜尋空間
        
        Args:
            n_components: SVD 維度列表
            k_neighbors: KNN 鄰居數列表
            min_item_ratings: 最小評分數列表
            use_svd: 是否使用 SVD
            use_item_bias: 是否使用 Item Bias
            use_time_decay: 是否使用時間衰減
            use_tfidf: 是否使用 TF-IDF
            n_samples: 評估樣本數
            top_n: 推薦數量
            random_state: 隨機種子
            
        Returns:
            參數組合列表
        """
        # 默認搜尋空間：SVD 使用 2^n (n=1..10)，KNN 使用 5*n (n=1..10)
        if n_components is None:
            n_components = [2**n for n in range(1, 11)]  # [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        if k_neighbors is None:
            k_neighbors = [5*n for n in range(1, 11)]    # [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        if min_item_ratings is None:
            min_item_ratings = [0]
        if use_svd is None:
            use_svd = [True]
        if use_item_bias is None:
            use_item_bias = [False]
        if use_time_decay is None:
            use_time_decay = [False]
        if use_tfidf is None:
            use_tfidf = [False]
        
        # 生成所有組合
        param_grid = {
            'n_components': n_components,
            'k_neighbors': k_neighbors,
            'min_item_ratings': min_item_ratings,
            'use_svd': use_svd,
            'use_item_bias': use_item_bias,
            'use_time_decay': use_time_decay,
            'use_tfidf': use_tfidf,
        }
        
        # 生成笛卡爾積
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(itertools.product(*values))
        
        configs = []
        for combo in combinations:
            config = dict(zip(keys, combo))
            # 添加固定參數
            config['data_limit'] = None
            config['n_samples'] = n_samples
            config['top_n'] = top_n
            config['random_state'] = random_state
            config['use_timestamp'] = False
            
            # 如果不使用 SVD，則不需要 n_components
            if not config['use_svd']:
                config['n_components'] = None
            
            configs.append(config)
        
        return configs
    
    def run_search(
        self,
        configs: List[Dict[str, Any]],
        metric: str = 'hit_rate',
        save_all: bool = True
    ) -> Dict[str, Any]:
        """
        執行網格搜尋
        
        Args:
            configs: 配置列表
            metric: 優化目標指標
            save_all: 是否保存所有結果
            
        Returns:
            搜尋結果摘要
        """
        total_configs = len(configs)
        print("=" * 80)
        print("🔍 網格搜尋開始")
        print("=" * 80)
        print(f"總配置數: {total_configs}")
        print(f"優化指標: {metric}")
        print(f"預計時間: ~{total_configs * 2 / 60:.1f} 分鐘 (假設每個配置 2 秒)")
        print("=" * 80)
        print()
        
        start_time = time.time()
        
        for idx, config_dict in enumerate(configs, 1):
            config_name = f"GRID_{idx:04d}"
            
            print(f"[{idx}/{total_configs}] 執行 {config_name}...")
            
            try:
                # 創建配置對象
                config = ExperimentConfig(**config_dict)
                
                # 執行實驗
                experiment = Experiment(config, config_name=config_name)
                metrics = experiment.run()
                
                # 記錄結果
                result = {
                    'config_name': config_name,
                    'config': config_dict,
                    'metrics': metrics,
                    'score': metrics[metric]
                }
                
                self.results.append(result)
                
                # 更新最佳配置
                if result['score'] > self.best_score:
                    self.best_score = result['score']
                    self.best_config = result
                    print(f"  ✨ 新的最佳配置! {metric} = {self.best_score:.4f}")
                
                # 保存結果
                if save_all:
                    result_file = self.output_dir / f"{config_name}.json"
                    with open(result_file, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
                
                # 進度報告
                elapsed = time.time() - start_time
                avg_time = elapsed / idx
                remaining = (total_configs - idx) * avg_time
                print(f"  {metric} = {result['score']:.4f} | "
                      f"已耗時: {elapsed/60:.1f}分 | "
                      f"預計剩餘: {remaining/60:.1f}分")
                
            except Exception as e:
                print(f"  ❌ 錯誤: {e}")
                continue
        
        total_time = time.time() - start_time
        
        print()
        print("=" * 80)
        print("✅ 網格搜尋完成")
        print("=" * 80)
        print(f"總執行時間: {total_time/60:.1f} 分鐘")
        print(f"成功配置數: {len(self.results)}/{total_configs}")
        print(f"最佳 {metric}: {self.best_score:.4f}")
        print("=" * 80)
        
        # 生成摘要
        summary = {
            'search_type': 'grid_search',
            'total_configs': total_configs,
            'successful_configs': len(self.results),
            'metric': metric,
            'best_score': self.best_score,
            'best_config': self.best_config,
            'total_time_seconds': total_time,
            'timestamp': datetime.now().isoformat()
        }
        
        # 保存摘要
        summary_file = self.output_dir / 'grid_search_summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        return summary
    
    def generate_report(self, metric: str = 'hit_rate'):
        """
        生成完整報告
        
        Args:
            metric: 主要指標
        """
        print("\n" + "=" * 80)
        print("📊 生成完整報告")
        print("=" * 80)
        
        if not self.results:
            print("❌ 沒有結果可供分析")
            return
        
        # 排序結果
        sorted_results = sorted(self.results, key=lambda x: x['score'], reverse=True)
        
        # 生成 Markdown 報告
        report_file = self.output_dir / 'grid_search_report.md'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 網格搜尋完整報告\n\n")
            f.write(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 摘要
            f.write("## 搜尋摘要\n\n")
            f.write(f"- **總配置數**: {len(self.results)}\n")
            f.write(f"- **優化指標**: {metric}\n")
            f.write(f"- **最佳得分**: {self.best_score:.4f}\n\n")
            
            # Top 10 配置
            f.write("## Top 10 配置\n\n")
            f.write("| 排名 | 配置 | Hit Rate | NDCG | RMSE | SVD | K | 備註 |\n")
            f.write("|------|------|----------|------|------|-----|---|------|\n")
            
            for rank, result in enumerate(sorted_results[:10], 1):
                config = result['config']
                metrics = result['metrics']
                
                svd_str = f"{config.get('n_components', 'N/A')}" if config.get('use_svd') else "無"
                k_str = str(config.get('k_neighbors', 'N/A'))
                
                notes = []
                if config.get('use_item_bias'):
                    notes.append("Bias")
                if config.get('use_time_decay'):
                    notes.append("TimeDecay")
                if config.get('use_tfidf'):
                    notes.append("TF-IDF")
                note_str = ", ".join(notes) if notes else "-"
                
                f.write(f"| {rank} | {result['config_name']} | "
                       f"{metrics['hit_rate']:.4f} | "
                       f"{metrics['ndcg']:.4f} | "
                       f"{metrics['rmse']:.4f} | "
                       f"{svd_str} | {k_str} | {note_str} |\n")
            
            f.write("\n")
            
            # 最佳配置詳情
            f.write("## 最佳配置詳情\n\n")
            f.write("```json\n")
            f.write(json.dumps(self.best_config, indent=2, ensure_ascii=False))
            f.write("\n```\n\n")
            
            # 參數分析
            f.write("## 參數影響分析\n\n")
            
            # SVD 維度分析
            if any(r['config'].get('use_svd') for r in self.results):
                f.write("### SVD 維度影響\n\n")
                svd_analysis = self._analyze_parameter('n_components', metric)
                f.write("| SVD 維度 | 平均 Hit Rate | 配置數 |\n")
                f.write("|----------|---------------|--------|\n")
                for value, avg_score, count in sorted(svd_analysis, key=lambda x: x[1], reverse=True):
                    if value is not None:
                        f.write(f"| {value} | {avg_score:.4f} | {count} |\n")
                f.write("\n")
            
            # K 值分析
            f.write("### KNN 鄰居數影響\n\n")
            k_analysis = self._analyze_parameter('k_neighbors', metric)
            f.write("| K 值 | 平均 Hit Rate | 配置數 |\n")
            f.write("|------|---------------|--------|\n")
            for value, avg_score, count in sorted(k_analysis, key=lambda x: x[1], reverse=True):
                f.write(f"| {value} | {avg_score:.4f} | {count} |\n")
            f.write("\n")
            
            # 所有結果
            f.write("## 所有配置結果\n\n")
            f.write("| 配置 | Hit Rate | NDCG | RMSE | 配置詳情 |\n")
            f.write("|------|----------|------|------|----------|\n")
            
            for result in sorted_results:
                config = result['config']
                metrics = result['metrics']
                
                config_str = f"SVD={config.get('n_components', 'N/A') if config.get('use_svd') else '無'}, K={config.get('k_neighbors')}"
                
                f.write(f"| {result['config_name']} | "
                       f"{metrics['hit_rate']:.4f} | "
                       f"{metrics['ndcg']:.4f} | "
                       f"{metrics['rmse']:.4f} | "
                       f"{config_str} |\n")
        
        print(f"✅ 報告已生成: {report_file}")
        
        # 生成可視化
        self._generate_visualizations(metric)
    
    def _analyze_parameter(self, param_name: str, metric: str) -> List[tuple]:
        """
        分析單個參數的影響
        
        Args:
            param_name: 參數名稱
            metric: 評估指標
            
        Returns:
            (參數值, 平均得分, 配置數) 列表
        """
        param_scores = {}
        
        for result in self.results:
            value = result['config'].get(param_name)
            score = result['metrics'][metric]
            
            if value not in param_scores:
                param_scores[value] = []
            param_scores[value].append(score)
        
        analysis = []
        for value, scores in param_scores.items():
            avg_score = sum(scores) / len(scores)
            count = len(scores)
            analysis.append((value, avg_score, count))
        
        return analysis
    
    def _generate_visualizations(self, metric: str):
        """生成可視化圖表"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # SVD 維度 vs 性能
            svd_results = {}
            for result in self.results:
                if result['config'].get('use_svd'):
                    n_comp = result['config'].get('n_components')
                    score = result['metrics'][metric]
                    if n_comp not in svd_results:
                        svd_results[n_comp] = []
                    svd_results[n_comp].append(score)
            
            if svd_results:
                fig, ax = plt.subplots(figsize=(10, 6))
                dimensions = sorted(svd_results.keys())
                avg_scores = [np.mean(svd_results[d]) for d in dimensions]
                std_scores = [np.std(svd_results[d]) for d in dimensions]
                
                ax.plot(dimensions, avg_scores, marker='o', linewidth=2, markersize=8)
                ax.fill_between(dimensions, 
                               [a - s for a, s in zip(avg_scores, std_scores)],
                               [a + s for a, s in zip(avg_scores, std_scores)],
                               alpha=0.3)
                ax.set_xlabel('SVD 維度', fontsize=12)
                ax.set_ylabel(f'{metric.upper()}', fontsize=12)
                ax.set_title('SVD 維度影響分析', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(self.output_dir / 'svd_analysis.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("✅ 已生成 SVD 分析圖")
            
            # K 值 vs 性能
            k_results = {}
            for result in self.results:
                k = result['config'].get('k_neighbors')
                score = result['metrics'][metric]
                if k not in k_results:
                    k_results[k] = []
                k_results[k].append(score)
            
            if k_results:
                fig, ax = plt.subplots(figsize=(10, 6))
                k_values = sorted(k_results.keys())
                avg_scores = [np.mean(k_results[k]) for k in k_values]
                std_scores = [np.std(k_results[k]) for k in k_values]
                
                ax.plot(k_values, avg_scores, marker='s', linewidth=2, markersize=8)
                ax.fill_between(k_values,
                               [a - s for a, s in zip(avg_scores, std_scores)],
                               [a + s for a, s in zip(avg_scores, std_scores)],
                               alpha=0.3)
                ax.set_xlabel('KNN 鄰居數 (K)', fontsize=12)
                ax.set_ylabel(f'{metric.upper()}', fontsize=12)
                ax.set_title('KNN 鄰居數影響分析', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(self.output_dir / 'knn_analysis.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("✅ 已生成 KNN 分析圖")
                
        except ImportError:
            print("⚠️  無法生成可視化 (需要 matplotlib)")


def main():
    parser = argparse.ArgumentParser(description='網格搜尋工具')
    parser.add_argument('--preset', type=str, choices=['quick', 'standard', 'full'],
                       default='standard', help='預設搜尋模式')
    parser.add_argument('--output', type=str, default='grid_search_results',
                       help='輸出目錄')
    parser.add_argument('--metric', type=str, default='hit_rate',
                       help='優化指標')
    parser.add_argument('--samples', type=int, default=500,
                       help='評估樣本數')
    
    args = parser.parse_args()
    
    # 初始化搜尋器
    searcher = GridSearch(output_dir=args.output)
    
    # 根據預設模式定義搜尋空間
    if args.preset == 'quick':
        # 快速搜尋：少量配置 (5×5=25個實驗)
        configs = searcher.define_search_space(
            n_components=[8, 32, 128, 512, 1024],  # 2^3, 2^5, 2^7, 2^9, 2^10
            k_neighbors=[5, 15, 25, 35, 50],        # 5*1, 5*3, 5*5, 5*7, 5*10
            n_samples=args.samples
        )
    elif args.preset == 'standard':
        # 標準搜尋：平衡配置 (10×10=100個實驗)
        configs = searcher.define_search_space(
            n_components=[2**n for n in range(1, 11)],  # [2, 4, 8, ..., 1024]
            k_neighbors=[5*n for n in range(1, 11)],     # [5, 10, 15, ..., 50]
            n_samples=args.samples
        )
    else:  # full
        # 完整搜尋：更密集的配置 (15×15=225個實驗)
        configs = searcher.define_search_space(
            n_components=[2**n for n in range(1, 11)] + [384, 512, 768, 1024, 1536],  # 擴展範圍
            k_neighbors=[5*n for n in range(1, 16)],  # [5, 10, 15, ..., 75]
            n_samples=args.samples
        )
    
    # 執行搜尋
    summary = searcher.run_search(configs, metric=args.metric)
    
    # 生成報告
    searcher.generate_report(metric=args.metric)
    
    print("\n" + "=" * 80)
    print("🎉 網格搜尋完成！")
    print("=" * 80)
    print(f"最佳配置: {summary['best_config']['config_name']}")
    print(f"最佳 {args.metric}: {summary['best_score']:.4f}")
    print(f"結果保存於: {args.output}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
