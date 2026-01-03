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
    
    def _load_grid_results(self):
        """載入 SVD_KNN_GRID 網格搜索結果"""
        log_dir = Path('log')
        grid_results = []
        
        # 載入所有 SVD_KNN_GRID 結果 (1-100)
        for i in range(1, 101):
            config_name = f'SVD_KNN_GRID_{i:03d}'
            json_file = log_dir / f'{config_name}.json'
            
            if json_file.exists():
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    # 從配置中讀取實際的參數值
                    config = data.get('config', {})
                    n_components = config.get('n_components')
                    k_neighbors = config.get('k_neighbors')
                    
                    # 跳過無效配置
                    if n_components is None or k_neighbors is None:
                        continue
                    
                    metrics = data.get('metrics', {})
                    time_records = data.get('time_records', {})
                    
                    grid_results.append({
                        'config_name': config_name,
                        'n_components': n_components,
                        'k_neighbors': k_neighbors,
                        'hit_rate': metrics.get('hit_rate', 0),
                        'ndcg': metrics.get('ndcg', 0),
                        'rmse': metrics.get('rmse', 0),
                        'total_time': sum(time_records.values()) if time_records else 0
                    })
                except Exception as e:
                    print(f"⚠️ 無法讀取 {json_file}: {e}")
        
        return grid_results
    
    def _load_expand_results(self):
        """載入 SVD_KNN_EXPAND 網格搜索結果"""
        log_dir = Path('log')
        expand_results = []
        
        # 載入所有 SVD_KNN_EXPAND 結果 (1-36)
        for i in range(1, 37):
            config_name = f'SVD_KNN_EXPAND_{i:03d}'
            json_file = log_dir / f'{config_name}.json'
            
            if json_file.exists():
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    # 從配置中讀取實際的參數值
                    config = data.get('config', {})
                    n_components = config.get('n_components')
                    k_neighbors = config.get('k_neighbors')
                    
                    # 跳過無效配置
                    if n_components is None or k_neighbors is None:
                        continue
                    
                    metrics = data.get('metrics', {})
                    time_records = data.get('time_records', {})
                    
                    expand_results.append({
                        'config_name': config_name,
                        'n_components': n_components,
                        'k_neighbors': k_neighbors,
                        'hit_rate': metrics.get('hit_rate', 0),
                        'ndcg': metrics.get('ndcg', 0),
                        'rmse': metrics.get('rmse', 0),
                        'total_time': sum(time_records.values()) if time_records else 0
                    })
                except Exception as e:
                    print(f"⚠️ 無法讀取 {json_file}: {e}")
        
        return expand_results
    
    def _load_knn_baseline_results(self):
        """載入 KNN_BASELINE 純KNN基準線結果"""
        log_dir = Path('log')
        baseline_results = []
        
        # 載入所有 KNN_BASELINE 結果 (1-10)
        for i in range(1, 11):
            config_name = f'KNN_BASELINE_{i:03d}'
            json_file = log_dir / f'{config_name}.json'
            
            if json_file.exists():
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    # 從配置中讀取實際的 k_neighbors 值
                    config = data.get('config', {})
                    k_neighbors = config.get('k_neighbors', 5 * i)  # fallback to 5*i
                    
                    metrics = data.get('metrics', {})
                    time_records = data.get('time_records', {})
                    
                    baseline_results.append({
                        'config_name': config_name,
                        'k_neighbors': k_neighbors,
                        'hit_rate': metrics.get('hit_rate', 0),
                        'ndcg': metrics.get('ndcg', 0),
                        'rmse': metrics.get('rmse', 0),
                        'total_time': sum(time_records.values()) if time_records else 0
                    })
                except Exception as e:
                    print(f"⚠️ 無法讀取 {json_file}: {e}")
        
        return baseline_results
    
    def generate_svd_plots(self) -> bool:
        """生成 SVD 維度分析圖（從網格搜索結果提取）"""
        # 從 SVD_KNN_GRID 結果中提取 SVD 分析數據
        grid_results = self._load_grid_results()
        
        if not grid_results:
            print("⚠️ SVD_KNN_GRID 結果不足，跳過圖表生成")
            return False
        
        # 按 SVD 維度分組，對每個維度取所有 K 值的平均
        svd_analysis = {}
        for result in grid_results:
            dim = result['n_components']
            if dim not in svd_analysis:
                svd_analysis[dim] = {'hit_rates': [], 'ndcgs': [], 'rmses': [], 'times': []}
            
            svd_analysis[dim]['hit_rates'].append(result['hit_rate'])
            svd_analysis[dim]['ndcgs'].append(result['ndcg'])
            svd_analysis[dim]['rmses'].append(result['rmse'])
            svd_analysis[dim]['times'].append(result['total_time'])
        
        # 計算平均值
        dims = sorted(svd_analysis.keys())
        hit_rates = [np.mean(svd_analysis[d]['hit_rates']) for d in dims]
        ndcgs = [np.mean(svd_analysis[d]['ndcgs']) for d in dims]
        rmses = [np.mean(svd_analysis[d]['rmses']) for d in dims]
        times = [np.mean(svd_analysis[d]['times']) for d in dims]
        
        # 創建圖表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('SVD Dimension Analysis', fontsize=16, fontweight='bold')
        
        # 1. Hit Rate vs Dimension
        ax1.plot(dims, hit_rates, 'o-', linewidth=2, markersize=8, color='#2E86AB')
        ax1.set_xlabel('SVD Dimension', fontsize=12)
        ax1.set_ylabel('Hit Rate@10', fontsize=12)
        ax1.set_title('Hit Rate@10 vs SVD Dimension', fontsize=14, fontweight='bold')
        ax1.set_xscale('log', base=2)  # 使用對數刻度，因為維度是 2^N
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
        ax2.set_xscale('log', base=2)  # 使用對數刻度
        ax2.grid(True, alpha=0.3)
        
        # 3. RMSE vs Dimension
        ax3.plot(dims, rmses, 'o-', linewidth=2, markersize=8, color='#C73E1D')
        ax3.set_xlabel('SVD Dimension', fontsize=12)
        ax3.set_ylabel('RMSE', fontsize=12)
        ax3.set_title('RMSE vs SVD Dimension', fontsize=14, fontweight='bold')
        ax3.set_xscale('log', base=2)  # 使用對數刻度
        ax3.grid(True, alpha=0.3)
        
        # 4. Execution Time vs Dimension
        ax4.plot(dims, times, 'o-', linewidth=2, markersize=8, color='#6A994E')
        ax4.set_xlabel('SVD Dimension', fontsize=12)
        ax4.set_ylabel('Execution Time (seconds)', fontsize=12)
        ax4.set_title('Execution Time vs SVD Dimension', fontsize=14, fontweight='bold')
        ax4.set_xscale('log', base=2)  # 使用對數刻度
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.figures_dir / 'svd_dimension_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ SVD 分析圖已保存: {output_path}")
        return True
    
    def generate_knn_plots(self) -> bool:
        """生成 KNN K值分析圖（從網格搜索結果提取）"""
        # 從 SVD_KNN_GRID 結果中提取 KNN 分析數據
        grid_results = self._load_grid_results()
        
        if not grid_results:
            print("⚠️ SVD_KNN_GRID 結果不足，跳過圖表生成")
            return False
        
        # 按 K 值分組，對每個 K 值取所有 SVD 維度的平均
        knn_analysis = {}
        for result in grid_results:
            k = result['k_neighbors']
            if k not in knn_analysis:
                knn_analysis[k] = {'hit_rates': [], 'ndcgs': [], 'rmses': [], 'times': []}
            
            knn_analysis[k]['hit_rates'].append(result['hit_rate'])
            knn_analysis[k]['ndcgs'].append(result['ndcg'])
            knn_analysis[k]['rmses'].append(result['rmse'])
            knn_analysis[k]['times'].append(result['total_time'])
        
        # 計算平均值
        ks = sorted(knn_analysis.keys())
        hit_rates = [np.mean(knn_analysis[k]['hit_rates']) for k in ks]
        ndcgs = [np.mean(knn_analysis[k]['ndcgs']) for k in ks]
        rmses = [np.mean(knn_analysis[k]['rmses']) for k in ks]
        times = [np.mean(knn_analysis[k]['times']) for k in ks]
        
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
        # 找出各階段的最佳配置
        stages = {}
        
        # 輔助函數：找出階段最佳配置
        def find_best_in_stage(stage_name: str):
            results = self.analyzer.load_stage_results(stage_name)
            if results:
                best = max(results, key=lambda x: x['data'].get('metrics', {}).get('hit_rate', 0))
                return best['config_name']
            return None
        
        # FILTER 最佳
        filter_best = find_best_in_stage('FILTER')
        if filter_best:
            stages['FILTER'] = filter_best
        
        # KNN_BASELINE 最佳
        knn_baseline_best = find_best_in_stage('KNN_BASELINE')
        if knn_baseline_best:
            stages['KNN_BASELINE'] = knn_baseline_best
        
        # SVD_KNN_GRID 最佳
        grid_results = self._load_grid_results()
        if grid_results:
            best_grid = max(grid_results, key=lambda x: x['hit_rate'])
            stages['SVD_KNN_GRID'] = best_grid['config_name']
        
        # BIAS 最佳
        bias_best = find_best_in_stage('BIAS')
        if bias_best:
            stages['BIAS'] = bias_best
        
        # OPT 最佳
        opt_best = find_best_in_stage('OPT')
        if opt_best:
            stages['OPT'] = opt_best
        
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
    
    def generate_grid_heatmap(self) -> bool:
        """生成 SVD×KNN 網格搜索熱圖"""
        grid_results = self._load_grid_results()
        
        if not grid_results:
            print("⚠️ SVD_KNN_GRID 結果不足，跳過熱圖生成")
            return False
        
        # 準備數據矩陣
        svd_dims = sorted(set(r['n_components'] for r in grid_results))
        k_values = sorted(set(r['k_neighbors'] for r in grid_results))
        
        # 創建熱圖數據
        hit_rate_matrix = np.zeros((len(svd_dims), len(k_values)))
        ndcg_matrix = np.zeros((len(svd_dims), len(k_values)))
        
        for result in grid_results:
            i = svd_dims.index(result['n_components'])
            j = k_values.index(result['k_neighbors'])
            hit_rate_matrix[i, j] = result['hit_rate']
            ndcg_matrix[i, j] = result['ndcg']
        
        # 創建圖表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        fig.suptitle('SVD × KNN Grid Search Heatmap', fontsize=16, fontweight='bold')
        
        # Hit Rate 熱圖
        im1 = ax1.imshow(hit_rate_matrix, cmap='YlOrRd', aspect='auto')
        ax1.set_xticks(range(len(k_values)))
        ax1.set_yticks(range(len(svd_dims)))
        ax1.set_xticklabels(k_values)
        ax1.set_yticklabels(svd_dims)
        ax1.set_xlabel('K (Number of Neighbors)', fontsize=12)
        ax1.set_ylabel('SVD Dimension', fontsize=12)
        ax1.set_title('Hit Rate@10', fontsize=14, fontweight='bold')
        
        # 添加數值標註
        for i in range(len(svd_dims)):
            for j in range(len(k_values)):
                text = ax1.text(j, i, f'{hit_rate_matrix[i, j]:.3f}',
                               ha="center", va="center", color="black", fontsize=9)
        
        plt.colorbar(im1, ax=ax1, label='Hit Rate@10')
        
        # NDCG 熱圖
        im2 = ax2.imshow(ndcg_matrix, cmap='YlGnBu', aspect='auto')
        ax2.set_xticks(range(len(k_values)))
        ax2.set_yticks(range(len(svd_dims)))
        ax2.set_xticklabels(k_values)
        ax2.set_yticklabels(svd_dims)
        ax2.set_xlabel('K (Number of Neighbors)', fontsize=12)
        ax2.set_ylabel('SVD Dimension', fontsize=12)
        ax2.set_title('NDCG@10', fontsize=14, fontweight='bold')
        
        # 添加數值標註
        for i in range(len(svd_dims)):
            for j in range(len(k_values)):
                text = ax2.text(j, i, f'{ndcg_matrix[i, j]:.3f}',
                               ha="center", va="center", color="black", fontsize=9)
        
        plt.colorbar(im2, ax=ax2, label='NDCG@10')
        
        plt.tight_layout()
        
        output_path = self.figures_dir / 'svd_knn_grid_heatmap.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 網格搜索熱圖已保存: {output_path}")
        
        # 生成分析報告
        best_result = max(grid_results, key=lambda x: x['hit_rate'])
        print(f"\n📊 網格搜索分析:")
        print(f"   最佳配置: SVD={best_result['n_components']}, K={best_result['k_neighbors']}")
        print(f"   Hit Rate@10: {best_result['hit_rate']:.4f}")
        print(f"   NDCG@10: {best_result['ndcg']:.4f}")
        
        # 分析趨勢
        print(f"\n🔍 趨勢分析:")
        
        # SVD 維度效果
        svd_avg_hit_rates = [hit_rate_matrix[i, :].mean() for i in range(len(svd_dims))]
        svd_trend = "遞增" if svd_avg_hit_rates[-1] > svd_avg_hit_rates[0] else "遞減"
        print(f"   SVD 維度放大效果: {svd_trend}")
        print(f"   - 維度 {svd_dims[0]}: 平均 Hit Rate = {svd_avg_hit_rates[0]:.4f}")
        print(f"   - 維度 {svd_dims[-1]}: 平均 Hit Rate = {svd_avg_hit_rates[-1]:.4f}")
        
        # KNN K值效果
        k_avg_hit_rates = [hit_rate_matrix[:, j].mean() for j in range(len(k_values))]
        k_trend = "遞增" if k_avg_hit_rates[-1] > k_avg_hit_rates[0] else "遞減"
        print(f"   KNN K值放大效果: {k_trend}")
        print(f"   - K={k_values[0]}: 平均 Hit Rate = {k_avg_hit_rates[0]:.4f}")
        print(f"   - K={k_values[-1]}: 平均 Hit Rate = {k_avg_hit_rates[-1]:.4f}")
        
        # 建議
        print(f"\n💡 建議:")
        if svd_avg_hit_rates[-1] > svd_avg_hit_rates[0] and (svd_avg_hit_rates[-1] - svd_avg_hit_rates[-2]) > 0.001:
            print(f"   ⚠️  SVD 維度仍在改善，建議測試更大的維度（如 512, 1024）")
        else:
            print(f"   ✅ SVD 維度已達收斂，當前範圍已足夠")
        
        if k_avg_hit_rates[-1] > k_avg_hit_rates[0] and (k_avg_hit_rates[-1] - k_avg_hit_rates[-2]) > 0.001:
            print(f"   ⚠️  KNN K值仍在改善，建議測試更大的 K 值（如 128, 256）")
        else:
            print(f"   ✅ KNN K值已達收斂，當前範圍已足夠")
        
        return True
    
    def generate_expand_heatmap(self) -> bool:
        """生成 SVD_KNN_EXPAND 擴展網格搜索熱圖"""
        expand_results = self._load_expand_results()
        
        if not expand_results:
            print("⚠️ SVD_KNN_EXPAND 結果不足，跳過熱圖生成")
            return False
        
        # 準備數據矩陣
        svd_dims = sorted(set(r['n_components'] for r in expand_results))
        k_values = sorted(set(r['k_neighbors'] for r in expand_results))
        
        # 創建熱圖數據
        hit_rate_matrix = np.zeros((len(svd_dims), len(k_values)))
        ndcg_matrix = np.zeros((len(svd_dims), len(k_values)))
        
        for result in expand_results:
            i = svd_dims.index(result['n_components'])
            j = k_values.index(result['k_neighbors'])
            hit_rate_matrix[i, j] = result['hit_rate']
            ndcg_matrix[i, j] = result['ndcg']
        
        # 創建圖表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle('SVD × KNN Expand Grid Search Heatmap', fontsize=16, fontweight='bold')
        
        # Hit Rate 熱圖
        im1 = ax1.imshow(hit_rate_matrix, cmap='YlOrRd', aspect='auto')
        ax1.set_xticks(range(len(k_values)))
        ax1.set_yticks(range(len(svd_dims)))
        ax1.set_xticklabels(k_values)
        ax1.set_yticklabels(svd_dims)
        ax1.set_xlabel('K (Number of Neighbors)', fontsize=12)
        ax1.set_ylabel('SVD Dimension', fontsize=12)
        ax1.set_title('Hit Rate@10', fontsize=14, fontweight='bold')
        
        # 添加數值標註
        for i in range(len(svd_dims)):
            for j in range(len(k_values)):
                text = ax1.text(j, i, f'{hit_rate_matrix[i, j]:.3f}',
                               ha="center", va="center", color="black", fontsize=9)
        
        plt.colorbar(im1, ax=ax1, label='Hit Rate@10')
        
        # NDCG 熱圖
        im2 = ax2.imshow(ndcg_matrix, cmap='YlGnBu', aspect='auto')
        ax2.set_xticks(range(len(k_values)))
        ax2.set_yticks(range(len(svd_dims)))
        ax2.set_xticklabels(k_values)
        ax2.set_yticklabels(svd_dims)
        ax2.set_xlabel('K (Number of Neighbors)', fontsize=12)
        ax2.set_ylabel('SVD Dimension', fontsize=12)
        ax2.set_title('NDCG@10', fontsize=14, fontweight='bold')
        
        # 添加數值標註
        for i in range(len(svd_dims)):
            for j in range(len(k_values)):
                text = ax2.text(j, i, f'{ndcg_matrix[i, j]:.3f}',
                               ha="center", va="center", color="black", fontsize=9)
        
        plt.colorbar(im2, ax=ax2, label='NDCG@10')
        
        plt.tight_layout()
        
        output_path = self.figures_dir / 'svd_knn_expand_heatmap.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 擴展網格搜索熱圖已保存: {output_path}")
        
        # 生成分析報告
        best_result = max(expand_results, key=lambda x: x['hit_rate'])
        print(f"\n📊 擴展網格搜索分析:")
        print(f"   最佳配置: SVD={best_result['n_components']}, K={best_result['k_neighbors']}")
        print(f"   Hit Rate@10: {best_result['hit_rate']:.4f}")
        print(f"   NDCG@10: {best_result['ndcg']:.4f}")
        
        return True
    
    def generate_svd_vs_baseline_comparison(self) -> bool:
        """生成有無 SVD 對比圖：純KNN vs SVD+KNN"""
        baseline_results = self._load_knn_baseline_results()
        grid_results = self._load_grid_results()
        
        if not baseline_results:
            print("⚠️ KNN_BASELINE 結果不足，跳過對比圖生成")
            return False
        
        if not grid_results:
            print("⚠️ SVD_KNN_GRID 結果不足，跳過對比圖生成")
            return False
        
        # 對於每個 K 值，找出 SVD+KNN 的最佳結果
        svd_knn_by_k = {}
        for result in grid_results:
            k = result['k_neighbors']
            if k not in svd_knn_by_k or result['hit_rate'] > svd_knn_by_k[k]['hit_rate']:
                svd_knn_by_k[k] = result
        
        # 只比較兩者都有的 K 值
        baseline_by_k = {r['k_neighbors']: r for r in baseline_results}
        common_ks = sorted(set(baseline_by_k.keys()) & set(svd_knn_by_k.keys()))
        
        if not common_ks:
            print("⚠️ 沒有共同的 K 值可供比較")
            return False
        
        # 準備數據
        baseline_hit_rates = [baseline_by_k[k]['hit_rate'] for k in common_ks]
        baseline_ndcgs = [baseline_by_k[k]['ndcg'] for k in common_ks]
        baseline_times = [baseline_by_k[k]['total_time'] for k in common_ks]
        
        svd_hit_rates = [svd_knn_by_k[k]['hit_rate'] for k in common_ks]
        svd_ndcgs = [svd_knn_by_k[k]['ndcg'] for k in common_ks]
        svd_times = [svd_knn_by_k[k]['total_time'] for k in common_ks]
        svd_dims = [svd_knn_by_k[k]['n_components'] for k in common_ks]
        
        # 創建圖表
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        fig.suptitle('Pure KNN vs SVD+KNN Comparison', fontsize=18, fontweight='bold')
        
        # 1. Hit Rate 比較（折線圖）
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(common_ks, baseline_hit_rates, 'o-', linewidth=2, markersize=8, 
                color='#E63946', label='Pure KNN (no SVD)')
        ax1.plot(common_ks, svd_hit_rates, 's-', linewidth=2, markersize=8, 
                color='#2E86AB', label='SVD+KNN (best SVD dim per K)')
        ax1.set_xlabel('K (Number of Neighbors)', fontsize=12)
        ax1.set_ylabel('Hit Rate@10', fontsize=12)
        ax1.set_title('Hit Rate@10: Pure KNN vs SVD+KNN', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 2. NDCG 比較（折線圖）
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(common_ks, baseline_ndcgs, 'o-', linewidth=2, markersize=8, 
                color='#E63946', label='Pure KNN')
        ax2.plot(common_ks, svd_ndcgs, 's-', linewidth=2, markersize=8, 
                color='#2E86AB', label='SVD+KNN')
        ax2.set_xlabel('K (Number of Neighbors)', fontsize=12)
        ax2.set_ylabel('NDCG@10', fontsize=12)
        ax2.set_title('NDCG@10: Pure KNN vs SVD+KNN', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 3. 改善百分比（柱狀圖）
        ax3 = fig.add_subplot(gs[1, 0])
        improvements = [(svd - base) / base * 100 if base > 0 else 0 
                       for base, svd in zip(baseline_hit_rates, svd_hit_rates)]
        colors = ['#06D6A0' if imp > 0 else '#EF476F' for imp in improvements]
        bars = ax3.bar(range(len(common_ks)), improvements, color=colors, alpha=0.7, edgecolor='black')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax3.set_xlabel('K (Number of Neighbors)', fontsize=12)
        ax3.set_ylabel('Improvement (%)', fontsize=12)
        ax3.set_title('SVD Improvement over Pure KNN (Hit Rate@10)', fontsize=14, fontweight='bold')
        ax3.set_xticks(range(len(common_ks)))
        ax3.set_xticklabels(common_ks)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 標註數值
        for i, (bar, imp) in enumerate(zip(bars, improvements)):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{imp:+.1f}%',
                    ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
        
        # 4. 執行時間比較（柱狀圖）
        ax4 = fig.add_subplot(gs[1, 1])
        x = np.arange(len(common_ks))
        width = 0.35
        bars1 = ax4.bar(x - width/2, baseline_times, width, label='Pure KNN', 
                       color='#E63946', alpha=0.7, edgecolor='black')
        bars2 = ax4.bar(x + width/2, svd_times, width, label='SVD+KNN', 
                       color='#2E86AB', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('K (Number of Neighbors)', fontsize=12)
        ax4.set_ylabel('Execution Time (seconds)', fontsize=12)
        ax4.set_title('Execution Time: Pure KNN vs SVD+KNN', fontsize=14, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(common_ks)
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. 最佳 SVD 維度分布（柱狀圖）
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.bar(range(len(common_ks)), svd_dims, color='#F77F00', alpha=0.7, edgecolor='black')
        ax5.set_xlabel('K (Number of Neighbors)', fontsize=12)
        ax5.set_ylabel('Best SVD Dimension', fontsize=12)
        ax5.set_title('Optimal SVD Dimension for Each K', fontsize=14, fontweight='bold')
        ax5.set_xticks(range(len(common_ks)))
        ax5.set_xticklabels(common_ks)
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 標註維度值
        for i, dim in enumerate(svd_dims):
            ax5.text(i, dim, f'{dim}', ha='center', va='bottom', fontsize=9)
        
        # 6. 摘要統計表格
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.axis('off')
        
        # 計算統計數據
        avg_improvement = np.mean(improvements)
        max_improvement = max(improvements)
        max_imp_k = common_ks[improvements.index(max_improvement)]
        
        avg_baseline_hit = np.mean(baseline_hit_rates)
        avg_svd_hit = np.mean(svd_hit_rates)
        
        avg_baseline_time = np.mean(baseline_times)
        avg_svd_time = np.mean(svd_times)
        time_overhead = (avg_svd_time - avg_baseline_time) / avg_baseline_time * 100 if avg_baseline_time > 0 else 0
        
        summary_text = f"""
        📊 Summary Statistics
        
        Performance (Hit Rate@10):
        • Pure KNN Average: {avg_baseline_hit:.4f}
        • SVD+KNN Average: {avg_svd_hit:.4f}
        • Average Improvement: {avg_improvement:+.2f}%
        • Max Improvement: {max_improvement:+.2f}% (at K={max_imp_k})
        
        Efficiency (Execution Time):
        • Pure KNN Average: {avg_baseline_time:.1f}s
        • SVD+KNN Average: {avg_svd_time:.1f}s
        • Time Overhead: {time_overhead:+.1f}%
        
        Conclusion:
        {'✅ SVD brings significant improvement!' if avg_improvement > 1 else '⚠️ SVD improvement is marginal'}
        {'⏱️ Acceptable time overhead' if time_overhead < 50 else '⚠️ Significant time cost'}
        """
        
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, 
                fontsize=11, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        
        output_path = self.figures_dir / 'svd_vs_baseline_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ SVD vs Baseline 對比圖已保存: {output_path}")
        
        # 輸出詳細分析
        print(f"\n📊 SVD vs Pure KNN 詳細分析:")
        print(f"   平均改善: {avg_improvement:+.2f}%")
        print(f"   最大改善: {max_improvement:+.2f}% (K={max_imp_k})")
        print(f"   時間開銷: {time_overhead:+.1f}%")
        
        if avg_improvement > 5:
            print(f"   💡 結論: SVD 帶來顯著效能提升，值得使用！")
        elif avg_improvement > 1:
            print(f"   💡 結論: SVD 有輕微改善，可考慮使用")
        else:
            print(f"   💡 結論: SVD 改善不明顯，純KNN可能更實用")
        
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
        results['plots']['grid_heatmap'] = self.generate_grid_heatmap()
        results['plots']['expand_heatmap'] = self.generate_expand_heatmap()
        print()
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
            "figures/svd_knn_grid_heatmap.png",
            "figures/svd_dimension_analysis.png",
            "figures/knn_k_value_analysis.png",
            "figures/svd_vs_baseline_comparison.png",
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


# 命令行接口
if __name__ == "__main__":
    generate_report()
