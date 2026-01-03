"""
實驗結果分析模組

提供各階段實驗結果的分析功能，包括：
- 進度檢查
- SVD 維度分析
- KNN K值分析
- 最佳配置提取
- 資料集統計分析
"""

import json
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict


class ExperimentAnalyzer:
    """實驗結果分析器"""
    
    def __init__(self, log_dir: str = 'log', run_dir: str = 'run', data_dir: str = 'data'):
        self.log_dir = Path(log_dir)
        self.run_dir = Path(run_dir)
        self.data_dir = Path(data_dir)
        
        # 各階段配置定義
        self.stages = {
            'DS': ['DS_001', 'DS_002', 'DS_003', 'DS_004'],
            'FILTER': ['FILTER_001', 'FILTER_002', 'FILTER_003', 
                      'FILTER_004', 'FILTER_005', 'FILTER_006'],
            'KNN_BASELINE': [f'KNN_BASELINE_{i:03d}' for i in range(1, 11)],
            'SVD': [f'SVD_{i:03d}' for i in range(1, 16)],
            'KNN': [f'KNN_{i:03d}' for i in range(1, 8)],
            'SVD_KNN_EXPAND': [f'SVD_KNN_EXPAND_{i:03d}' for i in range(1, 37)],
            'BIAS': ['BIAS_001', 'BIAS_002'],
            'OPT': ['OPT_001', 'OPT_002']
        }
    
    def load_result(self, config_name: str) -> Optional[Dict[str, Any]]:
        """載入單個配置的結果"""
        json_file = self.log_dir / f"{config_name}.json"
        if not json_file.exists():
            return None
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ 無法讀取 {json_file}: {e}")
            return None
    
    def load_stage_results(self, stage: str) -> List[Dict[str, Any]]:
        """載入指定階段的所有結果"""
        results = []
        for config_name in self.stages.get(stage, []):
            data = self.load_result(config_name)
            if data:
                results.append({
                    'config_name': config_name,
                    'data': data
                })
        return results
    
    def extract_param_from_config(self, config_name: str, param_pattern: str) -> Optional[int]:
        """從配置文件中提取參數值"""
        config_file = self.run_dir / f"{config_name}.py"
        if not config_file.exists():
            return None
        
        try:
            content = config_file.read_text(encoding='utf-8')
            match = re.search(param_pattern, content)
            if match:
                return int(match.group(1))
        except Exception:
            pass
        return None
    
    def check_progress(self) -> Dict[str, Any]:
        """檢查執行進度"""
        if not self.log_dir.exists():
            return {
                'total_completed': 0,
                'total_configs': sum(len(configs) for configs in self.stages.values()),
                'stages': {},
                'analysis_status': {'svd': False, 'knn': False}
            }
        
        stage_progress = {}
        total_completed = 0
        
        for stage_name, configs in self.stages.items():
            completed = []
            missing = []
            
            for config in configs:
                json_file = self.log_dir / f"{config}.json"
                if json_file.exists():
                    completed.append(config)
                else:
                    missing.append(config)
            
            stage_progress[stage_name] = {
                'completed': completed,
                'missing': missing,
                'total': len(configs),
                'count': len(completed),
                'rate': len(completed) / len(configs) * 100
            }
            total_completed += len(completed)
        
        # 檢查分析狀態
        best_svd = (self.log_dir / 'best_svd.json').exists()
        best_knn = (self.log_dir / 'best_knn.json').exists()
        
        return {
            'total_completed': total_completed,
            'total_configs': sum(len(configs) for configs in self.stages.values()),
            'stages': stage_progress,
            'analysis_status': {
                'svd': best_svd,
                'knn': best_knn
            }
        }
    
    def analyze_svd(self) -> Optional[Dict[str, Any]]:
        """分析 SVD 階段結果"""
        results = self.load_stage_results('SVD')
        
        if not results:
            return None
        
        # 提取並排序結果
        analysis = []
        for result in results:
            config_name = result['config_name']
            data = result['data']
            
            # 提取 SVD 維度
            n_components = self.extract_param_from_config(
                config_name, r'n_components\s*=\s*(\d+)'
            )
            
            if n_components:
                metrics = data.get('metrics', {})
                time_records = data.get('time_records', {})
                
                analysis.append({
                    'config_name': config_name,
                    'n_components': n_components,
                    'hit_rate': metrics.get('hit_rate', 0),
                    'ndcg': metrics.get('ndcg', 0),
                    'rmse': metrics.get('rmse', 0),
                    'total_time': sum(time_records.values()) if time_records else 0,
                    'memory_mb': data.get('peak_memory_mb', 0)
                })
        
        # 按 Hit Rate 排序
        analysis.sort(key=lambda x: x['hit_rate'], reverse=True)
        
        if analysis:
            best = analysis[0]
            # 保存最佳配置
            best_file = self.log_dir / 'best_svd.json'
            with open(best_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'best_config': best['config_name'],
                    'best_n_components': best['n_components'],
                    'metrics': {
                        'hit_rate': best['hit_rate'],
                        'ndcg': best['ndcg'],
                        'rmse': best['rmse']
                    }
                }, f, indent=2, ensure_ascii=False)
        
        return {
            'results': analysis,
            'best': analysis[0] if analysis else None,
            'count': len(analysis)
        }
    
    def analyze_knn(self) -> Optional[Dict[str, Any]]:
        """分析 KNN 階段結果"""
        results = self.load_stage_results('KNN')
        
        if not results:
            return None
        
        # 提取並排序結果
        analysis = []
        for result in results:
            config_name = result['config_name']
            data = result['data']
            
            # 提取 K 值
            k_neighbors = self.extract_param_from_config(
                config_name, r'k_neighbors\s*=\s*(\d+)'
            )
            
            if k_neighbors:
                metrics = data.get('metrics', {})
                time_records = data.get('time_records', {})
                
                analysis.append({
                    'config_name': config_name,
                    'k_neighbors': k_neighbors,
                    'hit_rate': metrics.get('hit_rate', 0),
                    'ndcg': metrics.get('ndcg', 0),
                    'rmse': metrics.get('rmse', 0),
                    'total_time': sum(time_records.values()) if time_records else 0
                })
        
        # 按 Hit Rate 排序
        analysis.sort(key=lambda x: x['hit_rate'], reverse=True)
        
        if analysis:
            best = analysis[0]
            # 保存最佳配置
            best_file = self.log_dir / 'best_knn.json'
            with open(best_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'best_config': best['config_name'],
                    'best_k_neighbors': best['k_neighbors'],
                    'metrics': {
                        'hit_rate': best['hit_rate'],
                        'ndcg': best['ndcg'],
                        'rmse': best['rmse']
                    }
                }, f, indent=2, ensure_ascii=False)
        
        return {
            'results': analysis,
            'best': analysis[0] if analysis else None,
            'count': len(analysis)
        }
    
    def get_best_configs(self) -> Dict[str, Any]:
        """獲取所有最佳配置"""
        best_configs = {}
        
        # 讀取保存的最佳配置
        best_svd_file = self.log_dir / 'best_svd.json'
        if best_svd_file.exists():
            with open(best_svd_file, 'r', encoding='utf-8') as f:
                best_configs['svd'] = json.load(f)
        
        best_knn_file = self.log_dir / 'best_knn.json'
        if best_knn_file.exists():
            with open(best_knn_file, 'r', encoding='utf-8') as f:
                best_configs['knn'] = json.load(f)
        
        return best_configs
    
    def generate_summary_table(self) -> List[Dict[str, Any]]:
        """生成所有階段的摘要表格"""
        summary = []
        
        # 移除 DS 階段，因為它測試的是資料量而非超參數
        for stage_name in ['FILTER', 'KNN_BASELINE', 'SVD_KNN_GRID', 'SVD_KNN_EXPAND', 'BIAS', 'OPT']:
            # SVD_KNN_GRID 和 SVD_KNN_EXPAND 特殊處理
            if stage_name in ['SVD_KNN_GRID', 'SVD_KNN_EXPAND']:
                # 載入所有結果
                log_dir = Path(self.log_dir)
                grid_results = []
                max_exp = 101 if stage_name == 'SVD_KNN_GRID' else 37
                for i in range(1, max_exp):
                    config_name = f'{stage_name}_{i:03d}'
                    json_file = log_dir / f'{config_name}.json'
                    if json_file.exists():
                        try:
                            with open(json_file, 'r') as f:
                                data = json.load(f)
                            grid_results.append({'config_name': config_name, 'data': data})
                        except:
                            pass
                results = grid_results
            else:
                results = self.load_stage_results(stage_name)
            
            if results:
                # 找出最佳結果
                best_result = max(
                    results, 
                    key=lambda x: x['data'].get('metrics', {}).get('hit_rate', 0)
                )
                
                metrics = best_result['data'].get('metrics', {})
                summary.append({
                    'stage': stage_name,
                    'best_config': best_result['config_name'],
                    'hit_rate': metrics.get('hit_rate', 0),
                    'ndcg': metrics.get('ndcg', 0),
                    'rmse': metrics.get('rmse', 0),
                    'count': len(results)
                })
        
        return summary


class DatasetAnalyzer:
    """資料集統計分析器"""
    
    def __init__(self, data_loader=None):
        """
        初始化資料集分析器
        
        Args:
            data_loader: DataLoader 實例，如果為 None 則創建新實例
        """
        if data_loader is None:
            from .data_loader import DataLoader
            self.data_loader = DataLoader()
        else:
            self.data_loader = data_loader
    
    def load_ratings_sample(self, limit: int = 100000) -> Optional[pd.DataFrame]:
        """載入評分資料樣本"""
        try:
            # 使用 DataLoader 載入資料
            _, ratings = self.data_loader.load_data(limit=limit)
            return ratings
        except Exception as e:
            print(f"❌ 載入資料失敗: {e}")
            return None
    
    def load_ratings_chunked(self, chunksize: int = 500000):
        """分批載入評分資料（生成器）
        
        Args:
            chunksize: 每批讀取的資料量
            
        Yields:
            pd.DataFrame: 評分資料批次
        """
        try:
            import kagglehub
            path = kagglehub.dataset_download(self.data_loader.dataset_name)
            rating_file = f"{path}/rating.csv"
            
            # 使用 chunksize 參數分批讀取
            for chunk in pd.read_csv(rating_file, usecols=['userId', 'movieId', 'rating'], 
                                     chunksize=chunksize):
                yield chunk
        except Exception as e:
            print(f"❌ 分批載入資料失敗: {e}")
            return
    
    def analyze_rating_distribution(self, ratings: pd.DataFrame = None, use_full_dataset: bool = False) -> Optional[Dict[str, Any]]:
        """分析評分分布
        
        Args:
            ratings: 預載入的評分資料（如果為 None 則自動載入）
            use_full_dataset: 是否使用完整資料集（分批處理）
        """
        if use_full_dataset and ratings is None:
            # 使用分批處理完整資料集
            print("📊 使用分批處理分析完整資料集...")
            rating_counts = {}
            total = 0
            sum_rating = 0
            sum_sq = 0
            
            for i, chunk in enumerate(self.load_ratings_chunked(), 1):
                chunk_counts = chunk['rating'].value_counts()
                for rating, count in chunk_counts.items():
                    rating_counts[rating] = rating_counts.get(rating, 0) + count
                
                total += len(chunk)
                sum_rating += chunk['rating'].sum()
                sum_sq += (chunk['rating'] ** 2).sum()
                
                if i % 10 == 0:
                    print(f"   處理批次 {i} (已處理 {total:,} 筆)...")
            
            mean = sum_rating / total
            std = np.sqrt(sum_sq / total - mean ** 2)
            
            # 計算中位數（從分布估算）
            sorted_ratings = sorted(rating_counts.items())
            cumsum = 0
            median = 3.0
            for rating, count in sorted_ratings:
                cumsum += count
                if cumsum >= total / 2:
                    median = rating
                    break
            
            stats = {
                'distribution': rating_counts,
                'mean': float(mean),
                'median': float(median),
                'std': float(std),
                'min': float(min(rating_counts.keys())),
                'max': float(max(rating_counts.keys())),
                'total_ratings': total
            }
            print(f"✅ 完成分析 {total:,} 筆評分")
            return stats
        
        # 使用樣本資料
        if ratings is None:
            ratings = self.load_ratings_sample()
        
        if ratings is None or ratings.empty:
            return None
        
        # 計算評分分布
        rating_counts = ratings['rating'].value_counts().sort_index()
        
        stats = {
            'distribution': rating_counts.to_dict(),
            'mean': float(ratings['rating'].mean()),
            'median': float(ratings['rating'].median()),
            'std': float(ratings['rating'].std()),
            'min': float(ratings['rating'].min()),
            'max': float(ratings['rating'].max()),
            'total_ratings': len(ratings)
        }
        
        return stats
    
    def analyze_user_activity(self, ratings: pd.DataFrame = None, use_full_dataset: bool = False, top_n: int = 10000) -> Optional[Dict[str, Any]]:
        """分析使用者活躍度（長尾效應）
        
        Args:
            ratings: 預載入的評分資料
            use_full_dataset: 是否使用完整資料集
            top_n: 返回前 N 個最活躍用戶（用於長尾圖）
        """
        if use_full_dataset and ratings is None:
            print("📊 使用分批處理分析用戶活躍度...")
            user_counts_dict = {}
            total = 0
            
            for i, chunk in enumerate(self.load_ratings_chunked(), 1):
                chunk_counts = chunk['userId'].value_counts()
                for user_id, count in chunk_counts.items():
                    user_counts_dict[user_id] = user_counts_dict.get(user_id, 0) + count
                
                total += len(chunk)
                if i % 10 == 0:
                    print(f"   處理批次 {i} (已處理 {total:,} 筆)...")
            
            # 轉換為排序後的數組（只保留前 top_n）
            user_counts = pd.Series(user_counts_dict).sort_values(ascending=False)
            user_counts_values = user_counts.head(top_n).values
            
            print(f"✅ 完成分析 {len(user_counts):,} 位用戶，保留前 {top_n:,} 位用於長尾圖")
        else:
            if ratings is None:
                ratings = self.load_ratings_sample()
            
            if ratings is None or ratings.empty:
                return None
            
            # 計算每個使用者的評分數量
            user_counts = ratings['userId'].value_counts()
            user_counts_values = user_counts.values
        
        # 統一處理統計數據
        if use_full_dataset:
            user_counts_series = pd.Series(sorted(user_counts_dict.values(), reverse=True))
            total_ratings = sum(user_counts_dict.values())
        else:
            user_counts_series = user_counts
            total_ratings = user_counts.sum()
        
        top_20_percent = int(len(user_counts_series) * 0.2)
        top_20_ratings = user_counts_series.iloc[:top_20_percent].sum()
        
        stats = {
            'total_users': len(user_counts_series),
            'mean_ratings_per_user': float(user_counts_series.mean()),
            'median_ratings_per_user': float(user_counts_series.median()),
            'std_ratings_per_user': float(user_counts_series.std()),
            'max_ratings': int(user_counts_series.max()),
            'min_ratings': int(user_counts_series.min()),
            'long_tail': {
                'top_20_percent_users': top_20_percent,
                'top_20_percent_ratings': int(top_20_ratings),
                'top_20_percent_ratio': float(top_20_ratings / total_ratings)
            },
            'quantiles': {
                '25%': int(user_counts_series.quantile(0.25)),
                '50%': int(user_counts_series.quantile(0.50)),
                '75%': int(user_counts_series.quantile(0.75)),
                '90%': int(user_counts_series.quantile(0.90)),
                '95%': int(user_counts_series.quantile(0.95)),
                '99%': int(user_counts_series.quantile(0.99))
            }
        }
        
        # 增加只用於繪圖的字段（不包含在 JSON 輸出中）
        stats['_plot_data'] = user_counts_values  # 前缀 _ 表示內部使用
        
        return stats
    
    def analyze_item_popularity(self, ratings: pd.DataFrame = None, use_full_dataset: bool = False, top_n: int = 10000) -> Optional[Dict[str, Any]]:
        """分析電影流行度（冷啟動問題）
        
        Args:
            ratings: 預載入的評分資料
            use_full_dataset: 是否使用完整資料集
            top_n: 返回前 N 個最熱門電影（用於長尾圖）
        """
        if use_full_dataset and ratings is None:
            print("📊 使用分批處理分析電影流行度...")
            item_counts_dict = {}
            total = 0
            
            for i, chunk in enumerate(self.load_ratings_chunked(), 1):
                chunk_counts = chunk['movieId'].value_counts()
                for movie_id, count in chunk_counts.items():
                    item_counts_dict[movie_id] = item_counts_dict.get(movie_id, 0) + count
                
                total += len(chunk)
                if i % 10 == 0:
                    print(f"   處理批次 {i} (已處理 {total:,} 筆)...")
            
            # 轉換為排序後的數組（只保留前 top_n）
            item_counts = pd.Series(item_counts_dict).sort_values(ascending=False)
            item_counts_values = item_counts.head(top_n).values
            
            print(f"✅ 完成分析 {len(item_counts):,} 部電影，保留前 {top_n:,} 部用於長尾圖")
        else:
            if ratings is None:
                ratings = self.load_ratings_sample()
            
            if ratings is None or ratings.empty:
                return None
            
            # 計算每部電影的評分數量
            item_counts = ratings['movieId'].value_counts()
            item_counts_values = item_counts.values
        
        # 統一處理統計數據
        if use_full_dataset:
            item_counts_series = pd.Series(sorted(item_counts_dict.values(), reverse=True))
            cold_items_1 = sum(1 for c in item_counts_dict.values() if c == 1)
            cold_items_5 = sum(1 for c in item_counts_dict.values() if c <= 5)
            cold_items_10 = sum(1 for c in item_counts_dict.values() if c <= 10)
        else:
            item_counts_series = item_counts
            cold_items_1 = (item_counts == 1).sum()
            cold_items_5 = (item_counts <= 5).sum()
            cold_items_10 = (item_counts <= 10).sum()
        
        stats = {
            'total_items': len(item_counts_series),
            'mean_ratings_per_item': float(item_counts_series.mean()),
            'median_ratings_per_item': float(item_counts_series.median()),
            'std_ratings_per_item': float(item_counts_series.std()),
            'max_ratings': int(item_counts_series.max()),
            'min_ratings': int(item_counts_series.min()),
            'cold_start': {
                'items_with_1_rating': int(cold_items_1),
                'items_with_le_5_ratings': int(cold_items_5),
                'items_with_le_10_ratings': int(cold_items_10),
                'cold_start_ratio_5': float(cold_items_5 / len(item_counts_series)),
                'cold_start_ratio_10': float(cold_items_10 / len(item_counts_series))
            },
            'quantiles': {
                '25%': int(item_counts_series.quantile(0.25)),
                '50%': int(item_counts_series.quantile(0.50)),
                '75%': int(item_counts_series.quantile(0.75)),
                '90%': int(item_counts_series.quantile(0.90)),
                '95%': int(item_counts_series.quantile(0.95)),
                '99%': int(item_counts_series.quantile(0.99))
            }
        }
        
        # 增加只用於繪圖的字段（不包含在 JSON 輸出中）
        stats['_plot_data'] = item_counts_values  # 前缀 _ 表示內部使用
        
        return stats
    
    def analyze_sparsity(self, ratings: pd.DataFrame = None) -> Optional[Dict[str, Any]]:
        """分析矩陣稀疏度"""
        if ratings is None:
            ratings = self.load_ratings_sample()
        
        if ratings is None or ratings.empty:
            return None
        
        n_users = ratings['userId'].nunique()
        n_items = ratings['movieId'].nunique()
        n_ratings = len(ratings)
        
        # 計算稀疏度
        total_possible = n_users * n_items
        sparsity = 1 - (n_ratings / total_possible)
        density = n_ratings / total_possible
        
        stats = {
            'n_users': int(n_users),
            'n_items': int(n_items),
            'n_ratings': int(n_ratings),
            'total_possible_ratings': int(total_possible),
            'sparsity': float(sparsity),
            'density': float(density),
            'sparsity_percentage': float(sparsity * 100),
            'density_percentage': float(density * 100)
        }
        
        return stats
    
    def generate_full_analysis(self, sample_size: int = 100000) -> Dict[str, Any]:
        """生成完整的資料集分析報告"""
        print("=" * 80)
        print("📊 資料集統計分析")
        print("=" * 80)
        print()
        
        # 載入資料
        print(f"📁 載入資料樣本 (前 {sample_size:,} 筆)...")
        try:
            ratings = self.load_ratings_sample(limit=sample_size)
        except Exception as e:
            print(f"❌ 載入資料時發生錯誤: {e}")
            return {}
        
        if ratings is None or ratings.empty:
            print("❌ 無法載入資料")
            return {}
        
        print(f"✅ 成功載入 {len(ratings):,} 筆評分資料")
        print()
        
        # 執行各項分析
        analysis = {
            'sample_size': len(ratings),
            'rating_distribution': self.analyze_rating_distribution(ratings),
            'user_activity': self.analyze_user_activity(ratings),
            'item_popularity': self.analyze_item_popularity(ratings),
            'sparsity': self.analyze_sparsity(ratings)
        }
        
        return analysis


def print_dataset_analysis(analyzer: DatasetAnalyzer, sample_size: int = 100000):
    """打印資料集分析報告"""
    analysis = analyzer.generate_full_analysis(sample_size)
    
    if not analysis:
        print("❌ 分析失敗")
        return
    
    # 評分分布
    if analysis.get('rating_distribution'):
        stats = analysis['rating_distribution']
        print("📊 評分分布統計")
        print("-" * 80)
        print(f"平均評分: {stats['mean']:.3f}")
        print(f"中位數: {stats['median']:.3f}")
        print(f"標準差: {stats['std']:.3f}")
        print(f"範圍: {stats['min']:.1f} - {stats['max']:.1f}")
        print(f"\n各評分數量:")
        for rating, count in sorted(stats['distribution'].items()):
            percentage = count / stats['total_ratings'] * 100
            print(f"  {rating:.1f} 星: {count:,} ({percentage:.2f}%)")
        print()
    
    # 使用者活躍度
    if analysis.get('user_activity'):
        stats = analysis['user_activity']
        print("👥 使用者活躍度分析")
        print("-" * 80)
        print(f"總使用者數: {stats['total_users']:,}")
        print(f"平均每人評分數: {stats['mean_ratings_per_user']:.1f}")
        print(f"中位數: {stats['median_ratings_per_user']:.0f}")
        print(f"標準差: {stats['std']:.1f}")
        print(f"最多評分: {stats['max_ratings']:,}")
        print(f"最少評分: {stats['min_ratings']:,}")
        print(f"\n長尾效應:")
        print(f"  前 20% 活躍用戶貢獻 {stats['top_20_percent_contribution']*100:.1f}% 的評分")
        print(f"\n活躍度分位數:")
        for p, v in stats['percentiles'].items():
            print(f"  {p}: {v:.0f} 評分")
        print()
    
    # 電影流行度
    if analysis.get('item_popularity'):
        stats = analysis['item_popularity']
        print("🎬 電影流行度分析")
        print("-" * 80)
        print(f"總電影數: {stats['total_items']:,}")
        print(f"平均每部電影評分數: {stats['mean_ratings_per_item']:.1f}")
        print(f"中位數: {stats['median_ratings_per_item']:.0f}")
        print(f"標準差: {stats['std']:.1f}")
        print(f"最多評分: {stats['max_ratings']:,}")
        print(f"最少評分: {stats['min_ratings']:,}")
        print(f"\n冷啟動問題:")
        for threshold, data in stats['cold_items'].items():
            print(f"  評分 {threshold}: {data['count']:,} 部 ({data['percentage']:.2f}%)")
        print(f"\n長尾效應:")
        print(f"  前 20% 熱門電影貢獻 {stats['top_20_percent_contribution']*100:.1f}% 的評分")
        print(f"\n流行度分位數:")
        for p, v in stats['percentiles'].items():
            print(f"  {p}: {v:.0f} 評分")
        print()
    
    # 稀疏度
    if analysis.get('sparsity'):
        stats = analysis['sparsity']
        print("📈 矩陣稀疏度分析")
        print("-" * 80)
        print(f"使用者數: {stats['n_users']:,}")
        print(f"電影數: {stats['n_items']:,}")
        print(f"評分數: {stats['n_ratings']:,}")
        print(f"可能的評分總數: {stats['total_possible_ratings']:,}")
        print(f"稀疏度: {stats['sparsity_percentage']:.2f}%")
        print(f"密度: {stats['density_percentage']:.4f}%")
        print()
    
    print("=" * 80)
    print()


def print_progress_report(analyzer: ExperimentAnalyzer):
    """打印進度報告"""
    progress = analyzer.check_progress()
    
    print("=" * 80)
    print("📊 重構計畫執行進度")
    print("=" * 80)
    print()
    
    for stage_name, stage_data in progress['stages'].items():
        status = "✅" if stage_data['rate'] == 100 else "⏳"
        print(f"{status} {stage_name} 階段: {stage_data['count']}/{stage_data['total']} ({stage_data['rate']:.0f}%)")
        
        if stage_data['missing']:
            print(f"   缺少: {', '.join(stage_data['missing'][:5])}")
            if len(stage_data['missing']) > 5:
                print(f"   ... 還有 {len(stage_data['missing']) - 5} 個")
        print()
    
    print("=" * 80)
    total = progress['total_configs']
    completed = progress['total_completed']
    print(f"📈 總體進度: {completed}/{total} ({completed/total*100:.1f}%)")
    print("=" * 80)
    print()
    
    # 分析狀態
    print("🎯 分析狀態:")
    svd_status = "✅ 已完成" if progress['analysis_status']['svd'] else "⏳ 未完成"
    knn_status = "✅ 已完成" if progress['analysis_status']['knn'] else "⏳ 未完成"
    print(f"  SVD 分析: {svd_status}")
    print(f"  KNN 分析: {knn_status}")
    print()


def print_svd_analysis(analyzer: ExperimentAnalyzer):
    """打印 SVD 分析結果"""
    result = analyzer.analyze_svd()
    
    if not result:
        print("❌ 未找到 SVD 階段結果")
        return
    
    print("=" * 76)
    print("📊 SVD 階段結果分析")
    print("=" * 76)
    print()
    print(f"📊 找到 {result['count']} 個 SVD 配置結果")
    print()
    print("🏆 Top 10 SVD 配置（按 Hit Rate@10）：")
    print()
    print(f"{'排名':<4} {'配置':<12} {'維度':<6} {'Hit Rate':<10} {'NDCG':<10} {'RMSE':<10} {'時間(s)':<10}")
    print("-" * 76)
    
    for i, r in enumerate(result['results'][:10], 1):
        print(f"{i:<4} {r['config_name']:<12} {r['n_components']:<6} "
              f"{r['hit_rate']:<10.4f} {r['ndcg']:<10.4f} "
              f"{r['rmse']:<10.4f} {r['total_time']:<10.2f}")
    
    if result['best']:
        best = result['best']
        print()
        print(f"✨ 最佳 SVD 配置：{best['config_name']}")
        print(f"   維度：{best['n_components']}")
        print(f"   Hit Rate@10：{best['hit_rate']:.4f}")
        print(f"   NDCG@10：{best['ndcg']:.4f}")
        print(f"   RMSE：{best['rmse']:.4f}")
        print(f"   執行時間：{best['total_time']:.2f} 秒")
        print(f"   記憶體峰值：{best['memory_mb']:.2f} MB")
        print()
        print("💾 已保存最佳配置至: log/best_svd.json")
    
    print("=" * 76)
    print()


def print_knn_analysis(analyzer: ExperimentAnalyzer):
    """打印 KNN 分析結果"""
    result = analyzer.analyze_knn()
    
    if not result:
        print("❌ 未找到 KNN 階段結果")
        return
    
    print("=" * 72)
    print("📊 KNN 階段結果分析")
    print("=" * 72)
    print()
    print(f"📊 找到 {result['count']} 個 KNN 配置結果")
    print()
    print("🏆 KNN 配置排名（按 Hit Rate@10）：")
    print()
    print(f"{'排名':<4} {'配置':<12} {'K值':<6} {'Hit Rate':<10} {'NDCG':<10} {'RMSE':<10} {'時間(s)':<10}")
    print("-" * 72)
    
    for i, r in enumerate(result['results'], 1):
        print(f"{i:<4} {r['config_name']:<12} {r['k_neighbors']:<6} "
              f"{r['hit_rate']:<10.4f} {r['ndcg']:<10.4f} "
              f"{r['rmse']:<10.4f} {r['total_time']:<10.2f}")
    
    if result['best']:
        best = result['best']
        print()
        print(f"✨ 最佳 KNN 配置：{best['config_name']}")
        print(f"   K 值：{best['k_neighbors']}")
        print(f"   Hit Rate@10：{best['hit_rate']:.4f}")
        print(f"   NDCG@10：{best['ndcg']:.4f}")
        print(f"   RMSE：{best['rmse']:.4f}")
        print(f"   執行時間：{best['total_time']:.2f} 秒")
        print()
        print("💾 已保存最佳配置至: log/best_knn.json")
    
    print("=" * 72)
    print()


# 命令行接口
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'progress':
            analyzer = ExperimentAnalyzer()
            print_progress_report(analyzer)
        elif command == 'svd':
            analyzer = ExperimentAnalyzer()
            print_svd_analysis(analyzer)
        elif command == 'knn':
            analyzer = ExperimentAnalyzer()
            print_knn_analysis(analyzer)
        elif command == 'dataset':
            dataset_analyzer = DatasetAnalyzer()
            print_dataset_analysis(dataset_analyzer)
        elif command == 'all':
            # 實驗結果分析
            analyzer = ExperimentAnalyzer()
            print_progress_report(analyzer)
            print()
            print_svd_analysis(analyzer)
            print()
            print_knn_analysis(analyzer)
            print()
            # 資料集分析
            dataset_analyzer = DatasetAnalyzer()
            print_dataset_analysis(dataset_analyzer)
        else:
            print("用法: python -m movie_recommendation.analysis [progress|svd|knn|dataset|all]")
    else:
        # 默認顯示進度
        analyzer = ExperimentAnalyzer()
        print_progress_report(analyzer)
