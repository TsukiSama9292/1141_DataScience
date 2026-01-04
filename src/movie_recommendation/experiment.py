"""
Experiment orchestration module.
"""

import time
import numpy as np
from typing import Optional, Dict
import logging
import inspect
from pathlib import Path
from .data_loader import DataLoader
from .feature_engineering import FeatureEngineer
from .models import KNNRecommender
from .evaluation import Evaluator
from .utils import TimeTracker, setup_logging, log_metrics
from movie_recommendation.hybrid_engine import GenomeHybridModel

logger = logging.getLogger(__name__)


class ExperimentConfig:
    """Configuration for a recommendation experiment."""
    
    def __init__(
        self,
        data_limit: Optional[int] = None,
        min_item_ratings: int = 0,
        use_timestamp: bool = False,
        use_item_bias: bool = False,
        use_svd: bool = False,
        n_components: int = 50,
        use_time_decay: bool = False,
        half_life_days: int = 500,
        use_tfidf: bool = False,
        k_neighbors: int = 20,
        amplification_factor: float = 1.0,
        n_samples: int = 500,
        top_n: int = 10,
        random_state: int = 42,

        use_genome_hybrid: bool = False,
        genome_alpha: float = 0.2,
        cold_start_threshold: int = 70
    ):
        """
        Initialize experiment configuration.
        
        Args:
            data_limit: Limit on number of ratings (None for full dataset)
            min_item_ratings: Minimum number of ratings per item (0 = no filtering)
            use_timestamp: Whether to load timestamp data
            use_item_bias: Whether to calculate item bias
            use_svd: Whether to apply SVD dimensionality reduction
            n_components: Number of SVD components
            use_time_decay: Whether to apply time decay weighting
            half_life_days: Half-life for time decay
            use_tfidf: Whether to apply TF-IDF weighting
            k_neighbors: Number of nearest neighbors
            amplification_factor: Similarity amplification factor
            n_samples: Number of samples for evaluation
            top_n: Number of recommendations to generate
            random_state: Random seed for reproducibility
        """
        self.data_limit = data_limit
        self.min_item_ratings = min_item_ratings
        self.use_timestamp = use_timestamp
        self.use_item_bias = use_item_bias
        self.use_svd = use_svd
        self.n_components = n_components
        self.use_time_decay = use_time_decay
        self.half_life_days = half_life_days
        self.use_tfidf = use_tfidf
        self.k_neighbors = k_neighbors
        self.amplification_factor = amplification_factor
        self.n_samples = n_samples
        self.top_n = top_n
        self.random_state = random_state


class Experiment:
    """Run a complete recommendation experiment."""
    
    def __init__(self, config: ExperimentConfig, config_name: Optional[str] = None):
        """
        Initialize experiment.
        
        Args:
            config: Experiment configuration
            config_name: Optional name for logging (auto-detected if None)
        """
        self.config = config
        
        # Auto-detect config name from calling script filename
        if config_name is None:
            frame = inspect.currentframe()
            if frame and frame.f_back and frame.f_back.f_back:
                caller_file = frame.f_back.f_back.f_code.co_filename
                config_name = Path(caller_file).stem
            else:
                config_name = "experiment"
        
        self.config_name = config_name
        self.logger = setup_logging(config_name, log_dir="log")
        self.tracker = TimeTracker()
        self.time_records = {}
        
        self.data_loader = DataLoader()
        self.feature_engineer = FeatureEngineer()
        self.recommender = KNNRecommender(
            n_neighbors=config.k_neighbors,
            metric='cosine',
            algorithm='brute',
            n_jobs=-1
        )
        self.evaluator = Evaluator()
        
        self.train_matrix = None
        self.user_features = None
        self.item_means = None
        self.global_mean = 3.5
        
    def _log_time(self, stage_name: str, start_time: float):
        """Log stage execution time."""
        elapsed = time.time() - start_time
        self.time_records[stage_name] = elapsed
        self.tracker.log_stage(stage_name, elapsed)
    
    def run(self) -> Dict[str, float]:
        """
        Run complete experiment pipeline.
        
        Returns:
            Dictionary of evaluation metrics
        """
        self.logger.info(f"開始配置: {self.config_name}")
        
        self.tracker.start()
        
        # Load data
        stage_start = time.time()
        if self.config.use_timestamp:
            movies, ratings = self.data_loader.load_with_timestamp(self.config.data_limit)
        else:
            movies, ratings = self.data_loader.load_data(self.config.data_limit, self.config.min_item_ratings)
        self.tracker.sample_memory()
        self._log_time("載入資料", stage_start)
        
        # Create mappings
        stage_start = time.time()
        ratings, user_map, movie_map = self.data_loader.create_user_item_mapping(ratings)
        n_users = len(user_map)
        n_items = len(movie_map)
        self.tracker.sample_memory()
        self._log_time("建立映射", stage_start)
        
        # Calculate item bias if needed
        if self.config.use_item_bias:
            stage_start = time.time()
            self.item_means, self.global_mean = self.data_loader.calculate_item_bias(ratings)
            self.tracker.sample_memory()
            self._log_time("計算 Item Bias", stage_start)
        
        # Build sparse matrix
        stage_start = time.time()
        
        if self.config.use_time_decay:
            decayed_ratings, original_ratings = self.feature_engineer.apply_time_decay(
                ratings, self.config.half_life_days
            )
            # Build two matrices
            ratings_decay = ratings.copy()
            ratings_decay['rating'] = decayed_ratings
            train_matrix_decay = self.feature_engineer.build_sparse_matrix(
                ratings_decay, n_users, n_items
            )
            ratings_orig = ratings.copy()
            ratings_orig['rating'] = original_ratings
            self.train_matrix = self.feature_engineer.build_sparse_matrix(
                ratings_orig, n_users, n_items
            )
            # Use decayed for KNN, original for scoring
            matrix_for_knn = train_matrix_decay
            
        elif self.config.use_tfidf:
            tfidf_ratings, original_ratings = self.feature_engineer.apply_tfidf_weighting(
                ratings, n_users
            )
            # Build two matrices
            ratings_tfidf = ratings.copy()
            ratings_tfidf['rating'] = tfidf_ratings
            train_matrix_tfidf = self.feature_engineer.build_sparse_matrix(
                ratings_tfidf, n_users, n_items
            )
            ratings_orig = ratings.copy()
            ratings_orig['rating'] = original_ratings
            self.train_matrix = self.feature_engineer.build_sparse_matrix(
                ratings_orig, n_users, n_items
            )
            # Use TF-IDF for KNN, original for scoring
            matrix_for_knn = train_matrix_tfidf
            
        else:
            self.train_matrix = self.feature_engineer.build_sparse_matrix(
                ratings, n_users, n_items
            )
            matrix_for_knn = self.train_matrix
        
        self.tracker.sample_memory()
        self._log_time("建立稀疏矩陣", stage_start)
        
        # Apply SVD if needed
        if self.config.use_svd:
            stage_start = time.time()
            self.user_features = self.feature_engineer.apply_svd(
                matrix_for_knn, self.config.n_components, self.config.random_state
            )
            self.tracker.sample_memory()
            self._log_time("SVD 降維", stage_start)
        else:
            # Keep user features as sparse matrix to avoid large dense allocations
            # Many sklearn routines accept CSR sparse matrices for distance computations.
            self.user_features = matrix_for_knn
        
        # Train KNN
        stage_start = time.time()
        self.recommender.fit(self.user_features)
        self.tracker.sample_memory()
        self._log_time("訓練 KNN", stage_start)
        
        # Evaluation
        stage_start = time.time()
        np.random.seed(self.config.random_state)
        
        # 動態調整樣本數，確保不超過可用用戶數
        actual_samples = min(self.config.n_samples, n_users)
        if actual_samples < self.config.n_samples:
            logger.warning(f"⚠️  可用用戶數 ({n_users}) 少於配置樣本數 ({self.config.n_samples})，調整為 {actual_samples}")
        
        test_users = np.random.choice(n_users, size=actual_samples, replace=False)
        
        # Override predict_rating if amplification is used
        if self.config.amplification_factor != 1.0:
            original_predict = self.recommender.predict_rating
            def amplified_predict(user_idx, movie_idx, train_matrix, user_features,
                                item_means=None, global_mean=3.5, amplification_factor=1.0):
                return original_predict(
                    user_idx, movie_idx, train_matrix, user_features,
                    item_means, global_mean, self.config.amplification_factor
                )
            self.recommender.predict_rating = amplified_predict
        
        if getattr(self.config, 'use_genome_hybrid', False):
            self.logger.info("🧬 [Hybrid] 正在啟動 Genome 冷啟動優化模組...")
            
            # 1. 初始化你的引擎
            # 確保路徑正確，根據你的專案結構調整
            hybrid_engine = GenomeHybridModel(genome_scores_path='data/raw/genome-scores.csv')
            
            # 2. 準備 ID 轉換表 (非常重要！因為 Recommender 用的是 0,1,2 索引，但 CSV 用的是真實 ID)
            # user_map: {Real_ID: Index} -> 轉成 -> {Index: Real_ID}
            idx_to_user_id = {v: k for k, v in user_map.items()}
            idx_to_movie_id = {v: k for k, v in movie_map.items()}
            
            # 3. 建立使用者觀影歷史 (用於計算相似度)
            # ratings 是 DataFrame，包含 [userId, movieId, rating]
            self.logger.info("   正在構建歷史觀影快取...")
            user_history_map = ratings.groupby('userId')['movieId'].apply(list).to_dict()
            
            # 4. 讀取超參數
            alpha_base = getattr(self.config, 'genome_alpha', 0.2)
            cold_thresh = getattr(self.config, 'cold_start_threshold', 5)
            
            # 5. 【核心魔法】偷換概念：攔截並覆寫 predict_rating 方法
            # 先把原本的預測函數存起來 (可能是 KNN 預測，也可能是已經被 amplification 改過的預測)
            base_predict_func = self.recommender.predict_rating
            
            def hybrid_predict_wrapper(user_idx, movie_idx, train_matrix, user_features,
                                     item_means=None, global_mean=3.5, *args, **kwargs):
                
                # A. 取得原本分數 (SVD/KNN)
                try:
                    base_score = base_predict_func(
                        user_idx, movie_idx, train_matrix, user_features,
                        item_means, global_mean, *args, **kwargs
                    )
                except:
                    base_score = global_mean

                # B. 轉換 Index -> Real ID
                real_uid = idx_to_user_id.get(user_idx)
                real_mid = idx_to_movie_id.get(movie_idx)
                
                # C. 計算 Genome 內容分數 (你的貢獻)
                # 如果找不到 ID (新電影)，回傳 0
                if real_uid is None or real_mid is None:
                    return base_score

                history = user_history_map.get(real_uid, [])
                content_sim = hybrid_engine.get_content_score(real_mid, history)
                
                # 將相似度 (0~1) 映射到評分 (3.0 ~ 5.0)
                # 這裡假設相似度高代表會給高分
                content_score = 3.0 + (content_sim * 2.0)
                
                # D. 動態權重 (冷啟動邏輯)
                if len(history) < cold_thresh:
                    current_alpha = getattr(self.config, 'genome_alpha')
                else:
                    current_alpha = alpha_base # 老手：協同為主
                
                # E. 融合
                final_score = (1 - current_alpha) * base_score + current_alpha * content_score
                
                # F. 確保分數合理
                return max(0.5, min(5.0, final_score))

            # 6. 把這個混合函數，「掛載」回去給 recommender
            self.recommender.predict_rating = hybrid_predict_wrapper
            self.logger.info("   ✅ 已成功注入混合預測邏輯 (Hybrid Logic Injected)")

        metrics = self.evaluator.evaluate_leave_one_out(
            self.recommender,
            self.train_matrix,
            self.user_features,
            test_users,
            top_n=self.config.top_n,
            item_means=self.item_means,
            global_mean=self.global_mean
        )
        self.tracker.sample_memory()
        self._log_time("評估", stage_start)
        
        # Log results
        peak_memory = self.tracker.get_peak_mb()
        
        # 準備配置字典
        config_dict = {
            'data_limit': self.config.data_limit,
            'min_item_ratings': self.config.min_item_ratings,
            'use_timestamp': self.config.use_timestamp,
            'use_item_bias': self.config.use_item_bias,
            'use_svd': self.config.use_svd,
            'n_components': self.config.n_components,
            'use_time_decay': self.config.use_time_decay,
            'half_life_days': self.config.half_life_days,
            'use_tfidf': self.config.use_tfidf,
            'k_neighbors': self.config.k_neighbors,
            'amplification_factor': self.config.amplification_factor,
            'top_n': self.config.top_n,
            'random_state': self.config.random_state
        }
        
        log_metrics(
            self.logger,
            metrics,
            metrics['n_samples'],
            self.time_records,
            peak_memory,
            config=config_dict
        )
        
        self.logger.info(f"配置完成: {self.config_name}")
        
        return metrics
