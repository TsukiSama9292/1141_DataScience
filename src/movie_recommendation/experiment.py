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
from .hybrid_engine import GenomeHybridModel

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
        cold_start_threshold: int = 50
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
        self.use_genome_hybrid = use_genome_hybrid
        self.genome_alpha = genome_alpha
        self.cold_start_threshold = cold_start_threshold


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

        hybrid_engine = None
        if getattr(self.config, 'use_genome_hybrid', False):
            self.logger.info("🧬 [Hybrid] 正在啟動 Genome 冷啟動優化模組...")
            
            # 直接呼叫剛剛寫好的方法
            genome_path = self.data_loader.get_genome_path()
            
            if genome_path:
                try:
                    hybrid_engine = GenomeHybridModel(
                        genome_scores_path=genome_path,
                        movie_map=movie_map
                    )
                    self.logger.info("   ✅ Genome 引擎初始化成功")
                except Exception as e:
                    self.logger.error(f"   ❌ 初始化失敗: {e}")
                    hybrid_engine = None
            else:
                self.logger.warning("   ⚠️ 找不到基因檔案，將降級為純 KNN 模式。")
        
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
        
        metrics = self.evaluator.evaluate_leave_one_out(
            self.recommender,
            self.train_matrix,
            self.user_features,
            test_users,
            top_n=self.config.top_n,
            item_means=self.item_means,
            global_mean=self.global_mean,
            hybrid_engine=hybrid_engine,
            hybrid_config={
                'alpha': getattr(self.config, 'genome_alpha', 0.2),
                'threshold': getattr(self.config, 'cold_start_threshold', 50)
            }
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
            'random_state': self.config.random_state,
            'use_genome_hybrid': getattr(self.config, 'use_genome_hybrid', False),
            'genome_alpha': getattr(self.config, 'genome_alpha', 0.2)
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
