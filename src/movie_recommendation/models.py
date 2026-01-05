"""
Recommendation models module.
"""

import numpy as np
from scipy.sparse import csr_matrix, issparse
from sklearn.neighbors import NearestNeighbors
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class KNNRecommender:
    """K-Nearest Neighbors based collaborative filtering recommender."""
    
    def __init__(
        self,
        n_neighbors: int = 20,
        metric: str = 'cosine',
        algorithm: str = 'brute',
        n_jobs: int = -1
    ):
        """
        Initialize KNN recommender.
        
        Args:
            n_neighbors: Number of neighbors to use
            metric: Distance metric ('cosine', 'euclidean', etc.)
            algorithm: Algorithm to use ('brute', 'ball_tree', 'kd_tree')
            n_jobs: Number of parallel jobs (-1 for all CPUs)
        """
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.algorithm = algorithm
        self.n_jobs = n_jobs
        self.knn = None
        self.trained = False
        
    def fit(self, features: np.ndarray):
        """
        Initialize KNN model (Lazy Learning - no training).
        
        Args:
            features: User feature matrix (n_users x n_features)
        """
        logger.info(f"初始化 KNN 索引: k={self.n_neighbors}, metric={self.metric} (Lazy Learning - 無訓練階段)")
        
        self.knn = NearestNeighbors(
            metric=self.metric,
            algorithm=self.algorithm,
            n_jobs=self.n_jobs
        )
        self.knn.fit(features)
        self.trained = True

        # Precompute neighbors for all users to avoid repeated kneighbors() calls
        # But skip full precompute when the user count is very large to avoid OOM.
        n_users = features.shape[0]
        max_precompute_users = 20000
        if n_users <= max_precompute_users:
            try:
                # request one extra neighbor to exclude self (k+1)
                distances, indices = self.knn.kneighbors(features, n_neighbors=self.n_neighbors + 1)
                # store arrays excluding the query itself (first column)
                self._neighbor_indices_all = indices[:, 1:]
                self._neighbor_similarities_all = 1 - distances[:, 1:]
            except Exception:
                # if global kneighbors fails (e.g., memory), fallback to on-demand
                self._neighbor_indices_all = None
                self._neighbor_similarities_all = None
        else:
            # skip precompute for large user sets; use on-demand or batched kneighbors in evaluation
            self._neighbor_indices_all = None
            self._neighbor_similarities_all = None

        logger.info("KNN 模型訓練完成")
    
    def find_neighbors(
        self,
        user_idx: int,
        user_features: np.ndarray,
        k: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k nearest neighbors for a user.
        
        Args:
            user_idx: Index of the query user
            user_features: User feature matrix
            k: Number of neighbors (uses self.n_neighbors if None)
            
        Returns:
            Tuple of (neighbor_indices, neighbor_similarities)
        """
        if not self.trained:
            raise ValueError("Model must be fitted before finding neighbors")
        
        k = k or self.n_neighbors

        # If we have a cached neighbor matrix (computed in fit), use it
        if getattr(self, '_neighbor_indices_all', None) is not None:
            inds = self._neighbor_indices_all[user_idx][:k]
            sims = self._neighbor_similarities_all[user_idx][:k]
            return inds, sims

        # Fallback: compute on-demand for this user
        # Support both dense ndarray and sparse matrix for user_features
        if issparse(user_features):
            query_vec = user_features[user_idx]
        else:
            query_vec = user_features[user_idx].reshape(1, -1)

        distances, indices = self.knn.kneighbors(query_vec, n_neighbors=k+1)

        neighbor_indices = indices.flatten()[1:]
        neighbor_similarities = 1 - distances.flatten()[1:]

        return neighbor_indices, neighbor_similarities
    
    def predict_rating(
        self,
        user_idx: int,
        movie_idx: int,
        train_matrix: csr_matrix,
        user_features: np.ndarray,
        item_means: Optional[Dict[int, float]] = None,
        global_mean: float = 3.5,
        amplification_factor: float = 1.0
    ) -> float:
        """
        Predict rating for a user-item pair.
        
        Args:
            user_idx: User index
            movie_idx: Movie index
            train_matrix: Sparse rating matrix
            user_features: User feature matrix
            item_means: Dict of movie means for fallback
            global_mean: Global mean rating for fallback
            amplification_factor: Similarity amplification factor (>1 amplifies)
            
        Returns:
            Predicted rating
        """
        neighbor_indices, neighbor_similarities = self.find_neighbors(user_idx, user_features)
        
        # Apply amplification if specified
        if amplification_factor != 1.0:
            neighbor_similarities = np.power(neighbor_similarities, amplification_factor)
        
        neighbor_ratings = []
        neighbor_weights = []
        
        for n_idx, sim in zip(neighbor_indices, neighbor_similarities):
            n_row = train_matrix[n_idx]
            if movie_idx in n_row.indices:
                idx_loc = np.where(n_row.indices == movie_idx)[0][0]
                r = n_row.data[idx_loc]
                neighbor_ratings.append(r)
                neighbor_weights.append(sim)
        
        if neighbor_weights:
            return np.average(neighbor_ratings, weights=neighbor_weights)
        elif item_means and movie_idx in item_means:
            return item_means[movie_idx]
        else:
            return global_mean
    
    def recommend_items(
        self,
        user_idx: int,
        train_matrix: csr_matrix,
        user_features: np.ndarray,
        top_n: int = 10,
        exclude_seen: bool = True,
        target_item: Optional[int] = None
    ) -> List[int]:
        """
        Generate top-N recommendations for a user.
        
        Args:
            user_idx: User index
            train_matrix: Sparse rating matrix
            user_features: User feature matrix
            top_n: Number of recommendations
            exclude_seen: Whether to exclude already seen items
            target_item: Optional target item to include in scoring
            
        Returns:
            List of recommended item indices
        """
        neighbor_indices, neighbor_similarities = self.find_neighbors(user_idx, user_features)
        
        user_row = train_matrix[user_idx]
        watched_items = set(user_row.indices)
        
        candidate_scores = {}
        
        for n_idx, sim in zip(neighbor_indices, neighbor_similarities):
            n_row = train_matrix[n_idx]
            for m_idx, r in zip(n_row.indices, n_row.data):
                if exclude_seen and m_idx in watched_items and m_idx != target_item:
                    continue
                
                if m_idx not in candidate_scores:
                    candidate_scores[m_idx] = 0
                candidate_scores[m_idx] += sim * r
        
        # Sort and return top N
        recommended = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        return [item_idx for item_idx, _ in recommended[:top_n]]
