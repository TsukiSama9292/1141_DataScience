"""
Feature engineering module for matrix construction and dimensionality reduction.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from typing import Optional, Tuple
import logging
import gc

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Build feature matrices and apply dimensionality reduction."""
    
    def __init__(self):
        """Initialize FeatureEngineer."""
        self.svd = None
        self.user_features = None
        
    def build_sparse_matrix(
        self,
        ratings: pd.DataFrame,
        n_users: int,
        n_items: int,
        user_col: str = 'user_idx',
        item_col: str = 'movie_idx',
        rating_col: str = 'rating'
    ) -> csr_matrix:
        """
        Build user-item sparse matrix.
        
        Args:
            ratings: Ratings DataFrame with user and item indices
            n_users: Number of users
            n_items: Number of items
            user_col: Name of user index column
            item_col: Name of item index column
            rating_col: Name of rating column
            
        Returns:
            CSR sparse matrix
        """
        logger.info(f"建立稀疏矩陣: {n_users} x {n_items}")
        
        row = ratings[user_col].values
        col = ratings[item_col].values
        data = ratings[rating_col].values
        
        sparse_matrix = csr_matrix((data, (row, col)), shape=(n_users, n_items))
        
        # Clean up
        del row, col, data
        gc.collect()
        
        logger.info(f"稀疏矩陣形狀: {sparse_matrix.shape}, 非零元素: {sparse_matrix.nnz}")
        return sparse_matrix
    
    def apply_svd(
        self,
        matrix: csr_matrix,
        n_components: int = 50,
        random_state: int = 42
    ) -> np.ndarray:
        """
        Apply TruncatedSVD for dimensionality reduction.
        
        Args:
            matrix: Input sparse matrix
            n_components: Number of components to keep
            random_state: Random state for reproducibility
            
        Returns:
            Reduced feature matrix
        """
        logger.info(f"執行 SVD 降維: {n_components} 維")
        
        self.svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        self.user_features = self.svd.fit_transform(matrix)
        
        explained_variance = self.svd.explained_variance_ratio_.sum()
        logger.info(f"SVD 完成: 解釋方差 = {explained_variance:.4f}")
        
        return self.user_features
    
    def apply_time_decay(
        self,
        ratings: pd.DataFrame,
        half_life_days: int = 500
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply time decay weighting to ratings.
        
        Args:
            ratings: Ratings DataFrame with timestamp column
            half_life_days: Half-life period in days
            
        Returns:
            Tuple of (decayed_ratings, original_ratings)
        """
        logger.info(f"應用時間衰減: 半衰期 = {half_life_days} 天")
        
        max_timestamp = ratings['timestamp'].max()
        time_diff_seconds = (max_timestamp - ratings['timestamp']).dt.total_seconds()
        
        half_life_seconds = half_life_days * 24 * 60 * 60
        lambda_decay = np.log(2) / half_life_seconds
        
        decay_factor = np.exp(-lambda_decay * time_diff_seconds)
        decayed_ratings = ratings['rating'].values * decay_factor.values
        original_ratings = ratings['rating'].values
        
        logger.info("時間衰減計算完成")
        return decayed_ratings, original_ratings
    
    def apply_tfidf_weighting(
        self,
        ratings: pd.DataFrame,
        n_users: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply TF-IDF style weighting to ratings.
        
        Args:
            ratings: Ratings DataFrame with movie_idx
            n_users: Total number of users
            
        Returns:
            Tuple of (tfidf_weighted_ratings, original_ratings)
        """
        logger.info("應用 TF-IDF 權重")
        
        # Calculate IDF
        movie_counts = ratings.groupby('movie_idx')['userId'].count()
        idf_weights = np.log(n_users / (movie_counts + 1))
        idf_dict = idf_weights.to_dict()
        
        # Apply IDF to ratings
        ratings['idf'] = ratings['movie_idx'].map(idf_dict)
        tfidf_ratings = ratings['rating'].values * ratings['idf'].values
        original_ratings = ratings['rating'].values
        
        logger.info("TF-IDF 權重計算完成")
        return tfidf_ratings, original_ratings
