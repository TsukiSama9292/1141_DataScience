"""
Evaluation metrics module.
"""

import numpy as np
from scipy.sparse import csr_matrix
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluate recommendation performance with various metrics."""
    
    def __init__(self):
        """Initialize Evaluator."""
        self.metrics = {}
        
    def precision_at_k(self, recommended: List[int], relevant: List[int], k: int) -> float:
        """
        Calculate Precision@K.
        
        Args:
            recommended: List of recommended item indices
            relevant: List of relevant item indices
            k: Cutoff position
            
        Returns:
            Precision@K score
        """
        recommended_k = recommended[:k]
        hits = len(set(recommended_k) & set(relevant))
        return hits / k if k > 0 else 0.0
    
    def recall_at_k(self, recommended: List[int], relevant: List[int], k: int) -> float:
        """
        Calculate Recall@K.
        
        Args:
            recommended: List of recommended item indices
            relevant: List of relevant item indices
            k: Cutoff position
            
        Returns:
            Recall@K score
        """
        recommended_k = recommended[:k]
        hits = len(set(recommended_k) & set(relevant))
        return hits / len(relevant) if relevant else 0.0
    
    def hit_rate_at_k(self, recommended: List[int], relevant: List[int], k: int) -> int:
        """
        Calculate Hit Rate@K (binary hit).
        
        Args:
            recommended: List of recommended item indices
            relevant: List of relevant item indices
            k: Cutoff position
            
        Returns:
            1 if hit, 0 otherwise
        """
        recommended_k = recommended[:k]
        return int(bool(set(recommended_k) & set(relevant)))
    
    def mrr(self, recommended: List[int], relevant: List[int]) -> float:
        """
        Calculate Mean Reciprocal Rank.
        
        Args:
            recommended: List of recommended item indices (ordered)
            relevant: List of relevant item indices
            
        Returns:
            MRR score
        """
        for i, item in enumerate(recommended):
            if item in relevant:
                return 1.0 / (i + 1)
        return 0.0
    
    def ndcg_at_k(self, recommended: List[int], relevant: List[int], k: int) -> float:
        """
        Calculate NDCG@K.
        
        Args:
            recommended: List of recommended item indices (ordered)
            relevant: List of relevant item indices
            k: Cutoff position
            
        Returns:
            NDCG@K score
        """
        recommended_k = recommended[:k]
        
        # For binary relevance (Leave-One-Out)
        dcg = 0.0
        for i, item in enumerate(recommended_k):
            if item in relevant:
                dcg += 1.0 / np.log2(i + 2)  # i+2 because i is 0-indexed
        
        # IDCG for single relevant item
        idcg = 1.0
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def mae(self, true_ratings: np.ndarray, predicted_ratings: np.ndarray) -> float:
        """
        Calculate Mean Absolute Error.
        
        Args:
            true_ratings: Array of true ratings
            predicted_ratings: Array of predicted ratings
            
        Returns:
            MAE score
        """
        return np.mean(np.abs(true_ratings - predicted_ratings))
    
    def rmse(self, true_ratings: np.ndarray, predicted_ratings: np.ndarray) -> float:
        """
        Calculate Root Mean Squared Error.
        
        Args:
            true_ratings: Array of true ratings
            predicted_ratings: Array of predicted ratings
            
        Returns:
            RMSE score
        """
        return np.sqrt(np.mean((true_ratings - predicted_ratings) ** 2))
    
    def evaluate_leave_one_out(
        self,
        recommender,
        train_matrix: csr_matrix,
        user_features: np.ndarray,
        test_users: np.ndarray,
        top_n: int = 10,
        item_means: Dict[int, float] = None,
        global_mean: float = 3.5
    ) -> Dict[str, float]:
        """
        Evaluate using Leave-One-Out strategy.
        
        Args:
            recommender: KNNRecommender instance
            train_matrix: Sparse rating matrix
            user_features: User feature matrix
            test_users: Array of user indices to test
            top_n: Number of recommendations
            item_means: Dict of movie means
            global_mean: Global mean rating
            
        Returns:
            Dictionary of metric scores
        """
        logger.info(f"開始評估 (樣本數: {len(test_users)})")
        
        hits = 0
        reciprocal_ranks = []
        ndcg_scores = []
        squared_errors = []
        absolute_errors = []
        
        valid_samples = 0
        
        for u_idx in test_users:
            user_row = train_matrix[u_idx]
            watched_items = user_row.indices
            watched_ratings = user_row.data
            
            if len(watched_items) < 5:
                continue
            
            valid_samples += 1
            
            # Leave-One-Out: target is highest rated item
            target_idx_in_row = np.argmax(watched_ratings)
            target_movie_idx = watched_items[target_idx_in_row]
            true_rating = watched_ratings[target_idx_in_row]
            
            # Predict rating
            pred_rating = recommender.predict_rating(
                u_idx, target_movie_idx, train_matrix, user_features,
                item_means, global_mean
            )
            
            squared_errors.append((true_rating - pred_rating) ** 2)
            absolute_errors.append(abs(true_rating - pred_rating))
            
            # Generate recommendations
            recommended = recommender.recommend_items(
                u_idx, train_matrix, user_features,
                top_n=top_n, target_item=target_movie_idx
            )
            
            # Calculate ranking metrics
            if target_movie_idx in recommended:
                hits += 1
                rank = recommended.index(target_movie_idx) + 1
                reciprocal_ranks.append(1.0 / rank)
                ndcg_scores.append(1.0 / np.log2(rank + 1))
            else:
                reciprocal_ranks.append(0.0)
                ndcg_scores.append(0.0)
        
        # Aggregate results
        hit_rate = hits / valid_samples if valid_samples > 0 else 0.0
        
        results = {
            'n_samples': valid_samples,
            'hit_rate': hit_rate,
            'precision': hit_rate / top_n,
            'recall': hit_rate,  # For Leave-One-Out with single target
            'mrr': np.mean(reciprocal_ranks),
            'ndcg': np.mean(ndcg_scores),
            'rmse': np.sqrt(np.mean(squared_errors)),
            'mae': np.mean(absolute_errors)
        }
        
        logger.info(f"評估完成: Hit Rate = {results['hit_rate']:.4f}, RMSE = {results['rmse']:.4f}")
        return results
