"""
Data loading and preprocessing module.
"""

import pandas as pd
import numpy as np
import kagglehub
from typing import Tuple, Optional
import logging
import os

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and preprocess MovieLens dataset."""
    
    def __init__(self, dataset_name: str = "grouplens/movielens-20m-dataset"):
        """
        Initialize DataLoader.
        
        Args:
            dataset_name: Name of the dataset in kagglehub
        """
        self.dataset_name = dataset_name
        self.path = None
        self.movies = None
        self.ratings = None
        
    def load_data(self, limit: Optional[int] = None, min_item_ratings: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load MovieLens dataset.
        
        Args:
            limit: Optional limit on number of ratings to load
            min_item_ratings: Minimum number of ratings per item (0 = no filtering)
            
        Returns:
            Tuple of (movies, ratings) DataFrames
        """
        logger.info(f"載入資料集: {self.dataset_name}")
        self.path = kagglehub.dataset_download(self.dataset_name)
        
        self.movies = pd.read_csv(f"{self.path}/movie.csv")[["movieId", "title"]]
        
        if limit:
            ratings_full = pd.read_csv(f"{self.path}/rating.csv")
            self.ratings = ratings_full[["userId", "movieId", "rating"]].iloc[:limit]
        else:
            self.ratings = pd.read_csv(f"{self.path}/rating.csv")[["userId", "movieId", "rating"]]
        
        # 過濾長尾電影
        if min_item_ratings > 0:
            self.ratings = self.filter_cold_items(self.ratings, min_item_ratings)
        
        logger.info(f"載入完成: {len(self.movies)} 部電影, {len(self.ratings)} 筆評分")
        return self.movies, self.ratings
    
    def load_with_timestamp(self, limit: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load MovieLens dataset with timestamp for time-based features.
        
        Args:
            limit: Optional limit on number of ratings to load
            
        Returns:
            Tuple of (movies, ratings) DataFrames with timestamp
        """
        logger.info(f"載入資料集 (含時間戳記): {self.dataset_name}")
        self.path = kagglehub.dataset_download(self.dataset_name)
        
        self.movies = pd.read_csv(f"{self.path}/movie.csv")[["movieId", "title"]]
        
        if limit:
            ratings_full = pd.read_csv(
                f"{self.path}/rating.csv",
                usecols=["userId", "movieId", "rating", "timestamp"],
                dtype={'userId': int, 'movieId': int, 'rating': float, 'timestamp': object}
            )
            self.ratings = ratings_full.iloc[:limit]
        else:
            self.ratings = pd.read_csv(
                f"{self.path}/rating.csv",
                usecols=["userId", "movieId", "rating", "timestamp"],
                dtype={'userId': int, 'movieId': int, 'rating': float, 'timestamp': object}
            )
        
        # Parse timestamp
        self.ratings['timestamp'] = pd.to_datetime(self.ratings['timestamp'], errors='coerce')
        self.ratings = self.ratings.dropna(subset=['timestamp'])
        
        logger.info(f"載入完成: {len(self.movies)} 部電影, {len(self.ratings)} 筆有效評分")
        return self.movies, self.ratings
    
    def create_user_item_mapping(self, ratings: pd.DataFrame) -> Tuple[pd.DataFrame, dict, dict]:
        """
        Create user and item index mappings.
        
        Args:
            ratings: Ratings DataFrame
            
        Returns:
            Tuple of (ratings with indices, user_map, movie_map)
        """
        ratings = ratings.copy()
        ratings['user_idx'] = ratings['userId'].astype('category').cat.codes
        ratings['movie_idx'] = ratings['movieId'].astype('category').cat.codes
        
        user_map = dict(enumerate(ratings['userId'].astype('category').cat.categories))
        movie_map = dict(enumerate(ratings['movieId'].astype('category').cat.categories))
        
        n_users = len(user_map)
        n_items = len(movie_map)
        
        logger.info(f"建立映射: {n_users} 位使用者, {n_items} 部電影")
        return ratings, user_map, movie_map
    
    def calculate_item_bias(self, ratings: pd.DataFrame) -> Tuple[dict, float]:
        """
        Calculate item mean ratings (item bias).
        
        Args:
            ratings: Ratings DataFrame with movie_idx
            
        Returns:
            Tuple of (item_means dict, global_mean)
        """
        item_means = ratings.groupby('movie_idx')['rating'].mean().to_dict()
        global_mean = ratings['rating'].mean()
        
        logger.info(f"計算 Item Bias: 全局平均 = {global_mean:.4f}")
        return item_means, global_mean
    
    def filter_cold_items(
        self,
        ratings: pd.DataFrame,
        min_item_ratings: int = 10
    ) -> pd.DataFrame:
        """
        過濾評分數不足的電影（長尾電影）
        
        Args:
            ratings: 評分資料
            min_item_ratings: 電影最少評分數閾值
            
        Returns:
            過濾後的評分資料
        """
        if min_item_ratings <= 0:
            return ratings
        
        item_counts = ratings.groupby('movieId').size()
        valid_items = item_counts[item_counts >= min_item_ratings].index
        filtered_ratings = ratings[ratings['movieId'].isin(valid_items)]
        
        removed_items = len(item_counts) - len(valid_items)
        removed_ratings = len(ratings) - len(filtered_ratings)
        
        logger.info(
            f"電影過濾: 移除 {removed_items} 部電影 "
            f"({removed_items/len(item_counts)*100:.2f}%), "
            f"{removed_ratings:,} 筆評分 "
            f"({removed_ratings/len(ratings)*100:.2f}%)"
        )
        
        return filtered_ratings

    def get_genome_path(self) -> Optional[str]:
        """
        嘗試尋找並回傳 genome-scores.csv 的絕對路徑
        支援遞迴搜尋、不同命名格式 (底線/連字號) 與大小寫忽略
        """
        # 1. 確保資料已下載
        if self.path is None:
            logger.info("檢查基因資料中，正在確認資料集下載狀態...")
            self.path = kagglehub.dataset_download(self.dataset_name)
            
        logger.info(f"正在搜尋基因檔案，搜尋根目錄: {self.path}")
        
        # 2. 定義目標檔名 (轉成小寫以方便比對)
        # Kaggle 資料集有時候會改檔名，例如把連字號改成底線
        target_names = {'genome-scores.csv', 'genome_scores.csv', 'genome_scores.csv.zip'}
        
        # 3. 使用 os.walk 進行地毯式搜索
        for root, dirs, files in os.walk(self.path):
            for filename in files:
                # 轉小寫比對，增加容錯率
                if filename.lower() in target_names:
                    full_path = os.path.join(root, filename)
                    logger.info(f"✅ 成功定位基因檔案: {full_path}")
                    return full_path
        
        # 4. 如果跑到這裡代表真的沒找到，印出目錄結構幫助除錯
        logger.error("❌ 找不到目標檔案。以下是搜尋過的目錄結構摘要:")
        try:
            # 只印出前 3 層目錄結構，避免 Log 爆炸
            level_limit = 3
            root_depth = self.path.count(os.sep)
            
            for root, dirs, files in os.walk(self.path):
                current_depth = root.count(os.sep)
                if current_depth - root_depth < level_limit:
                    indent = "  " * (current_depth - root_depth)
                    logger.error(f"{indent}📂 {os.path.basename(root)}/")
                    for f in files[:5]: # 每個目錄只列出前 5 個檔案
                        logger.error(f"{indent}  📄 {f}")
                    if len(files) > 5:
                        logger.error(f"{indent}  ... (還有 {len(files)-5} 個檔案)")
        except Exception as e:
            logger.error(f"無法列印目錄結構: {e}")
            
        return None

