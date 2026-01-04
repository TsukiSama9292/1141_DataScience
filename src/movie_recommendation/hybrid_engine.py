import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
import logging

logger = logging.getLogger(__name__)

class GenomeHybridModel:
    def __init__(self, genome_scores_path, movie_map=None):
        """
        Args:
            genome_scores_path: 基因資料路徑 (由 DataLoader 提供)
            movie_map: {Internal_Index: Real_Movie_ID} 的對照表
        """
        logger.info(f"🧬 初始化 Genome Hybrid 模型 (來源: {genome_scores_path})...")
        self.movie_map = movie_map
        
        # 1. 讀取資料 (優化：指定型別以節省記憶體)
        # tagId 和 relevance 是數值，不需要 object 型態
        try:
            df = pd.read_csv(
                genome_scores_path, 
                dtype={'movieId': 'int32', 'tagId': 'int32', 'relevance': 'float32'}
            )
        except Exception as e:
            logger.error(f"讀取 Genome CSV 失敗，請確認檔案格式: {e}")
            raise e
        
        # 2. 建立 Movie-Tag 矩陣
        # index=movieId (Real ID), columns=tagId
        self.tag_matrix_df = df.pivot(index='movieId', columns='tagId', values='relevance').fillna(0)
        
        # 3. 轉成 numpy 並正規化
        self.movie_ids = self.tag_matrix_df.index.values
        self.tag_matrix = normalize(self.tag_matrix_df.values, axis=1)
        
        # 4. 建立快速查詢索引 (RealID -> Matrix Row Index)
        # 用來查某個真實 ID 的電影在矩陣的第幾列
        self.real_id_to_matrix_idx = {mid: i for i, mid in enumerate(self.movie_ids)}
        
        # 5. 建立 Internal Index -> Matrix Row Index 的映射
        # 【修正點】DataLoader 的 movie_map 是 {Internal: Real}
        self.internal_to_matrix_idx = {}
        if movie_map:
            for internal_idx, real_id in movie_map.items():
                if real_id in self.real_id_to_matrix_idx:
                    # 只有當 Genome 資料集裡也有這部電影時，才建立映射
                    self.internal_to_matrix_idx[internal_idx] = self.real_id_to_matrix_idx[real_id]

        logger.info(f"🧬 Genome 模型就緒。涵蓋電影數: {len(self.internal_to_matrix_idx)} / {len(movie_map) if movie_map else 0}")

    def get_user_profile(self, history_internal_indices):
        """
        根據使用者看過的電影 (Internal Indices)，合成一個「使用者基因向量」
        """
        valid_vectors = []
        for idx in history_internal_indices:
            # 檢查這個 Internal Index 是否有對應的基因資料
            if idx in self.internal_to_matrix_idx:
                matrix_idx = self.internal_to_matrix_idx[idx]
                valid_vectors.append(self.tag_matrix[matrix_idx])
        
        if not valid_vectors:
            return None
            
        # 計算平均向量
        user_vector = np.mean(valid_vectors, axis=0)
        return user_vector

    def calculate_batch_scores(self, candidate_internal_indices, user_profile):
        """
        一次計算所有候選電影的相似度
        """
        if user_profile is None:
            return np.zeros(len(candidate_internal_indices))
            
        # 找出候選電影對應的矩陣列索引
        matrix_indices = []
        valid_mask = [] 
        
        for idx in candidate_internal_indices:
            if idx in self.internal_to_matrix_idx:
                matrix_indices.append(self.internal_to_matrix_idx[idx])
                valid_mask.append(True)
            else:
                valid_mask.append(False)
        
        if not matrix_indices:
            return np.zeros(len(candidate_internal_indices))

        # 取出候選向量群
        candidate_vectors = self.tag_matrix[matrix_indices]
        
        # 矩陣乘法
        scores = np.dot(candidate_vectors, user_profile)
        
        # 填回完整長度
        final_scores = np.zeros(len(candidate_internal_indices))
        final_scores[np.array(valid_mask)] = scores
        
        return final_scores