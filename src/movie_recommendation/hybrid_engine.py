import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class GenomeHybridModel:
    def __init__(self, genome_scores_path='data/genome-scores.csv'):
        """
        初始化：讀取基因數據並建立向量矩陣
        """
        print(f"正在載入電影基因數據: {genome_scores_path} ...")
        self.genome_df = pd.read_csv(genome_scores_path)
        
        # 建立 Movie-Tag 矩陣 (列是電影, 欄是標籤特徵)
        # 這就是組員說的「向量化」，這裡我們做成了 Dense Matrix
        self.movie_tag_matrix = self.genome_df.pivot(
            index='movieId', 
            columns='tagId', 
            values='relevance'
        ).fillna(0)
        
        # 預先計算所有電影之間的相似度矩陣 (這是一個優化，避免重複計算)
        # 這對應組員說的「計算暫存」
        print("正在計算內容相似度矩陣 (Content-Based Similarity)...")
        # 為了節省記憶體，這裡只計算矩陣本身，如果資料量太大可改用其他方式
        # 但 MovieLens 20M 的 tag 矩陣通常記憶體還吃得消
        self.similarity_matrix = cosine_similarity(self.movie_tag_matrix)
        
        # 建立 movieId 到 矩陣索引 的對照表
        self.movie_id_to_index = {mid: i for i, mid in enumerate(self.movie_tag_matrix.index)}
        self.index_to_movie_id = {i: mid for i, mid in enumerate(self.movie_tag_matrix.index)}
        print("基因模型初始化完成！")

    def get_content_score(self, target_movie_id, user_history_movie_ids):
        """
        計算某部電影(target)與使用者看過的歷史電影(history)的平均相似度
        """
        if target_movie_id not in self.movie_id_to_index:
            return 0.0 # 冷啟動：如果是沒基因資料的新電影，回傳 0

        target_idx = self.movie_id_to_index[target_movie_id]
        
        sim_scores = []
        for hist_mid in user_history_movie_ids:
            if hist_mid in self.movie_id_to_index:
                hist_idx = self.movie_id_to_index[hist_mid]
                # 直接查表取得相似度 (速度快)
                score = self.similarity_matrix[target_idx][hist_idx]
                sim_scores.append(score)
        
        if not sim_scores:
            return 0.0
            
        # 回傳平均相似度 (也可以改用最大值 max 或加權平均)
        return np.mean(sim_scores)