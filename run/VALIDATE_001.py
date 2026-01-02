"""
驗證實驗: 大樣本驗證測試 (20,000 個用戶)
目的: 使用更大樣本驗證最佳配置的性能穩定性
配置: 20M資料, SVD=200維, KNN=40, 20,000 個用戶驗證
統計意義: 95% 置信區間誤差約 ±0.7%
執行時間: 約 3-5 分鐘
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from movie_recommendation.experiment import Experiment, ExperimentConfig


def main():
    # 使用最佳配置進行大樣本驗證
    config = ExperimentConfig(
        data_limit=None,              # 使用完整 20M 資料集
        min_item_ratings=0,           # 不過濾長尾電影
        use_timestamp=False,          # 不使用時間戳
        use_item_bias=False,          # 不使用 Item Bias
        use_svd=True,                 # 使用 SVD 降維
        n_components=200,             # 最佳 SVD 維度
        k_neighbors=40,               # 最佳 KNN 鄰居數
        n_samples=20000,              # ⭐ 關鍵: 使用 20,000 個用戶 (14.4%)
        top_n=10,                     # Top-10 推薦
        random_state=42               # 固定隨機種子
    )
    
    experiment = Experiment(config)
    results = experiment.run()
    
    print(f"\n✅ 驗證實驗完成: 大樣本驗證測試")


if __name__ == "__main__":
    main()
