"""
實驗20: 20M資料, SVD=128維, KNN=50, Item Bias + 相似度放大
目的: 測試相似度放大策略
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from movie_recommendation.experiment import Experiment, ExperimentConfig


def main():
    config = ExperimentConfig(
        data_limit=None,
        use_timestamp=False,
        use_item_bias=True,
        use_svd=True,
        n_components=128,
        amplification_factor=2.5,
        k_neighbors=50,
        n_samples=500,
        top_n=10,
        random_state=42
    )
    
    experiment = Experiment(config)
    results = experiment.run()
    
    print(f"✅ 運行完成")


if __name__ == "__main__":
    main()
