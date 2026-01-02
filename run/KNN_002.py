"""
KNN_002: KNN 鄰居數 20（使用最佳 SVD）
目的: 基準 KNN 值測試
注意: SVD 維度需在 SVD 階段完成後更新為最佳值
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from movie_recommendation.experiment import Experiment, ExperimentConfig


def main():
    config = ExperimentConfig(
        data_limit=20_000_000,
        min_item_ratings=0,
        use_timestamp=False,
        use_item_bias=False,
        use_svd=True,
        n_components=128,  # TODO: 更新為最佳 SVD 維度
        k_neighbors=20,
        n_samples=500,
        top_n=10,
        random_state=42
    )
    
    experiment = Experiment(config)
    results = experiment.run()
    
    print(f"✅ 運行完成")
    return results


if __name__ == "__main__":
    main()
