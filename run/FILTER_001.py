"""
FILTER_001: 基準線 - 不過濾任何電影
目的: 建立基準線，測試完整資料集的表現
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from movie_recommendation.experiment import Experiment, ExperimentConfig


def main():
    config = ExperimentConfig(
        data_limit=20_000_000,
        min_item_ratings=0,  # 不過濾
        use_timestamp=False,
        use_item_bias=False,
        use_svd=True,
        n_components=128,
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
