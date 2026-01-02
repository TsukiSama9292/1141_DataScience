"""
實驗1: Baseline - 1M資料, 無SVD, KNN=20
目的: 建立最小基線，驗證基本KNN在小數據上的表現
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from movie_recommendation.experiment import Experiment, ExperimentConfig


def main():
    config = ExperimentConfig(
        data_limit=1_000_000,
        use_timestamp=False,
        use_item_bias=False,
        use_svd=False,
        k_neighbors=20,
        n_samples=500,
        top_n=10,
        random_state=42
    )
    
    experiment = Experiment(config)
    results = experiment.run()
    
    print(f"✅ 運行完成")


if __name__ == "__main__":
    main()
