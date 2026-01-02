"""
實驗12: 20M資料, SVD=128維, KNN=10
目的: 測試較少鄰居數的影響
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from movie_recommendation.experiment import Experiment, ExperimentConfig


def main():
    config = ExperimentConfig(
        name="實驗12",
        data_limit=None,
        use_timestamp=False,
        use_item_bias=False,
        use_svd=True,
        n_components=128,
        k_neighbors=10,
        n_samples=500,
        top_n=10,
        random_state=42
    )
    
    experiment = Experiment(config)
    results = experiment.run()
    
    print(f"\n✅ 實驗完成: {config.name}")


if __name__ == "__main__":
    main()
