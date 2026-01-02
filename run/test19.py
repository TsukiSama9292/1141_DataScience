"""
實驗19: 20M資料, SVD=128維, KNN=50, 有Item Bias ⭐最佳基線⭐
目的: 證明Item Bias的關鍵作用
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from movie_recommendation.experiment import Experiment, ExperimentConfig


def main():
    config = ExperimentConfig(
        name="實驗19",
        data_limit=None,
        use_timestamp=False,
        use_item_bias=True,
        use_svd=True,
        n_components=128,
        k_neighbors=50,
        n_samples=500,
        top_n=10,
        random_state=42
    )
    
    experiment = Experiment(config)
    results = experiment.run()
    
    print(f"\n✅ 實驗完成: {config.name} - 最佳基線配置")


if __name__ == "__main__":
    main()
