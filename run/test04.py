"""
實驗4: 20M全量資料, 無SVD, KNN=20
目的: 完整資料集的無降維基線
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from movie_recommendation.experiment import Experiment, ExperimentConfig


def main():
    config = ExperimentConfig(
        name="實驗4",
        data_limit=None,  # Full dataset
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
    
    print(f"\n✅ 實驗完成: {config.name}")


if __name__ == "__main__":
    main()
