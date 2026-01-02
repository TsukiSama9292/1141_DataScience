"""
實驗7: 20M資料, SVD=50維, KNN=20
目的: 測試中低維度SVD
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from movie_recommendation.experiment import Experiment, ExperimentConfig


def main():
    config = ExperimentConfig(
        name="實驗7",
        data_limit=None,
        use_timestamp=False,
        use_item_bias=False,
        use_svd=True,
        n_components=50,
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
