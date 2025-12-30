"""
Test 1 (Refactored): Baseline KNN without SVD
使用 1M 筆評分資料，不使用 SVD 降維
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from movie_recommendation.experiment import Experiment, ExperimentConfig


def main():
    """Run Test 1 experiment."""
    config = ExperimentConfig(
        name="實驗1",
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
    
    print(f"\n實驗完成: {config.name}")


if __name__ == "__main__":
    main()
