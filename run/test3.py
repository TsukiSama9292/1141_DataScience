"""
Test 3 (Refactored): 全量資料 + 稀疏矩陣優化
使用完整 MovieLens 20M 資料集，CSR 稀疏矩陣 + SVD 50 維
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from movie_recommendation.experiment import Experiment, ExperimentConfig


def main():
    """Run Test 3 experiment."""
    config = ExperimentConfig(
        name="實驗3",
        data_limit=None,  # Full dataset
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
    
    print(f"\n實驗完成: {config.name}")


if __name__ == "__main__":
    main()
