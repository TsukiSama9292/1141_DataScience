"""
Test 2 (Refactored): SVD 降維 (50 維) + Top 5k 電影
使用 1M 筆評分資料，SVD 50 維降維，篩選熱門 5000 部電影
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from movie_recommendation.experiment import Experiment, ExperimentConfig


def main():
    """Run Test 2 experiment."""
    config = ExperimentConfig(
        name="實驗2",
        data_limit=1_000_000,
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
