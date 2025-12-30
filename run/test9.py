"""
Test 9 (Refactored): TF-IDF Weighting
使用 TF-IDF 權重處理熱門電影偏差
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from movie_recommendation.experiment import Experiment, ExperimentConfig


def main():
    """Run Test 9 experiment."""
    config = ExperimentConfig(
        name="實驗9",
        data_limit=None,
        use_timestamp=False,
        use_item_bias=False,
        use_svd=True,
        n_components=128,
        use_tfidf=True,  # Use TF-IDF weighting
        k_neighbors=50,
        n_samples=500,
        top_n=10,
        random_state=42
    )
    
    experiment = Experiment(config)
    results = experiment.run()
    
    print(f"\n實驗完成: {config.name}")


if __name__ == "__main__":
    main()
