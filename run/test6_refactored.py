"""
Test 6 (Refactored): Best baseline with Item Bias + SVD + KNN
使用完整資料集，Item Bias + SVD 128 維 + KNN 50 鄰居
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from movie_recommendation.experiment import Experiment, ExperimentConfig


def main():
    """Run Test 6 experiment."""
    config = ExperimentConfig(
        name="實驗6",
        data_limit=None,  # Full dataset
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
    
    print(f"\n實驗完成: {config.name}")

if __name__ == "__main__":
    main()
