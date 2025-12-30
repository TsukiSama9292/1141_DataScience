"""
Test 5 (Refactored): 引入 NDCG 指標
全量資料，完整評估指標包含 NDCG
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from movie_recommendation.experiment import Experiment, ExperimentConfig


def main():
    """Run Test 5 experiment."""
    config = ExperimentConfig(
        name="實驗5",
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
