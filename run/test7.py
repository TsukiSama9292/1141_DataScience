"""
Test 7 (Refactored): Time Decay Weighting
使用時間衰減加權，較近期的評分權重較高
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from movie_recommendation.experiment import Experiment, ExperimentConfig


def main():
    """Run Test 7 experiment."""
    config = ExperimentConfig(
        name="實驗7",
        data_limit=None,
        use_timestamp=True,  # Need timestamp for time decay
        use_item_bias=False,
        use_svd=True,
        n_components=128,
        use_time_decay=True,
        half_life_days=500,
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
