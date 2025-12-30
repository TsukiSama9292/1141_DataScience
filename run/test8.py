"""
Test 8 (Refactored): Similarity Amplification
使用相似度放大，強化鄰居相似度影響
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from movie_recommendation.experiment import Experiment, ExperimentConfig


def main():
    """Run Test 8 experiment."""
    config = ExperimentConfig(
        name="實驗8",
        data_limit=None,
        use_timestamp=False,
        use_item_bias=True,
        use_svd=True,
        n_components=128,
        k_neighbors=50,
        amplification_factor=2.5,  # Amplify similarities
        n_samples=500,
        top_n=10,
        random_state=42
    )
    
    experiment = Experiment(config)
    results = experiment.run()
    
    print(f"\n實驗完成: {config.name}")


if __name__ == "__main__":
    main()
