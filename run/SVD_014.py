"""
SVD_014: SVD 維度 1000
目的: 極限測試，驗證降維必要性
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from movie_recommendation.experiment import Experiment, ExperimentConfig


def main():
    config = ExperimentConfig(
        data_limit=20_000_000,
        min_item_ratings=0,
        use_timestamp=False,
        use_item_bias=False,
        use_svd=True,
        n_components=1000,
        k_neighbors=20,
        n_samples=500,
        top_n=10,
        random_state=42
    )
    
    experiment = Experiment(config)
    results = experiment.run()
    
    print(f"✅ 運行完成")
    return results


if __name__ == "__main__":
    main()
