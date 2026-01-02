"""
實驗23: 20M資料, SVD=128維, KNN=50, TF-IDF (負面結果)
目的: 驗證TF-IDF在電影推薦中的災難性效果
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from movie_recommendation.experiment import Experiment, ExperimentConfig


def main():
    config = ExperimentConfig(
        name="實驗23",
        data_limit=None,
        use_timestamp=False,
        use_item_bias=False,
        use_svd=True,
        n_components=128,
        use_tfidf=True,
        k_neighbors=50,
        n_samples=500,
        top_n=10,
        random_state=42
    )
    
    experiment = Experiment(config)
    results = experiment.run()
    
    print(f"\n✅ 實驗完成: {config.name} - 負面結果")


if __name__ == "__main__":
    main()
