"""
Visualization module for experiment results and data analysis.
"""

import os
import re
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Import DataLoader from the project source
from src.movie_recommendation.data_loader import DataLoader 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaseVisualizer:
    """
    Base class for visualization.
    Handles common settings like fonts, styles, and directory management.
    """
    def __init__(self, output_dir="reports/figures"):
        self.output_dir = output_dir
        
        # 1. Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 2. Set plot style (support Chinese characters)
        sns.set_theme(style="whitegrid")
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
        plt.rcParams['axes.unicode_minus'] = False

    def save_figure(self, filename):
        """Unified figure saving logic to ensure font consistency."""
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, bbox_inches='tight', dpi=300) # dpi=300 for high resolution
        plt.close() # Close plot to release memory
        logger.info(f"圖表已儲存: {save_path}")


class ExperimentVisualizer(BaseVisualizer):
    """
    Visualizes experiment performance metrics.
    Corresponds to the requirement: Evaluate model performance under different conditions.
    """
    def __init__(self, log_dir="log"):
        super().__init__()
        self.log_dir = log_dir

    def parse_logs(self):
        """Scans the log directory and extracts metrics."""
        results = []
        if not os.path.exists(self.log_dir):
            logger.warning(f"Log 目錄不存在: {self.log_dir}")
            return pd.DataFrame()

        log_files = sorted([f for f in os.listdir(self.log_dir) if f.startswith("實驗") and f.endswith(".log")])
        
        for file_name in log_files:
            path = os.path.join(self.log_dir, file_name)
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Extract key metrics using Regex
                hit_rate = re.search(r"Hit Rate@10\s+=\s+([\d.]+)", content)
                ndcg = re.search(r"NDCG@10\s+=\s+([\d.]+)", content)
                rmse = re.search(r"RMSE\s+=\s+([\d.]+)", content)
                
                if hit_rate and ndcg and rmse:
                    results.append({
                        "實驗": file_name.replace(".log", ""),
                        "Hit Rate@10": float(hit_rate.group(1)),
                        "NDCG@10": float(ndcg.group(1)),
                        "RMSE": float(rmse.group(1))
                    })
        return pd.DataFrame(results)

    def run(self):
        """Executes the experiment data visualization process."""
        df = self.parse_logs()
        if df.empty:
            logger.error("無有效實驗數據，略過繪圖")
            return

        plt.figure(figsize=(14, 7)) # Increase size to prevent label overlapping
        
        # Plot lines
        sns.lineplot(data=df, x="實驗", y="Hit Rate@10", marker='o', label="Hit Rate@10", linewidth=2.5)
        sns.lineplot(data=df, x="實驗", y="NDCG@10", marker='s', label="NDCG@10", linewidth=2)
        
        # Annotate values for each point
        for index, row in df.iterrows():
            # Annotate Hit Rate (above the point)
            plt.text(row["實驗"], row["Hit Rate@10"] + 0.005, f'{row["Hit Rate@10"]:.3f}', 
                     ha='center', va='bottom', fontsize=10, color='blue', fontweight='bold')
            
            # Annotate NDCG (below the point)
            plt.text(row["實驗"], row["NDCG@10"] - 0.015, f'{row["NDCG@10"]:.3f}', 
                     ha='center', va='top', fontsize=10, color='green', fontweight='bold')

        plt.title("MovieLens 20M 實驗成效比較", fontsize=16)
        plt.ylabel("指標分數")
        plt.xlabel("實驗 ID")
        plt.ylim(0.2, 0.75) # Manual Y-axis adjustment for labels
        plt.xticks(rotation=45)
        plt.legend(loc='upper left') # Move legend to avoid obscuring data
        
        self.save_figure("experiment_performance.png")


class DatasetVisualizer(BaseVisualizer):
    """
    Visualizes raw dataset characteristics.
    Corresponds to the requirement: Data preprocessing and feature visualization.
    """
    def __init__(self):
        super().__init__()
        self.loader = DataLoader() # Initialize DataLoader

    def run(self):
        """Executes the dataset visualization process."""
        logger.info("讀取資料集樣本以繪製特徵 (Top 100k)...")
        
        # Load only top 100,000 records for visualization efficiency
        _, ratings = self.loader.load_data(limit=100000)
        
        if ratings is None or ratings.empty:
            logger.error("資料載入失敗，略過繪圖")
            return

        # 1. Plot Rating Distribution
        plt.figure(figsize=(10, 5))
        sns.countplot(x=ratings['rating'], palette="viridis", hue=ratings['rating'], legend=False)
        plt.title("評分分佈統計 (Rating Distribution)", fontsize=15)
        plt.xlabel("評分 (Stars)")
        plt.ylabel("數量 (Count)")
        
        # Annotate counts on bars
        ax = plt.gca()
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='bottom', fontsize=10)

        self.save_figure("data_rating_distribution.png")

        # 2. Plot Long Tail Effect (User Activity)
        user_counts = ratings['userId'].value_counts().values
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(user_counts)), user_counts, color='blue', linewidth=1.5)
        plt.fill_between(range(len(user_counts)), user_counts, color='blue', alpha=0.1)
        
        plt.title("使用者活躍度長尾分佈 (Long Tail Effect)", fontsize=15)
        plt.xlabel("使用者 (按活躍度排序)")
        plt.ylabel("評分數量")
        plt.xlim(0, 2000) 
        self.save_figure("data_user_activity_long_tail.png")


if __name__ == "__main__":
    # 1. Visualize experiment performance
    exp_viz = ExperimentVisualizer()
    exp_viz.run()

    # 2. Visualize dataset features
    data_viz = DatasetVisualizer()
    data_viz.run()