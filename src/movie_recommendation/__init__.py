"""
Movie Recommendation System
A modular framework for collaborative filtering based recommendation.
"""

__version__ = "0.0.1"

# 核心模組 - 直接導入
from .data_loader import DataLoader
from .feature_engineering import FeatureEngineer
from .models import KNNRecommender
from .evaluation import Evaluator
from .experiment import Experiment

# 分析和報告模組 - 延遲導入以避免循環依賴
# 使用時請用: from src.movie_recommendation.analysis import ExperimentAnalyzer
# 或: from src.movie_recommendation.report_generator import generate_report

__all__ = [
    # 核心模組
    "DataLoader",
    "FeatureEngineer",
    "KNNRecommender",
    "Evaluator",
    "Experiment",
]


def get_analyzer():
    """獲取實驗分析器（延遲導入）"""
    from .analysis import ExperimentAnalyzer
    return ExperimentAnalyzer


def get_dataset_analyzer():
    """獲取資料集分析器（延遲導入）"""
    from .analysis import DatasetAnalyzer
    return DatasetAnalyzer


def get_report_generator():
    """獲取報告生成器（延遲導入）"""
    from .report_generator import ReportGenerator
    return ReportGenerator


def generate_report(*args, **kwargs):
    """生成報告（延遲導入）"""
    from .report_generator import generate_report as _generate_report
    return _generate_report(*args, **kwargs)
