"""
Movie Recommendation System
A modular framework for collaborative filtering based recommendation.

重構後的模組結構：
- core/: 核心實驗執行邏輯
- data/: 資料載入和特徵工程
- models/: 推薦模型（KNN, Hybrid）
- evaluation/: 評估指標
- analysis/: 結果分析和報告生成
- config/: 配置管理
- utils/: 通用工具函數
"""

__version__ = "2.0.0"

# 提供向後兼容的導入
from .data.loader import DataLoader
from .data.feature_engineering import FeatureEngineer
from .models.knn import KNNRecommender
from .models.hybrid import GenomeHybridModel
from .evaluation.evaluator import Evaluator
from .core.experiment import Experiment, ExperimentConfig
from .core.experiment_runner import ExperimentRunner
from .config.loader import ConfigLoader, ExperimentSpec
from .utils.common import setup_logging, TimeTracker

__all__ = [
    # 資料模組
    "DataLoader",
    "FeatureEngineer",
    
    # 模型模組
    "KNNRecommender",
    "GenomeHybridModel",
    
    # 評估模組
    "Evaluator",
    
    # 核心模組
    "Experiment",
    "ExperimentConfig",
    "ExperimentRunner",
    
    # 配置模組
    "ConfigLoader",
    "ExperimentSpec",
    
    # 工具模組
    "setup_logging",
    "TimeTracker",
]


def get_analyzer():
    """獲取實驗分析器（延遲導入）"""
    from .analysis.analyzer import ExperimentAnalyzer
    return ExperimentAnalyzer


def get_dataset_analyzer():
    """獲取資料集分析器（延遲導入）"""
    from .analysis.analyzer import DatasetAnalyzer
    return DatasetAnalyzer


def generate_report(*args, **kwargs):
    """生成報告（延遲導入）"""
    from .analysis.report_generator import generate_report as _generate_report
    return _generate_report(*args, **kwargs)
