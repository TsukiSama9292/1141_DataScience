"""
Movie Recommendation System
A modular framework for collaborative filtering based recommendation.
"""

__version__ = "0.0.1"

from .data_loader import DataLoader
from .feature_engineering import FeatureEngineer
from .models import KNNRecommender
from .evaluation import Evaluator
from .experiment import Experiment

__all__ = [
    "DataLoader",
    "FeatureEngineer",
    "KNNRecommender",
    "Evaluator",
    "Experiment",
]
