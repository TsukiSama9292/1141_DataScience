"""
Utilities module for logging, time tracking, and memory monitoring.
"""

import logging
import time
import gc
from typing import Dict, Optional
from pathlib import Path

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    import resource
    PSUTIL_AVAILABLE = False


def setup_logging(log_name: str, log_dir: str = "log") -> logging.Logger:
    """
    Set up logging with file and console handlers.
    
    Args:
        log_name: Name of the log file (without extension)
        log_dir: Directory to save log files
        
    Returns:
        Logger instance
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    # File handler
    file_handler = logging.FileHandler(log_path / f"{log_name}.log", encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


class TimeTracker:
    """Track execution time and memory usage."""
    
    def __init__(self):
        """Initialize TimeTracker."""
        self.stages = {}
        self.memory_samples = []
        self.start_time = None
        
    def start(self):
        """Start timing."""
        self.start_time = time.time()
        self.sample_memory()
        
    def log_stage(self, stage_name: str, elapsed_time: float):
        """
        Log a stage execution time.
        
        Args:
            stage_name: Name of the stage
            elapsed_time: Time elapsed in seconds
        """
        self.stages[stage_name] = elapsed_time
        
    def sample_memory(self):
        """Sample current memory usage."""
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            mem_mb = process.memory_info().rss / (1024 * 1024)
        else:
            mem_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if hasattr(resource, 'RLIMIT_RSS'):
                mem_mb = mem_kb / 1024
            else:
                mem_mb = mem_kb / 1024
        
        self.memory_samples.append(mem_mb)
        gc.collect()
        
    def get_peak_mb(self) -> float:
        """
        Get peak memory usage.
        
        Returns:
            Peak memory in MB
        """
        return max(self.memory_samples) if self.memory_samples else 0.0
    
    def get_summary(self) -> Dict[str, float]:
        """
        Get summary of all tracked metrics.
        
        Returns:
            Dictionary with total_time, peak_memory, and stage times
        """
        total_time = sum(self.stages.values())
        return {
            'total_time': total_time,
            'peak_memory': self.get_peak_mb(),
            **self.stages
        }


def format_time_stats(time_records: Dict[str, float]) -> str:
    """
    Format time statistics with percentages.
    
    Args:
        time_records: Dictionary of stage names and times
        
    Returns:
        Formatted string with time breakdown
    """
    total_time = sum(time_records.values())
    
    lines = [f"總執行時間: {total_time:.2f} 秒\n各階段執行時間:"]
    for stage, t in time_records.items():
        percentage = (t / total_time * 100) if total_time > 0 else 0
        lines.append(f"  {stage}: {t:.2f} 秒 ({percentage:.1f}%)")
    
    return "\n".join(lines)


def log_metrics(
    logger: logging.Logger,
    metrics: Dict[str, float],
    n_samples: int,
    time_records: Dict[str, float],
    peak_memory: float
):
    """
    Log all metrics in a standardized format.
    
    Args:
        logger: Logger instance
        metrics: Dictionary of metric names and values
        n_samples: Number of samples evaluated
        time_records: Dictionary of timing information
        peak_memory: Peak memory usage in MB
    """
    logger.info("")
    logger.info(f"評估結果 (樣本數: {n_samples})")
    logger.info("")
    
    # Ranking metrics
    logger.info("排序指標:")
    logger.info(f"  Precision@10 = {metrics.get('precision', 0):.4f}")
    logger.info(f"  Recall@10    = {metrics.get('recall', 0):.4f}")
    logger.info(f"  Hit Rate@10  = {metrics.get('hit_rate', 0):.4f}")
    logger.info(f"  MRR          = {metrics.get('mrr', 0):.4f}")
    logger.info(f"  NDCG@10      = {metrics.get('ndcg', 0):.4f}")
    logger.info("")  # 空行分隔
    
    # Rating metrics
    logger.info("評分預測指標:")
    logger.info(f"  RMSE = {metrics.get('rmse', 0):.4f}")
    logger.info(f"  MAE  = {metrics.get('mae', 0):.4f}")
    logger.info("")  # 空行分隔
    
    # System metrics
    logger.info("系統資源:")
    logger.info(f"  記憶體峰值: {peak_memory:.2f} MB")
    logger.info("")  # 空行分隔
    
    # Time statistics
    total_time = sum(time_records.values())
    logger.info(f"總執行時間: {total_time:.2f} 秒")
    logger.info("各階段執行時間:")
    for stage, t in time_records.items():
        percentage = (t / total_time * 100) if total_time > 0 else 0
        logger.info(f"  {stage}: {t:.2f} 秒 ({percentage:.1f}%)")
    
    logger.info("")
