"""
Evaluation Module
=================

This module provides evaluation metrics and benchmarking tools:

- Classification metrics (accuracy, precision, recall, F1)
- Regression metrics (MAE, MSE, RMSE)
- Statistical significance testing (paired t-test)
- SOTA comparison and benchmarking
- Confusion matrices and classification reports
- Performance visualization
"""

from .metrics import ClassificationMetrics, RegressionMetrics, StatisticalTests
from .benchmarking import SOTAComparison, BenchmarkRunner

__all__ = [
    "ClassificationMetrics",
    "RegressionMetrics", 
    "StatisticalTests",
    "SOTAComparison",
    "BenchmarkRunner"
] 