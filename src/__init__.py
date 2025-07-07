"""
Multi-Corpus Speaker Profiling Benchmark
========================================

This package provides a comprehensive benchmarking suite for speaker profiling 
models (gender and age classification) across VoxCeleb1, Common Voice, and TIMIT datasets.

Authors: Jorge Jorrin-Coz et al., 2025
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Jorge Jorrin-Coz et al."
__email__ = "jljorrincoz@gmail.com"

from . import preprocessing
from . import models
from . import datasets
from . import training
from . import evaluation
from . import utils

__all__ = [
    "preprocessing",
    "models", 
    "datasets",
    "training",
    "evaluation",
    "utils"
] 