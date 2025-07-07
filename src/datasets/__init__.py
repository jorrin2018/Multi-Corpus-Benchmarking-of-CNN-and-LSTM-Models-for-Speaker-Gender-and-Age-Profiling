"""
Datasets Module
===============

This module provides dataset loaders and utilities for:
- VoxCeleb1: Gender classification (563F/688M speakers)  
- Common Voice 17.0: Gender + Age classification (2,953F/10,107M)
- TIMIT: Gender classification + Age regression (192F/438M)

Each dataset class handles:
- Data loading and validation
- Metadata processing
- Train/validation splits
- Balanced sampling
"""

from .voxceleb1 import VoxCeleb1Dataset
from .common_voice import CommonVoiceDataset  
from .timit import TIMITDataset

__all__ = [
    "VoxCeleb1Dataset",
    "CommonVoiceDataset",
    "TIMITDataset"
] 