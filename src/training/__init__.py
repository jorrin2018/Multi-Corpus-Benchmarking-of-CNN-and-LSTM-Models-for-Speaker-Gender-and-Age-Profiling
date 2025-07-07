"""
Training Module
===============

This module provides training utilities and components:

- Two-stage training pipeline (selection + fine-tuning)
- Trainer class with Adam optimizer and ReduceLROnPlateau scheduler
- Early stopping and model checkpointing
- Multi-seed training for robust results (10 seeds)
- Class weight balancing for imbalanced datasets
"""

from .trainer import Trainer
from .callbacks import EarlyStopping, ModelCheckpoint, LRScheduler

__all__ = [
    "Trainer",
    "EarlyStopping", 
    "ModelCheckpoint",
    "LRScheduler"
] 