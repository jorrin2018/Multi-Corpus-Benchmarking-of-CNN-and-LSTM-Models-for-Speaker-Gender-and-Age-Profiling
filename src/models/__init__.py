"""
Models Module
=============

This module contains CNN and LSTM architectures for speaker profiling:

CNN Models (Transfer Learning from ImageNet):
    - MobileNetV2, EfficientNet-B0, ResNet50, ResNet18
    - VGG16, AlexNet, DenseNet
    - All adapted with conv1 for single channel input

LSTM Models:
    - Bidirectional LSTM with configurable layers
    - Hidden sizes: 128, 256, 512
    - Dropout and fully connected layers
"""

from .cnn_models import (
    MobileNetV2Model, EfficientNetB0Model, ResNet50Model, ResNet18Model,
    VGG16Model, AlexNetModel, DenseNetModel
)
from .lstm_models import BidirectionalLSTMModel

__all__ = [
    "MobileNetV2Model",
    "EfficientNetB0Model", 
    "ResNet50Model",
    "ResNet18Model",
    "VGG16Model",
    "AlexNetModel",
    "DenseNetModel",
    "BidirectionalLSTMModel"
] 