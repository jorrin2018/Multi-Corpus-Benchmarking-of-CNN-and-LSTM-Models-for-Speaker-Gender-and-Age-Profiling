"""
CNN Models Module
================

This module implements CNN architectures with transfer learning from ImageNet
as specified in "Multi-Corpus Benchmarking of CNN & LSTM Models for Speaker Profiling".

Supported architectures:
- MobileNet-V2: Lightweight, deployment-ready
- EfficientNet-B0: Compound scaling efficiency
- ResNet50/ResNet18: Deep residual networks
- VGG16: Classic CNN baseline
- AlexNet: Historic deep learning
- DenseNet121: Dense connectivity

All models are adapted with:
- conv1 adjusted to 1 channel (grayscale input)
- Frozen backbone until penultimate block
- Custom classifier for 2 classes (gender) or 6 classes (age)

Authors: Jorge Jorrin-Coz et al., 2025
License: MIT
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Union, Dict, Any
import warnings

class BaseCNNModel(nn.Module):
    """
    Base class for CNN models with transfer learning.
    
    Provides common functionality for all CNN architectures including:
    - ImageNet weight loading
    - Backbone freezing/unfreezing
    - Custom classifier replacement
    - Model adaptation for single-channel input
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.5,
        freeze_backbone: bool = True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.dropout = dropout
        self.freeze_backbone = freeze_backbone
        
        # To be implemented by subclasses
        self.backbone = None
        self.feature_dim = None
        
    def _adapt_first_layer(self, first_layer: nn.Module) -> nn.Module:
        """
        Adapt the first convolutional layer for single-channel input.
        
        Args:
            first_layer (nn.Module): Original first layer (typically Conv2d)
            
        Returns:
            nn.Module: Adapted layer for single-channel input
        """
        if isinstance(first_layer, nn.Conv2d):
            # Get original layer parameters
            out_channels = first_layer.out_channels
            kernel_size = first_layer.kernel_size
            stride = first_layer.stride
            padding = first_layer.padding
            bias = first_layer.bias is not None
            
            # Create new layer with 1 input channel
            new_layer = nn.Conv2d(
                in_channels=1,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias
            )
            
            # Initialize weights by averaging across input channels
            with torch.no_grad():
                if self.pretrained:
                    # Average the pretrained weights across the input channel dimension
                    original_weights = first_layer.weight.data
                    new_weights = original_weights.mean(dim=1, keepdim=True)
                    new_layer.weight.data = new_weights
                    
                    if bias:
                        new_layer.bias.data = first_layer.bias.data
                        
            return new_layer
        else:
            warnings.warn(f"First layer type {type(first_layer)} not supported for adaptation")
            return first_layer
    
    def _freeze_backbone_layers(self, num_layers_to_unfreeze: int = 2):
        """
        Freeze backbone layers except for the last few.
        
        Args:
            num_layers_to_unfreeze (int): Number of layers to keep unfrozen from the end
        """
        if self.backbone is None:
            return
            
        # Freeze all parameters first
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Unfreeze last few layers
        layers = list(self.backbone.children())
        for layer in layers[-num_layers_to_unfreeze:]:
            for param in layer.parameters():
                param.requires_grad = True
                
    def _create_classifier(self, input_dim: int) -> nn.Module:
        """
        Create custom classifier head.
        
        Args:
            input_dim (int): Input dimension from backbone
            
        Returns:
            nn.Module: Custom classifier
        """
        return nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(self.dropout * 0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(self.dropout * 0.3),
            nn.Linear(256, self.num_classes)
        )
        
    def unfreeze_last_layers(self, num_layers: int = 2):
        """
        Unfreeze the last N layers for fine-tuning.
        
        Args:
            num_layers (int): Number of layers to unfreeze from the end
        """
        self._freeze_backbone_layers(num_layers_to_unfreeze=num_layers)
        
    def get_num_trainable_params(self) -> tuple:
        """
        Get the number of trainable and total parameters.
        
        Returns:
            tuple: (trainable_params, total_params)
        """
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        return trainable_params, total_params

class MobileNetV2Model(BaseCNNModel):
    """
    MobileNet-V2 model for speaker profiling.
    
    Lightweight architecture suitable for deployment.
    Good balance between accuracy and computational efficiency.
    """
    
    def __init__(self, num_classes: int = 2, **kwargs):
        super().__init__(num_classes=num_classes, **kwargs)
        
        # Load pretrained MobileNet-V2
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if self.pretrained else None
        self.backbone = models.mobilenet_v2(weights=weights)
        
        # Adapt first layer for single-channel input
        self.backbone.features[0][0] = self._adapt_first_layer(self.backbone.features[0][0])
        
        # Get feature dimension
        self.feature_dim = self.backbone.classifier[1].in_features
        
        # Replace classifier
        self.backbone.classifier = self._create_classifier(self.feature_dim)
        
        # Freeze backbone if requested
        if self.freeze_backbone:
            self._freeze_backbone_layers(num_layers_to_unfreeze=2)
    
    def forward(self, x):
        return self.backbone(x)

class EfficientNetB0Model(BaseCNNModel):
    """
    EfficientNet-B0 model for speaker profiling.
    
    Efficient architecture with compound scaling.
    Optimizes depth, width, and resolution simultaneously.
    """
    
    def __init__(self, num_classes: int = 2, **kwargs):
        super().__init__(num_classes=num_classes, **kwargs)
        
        # Load pretrained EfficientNet-B0
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if self.pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        
        # Adapt first layer for single-channel input
        self.backbone.features[0][0] = self._adapt_first_layer(self.backbone.features[0][0])
        
        # Get feature dimension
        self.feature_dim = self.backbone.classifier[1].in_features
        
        # Replace classifier
        self.backbone.classifier = self._create_classifier(self.feature_dim)
        
        # Freeze backbone if requested
        if self.freeze_backbone:
            self._freeze_backbone_layers(num_layers_to_unfreeze=2)
    
    def forward(self, x):
        return self.backbone(x)

class ResNet50Model(BaseCNNModel):
    """
    ResNet-50 model for speaker profiling.
    
    Deep residual network with skip connections.
    Strong baseline with proven performance.
    """
    
    def __init__(self, num_classes: int = 2, **kwargs):
        super().__init__(num_classes=num_classes, **kwargs)
        
        # Load pretrained ResNet-50
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if self.pretrained else None
        self.backbone = models.resnet50(weights=weights)
        
        # Adapt first layer for single-channel input
        self.backbone.conv1 = self._adapt_first_layer(self.backbone.conv1)
        
        # Get feature dimension
        self.feature_dim = self.backbone.fc.in_features
        
        # Replace classifier
        self.backbone.fc = self._create_classifier(self.feature_dim)
        
        # Freeze backbone if requested
        if self.freeze_backbone:
            self._freeze_backbone_layers(num_layers_to_unfreeze=2)
    
    def forward(self, x):
        return self.backbone(x)

class ResNet18Model(BaseCNNModel):
    """
    ResNet-18 model for speaker profiling.
    
    Lighter ResNet variant with fewer layers.
    Faster training and inference while maintaining good performance.
    """
    
    def __init__(self, num_classes: int = 2, **kwargs):
        super().__init__(num_classes=num_classes, **kwargs)
        
        # Load pretrained ResNet-18
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if self.pretrained else None
        self.backbone = models.resnet18(weights=weights)
        
        # Adapt first layer for single-channel input
        self.backbone.conv1 = self._adapt_first_layer(self.backbone.conv1)
        
        # Get feature dimension
        self.feature_dim = self.backbone.fc.in_features
        
        # Replace classifier
        self.backbone.fc = self._create_classifier(self.feature_dim)
        
        # Freeze backbone if requested
        if self.freeze_backbone:
            self._freeze_backbone_layers(num_layers_to_unfreeze=2)
    
    def forward(self, x):
        return self.backbone(x)

class VGG16Model(BaseCNNModel):
    """
    VGG-16 model for speaker profiling.
    
    Classic CNN architecture with simple structure.
    Provides interpretable features and good baseline performance.
    """
    
    def __init__(self, num_classes: int = 2, **kwargs):
        super().__init__(num_classes=num_classes, **kwargs)
        
        # Load pretrained VGG-16
        weights = models.VGG16_Weights.IMAGENET1K_V1 if self.pretrained else None
        self.backbone = models.vgg16(weights=weights)
        
        # Adapt first layer for single-channel input
        self.backbone.features[0] = self._adapt_first_layer(self.backbone.features[0])
        
        # Get feature dimension (VGG has different structure)
        self.feature_dim = self.backbone.classifier[0].in_features
        
        # Replace classifier
        self.backbone.classifier = self._create_classifier(self.feature_dim)
        
        # Freeze backbone if requested
        if self.freeze_backbone:
            self._freeze_backbone_layers(num_layers_to_unfreeze=2)
    
    def forward(self, x):
        return self.backbone(x)

class AlexNetModel(BaseCNNModel):
    """
    AlexNet model for speaker profiling.
    
    Historic deep CNN that started the deep learning revolution.
    Simple architecture with good interpretability.
    """
    
    def __init__(self, num_classes: int = 2, **kwargs):
        super().__init__(num_classes=num_classes, **kwargs)
        
        # Load pretrained AlexNet
        weights = models.AlexNet_Weights.IMAGENET1K_V1 if self.pretrained else None
        self.backbone = models.alexnet(weights=weights)
        
        # Adapt first layer for single-channel input
        self.backbone.features[0] = self._adapt_first_layer(self.backbone.features[0])
        
        # Get feature dimension
        self.feature_dim = self.backbone.classifier[1].in_features
        
        # Replace classifier
        self.backbone.classifier = self._create_classifier(self.feature_dim)
        
        # Freeze backbone if requested
        if self.freeze_backbone:
            self._freeze_backbone_layers(num_layers_to_unfreeze=2)
    
    def forward(self, x):
        return self.backbone(x)

class DenseNetModel(BaseCNNModel):
    """
    DenseNet-121 model for speaker profiling.
    
    Dense connectivity patterns with feature reuse.
    Efficient parameter usage and gradient flow.
    """
    
    def __init__(self, num_classes: int = 2, **kwargs):
        super().__init__(num_classes=num_classes, **kwargs)
        
        # Load pretrained DenseNet-121
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if self.pretrained else None
        self.backbone = models.densenet121(weights=weights)
        
        # Adapt first layer for single-channel input
        self.backbone.features.conv0 = self._adapt_first_layer(self.backbone.features.conv0)
        
        # Get feature dimension
        self.feature_dim = self.backbone.classifier.in_features
        
        # Replace classifier
        self.backbone.classifier = self._create_classifier(self.feature_dim)
        
        # Freeze backbone if requested
        if self.freeze_backbone:
            self._freeze_backbone_layers(num_layers_to_unfreeze=2)
    
    def forward(self, x):
        return self.backbone(x)

# Model factory function
def create_cnn_model(
    architecture: str,
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.5,
    freeze_backbone: bool = True
) -> BaseCNNModel:
    """
    Factory function to create CNN models.
    
    Args:
        architecture (str): Model architecture name
        num_classes (int): Number of output classes
        pretrained (bool): Use ImageNet pretrained weights
        dropout (float): Dropout rate for classifier
        freeze_backbone (bool): Whether to freeze backbone initially
        
    Returns:
        BaseCNNModel: Initialized CNN model
        
    Raises:
        ValueError: If architecture is not supported
        
    Example:
        >>> model = create_cnn_model('mobilenet_v2', num_classes=2)
        >>> model = create_cnn_model('resnet50', num_classes=6, dropout=0.3)
    """
    architecture = architecture.lower().replace('-', '_')
    
    model_classes = {
        'mobilenet_v2': MobileNetV2Model,
        'efficientnet_b0': EfficientNetB0Model,
        'resnet50': ResNet50Model,
        'resnet18': ResNet18Model,
        'vgg16': VGG16Model,
        'alexnet': AlexNetModel,
        'densenet121': DenseNetModel,
        'densenet': DenseNetModel,  # Alias
    }
    
    if architecture not in model_classes:
        available = list(model_classes.keys())
        raise ValueError(f"Unknown architecture: {architecture}. Available: {available}")
    
    model_class = model_classes[architecture]
    
    return model_class(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        freeze_backbone=freeze_backbone
    )

def get_model_info(architecture: str) -> Dict[str, Any]:
    """
    Get information about a specific model architecture.
    
    Args:
        architecture (str): Model architecture name
        
    Returns:
        Dict[str, Any]: Model information including description and characteristics
    """
    model_info = {
        'mobilenet_v2': {
            'description': 'Lightweight model with depthwise separable convolutions',
            'parameters': '~3.5M',
            'characteristics': ['lightweight', 'mobile-friendly', 'efficient'],
            'use_case': 'deployment and real-time applications'
        },
        'efficientnet_b0': {
            'description': 'Compound scaling with optimized depth, width, and resolution',
            'parameters': '~5.3M', 
            'characteristics': ['efficient', 'balanced', 'scalable'],
            'use_case': 'best accuracy-efficiency trade-off'
        },
        'resnet50': {
            'description': 'Deep residual network with skip connections',
            'parameters': '~25.6M',
            'characteristics': ['proven', 'robust', 'deep'],
            'use_case': 'strong baseline and transfer learning'
        },
        'resnet18': {
            'description': 'Lighter ResNet variant with fewer layers',
            'parameters': '~11.7M',
            'characteristics': ['fast', 'simple', 'effective'],
            'use_case': 'quick experiments and resource-constrained environments'
        },
        'vgg16': {
            'description': 'Classic CNN with simple sequential structure',
            'parameters': '~138M',
            'characteristics': ['interpretable', 'simple', 'large'],
            'use_case': 'feature analysis and traditional computer vision'
        },
        'alexnet': {
            'description': 'Historic deep CNN architecture',
            'parameters': '~61M',
            'characteristics': ['historic', 'simple', 'interpretable'],
            'use_case': 'educational purposes and simple baselines'
        },
        'densenet121': {
            'description': 'Dense connectivity with feature reuse',
            'parameters': '~8M',
            'characteristics': ['efficient', 'dense-connections', 'gradient-friendly'],
            'use_case': 'parameter efficiency and feature reuse'
        }
    }
    
    architecture = architecture.lower().replace('-', '_')
    if architecture == 'densenet':
        architecture = 'densenet121'
        
    return model_info.get(architecture, {'description': 'Unknown architecture'})

def list_available_architectures() -> list:
    """
    List all available CNN architectures.
    
    Returns:
        list: List of available architecture names
    """
    return [
        'mobilenet_v2',
        'efficientnet_b0', 
        'resnet50',
        'resnet18',
        'vgg16',
        'alexnet',
        'densenet121'
    ] 