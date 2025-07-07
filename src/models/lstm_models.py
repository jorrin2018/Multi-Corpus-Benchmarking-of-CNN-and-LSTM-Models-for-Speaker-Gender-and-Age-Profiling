"""
LSTM Models Module
=================

This module implements LSTM architectures for speaker profiling
as specified in "Multi-Corpus Benchmarking of CNN & LSTM Models for Speaker Profiling".

Supported configurations (9 total):
- Hidden units: 128, 256, 512
- Number of layers: 1, 2, 3
- Bidirectional LSTM with dropout
- Attention mechanism options
- Batch normalization and residual connections

All models process sequential features (MFCC, mel-spectrograms) with:
- Input shape: (batch_size, sequence_length, feature_dim)
- Output: (batch_size, num_classes)

Authors: Jorge Jorrin-Coz et al., 2025
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Dict, Any
import math

class AttentionMechanism(nn.Module):
    """
    Attention mechanism for LSTM models.
    
    Computes attention weights over the sequence dimension to focus on
    important time steps for speaker profiling.
    """
    
    def __init__(self, hidden_dim: int, attention_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        
        # Attention layers
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
        
    def forward(self, lstm_output: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention mechanism to LSTM output.
        
        Args:
            lstm_output (torch.Tensor): LSTM output with shape (batch_size, seq_len, hidden_dim)
            lengths (Optional[torch.Tensor]): Actual sequence lengths for masking
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (attended_output, attention_weights)
        """
        batch_size, seq_len, hidden_dim = lstm_output.shape
        
        # Compute attention scores
        attention_scores = self.attention(lstm_output.reshape(-1, hidden_dim))
        attention_scores = attention_scores.view(batch_size, seq_len)
        
        # Apply mask if lengths are provided
        if lengths is not None:
            mask = torch.arange(seq_len, device=lstm_output.device).expand(batch_size, seq_len)
            mask = mask >= lengths.unsqueeze(1)
            attention_scores.masked_fill_(mask, -float('inf'))
        
        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Apply attention to LSTM output
        attended_output = torch.bmm(attention_weights.unsqueeze(1), lstm_output)
        attended_output = attended_output.squeeze(1)
        
        return attended_output, attention_weights

class BaseLSTMModel(nn.Module):
    """
    Base class for LSTM models.
    
    Provides common functionality for all LSTM configurations including:
    - Bidirectional LSTM layers
    - Dropout regularization
    - Batch normalization
    - Attention mechanism
    - Residual connections
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        num_classes: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        use_attention: bool = True,
        use_batch_norm: bool = True,
        use_residual: bool = False
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual
        
        # Calculate effective hidden dimension
        self.effective_hidden_dim = hidden_dim * (2 if bidirectional else 1)
        
        # Input projection (if needed)
        self.input_projection = None
        if use_residual and input_dim != self.effective_hidden_dim:
            self.input_projection = nn.Linear(input_dim, self.effective_hidden_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Batch normalization
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(self.effective_hidden_dim)
        
        # Attention mechanism
        if use_attention:
            self.attention = AttentionMechanism(self.effective_hidden_dim)
        
        # Classifier
        self.classifier = self._create_classifier()
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # Initialize weights
        self._initialize_weights()
    
    def _create_classifier(self) -> nn.Module:
        """Create classifier head."""
        layers = []
        
        # First layer
        layers.append(nn.Linear(self.effective_hidden_dim, 512))
        if self.use_batch_norm:
            layers.append(nn.BatchNorm1d(512))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(self.dropout))
        
        # Second layer
        layers.append(nn.Linear(512, 256))
        if self.use_batch_norm:
            layers.append(nn.BatchNorm1d(256))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(self.dropout * 0.5))
        
        # Output layer
        layers.append(nn.Linear(256, self.num_classes))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    # Xavier initialization for LSTM weights
                    nn.init.xavier_uniform_(param.data)
                else:
                    # He initialization for other layers
                    nn.init.kaiming_uniform_(param.data, nonlinearity='relu')
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the LSTM model.
        
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, seq_len, input_dim)
            lengths (Optional[torch.Tensor]): Actual sequence lengths for padding
            
        Returns:
            torch.Tensor: Output logits with shape (batch_size, num_classes)
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Store input for residual connection
        residual = x
        
        # Pack padded sequence if lengths are provided
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
        # LSTM forward pass
        lstm_output, (hidden, cell) = self.lstm(x)
        
        # Unpack sequence if it was packed
        if lengths is not None:
            lstm_output, _ = nn.utils.rnn.pad_packed_sequence(lstm_output, batch_first=True)
        
        # Apply residual connection if enabled
        if self.use_residual:
            if self.input_projection is not None:
                residual = self.input_projection(residual)
            # Ensure shapes match for residual connection
            if residual.shape[-1] == lstm_output.shape[-1]:
                lstm_output = lstm_output + residual
        
        # Apply attention or use last hidden state
        if self.use_attention:
            attended_output, attention_weights = self.attention(lstm_output, lengths)
            output = attended_output
        else:
            # Use last hidden state
            if self.bidirectional:
                # Concatenate final forward and backward hidden states
                forward_hidden = hidden[-2]  # Last forward layer
                backward_hidden = hidden[-1]  # Last backward layer
                output = torch.cat([forward_hidden, backward_hidden], dim=1)
            else:
                output = hidden[-1]  # Last layer
        
        # Apply batch normalization
        if self.use_batch_norm:
            output = self.batch_norm(output)
        
        # Apply dropout
        output = self.dropout_layer(output)
        
        # Classification
        logits = self.classifier(output)
        
        return logits
    
    def get_num_trainable_params(self) -> tuple:
        """
        Get the number of trainable and total parameters.
        
        Returns:
            tuple: (trainable_params, total_params)
        """
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        return trainable_params, total_params

class LSTMModel(BaseLSTMModel):
    """
    Standard LSTM model for speaker profiling.
    
    Configurable LSTM with various architectural options.
    """
    
    def __init__(self, input_dim: int, **kwargs):
        super().__init__(input_dim=input_dim, **kwargs)

def create_lstm_model(
    input_dim: int,
    hidden_dim: int = 128,
    num_layers: int = 1,
    num_classes: int = 2,
    dropout: float = 0.3,
    bidirectional: bool = True,
    use_attention: bool = True,
    use_batch_norm: bool = True,
    use_residual: bool = False
) -> BaseLSTMModel:
    """
    Factory function to create LSTM models with specified configuration.
    
    Args:
        input_dim (int): Input feature dimension
        hidden_dim (int): Hidden dimension (128, 256, or 512)
        num_layers (int): Number of LSTM layers (1, 2, or 3)
        num_classes (int): Number of output classes
        dropout (float): Dropout rate
        bidirectional (bool): Use bidirectional LSTM
        use_attention (bool): Use attention mechanism
        use_batch_norm (bool): Use batch normalization
        use_residual (bool): Use residual connections
        
    Returns:
        BaseLSTMModel: Configured LSTM model
        
    Example:
        >>> # Create basic LSTM
        >>> model = create_lstm_model(input_dim=40, hidden_dim=256, num_layers=2)
        >>> # Create advanced LSTM with attention
        >>> model = create_lstm_model(
        ...     input_dim=13, hidden_dim=512, num_layers=3,
        ...     use_attention=True, use_residual=True
        ... )
    """
    return LSTMModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        bidirectional=bidirectional,
        use_attention=use_attention,
        use_batch_norm=use_batch_norm,
        use_residual=use_residual
    )

def create_paper_lstm_configs(input_dim: int, num_classes: int = 2) -> Dict[str, BaseLSTMModel]:
    """
    Create all 9 LSTM configurations specified in the paper.
    
    Configurations:
    - Hidden units: 128, 256, 512
    - Number of layers: 1, 2, 3
    - All bidirectional with dropout
    
    Args:
        input_dim (int): Input feature dimension
        num_classes (int): Number of output classes
        
    Returns:
        Dict[str, BaseLSTMModel]: Dictionary of model configurations
        
    Example:
        >>> models = create_paper_lstm_configs(input_dim=40, num_classes=2)
        >>> print(list(models.keys()))
        ['lstm_128_1', 'lstm_128_2', 'lstm_128_3', 'lstm_256_1', ...]
    """
    configurations = {}
    
    # Hidden dimensions and layer counts from the paper
    hidden_dims = [128, 256, 512]
    num_layers_list = [1, 2, 3]
    
    for hidden_dim in hidden_dims:
        for num_layers in num_layers_list:
            config_name = f"lstm_{hidden_dim}_{num_layers}"
            
            model = create_lstm_model(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_classes=num_classes,
                dropout=0.3,
                bidirectional=True,
                use_attention=True,
                use_batch_norm=True,
                use_residual=False
            )
            
            configurations[config_name] = model
    
    return configurations

def get_lstm_model_info(hidden_dim: int, num_layers: int) -> Dict[str, Any]:
    """
    Get information about a specific LSTM configuration.
    
    Args:
        hidden_dim (int): Hidden dimension
        num_layers (int): Number of layers
        
    Returns:
        Dict[str, Any]: Model information
    """
    # Estimate parameter count (approximate)
    input_dim = 40  # Example input dimension
    bidirectional_factor = 2
    
    # LSTM parameters: 4 * (input_dim + hidden_dim + 1) * hidden_dim per layer
    lstm_params = 4 * (input_dim + hidden_dim + 1) * hidden_dim * num_layers * bidirectional_factor
    
    # Classifier parameters (approximate)
    classifier_params = (hidden_dim * bidirectional_factor * 512) + (512 * 256) + (256 * 2)
    
    total_params = lstm_params + classifier_params
    
    return {
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'bidirectional': True,
        'estimated_params': f"~{total_params // 1000}K",
        'characteristics': [
            'bidirectional',
            'attention-based',
            'batch-normalized',
            f"{num_layers}-layer"
        ],
        'use_case': f"Sequential modeling with {hidden_dim} hidden units"
    }

def list_paper_lstm_configs() -> list:
    """
    List all LSTM configurations from the paper.
    
    Returns:
        list: List of configuration names
    """
    configs = []
    for hidden_dim in [128, 256, 512]:
        for num_layers in [1, 2, 3]:
            configs.append(f"lstm_{hidden_dim}_{num_layers}")
    return configs

class LSTMFeatureExtractor(nn.Module):
    """
    LSTM-based feature extractor for transfer learning.
    
    Can be used as a feature extractor for other models or fine-tuning.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        use_attention: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        
        # Calculate effective hidden dimension
        self.effective_hidden_dim = hidden_dim * (2 if bidirectional else 1)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Attention mechanism
        if use_attention:
            self.attention = AttentionMechanism(self.effective_hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract features from sequential input.
        
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, seq_len, input_dim)
            lengths (Optional[torch.Tensor]): Actual sequence lengths
            
        Returns:
            torch.Tensor: Extracted features with shape (batch_size, feature_dim)
        """
        # Pack padded sequence if lengths are provided
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
        # LSTM forward pass
        lstm_output, (hidden, cell) = self.lstm(x)
        
        # Unpack sequence if it was packed
        if lengths is not None:
            lstm_output, _ = nn.utils.rnn.pad_packed_sequence(lstm_output, batch_first=True)
        
        # Apply attention or use last hidden state
        if self.use_attention:
            attended_output, _ = self.attention(lstm_output, lengths)
            features = attended_output
        else:
            # Use last hidden state
            if self.bidirectional:
                # Concatenate final forward and backward hidden states
                forward_hidden = hidden[-2]  # Last forward layer
                backward_hidden = hidden[-1]  # Last backward layer
                features = torch.cat([forward_hidden, backward_hidden], dim=1)
            else:
                features = hidden[-1]  # Last layer
        
        # Apply dropout
        features = self.dropout(features)
        
        return features 