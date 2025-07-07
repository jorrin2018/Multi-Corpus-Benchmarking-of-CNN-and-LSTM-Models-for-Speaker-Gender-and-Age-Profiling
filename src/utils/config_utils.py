"""
Configuration Utilities Module
==============================

This module provides utilities for configuration management in speaker profiling
as specified in "Multi-Corpus Benchmarking of CNN & LSTM Models for Speaker Profiling".

Features:
- YAML configuration loading and validation
- Configuration merging and overrides
- Environment variable substitution
- Configuration schema validation
- Default configuration management

Authors: Jorge Jorrin-Coz et al., 2025
License: MIT
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import warnings
from dataclasses import dataclass, field
from copy import deepcopy
import re

@dataclass
class DatasetConfig:
    """Configuration for dataset parameters."""
    name: str
    data_dir: str
    metadata_file: Optional[str] = None
    sample_rate: int = 16000
    feature_type: str = 'mel'
    n_mels: int = 128
    n_mfcc: int = 13
    chunk_duration: Optional[float] = None
    overlap: float = 0.5
    min_duration: float = 0.5
    max_duration: float = 10.0
    cache_features: bool = False
    cache_dir: Optional[str] = None

@dataclass
class ModelConfig:
    """Configuration for model parameters."""
    architecture: str
    model_type: str  # 'cnn' or 'lstm'
    num_classes: int = 2
    input_dim: Optional[int] = None
    pretrained: bool = True
    dropout: float = 0.5
    freeze_backbone: bool = True
    
    # LSTM specific
    hidden_dim: int = 256
    num_layers: int = 2
    bidirectional: bool = True
    use_attention: bool = True
    
    # Additional parameters
    model_kwargs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = 'adam'
    
    # Scheduling
    lr_scheduler: Optional[str] = None
    scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    monitor_metric: str = 'accuracy'
    monitor_mode: str = 'max'
    
    # Regularization
    max_grad_norm: float = 1.0
    
    # Two-stage pipeline
    stage1_epochs: int = 50
    stage2_epochs: int = 100
    unfreeze_layers: int = 4
    
    # Multi-seed training
    seeds: List[int] = field(default_factory=lambda: [42, 43, 44, 45, 46, 47, 48, 49, 50, 51])

@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    experiment_name: str
    output_dir: str
    
    # Configurations
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    
    # Additional settings
    device: str = 'auto'
    mixed_precision: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    verbose: bool = True

def load_yaml_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path (Union[str, Path]): Path to YAML configuration file
        
    Returns:
        Dict[str, Any]: Loaded configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML is invalid
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Substitute environment variables
        config = substitute_env_vars(config)
        
        return config
        
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file {config_path}: {e}")

def save_yaml_config(config: Dict[str, Any], output_path: Union[str, Path]):
    """
    Save configuration to YAML file.
    
    Args:
        config (Dict[str, Any]): Configuration to save
        output_path (Union[str, Path]): Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2, sort_keys=False)

def substitute_env_vars(config: Any) -> Any:
    """
    Recursively substitute environment variables in configuration.
    
    Environment variables should be specified as ${VAR_NAME} or ${VAR_NAME:default_value}
    
    Args:
        config (Any): Configuration object
        
    Returns:
        Any: Configuration with environment variables substituted
    """
    if isinstance(config, dict):
        return {key: substitute_env_vars(value) for key, value in config.items()}
    elif isinstance(config, list):
        return [substitute_env_vars(item) for item in config]
    elif isinstance(config, str):
        # Pattern to match ${VAR_NAME} or ${VAR_NAME:default}
        pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
        
        def replacer(match):
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) is not None else ''
            return os.environ.get(var_name, default_value)
        
        return re.sub(pattern, replacer, config)
    else:
        return config

def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configurations, with override_config taking precedence.
    
    Args:
        base_config (Dict[str, Any]): Base configuration
        override_config (Dict[str, Any]): Override configuration
        
    Returns:
        Dict[str, Any]: Merged configuration
    """
    merged = deepcopy(base_config)
    
    def _merge_recursive(base: Dict, override: Dict):
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                _merge_recursive(base[key], value)
            else:
                base[key] = value
    
    _merge_recursive(merged, override_config)
    return merged

def validate_dataset_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate dataset configuration.
    
    Args:
        config (Dict[str, Any]): Dataset configuration
        
    Returns:
        List[str]: List of validation errors
    """
    errors = []
    
    # Required fields
    required_fields = ['name', 'data_dir']
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    # Data directory existence
    if 'data_dir' in config:
        data_dir = Path(config['data_dir'])
        if not data_dir.exists():
            errors.append(f"Data directory does not exist: {data_dir}")
    
    # Valid feature types
    valid_feature_types = ['mel', 'mfcc', 'linear']
    if 'feature_type' in config and config['feature_type'] not in valid_feature_types:
        errors.append(f"Invalid feature_type. Must be one of: {valid_feature_types}")
    
    # Positive values
    positive_fields = ['sample_rate', 'n_mels', 'n_mfcc', 'min_duration', 'max_duration']
    for field in positive_fields:
        if field in config and config[field] <= 0:
            errors.append(f"{field} must be positive")
    
    # Duration constraints
    if 'min_duration' in config and 'max_duration' in config:
        if config['min_duration'] >= config['max_duration']:
            errors.append("min_duration must be less than max_duration")
    
    # Overlap constraints
    if 'overlap' in config:
        if not 0 <= config['overlap'] < 1:
            errors.append("overlap must be between 0 and 1")
    
    return errors

def validate_model_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate model configuration.
    
    Args:
        config (Dict[str, Any]): Model configuration
        
    Returns:
        List[str]: List of validation errors
    """
    errors = []
    
    # Required fields
    required_fields = ['architecture', 'model_type', 'num_classes']
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    # Valid model types
    valid_model_types = ['cnn', 'lstm']
    if 'model_type' in config and config['model_type'] not in valid_model_types:
        errors.append(f"Invalid model_type. Must be one of: {valid_model_types}")
    
    # LSTM specific validation
    if config.get('model_type') == 'lstm':
        if 'input_dim' not in config:
            errors.append("input_dim is required for LSTM models")
        
        lstm_fields = ['hidden_dim', 'num_layers']
        for field in lstm_fields:
            if field in config and config[field] <= 0:
                errors.append(f"{field} must be positive")
    
    # Positive values
    positive_fields = ['num_classes', 'hidden_dim', 'num_layers']
    for field in positive_fields:
        if field in config and config[field] <= 0:
            errors.append(f"{field} must be positive")
    
    # Dropout constraints
    if 'dropout' in config:
        if not 0 <= config['dropout'] <= 1:
            errors.append("dropout must be between 0 and 1")
    
    return errors

def validate_training_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate training configuration.
    
    Args:
        config (Dict[str, Any]): Training configuration
        
    Returns:
        List[str]: List of validation errors
    """
    errors = []
    
    # Positive values
    positive_fields = ['epochs', 'batch_size', 'learning_rate', 'patience']
    for field in positive_fields:
        if field in config and config[field] <= 0:
            errors.append(f"{field} must be positive")
    
    # Valid optimizers
    valid_optimizers = ['adam', 'sgd', 'adamw']
    if 'optimizer' in config and config['optimizer'] not in valid_optimizers:
        errors.append(f"Invalid optimizer. Must be one of: {valid_optimizers}")
    
    # Valid monitor modes
    valid_modes = ['max', 'min']
    if 'monitor_mode' in config and config['monitor_mode'] not in valid_modes:
        errors.append(f"Invalid monitor_mode. Must be one of: {valid_modes}")
    
    # Learning rate constraints
    if 'learning_rate' in config:
        if config['learning_rate'] > 1.0:
            errors.append("learning_rate seems too high (>1.0)")
    
    # Seeds validation
    if 'seeds' in config:
        if not isinstance(config['seeds'], list):
            errors.append("seeds must be a list")
        elif len(config['seeds']) == 0:
            errors.append("seeds list cannot be empty")
    
    return errors

def validate_experiment_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate complete experiment configuration.
    
    Args:
        config (Dict[str, Any]): Complete experiment configuration
        
    Returns:
        List[str]: List of validation errors
    """
    errors = []
    
    # Required top-level fields
    required_fields = ['experiment_name', 'output_dir', 'dataset', 'model', 'training']
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    # Validate sub-configurations
    if 'dataset' in config:
        errors.extend([f"dataset.{error}" for error in validate_dataset_config(config['dataset'])])
    
    if 'model' in config:
        errors.extend([f"model.{error}" for error in validate_model_config(config['model'])])
    
    if 'training' in config:
        errors.extend([f"training.{error}" for error in validate_training_config(config['training'])])
    
    # Cross-validation between configurations
    if 'dataset' in config and 'model' in config:
        dataset_config = config['dataset']
        model_config = config['model']
        
        # Check feature compatibility
        if model_config.get('model_type') == 'lstm' and 'input_dim' not in model_config:
            if dataset_config.get('feature_type') == 'mel':
                suggested_dim = dataset_config.get('n_mels', 128)
            elif dataset_config.get('feature_type') == 'mfcc':
                suggested_dim = dataset_config.get('n_mfcc', 13)
            else:
                suggested_dim = 257  # Default for linear spectrogram
            
            warnings.warn(f"LSTM model requires input_dim. Suggested: {suggested_dim}")
    
    return errors

def create_default_config() -> Dict[str, Any]:
    """
    Create default experiment configuration.
    
    Returns:
        Dict[str, Any]: Default configuration
    """
    return {
        'experiment_name': 'speaker_profiling_experiment',
        'output_dir': './experiments',
        'device': 'auto',
        'mixed_precision': True,
        'num_workers': 4,
        'pin_memory': True,
        'verbose': True,
        
        'dataset': {
            'name': 'example_dataset',
            'data_dir': './data',
            'sample_rate': 16000,
            'feature_type': 'mel',
            'n_mels': 128,
            'n_mfcc': 13,
            'min_duration': 0.5,
            'max_duration': 10.0,
            'cache_features': False
        },
        
        'model': {
            'architecture': 'resnet18',
            'model_type': 'cnn',
            'num_classes': 2,
            'pretrained': True,
            'dropout': 0.5,
            'freeze_backbone': True
        },
        
        'training': {
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'optimizer': 'adam',
            'early_stopping': True,
            'patience': 10,
            'monitor_metric': 'accuracy',
            'monitor_mode': 'max',
            'max_grad_norm': 1.0,
            'seeds': [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
        }
    }

def create_corpus_specific_config(corpus: str, task: str = 'gender') -> Dict[str, Any]:
    """
    Create corpus-specific configuration based on paper specifications.
    
    Args:
        corpus (str): Corpus name ('voxceleb1', 'common_voice', 'timit')
        task (str): Task type ('gender', 'age')
        
    Returns:
        Dict[str, Any]: Corpus-specific configuration
    """
    base_config = create_default_config()
    
    # Corpus-specific parameters from the paper
    corpus_params = {
        'voxceleb1': {
            'sample_rate': 16000,
            'n_mels': 224,
            'n_mfcc': 40,
            'chunk_duration': 3.0,
            'overlap': 0.5
        },
        'common_voice': {
            'sample_rate': 22050,
            'n_mels': 128,
            'n_mfcc': 13,
            'chunk_duration': None,
            'min_duration': 0.5,
            'max_duration': 10.0
        },
        'timit': {
            'sample_rate': 16000,
            'n_mels': 64,
            'n_mfcc': 13,
            'chunk_duration': None,
            'min_duration': 0.5,
            'max_duration': 5.0
        }
    }
    
    if corpus in corpus_params:
        base_config['dataset'].update(corpus_params[corpus])
        base_config['dataset']['name'] = corpus
    
    # Task-specific parameters
    if task == 'gender':
        base_config['model']['num_classes'] = 2
        base_config['training']['monitor_metric'] = 'accuracy'
    elif task == 'age':
        if corpus == 'timit':
            # Age regression for TIMIT
            base_config['model']['num_classes'] = 1
            base_config['training']['monitor_metric'] = 'mae'
            base_config['training']['monitor_mode'] = 'min'
        else:
            # Age classification for other corpora
            base_config['model']['num_classes'] = 6
            base_config['training']['monitor_metric'] = 'accuracy'
    
    # Update experiment name
    base_config['experiment_name'] = f"{corpus}_{task}_experiment"
    
    return base_config

def load_and_validate_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load and validate configuration from file.
    
    Args:
        config_path (Union[str, Path]): Path to configuration file
        
    Returns:
        Dict[str, Any]: Validated configuration
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Load configuration
    config = load_yaml_config(config_path)
    
    # Validate configuration
    errors = validate_experiment_config(config)
    
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
        raise ValueError(error_msg)
    
    return config

def update_config_from_args(config: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update configuration with command-line arguments.
    
    Args:
        config (Dict[str, Any]): Base configuration
        args (Dict[str, Any]): Command-line arguments
        
    Returns:
        Dict[str, Any]: Updated configuration
    """
    config = deepcopy(config)
    
    # Mapping from flat argument names to nested config paths
    arg_mapping = {
        'experiment_name': 'experiment_name',
        'output_dir': 'output_dir',
        'data_dir': 'dataset.data_dir',
        'batch_size': 'training.batch_size',
        'learning_rate': 'training.learning_rate',
        'epochs': 'training.epochs',
        'device': 'device',
        'architecture': 'model.architecture',
        'num_classes': 'model.num_classes'
    }
    
    for arg_name, value in args.items():
        if arg_name in arg_mapping and value is not None:
            config_path = arg_mapping[arg_name]
            
            # Navigate to nested configuration
            path_parts = config_path.split('.')
            current = config
            
            for part in path_parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Set the value
            current[path_parts[-1]] = value
    
    return config

class ConfigManager:
    """Manages configuration loading, validation, and updating."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path (Optional[Union[str, Path]]): Path to configuration file
        """
        self.config_path = Path(config_path) if config_path else None
        self.config = None
        
        if self.config_path:
            self.load_config()
    
    def load_config(self, config_path: Optional[Union[str, Path]] = None):
        """Load configuration from file."""
        if config_path:
            self.config_path = Path(config_path)
        
        if self.config_path and self.config_path.exists():
            self.config = load_and_validate_config(self.config_path)
        else:
            self.config = create_default_config()
    
    def save_config(self, output_path: Optional[Union[str, Path]] = None):
        """Save current configuration to file."""
        if output_path:
            save_path = Path(output_path)
        elif self.config_path:
            save_path = self.config_path
        else:
            raise ValueError("No output path specified")
        
        if self.config:
            save_yaml_config(self.config, save_path)
    
    def update_from_args(self, args: Dict[str, Any]):
        """Update configuration from command-line arguments."""
        if self.config:
            self.config = update_config_from_args(self.config, args)
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self.config or create_default_config()
    
    def validate(self) -> List[str]:
        """Validate current configuration."""
        if self.config:
            return validate_experiment_config(self.config)
        return ["No configuration loaded"] 