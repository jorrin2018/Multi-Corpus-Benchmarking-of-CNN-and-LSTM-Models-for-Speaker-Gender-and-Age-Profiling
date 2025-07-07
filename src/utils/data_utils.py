"""
Data Utilities Module
====================

This module provides utility functions for data handling in speaker profiling
as specified in "Multi-Corpus Benchmarking of CNN & LSTM Models for Speaker Profiling".

Utilities include:
- Data validation and preprocessing
- Train/validation/test splits
- Data transformations and augmentations
- File I/O operations
- Dataset statistics

Authors: Jorge Jorrin-Coz et al., 2025
License: MIT
"""

import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import warnings
import json
import yaml
import pickle
import shutil
from collections import Counter
import librosa

def validate_audio_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Validate an audio file and extract basic information.
    
    Args:
        file_path (Union[str, Path]): Path to audio file
        
    Returns:
        Dict[str, Any]: Validation results and file info
    """
    file_path = Path(file_path)
    
    result = {
        'valid': False,
        'file_path': str(file_path),
        'exists': file_path.exists(),
        'size_bytes': 0,
        'duration': 0.0,
        'sample_rate': 0,
        'channels': 0,
        'error': None
    }
    
    if not file_path.exists():
        result['error'] = "File does not exist"
        return result
    
    try:
        # Get file size
        result['size_bytes'] = file_path.stat().st_size
        
        # Load audio info using librosa
        info = librosa.get_duration(path=str(file_path))
        result['duration'] = info
        
        # Get more detailed info using torchaudio if available
        try:
            import torchaudio
            audio_info = torchaudio.info(str(file_path))
            result['sample_rate'] = audio_info.sample_rate
            result['channels'] = audio_info.num_channels
            result['duration'] = audio_info.num_frames / audio_info.sample_rate
        except ImportError:
            # Fallback to librosa
            audio, sr = librosa.load(str(file_path), sr=None)
            result['sample_rate'] = sr
            result['channels'] = 1 if len(audio.shape) == 1 else audio.shape[0]
        
        result['valid'] = True
        
    except Exception as e:
        result['error'] = str(e)
    
    return result

def validate_dataset_directory(
    data_dir: Union[str, Path],
    metadata_file: Optional[Union[str, Path]] = None,
    audio_extensions: List[str] = ['.wav', '.mp3', '.flac', '.m4a'],
    max_files_to_check: int = 1000
) -> Dict[str, Any]:
    """
    Validate a dataset directory structure and audio files.
    
    Args:
        data_dir (Union[str, Path]): Path to dataset directory
        metadata_file (Optional[Union[str, Path]]): Path to metadata file
        audio_extensions (List[str]): Supported audio file extensions
        max_files_to_check (int): Maximum number of files to validate
        
    Returns:
        Dict[str, Any]: Dataset validation results
    """
    data_dir = Path(data_dir)
    
    result = {
        'valid': False,
        'data_dir': str(data_dir),
        'exists': data_dir.exists(),
        'total_files': 0,
        'valid_files': 0,
        'invalid_files': 0,
        'audio_files': [],
        'file_errors': [],
        'metadata_valid': False,
        'statistics': {}
    }
    
    if not data_dir.exists():
        result['error'] = "Data directory does not exist"
        return result
    
    # Find audio files
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(data_dir.rglob(f'*{ext}'))
    
    result['total_files'] = len(audio_files)
    
    # Validate subset of files
    files_to_check = min(max_files_to_check, len(audio_files))
    valid_count = 0
    invalid_count = 0
    
    durations = []
    sample_rates = []
    
    for i, file_path in enumerate(audio_files[:files_to_check]):
        validation = validate_audio_file(file_path)
        
        if validation['valid']:
            valid_count += 1
            durations.append(validation['duration'])
            sample_rates.append(validation['sample_rate'])
            result['audio_files'].append({
                'path': str(file_path.relative_to(data_dir)),
                'duration': validation['duration'],
                'sample_rate': validation['sample_rate']
            })
        else:
            invalid_count += 1
            result['file_errors'].append({
                'path': str(file_path.relative_to(data_dir)),
                'error': validation['error']
            })
    
    result['valid_files'] = valid_count
    result['invalid_files'] = invalid_count
    
    # Calculate statistics
    if durations:
        result['statistics'] = {
            'duration': {
                'mean': np.mean(durations),
                'std': np.std(durations),
                'min': np.min(durations),
                'max': np.max(durations),
                'median': np.median(durations)
            },
            'sample_rate': {
                'unique_rates': list(set(sample_rates)),
                'most_common': Counter(sample_rates).most_common(1)[0] if sample_rates else None
            }
        }
    
    # Validate metadata file if provided
    if metadata_file:
        metadata_path = Path(metadata_file)
        if metadata_path.exists():
            try:
                if metadata_path.suffix.lower() == '.csv':
                    metadata = pd.read_csv(metadata_path)
                elif metadata_path.suffix.lower() == '.tsv':
                    metadata = pd.read_csv(metadata_path, sep='\t')
                else:
                    metadata = pd.read_csv(metadata_path)
                
                result['metadata_valid'] = True
                result['metadata_rows'] = len(metadata)
                result['metadata_columns'] = list(metadata.columns)
                
            except Exception as e:
                result['metadata_error'] = str(e)
    
    # Overall validation
    result['valid'] = (
        result['exists'] and 
        result['total_files'] > 0 and 
        result['valid_files'] > 0 and
        (not metadata_file or result['metadata_valid'])
    )
    
    return result

def create_stratified_split(
    data: Union[pd.DataFrame, List[Dict]],
    stratify_column: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[List, List, List]:
    """
    Create stratified train/validation/test splits.
    
    Args:
        data (Union[pd.DataFrame, List[Dict]]): Data to split
        stratify_column (str): Column to stratify by
        train_ratio (float): Proportion for training
        val_ratio (float): Proportion for validation
        test_ratio (float): Proportion for testing
        random_state (int): Random seed
        
    Returns:
        Tuple[List, List, List]: Train, validation, test indices
    """
    if isinstance(data, list):
        data = pd.DataFrame(data)
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    np.random.seed(random_state)
    
    # Get unique classes and their counts
    class_counts = data[stratify_column].value_counts()
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    for class_label, count in class_counts.items():
        # Get indices for this class
        class_indices = data[data[stratify_column] == class_label].index.tolist()
        np.random.shuffle(class_indices)
        
        # Calculate split sizes
        n_train = int(count * train_ratio)
        n_val = int(count * val_ratio)
        
        # Split indices
        train_indices.extend(class_indices[:n_train])
        val_indices.extend(class_indices[n_train:n_train + n_val])
        test_indices.extend(class_indices[n_train + n_val:])
    
    # Shuffle final indices
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)
    
    return train_indices, val_indices, test_indices

def calculate_dataset_statistics(
    data: Union[pd.DataFrame, List[Dict]],
    categorical_columns: List[str] = None,
    numerical_columns: List[str] = None
) -> Dict[str, Any]:
    """
    Calculate comprehensive dataset statistics.
    
    Args:
        data (Union[pd.DataFrame, List[Dict]]): Dataset to analyze
        categorical_columns (List[str]): Categorical columns to analyze
        numerical_columns (List[str]): Numerical columns to analyze
        
    Returns:
        Dict[str, Any]: Dataset statistics
    """
    if isinstance(data, list):
        data = pd.DataFrame(data)
    
    stats = {
        'total_samples': len(data),
        'total_features': len(data.columns),
        'missing_values': data.isnull().sum().to_dict(),
        'categorical_stats': {},
        'numerical_stats': {}
    }
    
    # Categorical statistics
    if categorical_columns:
        for col in categorical_columns:
            if col in data.columns:
                value_counts = data[col].value_counts()
                stats['categorical_stats'][col] = {
                    'unique_values': len(value_counts),
                    'value_counts': value_counts.to_dict(),
                    'most_common': value_counts.index[0] if len(value_counts) > 0 else None,
                    'distribution': (value_counts / len(data)).to_dict()
                }
    
    # Numerical statistics
    if numerical_columns:
        for col in numerical_columns:
            if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
                values = data[col].dropna()
                if len(values) > 0:
                    stats['numerical_stats'][col] = {
                        'mean': float(values.mean()),
                        'std': float(values.std()),
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'median': float(values.median()),
                        'q25': float(values.quantile(0.25)),
                        'q75': float(values.quantile(0.75))
                    }
    
    return stats

def save_data_splits(
    train_data: Any,
    val_data: Any,
    test_data: Any,
    output_dir: Union[str, Path],
    format: str = 'pickle'
) -> Dict[str, str]:
    """
    Save data splits to files.
    
    Args:
        train_data (Any): Training data
        val_data (Any): Validation data
        test_data (Any): Test data
        output_dir (Union[str, Path]): Output directory
        format (str): Save format ('pickle', 'json', 'csv')
        
    Returns:
        Dict[str, str]: Paths to saved files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}
    
    splits = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }
    
    for split_name, data in splits.items():
        if format == 'pickle':
            file_path = output_dir / f"{split_name}_data.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
                
        elif format == 'json':
            file_path = output_dir / f"{split_name}_data.json"
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        elif format == 'csv':
            file_path = output_dir / f"{split_name}_data.csv"
            if isinstance(data, pd.DataFrame):
                data.to_csv(file_path, index=False)
            elif isinstance(data, list):
                pd.DataFrame(data).to_csv(file_path, index=False)
            else:
                raise ValueError(f"Cannot save {type(data)} as CSV")
        
        else:
            raise ValueError(f"Unknown format: {format}")
        
        saved_files[split_name] = str(file_path)
    
    return saved_files

def load_data_splits(
    input_dir: Union[str, Path],
    format: str = 'pickle'
) -> Dict[str, Any]:
    """
    Load data splits from files.
    
    Args:
        input_dir (Union[str, Path]): Input directory
        format (str): Load format ('pickle', 'json', 'csv')
        
    Returns:
        Dict[str, Any]: Loaded data splits
    """
    input_dir = Path(input_dir)
    
    loaded_data = {}
    
    for split_name in ['train', 'val', 'test']:
        if format == 'pickle':
            file_path = input_dir / f"{split_name}_data.pkl"
            if file_path.exists():
                with open(file_path, 'rb') as f:
                    loaded_data[split_name] = pickle.load(f)
                    
        elif format == 'json':
            file_path = input_dir / f"{split_name}_data.json"
            if file_path.exists():
                with open(file_path, 'r') as f:
                    loaded_data[split_name] = json.load(f)
                    
        elif format == 'csv':
            file_path = input_dir / f"{split_name}_data.csv"
            if file_path.exists():
                loaded_data[split_name] = pd.read_csv(file_path)
        
        else:
            raise ValueError(f"Unknown format: {format}")
    
    return loaded_data

def normalize_audio_paths(
    metadata: pd.DataFrame,
    path_column: str,
    data_dir: Union[str, Path],
    check_existence: bool = True
) -> pd.DataFrame:
    """
    Normalize audio file paths in metadata.
    
    Args:
        metadata (pd.DataFrame): Metadata with file paths
        path_column (str): Column containing file paths
        data_dir (Union[str, Path]): Base data directory
        check_existence (bool): Whether to check if files exist
        
    Returns:
        pd.DataFrame: Metadata with normalized paths
    """
    data_dir = Path(data_dir)
    metadata = metadata.copy()
    
    normalized_paths = []
    valid_rows = []
    
    for idx, row in metadata.iterrows():
        original_path = row[path_column]
        
        # Try different path combinations
        possible_paths = [
            data_dir / original_path,
            data_dir / Path(original_path).name,  # Just filename
            Path(original_path) if Path(original_path).is_absolute() else None
        ]
        
        # Remove None values
        possible_paths = [p for p in possible_paths if p is not None]
        
        found_path = None
        for path in possible_paths:
            if path.exists():
                found_path = path
                break
        
        if found_path or not check_existence:
            normalized_paths.append(str(found_path or possible_paths[0]))
            valid_rows.append(idx)
        else:
            warnings.warn(f"Could not find audio file: {original_path}")
    
    # Update metadata
    if valid_rows:
        metadata = metadata.loc[valid_rows].copy()
        metadata[path_column] = normalized_paths
    
    return metadata

def create_balanced_subset(
    data: pd.DataFrame,
    balance_column: str,
    max_samples_per_class: int,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Create a balanced subset of data.
    
    Args:
        data (pd.DataFrame): Original data
        balance_column (str): Column to balance by
        max_samples_per_class (int): Maximum samples per class
        random_state (int): Random seed
        
    Returns:
        pd.DataFrame: Balanced subset
    """
    np.random.seed(random_state)
    
    balanced_data = []
    
    for class_label in data[balance_column].unique():
        class_data = data[data[balance_column] == class_label]
        
        if len(class_data) > max_samples_per_class:
            # Sample without replacement
            sampled_data = class_data.sample(
                n=max_samples_per_class, 
                random_state=random_state
            )
        else:
            sampled_data = class_data
        
        balanced_data.append(sampled_data)
    
    return pd.concat(balanced_data, ignore_index=True)

def verify_data_integrity(
    data_path: Union[str, Path],
    checksum_file: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Verify data integrity using checksums.
    
    Args:
        data_path (Union[str, Path]): Path to data file or directory
        checksum_file (Optional[Union[str, Path]]): Path to checksum file
        
    Returns:
        Dict[str, Any]: Integrity verification results
    """
    import hashlib
    
    data_path = Path(data_path)
    
    result = {
        'verified': False,
        'path': str(data_path),
        'exists': data_path.exists(),
        'checksum': None,
        'expected_checksum': None,
        'error': None
    }
    
    if not data_path.exists():
        result['error'] = "Data path does not exist"
        return result
    
    try:
        # Calculate checksum
        if data_path.is_file():
            # Single file
            with open(data_path, 'rb') as f:
                content = f.read()
                result['checksum'] = hashlib.md5(content).hexdigest()
        else:
            # Directory - calculate combined checksum
            all_files = sorted(data_path.rglob('*'))
            file_checksums = []
            
            for file_path in all_files:
                if file_path.is_file():
                    with open(file_path, 'rb') as f:
                        content = f.read()
                        file_checksum = hashlib.md5(content).hexdigest()
                        file_checksums.append(file_checksum)
            
            combined = ''.join(file_checksums)
            result['checksum'] = hashlib.md5(combined.encode()).hexdigest()
        
        # Load expected checksum if available
        if checksum_file and Path(checksum_file).exists():
            with open(checksum_file, 'r') as f:
                result['expected_checksum'] = f.read().strip()
        
        # Verify
        if result['expected_checksum']:
            result['verified'] = result['checksum'] == result['expected_checksum']
        else:
            result['verified'] = True  # No expected checksum to compare
            
    except Exception as e:
        result['error'] = str(e)
    
    return result

class DataTransforms:
    """Collection of data transformation utilities."""
    
    @staticmethod
    def normalize_features(
        features: np.ndarray,
        method: str = 'standard'
    ) -> np.ndarray:
        """
        Normalize features using different methods.
        
        Args:
            features (np.ndarray): Input features
            method (str): Normalization method
            
        Returns:
            np.ndarray: Normalized features
        """
        if method == 'standard':
            return (features - np.mean(features)) / np.std(features)
        elif method == 'minmax':
            return (features - np.min(features)) / (np.max(features) - np.min(features))
        elif method == 'robust':
            median = np.median(features)
            mad = np.median(np.abs(features - median))
            return (features - median) / mad
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    @staticmethod
    def augment_spectrogram(
        spectrogram: np.ndarray,
        time_mask_prob: float = 0.1,
        freq_mask_prob: float = 0.1,
        noise_factor: float = 0.01
    ) -> np.ndarray:
        """
        Apply data augmentation to spectrograms.
        
        Args:
            spectrogram (np.ndarray): Input spectrogram
            time_mask_prob (float): Probability of time masking
            freq_mask_prob (float): Probability of frequency masking
            noise_factor (float): Noise factor for additive noise
            
        Returns:
            np.ndarray: Augmented spectrogram
        """
        augmented = spectrogram.copy()
        
        # Time masking
        if np.random.random() < time_mask_prob:
            time_length = spectrogram.shape[1]
            mask_length = int(time_length * 0.1)
            start = np.random.randint(0, time_length - mask_length)
            augmented[:, start:start + mask_length] = 0
        
        # Frequency masking
        if np.random.random() < freq_mask_prob:
            freq_length = spectrogram.shape[0]
            mask_length = int(freq_length * 0.1)
            start = np.random.randint(0, freq_length - mask_length)
            augmented[start:start + mask_length, :] = 0
        
        # Additive noise
        if noise_factor > 0:
            noise = np.random.normal(0, noise_factor, spectrogram.shape)
            augmented += noise
        
        return augmented
    
    @staticmethod
    def resize_spectrogram(
        spectrogram: np.ndarray,
        target_shape: Tuple[int, int],
        method: str = 'bilinear'
    ) -> np.ndarray:
        """
        Resize spectrogram to target shape.
        
        Args:
            spectrogram (np.ndarray): Input spectrogram
            target_shape (Tuple[int, int]): Target (height, width)
            method (str): Interpolation method
            
        Returns:
            np.ndarray: Resized spectrogram
        """
        from scipy.ndimage import zoom
        
        current_shape = spectrogram.shape
        zoom_factors = (
            target_shape[0] / current_shape[0],
            target_shape[1] / current_shape[1]
        )
        
        if method == 'bilinear':
            order = 1
        elif method == 'nearest':
            order = 0
        elif method == 'cubic':
            order = 3
        else:
            order = 1
        
        return zoom(spectrogram, zoom_factors, order=order) 