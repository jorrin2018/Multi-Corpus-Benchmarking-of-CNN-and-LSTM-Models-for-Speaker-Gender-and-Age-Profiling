"""
TIMIT Dataset Module
===================

This module implements the TIMIT dataset loader for speaker profiling
as specified in "Multi-Corpus Benchmarking of CNN & LSTM Models for Speaker Profiling".

TIMIT specifications:
- 630 speakers total
- 16 kHz sample rate
- Short sentences (typically 2-4 seconds)
- 64 mel-bins, 13 MFCC coefficients
- Gender classification (binary)
- Age regression (continuous age values)

Authors: Jorge Jorrin-Coz et al., 2025
License: MIT
"""

import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
from pathlib import Path
import re

from ..preprocessing.audio_processing import preprocess_audio
from ..preprocessing.feature_extraction import extract_features_for_corpus

class TIMITDataset(Dataset):
    """
    TIMIT dataset for speaker profiling.
    
    Loads audio files and applies preprocessing and feature extraction
    according to the paper specifications.
    """
    
    def __init__(
        self,
        data_dir: str,
        metadata_file: Optional[str] = None,
        feature_type: str = 'mel',
        task: str = 'gender',
        sample_rate: int = 16000,
        min_duration: float = 0.5,
        max_duration: float = 10.0,
        max_samples_per_speaker: int = 10,
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
        cache_features: bool = False,
        cache_dir: Optional[str] = None,
        parse_speaker_info: bool = True
    ):
        """
        Initialize TIMIT dataset.
        
        Args:
            data_dir (str): Path to TIMIT audio files
            metadata_file (Optional[str]): Path to metadata file (optional for TIMIT)
            feature_type (str): Type of features ('mel', 'mfcc', 'linear')
            task (str): Task type ('gender', 'age')
            sample_rate (int): Target sample rate
            min_duration (float): Minimum audio duration to include
            max_duration (float): Maximum audio duration to include
            max_samples_per_speaker (int): Maximum samples per speaker
            transform (Optional[callable]): Transform to apply to features
            target_transform (Optional[callable]): Transform to apply to labels
            cache_features (bool): Whether to cache extracted features
            cache_dir (Optional[str]): Directory for feature cache
            parse_speaker_info (bool): Whether to parse speaker info from filenames
        """
        self.data_dir = Path(data_dir)
        self.metadata_file = metadata_file
        self.feature_type = feature_type
        self.task = task
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.max_samples_per_speaker = max_samples_per_speaker
        self.transform = transform
        self.target_transform = target_transform
        self.cache_features = cache_features
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.parse_speaker_info = parse_speaker_info
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Create file list
        self.file_list = self._create_file_list()
        
        # Create label encoders
        self.label_encoder = self._create_label_encoder()
        
        # Setup cache if enabled
        if self.cache_features and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"TIMIT dataset initialized:")
        print(f"  - Total files: {len(self.file_list)}")
        print(f"  - Unique speakers: {len(self.get_unique_speakers())}")
        print(f"  - Task: {self.task}")
        print(f"  - Feature type: {self.feature_type}")
        print(f"  - Sample rate: {self.sample_rate} Hz")
    
    def _load_metadata(self) -> pd.DataFrame:
        """Load metadata from file or parse from directory structure."""
        if self.metadata_file and Path(self.metadata_file).exists():
            try:
                metadata = pd.read_csv(self.metadata_file)
                return metadata
            except Exception as e:
                warnings.warn(f"Error loading metadata file: {e}")
        
        # If no metadata file, parse from TIMIT directory structure
        if self.parse_speaker_info:
            return self._parse_timit_structure()
        else:
            return pd.DataFrame()
    
    def _parse_timit_structure(self) -> pd.DataFrame:
        """Parse TIMIT directory structure to extract speaker information."""
        metadata = []
        
        # TIMIT has structure: TIMIT/TRAIN|TEST/dialect/speaker/sentence.WAV
        # Speaker ID format: dialect + speaker (e.g., DR1_FCJF0)
        
        for split_dir in ['TRAIN', 'TEST']:
            split_path = self.data_dir / split_dir
            if not split_path.exists():
                continue
                
            for dialect_dir in split_path.iterdir():
                if not dialect_dir.is_dir():
                    continue
                    
                for speaker_dir in dialect_dir.iterdir():
                    if not speaker_dir.is_dir():
                        continue
                    
                    speaker_id = f"{dialect_dir.name}_{speaker_dir.name}"
                    
                    # Parse speaker information from ID
                    # Format: DR1_FCJF0 where F/M = gender, CJF = initials, 0 = instance
                    gender = self._parse_gender_from_id(speaker_dir.name)
                    age = self._parse_age_from_id(speaker_dir.name)
                    
                    # Find audio files
                    for audio_file in speaker_dir.glob('*.WAV'):
                        metadata.append({
                            'speaker_id': speaker_id,
                            'file_path': str(audio_file.relative_to(self.data_dir)),
                            'gender': gender,
                            'age': age,
                            'dialect': dialect_dir.name,
                            'split': split_dir.lower()
                        })
        
        return pd.DataFrame(metadata)
    
    def _parse_gender_from_id(self, speaker_id: str) -> str:
        """Parse gender from TIMIT speaker ID."""
        # First character after DR* is gender (M/F)
        if len(speaker_id) > 0:
            gender_char = speaker_id[0].upper()
            if gender_char == 'M':
                return 'male'
            elif gender_char == 'F':
                return 'female'
        return 'unknown'
    
    def _parse_age_from_id(self, speaker_id: str) -> Optional[int]:
        """Parse age from TIMIT speaker ID (if available)."""
        # TIMIT doesn't typically include age in the ID
        # This would need to be provided via external metadata
        return None
    
    def _create_file_list(self) -> List[Dict[str, Any]]:
        """Create list of audio files with metadata."""
        file_list = []
        
        if self.metadata.empty:
            # Fallback: scan directory for audio files
            for audio_file in self.data_dir.rglob('*.WAV'):
                file_info = {
                    'file_path': str(audio_file),
                    'speaker_id': audio_file.parent.name,
                    'gender': 'unknown',
                    'age': None
                }
                file_list.append(file_info)
        else:
            # Use metadata
            for _, row in self.metadata.iterrows():
                file_path = self.data_dir / row['file_path']
                
                if not file_path.exists():
                    continue
                
                try:
                    # Get audio duration
                    info = torchaudio.info(str(file_path))
                    duration = info.num_frames / info.sample_rate
                    
                    # Filter by duration
                    if duration < self.min_duration or duration > self.max_duration:
                        continue
                    
                    file_info = {
                        'file_path': str(file_path),
                        'speaker_id': row['speaker_id'],
                        'gender': row['gender'],
                        'age': row.get('age', None),
                        'duration': duration,
                        'original_sample_rate': info.sample_rate
                    }
                    
                    file_list.append(file_info)
                    
                except Exception as e:
                    warnings.warn(f"Error processing {file_path}: {e}")
                    continue
        
        # Limit samples per speaker
        if self.max_samples_per_speaker:
            speaker_counts = {}
            filtered_list = []
            
            for item in file_list:
                speaker_id = item['speaker_id']
                if speaker_counts.get(speaker_id, 0) < self.max_samples_per_speaker:
                    filtered_list.append(item)
                    speaker_counts[speaker_id] = speaker_counts.get(speaker_id, 0) + 1
            
            file_list = filtered_list
        
        return file_list
    
    def _create_label_encoder(self) -> Dict[str, Any]:
        """Create label encoder for the specified task."""
        if self.task == 'gender':
            # Binary gender classification
            genders = [item['gender'] for item in self.file_list if item['gender'] != 'unknown']
            unique_labels = sorted(set(genders))
            
            if not unique_labels:
                # Default gender labels
                unique_labels = ['female', 'male']
            
            label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            
            return {
                'label_to_idx': label_to_idx,
                'idx_to_label': {idx: label for label, idx in label_to_idx.items()},
                'num_classes': len(unique_labels)
            }
            
        elif self.task == 'age':
            # Age regression - no encoding needed for continuous values
            ages = [item['age'] for item in self.file_list if item['age'] is not None]
            
            if ages:
                min_age = min(ages)
                max_age = max(ages)
            else:
                min_age, max_age = 18, 80  # Default range
            
            return {
                'min_age': min_age,
                'max_age': max_age,
                'num_classes': 1,  # Single continuous output
                'task_type': 'regression'
            }
        
        else:
            raise ValueError(f"Unknown task: {self.task}")
    
    def _get_cache_path(self, item_idx: int) -> Optional[Path]:
        """Get cache file path for feature caching."""
        if not self.cache_features or not self.cache_dir:
            return None
        
        file_info = self.file_list[item_idx]
        file_stem = Path(file_info['file_path']).stem
        cache_filename = f"{file_stem}_{self.feature_type}.pt"
        
        return self.cache_dir / cache_filename
    
    def _load_audio(self, file_info: Dict[str, Any]) -> torch.Tensor:
        """Load and preprocess audio file."""
        # Load audio
        audio, sr = torchaudio.load(file_info['file_path'])
        
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0)
        else:
            audio = audio.squeeze(0)
        
        # Convert to numpy for preprocessing
        audio = audio.numpy()
        
        # Apply preprocessing pipeline
        processed_audio, final_sr = preprocess_audio(
            audio=audio,
            sample_rate=sr,
            target_sr=self.sample_rate,
            target_duration=None,  # Keep original duration for TIMIT
            remove_silence_flag=True,
            silence_threshold=0.075,
            apply_preemphasis_flag=True,
            preemphasis_coeff=0.97,
            apply_filter=True,
            filter_cutoff=4000,
            filter_order=10,
            normalize_energy_flag=True,
            energy_method='zscore'
        )
        
        return processed_audio, final_sr
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item from dataset."""
        file_info = self.file_list[idx]
        
        # Check cache first
        cache_path = self._get_cache_path(idx)
        if cache_path and cache_path.exists():
            try:
                cached_data = torch.load(cache_path)
                features = cached_data['features']
                label = cached_data['label']
                
                # Apply transforms
                if self.transform:
                    features = self.transform(features)
                if self.target_transform:
                    label = self.target_transform(label)
                
                return features, label
            except Exception as e:
                warnings.warn(f"Error loading cached features: {e}")
        
        # Load and process audio
        try:
            audio, sr = self._load_audio(file_info)
            
            # Extract features
            features = extract_features_for_corpus(
                audio=audio,
                sample_rate=sr,
                corpus='timit',
                feature_type=self.feature_type,
                to_db=True,
                include_deltas=False
            )
            
            # Convert to tensor
            features = torch.FloatTensor(features)
            
            # Get label
            if self.task == 'gender':
                if file_info['gender'] == 'unknown':
                    # Default to first class if unknown
                    label = torch.LongTensor([0])
                else:
                    label_str = file_info['gender']
                    label = self.label_encoder['label_to_idx'][label_str]
                    label = torch.LongTensor([label])
                    
            elif self.task == 'age':
                if file_info['age'] is not None:
                    # Normalize age to [0, 1] range for regression
                    age = file_info['age']
                    min_age = self.label_encoder['min_age']
                    max_age = self.label_encoder['max_age']
                    normalized_age = (age - min_age) / (max_age - min_age)
                    label = torch.FloatTensor([normalized_age])
                else:
                    # Default age if unknown
                    label = torch.FloatTensor([0.5])  # Middle age
            else:
                raise ValueError(f"Unknown task: {self.task}")
            
            # Cache features if enabled
            if cache_path:
                try:
                    torch.save({
                        'features': features,
                        'label': label
                    }, cache_path)
                except Exception as e:
                    warnings.warn(f"Error caching features: {e}")
            
            # Apply transforms
            if self.transform:
                features = self.transform(features)
            if self.target_transform:
                label = self.target_transform(label)
            
            return features, label
            
        except Exception as e:
            warnings.warn(f"Error processing item {idx}: {e}")
            # Return dummy data
            if self.feature_type == 'mel':
                dummy_features = torch.zeros(64, 100)   # 64 mel-bins
            elif self.feature_type == 'mfcc':
                dummy_features = torch.zeros(13, 100)   # 13 MFCC coefficients
            else:
                dummy_features = torch.zeros(257, 100)  # 257 frequency bins
            
            if self.task == 'gender':
                dummy_label = torch.LongTensor([0])
            else:  # age regression
                dummy_label = torch.FloatTensor([0.5])
            
            return dummy_features, dummy_label
    
    def get_unique_speakers(self) -> List[str]:
        """Get list of unique speakers."""
        return list(set(item['speaker_id'] for item in self.file_list))
    
    def get_speaker_splits(self, 
                          train_ratio: float = 0.7,
                          val_ratio: float = 0.15,
                          test_ratio: float = 0.15,
                          random_state: int = 42) -> Dict[str, List[str]]:
        """
        Create speaker-based train/val/test splits.
        
        Args:
            train_ratio (float): Proportion of speakers for training
            val_ratio (float): Proportion of speakers for validation
            test_ratio (float): Proportion of speakers for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            Dict[str, List[str]]: Speaker IDs for each split
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        
        # Get unique speakers
        unique_speakers = self.get_unique_speakers()
        
        # Shuffle speakers
        np.random.seed(random_state)
        np.random.shuffle(unique_speakers)
        
        # Calculate split sizes
        n_speakers = len(unique_speakers)
        n_train = int(n_speakers * train_ratio)
        n_val = int(n_speakers * val_ratio)
        
        # Create splits
        train_speakers = unique_speakers[:n_train]
        val_speakers = unique_speakers[n_train:n_train + n_val]
        test_speakers = unique_speakers[n_train + n_val:]
        
        return {
            'train': train_speakers,
            'val': val_speakers,
            'test': test_speakers
        }

def create_timit_dataloaders(
    data_dir: str,
    metadata_file: Optional[str] = None,
    feature_type: str = 'mel',
    task: str = 'gender',
    batch_size: int = 32,
    num_workers: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """
    Create train/val/test dataloaders for TIMIT.
    
    Args:
        data_dir (str): Path to TIMIT audio files
        metadata_file (Optional[str]): Path to metadata file
        feature_type (str): Type of features ('mel', 'mfcc', 'linear')
        task (str): Task type ('gender', 'age')
        batch_size (int): Batch size for dataloaders
        num_workers (int): Number of workers for data loading
        train_ratio (float): Proportion of speakers for training
        val_ratio (float): Proportion of speakers for validation
        test_ratio (float): Proportion of speakers for testing
        random_state (int): Random seed for reproducibility
        **dataset_kwargs: Additional arguments for dataset
        
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]: 
            (train_loader, val_loader, test_loader, dataset_info)
    """
    # Create initial dataset to get speaker splits
    full_dataset = TIMITDataset(
        data_dir=data_dir,
        metadata_file=metadata_file,
        feature_type=feature_type,
        task=task,
        **dataset_kwargs
    )
    
    # Get speaker splits
    speaker_splits = full_dataset.get_speaker_splits(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state
    )
    
    # Create datasets for each split
    datasets = {}
    for split_name, speakers in speaker_splits.items():
        # Filter file list for this split
        split_file_list = [
            item for item in full_dataset.file_list 
            if item['speaker_id'] in speakers
        ]
        
        # Create dataset with filtered file list
        dataset = TIMITDataset(
            data_dir=data_dir,
            metadata_file=metadata_file,
            feature_type=feature_type,
            task=task,
            **dataset_kwargs
        )
        dataset.file_list = split_file_list
        
        datasets[split_name] = dataset
    
    # Create dataloaders
    train_loader = DataLoader(
        datasets['train'],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        datasets['val'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        datasets['test'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Dataset info
    dataset_info = {
        'corpus': 'timit',
        'task': task,
        'feature_type': feature_type,
        'num_classes': full_dataset.label_encoder['num_classes'],
        'label_encoder': full_dataset.label_encoder,
        'train_samples': len(datasets['train']),
        'val_samples': len(datasets['val']),
        'test_samples': len(datasets['test']),
        'speaker_splits': speaker_splits
    }
    
    return train_loader, val_loader, test_loader, dataset_info 