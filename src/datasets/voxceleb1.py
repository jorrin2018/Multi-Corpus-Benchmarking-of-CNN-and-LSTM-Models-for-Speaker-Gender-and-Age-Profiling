"""
VoxCeleb1 Dataset Module
=======================

This module implements the VoxCeleb1 dataset loader for speaker profiling
as specified in "Multi-Corpus Benchmarking of CNN & LSTM Models for Speaker Profiling".

VoxCeleb1 specifications:
- 1,251 speakers total
- 16 kHz sample rate
- 3-second audio chunks
- 224 mel-bins, 40 MFCC coefficients
- Gender classification (binary)
- Age classification (6 groups: 18-25, 26-35, 36-45, 46-55, 56-65, 66+)

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

from ..preprocessing.audio_processing import preprocess_audio
from ..preprocessing.feature_extraction import extract_features_for_corpus

class VoxCeleb1Dataset(Dataset):
    """
    VoxCeleb1 dataset for speaker profiling.
    
    Loads audio files and applies preprocessing and feature extraction
    according to the paper specifications.
    """
    
    def __init__(
        self,
        data_dir: str,
        metadata_file: str,
        feature_type: str = 'mel',
        task: str = 'gender',
        chunk_duration: float = 3.0,
        sample_rate: int = 16000,
        overlap: float = 0.5,
        min_duration: float = 1.0,
        max_chunks_per_speaker: int = 50,
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
        cache_features: bool = False,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize VoxCeleb1 dataset.
        
        Args:
            data_dir (str): Path to VoxCeleb1 audio files
            metadata_file (str): Path to metadata CSV file
            feature_type (str): Type of features ('mel', 'mfcc', 'linear')
            task (str): Task type ('gender', 'age')
            chunk_duration (float): Duration of audio chunks in seconds
            sample_rate (int): Target sample rate
            overlap (float): Overlap between chunks (0.0 to 1.0)
            min_duration (float): Minimum audio duration to include
            max_chunks_per_speaker (int): Maximum chunks per speaker
            transform (Optional[callable]): Transform to apply to features
            target_transform (Optional[callable]): Transform to apply to labels
            cache_features (bool): Whether to cache extracted features
            cache_dir (Optional[str]): Directory for feature cache
        """
        self.data_dir = Path(data_dir)
        self.metadata_file = metadata_file
        self.feature_type = feature_type
        self.task = task
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.overlap = overlap
        self.min_duration = min_duration
        self.max_chunks_per_speaker = max_chunks_per_speaker
        self.transform = transform
        self.target_transform = target_transform
        self.cache_features = cache_features
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Create file list with chunks
        self.file_list = self._create_file_list()
        
        # Create label encoders
        self.label_encoder = self._create_label_encoder()
        
        # Setup cache if enabled
        if self.cache_features and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"VoxCeleb1 dataset initialized:")
        print(f"  - Total files: {len(self.file_list)}")
        print(f"  - Unique speakers: {len(self.metadata)}")
        print(f"  - Task: {self.task}")
        print(f"  - Feature type: {self.feature_type}")
        print(f"  - Chunk duration: {self.chunk_duration}s")
    
    def _load_metadata(self) -> pd.DataFrame:
        """Load and validate metadata."""
        try:
            metadata = pd.read_csv(self.metadata_file)
            
            # Validate required columns
            required_columns = ['speaker_id', 'file_path', 'gender']
            if self.task == 'age':
                required_columns.append('age_group')
            
            missing_columns = [col for col in required_columns if col not in metadata.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Filter valid entries
            metadata = metadata.dropna(subset=required_columns)
            
            # Validate file paths
            valid_files = []
            for _, row in metadata.iterrows():
                file_path = self.data_dir / row['file_path']
                if file_path.exists():
                    valid_files.append(row)
                else:
                    warnings.warn(f"File not found: {file_path}")
            
            if not valid_files:
                raise ValueError("No valid audio files found")
            
            return pd.DataFrame(valid_files)
            
        except Exception as e:
            raise ValueError(f"Error loading metadata: {e}")
    
    def _create_file_list(self) -> List[Dict[str, Any]]:
        """Create list of audio chunks with metadata."""
        file_list = []
        
        for _, row in self.metadata.iterrows():
            file_path = self.data_dir / row['file_path']
            
            try:
                # Get audio duration
                info = torchaudio.info(str(file_path))
                duration = info.num_frames / info.sample_rate
                
                if duration < self.min_duration:
                    continue
                
                # Calculate chunk parameters
                chunk_samples = int(self.chunk_duration * self.sample_rate)
                hop_samples = int(chunk_samples * (1 - self.overlap))
                
                # Generate chunks
                chunks = []
                start_sample = 0
                
                while start_sample + chunk_samples <= info.num_frames:
                    chunk_info = {
                        'file_path': str(file_path),
                        'start_sample': start_sample,
                        'chunk_samples': chunk_samples,
                        'speaker_id': row['speaker_id'],
                        'gender': row['gender'],
                        'original_sample_rate': info.sample_rate
                    }
                    
                    if self.task == 'age' and 'age_group' in row:
                        chunk_info['age_group'] = row['age_group']
                    
                    chunks.append(chunk_info)
                    start_sample += hop_samples
                
                # Limit chunks per speaker
                if len(chunks) > self.max_chunks_per_speaker:
                    # Sample evenly across the audio
                    indices = np.linspace(0, len(chunks) - 1, self.max_chunks_per_speaker, dtype=int)
                    chunks = [chunks[i] for i in indices]
                
                file_list.extend(chunks)
                
            except Exception as e:
                warnings.warn(f"Error processing {file_path}: {e}")
                continue
        
        return file_list
    
    def _create_label_encoder(self) -> Dict[str, Any]:
        """Create label encoder for the specified task."""
        if self.task == 'gender':
            # Binary gender classification
            unique_labels = sorted(self.metadata['gender'].unique())
            label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            
            return {
                'label_to_idx': label_to_idx,
                'idx_to_label': {idx: label for label, idx in label_to_idx.items()},
                'num_classes': len(unique_labels)
            }
            
        elif self.task == 'age':
            # Age group classification
            if 'age_group' not in self.metadata.columns:
                raise ValueError("Age group column not found in metadata")
            
            unique_labels = sorted(self.metadata['age_group'].unique())
            label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            
            return {
                'label_to_idx': label_to_idx,
                'idx_to_label': {idx: label for label, idx in label_to_idx.items()},
                'num_classes': len(unique_labels)
            }
        
        else:
            raise ValueError(f"Unknown task: {self.task}")
    
    def _get_cache_path(self, item_idx: int) -> Optional[Path]:
        """Get cache file path for feature caching."""
        if not self.cache_features or not self.cache_dir:
            return None
        
        chunk_info = self.file_list[item_idx]
        file_stem = Path(chunk_info['file_path']).stem
        cache_filename = f"{file_stem}_{chunk_info['start_sample']}_{self.feature_type}.pt"
        
        return self.cache_dir / cache_filename
    
    def _load_audio_chunk(self, chunk_info: Dict[str, Any]) -> torch.Tensor:
        """Load and preprocess audio chunk."""
        # Load audio
        audio, sr = torchaudio.load(
            chunk_info['file_path'],
            frame_offset=chunk_info['start_sample'],
            num_frames=chunk_info['chunk_samples']
        )
        
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
            target_duration=self.chunk_duration,
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
        chunk_info = self.file_list[idx]
        
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
            audio, sr = self._load_audio_chunk(chunk_info)
            
            # Extract features
            features = extract_features_for_corpus(
                audio=audio,
                sample_rate=sr,
                corpus='voxceleb1',
                feature_type=self.feature_type,
                to_db=True,
                include_deltas=False
            )
            
            # Convert to tensor
            features = torch.FloatTensor(features)
            
            # Get label
            if self.task == 'gender':
                label_str = chunk_info['gender']
            elif self.task == 'age':
                label_str = chunk_info['age_group']
            else:
                raise ValueError(f"Unknown task: {self.task}")
            
            label = self.label_encoder['label_to_idx'][label_str]
            label = torch.LongTensor([label])
            
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
                dummy_features = torch.zeros(224, 300)  # 224 mel-bins, ~300 frames for 3s
            elif self.feature_type == 'mfcc':
                dummy_features = torch.zeros(40, 300)   # 40 MFCC coefficients
            else:
                dummy_features = torch.zeros(257, 300)  # 257 frequency bins
            
            dummy_label = torch.LongTensor([0])
            
            return dummy_features, dummy_label
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for balanced training."""
        if self.task == 'gender':
            labels = [chunk['gender'] for chunk in self.file_list]
        elif self.task == 'age':
            labels = [chunk['age_group'] for chunk in self.file_list]
        else:
            raise ValueError(f"Unknown task: {self.task}")
        
        label_counts = pd.Series(labels).value_counts()
        total_samples = len(labels)
        
        # Calculate weights (inverse frequency)
        weights = {}
        for label in label_counts.index:
            weights[label] = total_samples / (len(label_counts) * label_counts[label])
        
        # Convert to tensor
        class_weights = torch.zeros(self.label_encoder['num_classes'])
        for label, weight in weights.items():
            idx = self.label_encoder['label_to_idx'][label]
            class_weights[idx] = weight
        
        return class_weights
    
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
        unique_speakers = list(self.metadata['speaker_id'].unique())
        
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

def create_voxceleb1_dataloaders(
    data_dir: str,
    metadata_file: str,
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
    Create train/val/test dataloaders for VoxCeleb1.
    
    Args:
        data_dir (str): Path to VoxCeleb1 audio files
        metadata_file (str): Path to metadata CSV file
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
    full_dataset = VoxCeleb1Dataset(
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
        # Filter metadata for this split
        split_metadata = full_dataset.metadata[
            full_dataset.metadata['speaker_id'].isin(speakers)
        ]
        
        # Create temporary metadata file
        temp_metadata_file = f"temp_{split_name}_metadata.csv"
        split_metadata.to_csv(temp_metadata_file, index=False)
        
        # Create dataset
        datasets[split_name] = VoxCeleb1Dataset(
            data_dir=data_dir,
            metadata_file=temp_metadata_file,
            feature_type=feature_type,
            task=task,
            **dataset_kwargs
        )
        
        # Clean up temporary file
        os.remove(temp_metadata_file)
    
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
        'corpus': 'voxceleb1',
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