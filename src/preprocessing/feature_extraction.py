"""
Feature Extraction Module
=========================

This module implements feature extraction for audio signals as specified 
in "Multi-Corpus Benchmarking of CNN & LSTM Models for Speaker Profiling".

Supported features:
1. Linear spectrograms (STFT 25ms/10ms, n_fft=512, power=1)
2. Mel-spectrograms (dataset-specific bins: 224/128/64 + log scale)
3. MFCC (dataset-specific coefficients: 40/13/13)

Feature specifications per corpus:
- VoxCeleb1: 224 mel-bins, 40 MFCC
- Common Voice: 128 mel-bins, 13 MFCC  
- TIMIT: 64 mel-bins, 13 MFCC

Authors: Jorge Jorrin-Coz et al., 2025
License: MIT
"""

import numpy as np
import librosa
import librosa.display
from typing import Tuple, Optional, Union
import warnings

def extract_linear_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = 512,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: str = 'hann',
    power: float = 1.0
) -> np.ndarray:
    """
    Extract linear spectrogram using STFT as specified in the paper.
    
    Default parameters: STFT 25ms window / 10ms hop, n_fft=512, power=1
    
    Args:
        audio (np.ndarray): Input audio signal
        sample_rate (int): Sample rate of the audio
        n_fft (int): FFT window size (default: 512)
        hop_length (Optional[int]): Number of samples between successive frames
        win_length (Optional[int]): Window length in samples  
        window (str): Window function (default: 'hann')
        power (float): Exponent for magnitude spectrogram (default: 1.0)
        
    Returns:
        np.ndarray: Linear spectrogram with shape (n_fft//2 + 1, n_frames)
        
    Example:
        >>> spectrogram = extract_linear_spectrogram(audio, 16000)
        >>> print(spectrogram.shape)  # (257, n_frames)
    """
    # Set default hop_length and win_length based on paper specifications
    if hop_length is None:
        hop_length = int(sample_rate * 0.01)  # 10ms hop
    if win_length is None:
        win_length = int(sample_rate * 0.025)  # 25ms window
        
    try:
        # Compute STFT
        stft = librosa.stft(
            audio,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=True,
            pad_mode='constant'
        )
        
        # Convert to magnitude spectrogram
        magnitude = np.abs(stft) ** power
        
        return magnitude
        
    except Exception as e:
        warnings.warn(f"Error in linear spectrogram extraction: {e}")
        # Return empty spectrogram with correct shape
        n_frames = 1 + len(audio) // hop_length
        return np.zeros((n_fft // 2 + 1, n_frames))

def extract_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    n_mels: int = 128,
    n_fft: int = 512,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    power: float = 2.0,
    to_db: bool = True,
    ref: Union[float, str] = 'max'
) -> np.ndarray:
    """
    Extract mel-spectrogram with dataset-specific parameters.
    
    Corpus-specific n_mels values:
    - VoxCeleb1: 224 mel-bins
    - Common Voice: 128 mel-bins
    - TIMIT: 64 mel-bins
    
    Args:
        audio (np.ndarray): Input audio signal
        sample_rate (int): Sample rate of the audio
        n_mels (int): Number of mel bands (dataset-specific)
        n_fft (int): FFT window size (default: 512)
        hop_length (Optional[int]): Number of samples between frames
        win_length (Optional[int]): Window length in samples
        fmin (float): Minimum frequency (default: 0.0)
        fmax (Optional[float]): Maximum frequency (default: sr/2)
        power (float): Exponent for magnitude spectrogram (default: 2.0)
        to_db (bool): Convert to decibel scale (default: True)
        ref (Union[float, str]): Reference value for dB conversion
        
    Returns:
        np.ndarray: Mel-spectrogram with shape (n_mels, n_frames)
        
    Example:
        >>> # VoxCeleb1 configuration
        >>> mel_spec = extract_mel_spectrogram(audio, 16000, n_mels=224)
        >>> # Common Voice configuration  
        >>> mel_spec = extract_mel_spectrogram(audio, 22050, n_mels=128)
        >>> # TIMIT configuration
        >>> mel_spec = extract_mel_spectrogram(audio, 16000, n_mels=64)
    """
    # Set default parameters based on paper specifications
    if hop_length is None:
        hop_length = int(sample_rate * 0.01)  # 10ms hop
    if win_length is None:
        win_length = int(sample_rate * 0.025)  # 25ms window
    if fmax is None:
        fmax = sample_rate / 2
        
    try:
        # Compute mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            fmin=fmin,
            fmax=fmax,
            power=power,
            center=True,
            pad_mode='constant'
        )
        
        # Convert to decibel scale if requested
        if to_db:
            mel_spec_db = librosa.power_to_db(mel_spec, ref=ref)
            return mel_spec_db
        else:
            return mel_spec
            
    except Exception as e:
        warnings.warn(f"Error in mel-spectrogram extraction: {e}")
        # Return empty mel-spectrogram with correct shape
        n_frames = 1 + len(audio) // hop_length
        return np.zeros((n_mels, n_frames))

def extract_mfcc(
    audio: np.ndarray,
    sample_rate: int,
    n_mfcc: int = 13,
    n_mels: int = 128,
    n_fft: int = 512,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    dct_type: int = 2,
    lifter: int = 0,
    include_delta: bool = False,
    include_delta_delta: bool = False
) -> np.ndarray:
    """
    Extract MFCC features with dataset-specific parameters.
    
    Corpus-specific n_mfcc values:
    - VoxCeleb1: 40 coefficients
    - Common Voice: 13 coefficients
    - TIMIT: 13 coefficients
    
    Args:
        audio (np.ndarray): Input audio signal
        sample_rate (int): Sample rate of the audio
        n_mfcc (int): Number of MFCC coefficients (dataset-specific)
        n_mels (int): Number of mel bands for intermediate mel-spectrogram
        n_fft (int): FFT window size (default: 512)
        hop_length (Optional[int]): Number of samples between frames
        win_length (Optional[int]): Window length in samples
        fmin (float): Minimum frequency (default: 0.0)
        fmax (Optional[float]): Maximum frequency (default: sr/2)
        dct_type (int): Type of DCT (default: 2)
        lifter (int): Cepstral liftering parameter (default: 0)
        include_delta (bool): Include delta (first-order) features
        include_delta_delta (bool): Include delta-delta (second-order) features
        
    Returns:
        np.ndarray: MFCC features with shape (n_features, n_frames)
                   where n_features = n_mfcc * (1 + include_delta + include_delta_delta)
        
    Example:
        >>> # VoxCeleb1 configuration
        >>> mfcc = extract_mfcc(audio, 16000, n_mfcc=40)
        >>> # Common Voice / TIMIT configuration
        >>> mfcc = extract_mfcc(audio, 16000, n_mfcc=13)
        >>> # With delta features
        >>> mfcc_with_deltas = extract_mfcc(audio, 16000, n_mfcc=13, 
        ...                                include_delta=True, include_delta_delta=True)
    """
    # Set default parameters
    if hop_length is None:
        hop_length = int(sample_rate * 0.01)  # 10ms hop
    if win_length is None:
        win_length = int(sample_rate * 0.025)  # 25ms window
    if fmax is None:
        fmax = sample_rate / 2
        
    try:
        # Compute MFCC
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=n_mfcc,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            fmin=fmin,
            fmax=fmax,
            dct_type=dct_type,
            lifter=lifter,
            center=True
        )
        
        features = [mfcc]
        
        # Add delta features if requested
        if include_delta:
            delta = librosa.feature.delta(mfcc, order=1)
            features.append(delta)
            
        if include_delta_delta:
            delta_delta = librosa.feature.delta(mfcc, order=2)
            features.append(delta_delta)
        
        # Concatenate all features
        if len(features) > 1:
            combined_features = np.vstack(features)
            return combined_features
        else:
            return mfcc
            
    except Exception as e:
        warnings.warn(f"Error in MFCC extraction: {e}")
        # Return empty MFCC with correct shape
        n_frames = 1 + len(audio) // hop_length
        total_features = n_mfcc * (1 + include_delta + include_delta_delta)
        return np.zeros((total_features, n_frames))

def extract_features_for_corpus(
    audio: np.ndarray,
    sample_rate: int,
    corpus: str,
    feature_type: str = 'mel',
    to_db: bool = True,
    include_deltas: bool = False
) -> np.ndarray:
    """
    Extract features with corpus-specific parameters as defined in the paper.
    
    Automatically selects the correct parameters for each corpus:
    - VoxCeleb1: 224 mel-bins, 40 MFCC
    - Common Voice: 128 mel-bins, 13 MFCC
    - TIMIT: 64 mel-bins, 13 MFCC
    
    Args:
        audio (np.ndarray): Input audio signal
        sample_rate (int): Sample rate of the audio
        corpus (str): Corpus name ('voxceleb1', 'common_voice', 'timit')
        feature_type (str): Type of features ('mel', 'mfcc', 'linear')
        to_db (bool): Convert mel-spectrogram to dB scale (ignored for MFCC)
        include_deltas (bool): Include delta features for MFCC
        
    Returns:
        np.ndarray: Extracted features with corpus-specific parameters
        
    Raises:
        ValueError: If corpus or feature_type is not supported
        
    Example:
        >>> # Extract mel-spectrogram for VoxCeleb1
        >>> mel_features = extract_features_for_corpus(
        ...     audio, 16000, corpus='voxceleb1', feature_type='mel'
        ... )
        >>> # Extract MFCC for Common Voice
        >>> mfcc_features = extract_features_for_corpus(
        ...     audio, 22050, corpus='common_voice', feature_type='mfcc'
        ... )
    """
    # Corpus-specific parameters from the paper
    corpus_params = {
        'voxceleb1': {
            'n_mels': 224,
            'n_mfcc': 40
        },
        'common_voice': {
            'n_mels': 128,
            'n_mfcc': 13
        },
        'timit': {
            'n_mels': 64,
            'n_mfcc': 13
        }
    }
    
    # Normalize corpus name
    corpus = corpus.lower().replace(' ', '_').replace('-', '_')
    
    if corpus not in corpus_params:
        raise ValueError(f"Unsupported corpus: {corpus}. Supported: {list(corpus_params.keys())}")
    
    params = corpus_params[corpus]
    
    if feature_type == 'mel':
        return extract_mel_spectrogram(
            audio=audio,
            sample_rate=sample_rate,
            n_mels=params['n_mels'],
            to_db=to_db
        )
    elif feature_type == 'mfcc':
        return extract_mfcc(
            audio=audio,
            sample_rate=sample_rate,
            n_mfcc=params['n_mfcc'],
            n_mels=params['n_mels'],
            include_delta=include_deltas,
            include_delta_delta=include_deltas
        )
    elif feature_type == 'linear':
        return extract_linear_spectrogram(
            audio=audio,
            sample_rate=sample_rate
        )
    else:
        raise ValueError(f"Unsupported feature type: {feature_type}. Supported: ['mel', 'mfcc', 'linear']")

def normalize_features(
    features: np.ndarray,
    method: str = 'standard',
    axis: int = 1,
    epsilon: float = 1e-8
) -> np.ndarray:
    """
    Normalize extracted features.
    
    Args:
        features (np.ndarray): Input features with shape (n_features, n_frames)
        method (str): Normalization method ('standard', 'minmax', 'robust')
        axis (int): Axis along which to normalize (default: 1 for time axis)
        epsilon (float): Small value to avoid division by zero
        
    Returns:
        np.ndarray: Normalized features
        
    Example:
        >>> normalized = normalize_features(mfcc_features, method='standard')
    """
    if len(features) == 0:
        return features
        
    if method == 'standard':
        # Z-score normalization
        mean = np.mean(features, axis=axis, keepdims=True)
        std = np.std(features, axis=axis, keepdims=True)
        std = np.maximum(std, epsilon)  # Avoid division by zero
        normalized = (features - mean) / std
        
    elif method == 'minmax':
        # Min-max normalization to [0, 1]
        min_val = np.min(features, axis=axis, keepdims=True)
        max_val = np.max(features, axis=axis, keepdims=True)
        range_val = max_val - min_val
        range_val = np.maximum(range_val, epsilon)  # Avoid division by zero
        normalized = (features - min_val) / range_val
        
    elif method == 'robust':
        # Robust normalization using median and MAD
        median = np.median(features, axis=axis, keepdims=True)
        mad = np.median(np.abs(features - median), axis=axis, keepdims=True)
        mad = np.maximum(mad, epsilon)  # Avoid division by zero
        normalized = (features - median) / (1.4826 * mad)  # 1.4826 for normal distribution
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
        
    return normalized

def apply_feature_transforms(
    features: np.ndarray,
    transforms: list = None
) -> np.ndarray:
    """
    Apply various transformations to extracted features.
    
    Available transforms:
    - 'log': Apply logarithm (for power spectrograms)
    - 'sqrt': Apply square root
    - 'power': Apply power transformation
    - 'cmn': Cepstral mean normalization (for MFCC)
    
    Args:
        features (np.ndarray): Input features
        transforms (list): List of transform names to apply
        
    Returns:
        np.ndarray: Transformed features
        
    Example:
        >>> # Apply log transformation to mel-spectrogram
        >>> log_mel = apply_feature_transforms(mel_spec, ['log'])
        >>> # Apply CMN to MFCC
        >>> cmn_mfcc = apply_feature_transforms(mfcc, ['cmn'])
    """
    if transforms is None:
        return features
        
    transformed = features.copy()
    
    for transform in transforms:
        if transform == 'log':
            # Apply logarithm (add small value to avoid log(0))
            transformed = np.log(np.maximum(transformed, 1e-10))
            
        elif transform == 'sqrt':
            # Apply square root
            transformed = np.sqrt(np.maximum(transformed, 0))
            
        elif transform == 'power':
            # Apply power transformation (square)
            transformed = np.power(transformed, 2)
            
        elif transform == 'cmn':
            # Cepstral mean normalization (subtract mean across time)
            mean = np.mean(transformed, axis=1, keepdims=True)
            transformed = transformed - mean
            
        else:
            warnings.warn(f"Unknown transform: {transform}")
            
    return transformed

def convert_to_image_format(
    features: np.ndarray,
    target_height: int = 224,
    target_width: int = 224,
    channels: int = 1
) -> np.ndarray:
    """
    Convert features to image format for CNN input.
    
    Resizes features to target dimensions and adds channel dimension.
    
    Args:
        features (np.ndarray): Input features with shape (n_features, n_frames)
        target_height (int): Target height (default: 224 for ImageNet models)
        target_width (int): Target width (default: 224)
        channels (int): Number of channels (1 for grayscale, 3 for RGB)
        
    Returns:
        np.ndarray: Features resized to (target_height, target_width, channels)
        
    Example:
        >>> # Convert mel-spectrogram for CNN input
        >>> cnn_input = convert_to_image_format(mel_spec, target_height=224, target_width=224)
    """
    from scipy.ndimage import zoom
    
    if len(features.shape) != 2:
        raise ValueError(f"Expected 2D features, got shape {features.shape}")
    
    current_height, current_width = features.shape
    
    # Calculate zoom factors
    height_zoom = target_height / current_height
    width_zoom = target_width / current_width
    
    # Resize using interpolation
    resized = zoom(features, (height_zoom, width_zoom), order=1)
    
    # Add channel dimension
    if channels == 1:
        # Grayscale: add single channel
        resized = resized[:, :, np.newaxis]
    elif channels == 3:
        # RGB: repeat channel 3 times
        resized = np.repeat(resized[:, :, np.newaxis], 3, axis=2)
    else:
        raise ValueError(f"Unsupported number of channels: {channels}")
    
    return resized 