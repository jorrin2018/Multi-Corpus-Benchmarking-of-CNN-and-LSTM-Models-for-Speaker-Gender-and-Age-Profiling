"""
Audio Processing Module
======================

This module implements the standardized audio preprocessing pipeline as specified 
in "Multi-Corpus Benchmarking of CNN & LSTM Models for Speaker Profiling".

The pipeline includes:
1. Silence removal using adaptive threshold (q=0.075)
2. Pre-emphasis filter: y[t] = x[t] - 0.97*x[t-1]
3. Butterworth low-pass filter (10th order, 4 kHz cutoff)
4. Energy normalization (Z-score per file)
5. Resampling to target sample rates

Authors: Jorge Jorrin-Coz et al., 2025
License: MIT
"""

import numpy as np
import librosa
import scipy.signal as signal
from typing import Tuple, Optional, Union
import warnings

def remove_silence(
    audio: np.ndarray, 
    sample_rate: int, 
    threshold: float = 0.075,
    hop_length: int = 512,
    frame_length: int = 2048
) -> np.ndarray:
    """
    Remove silent segments from audio using adaptive threshold.
    
    This function implements the silence removal as specified in the paper,
    using an adaptive threshold q=0.075 (optimal range 0.05-0.10).
    
    Args:
        audio (np.ndarray): Input audio signal
        sample_rate (int): Sample rate of the audio
        threshold (float): Silence threshold (default: 0.075)
        hop_length (int): Number of samples between successive frames
        frame_length (int): Length of the windowed signal after padding
        
    Returns:
        np.ndarray: Audio signal with silence removed
        
    Example:
        >>> audio, sr = librosa.load("speech.wav", sr=16000)
        >>> clean_audio = remove_silence(audio, sr, threshold=0.075)
    """
    try:
        # Compute short-time energy
        energy = librosa.feature.rms(
            y=audio, 
            frame_length=frame_length, 
            hop_length=hop_length,
            center=True
        )[0]
        
        # Adaptive threshold based on energy percentiles
        energy_percentile = np.percentile(energy, 30)  # 30th percentile as baseline
        adaptive_threshold = max(threshold, energy_percentile * 1.5)
        
        # Find frames above threshold
        frames_above_threshold = energy > adaptive_threshold
        
        # Convert frame indices to sample indices
        sample_indices = librosa.frames_to_samples(
            np.where(frames_above_threshold)[0], 
            hop_length=hop_length
        )
        
        if len(sample_indices) == 0:
            # If no frames above threshold, return original audio
            warnings.warn("No frames above silence threshold, returning original audio")
            return audio
            
        # Extract non-silent segments
        segments = []
        start_idx = sample_indices[0]
        
        for i in range(1, len(sample_indices)):
            if sample_indices[i] - sample_indices[i-1] > hop_length * 2:
                # Gap detected, save current segment
                end_idx = sample_indices[i-1] + hop_length
                if end_idx <= len(audio):
                    segments.append(audio[start_idx:end_idx])
                start_idx = sample_indices[i]
        
        # Add final segment
        end_idx = min(sample_indices[-1] + hop_length, len(audio))
        segments.append(audio[start_idx:end_idx])
        
        # Concatenate all segments
        if segments:
            clean_audio = np.concatenate(segments)
        else:
            clean_audio = audio
            
        return clean_audio
        
    except Exception as e:
        warnings.warn(f"Error in silence removal: {e}. Returning original audio.")
        return audio

def apply_preemphasis(
    audio: np.ndarray, 
    coeff: float = 0.97
) -> np.ndarray:
    """
    Apply pre-emphasis filter to compensate for -6 dB/octave spectral tilt.
    
    Implements the filter: y[t] = x[t] - coeff*x[t-1]
    As specified in the paper with coeff = 0.97.
    
    Args:
        audio (np.ndarray): Input audio signal
        coeff (float): Pre-emphasis coefficient (default: 0.97)
        
    Returns:
        np.ndarray: Pre-emphasized audio signal
        
    Example:
        >>> audio = np.random.randn(16000)
        >>> preemphasized = apply_preemphasis(audio, coeff=0.97)
    """
    if len(audio) == 0:
        return audio
        
    # Apply pre-emphasis filter
    emphasized = np.append(audio[0], audio[1:] - coeff * audio[:-1])
    return emphasized

def apply_butterworth_filter(
    audio: np.ndarray,
    sample_rate: int,
    cutoff_freq: int = 4000,
    order: int = 10,
    filter_type: str = 'low'
) -> np.ndarray:
    """
    Apply Butterworth low-pass filter as specified in the paper.
    
    Default settings: 10th order, 4 kHz cutoff frequency.
    
    Args:
        audio (np.ndarray): Input audio signal  
        sample_rate (int): Sample rate of the audio
        cutoff_freq (int): Cutoff frequency in Hz (default: 4000)
        order (int): Filter order (default: 10)
        filter_type (str): Type of filter ('low', 'high', 'band', 'stop')
        
    Returns:
        np.ndarray: Filtered audio signal
        
    Example:
        >>> filtered = apply_butterworth_filter(audio, 16000, cutoff_freq=4000, order=10)
    """
    try:
        # Normalize cutoff frequency (0 to 1, where 1 is Nyquist frequency)
        nyquist = sample_rate / 2
        normalized_cutoff = cutoff_freq / nyquist
        
        # Ensure cutoff frequency is valid
        if normalized_cutoff >= 1.0:
            warnings.warn(f"Cutoff frequency {cutoff_freq} Hz is >= Nyquist frequency {nyquist} Hz. Skipping filtering.")
            return audio
            
        # Design Butterworth filter
        sos = signal.butter(
            order, 
            normalized_cutoff, 
            btype=filter_type, 
            output='sos'
        )
        
        # Apply filter (using filtfilt for zero-phase filtering)
        filtered_audio = signal.sosfiltfilt(sos, audio)
        
        return filtered_audio
        
    except Exception as e:
        warnings.warn(f"Error in Butterworth filtering: {e}. Returning original audio.")
        return audio

def normalize_energy(
    audio: np.ndarray,
    method: str = 'zscore'
) -> np.ndarray:
    """
    Apply energy normalization to audio signal.
    
    Implements Z-score normalization: (x - μ) / σ per file as specified in the paper.
    
    Args:
        audio (np.ndarray): Input audio signal
        method (str): Normalization method ('zscore', 'minmax', 'rms')
        
    Returns:
        np.ndarray: Normalized audio signal
        
    Example:
        >>> normalized = normalize_energy(audio, method='zscore')
    """
    if len(audio) == 0:
        return audio
        
    if method == 'zscore':
        # Z-score normalization: (x - μ) / σ
        mean = np.mean(audio)
        std = np.std(audio)
        
        if std == 0:
            # Handle constant signals
            return audio - mean
        
        normalized = (audio - mean) / std
        
    elif method == 'minmax':
        # Min-max normalization to [-1, 1]
        min_val = np.min(audio)
        max_val = np.max(audio)
        
        if max_val == min_val:
            return np.zeros_like(audio)
            
        normalized = 2 * (audio - min_val) / (max_val - min_val) - 1
        
    elif method == 'rms':
        # RMS normalization
        rms = np.sqrt(np.mean(audio**2))
        
        if rms == 0:
            return audio
            
        normalized = audio / rms
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
        
    return normalized

def resample_audio(
    audio: np.ndarray,
    original_sr: int,
    target_sr: int
) -> np.ndarray:
    """
    Resample audio to target sample rate.
    
    Args:
        audio (np.ndarray): Input audio signal
        original_sr (int): Original sample rate
        target_sr (int): Target sample rate
        
    Returns:
        np.ndarray: Resampled audio signal
        
    Example:
        >>> resampled = resample_audio(audio, original_sr=44100, target_sr=16000)
    """
    if original_sr == target_sr:
        return audio
        
    try:
        # Use librosa for high-quality resampling
        resampled = librosa.resample(
            audio, 
            orig_sr=original_sr, 
            target_sr=target_sr,
            res_type='kaiser_best'  # High quality resampling
        )
        return resampled
        
    except Exception as e:
        warnings.warn(f"Error in resampling: {e}. Returning original audio.")
        return audio

def normalize_audio_length(
    audio: np.ndarray,
    target_length: int,
    method: str = 'pad_or_trim'
) -> np.ndarray:
    """
    Normalize audio to target length by padding or trimming.
    
    Args:
        audio (np.ndarray): Input audio signal
        target_length (int): Target length in samples  
        method (str): Method for normalization ('pad_or_trim', 'repeat', 'random_crop')
        
    Returns:
        np.ndarray: Audio normalized to target length
        
    Example:
        >>> normalized = normalize_audio_length(audio, target_length=48000)
    """
    current_length = len(audio)
    
    if current_length == target_length:
        return audio
        
    if method == 'pad_or_trim':
        if current_length < target_length:
            # Pad with zeros
            padding = target_length - current_length
            padded = np.pad(audio, (0, padding), mode='constant', constant_values=0)
            return padded
        else:
            # Trim from center
            start = (current_length - target_length) // 2
            trimmed = audio[start:start + target_length]
            return trimmed
            
    elif method == 'repeat':
        if current_length < target_length:
            # Repeat audio to fill target length
            repeats = target_length // current_length + 1
            repeated = np.tile(audio, repeats)
            return repeated[:target_length]
        else:
            # Trim randomly
            start = np.random.randint(0, current_length - target_length + 1)
            return audio[start:start + target_length]
            
    elif method == 'random_crop':
        if current_length >= target_length:
            # Random crop
            start = np.random.randint(0, current_length - target_length + 1)
            return audio[start:start + target_length]
        else:
            # Pad randomly
            total_padding = target_length - current_length
            left_padding = np.random.randint(0, total_padding + 1)
            right_padding = total_padding - left_padding
            return np.pad(audio, (left_padding, right_padding), mode='constant')
            
    else:
        raise ValueError(f"Unknown length normalization method: {method}")

def preprocess_audio(
    audio: np.ndarray,
    sample_rate: int,
    target_sr: Optional[int] = None,
    target_duration: Optional[float] = None,
    remove_silence_flag: bool = True,
    silence_threshold: float = 0.075,
    apply_preemphasis_flag: bool = True,
    preemphasis_coeff: float = 0.97,
    apply_filter: bool = True,
    filter_cutoff: int = 4000,
    filter_order: int = 10,
    normalize_energy_flag: bool = True,
    energy_method: str = 'zscore'
) -> Tuple[np.ndarray, int]:
    """
    Complete audio preprocessing pipeline as specified in the paper.
    
    Applies the full preprocessing chain:
    1. Silence removal (optional)
    2. Pre-emphasis filter (optional)  
    3. Butterworth low-pass filter (optional)
    4. Energy normalization (optional)
    5. Resampling (if target_sr specified)
    6. Length normalization (if target_duration specified)
    
    Args:
        audio (np.ndarray): Input audio signal
        sample_rate (int): Original sample rate
        target_sr (Optional[int]): Target sample rate for resampling
        target_duration (Optional[float]): Target duration in seconds
        remove_silence_flag (bool): Whether to remove silence
        silence_threshold (float): Threshold for silence removal
        apply_preemphasis_flag (bool): Whether to apply pre-emphasis
        preemphasis_coeff (float): Pre-emphasis coefficient
        apply_filter (bool): Whether to apply Butterworth filter
        filter_cutoff (int): Butterworth filter cutoff frequency
        filter_order (int): Butterworth filter order
        normalize_energy_flag (bool): Whether to normalize energy
        energy_method (str): Energy normalization method
        
    Returns:
        Tuple[np.ndarray, int]: Preprocessed audio and final sample rate
        
    Example:
        >>> processed_audio, final_sr = preprocess_audio(
        ...     audio, 44100, target_sr=16000, target_duration=3.0
        ... )
    """
    processed = audio.copy()
    current_sr = sample_rate
    
    # Step 1: Remove silence
    if remove_silence_flag:
        processed = remove_silence(
            processed, 
            current_sr, 
            threshold=silence_threshold
        )
    
    # Step 2: Apply pre-emphasis
    if apply_preemphasis_flag:
        processed = apply_preemphasis(processed, coeff=preemphasis_coeff)
    
    # Step 3: Apply Butterworth filter
    if apply_filter:
        processed = apply_butterworth_filter(
            processed,
            current_sr,
            cutoff_freq=filter_cutoff,
            order=filter_order
        )
    
    # Step 4: Resample if needed
    if target_sr is not None and target_sr != current_sr:
        processed = resample_audio(processed, current_sr, target_sr)
        current_sr = target_sr
    
    # Step 5: Normalize energy
    if normalize_energy_flag:
        processed = normalize_energy(processed, method=energy_method)
    
    # Step 6: Normalize length if needed
    if target_duration is not None:
        target_length = int(target_duration * current_sr)
        processed = normalize_audio_length(processed, target_length)
    
    return processed, current_sr 