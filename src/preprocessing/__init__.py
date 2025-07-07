"""
Audio Preprocessing Module
=========================

This module provides audio preprocessing functionality including:
- Silence removal using adaptive thresholds
- Pre-emphasis filtering
- Butterworth low-pass filtering  
- Energy normalization
- Resampling

Functions:
    preprocess_audio: Main preprocessing pipeline
    remove_silence: Remove silent segments from audio
    apply_preemphasis: Apply pre-emphasis filter
    apply_butterworth: Apply Butterworth low-pass filter
    normalize_energy: Apply z-score normalization
"""

from .audio_processing import preprocess_audio, remove_silence, apply_preemphasis, apply_butterworth, normalize_energy
from .feature_extraction import extract_mel_spectrogram, extract_mfcc, extract_linear_spectrogram

__all__ = [
    "preprocess_audio",
    "remove_silence", 
    "apply_preemphasis",
    "apply_butterworth",
    "normalize_energy",
    "extract_mel_spectrogram",
    "extract_mfcc", 
    "extract_linear_spectrogram"
] 