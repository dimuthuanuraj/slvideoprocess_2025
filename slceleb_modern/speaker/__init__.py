"""
Speaker Detection Module

This module handles active speaker detection using audio-visual correlation.
Replaces old SyncNet (2016) with modern MediaPipe-based approach.

Components:
- LipTracker: Extract and track lip motion features
- AudioFeatureExtractor: Extract audio features (MFCC, mel-spectrogram)
- AudioVisualCorrelator: Correlate lip motion with audio
- SpeakerValidator: Backward-compatible interface
"""

from .lip_tracker import LipTracker, LipState, LipMotionFeatures
from .audio_extractor import AudioFeatureExtractor, AudioFeatures
from .av_correlator import AudioVisualCorrelator, CorrelationResult

__all__ = [
    'LipTracker', 'LipState', 'LipMotionFeatures',
    'AudioFeatureExtractor', 'AudioFeatures',
    'AudioVisualCorrelator', 'CorrelationResult'
]
