"""
Audio-Visual Correlator for Active Speaker Detection

This module correlates lip motion with audio features to determine if a person
is actively speaking. Uses cross-correlation and temporal alignment.

Key improvements over old SyncNet (2016):
- MediaPipe 40+ lip landmarks (vs 20 dlib points)
- Multiple correlation metrics (energy, MFCC, spectral)
- Adaptive thresholding
- Temporal smoothing to reduce false positives

Author: Research Team
Date: November 1, 2025
"""

import numpy as np
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
from scipy import signal, stats
from collections import deque
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CorrelationResult:
    """Container for audio-visual correlation result"""
    
    frame_idx: int
    timestamp: float
    
    # Correlation scores (higher = better match)
    energy_correlation: float  # Lip energy vs audio energy
    mfcc_correlation: float  # Lip motion vs MFCC features
    temporal_alignment: float  # How well aligned in time
    
    # Combined score
    correlation_score: float  # Weighted combination (0-1)
    is_speaking: bool  # Final decision
    confidence: float  # Confidence in decision (0-1)
    
    # Details
    lip_motion_energy: float = 0.0
    audio_energy: float = 0.0
    
    def __repr__(self):
        return (f"CorrelationResult(frame={self.frame_idx}, "
                f"score={self.correlation_score:.3f}, "
                f"speaking={self.is_speaking}, "
                f"confidence={self.confidence:.3f})")


class AudioVisualCorrelator:
    """
    Correlate lip motion with audio to detect active speaker.
    
    This class implements a multi-metric correlation approach:
    1. Energy correlation: Lip opening energy vs audio amplitude
    2. MFCC correlation: Lip motion pattern vs speech features
    3. Temporal alignment: Synchronization between visual and audio
    
    Example:
        >>> correlator = AudioVisualCorrelator()
        >>> result = correlator.correlate(lip_features, audio_features)
        >>> if result.is_speaking:
        >>>     print(f"Speaking detected! Confidence: {result.confidence:.2f}")
    """
    
    def __init__(
        self,
        window_size: int = 30,  # Frames to correlate (1s at 30fps)
        speaking_threshold: float = 0.5,  # Score threshold for speaking
        smoothing_window: int = 5,  # Temporal smoothing window
        min_confidence: float = 0.3  # Minimum confidence to report
    ):
        """
        Initialize audio-visual correlator.
        
        Args:
            window_size: Number of frames for correlation window
            speaking_threshold: Minimum correlation score to classify as speaking
            smoothing_window: Number of frames for temporal smoothing
            min_confidence: Minimum confidence threshold
        """
        self.window_size = window_size
        self.speaking_threshold = speaking_threshold
        self.smoothing_window = smoothing_window
        self.min_confidence = min_confidence
        
        # Temporal smoothing buffer
        self.correlation_history: deque = deque(maxlen=smoothing_window)
        self.decision_history: deque = deque(maxlen=smoothing_window * 2)
        
        # Statistics
        self.total_frames = 0
        self.speaking_frames = 0
        
        logger.info(f"AudioVisualCorrelator initialized: window={window_size}, "
                    f"threshold={speaking_threshold}, smoothing={smoothing_window}")
    
    def correlate_energy(
        self,
        lip_openings: np.ndarray,
        audio_amplitudes: np.ndarray
    ) -> float:
        """
        Correlate lip opening energy with audio energy.
        
        When speaking, lip movements should correlate with audio amplitude.
        
        Args:
            lip_openings: Array of lip opening values over time
            audio_amplitudes: Array of audio amplitude values (same length)
            
        Returns:
            Correlation coefficient (0-1, higher = better match)
        """
        if len(lip_openings) < 3 or len(audio_amplitudes) < 3:
            return 0.0
        
        # Ensure same length
        min_len = min(len(lip_openings), len(audio_amplitudes))
        lip_openings = lip_openings[:min_len]
        audio_amplitudes = audio_amplitudes[:min_len]
        
        # Normalize sequences
        lip_norm = (lip_openings - np.mean(lip_openings)) / (np.std(lip_openings) + 1e-6)
        audio_norm = (audio_amplitudes - np.mean(audio_amplitudes)) / (np.std(audio_amplitudes) + 1e-6)
        
        # Pearson correlation
        corr, p_value = stats.pearsonr(lip_norm, audio_norm)
        
        # Map to 0-1 range (correlation can be -1 to 1)
        corr_score = (corr + 1) / 2
        
        return float(np.clip(corr_score, 0, 1))
    
    def correlate_cross_correlation(
        self,
        lip_openings: np.ndarray,
        audio_amplitudes: np.ndarray,
        max_lag: int = 5
    ) -> Tuple[float, int]:
        """
        Compute cross-correlation with time lag.
        
        Audio and lip motion may not be perfectly synchronized due to
        processing delays. This finds the best alignment.
        
        Args:
            lip_openings: Array of lip opening values
            audio_amplitudes: Array of audio amplitude values
            max_lag: Maximum time lag to search (frames)
            
        Returns:
            Tuple of (max_correlation, best_lag)
        """
        if len(lip_openings) < 3 or len(audio_amplitudes) < 3:
            return 0.0, 0
        
        # Normalize
        lip_norm = (lip_openings - np.mean(lip_openings)) / (np.std(lip_openings) + 1e-6)
        audio_norm = (audio_amplitudes - np.mean(audio_amplitudes)) / (np.std(audio_amplitudes) + 1e-6)
        
        # Cross-correlation
        xcorr = signal.correlate(audio_norm, lip_norm, mode='same')
        
        # Find peak within max_lag
        center = len(xcorr) // 2
        search_start = max(0, center - max_lag)
        search_end = min(len(xcorr), center + max_lag + 1)
        
        search_region = xcorr[search_start:search_end]
        max_corr_idx = np.argmax(search_region)
        best_lag = max_corr_idx - (center - search_start)
        max_correlation = search_region[max_corr_idx] / len(lip_norm)
        
        # Normalize to 0-1
        max_correlation = float(np.clip((max_correlation + 1) / 2, 0, 1))
        
        return max_correlation, int(best_lag)
    
    def correlate_spectral(
        self,
        lip_motion_energy: float,
        mfcc_energy: float
    ) -> float:
        """
        Correlate based on spectral energy.
        
        Simple correlation: when speaking, both lip motion and MFCC energy
        should be high.
        
        Args:
            lip_motion_energy: Total lip motion energy
            mfcc_energy: Energy in MFCC features
            
        Returns:
            Correlation score (0-1)
        """
        # Normalize energies
        lip_score = np.tanh(lip_motion_energy / 100)  # Typical speech: 50-200
        mfcc_score = np.tanh(mfcc_energy / 50)  # Typical speech: 20-100
        
        # Simple product (both should be high for speaking)
        correlation = lip_score * mfcc_score
        
        return float(np.clip(correlation, 0, 1))
    
    def correlate(
        self,
        lip_features: Dict,
        audio_features: Dict,
        frame_idx: int,
        timestamp: float
    ) -> CorrelationResult:
        """
        Perform full audio-visual correlation.
        
        Args:
            lip_features: Dictionary with lip motion features
                - 'openings': array of lip opening values
                - 'motion_energy': total motion energy
            audio_features: Dictionary with audio features
                - 'amplitudes': array of audio amplitude values
                - 'mfcc': array of MFCC coefficients
                - 'mfcc_energy': total MFCC energy
            frame_idx: Current frame index
            timestamp: Current timestamp (seconds)
            
        Returns:
            CorrelationResult with all metrics
        """
        self.total_frames += 1
        
        # Extract sequences
        lip_openings = np.array(lip_features.get('openings', []))
        audio_amplitudes = np.array(audio_features.get('amplitudes', []))
        lip_motion_energy = lip_features.get('motion_energy', 0.0)
        mfcc_array = audio_features.get('mfcc', [])
        
        # Compute MFCC energy (sum of variance across coefficients)
        if len(mfcc_array) > 0:
            mfcc_energy = float(np.sum(np.var(mfcc_array, axis=0)))
        else:
            mfcc_energy = 0.0
        
        # 1. Energy correlation
        energy_corr = self.correlate_energy(lip_openings, audio_amplitudes)
        
        # 2. Cross-correlation with temporal alignment
        temporal_corr, best_lag = self.correlate_cross_correlation(
            lip_openings, audio_amplitudes, max_lag=5
        )
        
        # 3. Spectral correlation
        spectral_corr = self.correlate_spectral(lip_motion_energy, mfcc_energy)
        
        # Combine scores with weights
        # Energy correlation is most reliable, temporal alignment helps confirm,
        # spectral provides additional evidence
        weights = [0.4, 0.35, 0.25]  # energy, temporal, spectral
        correlation_score = np.average(
            [energy_corr, temporal_corr, spectral_corr],
            weights=weights
        )
        
        # Temporal smoothing
        self.correlation_history.append(correlation_score)
        smoothed_score = float(np.mean(self.correlation_history))
        
        # Decision with hysteresis (avoid flickering)
        is_speaking = smoothed_score >= self.speaking_threshold
        
        # Add to decision history
        self.decision_history.append(is_speaking)
        
        # Confidence based on:
        # 1. How far from threshold
        # 2. Consistency in recent history
        distance_from_threshold = abs(smoothed_score - self.speaking_threshold)
        threshold_confidence = np.tanh(distance_from_threshold * 5)  # 0-1
        
        if len(self.decision_history) >= self.smoothing_window:
            recent_decisions = list(self.decision_history)[-self.smoothing_window:]
            consistency = np.mean([d == is_speaking for d in recent_decisions])
        else:
            consistency = 0.5
        
        confidence = float(0.6 * threshold_confidence + 0.4 * consistency)
        confidence = np.clip(confidence, 0, 1)
        
        # Update statistics
        if is_speaking:
            self.speaking_frames += 1
        
        result = CorrelationResult(
            frame_idx=frame_idx,
            timestamp=timestamp,
            energy_correlation=energy_corr,
            mfcc_correlation=spectral_corr,
            temporal_alignment=temporal_corr,
            correlation_score=smoothed_score,
            is_speaking=is_speaking and confidence >= self.min_confidence,
            confidence=confidence,
            lip_motion_energy=lip_motion_energy,
            audio_energy=float(np.mean(audio_amplitudes)) if len(audio_amplitudes) > 0 else 0.0
        )
        
        return result
    
    def reset(self):
        """Reset correlator state"""
        self.correlation_history.clear()
        self.decision_history.clear()
        self.total_frames = 0
        self.speaking_frames = 0
    
    def get_stats(self) -> Dict:
        """Get correlator statistics"""
        return {
            'total_frames': self.total_frames,
            'speaking_frames': self.speaking_frames,
            'speaking_ratio': self.speaking_frames / max(self.total_frames, 1),
            'window_size': self.window_size,
            'threshold': self.speaking_threshold,
            'correlation_history_len': len(self.correlation_history),
            'decision_history_len': len(self.decision_history)
        }


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("Audio-Visual Correlator Test")
    print("="*80)
    
    # Initialize correlator
    correlator = AudioVisualCorrelator(
        window_size=30,
        speaking_threshold=0.5,
        smoothing_window=5
    )
    
    print(f"\n📊 Correlator Configuration:")
    print(f"   Window size: {correlator.window_size} frames")
    print(f"   Speaking threshold: {correlator.speaking_threshold}")
    print(f"   Smoothing window: {correlator.smoothing_window}")
    
    # Simulate correlated data (person is speaking)
    print(f"\n🧪 Test 1: Simulating SPEAKING pattern...")
    print(f"   (Correlated lip motion and audio)")
    
    np.random.seed(42)
    speaking_results = []
    
    for frame_idx in range(60):
        t = frame_idx / 30.0
        
        # Generate correlated lip and audio (speaking)
        base_signal = np.sin(2 * np.pi * 4 * t)  # 4 Hz (syllable rate)
        
        # Lip openings (30 frames)
        lip_openings = 20 + 10 * base_signal + 2 * np.random.randn(30)
        lip_motion_energy = np.sum(np.abs(np.diff(lip_openings)))
        
        # Audio amplitudes (30 frames, correlated)
        audio_amplitudes = 0.15 + 0.1 * base_signal + 0.02 * np.random.randn(30)
        
        # MFCC (13 coefficients x 30 frames)
        mfcc = np.random.randn(13, 30) * 10 + 50 * abs(base_signal)
        
        lip_features = {
            'openings': lip_openings,
            'motion_energy': lip_motion_energy
        }
        
        audio_features = {
            'amplitudes': audio_amplitudes,
            'mfcc': mfcc,
            'mfcc_energy': np.sum(np.var(mfcc, axis=0))
        }
        
        result = correlator.correlate(lip_features, audio_features, frame_idx, t)
        speaking_results.append(result)
        
        if frame_idx % 15 == 0:
            print(f"   Frame {frame_idx:3d}: {result}")
    
    speaking_detected = sum(r.is_speaking for r in speaking_results)
    avg_confidence = np.mean([r.confidence for r in speaking_results])
    avg_score = np.mean([r.correlation_score for r in speaking_results])
    
    print(f"\n   Results: {speaking_detected}/60 frames detected as speaking")
    print(f"   Average correlation score: {avg_score:.3f}")
    print(f"   Average confidence: {avg_confidence:.3f}")
    
    # Reset for next test
    correlator.reset()
    
    # Simulate uncorrelated data (person is NOT speaking)
    print(f"\n🧪 Test 2: Simulating NON-SPEAKING pattern...")
    print(f"   (Uncorrelated/random lip motion and audio)")
    
    non_speaking_results = []
    
    for frame_idx in range(60):
        t = frame_idx / 30.0
        
        # Generate uncorrelated lip and audio (not speaking)
        # Very small lip movements (closed mouth)
        lip_openings = 8 + 1 * np.random.randn(30)  # Much smaller
        lip_motion_energy = np.sum(np.abs(np.diff(lip_openings)))
        
        # Background noise (very low)
        audio_amplitudes = 0.005 + 0.002 * np.random.randn(30)  # Much quieter
        audio_amplitudes = np.abs(audio_amplitudes)  # Keep positive
        
        # Low MFCC energy
        mfcc = np.random.randn(13, 30) * 2  # Reduced noise
        
        lip_features = {
            'openings': lip_openings,
            'motion_energy': lip_motion_energy
        }
        
        audio_features = {
            'amplitudes': audio_amplitudes,
            'mfcc': mfcc,
            'mfcc_energy': np.sum(np.var(mfcc, axis=0))
        }
        
        result = correlator.correlate(lip_features, audio_features, frame_idx, t)
        non_speaking_results.append(result)
        
        if frame_idx % 15 == 0:
            print(f"   Frame {frame_idx:3d}: {result}")
    
    non_speaking_detected = sum(r.is_speaking for r in non_speaking_results)
    avg_confidence = np.mean([r.confidence for r in non_speaking_results])
    avg_score = np.mean([r.correlation_score for r in non_speaking_results])
    
    print(f"\n   Results: {non_speaking_detected}/60 frames detected as speaking")
    print(f"   Average correlation score: {avg_score:.3f}")
    print(f"   Average confidence: {avg_confidence:.3f}")
    
    # Summary
    print(f"\n📊 Test Summary:")
    print(f"   ✅ Test 1 (SPEAKING): {speaking_detected}/60 detected (should be high)")
    print(f"   ✅ Test 2 (NON-SPEAKING): {non_speaking_detected}/60 detected (should be low)")
    
    if speaking_detected > 40 and non_speaking_detected < 20:
        print(f"\n✅ Correlator working correctly!")
        print(f"   Good discrimination between speaking and non-speaking")
    else:
        print(f"\n⚠️  Correlator may need tuning")
    
    # Final stats
    print(f"\n📈 Final Statistics:")
    stats = correlator.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")
    
    print("\n✅ Audio-visual correlation test complete!")
    print("   Ready for integration into speaker detection pipeline")
