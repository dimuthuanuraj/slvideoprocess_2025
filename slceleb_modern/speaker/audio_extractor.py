"""
Audio Feature Extractor for Active Speaker Detection

This module extracts audio features for correlation with lip motion.
Uses librosa for robust audio analysis.

Key features:
- MFCC (Mel-frequency cepstral coefficients) - standard for speech
- Mel-spectrogram - frequency content over time
- Amplitude envelope - energy over time
- Frame-level synchronization with video

Author: Research Team
Date: November 1, 2025
"""

import numpy as np
import librosa
import logging
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from scipy import signal

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AudioFeatures:
    """Container for extracted audio features at frame level"""
    
    frame_idx: int
    timestamp: float  # in seconds
    
    # Energy-based features
    amplitude_envelope: float  # RMS energy in this frame's window
    zero_crossing_rate: float  # How often signal crosses zero
    
    # Spectral features
    mfcc: np.ndarray  # MFCC coefficients (typically 13-20 values)
    mel_spectrogram: Optional[np.ndarray] = None  # Mel-frequency spectrogram
    spectral_centroid: float = 0.0  # Center of mass of spectrum
    
    # Voice activity
    is_voiced: bool = False  # Whether this frame contains speech
    voice_confidence: float = 0.0  # Confidence in voice activity
    
    def __repr__(self):
        return (f"AudioFeatures(frame={self.frame_idx}, "
                f"energy={self.amplitude_envelope:.3f}, "
                f"voiced={self.is_voiced}, "
                f"confidence={self.voice_confidence:.3f})")


class AudioFeatureExtractor:
    """
    Extract audio features synchronized with video frames.
    
    This class processes audio to extract features that can be correlated
    with lip motion for active speaker detection.
    
    Example:
        >>> extractor = AudioFeatureExtractor(sr=16000, fps=30)
        >>> extractor.load_audio("video.mp4")
        >>> features = extractor.extract_features_at_frame(frame_idx=100)
        >>> print(f"Energy: {features.amplitude_envelope:.3f}")
    """
    
    def __init__(
        self,
        sr: int = 16000,  # Sample rate (16kHz is standard for speech)
        fps: float = 30.0,  # Video frame rate
        n_mfcc: int = 13,  # Number of MFCC coefficients
        hop_length: int = 512,  # Hop length for STFT
        n_mels: int = 128,  # Number of mel bands
        frame_buffer_ms: int = 40  # Audio window for each video frame (ms)
    ):
        """
        Initialize audio feature extractor.
        
        Args:
            sr: Audio sample rate (Hz)
            fps: Video frame rate (frames per second)
            n_mfcc: Number of MFCC coefficients to extract
            hop_length: Hop length for short-time Fourier transform
            n_mels: Number of mel frequency bands
            frame_buffer_ms: Audio window size per video frame (ms)
        """
        self.sr = sr
        self.fps = fps
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.frame_buffer_ms = frame_buffer_ms
        
        # Audio data
        self.audio: Optional[np.ndarray] = None
        self.duration: float = 0.0
        
        # Pre-computed features (cached for efficiency)
        self._mfcc_cache: Optional[np.ndarray] = None
        self._mel_spec_cache: Optional[np.ndarray] = None
        self._rms_cache: Optional[np.ndarray] = None
        
        logger.info(f"AudioFeatureExtractor initialized: sr={sr}Hz, fps={fps}, "
                    f"n_mfcc={n_mfcc}, frame_buffer={frame_buffer_ms}ms")
    
    def load_audio(self, audio_path: str, offset: float = 0.0, duration: Optional[float] = None) -> bool:
        """
        Load audio from file.
        
        Args:
            audio_path: Path to audio file (or video file with audio)
            offset: Start time in seconds
            duration: Duration to load (None = full file)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Loading audio from: {audio_path}")
            self.audio, _ = librosa.load(
                audio_path,
                sr=self.sr,
                offset=offset,
                duration=duration,
                mono=True
            )
            self.duration = len(self.audio) / self.sr
            
            # Clear cache
            self._mfcc_cache = None
            self._mel_spec_cache = None
            self._rms_cache = None
            
            logger.info(f"Audio loaded: {self.duration:.2f}s, {len(self.audio)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            return False
    
    def load_audio_array(self, audio: np.ndarray, sr: int) -> bool:
        """
        Load audio from numpy array.
        
        Args:
            audio: Audio samples (1D array)
            sr: Sample rate of audio
            
        Returns:
            True if successful
        """
        try:
            # Resample if needed
            if sr != self.sr:
                logger.info(f"Resampling audio from {sr}Hz to {self.sr}Hz")
                self.audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr)
            else:
                self.audio = audio.copy()
            
            self.duration = len(self.audio) / self.sr
            
            # Clear cache
            self._mfcc_cache = None
            self._mel_spec_cache = None
            self._rms_cache = None
            
            logger.info(f"Audio array loaded: {self.duration:.2f}s, {len(self.audio)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load audio array: {e}")
            return False
    
    def _ensure_features_computed(self):
        """Pre-compute features for entire audio (cached)"""
        if self.audio is None:
            raise ValueError("No audio loaded")
        
        if self._mfcc_cache is None:
            logger.info("Computing audio features...")
            
            # MFCC
            self._mfcc_cache = librosa.feature.mfcc(
                y=self.audio,
                sr=self.sr,
                n_mfcc=self.n_mfcc,
                hop_length=self.hop_length
            )
            
            # Mel-spectrogram
            self._mel_spec_cache = librosa.feature.melspectrogram(
                y=self.audio,
                sr=self.sr,
                n_mels=self.n_mels,
                hop_length=self.hop_length
            )
            
            # RMS energy
            self._rms_cache = librosa.feature.rms(
                y=self.audio,
                hop_length=self.hop_length
            )[0]
            
            logger.info(f"Features computed: MFCC shape={self._mfcc_cache.shape}, "
                       f"Mel shape={self._mel_spec_cache.shape}")
    
    def extract_features_at_frame(self, frame_idx: int) -> AudioFeatures:
        """
        Extract audio features corresponding to a video frame.
        
        Args:
            frame_idx: Video frame index
            
        Returns:
            AudioFeatures for this frame
        """
        if self.audio is None:
            raise ValueError("No audio loaded")
        
        # Ensure features are computed
        self._ensure_features_computed()
        
        # Calculate timestamp
        timestamp = frame_idx / self.fps
        
        # Calculate audio sample range for this frame
        # Use a window centered on the frame time
        window_samples = int(self.frame_buffer_ms * self.sr / 1000)
        center_sample = int(timestamp * self.sr)
        start_sample = max(0, center_sample - window_samples // 2)
        end_sample = min(len(self.audio), center_sample + window_samples // 2)
        
        # Extract audio segment
        audio_segment = self.audio[start_sample:end_sample]
        
        # Convert sample position to spectrogram frame
        spec_frame_idx = int(timestamp * self.sr / self.hop_length)
        spec_frame_idx = min(spec_frame_idx, self._mfcc_cache.shape[1] - 1)
        
        # Extract MFCC
        mfcc = self._mfcc_cache[:, spec_frame_idx]
        
        # Extract mel-spectrogram column
        mel_spec = self._mel_spec_cache[:, spec_frame_idx]
        
        # Compute amplitude envelope (RMS)
        if len(audio_segment) > 0:
            amplitude_envelope = float(np.sqrt(np.mean(audio_segment ** 2)))
        else:
            amplitude_envelope = 0.0
        
        # Zero crossing rate
        if len(audio_segment) > 1:
            zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_segment)))) / 2
            zero_crossing_rate = float(zero_crossings / len(audio_segment))
        else:
            zero_crossing_rate = 0.0
        
        # Spectral centroid (center of mass of spectrum)
        spectral_centroid = float(np.sum(np.arange(len(mel_spec)) * mel_spec) / 
                                   (np.sum(mel_spec) + 1e-10))
        
        # Voice activity detection (simple energy-based)
        # Typical speech energy: > 0.02 RMS, ZCR in 0.1-0.5 range
        is_voiced = amplitude_envelope > 0.02 and 0.1 < zero_crossing_rate < 0.5
        voice_confidence = float(np.tanh(amplitude_envelope * 50))  # Normalize to 0-1
        
        features = AudioFeatures(
            frame_idx=frame_idx,
            timestamp=timestamp,
            amplitude_envelope=amplitude_envelope,
            zero_crossing_rate=zero_crossing_rate,
            mfcc=mfcc,
            mel_spectrogram=mel_spec,
            spectral_centroid=spectral_centroid,
            is_voiced=is_voiced,
            voice_confidence=voice_confidence
        )
        
        return features
    
    def extract_features_sequence(
        self,
        start_frame: int,
        end_frame: int
    ) -> List[AudioFeatures]:
        """
        Extract audio features for a sequence of frames.
        
        Args:
            start_frame: First frame index
            end_frame: Last frame index (inclusive)
            
        Returns:
            List of AudioFeatures
        """
        features_list = []
        for frame_idx in range(start_frame, end_frame + 1):
            features = self.extract_features_at_frame(frame_idx)
            features_list.append(features)
        return features_list
    
    def get_amplitude_envelope_sequence(
        self,
        start_frame: int,
        end_frame: int
    ) -> np.ndarray:
        """
        Get amplitude envelope time series.
        
        Args:
            start_frame: First frame index
            end_frame: Last frame index
            
        Returns:
            Array of amplitude values
        """
        features_list = self.extract_features_sequence(start_frame, end_frame)
        return np.array([f.amplitude_envelope for f in features_list])
    
    def get_voice_activity_sequence(
        self,
        start_frame: int,
        end_frame: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get voice activity detection sequence.
        
        Args:
            start_frame: First frame index
            end_frame: Last frame index
            
        Returns:
            Tuple of (is_voiced array, confidence array)
        """
        features_list = self.extract_features_sequence(start_frame, end_frame)
        is_voiced = np.array([f.is_voiced for f in features_list])
        confidence = np.array([f.voice_confidence for f in features_list])
        return is_voiced, confidence
    
    def get_stats(self) -> Dict:
        """Get extractor statistics"""
        if self.audio is None:
            return {'loaded': False}
        
        return {
            'loaded': True,
            'duration': self.duration,
            'samples': len(self.audio),
            'sample_rate': self.sr,
            'fps': self.fps,
            'total_frames': int(self.duration * self.fps),
            'n_mfcc': self.n_mfcc,
            'features_cached': self._mfcc_cache is not None
        }


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("Audio Feature Extractor Test")
    print("="*80)
    
    # Initialize extractor
    extractor = AudioFeatureExtractor(sr=16000, fps=30.0, n_mfcc=13)
    
    print(f"\n📊 Extractor Configuration:")
    print(f"   Sample rate: {extractor.sr} Hz")
    print(f"   Video FPS: {extractor.fps}")
    print(f"   MFCC coefficients: {extractor.n_mfcc}")
    print(f"   Frame buffer: {extractor.frame_buffer_ms} ms")
    
    # Generate synthetic audio (speech-like pattern)
    print(f"\n🧪 Generating synthetic audio...")
    duration = 2.0  # 2 seconds
    t = np.linspace(0, duration, int(16000 * duration))
    
    # Simulate speech: mixture of frequencies with amplitude modulation
    # Fundamental frequency around 150 Hz (typical male voice)
    # Formants at 500, 1500, 2500 Hz
    audio = (
        0.3 * np.sin(2 * np.pi * 150 * t) +  # F0
        0.2 * np.sin(2 * np.pi * 500 * t) +  # F1
        0.15 * np.sin(2 * np.pi * 1500 * t) +  # F2
        0.1 * np.sin(2 * np.pi * 2500 * t)    # F3
    )
    
    # Add amplitude modulation (syllable rate ~4 Hz)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)
    audio = audio * envelope
    
    # Add some noise
    audio += 0.05 * np.random.randn(len(audio))
    
    # Load audio
    success = extractor.load_audio_array(audio, sr=16000)
    if success:
        print(f"   ✅ Audio loaded successfully")
    else:
        print(f"   ❌ Failed to load audio")
        exit(1)
    
    # Extract features at various frames
    print(f"\n📈 Extracting features at sample frames:")
    test_frames = [0, 15, 30, 45, 59]
    
    for frame_idx in test_frames:
        features = extractor.extract_features_at_frame(frame_idx)
        print(f"\n   Frame {frame_idx} (t={features.timestamp:.3f}s):")
        print(f"      Energy: {features.amplitude_envelope:.4f}")
        print(f"      ZCR: {features.zero_crossing_rate:.4f}")
        print(f"      Voiced: {features.is_voiced} (confidence: {features.voice_confidence:.3f})")
        print(f"      MFCC shape: {features.mfcc.shape}")
        print(f"      MFCC[0:3]: [{features.mfcc[0]:.2f}, {features.mfcc[1]:.2f}, {features.mfcc[2]:.2f}]")
    
    # Extract sequence
    print(f"\n🎬 Extracting feature sequence (frames 0-59):")
    amplitude_seq = extractor.get_amplitude_envelope_sequence(0, 59)
    is_voiced, confidence = extractor.get_voice_activity_sequence(0, 59)
    
    print(f"   Amplitude envelope: mean={np.mean(amplitude_seq):.4f}, "
          f"std={np.std(amplitude_seq):.4f}")
    print(f"   Voice activity: {np.sum(is_voiced)}/60 frames ({100*np.mean(is_voiced):.1f}%)")
    print(f"   Mean confidence: {np.mean(confidence):.3f}")
    
    # Stats
    print(f"\n📊 Extractor Statistics:")
    stats = extractor.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")
    
    print("\n✅ Audio feature extraction test complete!")
    print("   Ready for correlation with lip motion in Phase 4")
