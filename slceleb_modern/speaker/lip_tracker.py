"""
MediaPipe-based Lip Tracker for Active Speaker Detection

This module extracts and tracks lip landmarks over time for audio-visual
correlation analysis. Uses MediaPipe's 478-landmark face mesh to get
detailed lip information (40+ points vs 20 in old dlib system).

Author: Research Team
Date: November 1, 2025
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from collections import deque
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LipState:
    """Container for lip state at a single time point"""
    
    frame_idx: int
    timestamp: float  # in seconds
    outer_lip_landmarks: np.ndarray  # Nx2 array of outer lip contour
    inner_lip_landmarks: Optional[np.ndarray] = None  # Nx2 array of inner lip contour
    lip_height: float = 0.0  # Vertical mouth opening
    lip_width: float = 0.0  # Horizontal mouth width
    lip_area: float = 0.0  # Area of mouth opening
    aspect_ratio: float = 0.0  # height/width ratio
    
    def __post_init__(self):
        """Calculate derived metrics"""
        if self.outer_lip_landmarks is not None and len(self.outer_lip_landmarks) > 0:
            # Calculate bounding box
            min_y, max_y = self.outer_lip_landmarks[:, 1].min(), self.outer_lip_landmarks[:, 1].max()
            min_x, max_x = self.outer_lip_landmarks[:, 0].min(), self.outer_lip_landmarks[:, 0].max()
            
            self.lip_height = max_y - min_y
            self.lip_width = max_x - min_x
            self.aspect_ratio = self.lip_height / (self.lip_width + 1e-6)
            
            # Estimate area (approximate as ellipse)
            self.lip_area = np.pi * (self.lip_height / 2) * (self.lip_width / 2)


@dataclass
class LipMotionFeatures:
    """Container for temporal lip motion features"""
    
    mean_lip_height: float = 0.0
    std_lip_height: float = 0.0
    max_lip_height: float = 0.0
    min_lip_height: float = 0.0
    
    mean_lip_opening_velocity: float = 0.0  # Rate of change
    max_lip_opening_velocity: float = 0.0
    
    mean_aspect_ratio: float = 0.0
    std_aspect_ratio: float = 0.0
    
    motion_energy: float = 0.0  # Total motion across window
    periodicity_score: float = 0.0  # How periodic the motion is (speaking has rhythm)
    
    num_frames: int = 0
    confidence: float = 0.0


class LipTracker:
    """
    Track lip landmarks over time and extract motion features.
    
    This class maintains a temporal window of lip states and computes
    features useful for active speaker detection, such as:
    - Lip opening/closing velocity
    - Motion energy
    - Periodicity (speaking has rhythmic patterns)
    
    Example:
        >>> tracker = LipTracker(window_size=30, fps=30)
        >>> for frame_idx, landmarks in enumerate(video_landmarks):
        >>>     tracker.update(frame_idx, landmarks)
        >>>     if tracker.is_ready():
        >>>         features = tracker.get_motion_features()
        >>>         print(f"Lip height: {features.mean_lip_height:.2f}")
    """
    
    def __init__(
        self,
        window_size: int = 30,  # Number of frames to track (1 second at 30fps)
        fps: float = 30.0,
        min_lip_height: float = 5.0  # Minimum pixels for valid lip detection
    ):
        """
        Initialize lip tracker.
        
        Args:
            window_size: Number of frames to keep in temporal window
            fps: Video frame rate (for timing calculations)
            min_lip_height: Minimum lip height in pixels to consider valid
        """
        self.window_size = window_size
        self.fps = fps
        self.min_lip_height = min_lip_height
        
        # Temporal window of lip states
        self.lip_states: deque[LipState] = deque(maxlen=window_size)
        
        # Statistics
        self.total_frames_processed = 0
        self.valid_frames = 0
        
        logger.info(f"LipTracker initialized: window={window_size} frames ({window_size/fps:.2f}s)")
    
    def update(
        self,
        frame_idx: int,
        landmarks: np.ndarray,
        timestamp: Optional[float] = None
    ) -> bool:
        """
        Update tracker with new lip landmarks.
        
        Args:
            frame_idx: Current frame index
            landmarks: MediaPipe landmarks (478x2 or 478x3 array)
            timestamp: Optional timestamp in seconds (computed from frame_idx if None)
            
        Returns:
            True if update successful, False if landmarks invalid
        """
        self.total_frames_processed += 1
        
        if timestamp is None:
            timestamp = frame_idx / self.fps
        
        # Extract lip landmarks from MediaPipe 478-point mesh
        # MediaPipe lip indices: outer (61 vertices), inner (40 vertices)
        outer_lip_indices = [
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0,
            267, 269, 270, 409, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78
        ]
        
        inner_lip_indices = [
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 191, 80, 81, 82
        ]
        
        try:
            # Extract outer lip landmarks
            if landmarks.shape[0] >= max(outer_lip_indices):
                outer_lips = landmarks[outer_lip_indices, :2]  # Only x, y
                inner_lips = landmarks[inner_lip_indices, :2] if landmarks.shape[0] >= max(inner_lip_indices) else None
                
                # Create lip state
                lip_state = LipState(
                    frame_idx=frame_idx,
                    timestamp=timestamp,
                    outer_lip_landmarks=outer_lips,
                    inner_lip_landmarks=inner_lips
                )
                
                # Validate lip detection
                if lip_state.lip_height >= self.min_lip_height:
                    self.lip_states.append(lip_state)
                    self.valid_frames += 1
                    return True
                else:
                    logger.debug(f"Frame {frame_idx}: Lip height too small ({lip_state.lip_height:.1f}px)")
                    return False
            else:
                logger.warning(f"Frame {frame_idx}: Insufficient landmarks")
                return False
                
        except Exception as e:
            logger.error(f"Frame {frame_idx}: Error extracting lip landmarks: {e}")
            return False
    
    def is_ready(self) -> bool:
        """Check if tracker has enough frames for feature extraction"""
        return len(self.lip_states) >= self.window_size // 2
    
    def get_current_lip_opening(self) -> float:
        """Get current lip opening (most recent frame)"""
        if len(self.lip_states) == 0:
            return 0.0
        return self.lip_states[-1].lip_height
    
    def get_lip_opening_sequence(self) -> np.ndarray:
        """Get time series of lip opening values"""
        if len(self.lip_states) == 0:
            return np.array([])
        return np.array([state.lip_height for state in self.lip_states])
    
    def get_motion_features(self) -> LipMotionFeatures:
        """
        Extract motion features from current window.
        
        Returns:
            LipMotionFeatures with temporal statistics
        """
        if not self.is_ready():
            logger.warning("Not enough frames for feature extraction")
            return LipMotionFeatures(num_frames=len(self.lip_states))
        
        # Extract time series
        lip_heights = np.array([state.lip_height for state in self.lip_states])
        aspect_ratios = np.array([state.aspect_ratio for state in self.lip_states])
        timestamps = np.array([state.timestamp for state in self.lip_states])
        
        # Compute velocity (rate of change)
        dt = np.diff(timestamps)
        dt[dt == 0] = 1e-6  # Avoid division by zero
        velocities = np.diff(lip_heights) / dt
        
        # Compute motion energy (total variation)
        motion_energy = np.sum(np.abs(velocities))
        
        # Compute periodicity using autocorrelation
        periodicity_score = self._compute_periodicity(lip_heights)
        
        # Create features
        features = LipMotionFeatures(
            mean_lip_height=float(np.mean(lip_heights)),
            std_lip_height=float(np.std(lip_heights)),
            max_lip_height=float(np.max(lip_heights)),
            min_lip_height=float(np.min(lip_heights)),
            
            mean_lip_opening_velocity=float(np.mean(np.abs(velocities))),
            max_lip_opening_velocity=float(np.max(np.abs(velocities))),
            
            mean_aspect_ratio=float(np.mean(aspect_ratios)),
            std_aspect_ratio=float(np.std(aspect_ratios)),
            
            motion_energy=float(motion_energy),
            periodicity_score=float(periodicity_score),
            
            num_frames=len(self.lip_states),
            confidence=float(self.valid_frames / max(self.total_frames_processed, 1))
        )
        
        return features
    
    def _compute_periodicity(self, signal: np.ndarray) -> float:
        """
        Compute periodicity score using autocorrelation.
        
        Speaking has rhythmic patterns (syllables, words) which show up
        as peaks in the autocorrelation function.
        
        Args:
            signal: Time series (e.g., lip heights)
            
        Returns:
            Periodicity score (0-1, higher = more periodic)
        """
        if len(signal) < 10:
            return 0.0
        
        # Normalize signal
        signal = signal - np.mean(signal)
        signal = signal / (np.std(signal) + 1e-6)
        
        # Compute autocorrelation
        autocorr = np.correlate(signal, signal, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Keep only positive lags
        
        # Normalize
        autocorr = autocorr / autocorr[0]
        
        # Look for peaks in typical speech frequency range
        # Syllables: 3-8 Hz, so at 30 fps, peaks at lags 4-10 frames
        min_lag = 4
        max_lag = min(15, len(autocorr))
        
        if max_lag <= min_lag:
            return 0.0
        
        # Find maximum peak in range
        peak_value = np.max(autocorr[min_lag:max_lag])
        
        return np.clip(peak_value, 0.0, 1.0)
    
    def compute_speaking_probability(self) -> float:
        """
        Compute probability that person is currently speaking.
        
        This is a heuristic based on lip motion features:
        - High motion energy
        - Moderate periodicity
        - Sufficient lip opening variation
        
        Returns:
            Probability (0-1) that person is speaking
        """
        if not self.is_ready():
            return 0.0
        
        features = self.get_motion_features()
        
        # Heuristic weights (these would ideally be learned from data)
        scores = []
        
        # Motion energy (normalized)
        # Speaking typically has energy > 50
        motion_score = np.tanh(features.motion_energy / 100.0)
        scores.append(motion_score)
        
        # Periodicity (speaking is somewhat periodic)
        # Sweet spot: 0.2-0.6 (too high might be repetitive motion, too low is random)
        periodicity_score = 1.0 - abs(features.periodicity_score - 0.4) / 0.4
        periodicity_score = np.clip(periodicity_score, 0, 1)
        scores.append(periodicity_score)
        
        # Variation in lip height (speaking has variation)
        # Typical std when speaking: 2-10 pixels
        variation_score = np.tanh(features.std_lip_height / 5.0)
        scores.append(variation_score)
        
        # Velocity (speaking has dynamic movement)
        velocity_score = np.tanh(features.mean_lip_opening_velocity / 2.0)
        scores.append(velocity_score)
        
        # Combine scores (weighted average)
        weights = [0.3, 0.2, 0.3, 0.2]  # Motion and variation most important
        probability = np.average(scores, weights=weights)
        
        return float(probability)
    
    def reset(self):
        """Reset tracker state"""
        self.lip_states.clear()
        self.total_frames_processed = 0
        self.valid_frames = 0
    
    def get_stats(self) -> Dict:
        """Get tracker statistics"""
        return {
            'total_frames': self.total_frames_processed,
            'valid_frames': self.valid_frames,
            'current_window_size': len(self.lip_states),
            'max_window_size': self.window_size,
            'validity_rate': self.valid_frames / max(self.total_frames_processed, 1),
            'is_ready': self.is_ready()
        }


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("MediaPipe Lip Tracker Test")
    print("="*80)
    
    # Initialize tracker
    tracker = LipTracker(window_size=30, fps=30.0)
    
    print(f"\n📊 Tracker Configuration:")
    print(f"   Window size: {tracker.window_size} frames ({tracker.window_size/tracker.fps:.2f}s)")
    print(f"   FPS: {tracker.fps}")
    print(f"   Min lip height: {tracker.min_lip_height}px")
    
    # Simulate some lip movements (speaking pattern)
    print(f"\n🧪 Simulating lip movements...")
    
    # Generate synthetic landmarks (478 points)
    for frame_idx in range(60):
        # Create synthetic 478-landmark array
        landmarks = np.random.rand(478, 2) * 100
        
        # Simulate speaking motion (periodic opening/closing)
        # Speaking frequency: ~4 syllables/second = 0.25s period
        t = frame_idx / 30.0
        lip_opening = 20 + 10 * np.sin(2 * np.pi * 4 * t)  # 4 Hz oscillation
        
        # Modify lip landmarks to simulate opening
        lip_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
        for idx in lip_indices:
            landmarks[idx, 1] += lip_opening
        
        # Update tracker
        success = tracker.update(frame_idx, landmarks)
        
        if frame_idx % 10 == 0:
            if tracker.is_ready():
                prob = tracker.compute_speaking_probability()
                features = tracker.get_motion_features()
                print(f"   Frame {frame_idx:3d}: Speaking prob = {prob:.3f}, "
                      f"Motion energy = {features.motion_energy:.1f}")
            else:
                print(f"   Frame {frame_idx:3d}: Warming up...")
    
    # Final stats
    print(f"\n📈 Final Statistics:")
    stats = tracker.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")
    
    print("\n✅ Lip tracker test complete!")
    print("   Ready for integration with audio features in Phase 4")
