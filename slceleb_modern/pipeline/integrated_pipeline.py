"""
Integrated Pipeline for Celebrity Audio Extraction

This module orchestrates the complete video processing pipeline by integrating:
1. Face Detection & Tracking (MediaPipe Face Mesh - Phase 2)
2. Face Recognition (InsightFace Buffalo_L - Phase 3)  
3. Active Speaker Detection (Audio-Visual Correlation - Phase 4)

The pipeline processes videos end-to-end to extract audio segments where
specific celebrities (POIs - Persons of Interest) are speaking.

Key Features:
- End-to-end video processing
- Real-time or batch processing
- Configurable thresholds for each component
- Comprehensive logging and statistics
- Production-ready error handling

Example:
    >>> from slceleb_modern.pipeline import IntegratedPipeline
    >>> pipeline = IntegratedPipeline()
    >>> pipeline.load_poi_references(['poi1.jpg', 'poi2.jpg'])
    >>> results = pipeline.process_video('lecture.mp4')
    >>> pipeline.export_audio_segments(results, 'output_dir/')

Author: Research Team
Date: November 1, 2025
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
from collections import defaultdict
import json
import time
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import all modern components
from slceleb_modern.detection import MediaPipeFaceDetector
from slceleb_modern.recognition import ModernFaceRecognizer
from slceleb_modern.speaker import (
    LipTracker,
    AudioFeatureExtractor,
    AudioVisualCorrelator
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FrameResult:
    """Container for per-frame processing results"""
    
    frame_idx: int
    timestamp: float
    
    # Detection results
    faces_detected: int = 0
    face_bboxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    face_landmarks: List[np.ndarray] = field(default_factory=list)
    
    # Recognition results
    face_identities: List[str] = field(default_factory=list)  # "POI" or "Unknown"
    face_confidences: List[float] = field(default_factory=list)
    
    # Speaker detection results
    is_speaking: List[bool] = field(default_factory=list)
    speaking_confidences: List[float] = field(default_factory=list)
    
    # POI status (main output)
    poi_present: bool = False
    poi_speaking: bool = False
    poi_index: Optional[int] = None  # Index of POI face in lists
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'frame_idx': int(self.frame_idx),
            'timestamp': float(self.timestamp),
            'faces_detected': int(self.faces_detected),
            'poi_present': bool(self.poi_present),
            'poi_speaking': bool(self.poi_speaking),
            'face_identities': self.face_identities,
            'face_confidences': [float(c) for c in self.face_confidences],
            'is_speaking': [bool(s) for s in self.is_speaking],
            'speaking_confidences': [float(c) for c in self.speaking_confidences]
        }


@dataclass
class VideoResults:
    """Container for complete video processing results"""
    
    video_path: str
    total_frames: int
    duration: float
    fps: float
    
    frame_results: List[FrameResult] = field(default_factory=list)
    
    # Summary statistics
    frames_with_poi: int = 0
    frames_with_poi_speaking: int = 0
    total_processing_time: float = 0.0
    avg_processing_fps: float = 0.0
    
    # Speaking segments (start_time, end_time, confidence)
    speaking_segments: List[Tuple[float, float, float]] = field(default_factory=list)
    
    def compute_summary(self):
        """Compute summary statistics from frame results"""
        self.frames_with_poi = sum(1 for r in self.frame_results if r.poi_present)
        self.frames_with_poi_speaking = sum(1 for r in self.frame_results if r.poi_speaking)
        
        if self.total_processing_time > 0:
            self.avg_processing_fps = len(self.frame_results) / self.total_processing_time
        
        # Extract speaking segments (consecutive frames where POI is speaking)
        self.speaking_segments = []
        segment_start = None
        segment_start_idx = None
        segment_confidences = []
        
        for i, result in enumerate(self.frame_results):
            if result.poi_speaking:
                if segment_start is None:
                    segment_start = result.timestamp
                    segment_start_idx = i
                    segment_confidences = []
                if result.poi_index is not None and result.poi_index < len(result.speaking_confidences):
                    segment_confidences.append(result.speaking_confidences[result.poi_index])
            else:
                if segment_start is not None:
                    # End current segment - use previous frame's timestamp
                    segment_end = self.frame_results[i - 1].timestamp if i > 0 else segment_start
                    avg_confidence = np.mean(segment_confidences) if segment_confidences else 0.0
                    self.speaking_segments.append((segment_start, segment_end, float(avg_confidence)))
                    segment_start = None
                    segment_start_idx = None
                    segment_confidences = []
        
        # Close final segment if still open
        if segment_start is not None:
            segment_end = self.frame_results[-1].timestamp
            avg_confidence = np.mean(segment_confidences) if segment_confidences else 0.0
            self.speaking_segments.append((segment_start, segment_end, float(avg_confidence)))
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'video_path': self.video_path,
            'total_frames': self.total_frames,
            'duration': self.duration,
            'fps': self.fps,
            'frames_with_poi': self.frames_with_poi,
            'frames_with_poi_speaking': self.frames_with_poi_speaking,
            'total_processing_time': self.total_processing_time,
            'avg_processing_fps': self.avg_processing_fps,
            'speaking_segments': [
                {'start': s, 'end': e, 'confidence': c}
                for s, e, c in self.speaking_segments
            ],
            'frame_results': [r.to_dict() for r in self.frame_results]
        }


class IntegratedPipeline:
    """
    Integrated pipeline combining face detection, recognition, and speaker detection.
    
    This pipeline processes videos to identify frames where specific persons of
    interest (POIs) are present and actively speaking.
    
    Example:
        >>> pipeline = IntegratedPipeline()
        >>> pipeline.load_poi_references(['celebrity1.jpg', 'celebrity2.jpg'])
        >>> results = pipeline.process_video('lecture.mp4', max_frames=1000)
        >>> print(f"POI speaking: {results.frames_with_poi_speaking} frames")
    """
    
    def __init__(
        self,
        detection_model: str = 'mediapipe',
        recognition_model: str = 'buffalo_l',
        detection_confidence: float = 0.5,
        recognition_threshold: float = 0.252,
        speaking_threshold: float = 0.5,
        use_gpu: bool = True
    ):
        """
        Initialize integrated pipeline.
        
        Args:
            detection_model: Face detection model ('mediapipe')
            recognition_model: Face recognition model ('buffalo_l')
            detection_confidence: Confidence threshold for face detection
            recognition_threshold: Similarity threshold for face recognition
            speaking_threshold: Correlation threshold for speaker detection
            use_gpu: Whether to use GPU acceleration
        """
        self.detection_model = detection_model
        self.recognition_model = recognition_model
        self.detection_confidence = detection_confidence
        self.recognition_threshold = recognition_threshold
        self.speaking_threshold = speaking_threshold
        self.use_gpu = use_gpu
        
        # Initialize components
        logger.info("Initializing IntegratedPipeline components...")
        
        # Phase 2: Face Detection
        self.detector = MediaPipeFaceDetector(
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=0.5
        )
        logger.info(f"✓ Face Detector initialized ({detection_model})")
        
        # Phase 3: Face Recognition
        self.recognizer = ModernFaceRecognizer(
            model_name=recognition_model,
            similarity_metric='cosine',
            adaptive_threshold=True
        )
        # Set custom threshold if provided
        if recognition_threshold is not None:
            self.recognizer.set_threshold(recognition_threshold)
        logger.info(f"✓ Face Recognizer initialized ({recognition_model})")
        
        # Phase 4: Speaker Detection components
        self.lip_tracker = LipTracker(window_size=30, fps=30.0)
        self.audio_extractor = AudioFeatureExtractor(sr=16000, fps=30.0)
        self.correlator = AudioVisualCorrelator(
            window_size=30,
            speaking_threshold=speaking_threshold
        )
        logger.info(f"✓ Speaker Detection initialized")
        
        # State
        self.poi_loaded = False
        self.audio_loaded = False
        
        logger.info("IntegratedPipeline initialization complete!")
    
    def load_poi_references(self, image_paths: List[str]) -> bool:
        """
        Load POI (Person of Interest) reference images.
        
        Args:
            image_paths: List of paths to reference images
            
        Returns:
            True if successful
        """
        logger.info(f"Loading {len(image_paths)} POI reference images...")
        success = self.recognizer.load_reference_images(image_paths)
        
        if success:
            self.poi_loaded = True
            logger.info(f"✓ POI references loaded successfully")
        else:
            logger.error("✗ Failed to load POI references")
        
        return success
    
    def process_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        skip_frames: int = 0,
        show_progress: bool = True
    ) -> VideoResults:
        """
        Process video through full pipeline.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum frames to process (None = all)
            skip_frames: Number of frames to skip at start
            show_progress: Whether to show progress messages
            
        Returns:
            VideoResults with complete processing results
        """
        if not self.poi_loaded:
            logger.warning("No POI references loaded! All faces will be marked as Unknown.")
        
        logger.info(f"Processing video: {video_path}")
        
        # Load audio
        logger.info("Loading audio from video...")
        audio_success = self.audio_extractor.load_audio(video_path)
        if not audio_success:
            logger.warning("Failed to load audio - speaker detection will not work")
            self.audio_loaded = False
        else:
            self.audio_loaded = True
            logger.info(f"✓ Audio loaded: {self.audio_extractor.duration:.2f}s")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        # Get video properties
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_video_frames / fps if fps > 0 else 0
        
        if max_frames is None:
            max_frames = total_video_frames
        
        logger.info(f"Video: {total_video_frames} frames, {fps:.2f} FPS, {duration:.2f}s")
        logger.info(f"Processing: {max_frames} frames (skipping first {skip_frames})")
        
        # Initialize results
        results = VideoResults(
            video_path=video_path,
            total_frames=max_frames,
            duration=duration,
            fps=fps
        )
        
        # Process frames
        frame_idx = 0
        processed_count = 0
        start_time = time.time()
        
        # Skip initial frames
        for _ in range(skip_frames):
            cap.read()
            frame_idx += 1
        
        try:
            while frame_idx < (skip_frames + max_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                frame_result = self._process_frame(frame, frame_idx, fps)
                results.frame_results.append(frame_result)
                
                processed_count += 1
                frame_idx += 1
                
                # Progress reporting
                if show_progress and processed_count % 100 == 0:
                    elapsed = time.time() - start_time
                    current_fps = processed_count / elapsed
                    logger.info(f"Processed {processed_count}/{max_frames} frames "
                              f"({100*processed_count/max_frames:.1f}%) - "
                              f"{current_fps:.1f} FPS")
        
        finally:
            cap.release()
        
        # Compute summary
        results.total_processing_time = time.time() - start_time
        results.compute_summary()
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing Complete!")
        logger.info(f"{'='*80}")
        logger.info(f"Frames processed: {len(results.frame_results)}")
        logger.info(f"Processing time: {results.total_processing_time:.2f}s")
        logger.info(f"Processing FPS: {results.avg_processing_fps:.2f}")
        logger.info(f"Frames with POI: {results.frames_with_poi} "
                   f"({100*results.frames_with_poi/len(results.frame_results):.1f}%)")
        logger.info(f"Frames with POI speaking: {results.frames_with_poi_speaking} "
                   f"({100*results.frames_with_poi_speaking/len(results.frame_results):.1f}%)")
        logger.info(f"Speaking segments found: {len(results.speaking_segments)}")
        logger.info(f"{'='*80}\n")
        
        return results
    
    def _process_frame(self, frame: np.ndarray, frame_idx: int, fps: float) -> FrameResult:
        """
        Process single frame through all pipeline stages.
        
        Args:
            frame: Video frame (BGR)
            frame_idx: Frame index
            fps: Video FPS
            
        Returns:
            FrameResult with all processing outputs
        """
        timestamp = frame_idx / fps
        result = FrameResult(frame_idx=frame_idx, timestamp=timestamp)
        
        # Stage 1: Face Detection (Phase 2)
        detections = self.detector.detect(frame)
        result.faces_detected = len(detections)
        
        if result.faces_detected == 0:
            return result
        
        # Extract bboxes and landmarks
        for detection in detections:
            result.face_bboxes.append(detection.bbox)
            result.face_landmarks.append(detection.landmarks_2d)
        
        # Stage 2: Face Recognition (Phase 3)
        if self.poi_loaded:
            for bbox in result.face_bboxes:
                recognition = self.recognizer.recognize_face(frame, bbox)
                
                if recognition.is_poi:
                    result.face_identities.append(recognition.person_id if hasattr(recognition, 'person_id') else "POI")
                    result.face_confidences.append(recognition.confidence)
                    
                    # Mark POI present
                    if not result.poi_present:
                        result.poi_present = True
                        result.poi_index = len(result.face_identities) - 1
                else:
                    result.face_identities.append("Unknown")
                    result.face_confidences.append(0.0)
        else:
            # No POI loaded, mark all as unknown
            result.face_identities = ["Unknown"] * result.faces_detected
            result.face_confidences = [0.0] * result.faces_detected
        
        # Stage 3: Speaker Detection (Phase 4)
        if self.audio_loaded and result.faces_detected > 0:
            for i, landmarks in enumerate(result.face_landmarks):
                # Update lip tracker
                self.lip_tracker.update(frame_idx, landmarks)
                
                # Check if ready for correlation
                if self.lip_tracker.is_ready():
                    # Get lip features
                    lip_openings = self.lip_tracker.get_lip_opening_sequence()
                    motion_features = self.lip_tracker.get_motion_features()
                    
                    # Get audio features
                    start_frame = max(0, frame_idx - 29)
                    audio_seq = self.audio_extractor.get_amplitude_envelope_sequence(
                        start_frame, frame_idx
                    )
                    
                    # Get MFCC
                    mfcc_list = []
                    for f_idx in range(start_frame, frame_idx + 1):
                        feat = self.audio_extractor.extract_features_at_frame(f_idx)
                        mfcc_list.append(feat.mfcc)
                    mfcc_array = np.array(mfcc_list).T if mfcc_list else np.array([])
                    
                    # Correlate
                    lip_features = {
                        'openings': lip_openings,
                        'motion_energy': motion_features.motion_energy
                    }
                    
                    audio_features = {
                        'amplitudes': audio_seq,
                        'mfcc': mfcc_array,
                        'mfcc_energy': np.sum(np.var(mfcc_array, axis=1)) if mfcc_array.size > 0 else 0.0
                    }
                    
                    correlation = self.correlator.correlate(
                        lip_features, audio_features, frame_idx, timestamp
                    )
                    
                    result.is_speaking.append(correlation.is_speaking)
                    result.speaking_confidences.append(correlation.confidence)
                    
                    # Check if POI is speaking
                    if i == result.poi_index and correlation.is_speaking:
                        result.poi_speaking = True
                else:
                    # Not ready yet
                    result.is_speaking.append(False)
                    result.speaking_confidences.append(0.0)
        else:
            # No audio or no faces
            result.is_speaking = [False] * result.faces_detected
            result.speaking_confidences = [0.0] * result.faces_detected
        
        return result
    
    def export_results(self, results: VideoResults, output_path: str):
        """
        Export results to JSON file.
        
        Args:
            results: VideoResults to export
            output_path: Path to output JSON file
        """
        logger.info(f"Exporting results to: {output_path}")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)
        
        logger.info(f"✓ Results exported successfully")
    
    def get_stats(self) -> Dict:
        """Get pipeline statistics"""
        return {
            'poi_loaded': self.poi_loaded,
            'audio_loaded': self.audio_loaded,
            'detector': self.detector.get_stats(),
            'recognizer': {
                'model': self.recognition_model,
                'threshold': self.recognition_threshold,
                'references_loaded': self.recognizer.reference_embeddings is not None
            },
            'speaker_detection': {
                'lip_tracker': self.lip_tracker.get_stats(),
                'audio_extractor': self.audio_extractor.get_stats(),
                'correlator': self.correlator.get_stats()
            }
        }


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("Integrated Pipeline Test")
    print("="*80)
    
    # Initialize pipeline
    pipeline = IntegratedPipeline(
        detection_confidence=0.5,
        recognition_threshold=0.252,
        speaking_threshold=0.5
    )
    
    print("\n✓ Pipeline initialized successfully")
    print(f"\nPipeline configuration:")
    print(f"  Detection: MediaPipe Face Mesh")
    print(f"  Recognition: InsightFace Buffalo_L")
    print(f"  Speaker: Audio-Visual Correlation")
    
    print("\n✓ Integrated pipeline ready for video processing")
    print("\nTo process a video:")
    print("  pipeline.load_poi_references(['ref1.jpg', 'ref2.jpg'])")
    print("  results = pipeline.process_video('video.mp4')")
    print("  pipeline.export_results(results, 'results.json')")
    
    print("\n" + "="*80)
