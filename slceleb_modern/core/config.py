"""
Core Configuration Module
=========================

Centralized configuration management for the SLCeleb video processing pipeline.
Replaces the old common.py with a more flexible, modern approach.

Author: Research Team
Date: October 31, 2025
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import yaml


@dataclass
class PathConfig:
    """Configuration for file paths."""
    # Base directories
    video_base_dir: str = "./videos"
    image_base_dir: str = "./images"
    output_dir: str = "./output"
    temp_dir: str = "./temp"
    log_dir: str = "./logs"
    
    # Model paths
    landmark_model: str = "model/dlib/shape_predictor_68_face_landmarks.dat"  # Legacy, unused
    retinaface_model: str = "model/retinaface_model/mnet.25/mnet.25"  # Legacy
    syncnet_model: str = "model/syncnet_v2.model"
    
    def validate(self):
        """Create directories if they don't exist."""
        for dir_path in [self.output_dir, self.temp_dir, self.log_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


@dataclass
class MediaPipeConfig:
    """Configuration for MediaPipe Face Mesh."""
    static_image_mode: bool = False
    max_num_faces: int = 5
    refine_landmarks: bool = True
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    
    # Landmark extraction
    extract_lip_landmarks: bool = True
    extract_eye_landmarks: bool = True
    lip_expansion_factor: float = 1.5  # For lip region bounding box


@dataclass
class FaceRecognitionConfig:
    """Configuration for face recognition (InsightFace)."""
    # Model selection
    model_name: str = "buffalo_l"  # Options: buffalo_s, buffalo_l, antelopev2
    
    # Matching thresholds
    cosine_threshold: float = 0.8  # For cosine similarity (lower = stricter)
    distance_threshold: float = 1.24  # For euclidean distance (lower = stricter)
    
    # Processing
    use_gpu: bool = True
    gpu_id: int = 0
    batch_size: int = 32


@dataclass
class SpeakerDetectionConfig:
    """Configuration for active speaker detection."""
    # Detection method: 'syncnet', 'mediapipe_correlation', 'syncformer'
    method: str = 'mediapipe_correlation'
    
    # Confidence thresholds
    starting_confidence: float = 4.0  # Threshold to start speaking segment
    patient_confidence: float = 3.0   # Threshold to continue speaking
    
    # Audio-visual correlation parameters
    correlation_window: int = 25  # Frames to analyze for correlation
    min_segment_length: int = 5   # Minimum speaking segment length (frames)
    
    # Audio features
    audio_sample_rate: int = 16000
    audio_features: str = 'mfcc'  # Options: 'mfcc', 'mel', 'raw'


@dataclass
class ProcessingConfig:
    """Configuration for video processing."""
    # Video constraints
    target_fps: int = 25
    max_resolution: tuple = (1280, 720)
    resize_if_larger: bool = True
    
    # Processing options
    use_gpu: bool = True
    gpu_id: int = 0
    batch_size: int = 8
    num_workers: int = 4
    
    # Debugging
    debug: bool = False
    show_visualization: bool = False
    save_visualization: bool = False
    verbose: bool = True


@dataclass
class Config:
    """
    Main configuration class for SLCeleb video processing.
    
    Example:
        # From code
        config = Config(
            paths=PathConfig(video_base_dir='/path/to/videos'),
            processing=ProcessingConfig(debug=True)
        )
        
        # From YAML file
        config = Config.from_yaml('config.yaml')
    """
    paths: PathConfig = field(default_factory=PathConfig)
    mediapipe: MediaPipeConfig = field(default_factory=MediaPipeConfig)
    recognition: FaceRecognitionConfig = field(default_factory=FaceRecognitionConfig)
    speaker: SpeakerDetectionConfig = field(default_factory=SpeakerDetectionConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.paths.validate()
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """
        Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            Config object
        """
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls(
            paths=PathConfig(**data.get('paths', {})),
            mediapipe=MediaPipeConfig(**data.get('mediapipe', {})),
            recognition=FaceRecognitionConfig(**data.get('recognition', {})),
            speaker=SpeakerDetectionConfig(**data.get('speaker', {})),
            processing=ProcessingConfig(**data.get('processing', {}))
        )
    
    def to_yaml(self, yaml_path: str):
        """
        Save configuration to YAML file.
        
        Args:
            yaml_path: Path to save YAML configuration
        """
        data = {
            'paths': self.paths.__dict__,
            'mediapipe': self.mediapipe.__dict__,
            'recognition': self.recognition.__dict__,
            'speaker': self.speaker.__dict__,
            'processing': self.processing.__dict__
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'paths': self.paths.__dict__,
            'mediapipe': self.mediapipe.__dict__,
            'recognition': self.recognition.__dict__,
            'speaker': self.speaker.__dict__,
            'processing': self.processing.__dict__
        }
    
    def print_summary(self):
        """Print configuration summary."""
        print("=" * 60)
        print("Configuration Summary")
        print("=" * 60)
        
        print("\n[Paths]")
        print(f"  Video Base: {self.paths.video_base_dir}")
        print(f"  Image Base: {self.paths.image_base_dir}")
        print(f"  Output: {self.paths.output_dir}")
        
        print("\n[MediaPipe]")
        print(f"  Max Faces: {self.mediapipe.max_num_faces}")
        print(f"  Confidence: {self.mediapipe.min_detection_confidence}")
        
        print("\n[Face Recognition]")
        print(f"  Model: {self.recognition.model_name}")
        print(f"  Threshold: {self.recognition.cosine_threshold}")
        
        print("\n[Speaker Detection]")
        print(f"  Method: {self.speaker.method}")
        print(f"  Thresholds: {self.speaker.starting_confidence} / {self.speaker.patient_confidence}")
        
        print("\n[Processing]")
        print(f"  GPU: {self.processing.use_gpu} (ID: {self.processing.gpu_id})")
        print(f"  Debug: {self.processing.debug}")
        
        print("=" * 60)


# Default configuration instance
DEFAULT_CONFIG = Config()


if __name__ == "__main__":
    """Test configuration module."""
    
    # Create default config
    config = Config()
    config.print_summary()
    
    # Save to YAML
    config.to_yaml('config_example.yaml')
    print("\n✓ Saved to config_example.yaml")
    
    # Load from YAML
    loaded_config = Config.from_yaml('config_example.yaml')
    print("\n✓ Loaded from YAML successfully")
    
    # Modify and save
    loaded_config.processing.debug = True
    loaded_config.recognition.model_name = "antelopev2"
    loaded_config.to_yaml('config_modified.yaml')
    print("✓ Modified config saved")
