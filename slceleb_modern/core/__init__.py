"""Core module initialization."""

from .config import (
    Config,
    PathConfig,
    MediaPipeConfig,
    FaceRecognitionConfig,
    SpeakerDetectionConfig,
    ProcessingConfig,
    DEFAULT_CONFIG
)

__all__ = [
    'Config',
    'PathConfig',
    'MediaPipeConfig',
    'FaceRecognitionConfig',
    'SpeakerDetectionConfig',
    'ProcessingConfig',
    'DEFAULT_CONFIG'
]
