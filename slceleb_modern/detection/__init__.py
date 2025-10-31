"""
Face Detection Module
=====================

Modern face detection and tracking using MediaPipe Face Mesh.
Provides 478 3D landmarks for detailed facial analysis.
"""

from .face_detector import (
    MediaPipeFaceDetector,
    FaceDetection,
    detect_faces
)

__all__ = [
    'MediaPipeFaceDetector',
    'FaceDetection',
    'detect_faces',
]

__all__ = []
