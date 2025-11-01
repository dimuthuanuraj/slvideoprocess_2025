"""
Face Recognition Module

Modern face recognition using InsightFace buffalo_l model (512D embeddings).
Replaces old MobileNet-based recognition (128D embeddings).
"""

from .face_recognizer import (
    ModernFaceRecognizer,
    FaceEmbedding,
    RecognitionResult
)

__all__ = [
    'ModernFaceRecognizer',
    'FaceEmbedding',
    'RecognitionResult'
]
