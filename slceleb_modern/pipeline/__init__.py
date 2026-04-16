"""
Pipeline Module - Integrated video processing orchestration

This module provides the main pipeline that integrates:
- Face Detection (Phase 2): MediaPipe Face Mesh
- Face Recognition (Phase 3): InsightFace Buffalo_L
- Speaker Detection (Phase 4): Audio-Visual Correlation

Components:
- IntegratedPipeline: Main orchestrator for end-to-end processing
- FrameResult: Per-frame processing results
- VideoResults: Complete video processing results
"""

from .integrated_pipeline import IntegratedPipeline, FrameResult, VideoResults

__all__ = ['IntegratedPipeline', 'FrameResult', 'VideoResults']
