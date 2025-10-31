"""
Modern SLCeleb Video Processing - Modular Architecture
======================================================

This package provides a modular, maintainable architecture for celebrity
audio extraction from videos using state-of-the-art deep learning models.

Directory Structure:
-------------------
slceleb_modern/
├── __init__.py                 (this file)
├── core/
│   ├── __init__.py
│   ├── config.py              (Configuration management)
│   └── utils.py               (Common utilities)
├── detection/
│   ├── __init__.py
│   ├── face_detector.py       (MediaPipe Face Mesh wrapper)
│   └── face_tracker.py        (Face tracking across frames)
├── recognition/
│   ├── __init__.py
│   ├── face_recognizer.py     (InsightFace wrapper)
│   └── embedding_matcher.py   (Face matching logic)
├── speaker/
│   ├── __init__.py
│   ├── speaker_detector.py    (Active speaker detection)
│   └── audio_processor.py     (Audio feature extraction)
└── pipeline/
    ├── __init__.py
    ├── video_processor.py     (Main pipeline)
    └── video_clipper.py       (Video clipping and saving)

Usage:
------
    from slceleb_modern import VideoProcessor, Config
    
    config = Config(video_dir='path/to/video.mp4',
                   poi_images=['path/to/img1.jpg', 'path/to/img2.jpg'])
    
    processor = VideoProcessor(config)
    results = processor.process()

Author: Research Team
Date: October 31, 2025
Version: 2.0.0 (Modern SOTA Implementation)
"""

__version__ = "2.0.0"
__author__ = "Research Team"
__date__ = "2025-10-31"

# Import main components for easy access
from .core.config import Config

# VideoProcessor will be imported later when implemented
# from .pipeline.video_processor import VideoProcessor

__all__ = [
    'Config',
    # 'VideoProcessor',  # TODO: Implement in Phase 2
    '__version__',
    '__author__',
]

# Module information
MODULES = {
    'core': 'Configuration and utilities',
    'detection': 'Face detection and tracking (MediaPipe)',
    'recognition': 'Face recognition (InsightFace)',
    'speaker': 'Active speaker detection',
    'pipeline': 'Main processing pipeline'
}

def get_module_info():
    """Get information about available modules."""
    return MODULES

def print_info():
    """Print package information."""
    print(f"SLCeleb Video Processing - Modern Version {__version__}")
    print(f"Author: {__author__}")
    print(f"Date: {__date__}")
    print("\nAvailable Modules:")
    for module, description in MODULES.items():
        print(f"  - {module}: {description}")
