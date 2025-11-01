# SLCeleb Modern - Modular Architecture

## Overview

This directory contains the modernized, modular implementation of the SLCeleb video processing pipeline with state-of-the-art components.

## Directory Structure

```
slceleb_modern/
├── __init__.py                    # Package initialization
├── core/                          # Core configuration and utilities
│   ├── __init__.py
│   ├── config.py                  # Configuration management (NEW)
│   └── utils.py                   # Common utilities (TODO)
├── detection/                     # Face detection and tracking
│   ├── __init__.py
│   ├── face_detector.py          # MediaPipe Face Mesh wrapper (TODO)
│   └── face_tracker.py           # Face tracking across frames (TODO)
├── recognition/                   # Face recognition
│   ├── __init__.py
│   ├── face_recognizer.py        # InsightFace wrapper (TODO)
│   └── embedding_matcher.py      # Face matching logic (TODO)
├── speaker/                       # Active speaker detection
│   ├── __init__.py
│   ├── speaker_detector.py       # Active speaker detection (TODO)
│   └── audio_processor.py        # Audio feature extraction (TODO)
└── pipeline/                      # Main processing pipeline
    ├── __init__.py
    ├── video_processor.py        # Main pipeline orchestrator (TODO)
    └── video_clipper.py          # Video clipping and saving (TODO)
```

## Design Principles

### 1. Separation of Concerns
Each module handles a specific aspect of the pipeline:
- **Core**: Configuration and shared utilities
- **Detection**: Face detection and tracking
- **Recognition**: Face identification
- **Speaker**: Active speaker detection
- **Pipeline**: Main orchestration

### 2. Configuration-Driven
All settings are managed through `core/config.py`:
- Easy to modify without changing code
- Support for YAML configuration files
- Type-safe with dataclasses
- Validation built-in

### 3. Extensibility
Easy to add new features:
- Add new detection methods
- Swap recognition models
- Integrate new speaker detection algorithms

### 4. Testability
Modular design enables:
- Unit testing individual components
- Mock objects for testing
- Integration testing

## Usage

### Basic Usage

```python
from slceleb_modern import Config, VideoProcessor

# Create configuration
config = Config()
config.paths.video_base_dir = '/path/to/videos'
config.paths.image_base_dir = '/path/to/images'
config.processing.debug = True

# Process video
processor = VideoProcessor(config)
results = processor.process_video('celebrity_name', 'video.mp4')
```

### Using YAML Configuration

```python
from slceleb_modern import Config, VideoProcessor

# Load from YAML
config = Config.from_yaml('my_config.yaml')

# Process
processor = VideoProcessor(config)
results = processor.process_video('celebrity_name', 'video.mp4')
```

### Example YAML Configuration

```yaml
paths:
  video_base_dir: /data/videos
  image_base_dir: /data/images
  output_dir: /data/output
  
mediapipe:
  max_num_faces: 5
  min_detection_confidence: 0.5
  
recognition:
  model_name: buffalo_l
  cosine_threshold: 0.8
  
speaker:
  method: mediapipe_correlation
  starting_confidence: 4.0
  
processing:
  debug: false
  use_gpu: true
```

## Implementation Status

### ✅ Completed (Phase 1)
- [x] Package structure
- [x] Configuration management
- [x] Core module setup

### 🚧 In Progress
- [ ] Detection module (MediaPipe integration)
- [ ] Recognition module (InsightFace)
- [ ] Speaker detection module
- [ ] Pipeline orchestration

### 📋 Planned (Phase 2-4)
- [ ] Audio processing utilities
- [ ] Video clipping logic
- [ ] Batch processing
- [ ] Performance monitoring
- [ ] Comprehensive testing

## Migration from Old Code

The old code structure:
```
common.py              → core/config.py
face_detection.py      → detection/face_detector.py
face_validation.py     → recognition/face_recognizer.py
speaker_validation.py  → speaker/speaker_detector.py
cv_tracker.py          → detection/face_tracker.py
run.py                 → pipeline/video_processor.py
```

## Advantages Over Old Structure

1. **Better Organization**: Clear separation of concerns
2. **Easier Maintenance**: Locate and update specific functionality
3. **Flexible Configuration**: No hardcoded values
4. **Better Testing**: Each module can be tested independently
5. **Documentation**: Clear module purposes and interfaces
6. **Extensibility**: Easy to add new features without breaking existing code

## Development Guidelines

### Adding a New Module

1. Create the module file in appropriate directory
2. Import in `__init__.py`
3. Update `__all__` list
4. Add tests in `tests/` directory
5. Update this README

### Code Style

- Follow PEP 8
- Use type hints
- Add docstrings (Google style)
- Keep functions focused and small
- Use meaningful variable names

### Testing

```bash
# Run all tests
pytest tests/

# Run specific module tests
pytest tests/test_detection.py

# With coverage
pytest --cov=slceleb_modern tests/
```

## Dependencies

See `requirements.txt` in the root directory for all dependencies.

Key libraries:
- `mediapipe`: Face mesh detection
- `insightface`: Face recognition
- `torch`: Deep learning framework
- `opencv-python`: Image processing
- `librosa`: Audio processing

## Performance Considerations

- GPU acceleration enabled by default
- Batch processing for efficiency
- Caching of embeddings
- Optimized video I/O
- Memory management for large videos

## Future Enhancements

1. **Multi-GPU support**: Process multiple videos in parallel
2. **Cloud deployment**: AWS/GCP integration
3. **REST API**: Web service interface
4. **Real-time processing**: Streaming video support
5. **Model optimization**: ONNX export for faster inference

## Contributing

When contributing to this module:
1. Follow the existing code structure
2. Add tests for new features
3. Update documentation
4. Run linting and formatting
5. Create detailed commit messages

## Version History

- **2.0.0** (2025-10-31): Modern modular architecture
  - Refactored into clean modules
  - Added configuration system
  - Prepared for SOTA model integration

- **1.0.0** (2019): Original implementation
  - Monolithic structure
  - Hardcoded configurations
  - Legacy models

---

**Status**: Phase 1 Complete - Core structure established  
**Next**: Implement detection and recognition modules
