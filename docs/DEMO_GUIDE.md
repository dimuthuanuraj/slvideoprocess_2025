# 🎬 Demo Visualization Guide

## Overview
The `demo_visualization.py` script creates an annotated video showing real-time face detection, POI recognition, and speaker tracking. Perfect for demonstrations and presentations!

## Features

### Visual Annotations
- ✅ **Face Detection**: Bounding boxes around detected faces
- ✅ **POI Recognition**: Green boxes for POI, gray for others
- ✅ **Speaking Detection**: Yellow highlight when POI is speaking
- ✅ **Facial Landmarks**: 478 MediaPipe landmarks visualization
- ✅ **Lip Region**: Red highlight on lips when speaking
- ✅ **Real-time Metrics**: FPS, detection stats, cache performance

### Color Coding
- 🟢 **Green Box**: POI detected (not speaking)
- 🟡 **Yellow Box**: POI speaking
- ⚪ **Gray Box**: Other person (not POI)
- 🔴 **Red**: Lip region when speaking
- 🟣 **Magenta**: Facial landmarks

## Quick Start

### 1. Activate Environment
```bash
conda activate slceleb_modern
cd /mnt/ricproject3/node5/SLCeleb_Videoprocess/slvideoprocess_2025
```

### 2. Basic Demo (300 frames)
```bash
python demo_visualization.py \
  --video "test_videos/sample video2.mp4" \
  --poi-dir images/pipe_test_persons \
  --output demo_output.mp4 \
  --max-frames 300
```

### 3. Full Video Demo
```bash
python demo_visualization.py \
  --video "test_videos/sample video2.mp4" \
  --poi-dir images/pipe_test_persons \
  --output demo_full.mp4
```

## Usage Options

### Basic Options
```bash
python demo_visualization.py \
  --video INPUT_VIDEO \
  --poi-dir POI_IMAGES_DIR \
  --output OUTPUT_VIDEO \
  [OPTIONS]
```

### Advanced Options

**Control Processing:**
```bash
--max-frames 500              # Process only first 500 frames (for quick demos)
--no-optimize                 # Use standard pipeline (slower, no caching)
--model buffalo_l             # Use larger model (more accurate, slower)
```

**Control Visualization:**
```bash
--no-landmarks                # Don't show facial landmarks (cleaner look)
--no-lip-region               # Don't highlight lip region
--no-metrics                  # Don't show metrics panel at bottom
```

## Example Commands

### Quick Demo (Fast, for testing)
```bash
python demo_visualization.py \
  --video input.mp4 \
  --poi-dir poi_faces/ \
  --output quick_demo.mp4 \
  --max-frames 200 \
  --no-landmarks
```

### Full Quality Demo (All features)
```bash
python demo_visualization.py \
  --video input.mp4 \
  --poi-dir poi_faces/ \
  --output full_demo.mp4 \
  --model buffalo_l
```

### Minimal Visualization (Clean, for presentations)
```bash
python demo_visualization.py \
  --video input.mp4 \
  --poi-dir poi_faces/ \
  --output minimal_demo.mp4 \
  --no-landmarks \
  --no-lip-region \
  --max-frames 500
```

### High-Performance Demo (With caching stats)
```bash
python demo_visualization.py \
  --video input.mp4 \
  --poi-dir poi_faces/ \
  --output performance_demo.mp4 \
  --max-frames 1000
```

## Output

### Video Output
The output video will include:
- ✅ Original video with overlays
- ✅ Bounding boxes with confidence scores
- ✅ Speaking status indicators
- ✅ Real-time performance metrics
- ✅ Statistics panel at bottom

### Console Output
```
INFO:Processing video...
INFO:Processed 1000/1000 frames (100%) - 34.8 FPS
INFO:✓ Demo video created: demo_output.mp4
INFO:  Processed 1000 frames in 28.7s (34.8 FPS)
INFO:  POI detected in 528 frames (52.8%)
INFO:  Speaking detected in 488 frames (48.8%)
```

## Performance

### Processing Speed
- **Optimized (default)**: ~35 FPS
- **Standard**: ~10 FPS
- **Video Creation**: Real-time speed

### Example Timing
- **300 frames**: ~10 seconds
- **1000 frames**: ~30 seconds
- **Full video (18,000 frames)**: ~10 minutes

## Metrics Panel

The bottom panel shows:
- **Frame number** and **current FPS**
- **POI detection status** (green if detected)
- **Speaking status** (yellow if speaking)
- **Cache hit rate** (if using optimized pipeline)
- **Total POI frames** count
- **Speaking frames** count

## Troubleshooting

### Common Issues

**1. Module not found error**
```bash
# Solution: Activate conda environment
conda activate slceleb_modern
```

**2. Video not created**
```bash
# Solution: Check ffmpeg is installed
which ffmpeg
# If not installed: conda install ffmpeg
```

**3. Slow processing**
```bash
# Solution: Use optimized pipeline (default)
# Or reduce frames
python demo_visualization.py ... --max-frames 200
```

**4. Out of memory**
```bash
# Solution: Process in smaller batches
python demo_visualization.py ... --max-frames 500
```

### Debug Mode
Add verbose logging:
```bash
# Set logging level to DEBUG
export PYTHONPATH=.
python -u demo_visualization.py ... 2>&1 | tee demo.log
```

## Tips for Best Results

### 1. Choose Good POI Images
- Use clear, front-facing photos
- Multiple angles recommended
- Good lighting
- No obstructions

### 2. Optimize Frame Count
- **Quick demos**: 200-300 frames (~10s)
- **Full demos**: 1000-2000 frames (~40-80s)  
- **Presentations**: 500 frames (~20s)

### 3. Video Resolution
- Works best with 720p-1080p
- Higher resolutions are supported but slower
- 4K may require more memory

### 4. For Presentations
Use minimal visualization for cleaner look:
```bash
python demo_visualization.py \
  --video presentation.mp4 \
  --poi-dir speaker_photos/ \
  --output presentation_demo.mp4 \
  --no-landmarks \
  --max-frames 500
```

## Example Output Description

When you play the output video, you'll see:

1. **Top Area**: Color-coded legend
2. **Main Video**: 
   - Bounding boxes around faces
   - Labels: "SPEAKING (0.85)" or "NOT POI"
   - Facial landmarks (optional)
   - Lip highlighting when speaking (optional)
3. **Bottom Panel**:
   - Current frame and FPS
   - Detection status
   - Speaking status
   - Cache statistics
   - Cumulative counts

## Advanced Usage

### Custom Thresholds
Modify thresholds in the script or pipeline:
```python
# In integrated_pipeline.py
detection_confidence = 0.5    # Face detection
recognition_threshold = 0.252  # POI recognition
speaking_threshold = 0.5       # Speaking detection
```

### Different Models
```bash
# Fast but less accurate
--model buffalo_s  # Default

# Slower but more accurate
--model buffalo_l
```

## Integration with Production Pipeline

The demo visualization uses the same pipeline as production:
```python
from slceleb_modern.pipeline import IntegratedPipeline

# Same pipeline used in production_run.py
pipeline = IntegratedPipeline()
results = pipeline.process_video(video_path)
```

Results are identical to production output, just with visual annotations added.

## Next Steps

After creating demo:
1. Review output video
2. Adjust parameters if needed
3. Use for presentations/documentation
4. Run full production pipeline for batch processing

For production batch processing, use:
```bash
python production_run.py \
  --video-list videos.txt \
  --poi-dir celebrities/ \
  --output-dir output/
```

---

**Created**: November 2, 2025  
**Version**: 2.1.0  
**Status**: ✅ Production Ready

For more information, see:
- `README_MODERN.md` - Modern pipeline overview
- `PRODUCTION_GUIDE.md` - Production usage guide
- `EXECUTIVE_SUMMARY.md` - Project summary
