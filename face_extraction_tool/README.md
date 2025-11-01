# 👤 Face Extraction & Clustering Tool

## Overview
Automatically extract faces from videos and organize them by person using intelligent face clustering. Perfect for building POI (Person of Interest) reference image datasets.

## Features

### Automatic Processing
- ✅ **Face Detection**: MediaPipe detection with 478 landmarks
- ✅ **Face Clustering**: Groups faces by person identity using embeddings
- ✅ **Quality Filtering**: Filters blurry, dark, or small faces
- ✅ **Duplicate Removal**: Keeps only best quality faces per person
- ✅ **Organized Output**: Saves faces in person-specific folders

### Quality Assessment
- **Sharpness**: Laplacian variance (blur detection)
- **Brightness**: Optimal lighting range
- **Size**: Larger faces preferred
- **Combined Score**: Weighted quality metric (0-1)

### Smart Clustering
- **Embedding-based**: Uses 512D InsightFace embeddings
- **Cosine Similarity**: Groups similar faces together
- **Configurable Threshold**: Adjust clustering strictness
- **Best Face Selection**: Saves highest quality faces only

## Quick Start

### Installation
```bash
cd /mnt/ricproject3/node5/SLCeleb_Videoprocess/slvideoprocess_2025/face_extraction_tool
conda activate slceleb_modern
```

### Basic Usage
```bash
# Extract faces from video
python extract_faces.py \
  --video ../test_videos/sample\ video2.mp4 \
  --output-dir extracted_faces
```

## Usage Examples

### 1. Quick Extraction (Default)
```bash
python extract_faces.py \
  --video input.mp4 \
  --output-dir faces_output
```
**Result**: Processes every 5th frame, saves up to 50 faces per person

### 2. High Quality Only
```bash
python extract_faces.py \
  --video input.mp4 \
  --output-dir high_quality_faces \
  --min-quality 0.5 \
  --min-face-size 100
```
**Result**: Only sharp, well-lit faces ≥100px

### 3. Fast Extraction (Fewer Faces)
```bash
python extract_faces.py \
  --video input.mp4 \
  --output-dir quick_extract \
  --skip-frames 10 \
  --max-faces-per-person 10
```
**Result**: Process every 10th frame, save 10 best faces per person

### 4. Strict Clustering (Fewer People)
```bash
python extract_faces.py \
  --video input.mp4 \
  --output-dir strict_cluster \
  --similarity-threshold 0.15
```
**Result**: Stricter matching = fewer people, more accurate grouping

### 5. Loose Clustering (More People)
```bash
python extract_faces.py \
  --video input.mp4 \
  --output-dir loose_cluster \
  --similarity-threshold 0.35
```
**Result**: Looser matching = more people, may split same person

### 6. Test Run (Limited Frames)
```bash
python extract_faces.py \
  --video input.mp4 \
  --output-dir test_run \
  --max-frames 500 \
  --max-faces-per-person 5
```
**Result**: Quick test on first 500 frames

## Command Options

### Required
```
--video PATH              Input video file
--output-dir PATH         Output directory for organized faces
```

### Processing Options
```
--skip-frames N           Process every Nth frame (default: 5)
                         Lower = more faces, slower
                         Higher = fewer faces, faster
                         
--max-frames N            Limit processing to N frames (for testing)
```

### Quality Filters
```
--min-quality FLOAT       Minimum quality score 0-1 (default: 0.3)
                         0.3 = balanced
                         0.5 = high quality only
                         0.2 = accept more faces
                         
--min-face-size INT       Minimum face dimension in pixels (default: 60)
                         Filters small/distant faces
```

### Clustering Options
```
--similarity-threshold    Face clustering threshold (default: 0.25)
  FLOAT                  Lower = stricter (fewer people)
                         Higher = looser (more people)
                         0.15 = very strict
                         0.25 = balanced (recommended)
                         0.35 = loose
                         
--max-faces-per-person    Maximum faces to save per person (default: 50)
  INT                    Saves best quality faces only
```

### Output Options
```
--no-preview             Don't create preview grids
```

## Output Structure

After extraction, you'll get:

```
extracted_faces/
├── person_000/                    # Most frequent person
│   ├── face_000_frame_000012_q0.85.jpg
│   ├── face_001_frame_000089_q0.82.jpg
│   ├── face_002_frame_000156_q0.79.jpg
│   ├── ...
│   └── preview_person_000.jpg     # Preview grid
├── person_001/                    # Second most frequent
│   ├── face_000_frame_000034_q0.91.jpg
│   ├── ...
│   └── preview_person_001.jpg
├── person_002/
│   └── ...
└── extraction_summary.json        # Detailed statistics
```

### File Naming
```
face_XXX_frame_YYYYYY_qZ.ZZ.jpg

XXX     = Face number (0-49)
YYYYYY  = Frame number in video
Z.ZZ    = Quality score (0.00-1.00)
```

### Preview Files
Each person folder contains a `preview_person_XXX.jpg` showing:
- Grid of up to 16 best faces
- Quality scores displayed
- Easy visual verification

### Summary JSON
```json
[
  {
    "person_id": 0,
    "total_faces": 287,
    "saved_faces": 50,
    "directory": "extracted_faces/person_000",
    "avg_quality": 0.64
  },
  ...
]
```

## Workflow: Extract → Review → Select POI

### Step 1: Extract Faces
```bash
python extract_faces.py \
  --video celebrity_interview.mp4 \
  --output-dir candidate_faces \
  --min-quality 0.4
```

### Step 2: Review Preview Files
```bash
# Open preview images to identify target person
eog candidate_faces/person_*/preview*.jpg
# Or use file browser
```

### Step 3: Select POI Folder
```bash
# Copy target person folder to POI directory
cp -r candidate_faces/person_003/ ../images/my_celebrity/

# Or create symbolic link
ln -s $(pwd)/candidate_faces/person_003 ../images/my_celebrity
```

### Step 4: Use in Production
```bash
cd ..
python production_run.py \
  --video-list videos.txt \
  --poi-dir images/my_celebrity \
  --output-dir output
```

## Tips & Best Practices

### For Best Results

**1. Adjust skip_frames based on video**
```bash
# Static camera/slow action
--skip-frames 10  # Process every 10th frame

# Dynamic camera/fast action  
--skip-frames 3   # Process every 3rd frame
```

**2. Set quality threshold appropriately**
```bash
# High quality training data
--min-quality 0.5 --min-face-size 100

# Maximum coverage (accept more faces)
--min-quality 0.2 --min-face-size 50
```

**3. Tune clustering for your use case**
```bash
# Single subject video (stricter)
--similarity-threshold 0.15

# Multi-person interview (balanced)
--similarity-threshold 0.25

# Crowd scene (looser)
--similarity-threshold 0.35
```

**4. Check preview files first**
- Always review `preview_person_XXX.jpg` files
- Verify faces are correctly grouped
- If one person split into multiple folders, increase threshold
- If multiple people in same folder, decrease threshold

### Performance

**Processing Speed**: ~20-40 FPS depending on:
- Frame skip rate
- Face count per frame
- GPU availability

**Example Timings**:
- 1000 frames, skip=5: ~1-2 minutes
- 5000 frames, skip=5: ~3-5 minutes
- 10000 frames, skip=10: ~3-5 minutes

### Common Issues

**Too many person folders (same person split)**
```bash
# Solution: Increase similarity threshold
--similarity-threshold 0.30
```

**Multiple people in same folder**
```bash
# Solution: Decrease similarity threshold
--similarity-threshold 0.20
```

**Not enough faces extracted**
```bash
# Solution: Lower quality requirements
--min-quality 0.2
--min-face-size 50
--skip-frames 3
```

**Too many low-quality faces**
```bash
# Solution: Raise quality requirements
--min-quality 0.5
--min-face-size 80
```

## Advanced Usage

### Custom Quality Thresholds
Edit `extract_faces.py` to adjust quality factors:
```python
quality = (
    blur_score * 0.4 +      # Sharpness weight
    brightness_score * 0.3 + # Lighting weight
    size_score * 0.3         # Size weight
)
```

### Batch Processing
```bash
#!/bin/bash
# Extract faces from multiple videos
for video in videos/*.mp4; do
    basename=$(basename "$video" .mp4)
    python extract_faces.py \
        --video "$video" \
        --output-dir "faces_batch/$basename" \
        --max-faces-per-person 20
done
```

### Integration with Pipeline
```python
# Use extracted faces as POI references
from pathlib import Path

poi_dir = "extracted_faces/person_003"  # Best person
poi_images = list(Path(poi_dir).glob("face_*.jpg"))

# Load in pipeline
pipeline.load_poi_references(poi_images)
```

## Comparison with Manual Selection

### Manual Method (Old Way)
1. ⏰ Play video and pause at good frames
2. 📸 Screenshot or export frames
3. ✂️ Manually crop faces
4. 📁 Organize into folders
5. ⏱️ **Time**: 1-2 hours per video

### Automatic Method (This Tool)
1. ▶️ Run extraction script
2. 👀 Review preview images
3. ✅ Select target person folder
4. ⏱️ **Time**: 2-5 minutes per video

**Speed**: **20-60x faster!**

## Output Quality Metrics

Quality scores indicate:
- **0.9-1.0**: Excellent (sharp, well-lit, large)
- **0.7-0.9**: Good (usable for training)
- **0.5-0.7**: Fair (acceptable)
- **0.3-0.5**: Poor (low quality)
- **<0.3**: Rejected (not saved)

## Integration with Main Pipeline

The extracted faces can be used directly as POI reference images:

```bash
# 1. Extract candidate faces
python face_extraction_tool/extract_faces.py \
  --video source_video.mp4 \
  --output-dir candidates

# 2. Review and identify target person (person_003)

# 3. Use in production pipeline
python production_run.py \
  --video-list videos.txt \
  --poi-dir candidates/person_003 \
  --output-dir results
```

## Troubleshooting

### No faces detected
- Check video quality
- Try lowering `--min-face-size`
- Ensure faces are visible (not masked/occluded)

### Script crashes
```bash
# Check conda environment
conda activate slceleb_modern

# Verify dependencies
python -c "import cv2, mediapipe, insightface; print('OK')"
```

### Slow processing
- Increase `--skip-frames` (process fewer frames)
- Use `--max-frames` for testing
- Ensure GPU is available

### Poor clustering
- Adjust `--similarity-threshold`
- Check preview files to understand grouping
- May need to manually merge/split folders

---

**Created**: November 2, 2025  
**Version**: 1.0.0  
**Status**: ✅ Ready to Use

**Next Steps**:
1. Extract faces from your videos
2. Review preview files
3. Select target person folders
4. Use as POI references in main pipeline

For questions or issues, check the main documentation:
- `README_MODERN.md` - Pipeline overview
- `PRODUCTION_GUIDE.md` - Production usage
- `DEMO_GUIDE.md` - Visualization demos
