# SLCeleb Video Processing - User Guide

**Version:** 2.0  
**Last Updated:** April 16, 2026  
**Author:** SLCeleb Video Processing Team

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Quick Start](#quick-start)
6. [Step-by-Step Usage Guide](#step-by-step-usage-guide)
7. [Understanding Outputs](#understanding-outputs)
8. [Advanced Usage](#advanced-usage)
9. [Troubleshooting](#troubleshooting)
10. [FAQ](#faq)

---

## 📖 Overview

The SLCeleb Video Processing system is an advanced pipeline for detecting and recognizing persons of interest (POI) in video files using state-of-the-art face detection, face recognition, and speaker verification technologies.

### Key Features
- ✅ High-accuracy face detection using MediaPipe (478 facial landmarks)
- ✅ Robust face recognition with InsightFace buffalo_s (512D embeddings)
- ✅ Speaker verification with audio-visual correlation
- ✅ Batch processing with automatic error recovery
- ✅ GPU acceleration support
- ✅ Real-time progress tracking with checkpoints

---

## 🔧 Prerequisites

### System Requirements
- **Operating System:** Linux (Ubuntu 18.04+ or CentOS 7+)
- **Python:** 3.8 or 3.9
- **GPU:** NVIDIA GPU with CUDA support (recommended)
- **RAM:** Minimum 8GB, recommended 16GB+
- **Storage:** 10GB+ free space

### Required Software
- Conda or Miniconda
- CUDA Toolkit 11.x (for GPU acceleration)
- FFmpeg (for video processing)

---

## 🚀 Installation

### Step 1: Clone or Navigate to Project Directory

```bash
cd /mnt/ricproject3/2025/SLCeleb_VideoProcess/slvideoprocess_2025
```

### Step 2: Set Up Conda Environment

```bash
# Run the setup script
bash config/setup_conda_env.sh

# Or manually create environment
conda create -n slceleb python=3.8 -y
conda activate slceleb
```

### Step 3: Install Dependencies

```bash
# Install required packages
pip install -r config/requirements.txt
```

### Step 4: Verify GPU Setup (Optional but Recommended)

```bash
# Check if CUDA is available
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# If you encounter NVIDIA driver issues
bash config/fix_nvidia_driver.sh
```

### Step 5: Verify Installation

```bash
# Run a simple test
python tests/test_face_detector.py
```

---

## ⚙️ Configuration

### Step 1: Prepare Your Data

Organize your data structure as follows:

```
your_project/
 videos/                      # Directory containing video files
   ├── video1.mp4
   ├── video2.mp4
   └── video3.avi

 poi/                         # Person of Interest images
   ├── person1/
   │   ├── face1.jpg
   │   ├── face2.jpg
   │   └── face3.jpg
   └── person2/
       ├── face1.jpg
       └── face2.jpg

 output/                      # Output directory (will be created)
```

### Step 2: Create Video List (Optional)

If processing specific videos, create a text file:

```bash
# Create video list file
nano my_videos.txt
```

Add video paths (one per line):
```
/path/to/videos/video1.mp4
/path/to/videos/video2.mp4
/path/to/videos/video3.mp4
```

### Step 3: Configure Processing Settings

Edit \`common.py\` if you need to adjust default settings:
- Detection confidence thresholds
- Recognition similarity thresholds
- Frame sampling rates
- Output formats

---

## 🎯 Quick Start

### Process a Single Video

```bash
python production_run.py \\
    --video-list path/to/single_video.txt \\
    --poi-dir /path/to/poi \\
    --output-dir /path/to/output
```

### Process Multiple Videos (Batch Mode)

```bash
python production_run.py \\
    --video-dir /path/to/videos \\
    --poi-dir /path/to/poi \\
    --output-dir /path/to/output
```

### Resume Failed Processing

```bash
python production_run.py \\
    --resume outputs/batch_checkpoint.json
```

---

## 📝 Step-by-Step Usage Guide

### Scenario 1: Processing Videos from a Directory

**Step 1:** Prepare your videos
```bash
# Ensure all videos are in one directory
ls /path/to/videos/*.mp4
```

**Step 2:** Prepare POI images
```bash
# Create POI directory structure
mkdir -p /path/to/poi/person_name
# Add face images for each person
cp face_image.jpg /path/to/poi/person_name/
```

**Step 3:** Create output directory
```bash
mkdir -p /path/to/output
```

**Step 4:** Run the pipeline
```bash
cd /mnt/ricproject3/2025/SLCeleb_VideoProcess/slvideoprocess_2025

python production_run.py \\
    --video-dir /path/to/videos \\
    --poi-dir /path/to/poi \\
    --output-dir /path/to/output \\
    --log-level INFO
```

**Step 5:** Monitor progress
```bash
# In another terminal, watch the log file
tail -f production_run.log
```

**Step 6:** Check results
```bash
# List generated outputs
ls -lh /path/to/output/
```

---

### Scenario 2: Processing Specific Videos from a List

**Step 1:** Create video list
```bash
cat > my_videos.txt << EOL
/mnt/data/videos/interview1.mp4
/mnt/data/videos/meeting2.mp4
/mnt/data/videos/presentation3.mp4
EOL
```

**Step 2:** Run with video list
```bash
python production_run.py \\
    --video-list my_videos.txt \\
    --poi-dir /path/to/poi \\
    --output-dir /path/to/output
```

---

### Scenario 3: Processing with Per-Video POI Configuration

**Step 1:** Create POI mapping file
```bash
cat > poi_mapping.json << EOL
{
    "interview1.mp4": "/path/to/poi/executives",
    "meeting2.mp4": "/path/to/poi/team_members",
    "presentation3.mp4": "/path/to/poi/speakers"
}
EOL
```

**Step 2:** Run with POI mapping
```bash
python production_run.py \\
    --video-list my_videos.txt \\
    --poi-mapping poi_mapping.json \\
    --output-dir /path/to/output
```

---

### Scenario 4: Resuming After Interruption

If processing is interrupted (power failure, crash, etc.):

**Step 1:** Locate checkpoint file
```bash
ls outputs/batch_checkpoint.json
```

**Step 2:** Resume from checkpoint
```bash
python production_run.py --resume outputs/batch_checkpoint.json
```

The system will skip already processed videos and continue from where it stopped.

---

## 📊 Understanding Outputs

### Output Directory Structure

```
output/
 video1/
   ├── segments/                    # Extracted video segments
   │   ├── person1_segment_001.mp4
   │   ├── person1_segment_002.mp4
   │   └── person2_segment_001.mp4
   ├── metadata.json                # Processing metadata
   └── detection_results.json       # Detailed detection results

 video2/
   └── ... (same structure)

 processing_summary.json          # Overall summary
```

### Metadata Files

**\`metadata.json\`** - Processing information:
```json
{
    "video_name": "interview1.mp4",
    "duration_seconds": 300.5,
    "fps": 30,
    "total_frames": 9015,
    "processing_time": 45.2,
    "persons_detected": ["person1", "person2"]
}
```

**\`detection_results.json\`** - Frame-by-frame detections:
```json
{
    "frame_1200": {
        "timestamp": "00:00:40.0",
        "detections": [
            {
                "person": "person1",
                "confidence": 0.95,
                "bbox": [100, 150, 300, 400],
                "is_speaking": true
            }
        ]
    }
}
```

### Video Segments

Extracted segments are named: \`{person_name}_segment_{number}.mp4\`
- Each segment contains continuous appearance of the detected person
- Includes audio track if the person is speaking

---

## 🔬 Advanced Usage

### Custom Detection Thresholds

```bash
python production_run.py \\
    --video-dir /path/to/videos \\
    --poi-dir /path/to/poi \\
    --output-dir /path/to/output \\
    --face-confidence 0.7 \\
    --recognition-threshold 0.6
```

### GPU Selection

```bash
# Use specific GPU
CUDA_VISIBLE_DEVICES=1 python production_run.py \\
    --video-dir /path/to/videos \\
    --poi-dir /path/to/poi \\
    --output-dir /path/to/output
```

### Parallel Processing

```bash
# Process multiple videos in parallel (if you have multiple GPUs)
python production_run.py \\
    --video-dir /path/to/videos \\
    --poi-dir /path/to/poi \\
    --output-dir /path/to/output \\
    --num-workers 4
```

### Frame Sampling

```bash
# Process every 5th frame (faster but less accurate)
python production_run.py \\
    --video-dir /path/to/videos \\
    --poi-dir /path/to/poi \\
    --output-dir /path/to/output \\
    --frame-skip 5
```

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### Issue 1: CUDA Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
```bash
# 1. Reduce batch size
python production_run.py ... --batch-size 1

# 2. Process smaller videos first
# 3. Close other GPU applications

# 4. Check GPU memory
nvidia-smi
```

---

#### Issue 2: No Faces Detected

**Symptoms:**
- Empty output directories
- "No faces detected" in logs

**Solutions:**
1. Check video quality: Ensure faces are visible and clear
2. Adjust detection confidence:
   ```bash
   python production_run.py ... --face-confidence 0.5
   ```
3. Verify POI images:
   ```bash
   ls /path/to/poi/*/
   ```
4. Run face detector test:
   ```bash
   python tests/test_face_detector.py
   ```

---

#### Issue 3: Poor Recognition Accuracy

**Symptoms:**
- Wrong person detected
- False positives

**Solutions:**
1. Increase recognition threshold:
   ```bash
   python production_run.py ... --recognition-threshold 0.7
   ```
2. Add more POI reference images (minimum 3-5 per person)
3. Use high-quality POI images:
   - Front-facing
   - Good lighting
   - Clear facial features
   - Different expressions

---

#### Issue 4: Slow Processing

**Symptoms:**
- Processing takes very long

**Solutions:**
1. Enable GPU acceleration:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```
2. Increase frame skip:
   ```bash
   python production_run.py ... --frame-skip 3
   ```
3. Use faster detection mode:
   ```bash
   python production_run.py ... --fast-mode
   ```

---

#### Issue 5: Import Errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'slceleb_modern'
```

**Solutions:**
```bash
# 1. Ensure you're in the correct directory
cd /mnt/ricproject3/2025/SLCeleb_VideoProcess/slvideoprocess_2025

# 2. Activate conda environment
conda activate slceleb

# 3. Reinstall dependencies
pip install -r config/requirements.txt

# 4. Check Python path
python -c "import sys; print('\\n'.join(sys.path))"
```

---

#### Issue 6: Permission Denied

**Symptoms:**
```
PermissionError: [Errno 13] Permission denied
```

**Solutions:**
```bash
# 1. Check file permissions
ls -l /path/to/output

# 2. Create output directory with correct permissions
mkdir -p /path/to/output
chmod 755 /path/to/output

# 3. Run with appropriate user permissions
```

---

## ❓ FAQ

### Q1: How many POI images do I need?

**A:** Minimum 3-5 images per person. More images (10-15) provide better accuracy. Use images with:
- Different angles
- Different lighting conditions
- Different expressions
- High resolution (at least 224x224 pixels)

---

### Q2: What video formats are supported?

**A:** The system supports all formats that OpenCV can read:
- MP4, AVI, MOV, MKV, FLV, WMV
- Codecs: H.264, H.265, VP8, VP9

---

### Q3: Can I process live video streams?

**A:** Currently, the system is optimized for recorded videos. For live streams, you would need to modify the pipeline to accept RTSP/HTTP streams.

---

### Q4: How long does processing take?

**A:** Processing time depends on:
- Video length and resolution
- Number of POIs
- Hardware (GPU vs CPU)
- Frame skip rate

**Estimates:**
- 1-minute video @ 1080p with GPU: ~1-2 minutes
- 1-minute video @ 1080p with CPU: ~5-10 minutes

---

### Q5: Can I run this on CPU-only machines?

**A:** Yes, but it will be significantly slower. To run on CPU:
```bash
CUDA_VISIBLE_DEVICES="" python production_run.py ...
```

---

### Q6: How do I update the system?

**A:** 
```bash
# 1. Pull latest changes (if using git)
git pull origin main

# 2. Update dependencies
pip install -r config/requirements.txt --upgrade

# 3. Test installation
python tests/test_integrated_pipeline.py
```

---

### Q7: Can I use this for real-time applications?

**A:** The current implementation is optimized for batch processing. For real-time applications, you would need:
- Lower resolution input
- Aggressive frame skipping
- Powerful GPU
- Modified pipeline for streaming

---

### Q8: How do I add new POI during processing?

**A:** Currently, POIs must be defined before processing. To add new POI:
1. Stop the current process
2. Add POI images to the POI directory
3. Resume with \`--resume\` flag (already processed videos will be skipped)

---

## 📞 Support and Contact

### Getting Help

1. **Documentation:** Check \`docs/\` folder for detailed guides
2. **Logs:** Review \`production_run.log\` for error details
3. **Tests:** Run tests in \`tests/\` to diagnose issues
4. **Issues:** Check \`docs/NVIDIA_DRIVER_ISSUES.md\` for GPU problems

### Useful Commands

```bash
# Check system status
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

# View recent logs
tail -100 production_run.log

# Check GPU usage
watch -n 1 nvidia-smi

# Monitor disk space
df -h

# Check processing progress
ls -lh /path/to/output/
```

---

## 📚 Additional Resources

- **[README.md](README.md)** - Project overview
- **[PRODUCTION_GUIDE.md](PRODUCTION_GUIDE.md)** - Deployment guide
- **[DEMO_GUIDE.md](DEMO_GUIDE.md)** - Demo examples
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Directory structure
- **[OLD_VS_NEW_SCRIPTS.md](OLD_VS_NEW_SCRIPTS.md)** - Migration guide

---

## 📝 Changelog

### Version 2.0 (April 16, 2026)
- Initial user guide creation
- Reorganized project structure
- Added comprehensive troubleshooting section
- Updated installation instructions

---

**Document Version:** 1.0  
**Last Updated:** April 16, 2026  
**Maintained by:** SLCeleb Video Processing Team
