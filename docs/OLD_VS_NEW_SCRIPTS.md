# Old Scripts vs New Implementation

## Summary
This document identifies legacy scripts from the 2019 implementation that are **NO LONGER USED** in the modern 2025 pipeline.

---

## 🔴 OBSOLETE - Old Scripts (2019 Technology)

### Main Processing Scripts
| Script | Purpose | Status | Replacement |
|--------|---------|--------|-------------|
| `run.py` | Main processing pipeline (449 lines) | ❌ **OBSOLETE** | `production_run.py` (717 lines) |
| `run_single.py` | Process single video | ❌ **OBSOLETE** | `production_run.py --video-list` |
| `common.py` | Configuration file | ❌ **OBSOLETE** | Modern config in each module |

**Why obsolete:**
- Uses TensorFlow 1.x / Keras
- Hard-coded paths and Windows-specific code
- No batch processing or error handling
- Uses deprecated CV2 trackers (BOOSTING, KCF, TLD, etc.)

---

### Component Scripts (Old)
| Script | Technology (2019) | Status | Replacement |
|--------|-------------------|--------|-------------|
| `face_detection.py` | RetinaFace (68 landmarks) | ❌ **OBSOLETE** | `slceleb_modern/detection/face_detector.py` (MediaPipe, 478 landmarks) |
| `face_validation.py` | InsightFace MobileNet (128D) | ❌ **OBSOLETE** | `slceleb_modern/recognition/face_recognizer.py` (InsightFace buffalo_s, 512D) |
| `speaker_validation.py` | SyncNet (2016) | ❌ **OBSOLETE** | `slceleb_modern/speaker/speaker_detector.py` (Audio-visual correlation) |
| `cv_tracker.py` | OpenCV trackers (162 lines) | ❌ **OBSOLETE** | MediaPipe face tracking (built-in) |

**Why obsolete:**
- Older models with lower accuracy
- No GPU optimization
- No caching mechanisms
- Deprecated dependencies

---

### Helper/Debug Scripts (Old)
| Script | Purpose | Status |
|--------|---------|--------|
| `evaluate.py` | Old evaluation logic | ❌ **OBSOLETE** |
| `audio_player.py` | Debug audio playback | ❌ **OBSOLETE** |
| `view_result.py` | Old result viewer | ❌ **OBSOLETE** |

---

## ✅ MODERN - New Scripts (2025 Technology)

### Main Production Scripts
| Script | Purpose | Lines | Technology |
|--------|---------|-------|------------|
| `production_run.py` | Production batch processor | 717 | Modern pipeline, checkpointing, error recovery |
| `slceleb_modern/pipeline/integrated_pipeline.py` | Unified modern pipeline | 587 | MediaPipe + InsightFace + Audio-visual |
| `slceleb_modern/recognition/face_recognizer_optimized.py` | Optimized recognizer | 319 | buffalo_s with caching (49% hit rate) |

### Component Modules (Modern)
```
slceleb_modern/
 detection/
   ├── face_detector.py          # MediaPipe (478 landmarks, 88 FPS)
   └── lip_tracker.py            # Lip motion tracking
 recognition/
   ├── face_recognizer.py        # InsightFace (512D embeddings)
   └── face_recognizer_optimized.py  # With caching (35.27 FPS)
 speaker/
   ├── audio_extractor.py        # MFCC + amplitude
   └── speaker_detector.py       # Audio-visual correlation
 pipeline/
    └── integrated_pipeline.py    # Orchestrator
```

### Testing & Benchmarking Scripts (Modern)
| Script | Purpose | Status |
|--------|---------|--------|
| `test_integrated_pipeline.py` | Integration tests | ✅ **ACTIVE** |
| `benchmark_old_vs_new.py` | Performance comparison | ✅ **ACTIVE** |
| `profile_performance.py` | Profiling | ✅ **ACTIVE** |
| `test_optimized.py` | Optimization tests | ✅ **ACTIVE** |

---

## 🔄 HYBRID - Scripts Used by Both

### Old Component Directories (Still Used by Old Scripts)
These directories contain 2019 technology but are referenced by old scripts:

| Directory | Technology | Used By | Status |
|-----------|------------|---------|--------|
| `RetinaFace/` | RetinaFace detector | Old `face_detection.py` | 🟡 Legacy, kept for reference |
| `FaceNet/` | FaceNet embeddings | Old `face_validation.py` | 🟡 Legacy, not used in modern pipeline |
| `SyncNet/` | SyncNet lip-sync | Old `speaker_validation.py` | 🟡 Legacy, replaced by audio-visual |
| `Speaker-Diarization/` | UIS-RNN diarization | Not integrated | 🟡 Optional component |
| `RetinaNet/` | Object detection | Not used | 🟡 Unused |

---

## 📊 Comparison: Old vs New

### Old Pipeline (run.py)
```python
# 2019 Implementation
run.py (449 lines)
 face_detection.py         # RetinaFace (68 pts)
 face_validation.py        # MobileNet (128D)
 speaker_validation.py     # SyncNet
 cv_tracker.py            # OpenCV trackers

Technology: TensorFlow 1.x, Keras, deprecated CV2
Performance: ~10 FPS
Features: Basic processing, no batch support
Error Handling: None
```

### New Pipeline (production_run.py)
```python
# 2025 Implementation  
production_run.py (717 lines)
 slceleb_modern/
    ├── detection/            # MediaPipe (478 pts, 88 FPS)
    ├── recognition/          # InsightFace (512D, 35 FPS)
    ├── speaker/              # Audio-visual correlation
    └── pipeline/             # Integrated orchestrator

Technology: Modern Python 3.10+, MediaPipe, InsightFace buffalo_s
Performance: 35.27 FPS (3.5x faster)
Features: Batch processing, checkpointing, segment merging, audio extraction
Error Handling: Comprehensive with resumption
```

---

## 🗑️ Safe to Remove (if not needed for reference)

### Scripts that can be deleted:
```bash
# Old main scripts
run.py
run_single.py
common.py

# Old components
face_detection.py
face_validation.py
speaker_validation.py
cv_tracker.py
evaluate.py
audio_player.py
view_result.py

# Legacy directories (if not needed)
FaceNet/          # Old face recognition
RetinaNet/        # Object detection (unused)
```

### Keep for reference/compatibility:
```bash
# Modern scripts
production_run.py
slceleb_modern/   # Entire modern implementation

# Testing/benchmarking
benchmark_*.py
test_*.py
profile_performance.py

# Documentation
README_MODERN.md
EXECUTIVE_SUMMARY.md
ResearchLog/
```

---

## 🚀 Migration Path

**To fully migrate to modern implementation:**

1. **Stop using:**
   - `run.py` → Use `production_run.py`
   - `run_single.py` → Use `production_run.py --video-list`
   - Old component scripts → Use `slceleb_modern/` modules

2. **Start using:**
   ```bash
   # Modern batch processing
   python production_run.py \
     --video-list videos.txt \
     --poi-dir poi_images/ \
     --output-dir output/ \
     --min-segment-duration 5.0 \
     --merge-gap 2.0
   ```

3. **Benefits:**
   - 3.5x faster (35.27 FPS vs 10 FPS)
   - Better accuracy (478 landmarks vs 68)
   - Larger embeddings (512D vs 128D)
   - Intelligent segment merging
   - Audio preservation
   - Batch processing with error recovery
   - Comprehensive logging and monitoring

---

## 
### Immediate Actions:
1. ✅ Use only `production_run.py` for all new work
2. ✅ All new features go into `slceleb_modern/`
3. ⚠️ Keep old scripts for reference/comparison only
4. ⚠️ Do NOT modify old scripts (they're legacy)

### Long-term:
1. Archive old scripts to `legacy/` directory
2. Remove unused dependencies from requirements.txt
3. Clean up old model directories if not needed
4. Update all documentation to reference only modern pipeline

---

**Last Updated:** November 2, 2025  
**Migration Status:** ✅ Complete (Modern pipeline is production-ready)

