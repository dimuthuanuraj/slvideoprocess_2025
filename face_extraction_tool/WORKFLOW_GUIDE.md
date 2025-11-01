# Face Extraction Workflow Guide

## Complete Workflow: Video → Organized Faces → POI Selection

```
┌──────────────────────────────────────────────────────────────┐
│                                                               │
│  STEP 1: EXTRACT FACES FROM VIDEO                           │
│                                                               │
│  Input:  celebrity_interview.mp4                            │
│  Output: extracted_faces/ with organized folders            │
│                                                               │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
        python extract_faces.py --video input.mp4 \
                               --output-dir extracted_faces
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  EXTRACTION PROCESS                                          │
│                                                               │
│  1. Video Processing                                         │
│     ├─ Frame reading (every 5th frame)                      │
│     ├─ Face detection (MediaPipe)                           │
│     └─ Quality assessment (blur, brightness, size)          │
│                                                               │
│  2. Feature Extraction                                       │
│     ├─ Face embedding (512D vector)                         │
│     └─ Landmarks detection (478 points)                     │
│                                                               │
│  3. Clustering                                               │
│     ├─ Similarity comparison (cosine distance)              │
│     ├─ Person grouping (automatic)                          │
│     └─ Quality ranking (best faces first)                   │
│                                                               │
│  4. Saving                                                   │
│     ├─ Create person folders (person_000, person_001, ...)  │
│     ├─ Save best faces per person (up to 50)                │
│     └─ Generate preview grids                                │
│                                                               │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  OUTPUT STRUCTURE                                            │
│                                                               │
│  extracted_faces/                                            │
│  ├── person_000/          ← Most frequent (Main speaker?)   │
│  │   ├── face_000_frame_000045_q0.87.jpg                    │
│  │   ├── face_001_frame_000123_q0.85.jpg                    │
│  │   ├── face_002_frame_000234_q0.83.jpg                    │
│  │   ├── ...                                                 │
│  │   └── preview_person_000.jpg  ← Visual grid             │
│  │                                                            │
│  ├── person_001/          ← Second most frequent            │
│  │   ├── face_000_frame_000067_q0.92.jpg                    │
│  │   ├── face_001_frame_000145_q0.89.jpg                    │
│  │   └── preview_person_001.jpg                             │
│  │                                                            │
│  ├── person_002/          ← Third most frequent             │
│  │   └── ...                                                 │
│  │                                                            │
│  └── extraction_summary.json  ← Statistics                  │
│                                                               │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                                                               │
│  STEP 2: REVIEW PREVIEW FILES                               │
│                                                               │
│  Open preview images to identify your target person:        │
│  $ eog extracted_faces/person_*/preview*.jpg                │
│                                                               │
│  Each preview shows:                                         │
│  ┌─────────────────────────────────────┐                   │
│  │ Person 0 - 16 faces                │                   │
│  ├─────┬─────┬─────┬─────┐             │                   │
│  │     │     │     │     │ Q:0.87      │                   │
│  │     │     │     │     │             │                   │
│  ├─────┼─────┼─────┼─────┤             │                   │
│  │ Q:0.85 Q:0.83 Q:0.81 Q:0.79 │       │                   │
│  │     │     │     │     │             │                   │
│  └─────┴─────┴─────┴─────┘             │                   │
│                                                               │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                                                               │
│  STEP 3: IDENTIFY TARGET PERSON                             │
│                                                               │
│  Example: person_000 is the main celebrity                  │
│                                                               │
│  $ ls extracted_faces/person_000/                           │
│  face_000_frame_000045_q0.87.jpg                            │
│  face_001_frame_000123_q0.85.jpg                            │
│  ...                                                          │
│  preview_person_000.jpg                                      │
│                                                               │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                                                               │
│  STEP 4: SELECT POI REFERENCE IMAGES                        │
│                                                               │
│  Option A: Copy entire folder                               │
│  $ cp -r extracted_faces/person_000 images/my_celebrity/    │
│                                                               │
│  Option B: Select best faces manually                       │
│  $ mkdir images/my_celebrity                                │
│  $ cp extracted_faces/person_000/face_00* images/my_celeb/ │
│                                                               │
│  Option C: Create symbolic link                             │
│  $ ln -s extracted_faces/person_000 images/my_celebrity     │
│                                                               │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                                                               │
│  STEP 5: USE IN PRODUCTION PIPELINE                         │
│                                                               │
│  python production_run.py \                                  │
│    --video-list videos_to_process.txt \                     │
│    --poi-dir images/my_celebrity \                          │
│    --output-dir final_results \                             │
│    --min-segment-duration 5.0 \                             │
│    --merge-gap 2.0                                           │
│                                                               │
│  The system will:                                            │
│  ├─ Use your extracted faces as POI references             │
│  ├─ Find the celebrity in all videos                        │
│  ├─ Detect when they're speaking                            │
│  └─ Extract speaking segments with audio                    │
│                                                               │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  FINAL OUTPUT                                                │
│                                                               │
│  final_results/                                              │
│  ├── video1/                                                 │
│  │   ├── video1_results.json                                │
│  │   ├── video1_segment_001.mp4  ← Celebrity speaking      │
│  │   ├── video1_segment_002.mp4                             │
│  │   └── ...                                                 │
│  ├── video2/                                                 │
│  │   └── ...                                                 │
│  └── batch_summary.json                                      │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

## Quick Reference Commands

### Extract Faces (Step 1)
```bash
python face_extraction_tool/extract_faces.py \
  --video source_video.mp4 \
  --output-dir extracted_faces \
  --min-quality 0.4 \
  --max-faces-per-person 30
```

### Review Previews (Step 2)
```bash
# View all preview grids
eog extracted_faces/person_*/preview*.jpg

# Or use file browser
nautilus extracted_faces/
```

### Select Target Person (Step 3 & 4)
```bash
# Copy target person folder
cp -r extracted_faces/person_003 images/target_celebrity/

# Verify copied faces
ls -lh images/target_celebrity/
```

### Run Production Pipeline (Step 5)
```bash
python production_run.py \
  --video-list videos.txt \
  --poi-dir images/target_celebrity \
  --output-dir results \
  --min-segment-duration 5.0
```

## Decision Guide: Extraction Parameters

### How many frames to process?
```
Short video (<5 min):     --skip-frames 3
Medium video (5-30 min):  --skip-frames 5  ← Default
Long video (>30 min):     --skip-frames 10
```

### How many faces per person?
```
Testing/preview:          --max-faces-per-person 10
General use:              --max-faces-per-person 30-50  ← Default
Training dataset:         --max-faces-per-person 100+
```

### Quality threshold?
```
High quality only:        --min-quality 0.5
Balanced (recommended):   --min-quality 0.3  ← Default
Maximum coverage:         --min-quality 0.2
```

### Clustering strictness?
```
Very strict (fewer people): --similarity-threshold 0.15
Balanced (recommended):     --similarity-threshold 0.25  ← Default
Loose (more people):        --similarity-threshold 0.35
```

## Common Scenarios

### Scenario 1: Single Person Interview
**Goal**: Extract faces of one main speaker

```bash
python extract_faces.py \
  --video interview.mp4 \
  --output-dir interview_faces \
  --similarity-threshold 0.20 \
  --max-faces-per-person 30 \
  --min-quality 0.4
```

**Result**: 1-2 person folders, high quality faces

### Scenario 2: Multi-Person Discussion
**Goal**: Extract all participants

```bash
python extract_faces.py \
  --video discussion.mp4 \
  --output-dir discussion_faces \
  --similarity-threshold 0.30 \
  --max-faces-per-person 20 \
  --skip-frames 5
```

**Result**: 3-5 person folders (one per participant)

### Scenario 3: Crowd/Event Video
**Goal**: Find specific person in crowd

```bash
python extract_faces.py \
  --video event.mp4 \
  --output-dir event_faces \
  --similarity-threshold 0.25 \
  --min-face-size 80 \
  --min-quality 0.4
```

**Result**: Multiple folders, find target in previews

### Scenario 4: Building Training Dataset
**Goal**: Maximum faces, high quality

```bash
python extract_faces.py \
  --video training_source.mp4 \
  --output-dir training_faces \
  --skip-frames 3 \
  --max-faces-per-person 100 \
  --min-quality 0.5 \
  --min-face-size 100
```

**Result**: Large high-quality dataset

## Verification Checklist

After extraction, verify:

- [ ] Preview files show correct grouping
- [ ] Target person has enough faces (>10)
- [ ] Faces are good quality (sharp, well-lit)
- [ ] No duplicate person folders
- [ ] extraction_summary.json looks correct

## Troubleshooting Guide

| Problem | Solution |
|---------|----------|
| Same person in multiple folders | Increase `--similarity-threshold` |
| Multiple people in same folder | Decrease `--similarity-threshold` |
| Not enough faces extracted | Lower `--min-quality`, increase `--max-frames` |
| Too many blurry faces | Raise `--min-quality` to 0.5+ |
| Processing too slow | Increase `--skip-frames` |
| Wrong person selected | Review all preview files carefully |

---

**Workflow Time Estimate**:
- Extraction: 2-5 minutes
- Review: 1-2 minutes  
- Selection: 30 seconds
- **Total: 5-10 minutes** (vs 1-2 hours manually!)
