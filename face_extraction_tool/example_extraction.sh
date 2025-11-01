#!/bin/bash
################################################################################
# Face Extraction Tool - Example Usage Script
################################################################################
# 
# Purpose: Demonstrate face extraction and clustering from video
# 
# This script extracts faces from a sample video and organizes them by person
# using automatic clustering. The extracted faces can be used as POI (Person
# of Interest) reference images in the main video processing pipeline.
#
# Workflow:
#   1. Process video frames (skip every 5th for speed)
#   2. Detect faces using MediaPipe (handles frontal and profile views)
#   3. Extract 512D embeddings using InsightFace
#   4. Cluster faces by person using cosine similarity
#   5. Save best quality faces organized in person_XXX folders
#   6. Generate preview grids for visual verification
#
# Author: SLCeleb Research Team
# Date: November 2, 2025
################################################################################

echo "================================"
echo "Face Extraction Example"
echo "================================"
echo ""

# Activate conda environment with required dependencies
echo "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate slceleb_modern

# Configure input/output paths
VIDEO="../test_videos/sample video2.mp4"
OUTPUT_DIR="example_output"

echo "Video: $VIDEO"
echo "Output: $OUTPUT_DIR"
echo ""

# Run face extraction with optimized parameters
echo "Starting face extraction..."
python extract_faces.py \
  --video "$VIDEO" \
  --output-dir "$OUTPUT_DIR" \
  --skip-frames 5 \              # Process every 5th frame for speed (20-40 FPS)
  --max-faces-per-person 30 \    # Keep top 30 best quality faces per person
  --min-quality 0.2 \            # Quality threshold (0.2 = include good side views)
  --max-frames 1000              # Limit to first 1000 frames for quick demo

echo ""
echo "================================"
echo "Extraction Complete!"
echo "================================"
echo ""
echo "Check the output directory for:"
echo "  1. person_XXX/ folders with extracted faces"
echo "  2. preview_person_XXX.jpg files showing face grids"
echo "  3. extraction_summary.json with detailed statistics"
echo ""
echo "To view preview files:"
echo "  eog $OUTPUT_DIR/person_*/preview*.jpg"
echo ""
echo "To use as POI references:"
echo "  python ../production_run.py \\"
echo "    --video-list videos.txt \\"
echo "    --poi-dir $OUTPUT_DIR/person_000 \\"
echo "    --output-dir results"
echo ""
