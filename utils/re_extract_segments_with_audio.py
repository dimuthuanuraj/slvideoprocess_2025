#!/usr/bin/env python3
"""
Re-extract speaking segments with audio from existing JSON results.
This script reads the segments from the JSON file and extracts them using ffmpeg.
"""

import json
import subprocess
import logging
from pathlib import Path
import argparse

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def extract_segments_with_audio(video_path: str, json_path: str, output_dir: str):
    """
    Extract speaking segments with audio using ffmpeg.
    
    Args:
        video_path: Path to source video
        json_path: Path to JSON results file with segments
        output_dir: Directory to save segment videos
    """
    # Load JSON results
    with open(json_path, 'r') as f:
        results = json.load(f)
    
    segments = results.get('segments', [])
    if not segments:
        logger.warning("No segments found in JSON")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_stem = Path(video_path).stem
    
    logger.info(f"Extracting {len(segments)} segments from {video_path}")
    
    for i, segment in enumerate(segments):
        # Segments are tuples: (start_time, end_time, confidence)
        if isinstance(segment, list) and len(segment) >= 2:
            start_time = segment[0]
            end_time = segment[1]
        else:
            logger.warning(f"Invalid segment format: {segment}")
            continue
        
        duration = end_time - start_time
        
        # Create output file
        segment_file = output_dir / f"{video_stem}_segment_{i+1:03d}.mp4"
        
        # Use ffmpeg to extract segment with audio
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file
            '-ss', str(start_time),  # Start time
            '-i', video_path,  # Input file
            '-t', str(duration),  # Duration
            '-c', 'copy',  # Copy streams without re-encoding (fast)
            '-avoid_negative_ts', 'make_zero',  # Fix timestamp issues
            str(segment_file)
        ]
        
        try:
            # Run ffmpeg with suppressed output
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            logger.info(f"✓ Saved segment {i+1}/{len(segments)}: {segment_file.name}")
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Failed to extract segment {i+1}: {e}")
    
    logger.info(f"\n✅ Extraction complete! {len(segments)} segments saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Re-extract speaking segments with audio from JSON results'
    )
    parser.add_argument(
        '--video',
        required=True,
        help='Path to source video file'
    )
    parser.add_argument(
        '--json',
        required=True,
        help='Path to JSON results file'
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Directory to save segment videos'
    )
    
    args = parser.parse_args()
    
    extract_segments_with_audio(args.video, args.json, args.output_dir)


if __name__ == '__main__':
    main()
