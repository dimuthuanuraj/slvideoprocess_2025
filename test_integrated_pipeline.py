"""
Test Integrated Pipeline on Sample Video

This script tests the complete pipeline (detection + recognition + speaker)
on a sample video to validate Phase 5 integration.

Author: Research Team
Date: November 1, 2025
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from slceleb_modern.pipeline import IntegratedPipeline
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    print("="*80)
    print("PHASE 5 TEST: Integrated Pipeline on Sample Video")
    print("="*80)
    
    # Configuration
    video_path = "test_videos/sample video2.mp4"
    poi_references = [
        "images/pipe_test_persons/ptp1.png",
        "images/pipe_test_persons/ptp2.png",
        "images/pipe_test_persons/ptp3.png",
        "images/pipe_test_persons/ptp4.png",
        "images/pipe_test_persons/ptp5.png"
    ]
    max_frames = 1500  # Process 1500 frames (30 seconds at 50 FPS)
    output_json = "results/integrated_test_results.json"
    
    print(f"\n📹 Video: {video_path}")
    print(f"👤 POI References: {len(poi_references)} images")
    print(f"🎬 Max Frames: {max_frames}")
    print(f"💾 Output: {output_json}")
    
    # Initialize pipeline
    print(f"\n{'='*80}")
    print("Initializing Integrated Pipeline...")
    print(f"{'='*80}\n")
    
    pipeline = IntegratedPipeline(
        detection_confidence=0.5,
        recognition_threshold=0.252,
        speaking_threshold=0.5,
        use_gpu=True
    )
    
    print(f"\n{'='*80}")
    print("Loading POI References...")
    print(f"{'='*80}\n")
    
    success = pipeline.load_poi_references(poi_references)
    
    if not success:
        logger.error("Failed to load POI references!")
        return 1
    
    print(f"\n{'='*80}")
    print("Processing Video...")
    print(f"{'='*80}\n")
    
    results = pipeline.process_video(
        video_path=video_path,
        max_frames=max_frames,
        skip_frames=0,
        show_progress=True
    )
    
    # Display results
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"📊 Overall Statistics:")
    print(f"   Total frames processed: {len(results.frame_results)}")
    print(f"   Video duration: {results.duration:.2f}s")
    print(f"   Video FPS: {results.fps:.2f}")
    print(f"   Processing time: {results.total_processing_time:.2f}s")
    print(f"   Processing FPS: {results.avg_processing_fps:.2f}")
    print(f"   Real-time factor: {results.fps / results.avg_processing_fps:.2f}x")
    
    print(f"\n👤 POI Detection:")
    print(f"   Frames with POI: {results.frames_with_poi} "
          f"({100*results.frames_with_poi/len(results.frame_results):.1f}%)")
    print(f"   Frames with POI speaking: {results.frames_with_poi_speaking} "
          f"({100*results.frames_with_poi_speaking/len(results.frame_results):.1f}%)")
    
    print(f"\n🎤 Speaking Segments:")
    print(f"   Total segments: {len(results.speaking_segments)}")
    
    if len(results.speaking_segments) > 0:
        total_speaking_duration = sum(end - start for start, end, _ in results.speaking_segments)
        print(f"   Total speaking time: {total_speaking_duration:.2f}s")
        print(f"   Average segment length: {total_speaking_duration/len(results.speaking_segments):.2f}s")
        
        print(f"\n   Top 5 segments:")
        for i, (start, end, conf) in enumerate(results.speaking_segments[:5], 1):
            print(f"      {i}. {start:.2f}s - {end:.2f}s (duration: {end-start:.2f}s, confidence: {conf:.3f})")
    
    # Export results
    print(f"\n{'='*80}")
    print("Exporting Results...")
    print(f"{'='*80}\n")
    
    pipeline.export_results(results, output_json)
    
    print(f"\n✅ Test Complete!")
    print(f"   Results saved to: {output_json}")
    print(f"\n{'='*80}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
