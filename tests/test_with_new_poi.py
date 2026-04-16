"""
Test Integrated Pipeline with New POI Images

Tests the pipeline using the updated POI images from pipe_test_persons folder.

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
    print("PHASE 5 TEST: Integrated Pipeline with New POI Images")
    print("="*80)
    
    # Configuration
    video_path = "test_videos/sample_video.mp4"
    poi_references = [
        "images/pipe_test_persons/ptp1.png",
        "images/pipe_test_persons/ptp2.png",
        "images/pipe_test_persons/ptp3.png",
        "images/pipe_test_persons/ptp4.png",
        "images/pipe_test_persons/ptp5.png"
    ]
    max_frames = 500  # Process first ~16 seconds (at 30 FPS)
    output_json = "results/new_poi_test_results.json"
    
    print(f"\n📹 Video: {video_path}")
    print(f"👤 POI References: {len(poi_references)} images from pipe_test_persons")
    print(f"🎬 Max Frames: {max_frames}")
    print(f"💾 Output: {output_json}")
    
    # Initialize pipeline
    print(f"\n{'='*80}")
    print("Initializing Integrated Pipeline...")
    print(f"{'='*80}")
    
    pipeline = IntegratedPipeline(
        detection_confidence=0.5,
        recognition_threshold=0.252,  # ArcFace cosine similarity threshold
        speaking_threshold=0.5
    )
    
    print("✅ Pipeline initialized successfully!")
    
    # Load POI references
    print(f"\n{'='*80}")
    print("Loading POI Reference Images...")
    print(f"{'='*80}")
    
    success_count = 0
    for ref_path in poi_references:
        try:
            pipeline.load_poi_references([ref_path])
            success_count += 1
            print(f"✅ Loaded: {ref_path}")
        except Exception as e:
            print(f"❌ Failed to load {ref_path}: {e}")
    
    print(f"\n📊 POI References Loaded: {success_count}/{len(poi_references)}")
    
    if success_count == 0:
        print("❌ No POI references loaded! Cannot proceed.")
        return
    
    # Process video
    print(f"\n{'='*80}")
    print("Processing Video...")
    print(f"{'='*80}")
    
    try:
        results = pipeline.process_video(
            video_path=video_path,
            max_frames=max_frames,
            show_progress=True
        )
        
        print(f"\n{'='*80}")
        print("RESULTS SUMMARY")
        print(f"{'='*80}")
        
        print(f"\n📊 Video Statistics:")
        print(f"   Total Frames Processed: {results.total_frames}")
        print(f"   Processing Time: {results.processing_time:.2f}s")
        print(f"   Processing FPS: {results.fps:.2f}")
        print(f"   Video FPS: {results.video_fps}")
        print(f"   Video Duration: {results.video_duration:.2f}s")
        
        print(f"\n👤 POI Detection:")
        print(f"   Frames with POI: {results.frames_with_poi} ({results.frames_with_poi/results.total_frames*100:.1f}%)")
        print(f"   Frames with POI Speaking: {results.frames_with_poi_speaking} ({results.frames_with_poi_speaking/results.total_frames*100:.1f}%)")
        
        print(f"\n🗣️ Speaking Segments:")
        print(f"   Total Segments: {len(results.speaking_segments)}")
        if results.speaking_segments:
            for i, seg in enumerate(results.speaking_segments[:5], 1):  # Show first 5
                print(f"   Segment {i}: Frames {seg['start_frame']}-{seg['end_frame']} "
                      f"({seg['duration']:.2f}s, confidence: {seg['confidence']:.3f})")
            if len(results.speaking_segments) > 5:
                print(f"   ... and {len(results.speaking_segments) - 5} more segments")
        
        # Export results
        print(f"\n💾 Exporting results to: {output_json}")
        pipeline.export_results(results, output_json)
        print("✅ Results exported successfully!")
        
        # Detailed frame analysis
        print(f"\n{'='*80}")
        print("DETAILED FRAME ANALYSIS")
        print(f"{'='*80}")
        
        frames_with_faces = sum(1 for fr in results.frame_results if fr.faces_detected > 0)
        print(f"\n📊 Face Detection:")
        print(f"   Frames with ANY faces: {frames_with_faces} ({frames_with_faces/results.total_frames*100:.1f}%)")
        
        # Sample frames with faces
        face_frames = [i for i, fr in enumerate(results.frame_results) if fr.faces_detected > 0]
        if face_frames:
            print(f"\n🎯 Sample frames with faces:")
            for frame_idx in face_frames[:10]:  # Show first 10
                fr = results.frame_results[frame_idx]
                identities = ', '.join(fr.face_identities) if fr.face_identities else 'None'
                print(f"   Frame {frame_idx}: {fr.faces_detected} face(s), Identities: {identities}, "
                      f"POI Present: {fr.poi_present}, Speaking: {fr.is_speaking}")
            if len(face_frames) > 10:
                print(f"   ... and {len(face_frames) - 10} more frames with faces")
        
        print(f"\n{'='*80}")
        print("TEST COMPLETE!")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"\n❌ Error processing video: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
