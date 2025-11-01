"""
Test script for Optimized Face Recognizer

Compares performance of standard vs optimized recognizer.
"""

import sys
import time
from pathlib import Path
from typing import List
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from slceleb_modern.detection.face_detector import MediaPipeFaceDetector
from slceleb_modern.recognition.face_recognizer_optimized import OptimizedFaceRecognizer


def test_optimized_recognizer(video_path: str, poi_images: List[str], num_frames: int = 100, start_frame: int = 500):
    """
    Test optimized recognizer performance.
    """
    print("="*80)
    print("OPTIMIZED RECOGNIZER TEST")
    print("="*80)
    
    # Initialize components
    print("\nInitializing detector...")
    detector = MediaPipeFaceDetector(min_detection_confidence=0.5)
    
    print("Initializing OPTIMIZED recognizer (buffalo_s with GPU + caching)...")
    recognizer = OptimizedFaceRecognizer(
        model_name="buffalo_s",  # Smaller/faster model
        use_gpu=True,
        cache_size=100
    )
    
    # Load POI references
    print(f"Loading {len(poi_images)} POI references...")
    recognizer.load_poi_references(poi_images)
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Process frames
    print(f"\nProcessing frames {start_frame} to {start_frame + num_frames}...")
    
    times = {
        'detection': [],
        'recognition': [],
        'total': []
    }
    
    face_count = 0
    poi_count = 0
    
    for i in range(start_frame, start_frame + num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_start = time.time()
        
        # Detection
        det_start = time.time()
        detections = detector.detect(frame)
        det_time = time.time() - det_start
        times['detection'].append(det_time * 1000)
        
        # Recognition
        if len(detections) > 0:
            rec_start = time.time()
            for detection in detections:
                result = recognizer.recognize_face(frame, detection.bbox, frame_number=i)
                if result.is_poi:
                    poi_count += 1
                face_count += 1
            rec_time = time.time() - rec_start
            times['recognition'].append(rec_time * 1000)
        
        frame_time = time.time() - frame_start
        times['total'].append(frame_time * 1000)
        
        if (i - start_frame + 1) % 20 == 0:
            print(f"  Processed {i - start_frame + 1}/{num_frames} frames...")
    
    cap.release()
    
    # Print results
    print("\n" + "="*80)
    print("PERFORMANCE RESULTS")
    print("="*80)
    
    avg_det = np.mean(times['detection'])
    avg_rec = np.mean(times['recognition']) if times['recognition'] else 0
    avg_total = np.mean(times['total'])
    
    print(f"\nTiming (milliseconds per frame):")
    print(f"  Detection:    {avg_det:>8.2f} ms  ({1000/avg_det:>6.2f} FPS)")
    print(f"  Recognition:  {avg_rec:>8.2f} ms  ({1000/avg_rec if avg_rec > 0 else 0:>6.2f} FPS)")
    print(f"  Total:        {avg_total:>8.2f} ms  ({1000/avg_total:>6.2f} FPS)")
    
    print(f"\nDetection:")
    print(f"  Faces detected: {face_count}")
    print(f"  POI detected:   {poi_count} ({poi_count/face_count*100 if face_count > 0 else 0:.1f}%)")
    
    # Cache statistics
    cache_stats = recognizer.get_cache_stats()
    print(f"\nCache Performance:")
    print(f"  Cache size:  {cache_stats['cache_size']}")
    print(f"  Cache hits:  {cache_stats['hits']}")
    print(f"  Cache misses: {cache_stats['misses']}")
    print(f"  Hit rate:    {cache_stats['hit_rate']*100:.1f}%")
    
    return {
        'avg_detection_ms': avg_det,
        'avg_recognition_ms': avg_rec,
        'avg_total_ms': avg_total,
        'fps': 1000 / avg_total,
        'cache_hit_rate': cache_stats['hit_rate']
    }


def main():
    """Main test script"""
    import argparse
    from typing import List
    
    parser = argparse.ArgumentParser(description="Test optimized face recognizer")
    parser.add_argument('--video', required=True, help='Path to test video')
    parser.add_argument('--poi-dir', default='images/pipe_test_persons', help='POI images directory')
    parser.add_argument('--frames', type=int, default=100, help='Number of frames to test')
    parser.add_argument('--start-frame', type=int, default=500, help='Frame to start from')
    
    args = parser.parse_args()
    
    # Find POI images
    poi_dir = Path(args.poi_dir)
    poi_images = sorted([str(p) for p in poi_dir.glob('*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    
    if not poi_images:
        print(f"Error: No POI images found in {poi_dir}")
        return 1
    
    print(f"Found {len(poi_images)} POI images")
    
    # Test optimized recognizer
    results = test_optimized_recognizer(args.video, poi_images, args.frames, args.start_frame)
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    print(f"\n🎯 Achieved {results['fps']:.2f} FPS")
    print(f"📊 Cache hit rate: {results['cache_hit_rate']*100:.1f}%")
    
    # Compare with baseline
    baseline_fps = 10.63  # From benchmark
    improvement = ((results['fps'] - baseline_fps) / baseline_fps) * 100
    print(f"\n📈 Improvement over baseline: {improvement:+.1f}%")
    
    if results['fps'] >= 15.0:
        print("\n✅ TARGET ACHIEVED: >= 15 FPS")
    else:
        print(f"\n⚠️  Target not met: {results['fps']:.2f} FPS < 15.0 FPS")
        print("   Consider: Reduce detection frequency, use face tracking")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
