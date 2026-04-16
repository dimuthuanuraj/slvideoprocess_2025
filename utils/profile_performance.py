"""
Performance Profiling Script

Analyzes the integrated pipeline to identify bottlenecks and optimization opportunities.
"""

import sys
import time
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List
import cProfile
import pstats
from io import StringIO

sys.path.insert(0, str(Path(__file__).parent))

from slceleb_modern.pipeline import IntegratedPipeline


def profile_component_times(video_path: str, poi_images: List[str], num_frames: int = 100, start_frame: int = 0):
    """
    Profile individual component processing times.
    
    Args:
        video_path: Path to test video
        poi_images: List of POI reference images
        num_frames: Number of frames to profile
        start_frame: Frame to start profiling from
    """
    print("="*80)
    print("COMPONENT-LEVEL PROFILING")
    print("="*80)
    
    # Initialize pipeline
    print("\nInitializing pipeline...")
    pipeline = IntegratedPipeline(
        detection_confidence=0.5,
        recognition_threshold=0.252,
        speaking_threshold=0.5
    )
    
    pipeline.load_poi_references(poi_images)
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Skip to start frame
    if start_frame > 0:
        print(f"Skipping to frame {start_frame}...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Load audio
    print("Loading audio...")
    audio_start = time.time()
    pipeline.audio_extractor.load_audio(video_path)
    audio_time = time.time() - audio_start
    print(f"  Audio loading: {audio_time:.2f}s")
    
    # Profile per-frame processing
    print(f"\nProfiling frames {start_frame} to {start_frame + num_frames}...")
    
    times = {
        'detection': [],
        'recognition': [],
        'speaker': [],
        'total': []
    }
    
    for i in range(start_frame, start_frame + num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Total frame time
        frame_start = time.time()
        
        # 1. Face Detection
        det_start = time.time()
        detections = pipeline.detector.detect(frame)
        det_time = time.time() - det_start
        times['detection'].append(det_time * 1000)  # ms
        
        # 2. Face Recognition
        if len(detections) > 0:
            rec_start = time.time()
            for detection in detections:
                result = pipeline.recognizer.recognize_face(frame, detection.bbox)
            rec_time = time.time() - rec_start
            times['recognition'].append(rec_time * 1000)  # ms
        
        # 3. Speaker Detection
        if len(detections) > 0 and pipeline.audio_loaded:
            spk_start = time.time()
            for detection in detections:
                pipeline.lip_tracker.update(i, detection.landmarks_2d)
                if pipeline.lip_tracker.is_ready():
                    lip_seq = pipeline.lip_tracker.get_lip_opening_sequence()
                    audio_seq = pipeline.audio_extractor.get_amplitude_envelope_sequence(
                        max(0, i - 29), i
                    )
                    pipeline.correlator.correlate(lip_seq, audio_seq)
            spk_time = time.time() - spk_start
            times['speaker'].append(spk_time * 1000)  # ms
        
        frame_time = time.time() - frame_start
        times['total'].append(frame_time * 1000)  # ms
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{num_frames} frames...")
    
    cap.release()
    
    # Calculate statistics
    print("\n" + "="*80)
    print("TIMING STATISTICS (milliseconds per frame)")
    print("="*80)
    
    for component, time_list in times.items():
        if time_list:
            avg = np.mean(time_list)
            std = np.std(time_list)
            min_t = np.min(time_list)
            max_t = np.max(time_list)
            fps = 1000 / avg if avg > 0 else 0
            
            print(f"\n{component.upper()}:")
            print(f"  Average:  {avg:>8.2f} ms  ({fps:>6.2f} FPS)")
            print(f"  Std Dev:  {std:>8.2f} ms")
            print(f"  Min:      {min_t:>8.2f} ms")
            print(f"  Max:      {max_t:>8.2f} ms")
            print(f"  % of Total: {(avg / np.mean(times['total']) * 100):>6.1f}%")
    
    # Bottleneck analysis
    print("\n" + "="*80)
    print("BOTTLENECK ANALYSIS")
    print("="*80)
    
    avg_times = {k: np.mean(v) if v else 0 for k, v in times.items() if k != 'total'}
    sorted_components = sorted(avg_times.items(), key=lambda x: x[1], reverse=True)
    
    print("\nComponents by processing time:")
    for i, (component, avg_time) in enumerate(sorted_components, 1):
        percentage = (avg_time / np.mean(times['total']) * 100)
        print(f"  {i}. {component:15s} {avg_time:>8.2f} ms  ({percentage:>5.1f}%)")
    
    return times


def profile_full_pipeline(video_path: str, poi_images: List[str], num_frames: int = 100):
    """
    Profile the full pipeline using cProfile.
    
    Args:
        video_path: Path to test video
        poi_images: List of POI reference images
        num_frames: Number of frames to profile
    """
    print("\n" + "="*80)
    print("FULL PIPELINE PROFILING (cProfile)")
    print("="*80)
    
    pipeline = IntegratedPipeline(
        detection_confidence=0.5,
        recognition_threshold=0.252,
        speaking_threshold=0.5
    )
    
    pipeline.load_poi_references(poi_images)
    
    # Profile with cProfile
    profiler = cProfile.Profile()
    profiler.enable()
    
    results = pipeline.process_video(
        video_path,
        max_frames=num_frames,
        show_progress=False
    )
    
    profiler.disable()
    
    # Print top functions
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(30)  # Top 30 functions
    
    print("\nTop 30 functions by cumulative time:")
    print(s.getvalue())
    
    return results


def analyze_gpu_usage():
    """
    Analyze GPU utilization.
    """
    print("\n" + "="*80)
    print("GPU USAGE ANALYSIS")
    print("="*80)
    
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        
        if gpus:
            for i, gpu in enumerate(gpus):
                print(f"\nGPU {i}: {gpu.name}")
                print(f"  Load:         {gpu.load * 100:.1f}%")
                print(f"  Memory Used:  {gpu.memoryUsed:.1f} MB / {gpu.memoryTotal:.1f} MB ({gpu.memoryUtil * 100:.1f}%)")
                print(f"  Temperature:  {gpu.temperature}°C")
        else:
            print("\nNo GPU detected or GPU monitoring not available")
    except Exception as e:
        print(f"\nGPU monitoring error: {e}")


def generate_optimization_recommendations(times: Dict[str, List[float]]):
    """
    Generate optimization recommendations based on profiling data.
    
    Args:
        times: Dictionary of component timing data
    """
    print("\n" + "="*80)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("="*80)
    
    avg_times = {k: np.mean(v) if v else 0 for k, v in times.items()}
    total_avg = avg_times.get('total', 0)
    
    recommendations = []
    
    # Check each component
    if avg_times.get('recognition', 0) > total_avg * 0.4:
        recommendations.append({
            'priority': 'HIGH',
            'component': 'Face Recognition',
            'issue': f"Takes {avg_times['recognition']:.1f}ms ({avg_times['recognition']/total_avg*100:.1f}% of total)",
            'solutions': [
                "Enable GPU acceleration for InsightFace",
                "Cache face embeddings for tracked faces (don't re-compute every frame)",
                "Use buffalo_s (smaller model) instead of buffalo_l for speed",
                "Batch process multiple faces together"
            ]
        })
    
    if avg_times.get('detection', 0) > total_avg * 0.3:
        recommendations.append({
            'priority': 'MEDIUM',
            'component': 'Face Detection',
            'issue': f"Takes {avg_times['detection']:.1f}ms ({avg_times['detection']/total_avg*100:.1f}% of total)",
            'solutions': [
                "Enable MediaPipe GPU delegate",
                "Reduce detection frequency (track faces between detections)",
                "Lower input resolution for detection",
                "Use static_image_mode=False for video"
            ]
        })
    
    if avg_times.get('speaker', 0) > total_avg * 0.3:
        recommendations.append({
            'priority': 'LOW',
            'component': 'Speaker Detection',
            'issue': f"Takes {avg_times['speaker']:.1f}ms ({avg_times['speaker']/total_avg*100:.1f}% of total)",
            'solutions': [
                "Pre-compute all audio features once (not per-frame)",
                "Reduce temporal window size (30 → 20 frames)",
                "Use GPU for MFCC computation",
                "Optimize numpy operations"
            ]
        })
    
    # Print recommendations
    for i, rec in enumerate(recommendations, 1):
        print(f"\n[{rec['priority']}] Recommendation {i}: Optimize {rec['component']}")
        print(f"  Issue: {rec['issue']}")
        print(f"  Solutions:")
        for j, solution in enumerate(rec['solutions'], 1):
            print(f"    {j}. {solution}")
    
    if not recommendations:
        print("\n✅ Pipeline is well-balanced. Consider general optimizations:")
        print("  1. Enable GPU acceleration across all components")
        print("  2. Implement batch processing for multiple videos")
        print("  3. Use multi-threading for I/O operations")


def main():
    """Main profiling script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Profile integrated pipeline performance")
    parser.add_argument('--video', required=True, help='Path to test video')
    parser.add_argument('--poi-dir', default='images/pipe_test_persons', help='POI images directory')
    parser.add_argument('--frames', type=int, default=100, help='Number of frames to profile')
    parser.add_argument('--start-frame', type=int, default=500, help='Frame to start profiling from')
    parser.add_argument('--full-profile', action='store_true', help='Run cProfile on full pipeline')
    
    args = parser.parse_args()
    
    # Find POI images
    poi_dir = Path(args.poi_dir)
    poi_images = sorted([str(p) for p in poi_dir.glob('*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    
    if not poi_images:
        print(f"Error: No POI images found in {poi_dir}")
        return 1
    
    print(f"Found {len(poi_images)} POI images")
    
    # Profile components
    times = profile_component_times(args.video, poi_images, args.frames, args.start_frame)
    
    # GPU analysis
    analyze_gpu_usage()
    
    # Generate recommendations
    generate_optimization_recommendations(times)
    
    # Full pipeline profiling (optional)
    if args.full_profile:
        profile_full_pipeline(args.video, poi_images, args.frames)
    
    print("\n" + "="*80)
    print("PROFILING COMPLETE")
    print("="*80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
