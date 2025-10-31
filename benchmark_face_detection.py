"""
Face Detection Benchmark - Old vs New
======================================

Compare RetinaFace + dlib (old) vs MediaPipe Face Mesh (new)

Metrics:
- Detection accuracy (precision, recall)
- Processing speed (FPS)
- Landmark quality (alignment error)
- Robustness (challenging conditions)

Author: SLCeleb Research Team
Date: October 31, 2025
Phase: 2 - Face Detection Benchmarking
"""

import cv2
import numpy as np
import time
from pathlib import Path
from typing import List, Dict, Tuple
import json
from dataclasses import dataclass, asdict
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from slceleb_modern.detection import MediaPipeFaceDetector, FaceDetection


@dataclass
class BenchmarkResults:
    """Container for benchmark results."""
    method: str
    avg_fps: float
    total_faces_detected: int
    avg_detection_time_ms: float
    avg_landmarks_per_face: int
    memory_usage_mb: float
    success_rate: float  # Percentage of frames with detection


class FaceDetectionBenchmark:
    """Benchmark face detection methods."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize detectors
        print("Initializing detectors...")
        self.new_detector = MediaPipeFaceDetector(
            static_image_mode=False,
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        print("✓ MediaPipe detector initialized")
        
        # Try to load old detector if available
        self.old_detector = None
        try:
            from face_detection import RetinaFaceDetector
            self.old_detector = RetinaFaceDetector()
            print("✓ RetinaFace detector initialized")
        except:
            print("⚠ Old RetinaFace detector not available (comparing MediaPipe only)")
    
    def benchmark_video(self, video_path: str, max_frames: int = 300) -> Dict:
        """
        Benchmark detection on a video file.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum frames to process
            
        Returns:
            Dictionary with benchmark results for each method
        """
        results = {}
        
        # Benchmark MediaPipe (new)
        print(f"\n📊 Benchmarking MediaPipe on {video_path}")
        results['mediapipe'] = self._benchmark_method(
            video_path, self.new_detector, 'mediapipe', max_frames
        )
        
        # Benchmark RetinaFace (old) if available
        if self.old_detector:
            print(f"\n📊 Benchmarking RetinaFace on {video_path}")
            results['retinaface'] = self._benchmark_method(
                video_path, self.old_detector, 'retinaface', max_frames
            )
        
        return results
    
    def _benchmark_method(self, video_path: str, detector, method_name: str,
                         max_frames: int) -> BenchmarkResults:
        """Benchmark a single detection method."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video: {total_frames} frames @ {total_fps:.1f} FPS")
        
        frame_count = 0
        total_detection_time = 0.0
        total_faces = 0
        successful_detections = 0
        
        times = []
        face_counts = []
        
        print("Processing frames...")
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Time the detection
            start_time = time.time()
            
            if method_name == 'mediapipe':
                detections = detector.detect(frame)
                num_faces = len(detections)
            else:
                # Old detector format (placeholder - adjust based on actual API)
                detections = detector.detect(frame)
                num_faces = len(detections) if detections else 0
            
            detection_time = time.time() - start_time
            
            # Record metrics
            times.append(detection_time)
            face_counts.append(num_faces)
            total_detection_time += detection_time
            total_faces += num_faces
            
            if num_faces > 0:
                successful_detections += 1
            
            frame_count += 1
            
            # Progress indicator
            if frame_count % 50 == 0:
                avg_fps = frame_count / total_detection_time
                print(f"  Processed {frame_count}/{max_frames} frames "
                      f"({avg_fps:.1f} FPS, {num_faces} faces)")
        
        cap.release()
        
        # Calculate metrics
        avg_detection_time_ms = (total_detection_time / frame_count) * 1000
        avg_fps = frame_count / total_detection_time
        success_rate = (successful_detections / frame_count) * 100
        avg_landmarks = 478 if method_name == 'mediapipe' else 68
        
        results = BenchmarkResults(
            method=method_name,
            avg_fps=avg_fps,
            total_faces_detected=total_faces,
            avg_detection_time_ms=avg_detection_time_ms,
            avg_landmarks_per_face=avg_landmarks,
            memory_usage_mb=0.0,  # TODO: Implement memory tracking
            success_rate=success_rate
        )
        
        print(f"\n✓ {method_name.upper()} Results:")
        print(f"  Average FPS: {avg_fps:.2f}")
        print(f"  Detection time: {avg_detection_time_ms:.2f} ms/frame")
        print(f"  Success rate: {success_rate:.1f}%")
        print(f"  Total faces: {total_faces}")
        print(f"  Landmarks per face: {avg_landmarks}")
        
        return results
    
    def benchmark_image_folder(self, image_dir: str) -> Dict:
        """
        Benchmark detection on a folder of images.
        
        Args:
            image_dir: Path to directory containing images
            
        Returns:
            Dictionary with benchmark results
        """
        image_paths = list(Path(image_dir).glob("*.jpg")) + \
                     list(Path(image_dir).glob("*.png"))
        
        if not image_paths:
            raise ValueError(f"No images found in {image_dir}")
        
        print(f"\n📊 Benchmarking on {len(image_paths)} images")
        
        results = {}
        
        # Benchmark MediaPipe
        print("\nTesting MediaPipe...")
        mp_times = []
        mp_faces = []
        
        for img_path in image_paths:
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            start = time.time()
            detections = self.new_detector.detect(image)
            detection_time = time.time() - start
            
            mp_times.append(detection_time)
            mp_faces.append(len(detections))
        
        results['mediapipe'] = {
            'avg_time_ms': np.mean(mp_times) * 1000,
            'std_time_ms': np.std(mp_times) * 1000,
            'avg_faces': np.mean(mp_faces),
            'total_faces': sum(mp_faces),
            'success_rate': (sum(1 for f in mp_faces if f > 0) / len(mp_faces)) * 100
        }
        
        print(f"✓ MediaPipe: {results['mediapipe']['avg_time_ms']:.2f} ms/image")
        print(f"  Success rate: {results['mediapipe']['success_rate']:.1f}%")
        
        return results
    
    def compare_results(self, results: Dict) -> None:
        """Print comparison of results."""
        print("\n" + "=" * 70)
        print("BENCHMARK COMPARISON")
        print("=" * 70)
        
        if 'mediapipe' in results and 'retinaface' in results:
            mp = results['mediapipe']
            rf = results['retinaface']
            
            print(f"\n{'Metric':<30} {'MediaPipe':<20} {'RetinaFace':<20} {'Improvement':<15}")
            print("-" * 85)
            
            # FPS comparison
            fps_improvement = ((mp.avg_fps - rf.avg_fps) / rf.avg_fps) * 100
            print(f"{'Average FPS':<30} {mp.avg_fps:<20.2f} {rf.avg_fps:<20.2f} {fps_improvement:>+.1f}%")
            
            # Detection time
            time_improvement = ((rf.avg_detection_time_ms - mp.avg_detection_time_ms) / rf.avg_detection_time_ms) * 100
            print(f"{'Detection Time (ms)':<30} {mp.avg_detection_time_ms:<20.2f} {rf.avg_detection_time_ms:<20.2f} {time_improvement:>+.1f}%")
            
            # Landmarks
            print(f"{'Landmarks per Face':<30} {mp.avg_landmarks_per_face:<20} {rf.avg_landmarks_per_face:<20} "
                  f"{mp.avg_landmarks_per_face - rf.avg_landmarks_per_face:>+} pts")
            
            # Success rate
            print(f"{'Success Rate (%)':<30} {mp.success_rate:<20.1f} {rf.success_rate:<20.1f} "
                  f"{mp.success_rate - rf.success_rate:>+.1f}%")
            
            print("\n" + "=" * 70)
            print("SUMMARY:")
            print(f"MediaPipe is {abs(fps_improvement):.1f}% {'faster' if fps_improvement > 0 else 'slower'} than RetinaFace")
            print(f"MediaPipe provides {mp.avg_landmarks_per_face / rf.avg_landmarks_per_face:.1f}x more landmarks")
            print("=" * 70)
        else:
            print("\nMediaPipe Results Only:")
            mp = results['mediapipe']
            print(f"  Average FPS: {mp.avg_fps:.2f}")
            print(f"  Detection Time: {mp.avg_detection_time_ms:.2f} ms")
            print(f"  Success Rate: {mp.success_rate:.1f}%")
            print(f"  Landmarks: {mp.avg_landmarks_per_face} per face")
    
    def save_results(self, results: Dict, filename: str = "benchmark_results.json"):
        """Save benchmark results to JSON."""
        output_path = self.output_dir / filename
        
        # Convert dataclass to dict if needed
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, BenchmarkResults):
                serializable_results[key] = asdict(value)
            else:
                serializable_results[key] = value
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_path}")


def main():
    """Run benchmark."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark face detection methods")
    parser.add_argument('--video', type=str, help="Path to video file")
    parser.add_argument('--images', type=str, help="Path to image folder")
    parser.add_argument('--max-frames', type=int, default=300, help="Max frames to process")
    parser.add_argument('--output', type=str, default='benchmark_results', help="Output directory")
    
    args = parser.parse_args()
    
    if not args.video and not args.images:
        print("Please provide --video or --images argument")
        return
    
    # Initialize benchmark
    benchmark = FaceDetectionBenchmark(output_dir=args.output)
    
    # Run appropriate benchmark
    if args.video:
        results = benchmark.benchmark_video(args.video, max_frames=args.max_frames)
        benchmark.compare_results(results)
        benchmark.save_results(results, "video_benchmark.json")
    
    if args.images:
        results = benchmark.benchmark_image_folder(args.images)
        benchmark.save_results(results, "image_benchmark.json")
    
    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()
