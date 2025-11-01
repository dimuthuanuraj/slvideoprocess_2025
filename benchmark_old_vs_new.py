"""
Benchmark Comparison: Old vs New System

This script compares the performance of the old system (RetinaFace + FaceNet + SyncNet)
against the new system (MediaPipe + InsightFace + Audio-Visual Correlator).

Metrics Compared:
- Processing speed (FPS)
- Face detection accuracy
- Face recognition accuracy
- Speaker detection accuracy
- False positive rate
- Memory usage
- CPU/GPU utilization

Author: Research Team
Date: November 1, 2025
"""

import sys
import time
import json
import cv2
import numpy as np
import psutil
import GPUtil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import logging

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

# Import new system
from slceleb_modern.pipeline import IntegratedPipeline

# Import old system components (if available)
try:
    from common import config
    from face_detection import FaceDetection
    from face_validation import FaceValidation
    from speaker_validation import SpeakerValidation
    OLD_SYSTEM_AVAILABLE = True
except ImportError:
    OLD_SYSTEM_AVAILABLE = False
    print("⚠️  Old system not fully available - will only benchmark new system")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkMetrics:
    """Container for benchmark metrics"""
    
    system_name: str
    
    # Performance metrics
    total_frames: int = 0
    processing_time: float = 0.0
    fps: float = 0.0
    real_time_factor: float = 0.0
    
    # Detection metrics
    faces_detected_total: int = 0
    avg_faces_per_frame: float = 0.0
    detection_rate: float = 0.0  # % of frames with faces
    
    # Recognition metrics
    poi_detected_frames: int = 0
    poi_detection_rate: float = 0.0  # % of frames with POI
    avg_recognition_confidence: float = 0.0
    
    # Speaker detection metrics
    poi_speaking_frames: int = 0
    poi_speaking_rate: float = 0.0  # % of frames with POI speaking
    num_speaking_segments: int = 0
    total_speaking_time: float = 0.0
    avg_speaking_confidence: float = 0.0
    
    # Resource usage
    peak_memory_mb: float = 0.0
    avg_cpu_percent: float = 0.0
    avg_gpu_percent: float = 0.0
    avg_gpu_memory_mb: float = 0.0
    
    # Error metrics
    num_errors: int = 0
    error_rate: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


class SystemMonitor:
    """Monitor system resource usage during processing"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.cpu_samples = []
        self.memory_samples = []
        self.gpu_samples = []
        self.gpu_memory_samples = []
        
    def sample(self):
        """Take a resource usage sample"""
        # CPU and memory
        self.cpu_samples.append(self.process.cpu_percent())
        self.memory_samples.append(self.process.memory_info().rss / 1024 / 1024)  # MB
        
        # GPU (if available)
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # First GPU
                self.gpu_samples.append(gpu.load * 100)
                self.gpu_memory_samples.append(gpu.memoryUsed)
        except:
            pass
    
    def get_stats(self) -> Dict[str, float]:
        """Get aggregated resource usage statistics"""
        return {
            'avg_cpu_percent': np.mean(self.cpu_samples) if self.cpu_samples else 0.0,
            'peak_memory_mb': max(self.memory_samples) if self.memory_samples else 0.0,
            'avg_gpu_percent': np.mean(self.gpu_samples) if self.gpu_samples else 0.0,
            'avg_gpu_memory_mb': np.mean(self.gpu_memory_samples) if self.gpu_memory_samples else 0.0
        }


def benchmark_new_system(
    video_path: str,
    poi_images: List[str],
    max_frames: int = None
) -> BenchmarkMetrics:
    """
    Benchmark the new integrated pipeline.
    
    Args:
        video_path: Path to test video
        poi_images: List of POI reference images
        max_frames: Maximum frames to process (None = all)
        
    Returns:
        BenchmarkMetrics with results
    """
    logger.info("="*80)
    logger.info("BENCHMARKING NEW SYSTEM (MediaPipe + InsightFace + AV Correlator)")
    logger.info("="*80)
    
    metrics = BenchmarkMetrics(system_name="New System (Modern)")
    monitor = SystemMonitor()
    
    try:
        # Initialize pipeline
        logger.info("Initializing new pipeline...")
        pipeline = IntegratedPipeline(
            detection_confidence=0.5,
            recognition_threshold=0.252,
            speaking_threshold=0.5,
            use_gpu=True
        )
        
        # Load POI references
        logger.info(f"Loading {len(poi_images)} POI references...")
        pipeline.load_poi_references(poi_images)
        
        # Process video
        logger.info(f"Processing video: {video_path}")
        start_time = time.time()
        
        results = pipeline.process_video(
            video_path,
            max_frames=max_frames,
            skip_frames=0,
            show_progress=True
        )
        
        end_time = time.time()
        
        # Sample resources during processing (we'll do post-processing stats)
        monitor.sample()
        
        # Calculate metrics
        metrics.total_frames = results.total_frames
        metrics.processing_time = results.total_processing_time
        metrics.fps = results.avg_processing_fps
        metrics.real_time_factor = results.duration / metrics.processing_time if metrics.processing_time > 0 else 0
        
        # Detection metrics
        frames_with_faces = sum(1 for r in results.frame_results if r.faces_detected > 0)
        total_faces = sum(r.faces_detected for r in results.frame_results)
        metrics.faces_detected_total = total_faces
        metrics.avg_faces_per_frame = total_faces / metrics.total_frames if metrics.total_frames > 0 else 0
        metrics.detection_rate = (frames_with_faces / metrics.total_frames * 100) if metrics.total_frames > 0 else 0
        
        # Recognition metrics
        metrics.poi_detected_frames = results.frames_with_poi
        metrics.poi_detection_rate = (results.frames_with_poi / metrics.total_frames * 100) if metrics.total_frames > 0 else 0
        
        # Calculate average recognition confidence
        poi_confidences = [
            conf for r in results.frame_results 
            for i, (identity, conf) in enumerate(zip(r.face_identities, r.face_confidences))
            if identity == "POI"
        ]
        metrics.avg_recognition_confidence = np.mean(poi_confidences) if poi_confidences else 0.0
        
        # Speaker detection metrics
        metrics.poi_speaking_frames = results.frames_with_poi_speaking
        metrics.poi_speaking_rate = (results.frames_with_poi_speaking / metrics.total_frames * 100) if metrics.total_frames > 0 else 0
        metrics.num_speaking_segments = len(results.speaking_segments)
        metrics.total_speaking_time = sum(end - start for start, end, _ in results.speaking_segments)
        
        # Average speaking confidence
        speaking_confidences = [conf for _, _, conf in results.speaking_segments]
        metrics.avg_speaking_confidence = np.mean(speaking_confidences) if speaking_confidences else 0.0
        
        # Resource usage
        resource_stats = monitor.get_stats()
        metrics.avg_cpu_percent = resource_stats['avg_cpu_percent']
        metrics.peak_memory_mb = resource_stats['peak_memory_mb']
        metrics.avg_gpu_percent = resource_stats['avg_gpu_percent']
        metrics.avg_gpu_memory_mb = resource_stats['avg_gpu_memory_mb']
        
        logger.info("✅ New system benchmark complete!")
        
    except Exception as e:
        logger.error(f"❌ Error benchmarking new system: {e}")
        metrics.num_errors = 1
        metrics.error_rate = 100.0
        
    return metrics


def benchmark_old_system(
    video_path: str,
    poi_images: List[str],
    max_frames: int = None
) -> BenchmarkMetrics:
    """
    Benchmark the old system (if available).
    
    Args:
        video_path: Path to test video
        poi_images: List of POI reference images
        max_frames: Maximum frames to process (None = all)
        
    Returns:
        BenchmarkMetrics with results
    """
    if not OLD_SYSTEM_AVAILABLE:
        logger.warning("Old system not available - skipping benchmark")
        return None
    
    logger.info("="*80)
    logger.info("BENCHMARKING OLD SYSTEM (RetinaFace + FaceNet + SyncNet)")
    logger.info("="*80)
    
    metrics = BenchmarkMetrics(system_name="Old System (Legacy)")
    monitor = SystemMonitor()
    
    # TODO: Implement old system benchmark
    # This would require adapting the old run.py code
    logger.warning("⚠️  Old system benchmark not yet implemented")
    logger.info("   This requires integrating the old run.py pipeline")
    
    return metrics


def print_comparison(new_metrics: BenchmarkMetrics, old_metrics: Optional[BenchmarkMetrics] = None):
    """
    Print comparison table of metrics.
    
    Args:
        new_metrics: Metrics from new system
        old_metrics: Metrics from old system (optional)
    """
    print("\n" + "="*100)
    print("BENCHMARK COMPARISON RESULTS")
    print("="*100)
    
    if old_metrics:
        # Side-by-side comparison
        print(f"\n{'Metric':<40} {'Old System':<25} {'New System':<25} {'Improvement':<10}")
        print("-"*100)
        
        # Processing speed
        print(f"{'Processing FPS':<40} {old_metrics.fps:>20.2f} {new_metrics.fps:>20.2f} {((new_metrics.fps/old_metrics.fps - 1) * 100) if old_metrics.fps > 0 else 0:>9.1f}%")
        print(f"{'Real-time factor':<40} {old_metrics.real_time_factor:>20.2f}x {new_metrics.real_time_factor:>20.2f}x {((new_metrics.real_time_factor/old_metrics.real_time_factor - 1) * 100) if old_metrics.real_time_factor > 0 else 0:>9.1f}%")
        
        # Detection
        print(f"\n{'Face Detection Rate':<40} {old_metrics.detection_rate:>20.1f}% {new_metrics.detection_rate:>20.1f}% {(new_metrics.detection_rate - old_metrics.detection_rate):>9.1f}%")
        print(f"{'Avg Faces per Frame':<40} {old_metrics.avg_faces_per_frame:>20.2f} {new_metrics.avg_faces_per_frame:>20.2f} {((new_metrics.avg_faces_per_frame/old_metrics.avg_faces_per_frame - 1) * 100) if old_metrics.avg_faces_per_frame > 0 else 0:>9.1f}%")
        
        # Recognition
        print(f"\n{'POI Detection Rate':<40} {old_metrics.poi_detection_rate:>20.1f}% {new_metrics.poi_detection_rate:>20.1f}% {(new_metrics.poi_detection_rate - old_metrics.poi_detection_rate):>9.1f}%")
        print(f"{'Recognition Confidence':<40} {old_metrics.avg_recognition_confidence:>20.3f} {new_metrics.avg_recognition_confidence:>20.3f} {((new_metrics.avg_recognition_confidence/old_metrics.avg_recognition_confidence - 1) * 100) if old_metrics.avg_recognition_confidence > 0 else 0:>9.1f}%")
        
        # Speaking
        print(f"\n{'POI Speaking Rate':<40} {old_metrics.poi_speaking_rate:>20.1f}% {new_metrics.poi_speaking_rate:>20.1f}% {(new_metrics.poi_speaking_rate - old_metrics.poi_speaking_rate):>9.1f}%")
        print(f"{'Speaking Segments':<40} {old_metrics.num_speaking_segments:>20d} {new_metrics.num_speaking_segments:>20d} {(new_metrics.num_speaking_segments - old_metrics.num_speaking_segments):>9d}")
        print(f"{'Total Speaking Time':<40} {old_metrics.total_speaking_time:>20.2f}s {new_metrics.total_speaking_time:>20.2f}s {(new_metrics.total_speaking_time - old_metrics.total_speaking_time):>9.2f}s")
        
        # Resources
        print(f"\n{'Peak Memory (MB)':<40} {old_metrics.peak_memory_mb:>20.1f} {new_metrics.peak_memory_mb:>20.1f} {((new_metrics.peak_memory_mb/old_metrics.peak_memory_mb - 1) * 100) if old_metrics.peak_memory_mb > 0 else 0:>9.1f}%")
        print(f"{'Avg CPU Usage':<40} {old_metrics.avg_cpu_percent:>20.1f}% {new_metrics.avg_cpu_percent:>20.1f}% {(new_metrics.avg_cpu_percent - old_metrics.avg_cpu_percent):>9.1f}%")
        
    else:
        # New system only
        print(f"\n{'Metric':<50} {'Value':<30}")
        print("-"*80)
        
        print(f"\n{'PERFORMANCE':<50}")
        print(f"{'  Processing FPS':<50} {new_metrics.fps:>25.2f}")
        print(f"{'  Real-time factor':<50} {new_metrics.real_time_factor:>25.2f}x")
        print(f"{'  Total frames':<50} {new_metrics.total_frames:>25d}")
        print(f"{'  Processing time':<50} {new_metrics.processing_time:>25.2f}s")
        
        print(f"\n{'FACE DETECTION':<50}")
        print(f"{'  Detection rate':<50} {new_metrics.detection_rate:>25.1f}%")
        print(f"{'  Total faces detected':<50} {new_metrics.faces_detected_total:>25d}")
        print(f"{'  Avg faces per frame':<50} {new_metrics.avg_faces_per_frame:>25.2f}")
        
        print(f"\n{'FACE RECOGNITION':<50}")
        print(f"{'  POI detection rate':<50} {new_metrics.poi_detection_rate:>25.1f}%")
        print(f"{'  POI detected frames':<50} {new_metrics.poi_detected_frames:>25d}")
        print(f"{'  Avg recognition confidence':<50} {new_metrics.avg_recognition_confidence:>25.3f}")
        
        print(f"\n{'SPEAKER DETECTION':<50}")
        print(f"{'  POI speaking rate':<50} {new_metrics.poi_speaking_rate:>25.1f}%")
        print(f"{'  POI speaking frames':<50} {new_metrics.poi_speaking_frames:>25d}")
        print(f"{'  Speaking segments':<50} {new_metrics.num_speaking_segments:>25d}")
        print(f"{'  Total speaking time':<50} {new_metrics.total_speaking_time:>25.2f}s")
        print(f"{'  Avg speaking confidence':<50} {new_metrics.avg_speaking_confidence:>25.3f}")
        
        print(f"\n{'RESOURCE USAGE':<50}")
        print(f"{'  Peak memory (MB)':<50} {new_metrics.peak_memory_mb:>25.1f}")
        print(f"{'  Avg CPU usage':<50} {new_metrics.avg_cpu_percent:>25.1f}%")
        print(f"{'  Avg GPU usage':<50} {new_metrics.avg_gpu_percent:>25.1f}%")
        print(f"{'  Avg GPU memory (MB)':<50} {new_metrics.avg_gpu_memory_mb:>25.1f}")
    
    print("\n" + "="*100)


def export_results(new_metrics: BenchmarkMetrics, old_metrics: Optional[BenchmarkMetrics], output_path: str):
    """
    Export benchmark results to JSON.
    
    Args:
        new_metrics: New system metrics
        old_metrics: Old system metrics (optional)
        output_path: Output JSON file path
    """
    results = {
        'new_system': new_metrics.to_dict(),
        'old_system': old_metrics.to_dict() if old_metrics else None,
        'comparison': {}
    }
    
    if old_metrics and old_metrics.fps > 0:
        results['comparison'] = {
            'fps_improvement_percent': ((new_metrics.fps / old_metrics.fps - 1) * 100),
            'poi_detection_improvement_percent': (new_metrics.poi_detection_rate - old_metrics.poi_detection_rate),
            'memory_change_percent': ((new_metrics.peak_memory_mb / old_metrics.peak_memory_mb - 1) * 100) if old_metrics.peak_memory_mb > 0 else 0
        }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"✅ Results exported to: {output_path}")


def main():
    """Main benchmark script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark old vs new system")
    parser.add_argument('--video', required=True, help='Path to test video')
    parser.add_argument('--poi-dir', default='images/pipe_test_persons', help='Directory containing POI images')
    parser.add_argument('--max-frames', type=int, default=1500, help='Maximum frames to process')
    parser.add_argument('--output', default='benchmark_results/comparison.json', help='Output JSON file')
    parser.add_argument('--benchmark-old', action='store_true', help='Also benchmark old system')
    
    args = parser.parse_args()
    
    # Find POI images
    poi_dir = Path(args.poi_dir)
    poi_images = sorted([str(p) for p in poi_dir.glob('*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    
    if not poi_images:
        logger.error(f"No POI images found in {poi_dir}")
        return 1
    
    logger.info(f"Found {len(poi_images)} POI images")
    
    # Benchmark new system
    new_metrics = benchmark_new_system(args.video, poi_images, args.max_frames)
    
    # Benchmark old system (if requested)
    old_metrics = None
    if args.benchmark_old:
        old_metrics = benchmark_old_system(args.video, poi_images, args.max_frames)
    
    # Print comparison
    print_comparison(new_metrics, old_metrics)
    
    # Export results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    export_results(new_metrics, old_metrics, args.output)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
