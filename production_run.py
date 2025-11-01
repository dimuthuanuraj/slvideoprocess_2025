"""
Production Video Processing Script

Batch process videos with optimized pipeline, error handling, progress tracking,
and resumption capability.

Usage:
    python production_run.py --video-dir /path/to/videos --poi-dir /path/to/poi --output-dir /path/to/output
    python production_run.py --video-list videos.txt --poi-dir /path/to/poi --output-dir /path/to/output
    python production_run.py --resume /path/to/checkpoint.json

Features:
    - Batch video processing with progress tracking
    - Automatic error recovery and resumption
    - Per-video POI configuration
    - Organized output structure
    - Performance monitoring and reporting
    - JSON checkpoint for crash recovery
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import time
from datetime import datetime, timedelta
import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from slceleb_modern.pipeline import IntegratedPipeline
from slceleb_modern.recognition.face_recognizer_optimized import OptimizedFaceRecognizer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('production_run.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class VideoJobConfig:
    """Configuration for a single video processing job."""
    video_path: str
    poi_images: List[str]
    output_dir: str
    max_frames: Optional[int] = None
    detection_confidence: float = 0.5
    recognition_threshold: float = 0.252
    speaking_threshold: float = 0.5


@dataclass
class VideoJobResult:
    """Result of processing a single video."""
    video_path: str
    success: bool
    frames_processed: int
    processing_time: float
    fps: float
    poi_detections: int
    poi_speaking_frames: int
    speaking_segments: int
    error_message: Optional[str] = None
    output_files: List[str] = None


@dataclass
class BatchProcessingCheckpoint:
    """Checkpoint for resuming batch processing."""
    start_time: str
    completed_videos: List[str]
    failed_videos: List[Dict]
    current_video: Optional[str]
    total_videos: int
    checkpoint_time: str


class ProductionPipeline:
    """
    Production-ready pipeline with optimized performance and error handling.
    """
    
    def __init__(
        self,
        use_optimized: bool = True,
        model_name: str = "buffalo_s",
        cache_size: int = 100
    ):
        """
        Initialize production pipeline.
        
        Args:
            use_optimized: Use optimized recognizer with caching
            model_name: InsightFace model (buffalo_s for speed, buffalo_l for accuracy)
            cache_size: Embedding cache size
        """
        self.use_optimized = use_optimized
        self.model_name = model_name
        self.cache_size = cache_size
        
        logger.info(f"Initializing ProductionPipeline (optimized={use_optimized}, model={model_name})")
        
        # Initialize pipeline with optimized recognizer
        if use_optimized:
            # Create pipeline but replace recognizer with optimized version
            self.pipeline = IntegratedPipeline()
            self.pipeline.recognizer = OptimizedFaceRecognizer(
                model_name=model_name,
                use_gpu=True,
                cache_size=cache_size
            )
            logger.info("✓ Optimized pipeline initialized")
        else:
            # Use standard pipeline
            self.pipeline = IntegratedPipeline()
            logger.info("✓ Standard pipeline initialized")
    
    def process_video(self, config: VideoJobConfig) -> VideoJobResult:
        """
        Process a single video.
        
        Args:
            config: Video processing configuration
            
        Returns:
            Processing result
        """
        video_path = config.video_path
        logger.info(f"Processing video: {video_path}")
        
        start_time = time.time()
        
        try:
            # Load POI references
            self.pipeline.load_poi_references(config.poi_images)
            
            # Process video
            results = self.pipeline.process_video(
                video_path=video_path,
                max_frames=config.max_frames,
                show_progress=True
            )
            
            processing_time = time.time() - start_time
            
            # Calculate statistics from VideoResults object
            total_frames = len(results.frame_results)
            poi_frames = results.frames_with_poi
            speaking_frames = results.frames_with_poi_speaking
            segments = results.speaking_segments
            
            # Extract speaking segments if not already done
            if not segments and hasattr(self.pipeline, 'extract_speaking_segments'):
                segments = self.pipeline.extract_speaking_segments(results.frame_results)
            
            # Save results
            output_dir = Path(config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save JSON results
            json_path = output_dir / f"{Path(video_path).stem}_results.json"
            self._save_results_json(results, segments, json_path)
            
            # Save speaking segments video (optional)
            segment_files = []
            if segments:
                segment_files = self._save_speaking_segments(
                    video_path, segments, output_dir
                )
            
            # Calculate cache statistics if using optimized
            cache_stats = {}
            if self.use_optimized:
                cache_stats = self.pipeline.recognizer.get_cache_stats()
                logger.info(f"Cache stats: {cache_stats}")
            
            logger.info(f"✓ Completed: {total_frames} frames, {processing_time:.1f}s, "
                       f"{total_frames/processing_time:.2f} FPS, "
                       f"{poi_frames} POI frames, {speaking_frames} speaking, "
                       f"{len(segments)} segments")
            
            return VideoJobResult(
                video_path=video_path,
                success=True,
                frames_processed=total_frames,
                processing_time=processing_time,
                fps=total_frames / processing_time if processing_time > 0 else 0,
                poi_detections=poi_frames,
                poi_speaking_frames=speaking_frames,
                speaking_segments=len(segments),
                error_message=None,
                output_files=[str(json_path)] + segment_files
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Error processing {video_path}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            return VideoJobResult(
                video_path=video_path,
                success=False,
                frames_processed=0,
                processing_time=processing_time,
                fps=0,
                poi_detections=0,
                poi_speaking_frames=0,
                speaking_segments=0,
                error_message=error_msg
            )
    
    def _save_results_json(self, results, segments: List, output_path: Path):
        """Save processing results to JSON."""
        # Handle both VideoResults object and list of FrameResult
        if hasattr(results, 'frame_results'):
            frame_results = results.frame_results
            total_frames = len(frame_results)
            poi_frames = results.frames_with_poi
            speaking_frames = results.frames_with_poi_speaking
        else:
            frame_results = results
            total_frames = len(results)
            poi_frames = sum(1 for r in results if r.poi_present)
            speaking_frames = sum(1 for r in results if r.poi_speaking)
        
        data = {
            'total_frames': total_frames,
            'poi_frames': poi_frames,
            'speaking_frames': speaking_frames,
            'speaking_segments': len(segments),
            'segments': segments,  # Already in correct format from VideoResults
            'frame_results': [
                {
                    'frame': r.frame_idx,
                    'timestamp': r.timestamp,
                    'poi_present': r.poi_present,
                    'poi_speaking': r.poi_speaking,
                    'num_faces': len(r.face_bboxes)
                }
                for r in frame_results
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved results to {output_path}")
    
    def _save_speaking_segments(
        self, 
        video_path: str, 
        segments: List,
        output_dir: Path
    ) -> List[str]:
        """Extract and save speaking segments as separate video files with audio."""
        import subprocess
        
        # Get video FPS for time calculation
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        segment_files = []
        video_stem = Path(video_path).stem
        
        for i, segment in enumerate(segments):
            # Handle both tuple (start_time, end_time, confidence) and dict formats
            if isinstance(segment, tuple):
                start_time, end_time, confidence = segment
            else:
                start_frame = segment['start_frame']
                end_frame = segment['end_frame']
                start_time = start_frame / fps
                end_time = end_frame / fps
            
            duration = end_time - start_time
            
            # Create output file
            segment_file = output_dir / f"{video_stem}_segment_{i+1:03d}.mp4"
            
            # Use ffmpeg to extract segment with audio
            # -ss: start time, -t: duration, -c copy: copy codecs without re-encoding
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
                segment_files.append(str(segment_file))
                logger.info(f"Saved segment {i+1}: {segment_file}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to extract segment {i+1}: {e}")
        
        return segment_files


class BatchProcessor:
    """
    Batch video processor with checkpointing and resumption.
    """
    
    def __init__(
        self,
        checkpoint_file: Optional[str] = None,
        checkpoint_interval: int = 5
    ):
        """
        Initialize batch processor.
        
        Args:
            checkpoint_file: Path to checkpoint file for resumption
            checkpoint_interval: Save checkpoint every N videos
        """
        self.checkpoint_file = checkpoint_file or "batch_checkpoint.json"
        self.checkpoint_interval = checkpoint_interval
        self.completed_videos = []
        self.failed_videos = []
        self.start_time = None
        
        # Load checkpoint if exists
        if Path(self.checkpoint_file).exists():
            self._load_checkpoint()
    
    def process_batch(
        self,
        video_configs: List[VideoJobConfig],
        pipeline: ProductionPipeline
    ) -> Dict:
        """
        Process a batch of videos with checkpointing.
        
        Args:
            video_configs: List of video processing configurations
            pipeline: Production pipeline instance
            
        Returns:
            Batch processing summary
        """
        self.start_time = datetime.now()
        total_videos = len(video_configs)
        
        logger.info(f"Starting batch processing: {total_videos} videos")
        
        # Filter out already completed videos
        remaining_configs = [
            cfg for cfg in video_configs 
            if cfg.video_path not in self.completed_videos
        ]
        
        if len(remaining_configs) < total_videos:
            logger.info(f"Resuming: {len(remaining_configs)} videos remaining")
        
        # Process each video
        results = []
        for i, config in enumerate(tqdm(remaining_configs, desc="Processing videos")):
            logger.info(f"\n{'='*80}")
            logger.info(f"Video {i+1}/{len(remaining_configs)}: {config.video_path}")
            logger.info(f"{'='*80}")
            
            result = pipeline.process_video(config)
            results.append(result)
            
            if result.success:
                self.completed_videos.append(config.video_path)
            else:
                self.failed_videos.append({
                    'video': config.video_path,
                    'error': result.error_message
                })
            
            # Save checkpoint periodically
            if (i + 1) % self.checkpoint_interval == 0:
                self._save_checkpoint(total_videos, config.video_path)
        
        # Final checkpoint
        self._save_checkpoint(total_videos, None)
        
        # Generate summary
        summary = self._generate_summary(results)
        
        return summary
    
    def _save_checkpoint(self, total_videos: int, current_video: Optional[str]):
        """Save processing checkpoint."""
        checkpoint = BatchProcessingCheckpoint(
            start_time=self.start_time.isoformat(),
            completed_videos=self.completed_videos,
            failed_videos=self.failed_videos,
            current_video=current_video,
            total_videos=total_videos,
            checkpoint_time=datetime.now().isoformat()
        )
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(asdict(checkpoint), f, indent=2)
        
        logger.info(f"Checkpoint saved: {len(self.completed_videos)}/{total_videos} completed")
    
    def _load_checkpoint(self):
        """Load processing checkpoint."""
        try:
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)
            
            self.completed_videos = data['completed_videos']
            self.failed_videos = data['failed_videos']
            self.start_time = datetime.fromisoformat(data['start_time'])
            
            logger.info(f"Loaded checkpoint: {len(self.completed_videos)} completed, "
                       f"{len(self.failed_videos)} failed")
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}")
    
    def _generate_summary(self, results: List[VideoJobResult]) -> Dict:
        """Generate batch processing summary."""
        total = len(results)
        successful = sum(1 for r in results if r.success)
        failed = total - successful
        
        total_frames = sum(r.frames_processed for r in results if r.success)
        total_time = sum(r.processing_time for r in results if r.success)
        avg_fps = total_frames / total_time if total_time > 0 else 0
        
        total_poi_frames = sum(r.poi_detections for r in results if r.success)
        total_speaking_frames = sum(r.poi_speaking_frames for r in results if r.success)
        total_segments = sum(r.speaking_segments for r in results if r.success)
        
        elapsed = datetime.now() - self.start_time
        
        summary = {
            'batch_info': {
                'total_videos': total,
                'successful': successful,
                'failed': failed,
                'success_rate': f"{successful/total*100:.1f}%" if total > 0 else "0%",
                'elapsed_time': str(elapsed),
                'start_time': self.start_time.isoformat(),
                'end_time': datetime.now().isoformat()
            },
            'processing_stats': {
                'total_frames': total_frames,
                'total_processing_time': f"{total_time:.1f}s",
                'average_fps': f"{avg_fps:.2f}",
                'total_poi_frames': total_poi_frames,
                'total_speaking_frames': total_speaking_frames,
                'total_segments': total_segments
            },
            'failed_videos': self.failed_videos,
            'video_results': [asdict(r) for r in results]
        }
        
        return summary


def load_video_configs_from_dir(
    video_dir: str,
    poi_dir: str,
    output_dir: str,
    **kwargs
) -> List[VideoJobConfig]:
    """Load video configurations from a directory."""
    video_dir = Path(video_dir)
    poi_dir = Path(poi_dir)
    
    # Find all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(video_dir.glob(f"*{ext}"))
    
    # Find POI images
    poi_images = sorted([
        str(p) for p in poi_dir.glob('*')
        if p.suffix.lower() in ['.jpg', '.jpeg', '.png']
    ])
    
    if not poi_images:
        raise ValueError(f"No POI images found in {poi_dir}")
    
    logger.info(f"Found {len(video_files)} videos and {len(poi_images)} POI images")
    
    # Create configs
    configs = []
    for video_file in sorted(video_files):
        config = VideoJobConfig(
            video_path=str(video_file),
            poi_images=poi_images,
            output_dir=str(Path(output_dir) / video_file.stem),
            **kwargs
        )
        configs.append(config)
    
    return configs


def load_video_configs_from_list(
    video_list_file: str,
    poi_dir: str,
    output_dir: str,
    **kwargs
) -> List[VideoJobConfig]:
    """Load video configurations from a text file list."""
    poi_dir = Path(poi_dir)
    
    # Load video paths
    with open(video_list_file, 'r') as f:
        video_paths = [line.strip() for line in f if line.strip()]
    
    # Find POI images
    poi_images = sorted([
        str(p) for p in poi_dir.glob('*')
        if p.suffix.lower() in ['.jpg', '.jpeg', '.png']
    ])
    
    if not poi_images:
        raise ValueError(f"No POI images found in {poi_dir}")
    
    logger.info(f"Loaded {len(video_paths)} videos from {video_list_file}")
    logger.info(f"Found {len(poi_images)} POI images")
    
    # Create configs
    configs = []
    for video_path in video_paths:
        if not Path(video_path).exists():
            logger.warning(f"Video not found: {video_path}")
            continue
        
        config = VideoJobConfig(
            video_path=video_path,
            poi_images=poi_images,
            output_dir=str(Path(output_dir) / Path(video_path).stem),
            **kwargs
        )
        configs.append(config)
    
    return configs


def main():
    """Main production script."""
    parser = argparse.ArgumentParser(
        description="Production video processing with optimized pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all videos in a directory
  python production_run.py --video-dir /path/to/videos --poi-dir /path/to/poi --output-dir /path/to/output
  
  # Process videos from a list file
  python production_run.py --video-list videos.txt --poi-dir /path/to/poi --output-dir /path/to/output
  
  # Resume from checkpoint
  python production_run.py --resume batch_checkpoint.json
  
  # Use standard (non-optimized) pipeline
  python production_run.py --video-dir /path/to/videos --poi-dir /path/to/poi --output-dir /path/to/output --no-optimize
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--video-dir', help='Directory containing videos to process')
    input_group.add_argument('--video-list', help='Text file with list of video paths')
    input_group.add_argument('--resume', help='Resume from checkpoint file')
    
    # Configuration
    parser.add_argument('--poi-dir', help='Directory containing POI reference images')
    parser.add_argument('--output-dir', default='production_output', help='Output directory')
    parser.add_argument('--checkpoint-file', default='batch_checkpoint.json', help='Checkpoint file path')
    parser.add_argument('--checkpoint-interval', type=int, default=5, help='Save checkpoint every N videos')
    
    # Pipeline options
    parser.add_argument('--no-optimize', action='store_true', help='Use standard pipeline (not optimized)')
    parser.add_argument('--model', default='buffalo_s', choices=['buffalo_s', 'buffalo_l'], 
                       help='InsightFace model')
    parser.add_argument('--cache-size', type=int, default=100, help='Embedding cache size')
    
    # Processing options
    parser.add_argument('--max-frames', type=int, help='Maximum frames per video (for testing)')
    parser.add_argument('--detection-confidence', type=float, default=0.5, help='Face detection confidence')
    parser.add_argument('--recognition-threshold', type=float, default=0.252, help='Face recognition threshold')
    parser.add_argument('--speaking-threshold', type=float, default=0.5, help='Speaking detection threshold')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ProductionPipeline(
        use_optimized=not args.no_optimize,
        model_name=args.model,
        cache_size=args.cache_size
    )
    
    # Initialize batch processor
    batch_processor = BatchProcessor(
        checkpoint_file=args.checkpoint_file,
        checkpoint_interval=args.checkpoint_interval
    )
    
    # Load video configurations
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        # Checkpoint already loaded in BatchProcessor.__init__
        # Need to reload the original configs - this requires storing them in checkpoint
        logger.error("Resume functionality requires checkpoint to store original configs")
        logger.error("For now, please re-run with original --video-dir or --video-list")
        return 1
    
    elif args.video_dir:
        if not args.poi_dir or not args.output_dir:
            parser.error("--poi-dir and --output-dir required with --video-dir")
        
        configs = load_video_configs_from_dir(
            args.video_dir,
            args.poi_dir,
            args.output_dir,
            max_frames=args.max_frames,
            detection_confidence=args.detection_confidence,
            recognition_threshold=args.recognition_threshold,
            speaking_threshold=args.speaking_threshold
        )
    
    elif args.video_list:
        if not args.poi_dir or not args.output_dir:
            parser.error("--poi-dir and --output-dir required with --video-list")
        
        configs = load_video_configs_from_list(
            args.video_list,
            args.poi_dir,
            args.output_dir,
            max_frames=args.max_frames,
            detection_confidence=args.detection_confidence,
            recognition_threshold=args.recognition_threshold,
            speaking_threshold=args.speaking_threshold
        )
    
    if not configs:
        logger.error("No videos to process")
        return 1
    
    # Process batch
    logger.info(f"\n{'='*80}")
    logger.info("STARTING BATCH PROCESSING")
    logger.info(f"{'='*80}\n")
    
    summary = batch_processor.process_batch(configs, pipeline)
    
    # Save summary
    summary_file = Path(args.output_dir) / 'batch_summary.json'
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info(f"{'='*80}\n")
    logger.info(f"Total Videos: {summary['batch_info']['total_videos']}")
    logger.info(f"Successful: {summary['batch_info']['successful']}")
    logger.info(f"Failed: {summary['batch_info']['failed']}")
    logger.info(f"Success Rate: {summary['batch_info']['success_rate']}")
    logger.info(f"Elapsed Time: {summary['batch_info']['elapsed_time']}")
    logger.info(f"\nTotal Frames Processed: {summary['processing_stats']['total_frames']}")
    logger.info(f"Average FPS: {summary['processing_stats']['average_fps']}")
    logger.info(f"POI Frames Detected: {summary['processing_stats']['total_poi_frames']}")
    logger.info(f"Speaking Frames: {summary['processing_stats']['total_speaking_frames']}")
    logger.info(f"Speaking Segments: {summary['processing_stats']['total_segments']}")
    logger.info(f"\nSummary saved to: {summary_file}")
    
    if summary['failed_videos']:
        logger.warning(f"\n{len(summary['failed_videos'])} videos failed:")
        for failed in summary['failed_videos']:
            logger.warning(f"  - {failed['video']}: {failed['error']}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
