"""
Demo Visualization Script

Creates an annotated video showing real-time detection, recognition, and speaker tracking.
Perfect for demonstrations and presentations.

Features:
    - Face detection with landmarks (MediaPipe 478 points)
    - POI recognition with confidence scores
    - Speaking detection with visual indicators
    - Real-time performance metrics
    - Color-coded annotations

Usage:
    python demo_visualization.py --video input.mp4 --poi-dir poi_images/ --output demo_output.mp4
    python demo_visualization.py --video input.mp4 --poi-dir poi_images/ --output demo_output.mp4 --max-frames 500
"""

import sys
import argparse
import logging
from pathlib import Path
import cv2
import numpy as np
from typing import List, Tuple, Optional
import time
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent))

from slceleb_modern.pipeline import IntegratedPipeline
from slceleb_modern.recognition.face_recognizer_optimized import OptimizedFaceRecognizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DemoVisualizer:
    """
    Create annotated demo videos with detection visualization.
    """
    
    # Color scheme (BGR format for OpenCV)
    COLOR_POI_FACE = (0, 255, 0)        # Green - POI detected
    COLOR_OTHER_FACE = (128, 128, 128)  # Gray - Other person
    COLOR_SPEAKING = (0, 255, 255)      # Yellow - Speaking
    COLOR_NOT_SPEAKING = (0, 165, 255)  # Orange - Not speaking
    COLOR_TEXT_BG = (0, 0, 0)           # Black background for text
    COLOR_TEXT = (255, 255, 255)        # White text
    COLOR_LANDMARKS = (255, 0, 255)     # Magenta - Facial landmarks
    COLOR_LIP = (0, 0, 255)             # Red - Lip region
    
    def __init__(
        self,
        pipeline: IntegratedPipeline,
        show_landmarks: bool = True,
        show_lip_region: bool = True,
        show_metrics: bool = True,
        font_scale: float = 0.6
    ):
        """
        Initialize demo visualizer.
        
        Args:
            pipeline: Integrated pipeline instance
            show_landmarks: Show facial landmarks
            show_lip_region: Highlight lip region
            show_metrics: Show performance metrics
            font_scale: Font size scale
        """
        self.pipeline = pipeline
        self.show_landmarks = show_landmarks
        self.show_lip_region = show_lip_region
        self.show_metrics = show_metrics
        self.font_scale = font_scale
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_thickness = 2
        
        # Performance tracking
        self.frame_times = []
        self.poi_count = 0
        self.speaking_count = 0
        
    def draw_text_with_background(
        self,
        frame: np.ndarray,
        text: str,
        position: Tuple[int, int],
        font_scale: float = None,
        color: Tuple[int, int, int] = None,
        bg_color: Tuple[int, int, int] = None,
        thickness: int = None
    ):
        """Draw text with background for better visibility."""
        font_scale = font_scale or self.font_scale
        color = color or self.COLOR_TEXT
        bg_color = bg_color or self.COLOR_TEXT_BG
        thickness = thickness or self.font_thickness
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, self.font, font_scale, thickness
        )
        
        x, y = position
        
        # Draw background rectangle
        cv2.rectangle(
            frame,
            (x - 5, y - text_height - 5),
            (x + text_width + 5, y + baseline + 5),
            bg_color,
            -1
        )
        
        # Draw text
        cv2.putText(
            frame,
            text,
            (x, y),
            self.font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA
        )
        
        return text_height + baseline + 10
    
    def draw_face_box(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        label: str,
        confidence: float,
        is_poi: bool,
        is_speaking: bool
    ):
        """Draw bounding box with label and confidence."""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Choose color based on status
        if is_poi:
            if is_speaking:
                color = self.COLOR_SPEAKING
                status = "SPEAKING"
            else:
                color = self.COLOR_POI_FACE
                status = "SILENT"
            box_thickness = 3
        else:
            color = self.COLOR_OTHER_FACE
            status = "OTHER"
            box_thickness = 2
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)
        
        # Draw label
        if is_poi:
            label_text = f"{status} ({confidence:.2f})"
        else:
            label_text = "NOT POI"
        
        self.draw_text_with_background(
            frame,
            label_text,
            (x1, y1 - 10),
            font_scale=self.font_scale * 0.9,
            color=self.COLOR_TEXT,
            bg_color=color
        )
    
    def draw_landmarks(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
        show_all: bool = False
    ):
        """Draw facial landmarks."""
        if landmarks is None or len(landmarks) == 0:
            return
        
        # Draw subset of landmarks for clarity (or all if requested)
        if show_all:
            # Draw all 478 landmarks
            for landmark in landmarks:
                x, y = int(landmark[0]), int(landmark[1])
                cv2.circle(frame, (x, y), 1, self.COLOR_LANDMARKS, -1)
        else:
            # Draw key landmarks only (eyes, nose, mouth outline)
            # MediaPipe landmark indices
            key_indices = [
                # Right eye
                33, 133, 160, 159, 158, 157, 173,
                # Left eye  
                263, 362, 387, 386, 385, 384, 398,
                # Nose
                1, 2, 98, 327,
                # Mouth outline
                61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
                # Face oval
                10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                93, 67, 109, 10
            ]
            
            for idx in key_indices:
                if idx < len(landmarks):
                    x, y = int(landmarks[idx][0]), int(landmarks[idx][1])
                    cv2.circle(frame, (x, y), 2, self.COLOR_LANDMARKS, -1)
    
    def draw_lip_region(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray
    ):
        """Highlight lip region."""
        if landmarks is None or len(landmarks) == 0:
            return
        
        # MediaPipe lip landmark indices
        upper_lip_indices = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
        lower_lip_indices = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
        
        # Get lip points
        try:
            upper_points = np.array([
                [int(landmarks[i][0]), int(landmarks[i][1])] 
                for i in upper_lip_indices if i < len(landmarks)
            ])
            lower_points = np.array([
                [int(landmarks[i][0]), int(landmarks[i][1])] 
                for i in lower_lip_indices if i < len(landmarks)
            ])
            
            # Draw lip contours
            if len(upper_points) > 0:
                cv2.polylines(frame, [upper_points], False, self.COLOR_LIP, 2)
            if len(lower_points) > 0:
                cv2.polylines(frame, [lower_points], False, self.COLOR_LIP, 2)
        except Exception as e:
            logger.debug(f"Could not draw lip region: {e}")
    
    def draw_metrics_panel(
        self,
        frame: np.ndarray,
        frame_idx: int,
        fps: float,
        poi_detected: bool,
        speaking: bool,
        cache_stats: dict = None
    ):
        """Draw performance metrics panel."""
        h, w = frame.shape[:2]
        panel_height = 180
        panel_y = h - panel_height
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, panel_y), (w, h), self.COLOR_TEXT_BG, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Metrics text
        y_offset = panel_y + 30
        line_height = 35
        
        # Frame info
        self.draw_text_with_background(
            frame,
            f"Frame: {frame_idx}  |  FPS: {fps:.1f}",
            (20, y_offset),
            font_scale=self.font_scale * 1.2,
            bg_color=(0, 0, 0, 0)  # Transparent
        )
        y_offset += line_height
        
        # Detection status
        poi_status = "POI DETECTED" if poi_detected else "No POI"
        poi_color = self.COLOR_POI_FACE if poi_detected else self.COLOR_OTHER_FACE
        self.draw_text_with_background(
            frame,
            poi_status,
            (20, y_offset),
            font_scale=self.font_scale * 1.1,
            color=poi_color,
            bg_color=(0, 0, 0, 0)
        )
        y_offset += line_height
        
        # Speaking status
        if poi_detected:
            speak_status = "SPEAKING" if speaking else "SILENT"
            speak_color = self.COLOR_SPEAKING if speaking else self.COLOR_NOT_SPEAKING
            self.draw_text_with_background(
                frame,
                speak_status,
                (20, y_offset),
                font_scale=self.font_scale * 1.1,
                color=speak_color,
                bg_color=(0, 0, 0, 0)
            )
            y_offset += line_height
        
        # Cache statistics (if available)
        if cache_stats and 'hit_rate' in cache_stats:
            cache_text = f"Cache Hit Rate: {cache_stats['hit_rate']:.1f}%"
            self.draw_text_with_background(
                frame,
                cache_text,
                (20, y_offset),
                font_scale=self.font_scale * 0.9,
                color=(100, 255, 100),
                bg_color=(0, 0, 0, 0)
            )
        
        # Statistics on right side
        stats_x = w - 280
        y_offset = panel_y + 30
        
        self.draw_text_with_background(
            frame,
            f"Total POI Frames: {self.poi_count}",
            (stats_x, y_offset),
            font_scale=self.font_scale * 0.9,
            bg_color=(0, 0, 0, 0)
        )
        y_offset += line_height
        
        self.draw_text_with_background(
            frame,
            f"Speaking Frames: {self.speaking_count}",
            (stats_x, y_offset),
            font_scale=self.font_scale * 0.9,
            bg_color=(0, 0, 0, 0)
        )
    
    def draw_legend(self, frame: np.ndarray):
        """Draw color legend."""
        h, w = frame.shape[:2]
        legend_x = w - 300
        legend_y = 20
        box_size = 20
        spacing = 30
        
        # Legend items
        items = [
            ("POI Detected", self.COLOR_POI_FACE),
            ("Speaking", self.COLOR_SPEAKING),
            ("Other Person", self.COLOR_OTHER_FACE)
        ]
        
        for i, (label, color) in enumerate(items):
            y = legend_y + i * spacing
            
            # Draw color box
            cv2.rectangle(
                frame,
                (legend_x, y),
                (legend_x + box_size, y + box_size),
                color,
                -1
            )
            cv2.rectangle(
                frame,
                (legend_x, y),
                (legend_x + box_size, y + box_size),
                (255, 255, 255),
                1
            )
            
            # Draw label
            cv2.putText(
                frame,
                label,
                (legend_x + box_size + 10, y + box_size - 5),
                self.font,
                self.font_scale * 0.7,
                self.COLOR_TEXT,
                1,
                cv2.LINE_AA
            )
    
    def annotate_frame(
        self,
        frame: np.ndarray,
        frame_result,
        frame_idx: int,
        fps: float,
        cache_stats: dict = None
    ) -> np.ndarray:
        """Annotate a single frame with all visualizations."""
        annotated = frame.copy()
        
        # Draw legend
        self.draw_legend(annotated)
        
        # Draw face boxes and landmarks
        if len(frame_result.face_bboxes) > 0:
            for i, bbox in enumerate(frame_result.face_bboxes):
                # Get POI status
                is_poi = frame_result.poi_present and (frame_result.poi_index == i if frame_result.poi_index is not None else i == 0)
                confidence = frame_result.face_confidences[i] if i < len(frame_result.face_confidences) else 0.0
                is_speaking = (frame_result.poi_speaking and is_poi) or (i < len(frame_result.is_speaking) and frame_result.is_speaking[i])
                
                # Draw face box
                self.draw_face_box(
                    annotated,
                    bbox,
                    "POI" if is_poi else "Person",
                    confidence,
                    is_poi,
                    is_speaking
                )
                
                # Draw landmarks for POI only
                if is_poi and self.show_landmarks and i < len(frame_result.face_landmarks):
                    landmarks = frame_result.face_landmarks[i]
                    self.draw_landmarks(annotated, landmarks, show_all=False)
                    
                    if self.show_lip_region and is_speaking:
                        self.draw_lip_region(annotated, landmarks)
        
        # Draw metrics panel
        if self.show_metrics:
            self.draw_metrics_panel(
                annotated,
                frame_idx,
                fps,
                frame_result.poi_present,
                frame_result.poi_speaking,
                cache_stats
            )
        
        # Update counters
        if frame_result.poi_present:
            self.poi_count += 1
        if frame_result.poi_speaking:
            self.speaking_count += 1
        
        return annotated
    
    def create_demo_video(
        self,
        video_path: str,
        output_path: str,
        max_frames: Optional[int] = None
    ):
        """
        Create annotated demo video.
        
        Args:
            video_path: Input video path
            output_path: Output video path
            max_frames: Maximum frames to process (None for all)
        """
        logger.info(f"Creating demo video: {video_path} -> {output_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video: {width}x{height} @ {fps} FPS, {total_frames} frames")
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            raise ValueError(f"Could not create output video: {output_path}")
        
        # Process video through pipeline
        logger.info("Processing video...")
        results = self.pipeline.process_video(
            video_path,
            max_frames=max_frames,
            show_progress=True
        )
        
        # Reset video capture
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Annotate and write frames
        logger.info("Creating annotated video...")
        frame_idx = 0
        processing_start = time.time()
        
        for frame_result in results.frame_results:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get cache stats if available
            cache_stats = None
            if hasattr(self.pipeline.recognizer, 'get_cache_stats'):
                cache_stats = self.pipeline.recognizer.get_cache_stats()
            
            # Calculate current FPS
            elapsed = time.time() - processing_start
            current_fps = (frame_idx + 1) / elapsed if elapsed > 0 else 0
            
            # Annotate frame
            annotated = self.annotate_frame(
                frame,
                frame_result,
                frame_idx,
                current_fps,
                cache_stats
            )
            
            # Write frame
            out.write(annotated)
            frame_idx += 1
            
            if max_frames and frame_idx >= max_frames:
                break
        
        # Cleanup
        cap.release()
        out.release()
        
        total_time = time.time() - processing_start
        avg_fps = frame_idx / total_time if total_time > 0 else 0
        
        logger.info(f"✓ Demo video created: {output_path}")
        logger.info(f"  Processed {frame_idx} frames in {total_time:.1f}s ({avg_fps:.1f} FPS)")
        logger.info(f"  POI detected in {self.poi_count} frames ({self.poi_count/frame_idx*100:.1f}%)")
        logger.info(f"  Speaking detected in {self.speaking_count} frames ({self.speaking_count/frame_idx*100:.1f}%)")


def main():
    """Main demo script."""
    parser = argparse.ArgumentParser(
        description="Create annotated demo video showing detection and tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create demo video with all frames
  python demo_visualization.py --video input.mp4 --poi-dir poi/ --output demo.mp4
  
  # Create demo with first 500 frames
  python demo_visualization.py --video input.mp4 --poi-dir poi/ --output demo.mp4 --max-frames 500
  
  # Minimal visualization (no landmarks)
  python demo_visualization.py --video input.mp4 --poi-dir poi/ --output demo.mp4 --no-landmarks
  
  # Use standard (non-optimized) pipeline
  python demo_visualization.py --video input.mp4 --poi-dir poi/ --output demo.mp4 --no-optimize
        """
    )
    
    # Required arguments
    parser.add_argument('--video', required=True, help='Input video file')
    parser.add_argument('--poi-dir', required=True, help='Directory with POI reference images')
    parser.add_argument('--output', required=True, help='Output video file')
    
    # Optional arguments
    parser.add_argument('--max-frames', type=int, help='Maximum frames to process')
    parser.add_argument('--no-landmarks', action='store_true', help='Don\'t show facial landmarks')
    parser.add_argument('--no-lip-region', action='store_true', help='Don\'t highlight lip region')
    parser.add_argument('--no-metrics', action='store_true', help='Don\'t show metrics panel')
    parser.add_argument('--no-optimize', action='store_true', help='Use standard pipeline (slower)')
    parser.add_argument('--model', default='buffalo_s', choices=['buffalo_s', 'buffalo_l'],
                       help='InsightFace model')
    
    args = parser.parse_args()
    
    # Verify input files
    if not Path(args.video).exists():
        parser.error(f"Video file not found: {args.video}")
    if not Path(args.poi_dir).exists():
        parser.error(f"POI directory not found: {args.poi_dir}")
    
    # Initialize pipeline
    logger.info("Initializing pipeline...")
    pipeline = IntegratedPipeline()
    
    # Use optimized recognizer if requested
    if not args.no_optimize:
        from slceleb_modern.recognition.face_recognizer_optimized import OptimizedFaceRecognizer
        pipeline.recognizer = OptimizedFaceRecognizer(
            model_name=args.model,
            use_gpu=True,
            cache_size=100
        )
        logger.info("✓ Using optimized recognizer with caching")
    
    # Load POI references
    poi_dir = Path(args.poi_dir)
    poi_images = [
        str(p) for p in poi_dir.glob('*')
        if p.suffix.lower() in ['.jpg', '.jpeg', '.png']
    ]
    
    if not poi_images:
        parser.error(f"No POI images found in {args.poi_dir}")
    
    logger.info(f"Loading {len(poi_images)} POI reference images...")
    pipeline.load_poi_references(poi_images)
    
    # Create visualizer
    visualizer = DemoVisualizer(
        pipeline=pipeline,
        show_landmarks=not args.no_landmarks,
        show_lip_region=not args.no_lip_region,
        show_metrics=not args.no_metrics
    )
    
    # Create demo video
    visualizer.create_demo_video(
        video_path=args.video,
        output_path=args.output,
        max_frames=args.max_frames
    )
    
    logger.info("✓ Demo complete!")


if __name__ == '__main__':
    main()
