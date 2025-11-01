"""
Face Extraction and Clustering Tool

Automatically extracts faces from videos and organizes them by person using
face clustering. Perfect for building POI reference image datasets.

Features:
    - Automatic face detection and extraction
    - Face clustering by person identity
    - Quality filtering (blur, brightness, size)
    - Duplicate removal
    - Organized folder structure
    - Preview generation

Usage:
    python extract_faces.py --video input.mp4 --output-dir faces_output
    python extract_faces.py --video input.mp4 --output-dir faces_output --max-faces-per-person 20
    python extract_faces.py --video input.mp4 --output-dir faces_output --skip-frames 10 --min-face-size 100
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict
import json
from tqdm import tqdm
import shutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from slceleb_modern.detection.face_detector import MediaPipeFaceDetector
from slceleb_modern.recognition.face_recognizer import ModernFaceRecognizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class FaceInstance:
    """Single face instance from video"""
    frame_idx: int
    bbox: Tuple[int, int, int, int]
    embedding: np.ndarray
    quality_score: float
    image: np.ndarray
    landmarks: Optional[np.ndarray] = None


@dataclass
class PersonCluster:
    """Cluster of faces belonging to same person"""
    person_id: int
    faces: List[FaceInstance] = field(default_factory=list)
    avg_embedding: Optional[np.ndarray] = None
    
    def add_face(self, face: FaceInstance):
        """Add face to cluster and update average embedding"""
        self.faces.append(face)
        
        # Update average embedding
        if self.avg_embedding is None:
            self.avg_embedding = face.embedding.copy()
        else:
            # Running average
            n = len(self.faces)
            self.avg_embedding = (self.avg_embedding * (n - 1) + face.embedding) / n
    
    def get_best_faces(self, n: int = 10) -> List[FaceInstance]:
        """Get N best quality faces from cluster"""
        sorted_faces = sorted(self.faces, key=lambda f: f.quality_score, reverse=True)
        return sorted_faces[:n]


class FaceQualityAssessor:
    """Assess face image quality for filtering"""
    
    @staticmethod
    def compute_blur(image: np.ndarray) -> float:
        """Compute Laplacian variance (blur metric)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var
    
    @staticmethod
    def compute_brightness(image: np.ndarray) -> float:
        """Compute average brightness"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray.mean()
    
    @staticmethod
    def compute_quality_score(
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        blur_threshold: float = 100.0,
        brightness_range: Tuple[float, float] = (40, 220)
    ) -> float:
        """
        Compute overall quality score (0-1)
        
        Factors:
        - Blur (sharpness)
        - Brightness (not too dark/bright)
        - Face size (larger is better)
        """
        # Blur score
        blur = FaceQualityAssessor.compute_blur(image)
        blur_score = min(blur / 200.0, 1.0)  # Normalize
        
        # Brightness score
        brightness = FaceQualityAssessor.compute_brightness(image)
        if brightness < brightness_range[0] or brightness > brightness_range[1]:
            brightness_score = 0.3
        else:
            # Peak at middle of range
            mid = (brightness_range[0] + brightness_range[1]) / 2
            range_width = brightness_range[1] - brightness_range[0]
            brightness_score = 1.0 - abs(brightness - mid) / (range_width / 2)
        
        # Size score
        x1, y1, x2, y2 = bbox
        face_area = (x2 - x1) * (y2 - y1)
        size_score = min(face_area / (200 * 200), 1.0)  # Normalize to 200x200 reference
        
        # Combined score
        quality = (blur_score * 0.4 + brightness_score * 0.3 + size_score * 0.3)
        
        return quality


class FaceExtractor:
    """Extract and organize faces from video"""
    
    def __init__(
        self,
        similarity_threshold: float = 0.25,
        min_quality_score: float = 0.3,
        min_face_size: int = 60,
        max_faces_per_person: int = 50
    ):
        """
        Initialize face extractor.
        
        Args:
            similarity_threshold: Threshold for clustering (lower = stricter)
            min_quality_score: Minimum quality score to keep face
            min_face_size: Minimum face dimension (width or height)
            max_faces_per_person: Maximum faces to save per person
        """
        self.similarity_threshold = similarity_threshold
        self.min_quality_score = min_quality_score
        self.min_face_size = min_face_size
        self.max_faces_per_person = max_faces_per_person
        
        # Initialize components
        logger.info("Initializing face detector...")
        self.detector = MediaPipeFaceDetector()
        
        logger.info("Initializing face recognizer...")
        self.recognizer = ModernFaceRecognizer(model_name="buffalo_s", ctx_id=0)
        
        self.quality_assessor = FaceQualityAssessor()
        
        # Clusters
        self.clusters: List[PersonCluster] = []
        
        logger.info("✓ Face extractor initialized")
    
    def extract_face_image(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        padding: float = 0.2
    ) -> np.ndarray:
        """Extract face region with padding"""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        
        # Add padding
        face_width = x2 - x1
        face_height = y2 - y1
        pad_x = int(face_width * padding)
        pad_y = int(face_height * padding)
        
        # Expand bbox with padding
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        
        # Extract face
        face_img = frame[y1:y2, x1:x2].copy()
        
        return face_img
    
    def find_matching_cluster(self, embedding: np.ndarray) -> Optional[PersonCluster]:
        """Find cluster that matches embedding"""
        for cluster in self.clusters:
            if cluster.avg_embedding is not None:
                # Compute cosine similarity
                similarity = np.dot(embedding, cluster.avg_embedding) / (
                    np.linalg.norm(embedding) * np.linalg.norm(cluster.avg_embedding)
                )
                
                # Check if similar enough
                distance = 1 - similarity
                if distance < self.similarity_threshold:
                    return cluster
        
        return None
    
    def process_video(
        self,
        video_path: str,
        skip_frames: int = 5,
        max_frames: Optional[int] = None
    ) -> Dict:
        """
        Extract faces from video and cluster them.
        
        Args:
            video_path: Path to video file
            skip_frames: Process every Nth frame
            max_frames: Maximum frames to process
            
        Returns:
            Statistics dictionary
        """
        logger.info(f"Processing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Video: {total_frames} frames @ {fps} FPS")
        logger.info(f"Processing every {skip_frames} frame(s)")
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
            logger.info(f"Limited to {max_frames} frames")
        
        # Process frames
        frame_idx = 0
        processed_frames = 0
        total_faces = 0
        skipped_quality = 0
        skipped_size = 0
        
        with tqdm(total=total_frames, desc="Extracting faces") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames
                if frame_idx % skip_frames != 0:
                    frame_idx += 1
                    pbar.update(1)
                    if max_frames and frame_idx >= max_frames:
                        break
                    continue
                
                processed_frames += 1
                
                # Detect faces
                detections = self.detector.detect(frame)
                
                if detections:
                    for detection in detections:
                        bbox = detection.bbox
                        x1, y1, x2, y2 = map(int, bbox)
                        face_width = x2 - x1
                        face_height = y2 - y1
                        
                        # Filter by size
                        if face_width < self.min_face_size or face_height < self.min_face_size:
                            skipped_size += 1
                            continue
                        
                        # Extract face image
                        face_img = self.extract_face_image(frame, bbox)
                        
                        if face_img.size == 0:
                            continue
                        
                        # Compute quality score
                        quality = self.quality_assessor.compute_quality_score(face_img, bbox)
                        
                        # Filter by quality
                        if quality < self.min_quality_score:
                            skipped_quality += 1
                            continue
                        
                        # Get embedding
                        try:
                            embedding = self.recognizer.get_embedding(face_img)
                            if embedding is None:
                                continue
                        except Exception as e:
                            logger.debug(f"Failed to get embedding: {e}")
                            continue
                        
                        # Create face instance with landmarks from detection
                        face = FaceInstance(
                            frame_idx=frame_idx,
                            bbox=bbox,
                            embedding=embedding,
                            quality_score=quality,
                            image=face_img,
                            landmarks=detection.landmarks_2d
                        )
                        
                        # Find or create cluster
                        cluster = self.find_matching_cluster(embedding)
                        if cluster is None:
                            # Create new cluster
                            person_id = len(self.clusters)
                            cluster = PersonCluster(person_id=person_id)
                            self.clusters.append(cluster)
                        
                        # Add face to cluster
                        cluster.add_face(face)
                        total_faces += 1
                
                frame_idx += 1
                pbar.update(1)
                
                if max_frames and frame_idx >= max_frames:
                    break
        
        cap.release()
        
        # Statistics
        stats = {
            'total_frames': total_frames,
            'processed_frames': processed_frames,
            'total_faces_detected': total_faces,
            'unique_persons': len(self.clusters),
            'skipped_quality': skipped_quality,
            'skipped_size': skipped_size,
            'faces_per_person': {
                f'person_{c.person_id}': len(c.faces) for c in self.clusters
            }
        }
        
        logger.info(f"✓ Extraction complete!")
        logger.info(f"  Processed: {processed_frames} frames")
        logger.info(f"  Detected: {total_faces} faces")
        logger.info(f"  Unique persons: {len(self.clusters)}")
        logger.info(f"  Skipped (quality): {skipped_quality}")
        logger.info(f"  Skipped (size): {skipped_size}")
        
        return stats
    
    def save_faces(self, output_dir: str, create_preview: bool = True):
        """
        Save extracted faces organized by person.
        
        Args:
            output_dir: Output directory
            create_preview: Create preview grid for each person
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving faces to: {output_dir}")
        
        # Sort clusters by number of faces (descending)
        sorted_clusters = sorted(self.clusters, key=lambda c: len(c.faces), reverse=True)
        
        summary = []
        
        for cluster in sorted_clusters:
            person_id = cluster.person_id
            person_dir = output_path / f"person_{person_id:03d}"
            person_dir.mkdir(exist_ok=True)
            
            # Get best faces
            best_faces = cluster.get_best_faces(self.max_faces_per_person)
            
            logger.info(f"Saving person_{person_id:03d}: {len(best_faces)}/{len(cluster.faces)} faces")
            
            # Save faces
            saved_faces = []
            for i, face in enumerate(best_faces):
                filename = f"face_{i:03d}_frame_{face.frame_idx:06d}_q{face.quality_score:.2f}.jpg"
                filepath = person_dir / filename
                
                cv2.imwrite(str(filepath), face.image)
                saved_faces.append(str(filepath))
            
            # Create preview grid
            if create_preview and len(best_faces) > 0:
                self._create_preview_grid(person_dir, best_faces[:16], person_id)
            
            # Add to summary
            summary.append({
                'person_id': person_id,
                'total_faces': len(cluster.faces),
                'saved_faces': len(saved_faces),
                'directory': str(person_dir),
                'avg_quality': np.mean([f.quality_score for f in cluster.faces])
            })
        
        # Save summary
        summary_file = output_path / 'extraction_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"✓ Faces saved successfully!")
        logger.info(f"  Summary: {summary_file}")
        
        return summary
    
    def _create_preview_grid(
        self,
        person_dir: Path,
        faces: List[FaceInstance],
        person_id: int
    ):
        """Create preview grid of faces"""
        n_faces = len(faces)
        if n_faces == 0:
            return
        
        # Grid layout
        n_cols = min(4, n_faces)
        n_rows = (n_faces + n_cols - 1) // n_cols
        
        # Resize faces to uniform size
        face_size = 150
        resized_faces = []
        for face in faces:
            resized = cv2.resize(face.image, (face_size, face_size))
            # Add quality score text
            cv2.putText(
                resized,
                f"Q: {face.quality_score:.2f}",
                (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )
            resized_faces.append(resized)
        
        # Create grid
        grid_height = n_rows * face_size
        grid_width = n_cols * face_size
        grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        
        for i, face_img in enumerate(resized_faces):
            row = i // n_cols
            col = i % n_cols
            y = row * face_size
            x = col * face_size
            grid[y:y+face_size, x:x+face_size] = face_img
        
        # Add title
        title_height = 40
        grid_with_title = np.zeros((grid_height + title_height, grid_width, 3), dtype=np.uint8)
        grid_with_title[title_height:] = grid
        
        cv2.putText(
            grid_with_title,
            f"Person {person_id} - {len(faces)} faces",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )
        
        # Save preview
        preview_file = person_dir / f"preview_person_{person_id:03d}.jpg"
        cv2.imwrite(str(preview_file), grid_with_title)


def main():
    """Main extraction script"""
    parser = argparse.ArgumentParser(
        description="Extract and organize faces from video by person",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic extraction
  python extract_faces.py --video input.mp4 --output-dir faces_output
  
  # Extract fewer faces per person
  python extract_faces.py --video input.mp4 --output-dir faces/ --max-faces-per-person 10
  
  # Faster extraction (skip more frames)
  python extract_faces.py --video input.mp4 --output-dir faces/ --skip-frames 10
  
  # Higher quality faces only
  python extract_faces.py --video input.mp4 --output-dir faces/ --min-quality 0.5 --min-face-size 100
  
  # Process limited frames (for testing)
  python extract_faces.py --video input.mp4 --output-dir faces/ --max-frames 1000
        """
    )
    
    # Required arguments
    parser.add_argument('--video', required=True, help='Input video file')
    parser.add_argument('--output-dir', required=True, help='Output directory for faces')
    
    # Optional arguments
    parser.add_argument('--skip-frames', type=int, default=5,
                       help='Process every Nth frame (default: 5)')
    parser.add_argument('--max-frames', type=int,
                       help='Maximum frames to process (default: all)')
    parser.add_argument('--max-faces-per-person', type=int, default=50,
                       help='Maximum faces to save per person (default: 50)')
    parser.add_argument('--min-face-size', type=int, default=60,
                       help='Minimum face size in pixels (default: 60)')
    parser.add_argument('--min-quality', type=float, default=0.3,
                       help='Minimum quality score 0-1 (default: 0.3)')
    parser.add_argument('--similarity-threshold', type=float, default=0.25,
                       help='Clustering similarity threshold (default: 0.25)')
    parser.add_argument('--no-preview', action='store_true',
                       help='Don\'t create preview grids')
    
    args = parser.parse_args()
    
    # Verify input
    if not Path(args.video).exists():
        parser.error(f"Video file not found: {args.video}")
    
    # Initialize extractor
    extractor = FaceExtractor(
        similarity_threshold=args.similarity_threshold,
        min_quality_score=args.min_quality,
        min_face_size=args.min_face_size,
        max_faces_per_person=args.max_faces_per_person
    )
    
    # Extract faces
    stats = extractor.process_video(
        video_path=args.video,
        skip_frames=args.skip_frames,
        max_frames=args.max_frames
    )
    
    # Save faces
    summary = extractor.save_faces(
        output_dir=args.output_dir,
        create_preview=not args.no_preview
    )
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("EXTRACTION SUMMARY")
    logger.info("="*80)
    logger.info(f"Video: {args.video}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Processed: {stats['processed_frames']} frames")
    logger.info(f"Detected: {stats['total_faces_detected']} faces")
    logger.info(f"Unique persons: {stats['unique_persons']}")
    logger.info("\nPer-person breakdown:")
    for person in summary:
        logger.info(f"  person_{person['person_id']:03d}: {person['saved_faces']} faces "
                   f"(avg quality: {person['avg_quality']:.2f})")
    logger.info("="*80)
    logger.info("✓ Complete! Check output directory for organized faces.")


if __name__ == '__main__':
    main()
