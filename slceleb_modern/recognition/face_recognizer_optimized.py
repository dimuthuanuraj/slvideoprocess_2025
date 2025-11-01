"""
Optimized Face Recognition Module with GPU acceleration and caching
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import numpy as np
import cv2
from insightface.app import FaceAnalysis
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class RecognitionResult:
    """Face recognition result."""
    is_poi: bool
    confidence: float
    person_id: str


class OptimizedFaceRecognizer:
    """
    Optimized face recognizer with:
    - GPU acceleration via CUDA
    - Face embedding caching for temporal coherence
    - Batch processing capability
    """
    
    def __init__(
        self,
        model_name: str = "buffalo_s",  # Use smaller model for speed
        use_gpu: bool = True,
        det_size: Tuple[int, int] = (640, 640),
        cache_size: int = 100
    ):
        """
        Initialize optimized face recognizer.
        
        Args:
            model_name: InsightFace model ('buffalo_s' for speed, 'buffalo_l' for accuracy)
            use_gpu: Enable GPU acceleration
            det_size: Detection size for face analysis
            cache_size: Maximum number of cached face embeddings
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.det_size = det_size
        self.cache_size = cache_size
        
        # Face embedding cache: bbox_hash -> (embedding, frame_number)
        self.embedding_cache: Dict[int, Tuple[np.ndarray, int]] = {}
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        
        # POI references
        self.poi_embeddings: List[np.ndarray] = []
        self.poi_names: List[str] = []
        self.recognition_threshold = 0.252
        
        # Initialize InsightFace with GPU
        logger.info(f"Initializing Optimized InsightFace with model: {model_name}")
        
        providers = ['CPUExecutionProvider']
        if use_gpu:
            try:
                import onnxruntime as ort
                available = ort.get_available_providers()
                if 'CUDAExecutionProvider' in available:
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                    logger.info("✓ CUDA GPU acceleration enabled")
                elif 'TensorrtExecutionProvider' in available:
                    providers = ['TensorrtExecutionProvider', 'CPUExecutionProvider']
                    logger.info("✓ TensorRT GPU acceleration enabled")
                else:
                    logger.warning("GPU requested but no GPU provider available, using CPU")
            except Exception as e:
                logger.warning(f"Could not enable GPU: {e}")
        
        self.app = FaceAnalysis(
            name=model_name,
            providers=providers
        )
        self.app.prepare(ctx_id=0 if use_gpu else -1, det_size=det_size)
        
        logger.info(f"✅ OptimizedFaceRecognizer initialized with {model_name}")
        logger.info(f"   GPU: {'Enabled' if use_gpu else 'Disabled'}")
        logger.info(f"   Cache size: {cache_size}")
        logger.info(f"   Providers: {providers}")
    
    def _bbox_to_hash(self, bbox: np.ndarray) -> int:
        """
        Convert bounding box to hash for caching.
        Uses center and size to allow for small variations.
        
        Args:
            bbox: Bounding box as [x1, y1, x2, y2] numpy array
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        
        # Quantize to 10-pixel grid to allow small movements
        center_x_q = int(center_x / 10)
        center_y_q = int(center_y / 10)
        width_q = int(width / 10)
        height_q = int(height / 10)
        
        return hash((center_x_q, center_y_q, width_q, height_q))
    
    def get_embedding(self, frame: np.ndarray, bbox: np.ndarray, frame_number: int = 0) -> Optional[np.ndarray]:
        """
        Get face embedding with caching.
        
        Args:
            frame: Input frame
            bbox: Face bounding box as [x1, y1, x2, y2]
            frame_number: Current frame number for cache management
            
        Returns:
            Face embedding vector or None
        """
        # Check cache
        bbox_hash = self._bbox_to_hash(bbox)
        if bbox_hash in self.embedding_cache:
            cached_embedding, cached_frame = self.embedding_cache[bbox_hash]
            # Use cache if within 5 frames (assuming face hasn't changed much)
            if abs(frame_number - cached_frame) < 5:
                self.cache_hit_count += 1
                return cached_embedding
        
        self.cache_miss_count += 1
        
        # Get embedding from InsightFace
        try:
            faces = self.app.get(frame)
            
            if not faces:
                return None
            
            # Find face matching the bbox
            best_face = None
            min_dist = float('inf')
            
            for face in faces:
                face_bbox = face.bbox.astype(int)
                face_center_x = (face_bbox[0] + face_bbox[2]) / 2
                face_center_y = (face_bbox[1] + face_bbox[3]) / 2
                
                bbox_center_x = (bbox[0] + bbox[2]) / 2
                bbox_center_y = (bbox[1] + bbox[3]) / 2
                
                dist = np.sqrt((face_center_x - bbox_center_x)**2 + (face_center_y - bbox_center_y)**2)
                
                if dist < min_dist:
                    min_dist = dist
                    best_face = face
            
            if best_face is None:
                return None
            
            embedding = best_face.embedding
            
            # Update cache
            if len(self.embedding_cache) >= self.cache_size:
                # Remove oldest entry
                oldest_hash = min(self.embedding_cache.items(), key=lambda x: x[1][1])[0]
                del self.embedding_cache[oldest_hash]
            
            self.embedding_cache[bbox_hash] = (embedding, frame_number)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return None
    
    def load_poi_references(self, image_paths: List[str]):
        """
        Load POI reference images.
        
        Args:
            image_paths: List of paths to POI images
        """
        self.poi_embeddings = []
        self.poi_names = []
        
        logger.info(f"Loading {len(image_paths)} POI reference images...")
        
        for img_path in image_paths:
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    logger.warning(f"Could not load: {img_path}")
                    continue
                
                faces = self.app.get(img)
                
                if not faces:
                    logger.warning(f"No face detected in: {img_path}")
                    continue
                
                # Use first face
                face = faces[0]
                embedding = face.embedding
                
                # Normalize embedding
                embedding = embedding / np.linalg.norm(embedding)
                
                self.poi_embeddings.append(embedding)
                self.poi_names.append(Path(img_path).stem)
                
                logger.info(f"✓ Loaded reference: {Path(img_path).name}")
                
            except Exception as e:
                logger.error(f"Error loading {img_path}: {e}")
        
        logger.info(f"\n✅ Loaded {len(self.poi_embeddings)}/{len(image_paths)} reference images")
        logger.info(f"   Embedding dimension: {len(self.poi_embeddings[0]) if self.poi_embeddings else 0}")
    
    def recognize_face(self, frame: np.ndarray, bbox: np.ndarray, frame_number: int = 0) -> RecognitionResult:
        """
        Recognize face with POI matching.
        
        Args:
            frame: Input frame
            bbox: Face bounding box as [x1, y1, x2, y2]
            frame_number: Current frame number
            
        Returns:
            Recognition result
        """
        # Get embedding (from cache or compute)
        embedding = self.get_embedding(frame, bbox, frame_number)
        
        if embedding is None:
            return RecognitionResult(
                is_poi=False,
                confidence=0.0,
                person_id="unknown"
            )
        
        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding)
        
        # Compare with POI references
        if not self.poi_embeddings:
            return RecognitionResult(
                is_poi=False,
                confidence=0.0,
                person_id="unknown"
            )
        
        # Compute cosine similarities
        similarities = [
            np.dot(embedding, poi_emb) 
            for poi_emb in self.poi_embeddings
        ]
        
        max_similarity = max(similarities)
        max_idx = similarities.index(max_similarity)
        
        is_poi = max_similarity >= self.recognition_threshold
        
        return RecognitionResult(
            is_poi=is_poi,
            confidence=float(max_similarity),
            person_id=self.poi_names[max_idx] if is_poi else "unknown"
        )
    
    def batch_recognize(self, frame: np.ndarray, bboxes: List[np.ndarray], frame_number: int = 0) -> List[RecognitionResult]:
        """
        Recognize multiple faces in batch (future optimization).
        
        Args:
            frame: Input frame
            bboxes: List of face bounding boxes as [x1, y1, x2, y2] arrays
            frame_number: Current frame number
            
        Returns:
            List of recognition results
        """
        # For now, process sequentially (can be optimized further)
        return [self.recognize_face(frame, bbox, frame_number) for bbox in bboxes]
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache performance statistics."""
        total = self.cache_hit_count + self.cache_miss_count
        hit_rate = self.cache_hit_count / total if total > 0 else 0
        
        return {
            'cache_size': len(self.embedding_cache),
            'hits': self.cache_hit_count,
            'misses': self.cache_miss_count,
            'hit_rate': hit_rate
        }
    
    def update_threshold(self, new_threshold: float):
        """Update recognition threshold."""
        self.recognition_threshold = new_threshold
        logger.info(f"Updated recognition threshold to {new_threshold}")
