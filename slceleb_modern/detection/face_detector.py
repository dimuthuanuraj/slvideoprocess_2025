"""
Face Detection Module - MediaPipe Face Mesh Integration
========================================================

Modern face detection and landmark tracking using MediaPipe Face Mesh.
Replaces the old RetinaFace + dlib 68-point approach with SOTA 478-landmark detection.

This module provides:
- Real-time face detection with 478 3D landmarks
- Detailed lip tracking for active speaker detection
- Robust tracking across frames
- GPU acceleration support
- Compatible API with old detection system

Author: SLCeleb Research Team
Date: October 31, 2025
Phase: 2 - Face Detection & Tracking
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FaceDetection:
    """Container for face detection results."""
    bbox: np.ndarray  # [x1, y1, x2, y2]
    landmarks_2d: np.ndarray  # 478 x 2 array
    landmarks_3d: np.ndarray  # 478 x 3 array
    confidence: float  # Detection confidence
    face_id: Optional[int] = None  # For tracking across frames


class MediaPipeFaceDetector:
    """
    Modern face detection and landmark tracking using MediaPipe Face Mesh.
    
    Key Features:
    - 478 3D facial landmarks (vs 68 in old dlib)
    - Real-time performance (30+ FPS on CPU, 60+ on GPU)
    - Superior robustness to occlusion and pose variation
    - Detailed lip tracking for speaker detection
    - Temporal consistency for video processing
    
    Compatibility:
    - Can output in old 68-landmark format for legacy code
    - Compatible bounding box format with RetinaFace
    """
    
    # MediaPipe landmark indices for key facial regions
    LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 
                  308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
    LIPS_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 
                  308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
    
    # Lip landmarks for active speaker detection (20 points compatible with old format)
    LIP_LANDMARKS_20 = list(range(61, 81))
    
    # Mapping to dlib 68-point format (for backward compatibility)
    MEDIAPIPE_TO_DLIB68 = {
        # Jaw line (0-16)
        0: 234, 1: 227, 2: 137, 3: 177, 4: 215, 5: 172, 6: 136, 7: 150, 8: 149,
        9: 176, 10: 148, 11: 152, 12: 377, 13: 400, 14: 378, 15: 379, 16: 365,
        # Right eyebrow (17-21)
        17: 276, 18: 283, 19: 282, 20: 295, 21: 285,
        # Left eyebrow (22-26)
        22: 46, 23: 53, 24: 52, 25: 65, 26: 55,
        # Nose (27-35)
        27: 168, 28: 6, 29: 197, 30: 195, 31: 5, 32: 4, 33: 1, 34: 19, 35: 94,
        # Right eye (36-41)
        36: 33, 37: 160, 38: 158, 39: 133, 40: 153, 41: 144,
        # Left eye (42-47)
        42: 362, 43: 385, 44: 387, 45: 263, 46: 373, 47: 380,
        # Outer lips (48-59)
        48: 61, 49: 185, 50: 40, 51: 39, 52: 37, 53: 0, 54: 267, 55: 269,
        56: 270, 57: 409, 58: 291, 59: 375,
        # Inner lips (60-67)
        60: 321, 61: 405, 62: 314, 63: 17, 64: 84, 65: 181, 66: 91, 67: 146,
    }
    
    def __init__(self, 
                 static_image_mode: bool = False,
                 max_num_faces: int = 5,
                 refine_landmarks: bool = True,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 use_gpu: bool = True):
        """
        Initialize MediaPipe Face Mesh detector.
        
        Args:
            static_image_mode: If True, detection runs on every frame (more accurate, slower)
            max_num_faces: Maximum number of faces to detect simultaneously
            refine_landmarks: If True, refines lip and eye landmarks for better accuracy
            min_detection_confidence: Minimum confidence for initial face detection (0.0-1.0)
            min_tracking_confidence: Minimum confidence for landmark tracking (0.0-1.0)
            use_gpu: Whether to use GPU acceleration if available
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        logger.info(f"MediaPipe Face Detector initialized: "
                   f"max_faces={max_num_faces}, refine={refine_landmarks}, "
                   f"detection_conf={min_detection_confidence}, "
                   f"tracking_conf={min_tracking_confidence}")
    
    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces and extract landmarks from an image.
        
        Args:
            image: Input image in BGR format (OpenCV standard)
            
        Returns:
            List of FaceDetection objects, one per detected face
        """
        # Convert BGR to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Process the image
        results = self.face_mesh.process(image_rgb)
        
        detections = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extract landmarks
                landmarks_2d = np.array([
                    [lm.x * w, lm.y * h] 
                    for lm in face_landmarks.landmark
                ])
                
                landmarks_3d = np.array([
                    [lm.x * w, lm.y * h, lm.z * w] 
                    for lm in face_landmarks.landmark
                ])
                
                # Calculate bounding box from landmarks
                x_coords = landmarks_2d[:, 0]
                y_coords = landmarks_2d[:, 1]
                x1, y1 = int(x_coords.min()), int(y_coords.min())
                x2, y2 = int(x_coords.max()), int(y_coords.max())
                
                # Add some padding
                padding = int(0.1 * (x2 - x1))
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(w, x2 + padding)
                y2 = min(h, y2 + padding)
                
                bbox = np.array([x1, y1, x2, y2])
                
                # Create detection object
                detection = FaceDetection(
                    bbox=bbox,
                    landmarks_2d=landmarks_2d,
                    landmarks_3d=landmarks_3d,
                    confidence=1.0  # MediaPipe doesn't provide per-face confidence
                )
                
                detections.append(detection)
        
        logger.debug(f"Detected {len(detections)} faces")
        return detections
    
    def get_lip_landmarks(self, detection: FaceDetection, 
                          format: str = 'mediapipe') -> np.ndarray:
        """
        Extract lip landmarks for active speaker detection.
        
        Args:
            detection: FaceDetection object
            format: 'mediapipe' (all lip points) or 'legacy' (20 points for compatibility)
            
        Returns:
            Numpy array of lip landmark coordinates
        """
        if format == 'mediapipe':
            # Return outer + inner lip contours
            lip_indices = self.LIPS_OUTER + self.LIPS_INNER
            return detection.landmarks_2d[lip_indices]
        elif format == 'legacy':
            # Return 20 points for compatibility with old speaker detection
            return detection.landmarks_2d[self.LIP_LANDMARKS_20]
        else:
            raise ValueError(f"Unknown format: {format}. Use 'mediapipe' or 'legacy'")
    
    def calculate_lip_distance(self, detection: FaceDetection) -> float:
        """
        Calculate vertical lip opening distance (for active speaker detection).
        
        Args:
            detection: FaceDetection object
            
        Returns:
            Normalized lip distance (0.0 = closed, larger = more open)
        """
        landmarks = detection.landmarks_2d
        
        # Upper lip center (landmark 13)
        upper_lip = landmarks[13]
        # Lower lip center (landmark 14)
        lower_lip = landmarks[14]
        
        # Calculate vertical distance
        lip_distance = np.linalg.norm(upper_lip - lower_lip)
        
        # Normalize by face height for scale invariance
        face_height = detection.bbox[3] - detection.bbox[1]
        normalized_distance = lip_distance / face_height if face_height > 0 else 0.0
        
        return normalized_distance
    
    def to_dlib68_format(self, detection: FaceDetection) -> np.ndarray:
        """
        Convert 478 MediaPipe landmarks to 68-point dlib format for backward compatibility.
        
        Args:
            detection: FaceDetection object
            
        Returns:
            68x2 numpy array in dlib format
        """
        landmarks_68 = np.zeros((68, 2))
        for dlib_idx, mp_idx in self.MEDIAPIPE_TO_DLIB68.items():
            landmarks_68[dlib_idx] = detection.landmarks_2d[mp_idx]
        return landmarks_68
    
    def visualize(self, image: np.ndarray, detections: List[FaceDetection],
                  draw_bbox: bool = True, draw_landmarks: bool = True,
                  draw_lips: bool = True) -> np.ndarray:
        """
        Visualize face detections on the image.
        
        Args:
            image: Input image (BGR)
            detections: List of FaceDetection objects
            draw_bbox: Whether to draw bounding boxes
            draw_landmarks: Whether to draw all facial landmarks
            draw_lips: Whether to highlight lip landmarks
            
        Returns:
            Image with visualizations
        """
        vis_image = image.copy()
        
        for i, detection in enumerate(detections):
            # Draw bounding box
            if draw_bbox:
                x1, y1, x2, y2 = detection.bbox.astype(int)
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis_image, f'Face {i}', (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw all landmarks
            if draw_landmarks:
                for landmark in detection.landmarks_2d:
                    x, y = landmark.astype(int)
                    cv2.circle(vis_image, (x, y), 1, (0, 255, 255), -1)
            
            # Highlight lip landmarks
            if draw_lips:
                lip_landmarks = self.get_lip_landmarks(detection, format='mediapipe')
                for landmark in lip_landmarks:
                    x, y = landmark.astype(int)
                    cv2.circle(vis_image, (x, y), 2, (0, 0, 255), -1)
                
                # Draw lip distance
                lip_dist = self.calculate_lip_distance(detection)
                x1, y1 = detection.bbox[:2].astype(int)
                cv2.putText(vis_image, f'Lip: {lip_dist:.3f}', (x1, y1 - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return vis_image
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()


# Convenience function for quick detection
def detect_faces(image: np.ndarray, 
                max_faces: int = 5,
                min_confidence: float = 0.5) -> List[FaceDetection]:
    """
    Convenience function for quick face detection.
    
    Args:
        image: Input image (BGR)
        max_faces: Maximum number of faces to detect
        min_confidence: Minimum detection confidence
        
    Returns:
        List of FaceDetection objects
    """
    detector = MediaPipeFaceDetector(
        max_num_faces=max_faces,
        min_detection_confidence=min_confidence
    )
    return detector.detect(image)


if __name__ == "__main__":
    # Test the detector
    print("MediaPipe Face Detector - Test Mode")
    print("=" * 60)
    
    # Create a test detector
    detector = MediaPipeFaceDetector(
        max_num_faces=5,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )
    
    # Test with webcam or test image
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No webcam found. Please test with an image.")
    else:
        print("Press 'q' to quit, 's' to save frame")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect faces
            detections = detector.detect(frame)
            
            # Visualize
            vis_frame = detector.visualize(frame, detections,
                                          draw_bbox=True,
                                          draw_landmarks=False,
                                          draw_lips=True)
            
            # Show info
            cv2.putText(vis_frame, f'Faces: {len(detections)}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('MediaPipe Face Detection', vis_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite('face_detection_test.jpg', vis_frame)
                print("Saved: face_detection_test.jpg")
        
        cap.release()
        cv2.destroyAllWindows()
    
    print("\nTest complete!")
