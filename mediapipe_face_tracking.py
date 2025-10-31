"""
MediaPipe Face Mesh Tracker - Modern Replacement for RetinaFace + dlib
=======================================================================

This module implements state-of-the-art face detection and landmark tracking
using Google's MediaPipe Face Mesh, which provides 478 3D facial landmarks
including detailed lip contours.

Advantages over old approach (RetinaFace + dlib 68-point):
- 7x more landmarks (478 vs 68)
- 3D coordinates for better pose handling
- Real-time performance (30+ FPS on CPU)
- Better robustness to occlusion and extreme angles
- Active maintenance and regular updates

Author: Research Team
Date: October 31, 2025
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple, Optional, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MediaPipeFaceTracker:
    """
    Modern face detection and landmark tracking using MediaPipe Face Mesh.
    
    Provides 478 3D landmarks per face, including:
    - Detailed lip contours (inner + outer lips)
    - Eye regions
    - Face oval
    - Eyebrows
    - Nose
    - Full facial mesh
    """
    
    # Landmark indices for key facial regions
    # Based on MediaPipe Face Mesh topology
    LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 
                  308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
    LIPS_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 
                  308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
    
    # Lip landmarks for sync analysis (compatible with old 20-point format)
    LIP_LANDMARKS_COMPAT = list(range(61, 81))  # 20 points for compatibility
    
    def __init__(self, 
                 static_image_mode: bool = False,
                 max_num_faces: int = 5,
                 refine_landmarks: bool = True,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize MediaPipe Face Mesh detector.
        
        Args:
            static_image_mode: If True, detection runs on every frame (slower but more accurate)
            max_num_faces: Maximum number of faces to detect
            refine_landmarks: If True, refines lip and eye landmarks for better accuracy
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for landmark tracking
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        logger.info(f"MediaPipe Face Mesh initialized: "
                   f"max_faces={max_num_faces}, "
                   f"refine={refine_landmarks}")
    
    def detect(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Detect faces and extract landmarks from an image.
        
        Args:
            image: Input image (BGR format, as used by OpenCV)
            
        Returns:
            Tuple of (bboxes, landmarks_2d, landmarks_3d):
                - bboxes: List of bounding boxes [x1, y1, x2, y2] for each face
                - landmarks_2d: List of 478 2D landmarks (x, y) for each face
                - landmarks_3d: List of 478 3D landmarks (x, y, z) for each face
        """
        # Convert BGR to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Process image
        results = self.face_mesh.process(image_rgb)
        
        bboxes = []
        landmarks_2d = []
        landmarks_3d = []
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extract landmarks
                landmarks_2d_array = np.array([
                    [lm.x * w, lm.y * h] 
                    for lm in face_landmarks.landmark
                ])
                
                landmarks_3d_array = np.array([
                    [lm.x * w, lm.y * h, lm.z * w]  # z is relative depth
                    for lm in face_landmarks.landmark
                ])
                
                # Calculate bounding box from landmarks
                x_coords = landmarks_2d_array[:, 0]
                y_coords = landmarks_2d_array[:, 1]
                x1, y1 = int(np.min(x_coords)), int(np.min(y_coords))
                x2, y2 = int(np.max(x_coords)), int(np.max(y_coords))
                
                # Add padding to bbox
                padding = int(0.1 * (x2 - x1))
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(w, x2 + padding)
                y2 = min(h, y2 + padding)
                
                bbox = np.array([x1, y1, x2, y2])
                
                bboxes.append(bbox)
                landmarks_2d.append(landmarks_2d_array)
                landmarks_3d.append(landmarks_3d_array)
        
        return bboxes, landmarks_2d, landmarks_3d
    
    def get_lip_landmarks(self, landmarks_2d: np.ndarray, 
                         include_inner: bool = True) -> Dict[str, np.ndarray]:
        """
        Extract lip landmarks from full facial landmarks.
        
        Args:
            landmarks_2d: Full 478-point landmarks (2D)
            include_inner: If True, also return inner lip contour
            
        Returns:
            Dictionary with 'outer' and optionally 'inner' lip landmarks
        """
        lip_dict = {
            'outer': landmarks_2d[self.LIPS_OUTER]
        }
        
        if include_inner:
            lip_dict['inner'] = landmarks_2d[self.LIPS_INNER]
        
        return lip_dict
    
    def get_lip_region_bbox(self, landmarks_2d: np.ndarray, 
                           expansion_factor: float = 1.5) -> Tuple[int, int, int, int]:
        """
        Get bounding box for lip region with optional expansion.
        
        Args:
            landmarks_2d: Full 478-point landmarks (2D)
            expansion_factor: Factor to expand bbox (1.0 = tight fit, 1.5 = 50% larger)
            
        Returns:
            Tuple (x1, y1, x2, y2) for lip region bounding box
        """
        # Get all lip landmarks
        lip_landmarks = np.vstack([
            landmarks_2d[self.LIPS_OUTER],
            landmarks_2d[self.LIPS_INNER]
        ])
        
        x_coords = lip_landmarks[:, 0]
        y_coords = lip_landmarks[:, 1]
        
        x1, y1 = int(np.min(x_coords)), int(np.min(y_coords))
        x2, y2 = int(np.max(x_coords)), int(np.max(y_coords))
        
        # Expand bbox
        width = x2 - x1
        height = y2 - y1
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        new_width = width * expansion_factor
        new_height = height * expansion_factor
        
        x1 = int(center_x - new_width / 2)
        y1 = int(center_y - new_height / 2)
        x2 = int(center_x + new_width / 2)
        y2 = int(center_y + new_height / 2)
        
        return x1, y1, x2, y2
    
    def calculate_lip_distance(self, landmarks_2d: np.ndarray) -> float:
        """
        Calculate vertical lip opening distance (useful for speech detection).
        
        Args:
            landmarks_2d: Full 478-point landmarks (2D)
            
        Returns:
            Normalized lip distance (0 = closed, higher = more open)
        """
        # Get key points for lip opening
        upper_lip_center = landmarks_2d[13]  # Upper lip center
        lower_lip_center = landmarks_2d[14]  # Lower lip center
        
        # Calculate vertical distance
        distance = np.linalg.norm(upper_lip_center - lower_lip_center)
        
        # Normalize by face size (use distance between eye corners)
        left_eye = landmarks_2d[33]
        right_eye = landmarks_2d[263]
        face_width = np.linalg.norm(left_eye - right_eye)
        
        normalized_distance = distance / face_width if face_width > 0 else 0
        
        return normalized_distance
    
    def visualize_landmarks(self, image: np.ndarray, 
                           landmarks_2d: List[np.ndarray],
                           draw_lips_only: bool = False) -> np.ndarray:
        """
        Draw landmarks on image for visualization.
        
        Args:
            image: Input image
            landmarks_2d: List of landmark arrays for each face
            draw_lips_only: If True, only draw lip landmarks
            
        Returns:
            Image with drawn landmarks
        """
        vis_image = image.copy()
        
        for face_landmarks in landmarks_2d:
            if draw_lips_only:
                # Draw only lip contours
                lip_outer = face_landmarks[self.LIPS_OUTER].astype(int)
                lip_inner = face_landmarks[self.LIPS_INNER].astype(int)
                
                cv2.polylines(vis_image, [lip_outer], True, (0, 255, 0), 2)
                cv2.polylines(vis_image, [lip_inner], True, (255, 0, 0), 2)
            else:
                # Draw all landmarks
                for landmark in face_landmarks:
                    x, y = int(landmark[0]), int(landmark[1])
                    cv2.circle(vis_image, (x, y), 1, (0, 255, 0), -1)
        
        return vis_image
    
    def __del__(self):
        """Cleanup MediaPipe resources."""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()


# Comparison function for benchmarking
def compare_with_old_method(image: np.ndarray) -> Dict:
    """
    Compare MediaPipe with old RetinaFace + dlib approach.
    
    Returns performance metrics for research log.
    """
    import time
    
    # MediaPipe approach
    mp_tracker = MediaPipeFaceTracker()
    
    start_time = time.time()
    bboxes_mp, landmarks_2d_mp, landmarks_3d_mp = mp_tracker.detect(image)
    mp_time = time.time() - start_time
    
    results = {
        'mediapipe': {
            'num_faces': len(bboxes_mp),
            'num_landmarks': len(landmarks_2d_mp[0]) if landmarks_2d_mp else 0,
            'processing_time': mp_time,
            'fps': 1.0 / mp_time if mp_time > 0 else 0
        }
    }
    
    # TODO: Add old method comparison when available
    # results['old_method'] = {...}
    
    return results


if __name__ == "__main__":
    """
    Test MediaPipe Face Tracker on sample image/video.
    """
    import sys
    
    # Initialize tracker
    tracker = MediaPipeFaceTracker(
        max_num_faces=5,
        min_detection_confidence=0.5
    )
    
    # Test on webcam or video file
    cap = cv2.VideoCapture(0)  # Use 0 for webcam, or path to video file
    
    print("MediaPipe Face Tracker Test")
    print("Press 'q' to quit, 's' to save screenshot")
    
    frame_count = 0
    total_time = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces and landmarks
        start_time = cv2.getTickCount()
        bboxes, landmarks_2d, landmarks_3d = tracker.detect(frame)
        end_time = cv2.getTickCount()
        
        # Calculate FPS
        processing_time = (end_time - start_time) / cv2.getTickFrequency()
        total_time += processing_time
        frame_count += 1
        fps = 1.0 / processing_time if processing_time > 0 else 0
        
        # Visualize
        vis_frame = tracker.visualize_landmarks(frame, landmarks_2d, draw_lips_only=True)
        
        # Draw bounding boxes and info
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Show lip distance
            if len(landmarks_2d) > i:
                lip_dist = tracker.calculate_lip_distance(landmarks_2d[i])
                cv2.putText(vis_frame, f"Lip: {lip_dist:.3f}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Show FPS
        cv2.putText(vis_frame, f"FPS: {fps:.1f} | Faces: {len(bboxes)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('MediaPipe Face Tracker', vis_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f'mediapipe_output_{frame_count}.jpg', vis_frame)
            print(f"Screenshot saved: mediapipe_output_{frame_count}.jpg")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Print average FPS
    if frame_count > 0:
        avg_fps = frame_count / total_time
        print(f"\nAverage FPS: {avg_fps:.2f}")
        print(f"Total frames: {frame_count}")
        print(f"Total time: {total_time:.2f}s")
