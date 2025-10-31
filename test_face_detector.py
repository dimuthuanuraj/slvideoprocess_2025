"""
Quick Test - MediaPipe Face Detector
=====================================

Simple test script to verify MediaPipe integration works correctly.

Usage:
    python test_face_detector.py --image <path>
    python test_face_detector.py --webcam
    
Author: SLCeleb Research Team
Date: October 31, 2025
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from slceleb_modern.detection import MediaPipeFaceDetector


def test_on_image(image_path: str):
    """Test detector on a single image."""
    print(f"Testing on image: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Could not load image: {image_path}")
        return
    
    print(f"Image shape: {image.shape}")
    
    # Create detector
    print("Initializing detector...")
    detector = MediaPipeFaceDetector(
        max_num_faces=10,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )
    
    # Detect faces
    print("Detecting faces...")
    detections = detector.detect(image)
    
    print(f"\n✓ Detected {len(detections)} faces")
    
    # Print details for each face
    for i, det in enumerate(detections):
        print(f"\nFace {i}:")
        print(f"  Bounding box: {det.bbox}")
        print(f"  Landmarks 2D: {det.landmarks_2d.shape}")
        print(f"  Landmarks 3D: {det.landmarks_3d.shape}")
        print(f"  Confidence: {det.confidence}")
        
        # Lip distance
        lip_dist = detector.calculate_lip_distance(det)
        print(f"  Lip distance: {lip_dist:.4f}")
        
        # Get lip landmarks
        lips = detector.get_lip_landmarks(det, format='mediapipe')
        print(f"  Lip landmarks: {lips.shape}")
    
    # Visualize
    print("\nVisualizing...")
    vis_image = detector.visualize(
        image, detections,
        draw_bbox=True,
        draw_landmarks=False,
        draw_lips=True
    )
    
    # Save result
    output_path = "test_detection_result.jpg"
    cv2.imwrite(output_path, vis_image)
    print(f"✓ Saved visualization to: {output_path}")
    
    # Display if possible
    try:
        cv2.imshow('Face Detection Test', vis_image)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print("(Display not available)")


def test_on_webcam():
    """Test detector on webcam stream."""
    print("Testing on webcam...")
    
    # Create detector
    print("Initializing detector...")
    detector = MediaPipeFaceDetector(
        static_image_mode=False,
        max_num_faces=5,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        return
    
    print("✓ Webcam opened")
    print("\nControls:")
    print("  'q' - Quit")
    print("  's' - Save current frame")
    print("  'l' - Toggle landmark drawing")
    print("  'b' - Toggle bounding box")
    
    draw_landmarks = False
    draw_bbox = True
    frame_count = 0
    
    import time
    fps_start = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces
        detections = detector.detect(frame)
        
        # Visualize
        vis_frame = detector.visualize(
            frame, detections,
            draw_bbox=draw_bbox,
            draw_landmarks=draw_landmarks,
            draw_lips=True
        )
        
        # Calculate FPS
        frame_count += 1
        if frame_count % 30 == 0:
            fps = 30 / (time.time() - fps_start)
            fps_start = time.time()
        else:
            fps = 0
        
        # Display info
        info_text = [
            f'Faces: {len(detections)}',
            f'FPS: {fps:.1f}' if fps > 0 else 'FPS: calculating...',
            f'Frame: {frame_count}'
        ]
        
        y_offset = 30
        for text in info_text:
            cv2.putText(vis_frame, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
        
        # Show frame
        cv2.imshow('MediaPipe Face Detection - Webcam Test', vis_frame)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f'webcam_capture_{frame_count}.jpg'
            cv2.imwrite(filename, vis_frame)
            print(f"Saved: {filename}")
        elif key == ord('l'):
            draw_landmarks = not draw_landmarks
            print(f"Landmarks: {'ON' if draw_landmarks else 'OFF'}")
        elif key == ord('b'):
            draw_bbox = not draw_bbox
            print(f"Bounding boxes: {'ON' if draw_bbox else 'OFF'}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Test complete")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test MediaPipe Face Detector")
    parser.add_argument('--image', type=str, help="Path to test image")
    parser.add_argument('--webcam', action='store_true', help="Test on webcam")
    
    args = parser.parse_args()
    
    if args.image:
        test_on_image(args.image)
    elif args.webcam:
        test_on_webcam()
    else:
        print("Please specify --image <path> or --webcam")
        print("\nExample:")
        print("  python test_face_detector.py --webcam")
        print("  python test_face_detector.py --image sample.jpg")


if __name__ == "__main__":
    main()
