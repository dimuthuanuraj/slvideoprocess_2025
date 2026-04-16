"""
Visualize a detected face from the video to see who's in it
"""
import cv2
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from slceleb_modern.detection import MediaPipeFaceDetector

# Load video
video_path = "test_videos/sample_video.mp4"
cap = cv2.VideoCapture(video_path)

# Initialize detector
detector = MediaPipeFaceDetector()

# Go to frame 410 (we know it has a face)
frame_idx = 410
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
ret, frame = cap.read()

if ret:
    # Detect faces
    detections = detector.detect(frame)
    
    print(f"\n📹 Frame {frame_idx}:")
    print(f"   Faces detected: {len(detections)}")
    
    # Draw bounding boxes
    for i, det in enumerate(detections):
        bbox = det.bbox
        x1, y1, x2, y2 = bbox
        
        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Face {i+1}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        print(f"   Face {i+1}: bbox={bbox}, confidence={det.confidence:.3f}")
    
    # Save image
    output_path = "results/sample_face_frame410.jpg"
    cv2.imwrite(output_path, frame)
    print(f"\n✅ Saved visualization to: {output_path}")
    print(f"   You can view this image to see who's in the video")
    
    # Also save the cropped face
    if len(detections) > 0:
        det = detections[0]
        x1, y1, x2, y2 = det.bbox
        face_crop = frame[y1:y2, x1:x2]
        face_path = "results/sample_face_crop_frame410.jpg"
        cv2.imwrite(face_path, face_crop)
        print(f"   Cropped face saved to: {face_path}")
else:
    print("Failed to read frame")

cap.release()
