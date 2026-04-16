"""
Test single frame processing with debug output
"""
import cv2
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from slceleb_modern.pipeline import IntegratedPipeline

# Initialize pipeline
print("Initializing pipeline...")
pipeline = IntegratedPipeline(
    detection_confidence=0.5,
    recognition_threshold=0.252,
    speaking_threshold=0.5
)

# Load POI
print("\nLoading POI references...")
poi_images = [
    "images/pipe_test_persons/ptp1.png",
    "images/pipe_test_persons/ptp2.png",
    "images/pipe_test_persons/ptp3.png",
    "images/pipe_test_persons/ptp4.png",
    "images/pipe_test_persons/ptp5.png"
]
pipeline.load_poi_references(poi_images)

# Check threshold
print(f"\nRecognizer thresholds: {pipeline.recognizer.thresholds}")
print(f"Using threshold: {pipeline.recognizer.thresholds['default']}")

# Load frame 410
print("\nLoading frame 410...")
video_path = "test_videos/sample_video.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.set(cv2.CAP_PROP_POS_FRAMES, 410)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Failed to read frame!")
    sys.exit(1)

# Process frame
print("\nProcessing frame...")
result = pipeline._process_frame(frame, 410, fps)

print(f"\n📊 Frame Processing Results:")
print(f"   Faces detected: {result.faces_detected}")
print(f"   Face identities: {result.face_identities}")
print(f"   Face confidences: {result.face_confidences}")
print(f"   POI present: {result.poi_present}")
print(f"   POI index: {result.poi_index}")

# Manually test recognition on the detected face
if result.faces_detected > 0:
    print(f"\n🔍 Manual Recognition Test:")
    bbox = result.face_bboxes[0]
    print(f"   Bbox: {bbox}")
    
    rec_result = pipeline.recognizer.recognize_face(frame, bbox)
    print(f"   Recognition result:")
    print(f"      is_match: {rec_result.is_match}")
    print(f"      confidence: {rec_result.confidence:.4f}")
    print(f"      distance: {rec_result.distance:.4f}")
    print(f"      threshold_used: {rec_result.threshold_used:.4f}")
    print(f"      all_distances: {rec_result.all_distances}")
