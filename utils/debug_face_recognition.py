"""
Debug face recognition - compare video face with POI references
to see actual similarity scores
"""
import cv2
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from slceleb_modern.recognition import ModernFaceRecognizer
from slceleb_modern.detection import MediaPipeFaceDetector

print("="*80)
print("FACE RECOGNITION DEBUG")
print("="*80)

# Initialize components
print("\n1. Initializing components...")
detector = MediaPipeFaceDetector()
recognizer = ModernFaceRecognizer(
    model_name='buffalo_l',
    similarity_metric='cosine',
    adaptive_threshold=True
)

# Load POI references
print("\n2. Loading POI reference images...")
poi_images = [
    "images/pipe_test_persons/ptp1.png",
    "images/pipe_test_persons/ptp2.png",
    "images/pipe_test_persons/ptp3.png",
    "images/pipe_test_persons/ptp4.png",
    "images/pipe_test_persons/ptp5.png"
]

recognizer.load_reference_images(poi_images)
print(f"✓ Loaded {len(poi_images)} POI references")

# Load video frame with detected face
print("\n3. Loading video frame 410 (known to have a face)...")
video_path = "test_videos/sample_video.mp4"
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 410)
ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ Failed to read frame")
    sys.exit(1)

print("✓ Frame loaded")

# Detect face
print("\n4. Detecting face in frame...")
detections = detector.detect(frame)
print(f"✓ Detected {len(detections)} face(s)")

if len(detections) == 0:
    print("❌ No face detected")
    sys.exit(1)

# Get the face
detection = detections[0]
bbox = detection.bbox
print(f"   Bounding box: {bbox}")
print(f"   Confidence: {detection.confidence:.3f}")

# Recognize face with detailed scores
print("\n5. Running face recognition...")
print("   (This will show similarity scores with each POI reference)")
print()

# Use recognizer to get embedding directly
result = recognizer.recognize_face(frame, bbox)
print(f"   Standard recognition result: {result.identity} (confidence: {result.confidence:.4f})")

# Now let's get the actual embedding and compare manually
# Extract embedding using recognizer's internal method
face_crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
faces = recognizer.app.get(frame)

if len(faces) == 0:
    print("❌ Could not extract face embedding from full frame")
    sys.exit(1)

# Find the face that matches our bbox (closest center)
bbox_center = [(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2]
best_face = None
min_dist = float('inf')

for face in faces:
    face_bbox = face.bbox.astype(int)
    face_center = [(face_bbox[0] + face_bbox[2])/2, (face_bbox[1] + face_bbox[3])/2]
    dist = np.sqrt((bbox_center[0] - face_center[0])**2 + (bbox_center[1] - face_center[1])**2)
    if dist < min_dist:
        min_dist = dist
        best_face = face

if best_face is None:
    print("❌ Could not find matching face")
    sys.exit(1)

video_embedding = best_face.embedding
print(f"✓ Extracted video face embedding (dim: {len(video_embedding)})")

# Compare with each POI reference
print("\n6. Comparing with POI references:")
print("-" * 80)

similarities = []
for i, (name, poi_embedding) in enumerate(recognizer.reference_embeddings.items()):
    # Calculate cosine similarity
    similarity = np.dot(video_embedding, poi_embedding) / (
        np.linalg.norm(video_embedding) * np.linalg.norm(poi_embedding)
    )
    similarities.append(similarity)
    
    # Determine if match
    is_match = "✓ MATCH" if similarity > recognizer.adaptive_threshold_value else "✗ No match"
    
    print(f"   {name:30s} | Similarity: {similarity:.4f} | {is_match}")

print("-" * 80)

# Show statistics
max_similarity = max(similarities)
avg_similarity = np.mean(similarities)
print(f"\n📊 Statistics:")
print(f"   Max similarity: {max_similarity:.4f}")
print(f"   Avg similarity: {avg_similarity:.4f}")
print(f"   Current threshold: {recognizer.adaptive_threshold_value:.4f}")
print(f"   Recognition result: {'POI' if max_similarity > recognizer.adaptive_threshold_value else 'Unknown'}")

# Recommendation
print(f"\n💡 Analysis:")
if max_similarity > 0.3 and max_similarity < recognizer.adaptive_threshold_value:
    print(f"   ⚠️  The best match ({max_similarity:.4f}) is below threshold ({recognizer.adaptive_threshold_value:.4f})")
    print(f"   This suggests the person IS your POI but threshold is too strict.")
    print(f"   Recommended threshold: {max_similarity - 0.05:.4f} to {max_similarity - 0.02:.4f}")
elif max_similarity > recognizer.adaptive_threshold_value:
    print(f"   ✓ Recognition should work - scores are above threshold!")
else:
    print(f"   ❌ Similarity is very low ({max_similarity:.4f})")
    print(f"   This suggests either:")
    print(f"      - Person in video is not the same as POI")
    print(f"      - POI reference images are poor quality")
    print(f"      - Lighting/angle is very different")

print("\n" + "="*80)
