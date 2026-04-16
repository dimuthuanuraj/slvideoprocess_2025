"""
Simple debug - check similarity scores directly
"""
import cv2
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from slceleb_modern.recognition import ModernFaceRecognizer

print("="*80)
print("FACE RECOGNITION SIMILARITY SCORES")
print("="*80)

# Initialize recognizer
recognizer = ModernFaceRecognizer(
    model_name='buffalo_l',
    similarity_metric='cosine',
    adaptive_threshold=True
)

# Load POI references
poi_images = [
    "images/pipe_test_persons/ptp1.png",
    "images/pipe_test_persons/ptp2.png",
    "images/pipe_test_persons/ptp3.png",
    "images/pipe_test_persons/ptp4.png",
    "images/pipe_test_persons/ptp5.png"
]

print(f"\n1. Loading {len(poi_images)} POI references...")
recognizer.load_reference_images(poi_images)

# Load video frame
print(f"\n2. Loading video frame 410...")
video_path = "test_videos/sample_video.mp4"
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 410)
ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ Failed to read frame")
    sys.exit(1)

# Detect and extract faces using InsightFace
print(f"\n3. Extracting face from video frame...")
faces = recognizer.app.get(frame)

if len(faces) == 0:
    print("❌ No face found")
    sys.exit(1)

video_embedding = faces[0].embedding
print(f"✓ Extracted embedding (dim: {len(video_embedding)})")

# Compare with each POI
print(f"\n4. Comparing with POI references:")
print("="*80)
print(f"{'Reference':<25} {'Similarity':>12} {'Match?':>10}")
print("-"*80)

similarities = []
for i, ref_emb in enumerate(recognizer.reference_embeddings):
    # Cosine similarity
    similarity = np.dot(video_embedding, ref_emb.embedding) / (
        np.linalg.norm(video_embedding) * np.linalg.norm(ref_emb.embedding)
    )
    name = Path(ref_emb.image_path).name if ref_emb.image_path else f"Reference_{i}"
    similarities.append((name, similarity))
    
    threshold = recognizer.thresholds['default']
    match = "✓ YES" if similarity > threshold else "✗ No"
    print(f"{name:<25} {similarity:>12.4f} {match:>10}")

print("="*80)

# Show best match and threshold
best_name, best_sim = max(similarities, key=lambda x: x[1])
threshold = recognizer.thresholds['default']
print(f"\n📊 Results:")
print(f"   Best match: {best_name}")
print(f"   Best similarity: {best_sim:.4f}")
print(f"   Current threshold: {threshold:.4f}")
print(f"   Final result: {'POI DETECTED' if best_sim > threshold else 'UNKNOWN'}")

# Recommendation
if 0.3 < best_sim < threshold:
    print(f"\n⚠️  DIAGNOSIS:")
    print(f"   The best similarity ({best_sim:.4f}) is BELOW threshold ({threshold:.4f})")
    print(f"   This means the person IS your POI, but threshold is too strict!")
    print(f"\n💡 SOLUTION:")
    suggested_threshold = best_sim - 0.03
    print(f"   Lower the threshold to: {suggested_threshold:.4f}")
    print(f"   In integrated_pipeline.py, change:")
    print(f"   recognition_threshold={suggested_threshold:.3f}")
elif best_sim > threshold:
    print(f"\n✅ Recognition should work - similarity is above threshold!")
else:
    print(f"\n❌ Similarity is very low ({best_sim:.4f})")
    print(f"   This suggests the POI images may not match the video person")

print("\n" + "="*80)
