"""
Simple POI Face Extraction using InsightFace

Extracts and clusters faces from video using InsightFace's integrated detection and recognition.
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import insightface
from insightface.app import FaceAnalysis
from collections import defaultdict
import json

def extract_faces_from_video(
    video_path: str,
    output_dir: str,
    max_faces_per_person: int = 30,
    similarity_threshold: float = 0.25,
    skip_frames: int = 5,
   min_face_size: int = 50
):
    """Extract and cluster faces from video."""
    
    # Initialize InsightFace
    print("Initializing InsightFace...")
    app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {total_frames} frames, {fps:.2f} FPS")
    
    # Storage for face clusters
    clusters = defaultdict(list)  # cluster_id -> list of (face_img, embedding, quality)
    next_cluster_id = 0
    
    # Process video
    frame_idx = 0
    pbar = tqdm(total=total_frames, desc="Extracting faces")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames
        if frame_idx % skip_frames != 0:
            frame_idx += 1
            pbar.update(1)
            continue
        
        # Detect faces
        faces = app.get(frame)
        
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            
            # Check face size
            if x2 - x1 < min_face_size or y2 - y1 < min_face_size:
                continue
            
            # Extract face image
            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:
                continue
            
            # Get embedding
            embedding = face.embedding
            
            # Calculate quality (simple: based on detection score and size)
            quality = face.det_score * ((x2 - x1) * (y2 - y1)) / (640 * 640)
            
            # Find matching cluster
            matched_cluster = None
            for cluster_id, cluster_faces in clusters.items():
                if len(cluster_faces) == 0:
                    continue
                # Compare with cluster average
                cluster_embeddings = np.array([f[1] for f in cluster_faces])
                avg_embedding = cluster_embeddings.mean(axis=0)
                similarity = np.dot(embedding, avg_embedding) / (
                    np.linalg.norm(embedding) * np.linalg.norm(avg_embedding)
                )
                if similarity > (1 - similarity_threshold):
                    matched_cluster = cluster_id
                    break
            
            # Add to cluster
            if matched_cluster is None:
                matched_cluster = next_cluster_id
                next_cluster_id += 1
            
            clusters[matched_cluster].append((face_img, embedding, quality))
        
        frame_idx += 1
        pbar.update(1)
    
    cap.release()
    pbar.close()
    
    # Save clustered faces
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nFound {len(clusters)} people")
    stats = {}
    
    for cluster_id, faces in clusters.items():
        if len(faces) == 0:
            continue
        
        # Sort by quality and keep top N
        faces.sort(key=lambda x: x[2], reverse=True)
        top_faces = faces[:max_faces_per_person]
        
        # Save faces
        person_dir = output_path / f"person_{cluster_id:03d}"
        person_dir.mkdir(exist_ok=True)
        
        for i, (face_img, _, quality) in enumerate(top_faces):
            face_path = person_dir / f"face_{i:03d}_q{quality:.3f}.jpg"
            cv2.imwrite(str(face_path), face_img)
        
        stats[f"person_{cluster_id:03d}"] = {
            "total_faces": len(faces),
            "saved_faces": len(top_faces),
            "avg_quality": float(np.mean([f[2] for f in faces]))
        }
        
        print(f"  Person {cluster_id}: {len(faces)} faces found, saved top {len(top_faces)}")
    
    # Save stats
    with open(output_path / "extraction_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✓ Faces saved to: {output_dir}")
    return output_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and cluster faces from video")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--output-dir", required=True, help="Output directory for faces")
    parser.add_argument("--max-faces-per-person", type=int, default=30, 
                       help="Maximum faces to save per person")
    parser.add_argument("--similarity-threshold", type=float, default=0.25,
                       help="Face similarity threshold for clustering")
    parser.add_argument("--skip-frames", type=int, default=5,
                       help="Process every Nth frame")
    parser.add_argument("--min-face-size", type=int, default=50,
                       help="Minimum face size in pixels")
    
    args = parser.parse_args()
    
    extract_faces_from_video(
        video_path=args.video,
        output_dir=args.output_dir,
        max_faces_per_person=args.max_faces_per_person,
        similarity_threshold=args.similarity_threshold,
        skip_frames=args.skip_frames,
        min_face_size=args.min_face_size
    )
