"""
Test MediaPipe Face Detector on Real Video
===========================================

Process sample video from dataset and measure performance.

Author: SLCeleb Research Team
Date: November 1, 2025
"""

import cv2
import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from slceleb_modern.detection import MediaPipeFaceDetector


def test_video_processing(video_path: str, max_frames: int = 300, save_output: bool = True):
    """
    Test face detection on video.
    
    Args:
        video_path: Path to video file
        max_frames: Maximum frames to process
        save_output: Whether to save visualization video
    """
    print("=" * 70)
    print("MediaPipe Face Detection - Video Test")
    print("=" * 70)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video: {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"\nVideo Information:")
    print(f"  File: {video_path}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total Frames: {total_frames}")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Processing: {min(max_frames, total_frames)} frames")
    
    # Initialize detector
    print(f"\nInitializing MediaPipe Face Detector...")
    detector = MediaPipeFaceDetector(
        static_image_mode=False,  # Use tracking mode for video
        max_num_faces=5,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    print("✓ Detector initialized")
    
    # Prepare output video if saving
    output_writer = None
    if save_output:
        output_path = 'test_videos/detection_result.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"✓ Output video: {output_path}")
    
    # Processing statistics
    frame_count = 0
    total_faces = 0
    total_detection_time = 0.0
    detection_times = []
    face_counts = []
    frames_with_faces = 0
    
    print(f"\nProcessing video...")
    print("-" * 70)
    
    start_time = time.time()
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces
        detect_start = time.time()
        detections = detector.detect(frame)
        detect_time = time.time() - detect_start
        
        num_faces = len(detections)
        
        # Update statistics
        detection_times.append(detect_time)
        face_counts.append(num_faces)
        total_detection_time += detect_time
        total_faces += num_faces
        if num_faces > 0:
            frames_with_faces += 1
        
        # Visualize
        vis_frame = detector.visualize(
            frame, detections,
            draw_bbox=True,
            draw_landmarks=False,
            draw_lips=True
        )
        
        # Add info overlay
        info_y = 30
        cv2.putText(vis_frame, f'Frame: {frame_count + 1}/{max_frames}', 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        info_y += 35
        cv2.putText(vis_frame, f'Faces: {num_faces}', 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        info_y += 35
        
        current_fps = 1.0 / detect_time if detect_time > 0 else 0
        cv2.putText(vis_frame, f'FPS: {current_fps:.1f}', 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add lip distance for each face
        for i, det in enumerate(detections):
            lip_dist = detector.calculate_lip_distance(det)
            x1, y1 = det.bbox[:2].astype(int)
            cv2.putText(vis_frame, f'Lip: {lip_dist:.3f}', 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Save frame
        if output_writer:
            output_writer.write(vis_frame)
        
        frame_count += 1
        
        # Progress indicator
        if frame_count % 50 == 0:
            avg_fps = frame_count / total_detection_time
            avg_faces = total_faces / frame_count
            print(f"  Frame {frame_count}/{max_frames} | "
                  f"Avg FPS: {avg_fps:.1f} | "
                  f"Avg Faces: {avg_faces:.2f} | "
                  f"Detection: {num_faces} faces")
    
    total_time = time.time() - start_time
    
    # Cleanup
    cap.release()
    if output_writer:
        output_writer.release()
    
    # Print results
    print("-" * 70)
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    avg_detection_time = np.mean(detection_times) * 1000  # ms
    std_detection_time = np.std(detection_times) * 1000   # ms
    min_detection_time = np.min(detection_times) * 1000   # ms
    max_detection_time = np.max(detection_times) * 1000   # ms
    
    avg_fps = frame_count / total_detection_time
    avg_faces = total_faces / frame_count if frame_count > 0 else 0
    success_rate = (frames_with_faces / frame_count) * 100 if frame_count > 0 else 0
    
    print(f"\nProcessing Statistics:")
    print(f"  Frames processed: {frame_count}")
    print(f"  Total time: {total_time:.2f} seconds")
    print(f"  Detection time: {total_detection_time:.2f} seconds")
    print(f"  Overhead: {(total_time - total_detection_time):.2f} seconds")
    
    print(f"\nPerformance Metrics:")
    print(f"  Average FPS: {avg_fps:.2f}")
    print(f"  Average detection time: {avg_detection_time:.2f} ms/frame")
    print(f"  Std detection time: {std_detection_time:.2f} ms")
    print(f"  Min detection time: {min_detection_time:.2f} ms")
    print(f"  Max detection time: {max_detection_time:.2f} ms")
    
    print(f"\nDetection Quality:")
    print(f"  Total faces detected: {total_faces}")
    print(f"  Average faces per frame: {avg_faces:.2f}")
    print(f"  Frames with faces: {frames_with_faces} ({success_rate:.1f}%)")
    print(f"  Frames without faces: {frame_count - frames_with_faces}")
    
    # Distribution of face counts
    unique_counts = sorted(set(face_counts))
    print(f"\nFace Count Distribution:")
    for count in unique_counts:
        freq = face_counts.count(count)
        pct = (freq / frame_count) * 100
        print(f"  {count} faces: {freq} frames ({pct:.1f}%)")
    
    print("\n" + "=" * 70)
    
    if save_output:
        print(f"\n✓ Output saved to: test_videos/detection_result.mp4")
        print(f"  You can view it with: vlc test_videos/detection_result.mp4")
    
    # Save results to file
    results_file = 'test_videos/detection_results.txt'
    with open(results_file, 'w') as f:
        f.write("MediaPipe Face Detection - Test Results\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Video: {video_path}\n")
        f.write(f"Resolution: {width}x{height}\n")
        f.write(f"Source FPS: {fps:.2f}\n")
        f.write(f"Frames processed: {frame_count}\n\n")
        f.write(f"Performance:\n")
        f.write(f"  Average FPS: {avg_fps:.2f}\n")
        f.write(f"  Average detection time: {avg_detection_time:.2f} ms\n")
        f.write(f"  Std detection time: {std_detection_time:.2f} ms\n\n")
        f.write(f"Detection Quality:\n")
        f.write(f"  Total faces: {total_faces}\n")
        f.write(f"  Average faces per frame: {avg_faces:.2f}\n")
        f.write(f"  Success rate: {success_rate:.1f}%\n\n")
        f.write(f"Face Count Distribution:\n")
        for count in unique_counts:
            freq = face_counts.count(count)
            pct = (freq / frame_count) * 100
            f.write(f"  {count} faces: {freq} frames ({pct:.1f}%)\n")
    
    print(f"✓ Results saved to: {results_file}")
    
    return {
        'avg_fps': avg_fps,
        'avg_detection_time_ms': avg_detection_time,
        'total_faces': total_faces,
        'avg_faces_per_frame': avg_faces,
        'success_rate': success_rate
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test face detection on video")
    parser.add_argument('--video', type=str, default='test_videos/sample_video.mp4',
                       help="Path to video file")
    parser.add_argument('--max-frames', type=int, default=300,
                       help="Maximum frames to process")
    parser.add_argument('--no-save', action='store_true',
                       help="Don't save output video")
    
    args = parser.parse_args()
    
    results = test_video_processing(
        args.video,
        max_frames=args.max_frames,
        save_output=not args.no_save
    )
