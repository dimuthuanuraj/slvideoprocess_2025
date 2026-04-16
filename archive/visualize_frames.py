"""
Visualize sample frames from test video to understand content
"""
import cv2
import os

def visualize_frames(video_path, frame_indices, output_dir='results/frame_samples'):
    """Save sample frames as images"""
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video has {total_frames} frames")
    
    saved_count = 0
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            output_path = os.path.join(output_dir, f'frame_{frame_idx:06d}.jpg')
            cv2.imwrite(output_path, frame)
            print(f"Saved frame {frame_idx} to {output_path}")
            saved_count += 1
        else:
            print(f"Error: Could not read frame {frame_idx}")
    
    cap.release()
    print(f"\nSaved {saved_count}/{len(frame_indices)} frames to {output_dir}")

if __name__ == '__main__':
    video_path = '/mnt/ricproject3/node5/SLCeleb_Videoprocess/slvideoprocess_2025/test_videos/sample_video.mp4'
    
    # Sample frames: start, middle, end, and the one with face (106)
    frame_indices = [0, 10, 50, 100, 106, 150, 200, 250, 299]
    
    print("Extracting sample frames...")
    visualize_frames(video_path, frame_indices)
    print("\nYou can now check results/frame_samples/ to see what's in the video")
