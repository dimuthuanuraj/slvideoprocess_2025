"""
Test audio loading with the fixed AudioFeatureExtractor
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from slceleb_modern.speaker import AudioFeatureExtractor

print("="*80)
print("TESTING AUDIO LOADING")
print("="*80)

# Initialize extractor
print("\n1. Initializing AudioFeatureExtractor...")
extractor = AudioFeatureExtractor(sr=16000, fps=50.0)

# Try to load audio from video
print("\n2. Loading audio from video...")
video_path = "test_videos/sample video2.mp4"
success = extractor.load_audio(video_path)

if success:
    print(f"\n✅ SUCCESS!")
    print(f"   Duration: {extractor.duration:.2f}s")
    print(f"   Samples: {len(extractor.audio)}")
    print(f"   Sample rate: {extractor.sr}Hz")
    
    # Test feature extraction
    print("\n3. Testing feature extraction at frame 100...")
    features = extractor.extract_features_at_frame(100)
    print(f"   MFCC shape: {features.mfcc.shape}")
    print(f"   Amplitude envelope: {features.amplitude_envelope:.4f}")
    print(f"   ZCR: {features.zero_crossing_rate:.4f}")
    print(f"   Spectral centroid: {features.spectral_centroid:.2f}Hz")
    
    print("\n✅ All audio loading tests passed!")
else:
    print("\n❌ FAILED to load audio")

print("\n" + "="*80)
