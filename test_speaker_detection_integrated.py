"""
Integrated Test for Active Speaker Detection (Phase 4)

This test combines all three Phase 4 components:
1. LipTracker - Extract lip motion features
2. AudioFeatureExtractor - Extract audio features  
3. AudioVisualCorrelator - Correlate for speaking detection

Tests on synthetic data simulating real video with speaking/non-speaking segments.

Author: Research Team
Date: November 1, 2025
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from slceleb_modern.speaker import (
    LipTracker, AudioFeatureExtractor, AudioVisualCorrelator
)

print("="*80)
print("PHASE 4 INTEGRATED TEST: Active Speaker Detection")
print("="*80)

# Initialize all components
print("\n📦 Initializing components...")
lip_tracker = LipTracker(window_size=30, fps=30.0)
audio_extractor = AudioFeatureExtractor(sr=16000, fps=30.0, n_mfcc=13)
correlator = AudioVisualCorrelator(window_size=30, speaking_threshold=0.5)

print(f"   ✅ LipTracker ready")
print(f"   ✅ AudioFeatureExtractor ready")
print(f"   ✅ AudioVisualCorrelator ready")

# Generate synthetic video+audio (3 seconds)
print(f"\n🎬 Generating synthetic video+audio (3 seconds)...")
print(f"   Segment 1 (0.0-1.0s): SPEAKING")
print(f"   Segment 2 (1.0-2.0s): NON-SPEAKING (silent)")
print(f"   Segment 3 (2.0-3.0s): SPEAKING")

duration = 3.0
fps = 30.0
sr = 16000
total_frames = int(duration * fps)  # 90 frames
total_samples = int(duration * sr)  # 48000 samples

# Generate audio
t_audio = np.linspace(0, duration, total_samples)
audio = np.zeros(total_samples)

# Segment 1: Speaking (0-1s)
mask1 = t_audio < 1.0
base_signal1 = np.sin(2 * np.pi * 4 * t_audio[mask1])  # 4 Hz syllables
audio[mask1] = (
    0.3 * np.sin(2 * np.pi * 150 * t_audio[mask1]) +  # F0
    0.2 * np.sin(2 * np.pi * 500 * t_audio[mask1]) +  # F1
    0.15 * np.sin(2 * np.pi * 1500 * t_audio[mask1])  # F2
) * (0.5 + 0.5 * base_signal1)

# Segment 2: Silent (1-2s)
mask2 = (t_audio >= 1.0) & (t_audio < 2.0)
audio[mask2] = 0.01 * np.random.randn(np.sum(mask2))  # Just noise

# Segment 3: Speaking (2-3s)
mask3 = t_audio >= 2.0
base_signal3 = np.sin(2 * np.pi * 4 * t_audio[mask3])
audio[mask3] = (
    0.3 * np.sin(2 * np.pi * 150 * t_audio[mask3]) +
    0.2 * np.sin(2 * np.pi * 500 * t_audio[mask3]) +
    0.15 * np.sin(2 * np.pi * 1500 * t_audio[mask3])
) * (0.5 + 0.5 * base_signal3)

# Add overall noise
audio += 0.02 * np.random.randn(len(audio))

# Load audio into extractor
audio_extractor.load_audio_array(audio, sr=16000)
print(f"   ✅ Audio generated: {duration}s, {total_samples} samples")

# Process all frames
print(f"\n⚙️  Processing {total_frames} frames...")
results = []

for frame_idx in range(total_frames):
    t = frame_idx / fps
    
    # Generate synthetic landmarks (478 points) with lip motion
    landmarks = np.random.rand(478, 2) * 100
    
    # Simulate lip motion based on audio
    if t < 1.0:  # Speaking
        lip_opening = 20 + 10 * np.sin(2 * np.pi * 4 * t)
    elif t < 2.0:  # Silent
        lip_opening = 10 + 1 * np.random.randn()
    else:  # Speaking
        lip_opening = 20 + 10 * np.sin(2 * np.pi * 4 * t)
    
    # Modify lip landmarks
    lip_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
    for idx in lip_indices:
        landmarks[idx, 1] += lip_opening
    
    # Update lip tracker
    lip_tracker.update(frame_idx, landmarks)
    
    # Check if ready for correlation
    if lip_tracker.is_ready() and frame_idx >= 30:
        # Get lip features
        lip_openings = lip_tracker.get_lip_opening_sequence()
        motion_features = lip_tracker.get_motion_features()
        
        # Get audio features
        audio_seq = audio_extractor.get_amplitude_envelope_sequence(
            max(0, frame_idx - 29), frame_idx
        )
        
        # Get MFCC for recent frames
        mfcc_list = []
        for f_idx in range(max(0, frame_idx - 29), frame_idx + 1):
            feat = audio_extractor.extract_features_at_frame(f_idx)
            mfcc_list.append(feat.mfcc)
        mfcc_array = np.array(mfcc_list).T  # 13 x 30
        
        # Prepare features for correlator
        lip_features = {
            'openings': lip_openings,
            'motion_energy': motion_features.motion_energy
        }
        
        audio_features = {
            'amplitudes': audio_seq,
            'mfcc': mfcc_array,
            'mfcc_energy': np.sum(np.var(mfcc_array, axis=1))
        }
        
        # Correlate
        result = correlator.correlate(lip_features, audio_features, frame_idx, t)
        results.append(result)

print(f"   ✅ Processed {len(results)} frames with correlation")

# Analyze results by segment
print(f"\n📊 Results by Segment:")

segment1_results = [r for r in results if r.timestamp < 1.0]
segment2_results = [r for r in results if 1.0 <= r.timestamp < 2.0]
segment3_results = [r for r in results if r.timestamp >= 2.0]

def analyze_segment(segment_results, segment_name, expected_speaking):
    """Analyze detection results for a segment"""
    if len(segment_results) == 0:
        print(f"\n   {segment_name}: No results")
        return
    
    speaking_frames = sum(r.is_speaking for r in segment_results)
    total = len(segment_results)
    speaking_ratio = speaking_frames / total
    avg_score = np.mean([r.correlation_score for r in segment_results])
    avg_confidence = np.mean([r.confidence for r in segment_results])
    
    # Accuracy
    if expected_speaking:
        accuracy = speaking_ratio  # Should detect speaking
        status = "✅" if accuracy > 0.7 else "⚠️"
    else:
        accuracy = 1.0 - speaking_ratio  # Should NOT detect speaking
        status = "✅" if accuracy > 0.7 else "⚠️"
    
    print(f"\n   {segment_name} ({total} frames):")
    print(f"      Expected: {'SPEAKING' if expected_speaking else 'SILENT'}")
    print(f"      Detected speaking: {speaking_frames}/{total} ({100*speaking_ratio:.1f}%)")
    print(f"      Avg correlation: {avg_score:.3f}")
    print(f"      Avg confidence: {avg_confidence:.3f}")
    print(f"      Accuracy: {status} {100*accuracy:.1f}%")

analyze_segment(segment1_results, "Segment 1 (0.0-1.0s)", expected_speaking=True)
analyze_segment(segment2_results, "Segment 2 (1.0-2.0s)", expected_speaking=False)
analyze_segment(segment3_results, "Segment 3 (2.0-3.0s)", expected_speaking=True)

# Overall statistics
print(f"\n📈 Overall Statistics:")
speaking_frames = sum(r.is_speaking for r in results)
total = len(results)
print(f"   Total frames analyzed: {total}")
print(f"   Speaking detected: {speaking_frames} ({100*speaking_frames/total:.1f}%)")
print(f"   Expected speaking: ~66.7% (2 of 3 segments)")

avg_score_all = np.mean([r.correlation_score for r in results])
print(f"   Average correlation score: {avg_score_all:.3f}")

# Component stats
print(f"\n🔧 Component Statistics:")
print(f"   LipTracker: {lip_tracker.get_stats()}")
print(f"   AudioExtractor: {audio_extractor.get_stats()}")
print(f"   Correlator: {correlator.get_stats()}")

# Success criteria
print(f"\n✅ Test Complete!")
print(f"\nSuccess Criteria:")
seg1_good = len(segment1_results) > 0 and sum(r.is_speaking for r in segment1_results) / len(segment1_results) > 0.7
seg2_good = len(segment2_results) > 0 and sum(r.is_speaking for r in segment2_results) / len(segment2_results) < 0.3
seg3_good = len(segment3_results) > 0 and sum(r.is_speaking for r in segment3_results) / len(segment3_results) > 0.7

if seg1_good and seg2_good and seg3_good:
    print("   ✅ All segments correctly classified!")
    print("   ✅ Phase 4 core components working as expected")
    print("   ✅ Ready for real video testing")
else:
    print("   ⚠️  Some segments need tuning")
    if not seg1_good:
        print("      - Segment 1 (speaking) under-detected")
    if not seg2_good:
        print("      - Segment 2 (silent) over-detected (false positives)")
    if not seg3_good:
        print("      - Segment 3 (speaking) under-detected")
    print("   💡 Consider adjusting correlation threshold or weights")

print("\n" + "="*80)
