#!/usr/bin/env python3
"""
Pipeline Visualization Tool
============================
Renders a debug visualization video showing how the pipeline detects,
tracks, and verifies the speaker (POI) with mouth-movement overlays.

Visualization layout:
  ┌──────────────────────────────────────┐
  │  Main video frame with overlays:     │
  │  - Face bboxes (green=POI, gray=unk) │
  │  - Lip landmarks drawn on face       │
  │  - Identity label + confidence       │
  │  - SPEAKING / SILENT badge           │
  ├──────────────────────────┬───────────┤
  │  Lip opening graph       │  Status   │
  │  (rolling 3-second       │  panel    │
  │   time series)           │           │
  └──────────────────────────┴───────────┘

Usage:
    python visualize_pipeline.py \
        --video "input video/project_V3_test1.mp4" \
        --poi-dir output/poi_reference_v3/ \
        --output output/visualization.mp4 \
        --max-frames 5000
"""

import argparse
import cv2
import numpy as np
import sys
import time
import logging
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).parent))

from slceleb_modern.pipeline.integrated_pipeline import IntegratedPipeline

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ── Colour palette ──────────────────────────────────────────────────────────
COL_POI   = (0, 255, 0)     # green
COL_UNK   = (160, 160, 160) # gray
COL_SPEAK = (0, 255, 100)   # bright green
COL_SILENT = (0, 0, 200)    # red
COL_LIP   = (255, 180, 0)   # cyan-ish (BGR)
COL_TEXT  = (255, 255, 255)  # white
COL_BG    = (30, 30, 30)    # dark bg
COL_GRAPH_LINE = (0, 200, 255)  # yellow-orange
COL_GRAPH_FILL = (0, 80, 120)
COL_AUDIO_LINE = (200, 100, 255)
COL_SEGMENT = (0, 180, 0)


# MediaPipe lip landmark indices (same as in lip_tracker.py)
OUTER_LIP_IDX = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0,
    267, 269, 270, 409, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78
]
INNER_LIP_IDX = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 191, 80, 81, 82]


def draw_lip_landmarks(frame, landmarks, color=COL_LIP, thickness=1):
    """Draw lip contour on the frame."""
    if landmarks is None or landmarks.shape[0] < max(OUTER_LIP_IDX):
        return
    pts_outer = landmarks[OUTER_LIP_IDX, :2].astype(np.int32)
    pts_inner = landmarks[INNER_LIP_IDX, :2].astype(np.int32)
    cv2.polylines(frame, [pts_outer], isClosed=True, color=color, thickness=thickness)
    cv2.polylines(frame, [pts_inner], isClosed=True, color=(0, 200, 255), thickness=thickness)
    for pt in pts_outer:
        cv2.circle(frame, tuple(pt), 1, color, -1)


def draw_bbox(frame, bbox, label, color, conf=None, speaking=None):
    """Draw bounding box with label and optional speaking badge."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Label background
    text = label
    if conf is not None:
        text += f" {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, text, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    # Speaking badge
    if speaking is not None:
        badge_text = "SPEAKING" if speaking else "SILENT"
        badge_col = COL_SPEAK if speaking else COL_SILENT
        bx = x1
        by = y2 + 4
        (btw, bth), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (bx, by), (bx + btw + 10, by + bth + 10), badge_col, -1)
        cv2.putText(frame, badge_text, (bx + 5, by + bth + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)


def draw_graph(canvas, data, y_min, y_max, color_line, color_fill=None, label=None):
    """Draw a time-series graph on a canvas (h x w x 3)."""
    h, w = canvas.shape[:2]
    n = len(data)
    if n < 2:
        return
    margin_top, margin_bot = 20, 10
    plot_h = h - margin_top - margin_bot

    # Normalise
    rng = max(y_max - y_min, 1e-6)
    pts = []
    for i, v in enumerate(data):
        x = int(i * (w - 1) / (n - 1))
        y = margin_top + int((1.0 - (v - y_min) / rng) * plot_h)
        y = max(margin_top, min(margin_top + plot_h, y))
        pts.append((x, y))

    # Fill area
    if color_fill is not None:
        fill_pts = pts + [(pts[-1][0], margin_top + plot_h), (pts[0][0], margin_top + plot_h)]
        cv2.fillPoly(canvas, [np.array(fill_pts, dtype=np.int32)], color_fill)

    # Line
    for i in range(len(pts) - 1):
        cv2.line(canvas, pts[i], pts[i + 1], color_line, 1, cv2.LINE_AA)

    # Label
    if label:
        cv2.putText(canvas, label, (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_TEXT, 1, cv2.LINE_AA)


def draw_status_panel(panel, frame_idx, timestamp, fps, poi_present,
                      poi_speaking, speak_conf, faces_detected,
                      poi_frames_total, speak_frames_total,
                      segment_count, total_frames):
    """Draw the info/status panel on the right side."""
    panel[:] = COL_BG
    y = 20
    lines = [
        f"Frame: {frame_idx}/{total_frames}",
        f"Time:  {timestamp:.2f}s",
        f"FPS:   {fps:.1f}",
        "",
        f"Faces: {faces_detected}",
        f"POI:   {'YES' if poi_present else 'no'}",
        f"Speak: {'YES' if poi_speaking else 'no'}",
        f"Conf:  {speak_conf:.3f}" if speak_conf > 0 else "Conf:  ---",
        "",
        f"POI frames:  {poi_frames_total}",
        f"Speak frames:{speak_frames_total}",
        f"Segments:    {segment_count}",
    ]
    for line in lines:
        if line == "":
            y += 10
            continue
        col = COL_TEXT
        if "POI:   YES" in line:
            col = COL_POI
        if "Speak: YES" in line:
            col = COL_SPEAK
        elif "Speak: no" in line:
            col = COL_SILENT
        cv2.putText(panel, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
        y += 18


def main():
    parser = argparse.ArgumentParser(description="Pipeline Visualization")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--poi-dir", required=True, help="POI reference images directory")
    parser.add_argument("--output", default="output/visualization.mp4", help="Output video path")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames (for testing)")
    parser.add_argument("--model", default="buffalo_s", help="InsightFace model")
    parser.add_argument("--recognition-threshold", type=float, default=0.25)
    parser.add_argument("--speaking-threshold", type=float, default=0.35)
    parser.add_argument("--detection-confidence", type=float, default=0.3)
    args = parser.parse_args()

    # ── Build pipeline ──────────────────────────────────────────────────────
    logger.info("Initialising pipeline…")
    pipeline = IntegratedPipeline(
        detection_confidence=args.detection_confidence,
        recognition_threshold=args.recognition_threshold,
        speaking_threshold=args.speaking_threshold,
    )
    # Replace recognizer with optimized one that matches the model
    from slceleb_modern.recognition.face_recognizer_optimized import OptimizedFaceRecognizer
    pipeline.recognizer = OptimizedFaceRecognizer(
        model_name=args.model, use_gpu=True, cache_size=200
    )

    # Load POI references
    poi_dir = Path(args.poi_dir)
    poi_images = sorted([str(p) for p in poi_dir.glob("*") if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])
    if not poi_images:
        logger.error(f"No reference images found in {poi_dir}")
        return
    pipeline.load_poi_references(poi_images)
    logger.info(f"Loaded {len(poi_images)} POI reference images")

    # ── Open video ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {args.video}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    max_frames = args.max_frames or total_frames
    logger.info(f"Video: {vid_w}x{vid_h} @ {vid_fps:.1f} FPS, {total_frames} frames")

    # Load audio into pipeline
    pipeline.audio_extractor.load_audio(args.video)
    pipeline.audio_loaded = True

    # Update speaker components to actual video FPS
    from slceleb_modern.speaker import LipTracker, AudioVisualCorrelator
    pipeline.lip_tracker = LipTracker(window_size=int(vid_fps), fps=vid_fps)
    pipeline.audio_extractor.fps = vid_fps
    pipeline.correlator = AudioVisualCorrelator(
        window_size=int(vid_fps),
        speaking_threshold=args.speaking_threshold,
    )
    logger.info(f"Speaker detection FPS set to {vid_fps:.1f}")

    # ── Layout sizes ────────────────────────────────────────────────────────
    # Scale main frame to fit a 1280-wide canvas
    CANVAS_W = 1280
    scale = CANVAS_W / vid_w
    main_w = CANVAS_W
    main_h = int(vid_h * scale)

    GRAPH_H = 160          # height of graph strip
    STATUS_W = 200         # width of status panel
    GRAPH_W = main_w - STATUS_W

    out_w = main_w
    out_h = main_h + GRAPH_H

    # ── Output writer ───────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, vid_fps, (out_w, out_h))
    if not writer.isOpened():
        logger.error(f"Cannot create output video: {out_path}")
        return

    # ── Rolling buffers for graphs ──────────────────────────────────────────
    graph_window = int(vid_fps * 3)  # 3 seconds of history
    lip_history = deque(maxlen=graph_window)
    audio_history = deque(maxlen=graph_window)
    speak_history = deque(maxlen=graph_window)  # 1.0 / 0.0

    # Counters
    poi_frames_total = 0
    speak_frames_total = 0
    segment_count = 0
    was_speaking = False

    t0 = time.time()
    frame_idx = 0
    logger.info(f"Rendering visualization → {out_path}")

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Run pipeline on this frame
        result = pipeline._process_frame(frame, frame_idx, vid_fps)

        # ── Gather info ──────────────────────────────────────────────────────
        poi_present = result.poi_present
        poi_speaking = result.poi_speaking
        speak_conf = 0.0
        if result.poi_index is not None and result.poi_index < len(result.speaking_confidences):
            speak_conf = result.speaking_confidences[result.poi_index]

        if poi_present:
            poi_frames_total += 1
        if poi_speaking:
            speak_frames_total += 1
            if not was_speaking:
                segment_count += 1
        was_speaking = poi_speaking

        timestamp = frame_idx / vid_fps

        # Lip opening for graph
        lip_val = pipeline.lip_tracker.get_current_lip_opening()
        lip_history.append(lip_val)

        # Audio amplitude for graph
        try:
            audio_feat = pipeline.audio_extractor.extract_features_at_frame(frame_idx)
            audio_history.append(audio_feat.amplitude_envelope)
        except Exception:
            audio_history.append(0.0)

        speak_history.append(1.0 if poi_speaking else 0.0)

        # ── Draw overlays on the frame ───────────────────────────────────────
        vis = frame.copy()

        for i in range(result.faces_detected):
            # Determine if this face is POI
            is_poi = (i < len(result.face_identities) and result.face_identities[i] == "POI")
            color = COL_POI if is_poi else COL_UNK
            conf = result.face_confidences[i] if i < len(result.face_confidences) else 0.0
            label = "POI" if is_poi else "Unknown"

            speaking = None
            if is_poi:
                speaking = poi_speaking

            bbox = result.face_bboxes[i] if i < len(result.face_bboxes) else None
            if bbox is not None:
                draw_bbox(vis, bbox, label, color, conf=conf, speaking=speaking)

            # Draw lip landmarks
            if i < len(result.face_landmarks):
                lm = result.face_landmarks[i]
                lip_col = COL_SPEAK if (is_poi and poi_speaking) else COL_LIP
                draw_lip_landmarks(vis, lm, color=lip_col, thickness=2 if is_poi else 1)

        # Timestamp overlay
        cv2.putText(vis, f"{timestamp:.2f}s  F{frame_idx}", (10, vid_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL_TEXT, 1, cv2.LINE_AA)

        # ── Resize main frame ────────────────────────────────────────────────
        main_frame = cv2.resize(vis, (main_w, main_h), interpolation=cv2.INTER_AREA)

        # ── Build graph strip ────────────────────────────────────────────────
        graph_canvas = np.full((GRAPH_H, GRAPH_W, 3), COL_BG, dtype=np.uint8)

        # Split into top half (lip) and bottom half (audio)
        half_h = GRAPH_H // 2
        lip_area = graph_canvas[0:half_h, :, :]
        audio_area = graph_canvas[half_h:GRAPH_H, :, :]

        # Draw lip opening graph
        lip_data = list(lip_history)
        draw_graph(lip_area, lip_data, 0, 40, COL_GRAPH_LINE, COL_GRAPH_FILL, "Lip Opening (px)")

        # Draw audio amplitude graph
        audio_data = list(audio_history)
        draw_graph(audio_area, audio_data, 0, 0.15, COL_AUDIO_LINE, (60, 30, 80), "Audio Amplitude")

        # Shade speaking regions on both graphs
        speak_data = list(speak_history)
        if len(speak_data) > 1:
            for g_idx in range(len(speak_data) - 1):
                if speak_data[g_idx] > 0.5:
                    x = int(g_idx * (GRAPH_W - 1) / (len(speak_data) - 1))
                    x2 = int((g_idx + 1) * (GRAPH_W - 1) / (len(speak_data) - 1))
                    overlay = graph_canvas[:, x:x2, :].copy()
                    green_tint = np.full_like(overlay, (0, 60, 0), dtype=np.uint8)
                    cv2.add(overlay, green_tint, overlay)
                    graph_canvas[:, x:x2, :] = overlay

        # Divider line
        cv2.line(graph_canvas, (0, half_h), (GRAPH_W, half_h), (80, 80, 80), 1)

        # ── Status panel ─────────────────────────────────────────────────────
        status_panel = np.full((GRAPH_H, STATUS_W, 3), COL_BG, dtype=np.uint8)
        proc_fps = (frame_idx + 1) / max(time.time() - t0, 0.01)
        draw_status_panel(
            status_panel, frame_idx, timestamp, proc_fps,
            poi_present, poi_speaking, speak_conf,
            result.faces_detected, poi_frames_total, speak_frames_total,
            segment_count, total_frames
        )

        # ── Compose final frame ──────────────────────────────────────────────
        bottom_strip = np.hstack([graph_canvas, status_panel])
        output_frame = np.vstack([main_frame, bottom_strip])

        writer.write(output_frame)
        frame_idx += 1

        if frame_idx % 100 == 0:
            elapsed = time.time() - t0
            logger.info(f"  {frame_idx}/{max_frames} frames "
                        f"({100 * frame_idx / max_frames:.1f}%) – "
                        f"{frame_idx / elapsed:.1f} FPS")

    cap.release()
    writer.release()
    elapsed = time.time() - t0
    logger.info(f"\nDone! {frame_idx} frames in {elapsed:.1f}s ({frame_idx / elapsed:.1f} FPS)")
    logger.info(f"POI frames: {poi_frames_total}, Speaking frames: {speak_frames_total}, Segments: {segment_count}")
    logger.info(f"Visualization saved to: {out_path}")


if __name__ == "__main__":
    main()
