"""
worker.py — Kubernetes pod entry point for the container video analytics path.

Runs inside the video-analytics-worker Docker image. Reads all configuration
from environment variables, executes the full YOLO + VLM pipeline, and writes
results to the shared output directory.

Environment variables (set by the Job spec):
  VIDEO_PATH     : absolute in-container path to the input video
  QUERY          : analytics query string
  TARGET_LABELS  : comma-separated YOLO label names (empty = all)
  JOB_ID         : unique identifier for this job
  OUTPUT_DIR     : directory to write output.json + annotated.mp4
  GEMINI_API_KEY : injected via Kubernetes Secret
"""

import os
import sys
import json
import cv2

# Ensure the src/ package is importable (PYTHONPATH=/app set in Dockerfile)
from src.cv_analyzer import CVAnalyzer
from src.vlm_analyzer import VLMAnalyzer
from src.semantic_trigger import SemanticTrigger
from src.video_editor import VideoEditor


def main():
    # ------------------------------------------------------------------ #
    # 1. Read configuration from environment
    # ------------------------------------------------------------------ #
    video_path    = os.environ.get("VIDEO_PATH", "")
    query         = os.environ.get("QUERY", "")
    target_labels = [l.strip() for l in os.environ.get("TARGET_LABELS", "").split(",") if l.strip()]
    job_id        = os.environ.get("JOB_ID", "unknown")
    output_dir    = os.environ.get("OUTPUT_DIR", f"/results/{job_id}")

    if not video_path or not os.path.exists(video_path):
        print(f"[Worker ERROR] VIDEO_PATH not found: {video_path!r}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    logs = []
    def log(msg: str):
        full = f"[Worker:{job_id}] {msg}"
        print(full, flush=True)
        logs.append(msg)

    log(f"Video: {video_path}")
    log(f"Query: {query}")
    log(f"Labels: {target_labels or '(all)'}")
    log(f"Output dir: {output_dir}")

    # ------------------------------------------------------------------ #
    # 2. Initialise analyzers
    # ------------------------------------------------------------------ #
    cv_analyzer      = CVAnalyzer()
    vlm_analyzer     = VLMAnalyzer()
    semantic_trigger = SemanticTrigger()

    # ------------------------------------------------------------------ #
    # 3. Frame extraction (1 frame per second)
    # ------------------------------------------------------------------ #
    log("Extracting frames...")
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    sample_rate = int(fps)          # sample 1 frame per second
    extracted_frames = []           # list of (path, timestamp_sec)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_rate == 0:
            name  = f"frame_{frame_idx // sample_rate:04d}.jpg"
            fpath = os.path.join(frames_dir, name)
            cv2.imwrite(fpath, frame)
            extracted_frames.append((fpath, float(frame_idx) / fps))
        frame_idx += 1
    cap.release()
    log(f"Extracted {len(extracted_frames)} frames.")

    # ------------------------------------------------------------------ #
    # 4. YOLO object detection
    # ------------------------------------------------------------------ #
    log("Running YOLO detection...")
    frame_paths  = [f[0] for f in extracted_frames]
    det_results  = cv_analyzer.detect_objects(frame_paths, target_labels)

    yolo_results = []
    for (path, ts), det in zip(extracted_frames, det_results):
        yolo_results.append({
            "path":       path,
            "timestamp":  ts,
            "detections": det["detections"],
        })
    log("YOLO complete.")

    # ------------------------------------------------------------------ #
    # 5. VLM trigger logic (mirrors container simulator)
    # ------------------------------------------------------------------ #
    needs_vlm_pre  = semantic_trigger.should_trigger_vlm_pre_execution(query)
    fallback_cands = [r for r in yolo_results
                      if semantic_trigger.should_trigger_vlm_fallback(r["detections"], target_labels)]
    vlm_triggered  = needs_vlm_pre or bool(fallback_cands)

    matched_frames = []

    if vlm_triggered:
        vlm_candidates = yolo_results if needs_vlm_pre else fallback_cands
        log(f"VLM triggered. Analyzing {len(vlm_candidates)} frame(s)...")
        for cand in vlm_candidates:
            res = vlm_analyzer.analyze_frame(cand["path"], query)
            if res["match"]:
                matched_frames.append({
                    "timestamp":  cand["timestamp"],
                    "confidence": res["confidence"],
                    "match_type": "vlm",
                    "detections": [{"label": "vlm", "confidence": res["confidence"],
                                    "bbox": res.get("bbox")}],
                })
        log(f"VLM found {len(matched_frames)} match(es).")
    else:
        log("VLM not triggered. Using YOLO results.")
        tll = [l.lower() for l in target_labels]
        for res in yolo_results:
            best_dets, max_conf = [], 0.0
            for det in res["detections"]:
                if not tll or det["label"].lower() in tll:
                    if det["confidence"] >= semantic_trigger.confidence_threshold:
                        best_dets.append(det)
                        max_conf = max(max_conf, det["confidence"])
            if best_dets:
                matched_frames.append({
                    "timestamp":  res["timestamp"],
                    "confidence": max_conf,
                    "match_type": "yolo",
                    "detections": best_dets,
                })
        log(f"YOLO matching found {len(matched_frames)} match(es).")

    # ------------------------------------------------------------------ #
    # 6. Merge matched timestamps into intervals (+/- 2 s padding)
    # ------------------------------------------------------------------ #
    matched_frames.sort(key=lambda x: x["timestamp"])
    intervals = []
    if matched_frames:
        raw = [(max(0.0, mf["timestamp"] - 2.0), mf["timestamp"] + 2.0)
               for mf in matched_frames]
        raw.sort()
        cs, ce = raw[0]
        for s, e in raw[1:]:
            if s <= ce:
                ce = max(ce, e)
            else:
                intervals.append((cs, ce))
                cs, ce = s, e
        intervals.append((cs, ce))
    log(f"Merged intervals: {intervals}")

    # ------------------------------------------------------------------ #
    # 7. Build frame annotations dict
    # ------------------------------------------------------------------ #
    frame_annotations = {
        mf["timestamp"]: {
            "match_type": mf["match_type"],
            "detections":  mf["detections"],
        }
        for mf in matched_frames
    }

    # ------------------------------------------------------------------ #
    # 8. Create annotated highlight video
    # ------------------------------------------------------------------ #
    annotated_path = None
    if intervals:
        annotated_path = os.path.join(output_dir, "annotated.mp4")
        ok = VideoEditor.create_annotated_highlight_video(
            video_path, intervals, frame_annotations, annotated_path, query
        )
        if ok:
            log(f"Annotated video: {annotated_path}")
        else:
            log("Annotated video creation failed (no frames written).")
            annotated_path = None
    else:
        log("No matches — skipping annotated video.")

    # ------------------------------------------------------------------ #
    # 9. Write output.json (read by k8s.py on the host)
    # ------------------------------------------------------------------ #
    output = {
        "job_id":            job_id,
        "matched_count":     len(matched_frames),
        "intervals":         intervals,
        "frame_annotations": {str(k): v for k, v in frame_annotations.items()},
        "annotated_video":   annotated_path,
        "logs":              logs,
    }
    out_json = os.path.join(output_dir, "output.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    log(f"Results written to {out_json}")


if __name__ == "__main__":
    main()
