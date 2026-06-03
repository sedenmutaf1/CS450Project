import os
import uuid
import cv2
import tempfile
import shutil
from src.cv_analyzer import CVAnalyzer
from src.vlm_analyzer import VLMAnalyzer
from src.semantic_trigger import SemanticTrigger
from src.video_editor import VideoEditor

class ContainerPathSimulator:
    def __init__(self, storage, cv_analyzer=None, vlm_analyzer=None, semantic_trigger=None):
        self.storage = storage
        self.cv_analyzer = cv_analyzer or CVAnalyzer()
        self.vlm_analyzer = vlm_analyzer or VLMAnalyzer()
        self.semantic_trigger = semantic_trigger or SemanticTrigger()

    def run(self, video_local_path, query, target_labels):
        """
        Simulates the entire Docker / Container worker execution path.
        Downloads video once, processes entirely on local disk, uploads final summary.
        Returns final S3 key of the summary video, and processing logs.
        """
        logs = []
        folder_id = str(uuid.uuid4())[:8]
        filename = os.path.basename(video_local_path)
        
        def log(msg):
            print(f"[Container-Worker] {msg}")
            logs.append(msg)

        # 1. Ingestion: Upload video to S3 (happens before scheduler routes job)
        log("Ingesting raw video to S3 bucket...")
        video_s3_key = f"videos/{folder_id}/{filename}"
        self.storage.put_file(video_local_path, video_s3_key)

        # 2. Worker Job Dispatch & Retrieval
        log(f"Worker container dispatched. Downloading video {filename} from S3...")
        # Local container workspace setup
        container_workspace = os.path.join(tempfile.gettempdir(), f"container_{folder_id}")
        os.makedirs(container_workspace, exist_ok=True)
        
        local_vid_path = os.path.join(container_workspace, filename)
        # Download once (1 GET request)
        self.storage.get_file(video_s3_key, local_vid_path)
        log("Video downloaded to local container workspace.")

        # 3. Frame Extraction (Directly to Local Disk - No S3 Writes)
        log("Extracting frames locally to container disk...")
        cap = cv2.VideoCapture(local_vid_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        sample_rate = int(fps)
        extracted_frames = []
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_rate == 0:
                frame_name = f"frame_{frame_idx // sample_rate:04d}.jpg"
                frame_path = os.path.join(container_workspace, "frames", frame_name)
                os.makedirs(os.path.dirname(frame_path), exist_ok=True)
                cv2.imwrite(frame_path, frame)
                extracted_frames.append((frame_path, float(frame_idx) / fps))
            frame_idx += 1
        cap.release()
        log(f"Extracted {len(extracted_frames)} frames locally.")

        # 4. YOLO Object Detection (Batch Processing on local disk)
        log("Running batch YOLO detection on local frames...")
        frame_paths = [f[0] for f in extracted_frames]
        detections = self.cv_analyzer.detect_objects(frame_paths, target_labels)
        
        yolo_results = []
        for (path, ts), det_res in zip(extracted_frames, detections):
            yolo_results.append({
                "path": path,
                "timestamp": ts,
                "detections": det_res["detections"]
            })
        log("YOLO detection completed.")

        # 5. Check Semantic VLM Triggers
        needs_vlm_pre = self.semantic_trigger.should_trigger_vlm_pre_execution(query)
        needs_vlm_fallback = False
        fallback_candidates = []
        
        for res in yolo_results:
            if self.semantic_trigger.should_trigger_vlm_fallback(res["detections"], target_labels):
                needs_vlm_fallback = True
                fallback_candidates.append(res)

        vlm_triggered = needs_vlm_pre or needs_vlm_fallback
        matched_frames = []

        if vlm_triggered:
            if needs_vlm_pre:
                log("VLM Analysis triggered: Query requires semantic or temporal reasoning.")
                vlm_candidates = yolo_results
            else:
                log(f"VLM Analysis triggered: YOLO confidence fallback. Escalate {len(fallback_candidates)} frame(s) to VLM.")
                vlm_candidates = fallback_candidates
                
            log(f"Running VLM analysis on {len(vlm_candidates)} frame(s) locally...")
            for candidate in vlm_candidates:
                vlm_res = self.vlm_analyzer.analyze_frame(candidate["path"], query)
                if vlm_res["match"]:
                    matched_frames.append({
                        "path": candidate["path"],
                        "timestamp": candidate["timestamp"],
                        "confidence": vlm_res["confidence"]
                    })
            log(f"VLM analysis finished. Found {len(matched_frames)} match(es).")
        else:
            log("VLM not triggered. Relying on YOLO detections.")
            target_labels_lower = [l.lower() for l in target_labels]
            for res in yolo_results:
                has_target = False
                max_conf = 0.0
                for det in res["detections"]:
                    if not target_labels_lower or det["label"].lower() in target_labels_lower:
                        if det["confidence"] >= self.semantic_trigger.confidence_threshold:
                            has_target = True
                            max_conf = max(max_conf, det["confidence"])
                if has_target:
                    matched_frames.append({
                        "path": res["path"],
                        "timestamp": res["timestamp"],
                        "confidence": max_conf
                    })
            log(f"YOLO matching finished. Found {len(matched_frames)} match(es).")

        # 6. Merge Intervals (Done locally)
        matched_frames.sort(key=lambda x: x["timestamp"])
        log("Merging highlight intervals...")
        
        intervals = []
        if matched_frames:
            raw_intervals = []
            for mf in matched_frames:
                t = mf["timestamp"]
                raw_intervals.append((max(0.0, t - 2.0), t + 2.0))
            
            raw_intervals.sort()
            current_start, current_end = raw_intervals[0]
            
            for start, end in raw_intervals[1:]:
                if start <= current_end:
                    current_end = max(current_end, end)
                else:
                    intervals.append((current_start, current_end))
                    current_start, current_end = start, end
            intervals.append((current_start, current_end))
            
        log(f"Merged intervals: {intervals}")

        # If no intervals found, return empty
        if not intervals:
            log("No highlight intervals found. Exiting.")
            shutil.rmtree(container_workspace, ignore_errors=True)
            return None, logs

        # 7. Clip Extraction (Directly from local video file - No S3 Writes)
        log("Cutting highlight clips on local disk...")
        local_clips = []
        for idx, (start, end) in enumerate(intervals):
            clip_path = os.path.join(container_workspace, "clips", f"clip_{idx:03d}.mp4")
            os.makedirs(os.path.dirname(clip_path), exist_ok=True)
            VideoEditor.cut_clip(local_vid_path, clip_path, start, end)
            local_clips.append(clip_path)
            
        # 8. Summary Assembly (Concatenated locally)
        log("Assembling final summary video locally...")
        summary_local_path = os.path.join(container_workspace, f"summary_{folder_id}.mp4")
        VideoEditor.merge_clips(local_clips, summary_local_path)

        # 9. Upload Final Summary to S3
        log("Uploading final summary video to S3 bucket...")
        final_s3_key = f"outputs/{folder_id}_summary.mp4"
        self.storage.put_file(summary_local_path, final_s3_key)

        # 10. Workspace Cleanup
        shutil.rmtree(container_workspace, ignore_errors=True)
        log(f"Container job finished. Output available at S3: {final_s3_key}")
        
        return final_s3_key, logs
