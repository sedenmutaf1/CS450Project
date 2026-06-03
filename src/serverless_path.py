import os
import uuid
import cv2
import tempfile
from concurrent.futures import ThreadPoolExecutor
from src.cv_analyzer import CVAnalyzer
from src.vlm_analyzer import VLMAnalyzer
from src.semantic_trigger import SemanticTrigger
from src.video_editor import VideoEditor

class ServerlessPathSimulator:
    def __init__(self, storage, cv_analyzer=None, vlm_analyzer=None, semantic_trigger=None):
        self.storage = storage
        self.cv_analyzer = cv_analyzer or CVAnalyzer()
        self.vlm_analyzer = vlm_analyzer or VLMAnalyzer()
        self.semantic_trigger = semantic_trigger or SemanticTrigger()

    def run(self, video_local_path, query, target_labels):
        """
        Simulates the entire Serverless AWS Lambda / Step Functions pipeline.
        Returns final S3 key of the summary video, and processing logs.
        """
        logs = []
        folder_id = str(uuid.uuid4())[:8]
        filename = os.path.basename(video_local_path)
        
        def log(msg):
            print(f"[Serverless-Lambda] {msg}")
            logs.append(msg)

        # 1. Ingestion: Upload video to S3
        log(f"Ingesting raw video {filename} to S3 bucket...")
        video_s3_key = f"videos/{folder_id}/{filename}"
        self.storage.put_file(video_local_path, video_s3_key)
        log("Video uploaded to S3.")

        # 2. Frame Extraction Lambda (CreateFrames)
        log("Invoking CreateFrames Lambda...")
        # Download video in Lambda
        temp_dir = tempfile.gettempdir()
        lambda_vid_path = os.path.join(temp_dir, f"lambda_{filename}")
        self.storage.get_file(video_s3_key, lambda_vid_path)
        
        # Extract frames locally in Lambda
        cap = cv2.VideoCapture(lambda_vid_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample at 1 fps
        sample_rate = int(fps)
        extracted_frames_local = []
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_rate == 0:
                frame_name = f"frame_{frame_idx // sample_rate:04d}.jpg"
                frame_path = os.path.join(temp_dir, frame_name)
                cv2.imwrite(frame_path, frame)
                extracted_frames_local.append((frame_name, frame_path, float(frame_idx) / fps))
            frame_idx += 1
        cap.release()
        
        # Upload frames to S3 (S3 PUT per frame)
        frame_s3_keys = []
        for name, path, ts in extracted_frames_local:
            key = f"frames/{folder_id}/{name}"
            self.storage.put_file(path, key)
            frame_s3_keys.append((key, ts))
            os.remove(path) # Cleanup Lambda local file
            
        os.remove(lambda_vid_path) # Cleanup Lambda video
        log(f"CreateFrames Lambda finished. Extracted {len(frame_s3_keys)} frames to S3.")

        # 3. Parallel Classical CV Detection (YOLO) Lambdas
        # Let's batch frames. Batch size = 10
        batch_size = 10
        batches = [frame_s3_keys[i:i + batch_size] for i in range(0, len(frame_s3_keys), batch_size)]
        
        yolo_results = []
        log(f"Spawning {len(batches)} parallel AnalyzeFrames Lambdas for YOLO detection...")
        
        def run_yolo_lambda(batch_idx, batch_frames):
            lambda_logs = []
            # Each Lambda downloads its frames from S3
            local_paths = []
            for key, ts in batch_frames:
                local_path = os.path.join(temp_dir, f"lambda_{batch_idx}_{os.path.basename(key)}")
                self.storage.get_file(key, local_path)
                local_paths.append(local_path)
                
            # Perform detection
            detections = self.cv_analyzer.detect_objects(local_paths, target_labels)
            
            # Map back to S3 keys and clean up
            batch_results = []
            for (key, ts), local_p, det_res in zip(batch_frames, local_paths, detections):
                batch_results.append({
                    "key": key,
                    "timestamp": ts,
                    "detections": det_res["detections"]
                })
                os.remove(local_p)
                
            return batch_results

        with ThreadPoolExecutor() as executor:
            future_results = [executor.submit(run_yolo_lambda, i, b) for i, b in enumerate(batches)]
            for fut in future_results:
                yolo_results.extend(fut.result())
                
        log("All YOLO AnalyzeFrames Lambdas completed.")

        # 4. Check Semantic VLM Triggers
        # Pre-execution: check if query itself needs VLM
        needs_vlm_pre = self.semantic_trigger.should_trigger_vlm_pre_execution(query)
        
        # Confidence fallback: check YOLO results for low-confidence detections
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
                # We check all frames using VLM
                vlm_candidates = yolo_results
            else:
                log(f"VLM Analysis triggered: YOLO confidence fallback. Escalate {len(fallback_candidates)} frame(s) to VLM.")
                vlm_candidates = fallback_candidates
                
            log(f"Spawning parallel VLM (Gemini) Lambdas for {len(vlm_candidates)} frames...")
            
            # Run VLM on candidates in parallel Lambdas
            def run_vlm_lambda(candidate):
                key = candidate["key"]
                ts = candidate["timestamp"]
                # Download frame to VLM Lambda
                local_frame = os.path.join(temp_dir, f"vlm_{folder_id}_{os.path.basename(key)}")
                self.storage.get_file(key, local_frame)
                
                # Query VLM
                vlm_res = self.vlm_analyzer.analyze_frame(local_frame, query)
                os.remove(local_frame)
                
                return {
                    "key": key,
                    "timestamp": ts,
                    "match": vlm_res["match"],
                    "confidence": vlm_res["confidence"]
                }
                
            with ThreadPoolExecutor() as executor:
                vlm_results = list(executor.map(run_vlm_lambda, vlm_candidates))
                
            for v_res in vlm_results:
                if v_res["match"]:
                    matched_frames.append(v_res)
                    
            log(f"VLM Lambdas completed. Found {len(matched_frames)} match(es).")
        else:
            log("VLM not triggered. Relying on YOLO detections.")
            # Map YOLO results directly to matched frames
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
                        "key": res["key"],
                        "timestamp": res["timestamp"],
                        "confidence": max_conf
                    })
            log(f"YOLO matching finished. Found {len(matched_frames)} match(es).")

        # 5. Merge Intervals (MergeIntervalsTask)
        # Sort matched frames by timestamp
        matched_frames.sort(key=lambda x: x["timestamp"])
        log("Invoking MergeIntervals Step Functions task...")
        
        intervals = []
        if matched_frames:
            # We want to create highlights around matched frames.
            # E.g., pad each matched frame by 2 seconds before and after
            raw_intervals = []
            for mf in matched_frames:
                t = mf["timestamp"]
                raw_intervals.append((max(0.0, t - 2.0), t + 2.0))
            
            # Merge overlapping intervals
            raw_intervals.sort()
            current_start, current_end = raw_intervals[0]
            
            for start, end in raw_intervals[1:]:
                if start <= current_end:
                    current_end = max(current_end, end)
                else:
                    intervals.append((current_start, current_end))
                    current_start, current_end = start, end
            intervals.append((current_start, current_end))
            
        log(f"Merged frame detections into {len(intervals)} highlight intervals: {intervals}")

        # If no intervals found, return empty
        if not intervals:
            log("No highlight intervals found. Exiting.")
            return None, logs

        # 6. Parallel Clip Extraction (CreateSingleClip Lambdas)
        log(f"Spawning {len(intervals)} parallel CreateSingleClip Lambdas to cut highlight clips...")
        clip_s3_keys = []
        
        def run_clip_lambda(idx, interval):
            start, end = interval
            clip_name = f"clip_{idx:03d}.mp4"
            local_clip = os.path.join(temp_dir, f"lambda_{folder_id}_{clip_name}")
            
            # Each Lambda downloads the full video from S3
            local_vid = os.path.join(temp_dir, f"lambda_clip_{idx}_{filename}")
            self.storage.get_file(video_s3_key, local_vid)
            
            # Cut clip using FFmpeg
            VideoEditor.cut_clip(local_vid, local_clip, start, end)
            
            # Upload clip to S3
            clip_key = f"clips/{folder_id}/{clip_name}"
            self.storage.put_file(local_clip, clip_key)
            
            # Cleanup local temp files
            os.remove(local_clip)
            os.remove(local_vid)
            
            return clip_key

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(run_clip_lambda, idx, val) for idx, val in enumerate(intervals)]
            for fut in futures:
                clip_s3_keys.append(fut.result())
                
        log("All CreateSingleClip Lambdas completed.")

        # 7. Summary Assembly (MergeClipsTask Step Function)
        log("Invoking MergeClips Step Functions task for summary assembly...")
        # Download all clips
        local_clips = []
        for key in clip_s3_keys:
            local_c = os.path.join(temp_dir, f"lambda_merge_{os.path.basename(key)}")
            self.storage.get_file(key, local_c)
            local_clips.append(local_c)
            
        # Concatenate clips
        summary_local_path = os.path.join(temp_dir, f"summary_{folder_id}.mp4")
        VideoEditor.merge_clips(local_clips, summary_local_path)
        
        # Upload final output to S3
        final_s3_key = f"outputs/{folder_id}_summary.mp4"
        self.storage.put_file(summary_local_path, final_s3_key)
        
        # Cleanup
        for path in local_clips:
            os.remove(path)
        os.remove(summary_local_path)
        
        # Cleanup S3 intermediate keys (frames and clips buckets) to free space
        self.storage.delete_folder(f"frames/{folder_id}")
        self.storage.delete_folder(f"clips/{folder_id}")
        
        log(f"Summary Assembly completed. Output available at S3: {final_s3_key}")
        
        return final_s3_key, logs
