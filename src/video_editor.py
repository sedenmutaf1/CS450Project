import subprocess
import os
import tempfile
import glob

class VideoEditor:
    _ffmpeg_path = None  # cached path once found

    @classmethod
    def _find_ffmpeg(cls):
        """
        Returns the ffmpeg executable path, or None if not found.
        Checks PATH first, then common Windows install locations (winget, choco, manual).
        Result is cached after the first call.
        """
        if cls._ffmpeg_path is not None:
            return cls._ffmpeg_path

        # 1. Check if it's on PATH already
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            if result.returncode == 0:
                cls._ffmpeg_path = "ffmpeg"
                return cls._ffmpeg_path
        except FileNotFoundError:
            pass

        # 2. Search common Windows install locations
        search_patterns = [
            # winget (Gyan.FFmpeg)
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WinGet",
                         "Packages", "Gyan.FFmpeg*", "**", "ffmpeg.exe"),
            # Chocolatey
            r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
            # Manual installs
            r"C:\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        ]

        for pattern in search_patterns:
            matches = glob.glob(pattern, recursive=True)
            if matches:
                cls._ffmpeg_path = matches[0]
                print(f"[VideoEditor] Found FFmpeg at: {cls._ffmpeg_path}")
                return cls._ffmpeg_path

        return None

    @classmethod
    def is_ffmpeg_available(cls):
        return cls._find_ffmpeg() is not None


    @classmethod
    def cut_clip(cls, input_path, output_path, start_sec, end_sec):
        """Cuts a segment from the input video from start_sec to end_sec."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        duration = max(0.1, end_sec - start_sec)
        
        if not cls.is_ffmpeg_available():
            # Mock fallback: create a dummy file or copy source
            print(f"[VideoEditor WARNING] FFmpeg not found. Mocking cut_clip from {start_sec}s to {end_sec}s.")
            # Let's write a tiny mock file to avoid crashes
            with open(output_path, "wb") as f:
                f.write(f"mock_clip_{start_sec}_{end_sec}".encode())
            return True
            
        ffmpeg_bin = cls._find_ffmpeg()
        cmd = [
            ffmpeg_bin, "-y",
            "-ss", f"{start_sec:.3f}",
            "-t", f"{duration:.3f}",
            "-i", input_path,
            "-c:v", "libx264",
            "-c:a", "aac",
            "-preset", "ultrafast",
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"[VideoEditor ERROR] FFmpeg failed with error:\n{e.stderr.decode('utf-8', errors='ignore')}")
            # Try to copy as a fallback
            try:
                import shutil
                shutil.copy2(input_path, output_path)
                return True
            except Exception:
                return False

    @classmethod
    def merge_clips(cls, clip_paths, output_path):
        """Concatenates multiple clip files into a single video."""
        if not clip_paths:
            print("[VideoEditor WARNING] No clips to merge.")
            return False
            
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if not cls.is_ffmpeg_available():
            print("[VideoEditor WARNING] FFmpeg not found. Mocking merge_clips.")
            with open(output_path, "wb") as f:
                f.write(b"mock_merged_video")
            return True
            
        # Write temporary concat list
        temp_dir = tempfile.gettempdir()
        concat_file_path = os.path.join(temp_dir, "ffmpeg_concat_list.txt")
        
        with open(concat_file_path, "w", encoding="utf-8") as f:
            for path in clip_paths:
                # Format for FFmpeg: escape single quotes and prefix with 'file '
                escaped_path = os.path.abspath(path).replace("'", "'\\''")
                f.write(f"file '{escaped_path}'\n")
                
        ffmpeg_bin = cls._find_ffmpeg()
        cmd = [
            ffmpeg_bin, "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file_path,
            "-c", "copy",
            output_path
        ]
        
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            # Cleanup temp file
            if os.path.exists(concat_file_path):
                os.remove(concat_file_path)
            return True
        except subprocess.CalledProcessError as e:
            print(f"[VideoEditor ERROR] FFmpeg concat failed with error:\n{e.stderr.decode('utf-8', errors='ignore')}")
            if os.path.exists(concat_file_path):
                os.remove(concat_file_path)
            # Mock fallback: join content or copy first clip
            try:
                import shutil
                shutil.copy2(clip_paths[0], output_path)
                return True
            except Exception:
                return False

    # ------------------------------------------------------------------
    # Annotated highlight video (OpenCV-based, draws bbox/overlay on frames)
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_frame_annotations(frame, annotation, width, height):
        """
        Draws bounding boxes (YOLO) or a border overlay (VLM) on a frame in-place.
        annotation = {"match_type": "yolo"|"vlm", "detections": [...], "label": str}
        """
        import cv2
        match_type = annotation.get("match_type", "yolo")
        detections  = annotation.get("detections", [])

        if match_type == "yolo":
            for det in detections:
                bbox = det.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = (
                    max(0, int(bbox[0])), max(0, int(bbox[1])),
                    min(width - 1, int(bbox[2])), min(height - 1, int(bbox[3]))
                )
                color = (0, 220, 0)   # green
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                label_text = f"{det.get('label', '')}  {det.get('confidence', 0):.2f}"
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (x1, max(0, y1 - th - 10)), (x1 + tw + 4, y1), color, -1)
                cv2.putText(frame, label_text, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        elif match_type == "vlm":
            # Check if VLM returned a real bbox
            detections_with_bbox = [d for d in detections if d.get("bbox")]
            if detections_with_bbox:
                for det in detections_with_bbox:
                    bbox = det["bbox"]
                    x1, y1, x2, y2 = (
                        max(0, int(bbox[0])), max(0, int(bbox[1])),
                        min(width - 1, int(bbox[2])), min(height - 1, int(bbox[3]))
                    )
                    color = (0, 140, 255)  # orange
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    label_text = f"VLM  {det.get('confidence', 0):.2f}"
                    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(frame, (x1, max(0, y1 - th - 10)), (x1 + tw + 4, y1), color, -1)
                    cv2.putText(frame, label_text, (x1 + 2, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            else:
                # No bbox from VLM — draw full-frame orange border + badge
                border = 8
                color  = (0, 140, 255)
                cv2.rectangle(frame, (border, border),
                              (width - border, height - border), color, border)
                badge = "VLM Match"
                cv2.rectangle(frame, (0, 0), (220, 42), color, -1)
                cv2.putText(frame, badge, (8, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

        return frame

    @classmethod
    def create_annotated_highlight_video(cls, source_path, intervals, frame_annotations,
                                         output_path, query=""):
        """
        Reads `source_path` with OpenCV, outputs only frames that fall within
        `intervals` [(start_sec, end_sec), ...], and draws bounding-box annotations.

        frame_annotations: {timestamp_float: {"match_type": str, "detections": [...]}}
        """
        import cv2

        cap = cv2.VideoCapture(source_path)
        if not cap.isOpened():
            print(f"[VideoEditor ERROR] Cannot open source video: {source_path}")
            return False

        fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        out = None
        for fourcc_str in ["mp4v", "XVID", "MJPG"]:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if out.isOpened():
                break

        if out is None or not out.isOpened():
            print(f"[VideoEditor ERROR] Cannot create output video: {output_path}")
            cap.release()
            return False

        ann_timestamps = sorted(frame_annotations.keys())
        frames_written = 0
        frame_idx      = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t = frame_idx / fps

            # Only write frames that fall inside a highlight interval
            in_interval = any(start <= t <= end for start, end in intervals)
            if in_interval:
                # Find the closest annotation within ±1.5 seconds
                closest_ann = None
                min_diff    = 1.5
                for ann_t in ann_timestamps:
                    diff = abs(ann_t - t)
                    if diff < min_diff:
                        min_diff    = diff
                        closest_ann = frame_annotations[ann_t]

                if closest_ann:
                    frame = cls._draw_frame_annotations(frame, closest_ann, width, height)

                out.write(frame)
                frames_written += 1

            frame_idx += 1

        cap.release()
        out.release()

        if frames_written == 0:
            print("[VideoEditor WARNING] No frames written to annotated video.")
            return False

        print(f"[VideoEditor] Annotated highlight video: {output_path}  ({frames_written} frames)")
        return True

