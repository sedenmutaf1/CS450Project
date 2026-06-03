import cv2
import numpy as np
import psutil
import os

class FeatureExtractor:
    @staticmethod
    def get_system_load(queue_size=0, alpha=5.0):
        """
        Calculates system load metric based on CPU/RAM usage and queue size:
        L = (CPU_% + MEM_%) / 2 + alpha * Q
        """
        try:
            cpu = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory().percent
        except Exception:
            cpu = 20.0
            mem = 30.0
            
        load = (cpu + mem) / 2.0 + alpha * queue_size
        return load

    @classmethod
    def extract_all(cls, video_path, queue_size=0):
        """
        Extracts duration, motion intensity, visual entropy, and scene changes.
        Returns a dict of features and the feature vector x.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found for feature extraction: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            # Return fallback metrics if OpenCV fails to open the video
            print(f"[FeatureExtractor WARNING] Could not open {video_path}. Using fallback values.")
            fallback = {
                "duration": 10.0,
                "motion_intensity": 1.0,
                "visual_entropy": 3.0,
                "scene_changes": 1,
                "system_load": cls.get_system_load(queue_size),
            }
            fallback["vector"] = [
                fallback["duration"],
                fallback["motion_intensity"],
                fallback["visual_entropy"],
                fallback["scene_changes"],
                fallback["system_load"]
            ]
            return fallback

        # 1. Get Duration
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps <= 0:
            fps = 25.0
        duration = total_frames / fps

        # Sample frames to make feature extraction extremely fast (e.g., max 100 frames evenly spaced)
        sample_interval = max(1, int(total_frames / 100))
        
        frames_grayscale = []
        frames_hsv = []
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_interval == 0:
                # Resize to small size for fast computation
                small_frame = cv2.resize(frame, (128, 128))
                gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
                frames_grayscale.append(gray)
                frames_hsv.append(hsv)
                
            frame_idx += 1
            
        cap.release()

        # 2. Motion Intensity
        motion_intensities = []
        for i in range(1, len(frames_grayscale)):
            # Pixel-level absolute difference
            diff = cv2.absdiff(frames_grayscale[i], frames_grayscale[i-1])
            mean_diff = np.mean(diff)
            motion_intensities.append(mean_diff)
        
        motion_intensity = float(np.mean(motion_intensities)) if motion_intensities else 0.0

        # 3. Visual Entropy (complexity of images)
        entropies = []
        for gray in frames_grayscale:
            hist, _ = np.histogram(gray.flatten(), bins=256, range=[0,256])
            # Normalize histogram to probabilities
            probs = hist / np.sum(hist)
            # Filter out zeros to avoid log(0)
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log2(probs))
            entropies.append(entropy)
            
        visual_entropy = float(np.mean(entropies)) if entropies else 0.0

        # 4. Scene Change Frequency
        scene_changes = 0
        for i in range(1, len(frames_hsv)):
            # Compute 2D Hue-Saturation histograms
            hist1 = cv2.calcHist([frames_hsv[i]], [0, 1], None, [8, 8], [0, 180, 0, 256])
            hist2 = cv2.calcHist([frames_hsv[i-1]], [0, 1], None, [8, 8], [0, 180, 0, 256])
            
            cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
            
            # Correlation method
            corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            
            # If correlation drops below threshold, we have a scene change
            if corr < 0.6:
                scene_changes += 1

        system_load = cls.get_system_load(queue_size)
        
        feature_vector = [duration, motion_intensity, visual_entropy, float(scene_changes), system_load]
        
        return {
            "duration": duration,
            "motion_intensity": motion_intensity,
            "visual_entropy": visual_entropy,
            "scene_changes": scene_changes,
            "system_load": system_load,
            "vector": feature_vector
        }
