import os
import re
import time
import threading


class VLMAnalyzer:
    # Rate limiter: set to 0 for paid plans, or ~12.0 for free tier (5 req/min)
    _rate_limit_lock = threading.Lock()
    _last_call_time: float = 0.0
    _min_interval: float = 0.0  # seconds between calls (0 = no throttling)

    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.gemini_available = False
        self.client = None
        self.model_name = "gemini-2.5-flash"

        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)

                # Determine the best model available
                try:
                    available = [m.name for m in self.client.models.list()]
                    for candidate in [
                        "models/gemini-2.5-flash",
                        "models/gemini-2.0-flash",
                        "models/gemini-1.5-flash",
                    ]:
                        if candidate in available:
                            self.model_name = candidate.replace("models/", "")
                            break
                except Exception:
                    pass

                self.gemini_available = True
                print(f"[VLMAnalyzer] Google Gemini API configured successfully with model: {self.model_name}")
            except Exception as e:
                print(f"[VLMAnalyzer WARNING] Failed to initialize Gemini API (using fallback simulator): {e}")
        else:
            print("[VLMAnalyzer INFO] GEMINI_API_KEY environment variable not set. Using fallback simulator.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_frame(self, frame_path, query, max_retries=4):
        """
        Sends frame and query to Gemini with automatic retry on rate-limit (429).
        Returns: {"success": bool, "match": bool, "confidence": float, "explanation": str}
        """
        if not (self.gemini_available and self.client):
            return self._fallback(frame_path, query)

        for attempt in range(max_retries):
            try:
                return self._call_gemini(frame_path, query)
            except Exception as e:
                error_str = str(e)
                if "429" in error_str:
                    wait = self._parse_retry_delay(error_str, attempt)
                    print(f"[VLMAnalyzer] Rate limited (429). Waiting {wait}s before retry "
                          f"({attempt + 1}/{max_retries})...")
                    time.sleep(wait)
                elif "503" in error_str or "UNAVAILABLE" in error_str:
                    wait = min(5 * (attempt + 1), 30)  # 5s, 10s, 15s, 20s
                    print(f"[VLMAnalyzer] Server unavailable (503). Waiting {wait}s before retry "
                          f"({attempt + 1}/{max_retries})...")
                    time.sleep(wait)
                else:
                    print(f"[VLMAnalyzer ERROR] Gemini API invocation failed: {e}. Falling back to simulation.")
                    return self._fallback(frame_path, query)

        # Exhausted retries
        print(f"[VLMAnalyzer ERROR] Gemini API failed after {max_retries} retries. Falling back to simulation.")
        return self._fallback(frame_path, query)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_gemini(self, frame_path, query):
        import PIL.Image
        from google.genai import types

        # --- Rate limiting: space out calls to stay under free-tier quota ---
        with VLMAnalyzer._rate_limit_lock:
            now = time.time()
            elapsed = now - VLMAnalyzer._last_call_time
            wait = VLMAnalyzer._min_interval - elapsed
            if wait > 0:
                print(f"[VLMAnalyzer] Rate limiter: waiting {wait:.1f}s to stay within free-tier quota...")
                time.sleep(wait)
            VLMAnalyzer._last_call_time = time.time()

        prompt = (
            f"You are a video analytics system analyzing surveillance frames.\n"
            f"User Query: '{query}'\n"
            f"Evaluate if this image shows the entity or action described in the query.\n"
            f"If YES, also provide a bounding box around the relevant object or region.\n"
            f"Format your response EXACTLY as follows (no extra text):\n"
            f"RESULT: <YES/NO>\n"
            f"CONFIDENCE: <score between 0.0 and 1.0>\n"
            f"BBOX: <[ymin, xmin, ymax, xmax] normalized to 0-1000, or NONE if not applicable>\n"
            f"EXPLANATION: <one line explaining your reasoning>\n"
        )

        with PIL.Image.open(frame_path) as img:
            frame_width, frame_height = img.size  # PIL gives (width, height)
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt, img],
            )
        text = response.text

        match_val = False
        confidence_val = 0.8
        explanation_val = ""
        bbox_val = None

        result_match = re.search(r"RESULT:\s*(YES|NO)", text, re.IGNORECASE)
        if result_match:
            match_val = result_match.group(1).upper() == "YES"

        conf_match = re.search(r"CONFIDENCE:\s*([\d.]+)", text)
        if conf_match:
            confidence_val = float(conf_match.group(1))

        # Parse BBOX: [ymin, xmin, ymax, xmax] normalized 0-1000
        bbox_match = re.search(r"BBOX:\s*\[?\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]?", text)
        if bbox_match and match_val:
            ymin = float(bbox_match.group(1))
            xmin = float(bbox_match.group(2))
            ymax = float(bbox_match.group(3))
            xmax = float(bbox_match.group(4))
            # Convert from 0-1000 normalized to pixel coordinates
            x1 = int((xmin / 1000.0) * frame_width)
            y1 = int((ymin / 1000.0) * frame_height)
            x2 = int((xmax / 1000.0) * frame_width)
            y2 = int((ymax / 1000.0) * frame_height)
            bbox_val = [x1, y1, x2, y2]

        exp_match = re.search(r"EXPLANATION:\s*(.*)", text, re.IGNORECASE)
        explanation_val = exp_match.group(1).strip() if exp_match else text.replace("\n", " ")

        return {
            "success": True,
            "match": match_val,
            "confidence": confidence_val,
            "bbox": bbox_val,
            "explanation": explanation_val,
        }


    def _parse_retry_delay(self, error_str: str, attempt: int) -> float:
        """Parses the suggested retry delay from a Gemini 429 error string."""
        # Try multiple patterns — error format varies between SDK versions
        patterns = [
            r'retry_delay\s*\{[^}]*seconds:\s*(\d+)',  # protobuf-style
            r'"retryDelay":\s*"(\d+)s"',               # JSON-style
            r'Retry in (\d+\.?\d*) second',             # plain text
            r'retry in (\d+\.?\d*)s',                   # short form
        ]
        for pat in patterns:
            m = re.search(pat, error_str, re.IGNORECASE | re.DOTALL)
            if m:
                return float(m.group(1)) + 2  # add 2s buffer
        # Exponential backoff fallback
        return min(5 * (2 ** attempt), 60)

    def _fallback(self, frame_path: str, query: str) -> dict:
        """Deterministic simulation when Gemini is unavailable."""
        fn_hash = sum(ord(c) for c in os.path.basename(frame_path))
        match_val = (fn_hash % 4 == 0)  # ~25% match rate
        confidence_val = 0.85 + 0.1 * ((fn_hash % 5) / 5.0)
        explanation_val = (
            f"(Simulated VLM decision) Detected activity matching query '{query}'."
            if match_val else
            f"(Simulated VLM decision) No activity matching query '{query}'."
        )
        return {
            "success": False,
            "match": match_val,
            "confidence": confidence_val,
            "bbox": None,
            "explanation": explanation_val,
        }

