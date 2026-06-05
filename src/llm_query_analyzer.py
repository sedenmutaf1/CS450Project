import json
import re

# YOLO-detectable object classes (COCO dataset labels that YOLOv8 knows)
YOLO_KNOWN_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]


class LLMQueryAnalyzer:
    """
    Uses a local Ollama LLM to parse a natural language video analytics query into:
      - target_labels: list of YOLO-detectable object names to search for
      - needs_vlm:     bool, whether semantic/behavioral understanding is required
      - reasoning:     short explanation of the decision

    Falls back to keyword-based logic if Ollama is not running.
    """

    def __init__(self, model: str = "llama3.2:1b"):
        self.model = model
        self.ollama_available = False
        self._ollama = None

        try:
            import ollama
            # Ping the Ollama server with a tiny request to confirm it's running
            ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": "hi"}],
                options={"num_predict": 1}
            )
            self._ollama = ollama
            self.ollama_available = True
            print(f"[LLMQueryAnalyzer] Ollama connected. Using model: {self.model}")
        except Exception as e:
            print(f"[LLMQueryAnalyzer] Ollama not available ({e}). "
                  f"Falling back to keyword-based analysis.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, query: str) -> dict:
        """
        Main entry point. Returns:
          {
            "target_labels": ["car", "person"],
            "needs_vlm": True,
            "reasoning": "...",
            "source": "llm" | "keyword_fallback"
          }
        """
        if self.ollama_available:
            result = self._analyze_with_llm(query)
            if result is not None:
                result["source"] = "llm"
                return result
            # LLM returned unparseable output — fall through to fallback
            print("[LLMQueryAnalyzer] LLM response could not be parsed. Using keyword fallback.")

        result = self._analyze_with_keywords(query)
        result["source"] = "keyword_fallback"
        return result

    # ------------------------------------------------------------------
    # LLM path
    # ------------------------------------------------------------------

    def _analyze_with_llm(self, query: str) -> dict | None:
        known_labels_str = ", ".join(YOLO_KNOWN_LABELS[:30]) + ", ..."  # show a sample

        prompt = f"""You are a video analytics assistant. Your job is to parse a user query about video footage and return a JSON object.

User Query: "{query}"

Known YOLO-detectable labels: {known_labels_str}

Instructions:
1. Identify which physical objects from the KNOWN YOLO LABELS LIST above the user wants to detect.
   - ONLY include labels from that exact list. Do NOT invent labels.
   - If the target object (e.g. "advertisement panel", "billboard", "sign", "logo", "text") is NOT in the YOLO list, leave target_labels as an empty array [].

2. Set needs_vlm=true if ANY of the following are true:
   - The target object is NOT in the YOLO label list (e.g. advertisements, billboards, signs, logos, scenes, emotions, text)
   - The query requires understanding behavior or actions ("waiting", "fighting", "holding", "running")
   - The query requires temporal reasoning ("before", "after", "longer than", "first", "then")
   - The query requires scene context ("near the entrance", "crowded area", "night time")
   Set needs_vlm=false ONLY if the query simply asks whether a known YOLO object is present.

Respond ONLY with a valid JSON object in this exact format, no extra text:
{{
  "target_labels": [],
  "needs_vlm": true,
  "reasoning": "one sentence explanation"
}}"""

        try:
            response = self._ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0, "num_predict": 200}
            )
            raw = response["message"]["content"].strip()
            return self._parse_json_response(raw, query)
        except Exception as e:
            print(f"[LLMQueryAnalyzer] Ollama call failed: {e}")
            return None

    def _parse_json_response(self, raw: str, query: str) -> dict | None:
        """Extracts and validates the JSON block from the LLM response."""
        # Try to find a JSON block even if the model added surrounding text
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not json_match:
            return None

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            return None

        # Validate required fields
        if "target_labels" not in data or "needs_vlm" not in data:
            return None

        # Sanitize: keep only unique labels that are actually in the YOLO known list
        seen = set()
        labels = []
        for l in data["target_labels"]:
            lc = str(l).lower().strip()
            if lc and lc not in seen and lc in YOLO_KNOWN_LABELS:
                seen.add(lc)
                labels.append(lc)

        needs_vlm = bool(data.get("needs_vlm", False))

        # Sanity check: if no valid YOLO labels were found, the query target is
        # not YOLO-detectable → VLM is required regardless of what the LLM said
        if not labels:
            if not needs_vlm:
                print("[LLMQueryAnalyzer] No valid YOLO labels found — overriding needs_vlm=True.")
            needs_vlm = True

        return {
            "target_labels": labels,
            "needs_vlm": needs_vlm,
            "reasoning": str(data.get("reasoning", "")).strip()
        }

    # ------------------------------------------------------------------
    # Keyword fallback (mirrors old SemanticTrigger logic)
    # ------------------------------------------------------------------

    SEMANTIC_KEYWORDS = [
        "waiting", "staying", "sitting", "standing", "holding", "interacting",
        "talking", "moving", "playing", "reading", "eating", "drinking",
        "stealing", "fighting", "running", "jumping", "walking", "crossing",
        "loitering", "carrying", "pushing", "pulling", "climbing", "falling",
        "chasing", "following", "approaching", "leaving"
    ]

    TEMPORAL_KEYWORDS = [
        "longer than", "duration", "before", "after", "then", "sequence",
        "first", "last", "between", "while", "until", "seconds", "minutes",
        "at the same time", "simultaneously", "followed by"
    ]

    def _analyze_with_keywords(self, query: str) -> dict:
        query_lower = query.lower()

        needs_vlm = (
            any(kw in query_lower for kw in self.SEMANTIC_KEYWORDS) or
            any(kw in query_lower for kw in self.TEMPORAL_KEYWORDS)
        )

        # Infer target labels by matching known YOLO labels against query words
        found_labels = [
            label for label in YOLO_KNOWN_LABELS
            if label in query_lower
        ]

        # If no YOLO label matched the query, the subject is not YOLO-detectable → VLM required
        if not found_labels:
            needs_vlm = True
            reasoning = "Target object is not in YOLO label set; VLM required for scene understanding."
        else:
            reasoning = (
                "Keyword match detected semantic/temporal reasoning requirement."
                if needs_vlm else
                "No behavioral or temporal keywords found; YOLO-only detection sufficient."
            )

        return {
            "target_labels": found_labels,
            "needs_vlm": needs_vlm,
            "reasoning": reasoning
        }
