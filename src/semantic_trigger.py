class SemanticTrigger:
    def __init__(self, confidence_threshold=0.65):
        self.confidence_threshold = confidence_threshold
        
        # Keywords suggesting semantic or behavioral analysis
        self.semantic_keywords = [
            "waiting", "staying", "sitting", "standing", "holding", "interacting",
            "talking", "moving", "playing", "reading", "eating", "drinking",
            "stealing", "fighting", "running", "jumping", "walking"
        ]
        
        # Keywords suggesting temporal relationship or duration
        self.temporal_keywords = [
            "longer than", "duration", "before", "after", "then", "sequence",
            "first", "last", "between", "while", "until", "seconds", "minutes"
        ]

    def is_semantic_query(self, query):
        """Checks if query has semantic/behavioral keywords."""
        query_lower = query.lower()
        return any(kw in query_lower for kw in self.semantic_keywords)

    def is_temporal_query(self, query):
        """Checks if query has temporal/sequential keywords."""
        query_lower = query.lower()
        return any(kw in query_lower for kw in self.temporal_keywords)

    def should_trigger_vlm_pre_execution(self, query):
        """
        Determines prior to execution if VLM is required based on query type.
        """
        return self.is_semantic_query(query) or self.is_temporal_query(query)

    def should_trigger_vlm_fallback(self, detections, target_labels):
        """
        Evaluates YOLO detection results to decide on confidence-based fallback.
        If YOLO detects a target object with confidence below the threshold,
        escalate to VLM.
        """
        target_labels = [l.lower() for l in target_labels]
        
        for det in detections:
            label = det["label"].lower()
            conf = det["confidence"]
            
            # If target object matches and is detected with low-to-moderate confidence
            if (not target_labels or label in target_labels) and (0.15 <= conf < self.confidence_threshold):
                return True
                
        return False
