import os
import re
import PIL.Image

class VLMAnalyzer:
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.gemini_available = False
        self.model = None
        
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                
                # Determine the best model available on this API key
                model_name = "gemini-2.5-flash"
                try:
                    available = [m.name for m in genai.list_models()]
                    for candidate in ["models/gemini-2.5-flash", "models/gemini-3.5-flash", "models/gemini-2.0-flash", "models/gemini-1.5-flash"]:
                        if candidate in available:
                            model_name = candidate.replace("models/", "")
                            break
                except Exception:
                    pass
                
                self.model = genai.GenerativeModel(model_name)
                self.gemini_available = True
                print(f"[VLMAnalyzer] Google Gemini API configured successfully with model: {model_name}")
            except Exception as e:
                print(f"[VLMAnalyzer WARNING] Failed to initialize Gemini API (using fallback simulator): {e}")
        else:
            print("[VLMAnalyzer INFO] GEMINI_API_KEY environment variable not set. Using fallback simulator.")

    def analyze_frame(self, frame_path, query):
        """
        Sends frame and query to Gemini.
        Returns a dict: {"success": bool, "match": bool, "confidence": float, "explanation": str}
        """
        if self.gemini_available and self.model is not None:
            try:
                with PIL.Image.open(frame_path) as img:
                    prompt = (
                        f"You are a video analytics system analyzing surveillance frames.\n"
                        f"User Query: '{query}'\n"
                        f"Evaluate if this image shows the entity or action described in the query.\n"
                        f"Format your response EXACTLY as follows:\n"
                        f"RESULT: <YES/NO>\n"
                        f"CONFIDENCE: <score between 0.0 and 1.0>\n"
                        f"EXPLANATION: <one line explaining your reasoning>\n"
                    )
                    
                    response = self.model.generate_content([prompt, img])
                    text = response.text
                
                # Parse output
                match_val = False
                confidence_val = 0.8
                explanation_val = ""
                
                result_match = re.search(r"RESULT:\s*(YES|NO)", text, re.IGNORECASE)
                if result_match:
                    match_val = (result_match.group(1).upper() == "YES")
                    
                conf_match = re.search(r"CONFIDENCE:\s*([\d\.]+)", text)
                if conf_match:
                    confidence_val = float(conf_match.group(1))
                    
                exp_match = re.search(r"EXPLANATION:\s*(.*)", text, re.IGNORECASE)
                if exp_match:
                    explanation_val = exp_match.group(1).strip()
                else:
                    explanation_val = text.replace("\n", " ")
                    
                return {
                    "success": True,
                    "match": match_val,
                    "confidence": confidence_val,
                    "explanation": explanation_val
                }
            except Exception as e:
                print(f"[VLMAnalyzer ERROR] Gemini API invocation failed: {e}. Falling back to simulation.")

        # Fallback simulation logic
        # We simulate the VLM results. VLM is higher precision, so we return matches for files
        # whose hashes align. To make it interesting, we check if YOLO detected something first.
        # Here we just generate simulated responses:
        fn_hash = sum(ord(c) for c in os.path.basename(frame_path))
        match_val = (fn_hash % 4 == 0) # 25% match rate
        confidence_val = 0.85 + 0.1 * ((fn_hash % 5) / 5.0)
        explanation_val = f"(Simulated VLM decision) Detected activity matching query '{query}'." if match_val else f"(Simulated VLM decision) No activity matching query '{query}'."
        
        return {
            "success": False,
            "match": match_val,
            "confidence": confidence_val,
            "explanation": explanation_val
        }
