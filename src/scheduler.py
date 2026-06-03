import os

class PredictiveScheduler:
    def __init__(self, lambda_weight=1.0):
        """
        lambda_weight (float): Balances cost and latency. 
                               Higher lambda prioritizes performance (low latency).
                               Lower lambda prioritizes monetary savings (low cost).
        """
        self.lambda_weight = lambda_weight
        
        # S3 Cost Parameters (in USD)
        self.S3_PUT_COST = 0.000005  # per PUT
        self.S3_GET_COST = 0.0000004 # per GET
        self.S3_DATA_TRANSFER_OUT_COST = 0.09 / (1024 * 1024 * 1024) # per byte ($0.09 / GB)
        
        # Serverless Cost Parameters
        self.LAMBDA_INVOKE_COST = 0.0000002 # flat fee per execution
        # 2GB RAM Lambda execution cost per millisecond: ~$0.0000000333
        self.LAMBDA_COMPUTE_MS_COST = 0.0000000333
        
        # VLM API Cost Parameter (Gemini VLM call cost, in USD)
        self.VLM_CALL_COST = 0.0025 # Estimate of token cost per image query
        
        # Container Cost Parameters
        # Amortized container cluster cost: let's say $0.18 / hour ($0.00005 / second)
        self.CONTAINER_SEC_COST = 0.00005 

    def estimate_serverless(self, features, will_trigger_vlm=False):
        """
        Estimates cost (USD) and latency (sec) for Serverless execution.
        """
        duration = features["duration"]
        motion = features["motion_intensity"]
        entropy = features["visual_entropy"]
        
        # --- Latency Estimation ---
        # Frame extraction: downloads video, runs FFmpeg, uploads frames to S3
        extract_latency = 2.0 + (0.05 * duration) # 2s base cold start/setup + 0.05s per video sec
        
        # YOLO batch detection: processed in parallel batches of 10 frames
        # S3 write/read overhead + batch execution overhead
        num_frames = max(1, int(duration)) # sampling at 1fps
        num_batches = max(1, (num_frames + 9) // 10)
        yolo_latency = 1.5 + (0.02 * num_frames) # parallelized, but scale-up overhead
        
        # VLM evaluation (multimodal model is slower)
        vlm_latency = 0.0
        if will_trigger_vlm:
            # Assume 1 VLM call per 5 frames (on selected candidates)
            num_vlm_calls = max(1, num_frames // 5)
            # Parallelized, but API response overhead
            vlm_latency = 2.5 + (0.1 * num_vlm_calls)
            
        # Assembly (Step functions clip assembly)
        assembly_latency = 3.0 + (0.1 * num_frames)
        
        total_latency = extract_latency + yolo_latency + vlm_latency + assembly_latency
        
        # --- Cost Estimation ---
        # Lambda invokations: 1 (extract) + num_batches (yolo) + 1 (merge) + num_clips (assemble)
        invocations = 1 + num_batches + 1 + 5
        invoke_cost = invocations * self.LAMBDA_INVOKE_COST
        
        # Compute cost
        # Extract Lambda: runs for (0.05 * duration) seconds
        extract_compute = (0.05 * duration * 1000) * self.LAMBDA_COMPUTE_MS_COST
        # YOLO Lambdas: each batch takes ~0.5s
        yolo_compute = (num_batches * 500) * self.LAMBDA_COMPUTE_MS_COST
        
        # VLM compute (if run as Lambda, but let's count VLM API cost separately)
        vlm_compute = 0.0
        vlm_api_cost = 0.0
        if will_trigger_vlm:
            num_vlm_calls = max(1, num_frames // 5)
            vlm_api_cost = num_vlm_calls * self.VLM_CALL_COST
            vlm_compute = (num_vlm_calls * 1200) * self.LAMBDA_COMPUTE_MS_COST
            
        # Assembly Compute
        assembly_compute = (5000) * self.LAMBDA_COMPUTE_MS_COST
        
        total_compute_cost = extract_compute + yolo_compute + vlm_compute + assembly_compute
        
        # S3 Cost: PUT frames + PUT clips + GET frames + GET clips
        # Approx 1.5MB per frame image
        frame_size = 1.5 * 1024 * 1024 
        s3_put_count = num_frames + 5 # frames + output clips
        s3_get_count = num_frames + 5 # frames read by YOLO + clips read by assemble
        
        s3_req_cost = (s3_put_count * self.S3_PUT_COST) + (s3_get_count * self.S3_GET_COST)
        # S3 Data transfer: reading frames in YOLO Lambdas (1.5MB per frame)
        # and reading clips for assembly
        bytes_transferred = (num_frames * frame_size) + (5 * frame_size * 5)
        s3_transfer_cost = bytes_transferred * self.S3_DATA_TRANSFER_OUT_COST
        
        total_cost = invoke_cost + total_compute_cost + vlm_api_cost + s3_req_cost + s3_transfer_cost
        
        return total_cost, total_latency

    def estimate_container(self, features, will_trigger_vlm=False):
        """
        Estimates cost (USD) and latency (sec) for Container execution.
        """
        duration = features["duration"]
        motion = features["motion_intensity"]
        system_load = features["system_load"]
        
        # --- Latency Estimation ---
        # Queue delay: if system load is high (CPU/RAM or pending jobs)
        # We model load threshold. Standard queue wait is load * 0.15s
        queue_delay = system_load * 0.15
        
        # Video download: download original video from S3 (e.g. 5MB/s)
        # Let's say video size is duration * 1MB
        video_size = duration * 1024 * 1024
        download_latency = video_size / (5.0 * 1024 * 1024) 
        
        # Local frame extraction (fast, direct to local disk, no S3 PUTs)
        extract_latency = 1.0 + (0.01 * duration)
        
        # YOLO batch detection (run sequentially inside the container, but fast local processing)
        num_frames = max(1, int(duration))
        # local inference is ~0.04s per frame (no network overhead)
        yolo_latency = 0.04 * num_frames
        
        # VLM evaluation
        vlm_latency = 0.0
        if will_trigger_vlm:
            num_vlm_calls = max(1, num_frames // 5)
            # Batch VLM calling has lower serialization overhead, but API time is sequential
            vlm_latency = 1.5 + (0.5 * num_vlm_calls)
            
        # Local assembly (very fast, no downloads needed from S3)
        assembly_latency = 0.5 + (0.01 * num_frames)
        
        # Local upload (upload only the final output video)
        upload_latency = 1.5
        
        total_latency = (queue_delay + download_latency + extract_latency + 
                         yolo_latency + vlm_latency + assembly_latency + upload_latency)
        
        # --- Cost Estimation ---
        # Dedicated container execution time:
        active_time = total_latency - queue_delay # We don't charge for queue delay on shared resources
        container_compute_cost = active_time * self.CONTAINER_SEC_COST
        
        # S3 Cost: Only 1 GET for original video, 1 PUT for final video, and final download
        s3_req_cost = (1 * self.S3_GET_COST) + (1 * self.S3_PUT_COST)
        s3_transfer_cost = video_size * self.S3_DATA_TRANSFER_OUT_COST
        
        # VLM API Cost
        vlm_api_cost = 0.0
        if will_trigger_vlm:
            num_vlm_calls = max(1, num_frames // 5)
            vlm_api_cost = num_vlm_calls * self.VLM_CALL_COST
            
        total_cost = container_compute_cost + s3_req_cost + s3_transfer_cost + vlm_api_cost
        
        return total_cost, total_latency

    def select_backend(self, features, will_trigger_vlm=False):
        """
        Calculates objectives for Serverless and Container pathways.
        Returns selected backend ('serverless' or 'container') and detailed metrics.
        """
        s_cost, s_latency = self.estimate_serverless(features, will_trigger_vlm)
        c_cost, c_latency = self.estimate_container(features, will_trigger_vlm)
        
        s_objective = s_cost + (self.lambda_weight * s_latency)
        c_objective = c_cost + (self.lambda_weight * c_latency)
        
        decision = "serverless" if s_objective <= c_objective else "container"
        
        return decision, {
            "serverless": {"cost": s_cost, "latency": s_latency, "objective": s_objective},
            "container": {"cost": c_cost, "latency": c_latency, "objective": c_objective},
            "decision": decision,
            "lambda": self.lambda_weight
        }
