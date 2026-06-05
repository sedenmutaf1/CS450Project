import os
import sys
import cv2
import numpy as np
import time
import argparse

# Load environment variables from .env file (if present)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


from src.storage import StorageSimulator
from src.features import FeatureExtractor
from src.scheduler import PredictiveScheduler
from src.cv_analyzer import CVAnalyzer
from src.vlm_analyzer import VLMAnalyzer
from src.semantic_trigger import SemanticTrigger
from src.serverless_path import ServerlessPathSimulator
from src.llm_query_analyzer import LLMQueryAnalyzer
from container.simulator import ContainerPathSimulator
from container.k8s import ContainerPathK8s
from src.video_editor import VideoEditor

def generate_synthetic_video(output_path, duration_sec=15, fps=25):
    """
    Generates a synthetic MP4 video of moving shapes (simulating traffic).
    This allows testing the pipeline locally without requiring a real video.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    width, height = 640, 480
    
    # Try different FOURCC codes for maximum compatibility across OS/FFmpeg configurations
    fourcc_options = [
        cv2.VideoWriter_fourcc(*'mp4v'),
        cv2.VideoWriter_fourcc(*'XVID'),
        cv2.VideoWriter_fourcc(*'MJPG')
    ]
    
    out = None
    for fourcc in fourcc_options:
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if out.isOpened():
            break
            
    if out is None or not out.isOpened():
        print("[Generator WARNING] OpenCV VideoWriter could not open. Writing empty mock video file instead.")
        with open(output_path, "wb") as f:
            f.write(b"mock_video_bytes_123456789")
        return False
        
    num_frames = duration_sec * fps
    
    # Motion configuration
    car_x = 50
    car_y = 200
    person_x = 300
    person_y = 100
    
    for f in range(num_frames):
        # Create a dark gray background
        frame = np.ones((height, width, 3), dtype=np.uint8) * 40
        
        # Draw a "road"
        cv2.rectangle(frame, (0, 180), (width, 300), (80, 80, 80), -1)
        cv2.line(frame, (0, 240), (width, 240), (255, 255, 255), 2)
        
        # 1. Simulate a moving "car" (red rectangle) from frame 0 to 200
        if f < 200:
            cv2.rectangle(frame, (car_x, car_y), (car_x + 80, car_y + 40), (0, 0, 255), -1)
            cv2.putText(frame, "CAR", (car_x + 10, car_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            car_x = (car_x + 4) % (width + 80)
            
        # 2. Simulate a moving "person" (green circle) from frame 100 to 300
        if 100 <= f < 300:
            cv2.circle(frame, (person_x, person_y), 20, (0, 255, 0), -1)
            cv2.putText(frame, "PERSON", (person_x - 30, person_y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            person_y = (person_y + 3) % (height + 40)
            
        # Add frame info text
        cv2.putText(frame, f"Frame: {f}/{num_frames} | Time: {f/fps:.1f}s", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        out.write(frame)
        
    out.release()
    print(f"[Generator] Synthetic video generated at {output_path} ({duration_sec}s, {fps}fps)")
    return True

def run_evaluation(video_path, query, target_labels, will_trigger_vlm_override=None, use_k8s=False):
    import uuid
    run_id = str(uuid.uuid4())[:8]
    output_dir = os.path.join("storage", "outputs", run_id)
    os.makedirs(output_dir, exist_ok=True)

    print("="*60)
    print("STARTING HYBRID VIDEO ANALYTICS EVALUATION")
    print("="*60)
    print(f"Run ID:           {run_id}")
    print(f"Output directory: {os.path.abspath(output_dir)}")
    print(f"Video file: {video_path}")
    print(f"Query: {query}")
    print(f"Targets: {target_labels}")
    
    # 1. Initialize core system analyzers
    storage = StorageSimulator()
    cv_analyzer = CVAnalyzer()
    vlm_analyzer = VLMAnalyzer()
    semantic_trigger = SemanticTrigger()
    
    # 2. Extract Features
    print("\n--- Extracting Video & System Features ---")
    # Simulate current system load (queue_size=2 tasks)
    features = FeatureExtractor.extract_all(video_path, queue_size=2)
    print(f"Duration:         {features['duration']:.2f} seconds")
    print(f"Motion Intensity: {features['motion_intensity']:.2f}")
    print(f"Visual Entropy:   {features['visual_entropy']:.2f}")
    print(f"Scene Changes:    {features['scene_changes']}")
    print(f"System Load L:    {features['system_load']:.2f}")
    
    # 3. Check VLM trigger ahead of scheduling
    if will_trigger_vlm_override is not None:
        will_trigger_vlm = will_trigger_vlm_override
    else:
        will_trigger_vlm = semantic_trigger.should_trigger_vlm_pre_execution(query)
    
    # 4. Compare across lambdas (different values of lambda_weight)
    lambdas = [0.001, 0.05, 1.0, 10.0]
    results = {}
    
    for l_val in lambdas:
        scheduler = PredictiveScheduler(lambda_weight=l_val)
        decision, metrics = scheduler.select_backend(features, will_trigger_vlm)
        results[l_val] = {
            "decision": decision,
            "metrics": metrics
        }
        
    print("\n--- Scheduler Backend Decisions ---")
    print(f"{'Lambda (Weight)':<15} | {'Serverless Objective':<22} | {'Container Objective':<22} | {'Selected Backend'}")
    print("-" * 80)
    for l_val, res in results.items():
        metrics = res["metrics"]
        s_obj = metrics["serverless"]["objective"]
        c_obj = metrics["container"]["objective"]
        dec = res["decision"].upper()
        print(f"{l_val:<15.4f} | {s_obj:<22.6f} | {c_obj:<22.6f} | {dec}")
        
    # 5. Run Execution Baselines
    serverless_sim = ServerlessPathSimulator(storage, cv_analyzer, vlm_analyzer, semantic_trigger)
    if use_k8s:
        print("[main] Using real Kubernetes container path (ContainerPathK8s).")
        container_sim = ContainerPathK8s(storage)
    else:
        container_sim = ContainerPathSimulator(storage, cv_analyzer, vlm_analyzer, semantic_trigger)
    
    print("\n--- Execution Baseline 1: SERVERLESS (Forced) ---")
    storage.reset_metrics()
    s_start = time.time()
    s_result = serverless_sim.run(video_path, query, target_labels)
    s_output, s_logs, s_ann = s_result if len(s_result) == 3 else (*s_result, {})
    s_actual_latency = time.time() - s_start
    s_metrics = storage.get_metrics()

    print("\n--- Execution Baseline 2: CONTAINER (Forced) ---")
    storage.reset_metrics()
    c_start = time.time()
    c_result = container_sim.run(video_path, query, target_labels)
    c_output, c_logs, c_ann = c_result if len(c_result) == 3 else (*c_result, {})
    c_actual_latency = time.time() - c_start
    c_metrics = storage.get_metrics()
    
    # Compare Costs & Latencies
    print("\n" + "="*60)
    print("PERFORMANCE EVALUATION SUMMARY")
    print("="*60)
    print(f"{'Metric':<25} | {'Serverless Path':<15} | {'Container Path':<15}")
    print("-" * 60)
    print(f"{'S3 PUT Requests':<25} | {s_metrics['put_requests']:<15} | {c_metrics['put_requests']:<15}")
    print(f"{'S3 GET Requests':<25} | {s_metrics['get_requests']:<15} | {c_metrics['get_requests']:<15}")
    print(f"{'Data Downloaded (MB)':<25} | {s_metrics['bytes_downloaded']/(1024*1024):<15.2f} | {c_metrics['bytes_downloaded']/(1024*1024):<15.2f}")
    print(f"{'Actual execution (sec)':<25} | {s_actual_latency:<15.2f} | {c_actual_latency:<15.2f}")
    print(f"{'Simulated cost (USD)':<25} | ${s_metrics['total_cost_usd']:<14.6f} | ${c_metrics['total_cost_usd']:<14.6f}")
    
    # Save a comparison summary file in storage
    report_path = os.path.join(output_dir, "evaluation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Video Analytics Performance Evaluation Report\n\n")
        f.write("## Workload Information\n")
        f.write(f"- **Video Path:** `{video_path}`\n")
        f.write(f"- **Duration:** {features['duration']:.2f} seconds\n")
        f.write(f"- **Motion Intensity:** {features['motion_intensity']:.2f}\n")
        f.write(f"- **Visual Entropy:** {features['visual_entropy']:.2f}\n")
        f.write(f"- **Scene Changes:** {features['scene_changes']}\n")
        f.write(f"- **System Load:** {features['system_load']:.2f}\n")
        f.write(f"- **Query:** `{query}`\n")
        f.write(f"- **Target Labels:** `{target_labels}`\n\n")
        
        f.write("## Execution Metrics Comparison\n\n")
        f.write("| Metric | Serverless (Lambda/Step Functions) | Container (Docker Worker) |\n")
        f.write("| --- | --- | --- |\n")
        f.write(f"| **S3 PUT Requests** | {s_metrics['put_requests']} | {c_metrics['put_requests']} |\n")
        f.write(f"| **S3 GET Requests** | {s_metrics['get_requests']} | {c_metrics['get_requests']} |\n")
        f.write(f"| **Data Transferred Out (MB)** | {s_metrics['bytes_downloaded']/(1024*1024):.2f} | {c_metrics['bytes_downloaded']/(1024*1024):.2f} |\n")
        f.write(f"| **Simulated Cost (USD)** | ${s_metrics['total_cost_usd']:.6f} | ${c_metrics['total_cost_usd']:.6f} |\n")
        f.write(f"| **Simulated Latency (sec)** | {s_actual_latency:.2f} s | {c_actual_latency:.2f} s |\n\n")
        
        f.write("## Scheduler Decision Analysis\n\n")
        f.write("The scheduler calculates $C_b(x) + \\lambda L_b(x)$ to select the optimal path. Below is the decision breakdown:\n\n")
        f.write("| Lambda Weight | Serverless Obj | Container Obj | Selection |\n")
        f.write("| --- | --- | --- | --- |\n")
        for l_val, res in results.items():
            f.write(f"| {l_val} | {res['metrics']['serverless']['objective']:.6f} | {res['metrics']['container']['objective']:.6f} | **{res['decision'].upper()}** |\n")
            
    print(f"\n[Evaluation] Evaluation report generated at {os.path.abspath(report_path)}")

    # 6. Create annotated highlight videos
    print("\n--- Generating Annotated Highlight Videos ---")
    abs_video = os.path.abspath(video_path)
    for tag, ann in [("serverless", s_ann), ("container", c_ann)]:
        if ann and ann.get("intervals") and ann.get("frame_annotations"):
            ann_path = os.path.join(output_dir, f"{tag}_annotated.mp4")
            VideoEditor.create_annotated_highlight_video(
                abs_video,
                ann["intervals"],
                ann["frame_annotations"],
                ann_path,
                query,
            )
        else:
            print(f"[Evaluation] No matches for {tag} path — skipping annotated video.")

    # 7. Log results to separate CSVs per path
    try:
        import csv

        shared_fields = ["run_id", "video", "query",
                         "duration", "motion_intensity", "visual_entropy",
                         "scene_changes", "system_load"]
        shared_values = {
            "run_id":           run_id,
            "video":            os.path.basename(video_path),
            "query":            query,
            "duration":         round(features["duration"], 3),
            "motion_intensity": round(features["motion_intensity"], 3),
            "visual_entropy":   round(features["visual_entropy"], 3),
            "scene_changes":    features["scene_changes"],
            "system_load":      round(features["system_load"], 3),
        }

        def append_csv(csv_path, row):
            fieldnames = list(row.keys())
            write_header = not os.path.exists(csv_path)
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerow(row)

        # Serverless results
        s_row = {**shared_values,
                 "cost":    round(s_metrics["total_cost_usd"], 6),
                 "latency": round(s_actual_latency, 3)}
        append_csv("serverless_results.csv", s_row)
        print(f"[DataLogger] Serverless row → serverless_results.csv")

        # Container results
        c_row = {**shared_values,
                 "cost":    round(c_metrics["total_cost_usd"], 6),
                 "latency": round(c_actual_latency, 3)}
        append_csv("container_results.csv", c_row)
        print(f"[DataLogger] Container row  → container_results.csv")


    except Exception as e:
        print(f"[DataLogger WARNING] Could not write training_data.csv: {e}")

    
    # 6. Generate Plot if Matplotlib is installed
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Plot predicted costs vs latencies
        s_pred_cost, s_pred_lat = results[1.0]["metrics"]["serverless"]["cost"], results[1.0]["metrics"]["serverless"]["latency"]
        c_pred_cost, c_pred_lat = results[1.0]["metrics"]["container"]["cost"], results[1.0]["metrics"]["container"]["latency"]
        
        ax.scatter([s_pred_lat], [s_pred_cost], color='red', s=150, label='Predicted Serverless', marker='o')
        ax.scatter([c_pred_lat], [c_pred_cost], color='blue', s=150, label='Predicted Container', marker='s')
        
        ax.scatter([s_actual_latency], [s_metrics['total_cost_usd']], color='darkred', s=150, label='Actual Serverless', marker='x')
        ax.scatter([c_actual_latency], [c_metrics['total_cost_usd']], color='darkblue', s=150, label='Actual Container', marker='x')
        
        ax.set_title("Cost-Latency Trade-Off Analysis")
        ax.set_xlabel("Latency (Seconds)")
        ax.set_ylabel("Cost (USD)")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        
        plot_path = os.path.join(output_dir, "cost_latency_tradeoff.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"[Evaluation] Plot saved at {os.path.abspath(plot_path)}")
    except Exception as e:
        print(f"[Evaluation WARNING] Could not generate plot: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid Video Analytics Simulation Runner")
    parser.add_argument("--video", type=str, default="storage/videos/test_traffic.mp4", help="Path to input video")
    parser.add_argument("--query", type=str, default="find all frames with a red car moving on the road", help="Semantic analytics query")
    parser.add_argument("--targets", type=str, default=None, help="(Optional) Comma-separated YOLO target objects. If omitted, inferred from query by LLM.")
    parser.add_argument("--model",    type=str, default="llama3.2:1b", help="Ollama model for query analysis")
    parser.add_argument("--duration", type=int, default=15, help="Synthetic video duration (seconds)")
    parser.add_argument("--k8s",      action="store_true",
                        help="Use real Kubernetes (Minikube) for container path instead of simulator")

    args = parser.parse_args()

    # If the video path doesn't exist and matches default, generate a synthetic one
    if not os.path.exists(args.video) and args.video == "storage/videos/test_traffic.mp4":
        print(f"Default video {args.video} not found. Generating synthetic video...")
        generate_synthetic_video(args.video, duration_sec=args.duration)

    # --- Query Analysis: LLM or manual override ---
    will_trigger_vlm_override = None
    if args.targets is not None:
        # User explicitly provided targets — skip LLM, use keyword check for VLM flag
        target_labels = [t.strip() for t in args.targets.split(",")]
        print(f"\n[QueryAnalysis] Using user-provided targets: {target_labels}")
    else:
        # Let the LLM analyze the query
        print("\n--- Query Analysis (LLM) ---")
        analyzer = LLMQueryAnalyzer(model=args.model)
        analysis = analyzer.analyze(args.query)
        target_labels = analysis["target_labels"]
        will_trigger_vlm_override = analysis["needs_vlm"]
        print(f"  Source:        {analysis['source']}")
        print(f"  Target labels: {target_labels}")
        print(f"  Needs VLM:     {will_trigger_vlm_override}")
        print(f"  Reasoning:     {analysis['reasoning']}")

    run_evaluation(args.video, args.query, target_labels, will_trigger_vlm_override, use_k8s=args.k8s)

