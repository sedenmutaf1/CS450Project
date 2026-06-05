"""
run_batch.py -- Batch runner for a SINGLE analytics path.

Reads workload.csv, runs every (video, query) pair through ONE path,
and appends results to the matching CSV file.

Usage:
    python run_batch.py                      -> serverless path -> serverless_results.csv
    python run_batch.py --k8s               -> K8s container path -> container_results.csv
    python run_batch.py --workload my.csv
"""

import argparse
import csv
import os
import sys
import time

from dotenv import load_dotenv
load_dotenv()


def load_workload(path: str) -> list:
    if not os.path.exists(path):
        print(f"[Batch ERROR] Workload file not found: {path}")
        sys.exit(1)
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("video", "").strip()]
    return rows


def append_csv(csv_path: str, row: dict):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_batch(workload_file: str, use_k8s: bool, model: str, video_dir: str):
    from src.storage import StorageSimulator
    from src.features import FeatureExtractor
    from src.cv_analyzer import CVAnalyzer
    from src.vlm_analyzer import VLMAnalyzer
    from src.semantic_trigger import SemanticTrigger
    from src.llm_query_analyzer import LLMQueryAnalyzer

    rows  = load_workload(workload_file)
    total = len(rows)
    if total == 0:
        print(f"[Batch] No entries in {workload_file}.")
        return

    path_label = "Kubernetes (Minikube)" if use_k8s else "Serverless (simulator)"
    out_csv    = "container_results.csv" if use_k8s else "serverless_results.csv"

    # Resolve video directory (absolute so relative ../input_videos works from any cwd)
    video_dir = os.path.abspath(video_dir)
    print(f"[Batch] Video directory: {video_dir}")

    print(f"\n[Batch] {total} entries  |  Path: {path_label}  |  Output: {out_csv}\n")

    # Shared analyzers (reuse across rows to avoid reloading YOLO each time)
    cv_analyzer      = CVAnalyzer()
    vlm_analyzer     = VLMAnalyzer()
    semantic_trigger = SemanticTrigger()
    llm_analyzer     = LLMQueryAnalyzer(model=model)

    # Build the path object once
    storage = StorageSimulator()
    if use_k8s:
        from container.k8s import ContainerPathK8s
        path = ContainerPathK8s(storage)
    else:
        from src.serverless_path import ServerlessPathSimulator
        path = ServerlessPathSimulator(storage, cv_analyzer, vlm_analyzer, semantic_trigger)

    passed, failed = 0, 0

    for i, row in enumerate(rows, start=1):
        video = row.get("video", "").strip()
        query = row.get("query", "").strip()

        print(f"{'='*60}")
        print(f"[Batch] Entry {i}/{total}: {video}")
        print(f"[Batch] Query : {query}")
        print(f"{'='*60}")

        video_name = video   # bare filename for the CSV row
        video      = os.path.join(video_dir, video_name)   # full path for processing

        if not os.path.exists(video):
            print(f"[Batch] SKIP -- video not found: {video}")
            failed += 1
            continue

        # LLM query analysis
        print("[Batch] Analysing query with LLM...")
        analysis      = llm_analyzer.analyze(query)
        target_labels = analysis["target_labels"]
        print(f"[Batch] Labels: {target_labels}  |  Needs VLM: {analysis['needs_vlm']}")

        # Video feature extraction (for the CSV)
        print("[Batch] Extracting video features...")
        extractor = FeatureExtractor()
        features  = extractor.extract_all(video)

        # Reset storage metrics before the run
        storage.reset_metrics()

        # Run the path
        print(f"[Batch] Running {path_label}...")
        start = time.time()
        try:
            result  = path.run(video, query, target_labels)
            output, logs, ann = result if len(result) == 3 else (*result, {})
            latency = time.time() - start
            metrics = storage.get_metrics()

            # Cost accounting:
            # - K8s: EC2 compute rate × real wall-clock time + S3 overhead
            # - Serverless: tracked per-operation by StorageSimulator
            EC2_RATE  = 0.0000115   # USD/sec (t3.medium on-demand)
            S3_PUT    = 0.000005
            S3_GET    = 0.0000004
            if use_k8s:
                cost = round(latency * EC2_RATE + S3_PUT + S3_GET, 6)
            else:
                cost = round(metrics["total_cost_usd"], 6)

            result_row = {
                "video":            video_name,
                "query":            query,
                "duration":         round(features["duration"], 3),
                "motion_intensity": round(features["motion_intensity"], 3),
                "visual_entropy":   round(features["visual_entropy"], 3),
                "scene_changes":    features["scene_changes"],
                "system_load":      round(features["system_load"], 3),
                "cost":             cost,
                "latency":          round(latency, 3),
                "matched":          bool(ann.get("intervals")),
            }

            append_csv(out_csv, result_row)
            print(f"[Batch] Done. Latency={latency:.1f}s  Cost=${metrics['total_cost_usd']:.6f}")
            print(f"[Batch] Row written to {out_csv}")
            passed += 1

        except Exception as e:
            latency = time.time() - start
            print(f"[Batch] FAILED after {latency:.1f}s: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"[Batch] Complete.  Passed: {passed}  Failed/Skipped: {failed}")
    print(f"[Batch] Results in {out_csv}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="CS450 Single-Path Batch Runner")
    parser.add_argument("--workload", default="workload.csv",
                        help="CSV file with columns: video, query")
    parser.add_argument("--k8s",     action="store_true",
                        help="Use Kubernetes (Minikube) container path")
    parser.add_argument("--model",     default="llama3.2:1b",
                        help="Ollama model for query analysis")
    parser.add_argument("--video-dir", default="../input_videos",
                        help="Directory containing the video files (default: ../input_videos)")
    args = parser.parse_args()
    run_batch(args.workload, args.k8s, args.model, args.video_dir)


if __name__ == "__main__":
    main()
