"""
analyze_dataset.py -- Feature distribution analysis for workload videos.

Extracts features from every video in workload.csv (or a directory),
shows distribution plots, and prints concrete recommendations about
what kinds of videos you still need to add.

Usage:
    python analyze_dataset.py                        # reads workload.csv
    python analyze_dataset.py --workload my.csv
    python analyze_dataset.py --dir storage/videos   # scan a directory
    python analyze_dataset.py --save                 # save plots to PNG
"""

import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend; works without a display
import matplotlib.pyplot as plt


# ── Feature extraction ──────────────────────────────────────────────────────

def extract_features_for_videos(video_paths: list[str]) -> list[dict]:
    from src.features import FeatureExtractor
    extractor = FeatureExtractor()
    records   = []
    total     = len(video_paths)
    for i, vp in enumerate(video_paths, 1):
        if not os.path.exists(vp):
            print(f"  [{i}/{total}] SKIP (not found): {vp}")
            continue
        print(f"  [{i}/{total}] Extracting: {os.path.basename(vp)}")
        try:
            feats = extractor.extract_all(vp)
            feats["video"] = os.path.basename(vp)
            records.append(feats)
        except Exception as e:
            print(f"           ERROR: {e}")
    return records


def load_video_paths_from_csv(csv_path: str) -> list[str]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        seen   = set()
        result = []
        for row in csv.DictReader(f):
            v = row.get("video", "").strip()
            if v and v not in seen:
                seen.add(v)
                result.append(v)
    return result


def load_video_paths_from_dir(directory: str) -> list[str]:
    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in exts
    ]


# ── Bucketing helpers ────────────────────────────────────────────────────────

FEATURE_BUCKETS = {
    "duration": [
        (0,   15,  "short  (<15s)"),
        (15,  60,  "medium (15-60s)"),
        (60,  999, "long   (>60s)"),
    ],
    "motion_intensity": [
        (0,   5,   "low    (<5)"),
        (5,   20,  "medium (5-20)"),
        (20,  999, "high   (>20)"),
    ],
    "visual_entropy": [
        (0,   6.0, "low    (<6)"),
        (6.0, 7.0, "medium (6-7)"),
        (7.0, 9.0, "high   (>7)"),
    ],
    "scene_changes": [
        (0,   1,   "none   (0)"),
        (1,   6,   "some   (1-5)"),
        (6,   999, "many   (>5)"),
    ],
}

MIN_PER_BUCKET = 6   # recommendation threshold


def bucket_label(value, buckets):
    for lo, hi, label in buckets:
        if lo <= value < hi:
            return label
    return buckets[-1][2]


# ── Text analysis ────────────────────────────────────────────────────────────

def print_distribution_report(records: list[dict]):
    if not records:
        print("No records to analyse.")
        return

    features = ["duration", "motion_intensity", "visual_entropy", "scene_changes", "system_load"]
    print("\n" + "="*60)
    print("FEATURE DISTRIBUTION SUMMARY")
    print("="*60)

    for feat in features:
        vals = [r[feat] for r in records if feat in r]
        if not vals:
            continue
        print(f"\n{feat}:")
        print(f"  n={len(vals)}  min={min(vals):.2f}  max={max(vals):.2f}"
              f"  mean={np.mean(vals):.2f}  std={np.std(vals):.2f}")

    print("\n" + "="*60)
    print("BUCKET COVERAGE  (target: >={} per bucket)".format(MIN_PER_BUCKET))
    print("="*60)

    recommendations = []
    for feat, buckets in FEATURE_BUCKETS.items():
        vals = [r[feat] for r in records if feat in r]
        print(f"\n{feat}:")
        for lo, hi, label in buckets:
            count = sum(1 for v in vals if lo <= v < hi)
            bar   = "█" * count + "░" * max(0, MIN_PER_BUCKET - count)
            flag  = "✓" if count >= MIN_PER_BUCKET else "⚠ NEED MORE"
            print(f"  {label:20s}  {bar}  {count}/{MIN_PER_BUCKET}  {flag}")
            if count < MIN_PER_BUCKET:
                recommendations.append((feat, label, count))

    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    if not recommendations:
        print("\n✓ All buckets are well covered. Your dataset looks balanced!")
    else:
        print(f"\nYou need more videos in these categories:\n")
        for feat, label, count in recommendations:
            need = MIN_PER_BUCKET - count
            tip  = _tip(feat, label)
            print(f"  • {feat} → {label.strip()}")
            print(f"    Add {need} more video(s).  Tip: {tip}")

    print_outlier_report(records)


def print_outlier_report(records: list[dict]):
    """IQR-based outlier detection per feature. Lists the specific video filenames."""
    features = ["duration", "motion_intensity", "visual_entropy", "scene_changes"]

    print("\n" + "="*60)
    print("OUTLIER DETECTION  (IQR method, threshold = 1.5×IQR)")
    print("="*60)

    any_outlier = False
    for feat in features:
        vals  = np.array([r[feat] for r in records if feat in r])
        names = [r["video"]  for r in records if feat in r]

        q1, q3 = np.percentile(vals, 25), np.percentile(vals, 75)
        iqr    = q3 - q1
        lo     = q1 - 1.5 * iqr
        hi     = q3 + 1.5 * iqr

        outliers = [(name, val) for name, val in zip(names, vals)
                    if val < lo or val > hi]

        if outliers:
            any_outlier = True
            print(f"\n{feat}  (IQR={iqr:.2f}, normal range [{lo:.2f}, {hi:.2f}]):")
            for name, val in sorted(outliers, key=lambda x: x[1]):
                direction = "▼ LOW" if val < lo else "▲ HIGH"
                print(f"  {direction}  {val:8.2f}  {name}")
        else:
            print(f"\n{feat}:  no outliers")

    if not any_outlier:
        print("\n✓ No outliers detected across all features.")

    print()


def _tip(feat, label):
    tips = {
        ("duration",        "short"):  "Pexels clips under 15s (GIFs/teasers)",
        ("duration",        "medium"): "Standard stock video 15-60s",
        ("duration",        "long"):   "Full scenes, dash-cam recordings, 1-2 min clips",
        ("motion_intensity","low"):    "Static camera: interview, landscape, timelapse",
        ("motion_intensity","medium"): "Walking cam, slow traffic, normal outdoor scenes",
        ("motion_intensity","high"):   "Sports, fast traffic, action camera footage",
        ("visual_entropy",  "low"):    "Simple scenes: sky, empty road, plain backgrounds",
        ("visual_entropy",  "medium"): "Typical street scenes, moderate clutter",
        ("visual_entropy",  "high"):   "Dense scenes: market, crowd, busy intersection",
        ("scene_changes",   "none"):   "Single continuous shot with no cuts",
        ("scene_changes",   "some"):   "Short film clips with 1-5 cuts",
        ("scene_changes",   "many"):   "News footage, sports highlights with many cuts",
    }
    key = (feat, label.split()[0].lower().strip("<(>"))
    return tips.get(key, "Search Pexels for variety in this dimension.")


# ── Plot ─────────────────────────────────────────────────────────────────────

def make_plots(records: list[dict], save: bool, out_dir: str = "."):
    if not records:
        return

    features = ["duration", "motion_intensity", "visual_entropy", "scene_changes"]
    labels   = ["Duration (s)", "Motion Intensity", "Visual Entropy", "Scene Changes"]
    data     = {f: [r[f] for r in records if f in r] for f in features}

    # Normalise each feature to [0, 1]
    norm_data = {}
    for feat in features:
        vals = data[feat]
        lo, hi = min(vals), max(vals)
        rng = hi - lo or 1
        norm_data[feat] = [(v - lo) / rng for v in vals]

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f0f0f0")

    x_pos = list(range(len(features)))
    for i in range(len(records)):
        y_vals = [norm_data[f][i] for f in features if i < len(norm_data[f])]
        ax.plot(x_pos[:len(y_vals)], y_vals,
                "o-", alpha=0.5, linewidth=1.2, markersize=4)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, color="#333333", fontsize=10)
    ax.set_ylabel("Normalised value", color="#555555", fontsize=9)
    ax.set_title(f"Parallel Coordinates  —  each line is one video  (n={len(records)})",
                 color="#222222", fontsize=11, fontweight="bold")
    ax.tick_params(colors="#555555")
    for spine in ax.spines.values():
        spine.set_edgecolor("#bbbbbb")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "dataset_analysis.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"[Plot] Saved to {os.path.abspath(out_path)}")
    plt.close()





# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Dataset feature distribution analyser")
    parser.add_argument("--workload", default="workload.csv",
                        help="CSV file with video paths (default: workload.csv)")
    parser.add_argument("--dir",  default=None,
                        help="Scan a directory for videos instead of reading CSV")
    parser.add_argument("--save", action="store_true",
                        help="Save the plot image (default: always saves to dataset_analysis.png)")
    args = parser.parse_args()

    print("\n[Analyser] Loading video list...")
    if args.dir:
        video_paths = load_video_paths_from_dir(args.dir)
        print(f"  Found {len(video_paths)} video(s) in {args.dir}")
    else:
        video_paths = load_video_paths_from_csv(args.workload)
        print(f"  Found {len(video_paths)} unique video(s) in {args.workload}")

    if not video_paths:
        print("No videos found. Add entries to workload.csv or specify --dir.")
        sys.exit(0)

    print("\n[Analyser] Extracting features...")
    records = extract_features_for_videos(video_paths)

    print_distribution_report(records)
    make_plots(records, args.save)


if __name__ == "__main__":
    main()
