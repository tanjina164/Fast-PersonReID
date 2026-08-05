"""
Generates all charts needed for the project report: training loss curves
(from metrics.json), Rank-1/mAP progression, with-vs-without-rerank bar
chart, comparison-with-reference bar chart, cross-dataset bar chart, and
dataset statistics bar chart. Uses the actual logged/recorded experiment
numbers throughout -- no synthetic data.

Usage:
    python scripts/generate_report_charts.py --metrics-json /path/to/metrics.json --output-dir /kaggle/working/report_charts
"""

import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--metrics-json", default=None, help="Path to metrics.json (optional, for loss curves)")
parser.add_argument("--output-dir", default="/kaggle/working/report_charts")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.titleweight": "bold"})

COLORS = {"cls": "#4C72B0", "triplet": "#DD8452", "mask": "#55A868", "total": "#C44E52"}


# ---------------------------------------------------------------------
# Chart 1: Training loss curves (real, from metrics.json)
# ---------------------------------------------------------------------
if args.metrics_json and os.path.exists(args.metrics_json):
    history = defaultdict(lambda: {"iter": [], "value": []})
    with open(args.metrics_json, "r") as f:
        for line in f:
            entry = json.loads(line)
            it = entry.get("iteration")
            for key, val in entry.items():
                if key == "iteration":
                    continue
                history[key]["iter"].append(it)
                history[key]["value"].append(val)

    plt.figure(figsize=(9, 5.5))
    for key, color in [("total_loss", COLORS["total"]), ("loss_cls", COLORS["cls"]),
                        ("loss_triplet", COLORS["triplet"]), ("loss_mask", COLORS["mask"])]:
        if key in history:
            plt.plot(history[key]["iter"], history[key]["value"], label=key, color=color, linewidth=1.6)
    plt.xlabel("Training Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves (100 Epochs)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "01_loss_curves.png"), dpi=150)
    plt.close()
    print("Saved: 01_loss_curves.png")
else:
    print("metrics.json not provided/found -- skipping Chart 1 (loss curves)")


# ---------------------------------------------------------------------
# Chart 2: Rank-1 / mAP progression across training (real logged checkpoints)
# ---------------------------------------------------------------------
epochs = [10, 20, 100]
rank1 = [82.33, 85.51, 93.17]
mAP = [59.15, 65.03, 79.72]

fig, ax1 = plt.subplots(figsize=(8, 5.5))
ax1.plot(epochs, rank1, marker="o", color=COLORS["cls"], linewidth=2, markersize=8, label="Rank-1")
ax1.plot(epochs, mAP, marker="s", color=COLORS["mask"], linewidth=2, markersize=8, label="mAP")
ax1.set_xlabel("Training Epoch")
ax1.set_ylabel("Accuracy (%)")
ax1.set_title("Rank-1 / mAP Progression on Market-1501 Test Set")
ax1.set_xticks(epochs)
ax1.legend()
ax1.grid(True, alpha=0.3)
for x, y in zip(epochs, rank1):
    ax1.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=10)
for x, y in zip(epochs, mAP):
    ax1.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, -16), ha="center", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, "02_rank1_map_progression.png"), dpi=150)
plt.close()
print("Saved: 02_rank1_map_progression.png")


# ---------------------------------------------------------------------
# Chart 3: With vs Without re-ranking (grouped bar chart)
# ---------------------------------------------------------------------
metrics_names = ["Rank-1", "Rank-5", "Rank-10", "mAP", "mINP"]
no_rerank = [93.17, 97.77, 98.57, 79.72, 44.09]
with_rerank = [95.13, 97.33, 98.13, 93.30, 85.87]

x = np.arange(len(metrics_names))
width = 0.35
fig, ax = plt.subplots(figsize=(9, 5.5))
b1 = ax.bar(x - width / 2, no_rerank, width, label="Without Re-ranking", color=COLORS["cls"])
b2 = ax.bar(x + width / 2, with_rerank, width, label="With Re-ranking", color=COLORS["mask"])
ax.set_ylabel("Score (%)")
ax.set_title("Effect of Re-Ranking on Market-1501 Evaluation Metrics")
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.legend()
ax.bar_label(b1, fmt="%.1f", fontsize=9, padding=2)
ax.bar_label(b2, fmt="%.1f", fontsize=9, padding=2)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, "03_rerank_comparison.png"), dpi=150)
plt.close()
print("Saved: 03_rerank_comparison.png")


# ---------------------------------------------------------------------
# Chart 4: This project vs Paper vs Deepreid baseline (grouped bar)
# ---------------------------------------------------------------------
systems = ["STCANet\n(Paper, 150ep)", "Deepreid\n(SHP-LIP, 50ep)", "This Project\n(SCHP+FastReID, 100ep)"]
mAP_no_rerank = [87.6, 81.9, 79.72]
mAP_rerank = [94.5, 85.1, 93.30]

x = np.arange(len(systems))
width = 0.35
fig, ax = plt.subplots(figsize=(9, 5.5))
b1 = ax.bar(x - width / 2, mAP_no_rerank, width, label="mAP (no re-rank)", color="#8172B2")
b2 = ax.bar(x + width / 2, mAP_rerank, width, label="mAP (with re-rank)", color="#CCB974")
ax.set_ylabel("mAP (%)")
ax.set_title("Comparison with Reference Implementations (Market-1501)")
ax.set_xticks(x)
ax.set_xticklabels(systems, fontsize=10)
ax.legend()
ax.bar_label(b1, fmt="%.1f", fontsize=9, padding=2)
ax.bar_label(b2, fmt="%.1f", fontsize=9, padding=2)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, "04_comparison_reference.png"), dpi=150)
plt.close()
print("Saved: 04_comparison_reference.png")


# ---------------------------------------------------------------------
# Chart 5: Cross-dataset generalization (Market1501 vs DukeMTMC, bar)
# ---------------------------------------------------------------------
datasets = ["Market-1501\n(in-domain)", "DukeMTMC-reID\n(zero-shot)"]
no_rerank_cross = [79.72, 20.93]
with_rerank_cross = [93.30, 34.06]

x = np.arange(len(datasets))
width = 0.35
fig, ax = plt.subplots(figsize=(7.5, 5.5))
b1 = ax.bar(x - width / 2, no_rerank_cross, width, label="mAP (no re-rank)", color=COLORS["cls"])
b2 = ax.bar(x + width / 2, with_rerank_cross, width, label="mAP (with re-rank)", color=COLORS["mask"])
ax.set_ylabel("mAP (%)")
ax.set_title("In-Domain vs Cross-Dataset (Zero-Shot) mAP")
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.legend()
ax.bar_label(b1, fmt="%.1f", fontsize=9, padding=2)
ax.bar_label(b2, fmt="%.1f", fontsize=9, padding=2)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, "05_cross_dataset.png"), dpi=150)
plt.close()
print("Saved: 05_cross_dataset.png")


# ---------------------------------------------------------------------
# Chart 6: Dataset statistics (Market-1501 train/query/gallery counts)
# ---------------------------------------------------------------------
subsets = ["Train", "Query", "Gallery"]
image_counts = [12936, 3368, 15913]
id_counts = [751, 750, 751]

fig, ax1 = plt.subplots(figsize=(8, 5.5))
bars = ax1.bar(subsets, image_counts, color=["#4C72B0", "#DD8452", "#55A868"])
ax1.set_ylabel("Number of Images")
ax1.set_title("Market-1501 Dataset Statistics")
ax1.bar_label(bars, fmt="%d", fontsize=10, padding=3)
for i, (s, idc) in enumerate(zip(subsets, id_counts)):
    ax1.annotate(f"{idc} identities", (i, image_counts[i] / 2), ha="center", color="white", fontsize=10, fontweight="bold")
ax1.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, "06_dataset_statistics.png"), dpi=150)
plt.close()
print("Saved: 06_dataset_statistics.png")

print(f"\nAll charts saved to: {args.output_dir}")
