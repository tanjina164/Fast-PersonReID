"""
Plots training loss curves (and validation loss, if present) from
FastReID's metrics.json log, for use in the project report.

Usage:
    python scripts/plot_learning_curves.py --metrics-json /path/to/metrics.json --output-dir /path/to/save/plots
"""
import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--metrics-json", required=True)
parser.add_argument("--output-dir", default=".")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

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

loss_keys = [k for k in history if "loss" in k.lower()]
plt.figure(figsize=(10, 6))
for key in loss_keys:
    plt.plot(history[key]["iter"], history[key]["value"], label=key)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training / Validation Loss Curves")
plt.legend()
plt.grid(True, alpha=0.3)
out_path = os.path.join(args.output_dir, "loss_curves.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
