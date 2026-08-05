"""
Generates a figure showing N sample training images alongside their 4
SCHP-generated body-part masks (head, upper_clothes, lower_clothes,
shoes), one row per image, for use in the project report / presentation.

Usage:
    python scripts/visualize_masks_grid.py --num-samples 5
"""

import argparse
import os
import random

import matplotlib.pyplot as plt
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--num-samples", type=int, default=5)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--output", default="/kaggle/working/mask_visualization_grid.png")
args = parser.parse_args()

MARKET1501_PATH = "/kaggle/working/Fast-PersonReID/data/masks_schp"
IMAGE_ROOT = "/kaggle/input/datasets/jiniyatanjina/market1501/Market-1501-v15.09.15/bounding_box_train"

GROUPS = ["foreground", "head", "upper_clothes", "lower_clothes", "shoes"]
GROUP_TITLES = ["Original", "Foreground", "Head", "Upper-clothes", "Lower-clothes", "Shoes"]

random.seed(args.seed)

# pick sample_ids that have a valid, fully-generated mask folder
all_ids = os.listdir(MARKET1501_PATH)
random.shuffle(all_ids)

sample_ids = []
for image_id in all_ids:
    mask_dir = os.path.join(MARKET1501_PATH, image_id)
    if all(os.path.exists(os.path.join(mask_dir, f"{g}.png")) for g in GROUPS):
        img_path = os.path.join(IMAGE_ROOT, image_id + ".jpg")
        if os.path.exists(img_path):
            sample_ids.append(image_id)
    if len(sample_ids) >= args.num_samples:
        break

print(f"Selected {len(sample_ids)} samples: {sample_ids}")

n_cols = 1 + len(GROUPS)  # original + foreground + 4 parts
fig, axes = plt.subplots(len(sample_ids), n_cols, figsize=(3 * n_cols, 3.4 * len(sample_ids)))
if len(sample_ids) == 1:
    axes = axes.reshape(1, -1)

for row, image_id in enumerate(sample_ids):
    img_path = os.path.join(IMAGE_ROOT, image_id + ".jpg")
    orig = Image.open(img_path).convert("RGB")
    axes[row, 0].imshow(orig)
    axes[row, 0].set_title(GROUP_TITLES[0] if row == 0 else "", fontsize=13, fontweight="bold")
    axes[row, 0].set_ylabel(image_id, fontsize=9, rotation=90)
    axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])

    for col, group in enumerate(GROUPS, start=1):
        mask_path = os.path.join(MARKET1501_PATH, image_id, f"{group}.png")
        mask = Image.open(mask_path).convert("L")
        axes[row, col].imshow(mask, cmap="gray")
        if row == 0:
            axes[row, col].set_title(GROUP_TITLES[col], fontsize=13, fontweight="bold")
        axes[row, col].set_xticks([]); axes[row, col].set_yticks([])

plt.suptitle("SCHP-Generated Body-Part Masks (Mission 1)", fontsize=16, fontweight="bold", y=1.0)
plt.tight_layout()
plt.savefig(args.output, dpi=150, bbox_inches="tight")
print(f"\nSaved: {args.output}")
