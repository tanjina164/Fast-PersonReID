"""
Links the SCHP-generated mask dataset (backed up as a permanent Kaggle
Dataset named 'mask-file') into the expected data/masks_schp location.

Kaggle automatically extracts uploaded .zip files when they become a
Dataset, so the mask folders are directly available under the dataset's
input path (no manual unzip needed). This script creates a symlink,
avoiding an unnecessary disk copy.

NOTE: Kaggle's actual input path for a dataset owned by a specific user
is /kaggle/input/datasets/<username>/<dataset-slug>/, not simply
/kaggle/input/<dataset-slug>/. Always verify the exact path with
`os.listdir("/kaggle/input")` if this script fails to find the source.

Since Kaggle sessions reset /kaggle/working/ on every restart, the
generated masks are stored as this permanent Kaggle Dataset instead of
being regenerated (SCHP inference) from scratch every time. Attach the
'mask-file' dataset to the notebook (Add Input -> Datasets -> mask-file),
then run this script.

Usage (from repo root):
    python scripts/mission1_schp/extract_masks_from_kaggle_dataset.py
"""

import os

MASK_SOURCE = "/kaggle/input/datasets/jiniyatanjina/mask-file"
MASK_TARGET = "data/masks_schp"


def link_masks():
    os.makedirs("data", exist_ok=True)

    if os.path.islink(MASK_TARGET):
        if os.path.exists(MASK_TARGET):
            print(f"{MASK_TARGET} is already a valid symlink, skipping.")
            return
        else:
            print(f"{MASK_TARGET} is a broken symlink, removing and relinking.")
            os.remove(MASK_TARGET)

    if os.path.exists(MASK_TARGET):
        if os.path.isdir(MASK_TARGET) and len(os.listdir(MASK_TARGET)) > 0:
            print(f"{MASK_TARGET} already exists and is not empty, skipping.")
            return
        os.rmdir(MASK_TARGET)

    if not os.path.exists(MASK_SOURCE):
        raise FileNotFoundError(
            f"{MASK_SOURCE} not found. Make sure the 'mask-file' dataset "
            "is attached to this notebook (Add Input -> Datasets -> mask-file). "
            "If this still fails, run os.listdir('/kaggle/input') and "
            "os.listdir('/kaggle/input/datasets/<your-username>') to find "
            "the correct path."
        )

    os.symlink(MASK_SOURCE, MASK_TARGET)
    print(f"Symlinked {MASK_TARGET} -> {MASK_SOURCE}")
    print(f"Total mask folders: {len(os.listdir(MASK_TARGET))}")


if __name__ == "__main__":
    link_masks()
