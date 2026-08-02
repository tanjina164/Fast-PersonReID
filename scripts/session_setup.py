"""
Run this once at the start of every new Kaggle session to restore all
symlinks and environment variables needed by Mission 1 (SCHP masks) and
Mission 2 (FastReID) scripts. Kaggle resets /kaggle/working/ on every
session restart, so anything not committed to the repo or backed by a
persistent Kaggle Dataset must be re-created here.

Usage (from repo root, in a notebook cell):
    exec(open("scripts/session_setup.py").read())
"""

import os
import sys

REPO_ROOT = "/kaggle/working/Fast-PersonReID"
os.chdir(REPO_ROOT)

DATASETS_ROOT = "/kaggle/working/datasets"
os.makedirs(DATASETS_ROOT, exist_ok=True)
market_link = os.path.join(DATASETS_ROOT, "Market-1501-v15.09.15")
market_source = "/kaggle/input/datasets/jiniyatanjina/market1501/Market-1501-v15.09.15"
if os.path.islink(market_link) and not os.path.exists(market_link):
    os.remove(market_link)
if not os.path.exists(market_link):
    os.symlink(market_source, market_link)
os.environ["FASTREID_DATASETS"] = DATASETS_ROOT

os.makedirs(os.path.join(REPO_ROOT, "data"), exist_ok=True)
mask_target = os.path.join(REPO_ROOT, "data", "masks_schp")
mask_source = "/kaggle/input/datasets/jiniyatanjina/mask-file"
if os.path.islink(mask_target) and not os.path.exists(mask_target):
    os.remove(mask_target)
if not os.path.exists(mask_target):
    os.symlink(mask_source, mask_target)

for p in ["/kaggle/working/Fast-PersonReID", "/kaggle/working/Fast-PersonReID/external/fast-reid"]:
    if p not in sys.path:
        sys.path.insert(0, p)

print("Session setup complete.")
print("  FASTREID_DATASETS =", os.environ["FASTREID_DATASETS"])
print("  Market1501 linked:", os.path.exists(market_link), "-", os.listdir(market_link) if os.path.exists(market_link) else "MISSING")
print("  Masks linked:", os.path.exists(mask_target), f"({len(os.listdir(mask_target))} folders)" if os.path.exists(mask_target) else "MISSING")
