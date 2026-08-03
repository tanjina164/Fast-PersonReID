"""
Lightweight backup script: copies ONLY the essential files needed for
future cross-validation/testing/resuming, avoiding a full training
output-folder backup (which can be several GB and fail to download).

Copies (if present):
    - model_final.pth      (final checkpoint)
    - model_best.pth        (best-mAP checkpoint, auto-saved by FastReID)
    - checkpoint_ep*.pth    (periodic checkpoints, for resuming mid-training)
    - config.yaml           (exact config used for this run)
    - metrics.json          (per-iteration logged metrics, for plotting loss curves)
    - last_checkpoint       (FastReID's pointer file, needed for --resume)
    - log.txt               (full training log, if present)

Usage:
    python scripts/backup_checkpoint.py --output-dir /kaggle/working/stcanet_full_training
    python scripts/backup_checkpoint.py --output-dir /kaggle/working/stcanet_full_training --zip-name /kaggle/working/my_backup
    python scripts/backup_checkpoint.py --output-dir /kaggle/working/stcanet_full_training --skip-periodic
"""
import argparse
import glob
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", required=True, help="FastReID OUTPUT_DIR of the training run")
parser.add_argument("--zip-name", default=None, help="Path (without .zip) for the output archive")
parser.add_argument("--skip-periodic", action="store_true",
                     help="Skip checkpoint_ep*.pth files (keep only final/best) to save space")
args = parser.parse_args()

output_dir = args.output_dir
run_name = os.path.basename(os.path.normpath(output_dir))
zip_name = args.zip_name or f"/kaggle/working/backup_{run_name}"

staging_dir = "/kaggle/working/_backup_staging"
if os.path.exists(staging_dir):
    shutil.rmtree(staging_dir)
os.makedirs(staging_dir)

files_to_backup = ["model_final.pth", "model_best.pth", "config.yaml",
                   "metrics.json", "last_checkpoint", "log.txt"]

if not args.skip_periodic:
    files_to_backup += [os.path.basename(p) for p in sorted(glob.glob(os.path.join(output_dir, "checkpoint_ep*.pth")))]

total_size = 0
found_any = False
for fname in files_to_backup:
    src = os.path.join(output_dir, fname)
    if os.path.exists(src):
        shutil.copy(src, staging_dir)
        size_mb = os.path.getsize(src) / 1e6
        total_size += size_mb
        print(f"  Staged: {fname} ({size_mb:.1f} MB)")
        found_any = True
    else:
        print(f"  Skipped (not found): {fname}")

if not found_any:
    raise FileNotFoundError(f"No backup-able files found in {output_dir}. Check --output-dir path.")

shutil.make_archive(zip_name, "zip", staging_dir)
zip_size_mb = os.path.getsize(zip_name + ".zip") / 1e6

print(f"\n{'='*50}")
print(f"Backup created: {zip_name}.zip")
print(f"Total size: {zip_size_mb:.1f} MB (uncompressed staged: {total_size:.1f} MB)")
print(f"{'='*50}")
print("\nNext steps:")
print(f"  1. Download {zip_name}.zip from the Output panel")
print(f"  2. Upload it as a new (or updated) Kaggle Dataset")
print(f"  3. In future sessions, use scripts/restore_checkpoint.py to bring it back")
