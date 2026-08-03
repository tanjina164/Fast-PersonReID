"""
Restores a checkpoint backup (created by backup_checkpoint.py, uploaded
as a Kaggle Dataset) back into the expected FastReID OUTPUT_DIR location,
so training can be resumed (--resume) or evaluated (--eval-only) directly.

Usage:
    python scripts/restore_checkpoint.py \
        --backup-dataset-path /kaggle/input/datasets/<username>/<dataset-name> \
        --output-dir /kaggle/working/stcanet_full_training
"""
import argparse
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--backup-dataset-path", required=True,
                     help="Path to the attached Kaggle Dataset containing the backup files")
parser.add_argument("--output-dir", required=True,
                     help="FastReID OUTPUT_DIR to restore files into")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

restored = 0
for fname in os.listdir(args.backup_dataset_path):
    src = os.path.join(args.backup_dataset_path, fname)
    dst = os.path.join(args.output_dir, fname)
    if os.path.isfile(src) and not os.path.exists(dst):
        shutil.copy(src, dst)
        print(f"  Restored: {fname}")
        restored += 1

print(f"\nRestored {restored} file(s) into {args.output_dir}")
print("You can now run train_stcanet.py with --resume, or --eval-only with MODEL.WEIGHTS pointing to a restored checkpoint.")
