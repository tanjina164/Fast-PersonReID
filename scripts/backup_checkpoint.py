"""
Comprehensive but lightweight backup script. Packages everything needed
for future inference, resuming, cross-validation, and report-writing
(learning curves) into a single small zip:

  1. model_weights.pth       -- weights-only (small, for inference/testing)
  2. model_full_resume.pth   -- full checkpoint incl. optimizer/scheduler
                                 state (for resuming training later)
  3. config.yaml              -- exact architecture/hyperparameters used
                                 (backbone, num_classes, input size, etc.)
  4. metrics.json              -- per-iteration loss/accuracy log, for
                                 plotting learning curves in the report
  5. log.txt                   -- full human-readable training log

Preprocessing parameters (pixel mean/std) are already recorded inside
config.yaml (MODEL.PIXEL_MEAN / MODEL.PIXEL_STD) -- no separate file
needed, since FastReID's Baseline model reads them directly from config
at inference time.

Usage:
    python scripts/backup_checkpoint.py --output-dir /kaggle/working/stcanet_full_training
"""
import argparse
import os
import shutil
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", required=True, help="FastReID OUTPUT_DIR of the training run")
parser.add_argument("--zip-name", default=None, help="Path (without .zip) for the output archive")
parser.add_argument("--checkpoint-name", default="model_final.pth",
                     help="Which checkpoint in output-dir to use as the primary source")
args = parser.parse_args()

output_dir = args.output_dir
run_name = os.path.basename(os.path.normpath(output_dir))
zip_name = args.zip_name or f"/kaggle/working/backup_{run_name}"

staging_dir = "/kaggle/working/_backup_staging"
if os.path.exists(staging_dir):
    shutil.rmtree(staging_dir)
os.makedirs(staging_dir)

ckpt_path = os.path.join(output_dir, args.checkpoint_name)
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # 1. Full checkpoint (weights + optimizer + scheduler), for resuming training
    full_out = os.path.join(staging_dir, "model_full_resume.pth")
    shutil.copy(ckpt_path, full_out)
    print(f"  Staged: model_full_resume.pth ({os.path.getsize(full_out)/1e6:.1f} MB, "
          f"includes optimizer/scheduler state -- use with --resume)")

    # 2. Weights-only (small), for inference/testing
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    weights_out = os.path.join(staging_dir, "model_weights.pth")
    torch.save({"model": state_dict}, weights_out)
    print(f"  Staged: model_weights.pth ({os.path.getsize(weights_out)/1e6:.1f} MB, "
          f"weights-only -- use with --eval-only / MODEL.WEIGHTS)")
else:
    print(f"  WARNING: {ckpt_path} not found, skipping checkpoint")

# 3. Best checkpoint's weights too, if different
best_path = os.path.join(output_dir, "model_best.pth")
if os.path.exists(best_path) and args.checkpoint_name != "model_best.pth":
    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    best_out = os.path.join(staging_dir, "model_best_weights.pth")
    torch.save({"model": state_dict}, best_out)
    print(f"  Staged: model_best_weights.pth ({os.path.getsize(best_out)/1e6:.1f} MB)")

# 4. Config (architecture, hyperparameters, preprocessing pixel mean/std)
config_src = os.path.join(output_dir, "config.yaml")
if os.path.exists(config_src):
    shutil.copy(config_src, staging_dir)
    print(f"  Staged: config.yaml (architecture + hyperparameters + pixel mean/std)")

# 5. Training history (for learning curve plots) and human-readable log
for fname in ["metrics.json", "log.txt"]:
    src = os.path.join(output_dir, fname)
    if os.path.exists(src):
        shutil.copy(src, staging_dir)
        print(f"  Staged: {fname}")

shutil.make_archive(zip_name, "zip", staging_dir)
zip_size_mb = os.path.getsize(zip_name + ".zip") / 1e6

print(f"\n{'='*50}")
print(f"Backup created: {zip_name}.zip ({zip_size_mb:.1f} MB)")
print(f"{'='*50}")
print("Contents:")
print("  - model_weights.pth        : inference/testing/cross-validation")
print("  - model_full_resume.pth    : resume training later (--resume)")
print("  - model_best_weights.pth   : best-mAP checkpoint, weights-only")
print("  - config.yaml              : architecture, hyperparameters, pixel mean/std")
print("  - metrics.json             : per-iteration loss/accuracy history (for learning curves)")
print("  - log.txt                  : full training log")
