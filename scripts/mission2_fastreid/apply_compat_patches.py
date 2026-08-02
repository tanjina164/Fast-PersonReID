"""
Applies all compatibility patches needed to run the JDAI-CV/fast-reid repo
(circa 2021, targeting PyTorch ~1.6) on a modern environment (Python 3.12,
PyTorch 2.6+).

Without these patches, training crashes with a series of unrelated-looking
errors, each caused by an API that FastReID's code relies on being removed
or changed in newer Python/PyTorch versions.

Usage (from repo root, after cloning fast-reid into external/fast-reid):
    python scripts/mission2_fastreid/apply_compat_patches.py
"""

import os

FASTREID_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "external", "fast-reid"
)


def patch_file(relative_path, replacements, description):
    """Apply a list of (old, new) string replacements to a file.
    Prints a warning (does not raise) if an expected 'old' string is not
    found, since the file may already be patched."""
    filepath = os.path.join(FASTREID_ROOT, relative_path)
    if not os.path.exists(filepath):
        print(f"[SKIP] {relative_path} not found")
        return

    with open(filepath, "r") as f:
        content = f.read()

    changed = False
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            changed = True
        elif new not in content:
            print(f"[WARN] Expected pattern not found in {relative_path}: {old[:60]}...")

    if changed:
        with open(filepath, "w") as f:
            f.write(content)
        print(f"[OK] {description}: {relative_path}")
    else:
        print(f"[SKIP] {description} (already applied or pattern missing): {relative_path}")


def main():
    # 1. Python 3.10+: collections.Mapping moved to collections.abc
    patch_file(
        "fastreid/evaluation/testing.py",
        [("from collections import Mapping, OrderedDict",
          "from collections import OrderedDict\nfrom collections.abc import Mapping")],
        "collections.abc.Mapping fix"
    )
    patch_file(
        "fastreid/data/build.py",
        [("from collections import Mapping", "from collections.abc import Mapping")],
        "collections.abc.Mapping fix"
    )

    # 2. PyTorch 2.6+: torch.load default changed to weights_only=True,
    #    which breaks loading legacy-format pretrained checkpoints.
    #    Precise string replacement is used (not regex) because torch.load
    #    calls often contain nested parentheses (e.g. torch.device('cpu')),
    #    which a naive regex would match incorrectly.
    torch_load_patches = [
        ("fastreid/modeling/backbones/resnet.py", [
            ("state_dict = torch.load(cached_file, map_location=torch.device('cpu'))",
             "state_dict = torch.load(cached_file, map_location=torch.device('cpu'), weights_only=False)"),
            ("state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'))",
             "state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'), weights_only=False)"),
        ]),
        ("fastreid/utils/checkpoint.py", [
            ('return torch.load(f, map_location=torch.device("cpu"))',
             'return torch.load(f, map_location=torch.device("cpu"), weights_only=False)'),
        ]),
        ("fastreid/modeling/backbones/osnet.py", [
            ("state_dict = torch.load(cached_file, map_location=torch.device('cpu'))",
             "state_dict = torch.load(cached_file, map_location=torch.device('cpu'), weights_only=False)"),
            ("state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'))",
             "state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'), weights_only=False)"),
        ]),
        ("fastreid/modeling/backbones/shufflenet.py", [
            ('state_dict = torch.load(pretrain_path)["state_dict"]',
             'state_dict = torch.load(pretrain_path, weights_only=False)["state_dict"]'),
        ]),
        ("fastreid/modeling/backbones/resnext.py", [
            ("state_dict = torch.load(cached_file, map_location=torch.device('cpu'))",
             "state_dict = torch.load(cached_file, map_location=torch.device('cpu'), weights_only=False)"),
            ("state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'))['model']",
             "state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'), weights_only=False)['model']"),
        ]),
        ("fastreid/modeling/backbones/mobilenet.py", [
            ("state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'))",
             "state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'), weights_only=False)"),
        ]),
        ("fastreid/modeling/backbones/repvgg.py", [
            ('state_dict = torch.load(pretrain_path, map_location=torch.device("cpu"))',
             'state_dict = torch.load(pretrain_path, map_location=torch.device("cpu"), weights_only=False)'),
        ]),
        ("fastreid/modeling/backbones/vision_transformer.py", [
            ("state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'))",
             "state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'), weights_only=False)"),
        ]),
        ("fastreid/modeling/backbones/mobilenetv3.py", [
            ("state_dict = torch.load(pretrain_path)",
             "state_dict = torch.load(pretrain_path, weights_only=False)"),
        ]),
        ("fastreid/modeling/backbones/regnet/effnet.py", [
            ('state_dict = torch.load(cached_file, map_location=torch.device("cpu"))["model_state"]',
             'state_dict = torch.load(cached_file, map_location=torch.device("cpu"), weights_only=False)["model_state"]'),
            ('state_dict = torch.load(pretrain_path, map_location=torch.device(\'cpu\'))["model_state"]',
             'state_dict = torch.load(pretrain_path, map_location=torch.device(\'cpu\'), weights_only=False)["model_state"]'),
        ]),
        ("fastreid/modeling/backbones/regnet/regnet.py", [
            ("state_dict = torch.load(cached_file, map_location=torch.device('cpu'))['model_state']",
             "state_dict = torch.load(cached_file, map_location=torch.device('cpu'), weights_only=False)['model_state']"),
            ("state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'))",
             "state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'), weights_only=False)"),
        ]),
        ("fastreid/modeling/backbones/resnest.py", [
            ("state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'))",
             "state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'), weights_only=False)"),
        ]),
    ]
    for relative_path, replacements in torch_load_patches:
        patch_file(relative_path, replacements, "torch.load weights_only=False fix")

    # 3. Deprecated torch.cuda.amp API (removed in favor of torch.amp with
    #    an explicit device string). Using the old API on newer PyTorch
    #    causes AMPTrainer to silently fail with
    #    "AssertionError: No inf checks were recorded for this optimizer."
    patch_file(
        "fastreid/engine/train_loop.py",
        [
            ("from torch.cuda.amp import GradScaler", "from torch.amp import GradScaler"),
            ("grad_scaler = GradScaler()", 'grad_scaler = GradScaler("cuda")'),
            ("from torch.cuda.amp import autocast", "from torch.amp import autocast"),
            ("with autocast():", 'with autocast("cuda"):'),
        ],
        "torch.amp API fix"
    )

    # 4. ContiguousParams optimizer wrapper is incompatible with newer
    #    PyTorch's GradScaler inf-check tracking (same root symptom as #3:
    #    "No inf checks were recorded for this optimizer"). Disabling it
    #    is a pure performance trade-off, not a correctness change.
    patch_file(
        "fastreid/engine/defaults.py",
        [("        return build_optimizer(cfg, model)",
          "        return build_optimizer(cfg, model, contiguous=False)")],
        "Disable ContiguousParams (incompatible with AMP on newer PyTorch)"
    )

    print("\nAll compatibility patches applied.")


if __name__ == "__main__":
    main()
