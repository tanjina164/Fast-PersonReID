# Mission 2: FastReID Integration -- Compatibility Notes

`external/fast-reid` (JDAI-CV/fast-reid) targets Python ~3.7 / PyTorch ~1.6
(circa 2020-2021). Running it as-is on a modern environment (Python 3.12,
PyTorch 2.6+) fails with a series of unrelated-looking errors. All fixes
are collected in `apply_compat_patches.py`.

## Issues encountered and fixes

### 1. `collections.Mapping` removed (Python 3.10+)
`collections.Mapping` was moved to `collections.abc.Mapping`. Affects
`fastreid/evaluation/testing.py` and `fastreid/data/build.py`.

### 2. `torch.load` default changed to `weights_only=True` (PyTorch 2.6+)
Legacy-format pretrained checkpoints (`.pth`/`.tar` saved with old PyTorch)
fail to load under the new default. Fix: pass `weights_only=False`
explicitly at every `torch.load(...)` call site across all backbone files
(resnet, osnet, resnext, mobilenet, mobilenetv3, repvgg, vision_transformer,
regnet, resnest, shufflenet, and `utils/checkpoint.py`).

**Note:** `weights_only=False` is safe here because all checkpoints are
either official PyTorch pretrained weights (`download.pytorch.org`) or
FastReID's own official releases -- not arbitrary untrusted files.

**Implementation note:** the patch uses precise string matching, not
regex, because a naive regex like `torch\.load\(([^)]*)\)` incorrectly
matches nested parentheses (e.g. `torch.device('cpu')` inside the call),
corrupting the patched line.

### 3. Deprecated `torch.cuda.amp` API
`torch.cuda.amp.GradScaler` / `torch.cuda.amp.autocast` are deprecated in
favor of `torch.amp.GradScaler("cuda")` / `torch.amp.autocast("cuda")`.
Using the old API with newer PyTorch causes AMP training to fail with:
