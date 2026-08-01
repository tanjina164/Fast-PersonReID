# Mission 1: SCHP Mask Generation -- Setup Notes

Helper scripts for Mission 1 (replacing SHP-LIP mask generation with SCHP),
including fixes for issues encountered when running SCHP on Kaggle/Colab.

## Issues encountered and fixes

### 1. `requirements.txt` install failure
SCHP's `requirements.txt` pins old (2020) package versions (e.g.
`opencv-python==4.4.0.46`) that have no prebuilt wheel for Python 3.12.
**Fix:** only install what is actually needed (`ninja`, `networkx`, `gdown`).
Everything else (torch, opencv, numpy, PIL) is already available on
Kaggle/Colab.

### 2. `InPlaceABNSync` CUDA extension compile failure
`modules/functions.py` tries to JIT-compile a custom CUDA extension
(`inplace_abn`) at runtime. Its C++/CUDA source (2019) uses a deprecated
PyTorch API (`Tensor.type()`), which fails to compile against newer
PyTorch/CUDA versions.

**Fix:** `bn_patch.py` replaces `InPlaceABNSync` with a plain
`nn.BatchNorm2d` subclass (not a wrapped submodule). No CUDA compilation
is needed, and the inference output is functionally equivalent.

**IMPORTANT:** `ABN` must directly subclass `nn.BatchNorm2d`
(`class ABN(nn.BatchNorm2d)`), not wrap it as `self.bn = nn.BatchNorm2d(...)`,
otherwise the checkpoint's state_dict keys (`bn1.weight`) will not match
the model's keys (`bn1.bn.weight`) and `load_state_dict()` will raise a
RuntimeError.

### 3. Large checkpoint file rejected by GitHub
SCHP checkpoints (~255MB) exceed GitHub's 100MB file size limit, so they
must never be committed. `download_checkpoint.py` downloads the checkpoint
at runtime instead, and `.gitignore` excludes `external/SCHP/checkpoints/`.

## Setup steps (fresh environment)

```bash
# 1. dependencies
pip install ninja networkx gdown --quiet

# 2. apply the bn.py CUDA-free patch
python scripts/mission1_schp/apply_bn_patch.py

# 3. download the LIP pretrained checkpoint
python scripts/mission1_schp/download_checkpoint.py

# 4. generate masks for a full image folder
python scripts/mission1_schp/generate_masks_batch.py \
    --input-dir <MARKET1501_PATH>/bounding_box_train \
    --output-dir data/masks_schp \
    --checkpoint external/SCHP/checkpoints/exp-schp-201908261155-lip.pth
```

`generate_masks_batch.py` uses `label_grouping.py` internally to convert
SCHP's 20-class LIP output into the 4-group (+foreground) binary masks
required by STCANet.
