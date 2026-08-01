# Mission 1: SCHP Mask Generation — Setup Notes

This folder contains helper scripts related to Mission 1 (SHP-LIP → SCHP replacement), along with solutions for issues encountered when running SCHP in environments like Kaggle/Colab.

## Issues Encountered and Solutions

### 1. `requirements.txt` installation failure
SCHP's `requirements.txt` contains old (2020) pinned versions (such as `opencv-python==4.4.0.46`) that do not have prebuilt wheels for Python 3.12.  
**Solution:** Instead of installing the entire requirements.txt, install only the actually required packages (`ninja`, `networkx`). All others (torch, opencv, numpy, PIL) are already pre-installed in Kaggle/Colab.

### 2. `InPlaceABNSync` CUDA extension compilation failure
SCHP's `modules/functions.py` attempts to compile a custom CUDA extension (`inplace_abn`) at runtime, whose C++/CUDA source (from 2019) uses deprecated PyTorch APIs (`Tensor.type()`), causing compilation to fail.  
**Solution:** `bn_patch.py` replaces `InPlaceABNSync` with standard `nn.BatchNorm2d` (direct subclass, not a nested submodule). For inference, this gives practically identical results without requiring any CUDA compilation.

**⚠️ Important:** The `ABN` class must directly inherit from `nn.BatchNorm2d` (`class ABN(nn.BatchNorm2d)`). Do not create a separate submodule like `self.bn = nn.BatchNorm2d(...)` — otherwise, the checkpoint's `state_dict` keys (`bn1.weight`) will not match the model's keys (`bn1.bn.weight`), resulting in a `RuntimeError` during `load_state_dict()`.

### 3. Inability to push as a submodule
Adding `external/SCHP` as a `git submodule` prevents direct pushing of internal patches (like `bn.py`), because submodules are separate repositories without push access.  
**Solution:** Convert the submodule into a normal tracked folder:
```bash
rm -rf external/SCHP/.git
git rm --cached external/SCHP
rm -f .gitmodules
git add external/SCHP
git commit -m "Convert SCHP from submodule to regular folder"
