"""
The original InPlaceABNSync CUDA extension in the modules/bn.py file of the SCHP (Self-Correction Human Parsing) repository uses the old PyTorch API (Tensor.type()), which does not compile on newer PyTorch/CUDA versions (causing nvcc build failures in environments like Kaggle).

This patch replaces InPlaceABNSync with standard nn.BatchNorm2d + activation—requiring no CUDA compilation, and providing inference outputs that are nearly identical.

Usage:

Copy (overwrite) the contents of this file into external/SCHP/modules/bn.py.

Then, ensure the following import is included in external/SCHP/modules/__init__.py:
from .bn import ABN, InPlaceABN, InPlaceABNSync

⚠️ Important: The ABN class must directly subclass nn.BatchNorm2d (instead of creating a separate self.bn sub-module). Otherwise, the state_dict key names (bn1.weight vs bn1.bn.weight) will not match the checkpoint, resulting in a RuntimeError during load_state_dict().
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ABN(nn.BatchNorm2d):
    """Activated Batch Normalization — CUDA-extension-free replacement for InPlaceABNSync.
    Inherits directly from BatchNorm2d so state_dict keys match the original checkpoint."""

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 activation="leaky_relu", slope=0.01):
        super(ABN, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine)
        self.activation = activation
        self.slope = slope

    def forward(self, x):
        x = super(ABN, self).forward(x)
        if self.activation == "leaky_relu":
            return F.leaky_relu(x, negative_slope=self.slope, inplace=True)
        elif self.activation in ("none", "identity"):
            return x
        elif self.activation == "elu":
            return F.elu(x, inplace=True)
        return F.relu(x, inplace=True)


InPlaceABN = ABN
InPlaceABNSync = ABN
