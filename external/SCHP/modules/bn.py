
import torch
import torch.nn as nn
import torch.nn.functional as F

class ABN(nn.BatchNorm2d):
    """Activated Batch Normalization — CUDA-extension-free replacement for InPlaceABNSync.
    Inherits directly from BatchNorm2d so state_dict keys match the original checkpoint
    (no extra nested submodule prefix)."""
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, activation="leaky_relu", slope=0.01):
        super(ABN, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine)
        self.activation = activation
        self.slope = slope

    def forward(self, x):
        x = super(ABN, self).forward(x)
        if self.activation == "leaky_relu":
            return F.leaky_relu(x, negative_slope=self.slope, inplace=True)
        elif self.activation == "none" or self.activation == "identity":
            return x
        elif self.activation == "elu":
            return F.elu(x, inplace=True)
        return F.relu(x, inplace=True)

InPlaceABN = ABN
InPlaceABNSync = ABN
