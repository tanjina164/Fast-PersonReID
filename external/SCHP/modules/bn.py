
import torch
import torch.nn as nn
import torch.nn.functional as F

class ABN(nn.Module):
    """Activated Batch Normalization — fallback replacement for InPlaceABNSync,
    CUDA-extension-free, works fine for inference."""
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, activation="leaky_relu", slope=0.01):
        super(ABN, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine)
        self.activation = activation
        self.slope = slope

    def forward(self, x):
        x = self.bn(x)
        if self.activation == "leaky_relu":
            return F.leaky_relu(x, negative_slope=self.slope, inplace=True)
        elif self.activation == "none":
            return x
        return F.relu(x, inplace=True)

InPlaceABN = ABN
InPlaceABNSync = ABN
