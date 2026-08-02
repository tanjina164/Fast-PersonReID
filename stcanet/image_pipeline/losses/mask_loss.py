"""
Mask-supervised spatial attention loss, ported from the Deepreid reference
notebook ("Image Based Person ReID STCANet.ipynb"). Logic (L2/L1/BCE
comparison against a bilinearly-resized ground-truth mask, averaged over
all elements) is unchanged from the original.

ONE ADAPTATION for FastReID's AMP (mixed-precision) training: nn.BCELoss
is not autocast-safe (PyTorch raises a RuntimeError if called under
autocast), because a sigmoid immediately followed by BCE in FP16 can lose
precision, biasing gradients. Since the IAT module already applies
sigmoid to produce its attention maps (so we only have the sigmoid
OUTPUT, not the pre-sigmoid logits, available here), we cannot switch to
BCEWithLogitsLoss without changing IAT's output contract. Instead, for
mode='ce', the loss computation is wrapped in
`torch.cuda.amp.autocast(enabled=False)` -- this forces just this loss
computation to run in FP32, which is numerically IDENTICAL to running
nn.BCELoss outside of AMP entirely; it does not change what is computed,
only the precision it is computed in. This is a standard, recommended
pattern for AMP-incompatible loss functions, not an architecture change.
"""

import torch
from torch import nn
from torch.nn import functional as F


class MaskLoss(nn.Module):
    """L2, L1, or cross-entropy (BCE) loss, averaged over all elements."""

    def __init__(self, mode='l2'):
        super(MaskLoss, self).__init__()
        self.mode = mode
        if mode == 'l2':
            self.loss = nn.MSELoss()
        elif mode == 'l1':
            self.loss = nn.L1Loss()
        elif mode == 'ce':
            self.loss = nn.BCELoss()
        else:
            raise ValueError(f"Unsupported MaskLoss mode: {mode}")

    def forward(self, inputs, targets):
        b, c, h, w = inputs.size()
        targets = F.interpolate(targets, (h, w), mode='bilinear', align_corners=True)
        inputs = inputs.view(b, -1)
        targets = targets.view(b, -1)

        if self.mode == 'ce':
            # BCELoss is not autocast-safe; compute in FP32 explicitly.
            with torch.cuda.amp.autocast(enabled=False):
                return self.loss(inputs.float(), targets.float())

        return self.loss(inputs, targets)


def DeepSupervision(criterion, xs, y):
    loss = 0.
    for x in xs:
        loss += criterion(x, y)
    loss /= len(xs)
    return loss
