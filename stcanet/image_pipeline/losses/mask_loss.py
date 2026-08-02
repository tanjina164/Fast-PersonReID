"""
Mask-supervised spatial attention loss, ported directly from the Deepreid
reference notebook ("Image Based Person ReID STCANet.ipynb"), unchanged.
This is the original STCANet paper's own architecture contribution, not a
deep-person-reid framework utility -- it is preserved exactly per the
project's core constraint (only the mask-generation source and the
backbone/training framework are being replaced).

MaskLoss compares a predicted (b, 1, h, w) attention channel against a
ground-truth (b, 1, h1, w1) mask, bilinearly interpolating the target to
match the prediction's (smaller) spatial resolution.

DeepSupervision averages MaskLoss across multiple predictions (here: the
IAT attention maps produced after layer2 and layer3).
"""

import torch
from torch import nn
from torch.nn import functional as F


class MaskLoss(nn.Module):
    """L2, L1, or cross-entropy (BCE) loss, averaged over all elements."""

    def __init__(self, mode='l2'):
        super(MaskLoss, self).__init__()
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
        return self.loss(inputs, targets)


def DeepSupervision(criterion, xs, y):
    loss = 0.
    for x in xs:
        loss += criterion(x, y)
    loss /= len(xs)
    return loss
