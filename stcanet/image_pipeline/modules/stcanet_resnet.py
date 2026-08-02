"""
STCANet-enhanced ResNet backbone for FastReID.

Wraps FastReID's standard ResNet backbone (built via build_resnet_backbone)
and injects the IAT (CAM+SAM) attention module after layer2 and layer3,
matching the original STCANet paper's 2D architecture exactly:

    conv1 -> layer1 -> layer2 -> IAT(512) -> layer3 -> IAT(1024) -> layer4

The attention maps produced by each IAT call (one per body-part group:
head, upper_clothes, lower_clothes, shoes) are stored as instance
attributes after forward() so the meta-architecture (e.g. Baseline) can
retrieve them and compute the mask-supervision loss against the SCHP
ground-truth masks.

This backbone is registered separately (build_stcanet_resnet_backbone) so
it can be selected via MODEL.BACKBONE.NAME in a config file without
modifying FastReID's original resnet.py -- keeping upstream diffs minimal
and baseline/IAT-enhanced configs easily comparable side by side.

CONSTRAINT: Non-local blocks (MODEL.BACKBONE.WITH_NL) are not supported
here, since this wrapper calls each ResNet stage as a plain nn.Sequential
(self.layer2(x)) rather than replicating FastReID's per-block NL-insertion
loop. This is fine for our use case, since STCANet's own CAM+SAM attention
serves the same "non-local" role. An explicit check raises an error if
WITH_NL is accidentally enabled, to avoid a silent architecture mismatch.
"""

import torch
from torch import nn

from fastreid.modeling.backbones.build import BACKBONE_REGISTRY
from fastreid.modeling.backbones.resnet import build_resnet_backbone

from .iat import IAT


class STCANetResNetBackbone(nn.Module):
    """Wraps a FastReID ResNet backbone, inserting IAT (CAM+SAM) attention
    after layer2 (512 channels) and layer3 (1024 channels)."""

    def __init__(self, base_backbone):
        super().__init__()
        self.conv1 = base_backbone.conv1
        self.bn1 = base_backbone.bn1
        self.relu = base_backbone.relu
        self.maxpool = base_backbone.maxpool
        self.layer1 = base_backbone.layer1
        self.layer2 = base_backbone.layer2
        self.layer3 = base_backbone.layer3
        self.layer4 = base_backbone.layer4

        self.IAT2 = IAT(512)
        self.IAT3 = IAT(1024)

        # populated after each forward() call, for the meta-arch's mask loss
        self.last_attn_layer2 = None
        self.last_attn_layer3 = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)

        x2 = self.layer2(x1)
        x2, a2 = self.IAT2(x2)
        self.last_attn_layer2 = a2

        x3 = self.layer3(x2)
        x3, a3 = self.IAT3(x3)
        self.last_attn_layer3 = a3

        x4 = self.layer4(x3)

        return x4


@BACKBONE_REGISTRY.register()
def build_stcanet_resnet_backbone(cfg):
    """
    Builds a STCANet-enhanced ResNet backbone: standard FastReID ResNet
    (with pretrained weights, IBN, etc. as configured) plus IAT (CAM+SAM)
    attention after layer2 and layer3.
    """
    if cfg.MODEL.BACKBONE.WITH_NL:
        raise NotImplementedError(
            "STCANetResNetBackbone does not support MODEL.BACKBONE.WITH_NL=True. "
            "STCANet's own CAM+SAM attention (IAT) serves the same non-local "
            "role; disable WITH_NL in the config."
        )

    base_backbone = build_resnet_backbone(cfg)
    return STCANetResNetBackbone(base_backbone)
