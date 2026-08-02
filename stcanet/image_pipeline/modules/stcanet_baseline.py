"""
STCANet-enhanced Baseline meta-architecture for FastReID.

Extends FastReID's standard Baseline (ID classification + backbone
feature extraction) by adding the mask-supervised attention loss from
the original STCANet paper (via IAT attention maps captured by
STCANetResNetBackbone during forward()).

Total training loss = (CE + Triplet, as in the standard Baseline)
                       + alpha * mask_supervision_loss

The mask loss is gated by self.mask_loss_active, a plain attribute
(not a buffer/parameter) toggled externally by MaskLossWarmupHook
(see stcanet_hooks.py), implementing the paper's warm-up schedule.

Registered separately (STCANetBaseline) so it can be selected via
MODEL.META_ARCHITECTURE in a config file without modifying FastReID's
original baseline.py.

NOTE: __init__ must be decorated with @configurable, matching the parent
Baseline class -- FastReID's build_model() calls META_ARCH_REGISTRY.get(name)(cfg),
and @configurable is what allows the class to accept a raw cfg object
(dispatching to from_config() to convert it into keyword arguments).
Without this decorator, __init__ only accepts explicit kwargs, causing
"takes 1 positional argument but 2 were given" when the registry passes cfg directly.
"""

import torch

from fastreid.config import configurable
from fastreid.modeling.meta_arch.baseline import Baseline
from fastreid.modeling.meta_arch.build import META_ARCH_REGISTRY

from ..losses.mask_loss import MaskLoss
from ..losses.attention_supervision import compute_attention_supervision_loss


@META_ARCH_REGISTRY.register()
class STCANetBaseline(Baseline):

    @configurable
    def __init__(self, *, mask_loss_alpha=0.5, mask_loss_mode='ce', **kwargs):
        super().__init__(**kwargs)
        self.mask_loss_alpha = mask_loss_alpha
        self.criterion_mask = MaskLoss(mode=mask_loss_mode)
        # toggled by MaskLossWarmupHook at the start of each iteration
        self.mask_loss_active = False

    @classmethod
    def from_config(cls, cfg):
        base_kwargs = Baseline.from_config(cfg)
        base_kwargs['mask_loss_alpha'] = cfg.MODEL.STCANET.ALPHA
        base_kwargs['mask_loss_mode'] = cfg.MODEL.STCANET.MASK_LOSS_MODE
        return base_kwargs

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images)

        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"]

            if targets.sum() < 0:
                targets.zero_()

            outputs = self.heads(features, targets)
            losses = self.losses(outputs, targets)

            if "masks" in batched_inputs:
                gt_masks = batched_inputs["masks"].to(self.device)
                if self.mask_loss_active:
                    mask_loss = compute_attention_supervision_loss(
                        self.criterion_mask,
                        self.backbone.last_attn_layer2,
                        self.backbone.last_attn_layer3,
                        gt_masks,
                    )
                    losses["loss_mask"] = mask_loss * self.mask_loss_alpha
                else:
                    losses["loss_mask"] = torch.zeros(
                        1, device=self.device, dtype=next(self.parameters()).dtype
                    ).squeeze() * self.mask_loss_alpha

            return losses
        else:
            outputs = self.heads(features)
            return outputs
