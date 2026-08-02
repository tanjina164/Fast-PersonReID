"""
MaskLossWarmupHook: toggles model.mask_loss_active based on the current
iteration, implementing the paper's mask-loss warm-up schedule (mask loss
disabled for the first WARMUP_EPOCHS epochs, letting the backbone
stabilize on ID classification + triplet loss first).
"""

from fastreid.engine.train_loop import HookBase


class MaskLossWarmupHook(HookBase):
    def __init__(self, warmup_epochs, iters_per_epoch):
        self.warmup_iters = warmup_epochs * iters_per_epoch

    def before_step(self):
        model = self.trainer.model
        target = model.module if hasattr(model, "module") else model
        target.mask_loss_active = self.trainer.iter >= self.warmup_iters
