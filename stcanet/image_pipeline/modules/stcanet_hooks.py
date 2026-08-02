"""
MaskLossWarmupHook: toggles model.mask_loss_active based on the current
iteration, implementing the paper's mask-loss warm-up schedule.

ValidationLossHook: periodically computes loss (CE + Triplet + Mask) on a
held-out validation subset (see build_stcanet_val_loader), without
updating model weights, and logs it (prefixed "val_") alongside the
training loss so overfitting can be monitored during training.

Computing loss (as opposed to running full mAP/Rank-1 evaluation) requires
the model's training-mode forward path (heads(features, targets) +
losses()), so validation batches are run with torch.no_grad() while the
model stays in .train() mode -- this disables gradient tracking/weight
updates but keeps the loss-computation code path active. BatchNorm running
stats are briefly affected by validation batches passing through in train
mode; since validation batches are a small fraction of total iterations,
this effect is negligible in practice.
"""

import logging

import torch

from fastreid.engine.train_loop import HookBase

logger = logging.getLogger(__name__)


class MaskLossWarmupHook(HookBase):
    def __init__(self, warmup_epochs, iters_per_epoch):
        self.warmup_iters = warmup_epochs * iters_per_epoch

    def before_step(self):
        model = self.trainer.model
        target = model.module if hasattr(model, "module") else model
        target.mask_loss_active = self.trainer.iter >= self.warmup_iters


class ValidationLossHook(HookBase):
    def __init__(self, val_loader, period_iters, num_batches=5):
        self.val_loader = val_loader
        self.val_loader_iter = iter(val_loader)
        self.period_iters = period_iters
        self.num_batches = num_batches

    def _next_batch(self):
        try:
            return next(self.val_loader_iter)
        except StopIteration:
            self.val_loader_iter = iter(self.val_loader)
            return next(self.val_loader_iter)

    def after_step(self):
        next_iter = self.trainer.iter + 1
        if self.period_iters <= 0 or next_iter % self.period_iters != 0:
            return

        model = self.trainer.model
        target = model.module if hasattr(model, "module") else model

        was_training = target.training
        target.train()  # keep the loss-computation forward path active

        accumulated = {}
        with torch.no_grad():
            for _ in range(self.num_batches):
                batch = self._next_batch()
                batch["images"] = batch["images"].to(target.device)
                batch["targets"] = batch["targets"].to(target.device)
                loss_dict = target(batch)
                for k, v in loss_dict.items():
                    accumulated[k] = accumulated.get(k, 0.0) + float(v)

        for k in accumulated:
            accumulated[k] /= self.num_batches

        total_val_loss = sum(accumulated.values())

        for k, v in accumulated.items():
            self.trainer.storage.put_scalar(f"val_{k}", v, smoothing_hint=False)
        self.trainer.storage.put_scalar("val_total_loss", total_val_loss, smoothing_hint=False)

        logger.info(
            "Validation loss @ iter {}: total={:.4f} ({})".format(
                self.trainer.iter,
                total_val_loss,
                ", ".join(f"{k}={v:.4f}" for k, v in accumulated.items())
            )
        )

        if not was_training:
            target.eval()
