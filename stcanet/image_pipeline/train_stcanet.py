#!/usr/bin/env python
"""
Training entry-point for the STCANet-enhanced FastReID pipeline (Mission 2).
Mirrors external/fast-reid/tools/train_net.py, with these additions:
  1. add_stcanet_config(cfg) registers MODEL.STCANET.* config keys
  2. build_stcanet_train_loader excludes a per-identity validation subset
     (MODEL.STCANET.VAL_RATIO) so training batches include the "masks" key
  3. MaskLossWarmupHook implements the mask-loss warm-up schedule
  4. ValidationLossHook periodically logs validation loss (CE+Triplet+Mask)
     on the held-out subset, to monitor overfitting during training
"""

import sys
import os

sys.path.append('.')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import stcanet.image_pipeline  # registers all STCANet components

from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from fastreid.utils.checkpoint import Checkpointer

from stcanet.image_pipeline.configs.stcanet_config import add_stcanet_config
from stcanet.image_pipeline.datasets.build_seg import build_stcanet_train_loader, build_stcanet_val_loader
from stcanet.image_pipeline.modules.stcanet_hooks import MaskLossWarmupHook, ValidationLossHook


class STCANetTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_stcanet_train_loader(cfg, val_ratio=cfg.MODEL.STCANET.VAL_RATIO)

    def build_hooks(self):
        hooks = super().build_hooks()

        warmup_epochs = self.cfg.MODEL.STCANET.WARMUP_EPOCHS
        iters_per_epoch = self.iters_per_epoch
        hooks.insert(0, MaskLossWarmupHook(warmup_epochs, iters_per_epoch))

        if self.cfg.MODEL.STCANET.VAL_RATIO > 0:
            val_loader = build_stcanet_val_loader(
                self.cfg, val_ratio=self.cfg.MODEL.STCANET.VAL_RATIO
            )
            hooks.insert(
                1,
                ValidationLossHook(val_loader, self.cfg.MODEL.STCANET.VAL_PERIOD_ITERS)
            )

        return hooks


def setup(args):
    cfg = get_cfg()
    add_stcanet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False
        model = STCANetTrainer.build_model(cfg)
        Checkpointer(model).load(cfg.MODEL.WEIGHTS)
        res = STCANetTrainer.test(cfg, model)
        return res

    trainer = STCANetTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
