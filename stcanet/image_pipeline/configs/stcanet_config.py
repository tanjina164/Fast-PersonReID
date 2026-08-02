"""
Registers custom MODEL.STCANET.* config keys used by STCANetBaseline,
MaskLossWarmupHook, and ValidationLossHook. Must be called
(add_stcanet_config(cfg)) before cfg.merge_from_file(...) in the training
script.
"""

from fastreid.config import CfgNode as CN


def add_stcanet_config(cfg):
    cfg.MODEL.STCANET = CN()
    cfg.MODEL.STCANET.ALPHA = 0.5
    cfg.MODEL.STCANET.MASK_LOSS_MODE = 'ce'
    cfg.MODEL.STCANET.WARMUP_EPOCHS = 10
    cfg.MODEL.STCANET.VAL_RATIO = 0.1          # fraction of train images held out per identity for validation
    cfg.MODEL.STCANET.VAL_PERIOD_ITERS = 200   # compute validation loss every N training iterations
    return cfg
