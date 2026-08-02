"""
Registers custom MODEL.STCANET.* config keys used by STCANetBaseline and
MaskLossWarmupHook. Must be called (add_stcanet_config(cfg)) before
cfg.merge_from_file(...) in the training script.
"""

from fastreid.config import CfgNode as CN


def add_stcanet_config(cfg):
    cfg.MODEL.STCANET = CN()
    cfg.MODEL.STCANET.ALPHA = 0.5
    cfg.MODEL.STCANET.MASK_LOSS_MODE = 'ce'
    cfg.MODEL.STCANET.WARMUP_EPOCHS = 10
    return cfg
