"""
Importing this package registers all STCANet components with FastReID's
registries (BACKBONE_REGISTRY, META_ARCH_REGISTRY, DATASET_REGISTRY):

    import stcanet.image_pipeline  # registers everything

After this import, the following names become available via FastReID's
config system:
    MODEL.BACKBONE.NAME:       build_stcanet_resnet_backbone
    MODEL.META_ARCHITECTURE:   STCANetBaseline
    DATASETS.NAMES:            STCANetMarket1501
"""

from .modules.iat import IAT
from .modules.stcanet_resnet import STCANetResNetBackbone, build_stcanet_resnet_backbone
from .modules.stcanet_baseline import STCANetBaseline
from .modules.stcanet_hooks import MaskLossWarmupHook
from .datasets.market1501_seg import STCANetMarket1501
from .datasets.common_seg import STCANetCommDataset
from .datasets.build_seg import build_stcanet_train_loader
from .configs.stcanet_config import add_stcanet_config
from .losses.mask_loss import MaskLoss, DeepSupervision
from .losses.attention_supervision import compute_attention_supervision_loss
