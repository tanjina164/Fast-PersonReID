from .modules.iat import IAT
from .modules.stcanet_resnet import STCANetResNetBackbone, build_stcanet_resnet_backbone
from .modules.stcanet_baseline import STCANetBaseline
from .modules.stcanet_hooks import MaskLossWarmupHook, ValidationLossHook
from .datasets.market1501_seg import STCANetMarket1501
from .datasets.common_seg import STCANetCommDataset
from .datasets.build_seg import build_stcanet_train_loader, build_stcanet_val_loader
from .configs.stcanet_config import add_stcanet_config
from .losses.mask_loss import MaskLoss, DeepSupervision
from .losses.attention_supervision import compute_attention_supervision_loss
