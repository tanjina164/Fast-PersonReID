"""
Mirrors fastreid.data.build.build_reid_train_loader, but constructs
STCANetCommDataset (mask-aware) instead of the plain CommDataset.

Kept as a separate function (not modifying FastReID's original
data/build.py) to keep upstream diffs minimal and make it trivial to
switch between the plain baseline loader and this mask-aware loader via
the training script.
"""

import logging

from fastreid.utils import comm
from fastreid.data import samplers
from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.transforms import build_transforms
from fastreid.data.data_utils import DataLoaderX
from fastreid.data.build import fast_batch_collator
import os

from .common_seg import STCANetCommDataset

_root = os.getenv("FASTREID_DATASETS", "datasets")


def build_stcanet_train_loader(cfg):
    transforms = build_transforms(cfg, is_train=True)

    train_items = list()
    for d in cfg.DATASETS.NAMES:
        data = DATASET_REGISTRY.get(d)(root=_root)
        if comm.is_main_process():
            data.show_train()
        train_items.extend(data.train)

    train_set = STCANetCommDataset(train_items, transforms, relabel=True)

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    num_instance = cfg.DATALOADER.NUM_INSTANCE
    mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // comm.get_world_size()

    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {} (STCANet mask-aware loader)".format(sampler_name))

    if sampler_name == "NaiveIdentitySampler":
        sampler = samplers.NaiveIdentitySampler(train_set.img_items, mini_batch_size, num_instance)
    elif sampler_name == "TrainingSampler":
        sampler = samplers.TrainingSampler(len(train_set))
    else:
        raise ValueError("Unsupported sampler for STCANet loader: {}".format(sampler_name))

    batch_sampler = __import__("torch").utils.data.sampler.BatchSampler(
        sampler, mini_batch_size, True
    )

    train_loader = DataLoaderX(
        comm.get_local_rank(),
        dataset=train_set,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=fast_batch_collator,
        pin_memory=True,
    )
    return train_loader
