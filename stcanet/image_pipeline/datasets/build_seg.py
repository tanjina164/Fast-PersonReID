"""
Mirrors fastreid.data.build.build_reid_train_loader, but constructs
STCANetCommDataset (mask-aware) instead of the plain CommDataset.

Also provides a held-out validation split: a fixed percentage of images
per identity are set aside (not used for training) so validation loss
(CE + Triplet + Mask) can be tracked during training to monitor
overfitting. The split is per-identity (stratified), so both the train
and validation subsets cover the same set of person IDs -- this keeps
the CE classifier's label space consistent between train and val loss
computation.
"""

import logging
import os
import random
from collections import defaultdict

from fastreid.utils import comm
from fastreid.data import samplers
from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.transforms import build_transforms
from fastreid.data.data_utils import DataLoaderX
from fastreid.data.build import fast_batch_collator

from .common_seg import STCANetCommDataset

_root = os.getenv("FASTREID_DATASETS", "datasets")


def _load_train_items(cfg):
    train_items = list()
    for d in cfg.DATASETS.NAMES:
        data = DATASET_REGISTRY.get(d)(root=_root)
        if comm.is_main_process():
            data.show_train()
        train_items.extend(data.train)
    return train_items


def _split_train_val(train_items, val_ratio=0.1, seed=42):
    """
    Per-identity stratified split. For each pid, val_ratio fraction of its
    images (at least 0, and only if the identity has more than 1 image --
    identities with just 1 image stay fully in train, since holding out
    their only image would remove them from the training label space)
    go to the validation subset; the rest stay in train.
    """
    by_pid = defaultdict(list)
    for item in train_items:
        by_pid[item[1]].append(item)

    rng = random.Random(seed)
    train_subset, val_subset = [], []

    for pid, items in by_pid.items():
        items = items.copy()
        rng.shuffle(items)
        n_val = int(len(items) * val_ratio)
        if len(items) <= 1:
            n_val = 0  # never remove an identity's only image from train
        val_subset.extend(items[:n_val])
        train_subset.extend(items[n_val:])

    return train_subset, val_subset


def _build_loader(cfg, items, transforms, is_train_sampler=True):
    dataset = STCANetCommDataset(items, transforms, relabel=True)

    mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // comm.get_world_size()
    num_instance = cfg.DATALOADER.NUM_INSTANCE

    if is_train_sampler:
        sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
        if sampler_name == "NaiveIdentitySampler":
            sampler = samplers.NaiveIdentitySampler(dataset.img_items, mini_batch_size, num_instance)
        elif sampler_name == "TrainingSampler":
            sampler = samplers.TrainingSampler(len(dataset))
        else:
            raise ValueError("Unsupported sampler for STCANet loader: {}".format(sampler_name))
    else:
        # validation: still use identity sampling so triplet loss is
        # computable (needs multiple instances per identity per batch)
        sampler = samplers.NaiveIdentitySampler(dataset.img_items, mini_batch_size, num_instance)

    batch_sampler = __import__("torch").utils.data.sampler.BatchSampler(
        sampler, mini_batch_size, True
    )

    loader = DataLoaderX(
        comm.get_local_rank(),
        dataset=dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=fast_batch_collator,
        pin_memory=True,
    )
    return loader, dataset


def build_stcanet_train_loader(cfg, val_ratio=0.0, seed=42):
    """
    If val_ratio > 0, splits off a per-identity validation subset first
    (not included in the returned train loader). Use
    build_stcanet_val_loader with the same val_ratio/seed to get the
    matching validation loader.
    """
    transforms = build_transforms(cfg, is_train=True)
    train_items = _load_train_items(cfg)

    if val_ratio > 0:
        train_items, _ = _split_train_val(train_items, val_ratio, seed)

    logger = logging.getLogger(__name__)
    logger.info(
        "Using training sampler {} (STCANet mask-aware loader, {} train images{})".format(
            cfg.DATALOADER.SAMPLER_TRAIN, len(train_items),
            f", {val_ratio:.0%} held out for validation" if val_ratio > 0 else ""
        )
    )

    loader, _ = _build_loader(cfg, train_items, transforms, is_train_sampler=True)
    return loader


def build_stcanet_val_loader(cfg, val_ratio=0.1, seed=42):
    """
    Builds a validation loader from the per-identity held-out subset
    (same seed/ratio as used when excluding it from the train loader).
    Uses eval-style transforms (no random erasing/flip) so validation
    loss reflects the model's behavior on clean, unaugmented images.
    """
    transforms = build_transforms(cfg, is_train=False)
    train_items = _load_train_items(cfg)
    _, val_items = _split_train_val(train_items, val_ratio, seed)

    logger = logging.getLogger(__name__)
    logger.info("Built STCANet validation set: {} images".format(len(val_items)))

    loader, _ = _build_loader(cfg, val_items, transforms, is_train_sampler=False)
    return loader
