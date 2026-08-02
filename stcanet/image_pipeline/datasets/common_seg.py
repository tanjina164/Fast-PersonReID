"""
CommDataset variant that additionally loads the 4 body-part masks
(head, upper_clothes, lower_clothes, shoes) when the underlying dataset
item is a 4-tuple (img_path, pid, camid, mask_dir), i.e. training data
from STCANetMarket1501.

The masks are returned as a single (4, H, W) float tensor, values in
[0, 1], stacked in the order [head, upper_clothes, lower_clothes, shoes]
-- matching the channel order STCANet's IAT module produces
(a[:, 0:1]=head, a[:, 1:2]=upper, a[:, 2:3]=lower, a[:, 3:4]=shoes).

No resizing/interpolation is done here; the mask-supervision loss
interpolates the ground-truth mask to match the (smaller) attention map
resolution at loss-computation time, exactly as in the original STCANet
reference implementation.
"""

import os.path as osp

import numpy as np
import torch
from PIL import Image

from fastreid.data.common import CommDataset

MASK_GROUP_ORDER = ["head", "upper_clothes", "lower_clothes", "shoes"]


def _load_masks(mask_dir):
    """Loads the 4 body-part masks from mask_dir, returns a (4, H, W)
    float tensor with values in [0, 1]."""
    masks = []
    for group in MASK_GROUP_ORDER:
        mask_path = osp.join(mask_dir, f"{group}.png")
        mask = Image.open(mask_path).convert("L")
        mask_arr = np.array(mask, dtype=np.float32) / 255.0
        masks.append(torch.from_numpy(mask_arr))
    return torch.stack(masks, dim=0)  # (4, H, W)


class STCANetCommDataset(CommDataset):
    """Same as CommDataset, but also returns a "masks" key (4, H, W)
    tensor for training items that include a mask_dir (4-tuple items)."""

    def __getitem__(self, index):
        img_item = self.img_items[index]
        img_path = img_item[0]
        pid = img_item[1]
        camid = img_item[2]

        img = self._read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        if self.relabel:
            pid = self.pid_dict[pid]
            camid = self.cam_dict[camid]

        result = {
            "images": img,
            "targets": pid,
            "camids": camid,
            "img_paths": img_path,
        }

        if len(img_item) >= 4:
            mask_dir = img_item[3]
            result["masks"] = _load_masks(mask_dir)

        return result

    @staticmethod
    def _read_image(img_path):
        from fastreid.data.data_utils import read_image
        return read_image(img_path)
