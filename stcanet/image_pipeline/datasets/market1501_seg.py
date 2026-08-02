"""
Market1501 dataset variant that additionally attaches, for each TRAIN
image, the path to its SCHP-generated mask directory (produced by
Mission 1's scripts/mission1_schp/generate_masks_batch.py):

    data/masks_schp/<image_name_without_ext>/{head,upper_clothes,lower_clothes,shoes,foreground}.png

Query/gallery items are left as plain (img_path, pid, camid) tuples --
masks are only needed during training for the attention supervision loss,
not for evaluation.

Registered separately (STCANetMarket1501) so it can be selected via
DATASETS.NAMES in a config file without modifying FastReID's original
market1501.py.

NOTE: mask_root defaults to an ABSOLUTE path resolved from this file's
location (repo_root/data/masks_schp), not a relative path. DataLoader
worker processes may have a different current working directory than
the main process (e.g. training scripts often os.chdir into
external/fast-reid), so a relative path would silently resolve to the
wrong location inside worker processes.
"""

import glob
import os
import os.path as osp
import re
import warnings

from fastreid.data.datasets.bases import ImageDataset
from fastreid.data.datasets import DATASET_REGISTRY

_THIS_FILE = osp.abspath(__file__)
_REPO_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.dirname(_THIS_FILE))))  # .../Fast-PersonReID
_DEFAULT_MASK_ROOT = osp.join(_REPO_ROOT, "data", "masks_schp")


@DATASET_REGISTRY.register()
class STCANetMarket1501(ImageDataset):
    _junk_pids = [0, -1]
    dataset_dir = ''
    dataset_name = "market1501"

    def __init__(self, root='datasets', mask_root=None, **kwargs):
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, 'Market-1501-v15.09.15')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated. Please '
                          'put data folders such as "bounding_box_train" under '
                          '"Market-1501-v15.09.15".')

        self.train_dir = osp.join(self.data_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'bounding_box_test')
        self.mask_root = mask_root if mask_root is not None else _DEFAULT_MASK_ROOT

        required_files = [self.data_dir, self.train_dir, self.query_dir, self.gallery_dir]
        self.check_before_run(required_files)

        train = lambda: self.process_dir(self.train_dir, is_train=True, with_masks=True)
        query = lambda: self.process_dir(self.query_dir, is_train=False)
        gallery = lambda: self.process_dir(self.gallery_dir, is_train=False)

        super(STCANetMarket1501, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True, with_masks=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue
            assert 0 <= pid <= 1501
            assert 1 <= camid <= 6
            camid -= 1
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)

            if with_masks:
                img_name = osp.splitext(osp.basename(img_path))[0]
                mask_dir = osp.join(self.mask_root, img_name)
                data.append((img_path, pid, camid, mask_dir))
            else:
                data.append((img_path, pid, camid))

        return data
