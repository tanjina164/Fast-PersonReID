"""
Generates SCHP human-parsing masks for every image in a Market1501-style
image folder, then converts them into STCANet's expected 4-group
(+foreground) mask format:

    <output_dir>/<image_name_without_ext>/head.png
    <output_dir>/<image_name_without_ext>/upper_clothes.png
    <output_dir>/<image_name_without_ext>/lower_clothes.png
    <output_dir>/<image_name_without_ext>/shoes.png
    <output_dir>/<image_name_without_ext>/foreground.png

This mirrors the inference logic in external/SCHP/simple_extractor.py,
reusing SimpleFolderDataset (for correct center/scale handling) and
transform_logits, then applies group mapping instead of saving the raw
20-class palette image.

Usage:
    python scripts/mission1_schp/generate_masks_batch.py \
        --input-dir /path/to/bounding_box_train \
        --output-dir data/masks_schp \
        --checkpoint external/SCHP/checkpoints/exp-schp-201908261155-lip.pth
"""

import os
import sys
import argparse
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Make label_grouping.py importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from label_grouping import generate_group_masks

# Make SCHP's internal modules (networks, utils, datasets) importable
SCHP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "external", "SCHP"))
sys.path.insert(0, SCHP_DIR)

import networks
from utils.transforms import transform_logits
from datasets.simple_extractor_dataset import SimpleFolderDataset

INPUT_SIZE = [473, 473]
NUM_CLASSES = 20  # LIP dataset


def load_schp_model(checkpoint_path, device):
    model = networks.init_model("resnet101", num_classes=NUM_CLASSES, pretrained=None)
    state_dict = torch.load(checkpoint_path, map_location=device)["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k  # strip 'module.' prefix if present
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model


def process_dataset(input_dir, output_dir, checkpoint_path, resume=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_schp_model(checkpoint_path, device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229]),
    ])
    dataset = SimpleFolderDataset(root=input_dir, input_size=INPUT_SIZE, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating masks"):
            image, meta = batch
            img_name = meta["name"][0]
            c = meta["center"].numpy()[0]
            s = meta["scale"].numpy()[0]
            w = meta["width"].numpy()[0]
            h = meta["height"].numpy()[0]

            name_no_ext = os.path.splitext(img_name)[0]
            save_dir = os.path.join(output_dir, name_no_ext)

            if resume and os.path.exists(os.path.join(save_dir, "foreground.png")):
                continue

            output = model(image.to(device))
            upsample = torch.nn.Upsample(size=INPUT_SIZE, mode="bilinear", align_corners=True)
            upsample_output = upsample(output[0][-1][0].unsqueeze(0))
            upsample_output = upsample_output.squeeze()
            upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC

            logits_result = transform_logits(
                upsample_output.data.cpu().numpy(), c, s, w, h, input_size=INPUT_SIZE
            )
            parsing_result = np.argmax(logits_result, axis=2).astype(np.uint8)  # (H, W), values 0-19

            group_masks = generate_group_masks(parsing_result)

            os.makedirs(save_dir, exist_ok=True)
            for group_name, mask_array in group_masks.items():
                Image.fromarray(mask_array).save(os.path.join(save_dir, f"{group_name}.png"))

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--no-resume", action="store_true", help="reprocess even if output already exists")
    args = parser.parse_args()

    process_dataset(args.input_dir, args.output_dir, args.checkpoint, resume=not args.no_resume)
