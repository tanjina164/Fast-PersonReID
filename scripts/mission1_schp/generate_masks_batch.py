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

Corrupt or unreadable images are skipped (logged to a text file) instead
of crashing the whole run.

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
import cv2
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from label_grouping import generate_group_masks

SCHP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "external", "SCHP"))
sys.path.insert(0, SCHP_DIR)

import networks
from utils.transforms import get_affine_transform, transform_logits
import torchvision.transforms as transforms

INPUT_SIZE = [473, 473]
NUM_CLASSES = 20  # LIP dataset


def load_schp_model(checkpoint_path, device):
    model = networks.init_model("resnet101", num_classes=NUM_CLASSES, pretrained=None)
    state_dict = torch.load(checkpoint_path, map_location=device)["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model


def _box2cs(box, aspect_ratio):
    x, y, w, h = box[:4]
    return _xywh2cs(x, y, w, h, aspect_ratio)


def _xywh2cs(x, y, w, h, aspect_ratio):
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5
    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array([w, h], dtype=np.float32)
    return center, scale


def preprocess_image(img_path, input_size, transform):
    """Load and preprocess a single image, mirroring SimpleFolderDataset's
    logic. Returns None if the image cannot be read."""
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        return None, None, None, None, None

    h, w, _ = img.shape
    aspect_ratio = input_size[1] * 1.0 / input_size[0]
    person_center, s = _box2cs([0, 0, w - 1, h - 1], aspect_ratio)
    r = 0
    trans = get_affine_transform(person_center, s, r, input_size)
    inp = cv2.warpAffine(
        img, trans, (int(input_size[1]), int(input_size[0])),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
    )
    inp = transform(inp)
    return inp, person_center, s, w, h


def process_dataset(input_dir, output_dir, checkpoint_path, resume=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_schp_model(checkpoint_path, device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229]),
    ])

    image_files = sorted([
        f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    print(f"Found {len(image_files)} images in {input_dir}")

    os.makedirs(output_dir, exist_ok=True)
    skipped_log_path = os.path.join(output_dir, "_skipped_files.txt")
    skipped = []

    with torch.no_grad():
        for img_name in tqdm(image_files, desc="Generating masks"):
            name_no_ext = os.path.splitext(img_name)[0]
            save_dir = os.path.join(output_dir, name_no_ext)

            if resume and os.path.exists(os.path.join(save_dir, "foreground.png")):
                continue

            img_path = os.path.join(input_dir, img_name)
            inp, c, s, w, h = preprocess_image(img_path, INPUT_SIZE, transform)

            if inp is None:
                skipped.append(img_name)
                continue

            try:
                image = inp.unsqueeze(0).to(device)
                output = model(image)
                upsample = torch.nn.Upsample(size=INPUT_SIZE, mode="bilinear", align_corners=True)
                upsample_output = upsample(output[0][-1][0].unsqueeze(0))
                upsample_output = upsample_output.squeeze().permute(1, 2, 0)

                logits_result = transform_logits(
                    upsample_output.data.cpu().numpy(), c, s, w, h, input_size=INPUT_SIZE
                )
                parsing_result = np.argmax(logits_result, axis=2).astype(np.uint8)

                group_masks = generate_group_masks(parsing_result)

                os.makedirs(save_dir, exist_ok=True)
                for group_name, mask_array in group_masks.items():
                    Image.fromarray(mask_array).save(os.path.join(save_dir, f"{group_name}.png"))
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
                skipped.append(img_name)

    if skipped:
        with open(skipped_log_path, "w") as f:
            f.write("\n".join(skipped))
        print(f"Skipped {len(skipped)} unreadable/failed images. See {skipped_log_path}")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--no-resume", action="store_true", help="reprocess even if output already exists")
    args = parser.parse_args()

    process_dataset(args.input_dir, args.output_dir, args.checkpoint, resume=not args.no_resume)
