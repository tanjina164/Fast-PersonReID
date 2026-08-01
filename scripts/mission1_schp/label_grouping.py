"""
LIP dataset (SCHP checkpoint 'lip') এর 20-class parsing output কে
STCANet architecture-এর দরকারি 4-group (+ foreground) binary mask-এ
কনভার্ট করার জন্য।

STCANet-এর IAT module hardcoded 4-channel spatial attention আউটপুট দেয়
(SpatialAttn(in_channels, number=4)), তাই group সংখ্যা অবশ্যই 4 রাখতে হবে —
architecture নিজেই এই সংখ্যার উপর নির্ভরশীল, এটা tunable hyperparameter না।

LIP label indices (mode='P' palette image থেকে np.array করলে সরাসরি এই
index পাওয়া যায়):
    0=Background, 1=Hat, 2=Hair, 3=Glove, 4=Sunglasses, 5=Upper-clothes,
    6=Dress, 7=Coat, 8=Socks, 9=Pants, 10=Jumpsuits, 11=Scarf, 12=Skirt,
    13=Face, 14=Left-arm, 15=Right-arm, 16=Left-leg, 17=Right-leg,
    18=Left-shoe, 19=Right-shoe
"""

import numpy as np

LABEL_GROUPS = {
    "head":          [1, 2, 4, 13],                  # Hat, Hair, Sunglasses, Face
    "upper_clothes": [3, 5, 6, 7, 10, 11, 14, 15],    # Glove, Upper-clothes, Dress, Coat, Jumpsuits, Scarf, L-arm, R-arm
    "lower_clothes": [9, 12, 16, 17],                 # Pants, Skirt, L-leg, R-leg
    "shoes":         [8, 18, 19],                     # Socks, L-shoe, R-shoe
}


def generate_group_masks(lip_mask_array: np.ndarray) -> dict:
    """
    Args:
        lip_mask_array: numpy array (H, W), values 0-19 (LIP label indices).
    Returns:
        dict {group_name: (H, W) uint8 array, values 0 or 255}
        keys: 'head', 'upper_clothes', 'lower_clothes', 'shoes', 'foreground'
    """
    masks = {}
    for group_name, label_list in LABEL_GROUPS.items():
        binary_mask = np.isin(lip_mask_array, label_list).astype(np.uint8) * 255
        masks[group_name] = binary_mask


    masks["foreground"] = (lip_mask_array != 0).astype(np.uint8) * 255

    return masks
