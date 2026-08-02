"""
Combines MaskLoss + DeepSupervision to compute the full 4-part
(head, upper_clothes, lower_clothes, shoes) attention supervision loss,
matching the Deepreid reference notebook's training loop exactly.

Channel order convention (matches IAT's SpatialAttn(number=4) output, and
Mission 1's label_grouping.py mask ordering):
    channel 0 = head, 1 = upper_clothes, 2 = lower_clothes, 3 = shoes
"""

from .mask_loss import MaskLoss, DeepSupervision


def compute_attention_supervision_loss(criterion_mask, attn_layer2, attn_layer3, gt_masks):
    """
    Args:
        criterion_mask: MaskLoss instance
        attn_layer2: (batch, 4, h2, w2) attention map from IAT after layer2
        attn_layer3: (batch, 4, h3, w3) attention map from IAT after layer3
        gt_masks: (batch, 4, H, W) ground-truth masks, channel order matching
            [head, upper_clothes, lower_clothes, shoes]

    Returns:
        scalar tensor: the averaged 4-part mask supervision loss
    """
    total = 0.
    for i in range(4):
        a2_part = attn_layer2[:, i:i + 1]
        a3_part = attn_layer3[:, i:i + 1]
        gt_part = gt_masks[:, i:i + 1]
        part_loss = DeepSupervision(criterion_mask, [a2_part, a3_part], gt_part)
        total += part_loss
    return total / 4.0
