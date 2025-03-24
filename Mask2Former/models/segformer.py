import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation

class SegFormerWrapper(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )

    def forward(self, x):
        h, w = x.shape[2:]
        outputs = self.model(pixel_values=x, return_dict=True)
        logits = outputs.logits  # [B, 1, H/4, W/4] or similar
        logits = nn.functional.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        return torch.sigmoid(logits)

# DiceLoss and dice_coefficient reused from unet.py
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice

def dice_coefficient(pred, target, smooth=1.0):
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    intersection = (pred_flat * target_flat).sum().item()
    return (2.0 * intersection + smooth) / (pred_flat.sum().item() + target_flat.sum().item() + smooth)

