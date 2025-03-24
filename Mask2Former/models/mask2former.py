import torch
import torch.nn as nn
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from PIL import Image
import numpy as np

class Mask2FormerWrapper(nn.Module):
    def __init__(self,  model_name="facebook/mask2former-swin-large-ade-semantic", num_classes=1):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            model_name,
            ignore_mismatched_sizes=True
        )
        self.num_classes = num_classes

    def forward(self, x):
        b, c, h, w = x.shape
        images = []
        for img in x:
            img_np = img.cpu().numpy().transpose(1, 2, 0) * 255.0
            img_pil = Image.fromarray(img_np.astype("uint8"))
            images.append(img_pil)

        inputs = self.processor(images=images, return_tensors="pt").to(x.device)

        outputs = self.model(**inputs)  # ❌ 不再使用 torch.no_grad()

        masks = outputs.masks_queries_logits  # [B, N, h', w']
        best_mask = masks[:, 0:1, :, :]       # [B, 1, h', w']
        best_mask = nn.functional.interpolate(best_mask, size=(h, w), mode="bilinear", align_corners=False)
        return torch.sigmoid(best_mask)

# Loss and metric
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
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
