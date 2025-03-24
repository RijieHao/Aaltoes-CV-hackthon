import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import argparse

from models import UNet, ResUNet, AttentionUNet, SegFormerWrapper, DiceLoss, dice_coefficient, Mask2FormerWrapper

# ------------------ Argument Parser ------------------
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--model', type=str, default="UNet")
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--threshold', type=float, default=0.5)
parser.add_argument('--train_image_dir', type=str, default="../Dataset/train/train/images")
parser.add_argument('--train_mask_dir', type=str, default="../Dataset/train/train/masks")
parser.add_argument('--test_image_dir', type=str, default="../Dataset/test/test/images")
parser.add_argument('--load_pretrain', type=str, default=None , help="Skip training and only do inference using pretrained model, path to the model")
args = parser.parse_args()

# ------------------ Dataset ------------------
class InpaintDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0

        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.float32)

        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)
        return image, mask

class InpaintTestDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        image_id = os.path.basename(img_path).split('.')[0]
        return image, image_id

# ------------------ Run Length Encoding ------------------
def mask2rle(img):
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# ------------------ Main Training + Inference ------------------
image_paths = sorted(glob.glob(os.path.join(args.train_image_dir, "*.png")))
mask_paths = sorted(glob.glob(os.path.join(args.train_mask_dir, "*.png")))
train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = train_test_split(image_paths, mask_paths, test_size=0.2, random_state=42)

train_dataset = InpaintDataset(train_img_paths, train_mask_paths)
val_dataset = InpaintDataset(val_img_paths, val_mask_paths)
test_image_paths = sorted(glob.glob(os.path.join(args.test_image_dir, "*.png")))
test_dataset = InpaintTestDataset(test_image_paths)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=arg.num_workers)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


if args.model == "UNet":
    model = UNet().cuda()
elif args.model == "ResUNet":
    model = ResUNet().cuda()
elif args.model == "AttentionUNet":
    model = AttentionUNet().cuda()
elif args.model == "SegFormer":
    model = SegFormerWrapper().cuda()
elif args.model == "Mask2Former":
    args.model = "mask2former-base"
    model = Mask2FormerWrapper(
    model_name="facebook/mask2former-swin-base-ade-semantic"# You can use other models like "facebook/mask2former-small-ade-semantic"
).cuda()
else:
    raise ValueError(f"Unsupported model: {args.model}")


best_dice = 0.0

if not args.load_pretrain:
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = lambda inputs, targets: 0.5 * nn.BCELoss()(inputs, targets) + 0.5 * DiceLoss()(inputs, targets)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            images, masks = images.cuda(), masks.cuda()
            outputs = model(images)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"[Train] Loss: {train_loss / len(train_loader):.4f}")

        model.eval()
        val_loss, dice_score = 0.0, 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.cuda(), masks.cuda()
                outputs = model(images)
                val_loss += criterion(outputs, masks).item()
                preds = (outputs > 0.5).float()
                for pred, mask in zip(preds, masks):
                    dice_score += dice_coefficient(pred, mask)

        avg_dice = dice_score / len(val_loader.dataset)
        print(f"[Val] Dice: {avg_dice:.4f}")
        scheduler.step(avg_dice)

        if avg_dice > best_dice:
            best_dice = avg_dice
            score_tag = f"{int(best_dice * 1000):03d}"
            save_folder = f"results/{args.model}_bs{args.batch_size}_lr{args.lr}_ep{args.epochs}_{score_tag}"
            os.makedirs(save_folder, exist_ok=True)
            model_path = os.path.join(save_folder, "best_model.pth")
            submission_path = os.path.join(save_folder, "submission.csv")
            torch.save(model.state_dict(), model_path)
            print("✅ Best model saved.")
else:
    score_tag = "pretrained"
    model_path = args.load_pretrain
    assert os.path.exists(model_path), f"Model path {model_path} does not exist."
    save_folder = os.path.dirname(model_path)
    submission_path = os.path.join(save_folder, "submission.csv")
    print(f"🔁 Skipping training. Loading pretrained model from {model_path}...")

# ------------------ Final Prediction ------------------
model.load_state_dict(torch.load(model_path))
model.eval()
submission = []
with torch.no_grad():
    for images, image_ids in tqdm(test_loader, desc="Test Inference"):
        images = images.cuda()
        outputs = model(images)
        preds = (outputs > args.threshold).float().cpu().numpy()
        for i, image_id in enumerate(image_ids):
            mask_pred = preds[i, 0, :, :]
            rle = mask2rle(mask_pred)
            submission.append([image_id, rle])

pd.DataFrame(submission, columns=["ImageId", "EncodedPixels"]).to_csv(submission_path, index=False)
print(f"📁 submission.csv saved to {submission_path}")

