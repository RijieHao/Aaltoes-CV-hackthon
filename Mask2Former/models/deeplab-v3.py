import torch
import torchvision
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as F
from torchvision.models.segmentation import deeplabv3_resnet101

# 加载预训练的 DeepLabV3+ 模型
model = deeplabv3_resnet101(pretrained=True)

# 如果需要微调，只需要替换分类器的输出层
model.classifier[4] = nn.Conv2d(256, 1, kernel_size=1)  # 将类别数设置为 1（生成区域 vs. 原始区域）

# 将模型移动到 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch

class CustomDataset(Dataset):
    def __init__(self, image_folder, mask_folder, transform=None):
        self.image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.png')]  # 确保是png文件
        self.mask_paths = [os.path.join(mask_folder, f) for f in os.listdir(mask_folder) if f.endswith('.png')]  # 确保是png文件
        self.transform = transform

        # 确保图像和掩膜文件数量一致
        assert len(self.image_paths) == len(self.mask_paths), "Number of images and masks do not match!"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")  # 确保图像是RGB模式
        mask = Image.open(self.mask_paths[idx]).convert("L")  # 确保掩膜是单通道（L模式）

        if self.transform:
            img = self.transform(img)
            # 对掩膜不进行 Normalize 仅转换为张量
            mask = transforms.ToTensor()(mask)

        return img, mask


# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 对图像进行标准化
])

# 路径设置
train_image_paths = "/home/cv-hacker/Dataset/train/train/images"
train_mask_paths = "/home/cv-hacker/Dataset/train/train/masks"

# 创建数据集和数据加载器
train_dataset = CustomDataset(train_image_paths, train_mask_paths, transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# 验证数据集加载
for images, masks in train_loader:
    print(images.shape, masks.shape)  # 查看输出的图像和掩膜的形状
    break

import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm  # 导入 tqdm 库

# 优化器
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

#dice损失函数
def dice_loss(pred, target, smooth=1e-6):
    # 将预测结果转换为二值
    pred = torch.sigmoid(pred)  # 对 logits 应用 sigmoid
    pred = pred.view(-1)
    target = target.view(-1)

    intersection = torch.sum(pred * target)
    dice = (2. * intersection + smooth) / (torch.sum(pred) + torch.sum(target) + smooth)
    return 1 - dice
# 训练循环
# 训练循环
def train_one_epoch(model, train_loader, optimizer, device):
    model.train()
    running_loss = 0.0
    
    # 使用 tqdm 为训练加载器添加进度条
    loop = tqdm(train_loader, desc="Training", unit="batch")
    
    for images, masks in loop:
        images = images.to(device)
        masks = masks.to(device)

        # 前向传播
        outputs = model(images)['out']
        
        # 损失函数改为 Dice Loss
        loss = dice_loss(outputs, masks)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # 更新进度条描述，显示当前批次的损失
        loop.set_postfix(loss=loss.item())

    return running_loss / len(train_loader)

    
# 训练模型
num_epochs = 1  # 可根据实际时间调整
for epoch in range(num_epochs):
    print(f"Start running epoch {epoch + 1}")
    loss = train_one_epoch(model, train_loader, optimizer, device)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")
    print(f"Finished running epoch {epoch + 1}")

import numpy as np
from sklearn.preprocessing import LabelEncoder

def rle_encode(mask):
    pixels = mask.flatten()
    pad = np.pad(pixels, (1, 0), mode='constant', constant_values=0)
    runs = np.where(pad[1:] != pad[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)





# 预测
def predict(model, image, device):
    model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)
        output = model(image)['out']
        prediction = torch.sigmoid(output).squeeze().cpu().numpy()
        return prediction

# 对预测结果进行 RLE 编码
mask = predict(model, some_image, device)
rle_mask = rle_encode(mask)
print(rle_mask)

import pandas as pd

def save_submission(rle_results, filename="submission.csv"):
    df = pd.DataFrame(rle_results, columns=["ImageId", "EncodedPixels"])
    df.to_csv(filename, index=False)

# 假设你保存了所有预测的结果
save_submission(rle_results)