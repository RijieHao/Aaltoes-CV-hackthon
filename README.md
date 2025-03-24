# Aaltoes-CV-Hackthon: Surf in Wild Rift

![Profile](pics/profile.png "Best profile pic award")

This repository contains a PyTorch-based pipeline for image segmentation (inpainting) tasks on a Kaggle competition dataset. It demonstrates how to train and evaluate popular segmentation models, such as **U-Net**, **ResUNet**, **Attention U-Net**, **SegFormer**, and **Mask2Former**. The code is designed to handle both CPU and GPU environments seamlessly.

## 📚 Table of Contents
1. [✨ Features](#features)  
2. [💻 Requirements](#requirements)  
3. [🗂️ Dataset](#data-preparation)  
4. [🚀 Usage](#usage)  
   - [📈 Training](#training)  
   - [🔍 Inference](#inference)  
5. [⚙️ Arguments and Options](#arguments-and-options)  
6. [📁 Project Structure](#project-structure)  
7. [🔗 Related resources](#resources)  

---

## Features

- **Multiple models in one framework**: Easily switch between U-Net, ResUNet, AttentionUNet, SegFormer, and Mask2Former.  
- **Flexible device setup**: Runs on GPU if available; falls back to CPU automatically otherwise.  
- **Configurable hyperparameters**: Adjust learning rate, batch size, epochs, threshold, etc.  
- **Run-Length Encoding (RLE)**: Automatically encodes predicted masks in RLE format for Kaggle submissions.  
- **Validation**: Splits the training dataset into train/val sets for monitoring performance via Dice coefficient.  
- **Easy inference**: Generate submission files for Kaggle evaluation.  

---

## Requirements

- [PyTorch](https://pytorch.org/) (with GPU support if you want to train on CUDA)
- [torchvision](https://pytorch.org/vision/stable/)
- [OpenCV](https://opencv.org/)
- [scikit-learn](https://scikit-learn.org/)
- [pandas](https://pandas.pydata.org/)
- [tqdm](https://github.com/tqdm/tqdm)

You can install the required Python packages via:

```bash
pip install -r requirements.txt
```

---

## Dataset

You can download dataset via
```bash
kaggle competitions download -c aaltoes-2025-computer-vision-v-1
```
Please place your data in Datasets folder.

### Overview
This competition provides a dataset of **46836 images**, each with its original version and corresponding binary mask indicating inpainted regions. All images are **256×256 pixels** in size.

### Files

- **train/** - Directory containing training data:
  - **images/** - Manipulated versions of the images
  - **masks/** - Binary masks indicating inpainted regions
  - **originals/** - Original versions of the images
- **test/** - Directory containing test data:
  - **images/** - Manipulated versions of the images
- **sample_submission.csv** - Example submission file in the correct format

### Data Format

#### Images
- **Format:** PNG
- **Resolution:** 256×256 pixels
- **Color space:** RGB (3 channels)

#### Masks
- **Format:** PNG
- **Resolution:** 256×256 pixels
- **Values:**
  - **0:** Original (non-inpainted) regions
  - **1:** Inpainted (manipulated) regions

---

## Usage
### Training
To train a model, run the main script with appropriate arguments. You can also you the run.sh to train or inference.

```bash
python main.py \
    --model "Mask2Fomer" \
    --batch_size 16 \
    --epochs 30 \
    --lr 1e-4 \
    --num_workers 4 \
    --train_image_dir "../Dataset/train/train/images" \
    --train_mask_dir "../Dataset/train/train/masks" \
    --test_image_dir "../Dataset/test/test/images" \
    --threshold 0.5
```

### Inference
For inference you need add model name and the path to the pretrained model.

```bash
python main.py \
    --model "Mask2Former" \
    --test_image_dir "../Dataset/test/test/images" \
    --load_pretrain "/path/to/your/best_model.pth" \
    --threshold 0.5
```
---

## License

Shield: [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project is licensed under a [MIT License](LICENSE).

---

## Related resources
We acknowledge all the open-source contributors for the following projects to make this work possible:

- [Mask2Former](https://github.com/facebookresearch/Mask2Former)
- [SegFormer](https://github.com/facebookresearch/Mask2Former)

- [IML-ViT](https://github.com/SunnyHaze/IML-ViT)
- [Huggingface 🤗](https://huggingface.co/)