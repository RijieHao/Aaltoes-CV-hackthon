#!/bin/bash

# Define models, learning rates, batch sizes
#declare -a models=("Unet" "ResUNet" "AttentionUNet")
declare -a models=("Mask2Former")
declare -a lrs=(1e-4)
declare -a bss=(16)
epochs=30
threshold=0.61 # 0.61 is the best threshold for Mask2Former, you can calculate the best threshold for other models by utils

for model in "${models[@]}"; do
  for lr in "${lrs[@]}"; do
    for bs in "${bss[@]}"; do
      echo "Running model=$model, lr=$lr, batch_size=$bs, epochs=$epochs"
      python train_base.py --model $model --lr $lr --batch_size $bs --epochs $epochs --threshold $threshold
    done
  done
done
