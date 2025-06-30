#!/bin/bash
export PYTHONWARNINGS="ignore"
n_models=1
max_epochs=50
lr=3e-2
weight_decay=0.0
batch_size=50
        
# Baseline model

python train.py \
        --name "D2_W4x4_baseline_S1_relu_16x16" \
        --logdir "../logs/wilson16x16/" \
        --num_models $n_models \
        --max_epochs $max_epochs \
        --lr $lr \
        --weight_decay $weight_decay \
        --batch_size $batch_size \
        --dims 16 16 \
        --conv_channels 4 4 6 6\
        --conv_kernel_size 2 2 1 1\
        --linear_sizes 4 \
        --activation "relu" \
        --out_mode "trW_4x4" \
        --train_path "../datasets16x16/D2_16/train.hdf5" \
        --val_path "../datasets16x16/D2_16/val.hdf5" \
        --test_path "../datasets16x16/D2_16/test.hdf5" \
        --gpus 1 \