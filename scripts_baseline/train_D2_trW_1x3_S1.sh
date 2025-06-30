#!/bin/bash
export PYTHONWARNINGS="ignore"
n_models=5
max_epochs=50
lr=3e-2
weight_decay=0.0
batch_size=50

# Baseline model     
   
python train.py \
        --name "D2_W1x3_baseline_S1_relu" \
        --logdir "../logs/wilson/" \
        --num_models $n_models \
        --max_epochs $max_epochs \
        --lr $lr \
        --weight_decay $weight_decay \
        --batch_size $batch_size \
        --dims 8 8 \
        --conv_channels 4 4 \
        --conv_kernel_size 2 2 \
        --linear_sizes 4 \
        --activation "relu" \
        --out_mode "trW_1x3" \
        --train_path "../datasets/D2_8/train.hdf5" \
        --val_path "../datasets/D2_8/val.hdf5" \
        --test_path "../datasets/D2_8/test.hdf5" \
        --gpus 1 \