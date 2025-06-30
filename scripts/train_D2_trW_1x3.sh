#!/bin/bash
export PYTHONWARNINGS="ignore"
n_models=5
max_epochs=50
lr=1e-3
weight_decay=0.0
batch_size=25

# Trivial LCNN model
python train_lcnn.py \
        --name "D2_W1x3_lcnn_s" \
        --logdir "../logs/wilson/" \
        --num_models $n_models \
        --max_epochs $max_epochs \
        --lr $lr \
        --weight_decay $weight_decay \
        --batch_size $batch_size \
        --dims 8 8 \
        --conv_channels 2 2 \
        --conv_kernel_size 2 1 \
        --conv_dilation 1 1 \
        --out_mode "trW_1x3" \
        --train_path "../datasets/D2_8/train.hdf5" \
        --val_path "../datasets/D2_8/val.hdf5" \
        --test_path "../datasets/D2_8/test.hdf5" \
        --gpus 1 \
