#!/bin/bash
export PYTHONWARNINGS="ignore"
n_models=1
max_epochs=50  # Increased epochs
lr=3e-4  # Reduced learning rate
weight_decay=0.0
batch_size=10

# Trivial LCNN model
python train_lcnn.py \
        --name "D2_W4x4_lcnn_s" \
        --logdir "../logs/wilson/" \
        --num_models $n_models \
        --max_epochs $max_epochs \
        --lr $lr \
        --weight_decay $weight_decay \
        --batch_size $batch_size \
        --dims 8 8 \
        --conv_channels 2 2 2 2 \
        --conv_kernel_size 2 2 3 3 \
        --conv_dilation 1 1 1 1 \
        --out_mode "trW_4x4" \
        --train_path "../datasets/D2_8/train.hdf5" \
        --val_path "../datasets/D2_8/val.hdf5" \
        --test_path "../datasets/D2_8/test.hdf5" \
        --gpus 1

# Medium LCNN model
python train_lcnn.py \
        --name "D2_W4x4_lcnn_m" \
        --logdir "../logs/wilson/" \
        --num_models $n_models \
        --max_epochs $max_epochs \
        --lr $lr \
        --weight_decay $weight_decay \
        --batch_size $batch_size \
        --dims 8 8 \
        --conv_channels 4 4 6 6 \
        --conv_kernel_size 3 3 4 4 \
        --conv_dilation 1 1 1 1 \
        --out_mode "trW_4x4" \
        --train_path "../datasets/D2_8/train.hdf5" \
        --val_path "../datasets/D2_8/val.hdf5" \
        --test_path "../datasets/D2_8/test.hdf5" \
        --gpus 1


# Large LCNN model

python train_lcnn.py \
        --name "D2_W4x4_lcnn_l" \
        --logdir "../logs/wilson/" \
        --num_models $n_models \
        --max_epochs $max_epochs \
        --lr $lr \
        --weight_decay $weight_decay \
        --batch_size $batch_size \
        --dims 8 8 \
        --conv_channels 8 8 8 8  \
        --conv_kernel_size 4 4 4 4 \
        --conv_dilation 1 1 1 1 \
        --out_mode "trW_4x4" \
        --train_path "../datasets/D2_8/train.hdf5" \
        --val_path "../datasets/D2_8/val.hdf5" \
        --test_path "../datasets/D2_8/test.hdf5" \
        --gpus 1

# Max LCNN model

python train_lcnn.py \
        --name "D2_W4x4_lcnn_max" \
        --logdir "../logs/wilson/" \
        --num_models $n_models \
        --max_epochs $max_epochs \
        --lr $lr \
        --weight_decay $weight_decay \
        --batch_size $batch_size \
        --dims 8 8 \
        --conv_channels 8 8 8 8 8 8 \
        --conv_kernel_size 4 4 4 4 4 4\
        --conv_dilation 1 1 1 1 1 1 \
        --out_mode "trW_4x4" \
        --train_path "../datasets/D2_8/train.hdf5" \
        --val_path "../datasets/D2_8/val.hdf5" \
        --test_path "../datasets/D2_8/test.hdf5" \
        --gpus 1        
