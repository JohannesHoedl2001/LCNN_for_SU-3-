export PYTHONWARNINGS="ignore"

# path to datasets
base_path="../datasets16x16"

# dataset parameters
n_train=1000
n_testval=100
n_sweep=100
n_warmup=10000
beta_min=5.5
beta_max=6.5
beta_steps=10

# 16x16 dataset (test)
mkdir -p $base_path"/D2_16/"

python generate_configs.py \
        --paths $base_path"/D2_16/train.hdf5" \
        $base_path"/D2_16/val.hdf5" \
        $base_path"/D2_16/test.hdf5" \
        --nums $n_train $n_testval $n_testval \
        --sweeps $n_sweep \
        --warmup $n_warmup \
        --beta_min $beta_min \
        --beta_max $beta_max \
        --beta_steps $beta_steps \
        --dims 16 16 \
        --conjugate \
        --random_seed 94578

# 32x32 dataset (test)
mkdir -p $base_path"/D2_32/"

python generate_configs.py \
        --paths $base_path"/D2_32/test.hdf5" \
        --nums $n_testval \
        --sweeps $n_sweep \
        --warmup $n_warmup \
        --beta_min $beta_min \
        --beta_max $beta_max \
        --beta_steps $beta_steps \
        --dims 32 32 \
        --conjugate \
        --random_seed 28362

# 64x64 dataset (test)
mkdir -p $base_path"/D2_64/"

python generate_configs.py \
        --paths $base_path"/D2_64/test.hdf5" \
        --nums $n_testval \
        --sweeps $n_sweep \
        --warmup $n_warmup \
        --beta_min $beta_min \
        --beta_max $beta_max \
        --beta_steps $beta_steps \
        --dims 64 64 \
        --conjugate \
        --random_seed 30834