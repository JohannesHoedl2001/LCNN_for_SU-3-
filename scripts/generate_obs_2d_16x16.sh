export PYTHONWARNINGS="ignore"

# path to datasets
base_path="../datasets16x16"

# generate observables (labels) for all 2D datasets
python generate_observables.py \
        --paths $base_path"/D2_16/train.hdf5" \
        $base_path"/D2_16/val.hdf5" \
        $base_path"/D2_16/test.hdf5" \
        $base_path"/D2_32/test.hdf5" \
        $base_path"/D2_64/test.hdf5" \
        --loops "1x1" "1x2" "2x2" "3x3" "4x4" \
        --loop_axes 0 1 \
        --polyakov