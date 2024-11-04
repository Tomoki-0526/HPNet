#!/bin/bash

train_flag=0

data_path="/home/szj/HPNet/data/synthetic/h5"
log_dir="./log/tomoki"
max_epoch=300
val_skip=1

if [ $train_flag -eq 0 ]; then
    checkpoint_path="/home/szj/HPNet/log/pretrained_models/abc_normal.tar"
    log_file="/home/szj/HPNet/log/train_$(date +%Y-%m-%d_%H:%M:%S).log" && touch "$log_file"
    CUDA_VISIBLE_DEVICES=4,5,6,7 nohup \
        python xtrain.py \
        --data_path="${data_path}" \
        --checkpoint_path="${checkpoint_path}" \
        --log_dir="${log_dir}" \
        --val_skip="${val_skip}" \
        --max_epoch="${max_epoch}" \
        > "${log_file}" 2>&1 &
else
    checkpoint_path="/home/szj/HPNet/checkpoints/real_column_e1500.tar"
    log_file="/home/szj/HPNet/log/test_$(date +%Y-%m-%d_%H:%M:%S).log" && touch "$log_file"
    CUDA_VISIBLE_DEVICES=7 nohup \
        python xtrain.py \
        --data_path="${data_path}" \
        --checkpoint_path="${checkpoint_path}" \
        --log_dir="${log_dir}" \
        --val_skip="${val_skip}" \
        --max_epoch="${max_epoch}" \
        --eval \
        > "${log_file}" 2>&1 &
fi
