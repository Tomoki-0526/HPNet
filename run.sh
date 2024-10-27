#!/bin/bash

train_flag=0

data_path="/home/szj/HPNet/data/synthetic/h5"
checkpoint_path="/home/szj/HPNet/log/pretrained_models/abc_normal.tar"
# checkpoint_path="/home/szj/HPNet/log/tomoki/checkpoint_eval299.tar"
log_dir="./log/tomoki"
max_epoch=300
val_skip=1

if [ $train_flag -eq 0 ]; then
    log_file="/home/szj/HPNet/log/train_$(date +%Y-%m-%d_%H:%M:%S).log" && touch "$log_file"
    nohup python xtrain.py \
        --data_path="${data_path}" \
        --checkpoint_path="${checkpoint_path}" \
        --log_dir="${log_dir}" \
        --val_skip="${val_skip}" \
        --max_epoch="${max_epoch}" \
        > "${log_file}" 2>&1 &
else
    log_file="/home/szj/HPNet/log/test_$(date +%Y-%m-%d_%H:%M:%S).log" && touch "$log_file"
    python xtrain.py \
        --data_path="${data_path}" \
        --checkpoint_path="${checkpoint_path}" \
        --log_dir="${log_dir}" \
        --val_skip="${val_skip}" \
        --eval
        > "${log_file}" 2>&1 &
fi
