#!/bin/bash

train_flag=1
val_skip=1

if [ $train_flag -eq 0 ]; then
    log_dir="./log/finetune"
    max_epoch=1500
    data_path="/home/szj/HPNet/data/synthetic/h5"
    checkpoint_path="/home/szj/HPNet/log/pretrained_models/abc_normal.tar"
    log_file="/home/szj/HPNet/log/finetune_$(date +%Y-%m-%d_%H:%M:%S).log" && touch "$log_file"
    CUDA_VISIBLE_DEVICES=4,5,6,7 nohup \
        python xtrain.py \
        --data_path="${data_path}" \
        --checkpoint_path="${checkpoint_path}" \
        --log_dir="${log_dir}" \
        --val_skip="${val_skip}" \
        --max_epoch="${max_epoch}" \
        > "${log_file}" 2>&1 &
elif [ $train_flag -eq 1 ]; then
    log_dir="./log/predict"
    max_epoch=300
    data_path="/home/szj/HPNet/data/predict/h5"
    checkpoint_path="/home/szj/HPNet/log/pretrained_models/abc_normal.tar"
    log_file="/home/szj/HPNet/log/predict_$(date +%Y-%m-%d_%H:%M:%S).log" && touch "$log_file"
    CUDA_VISIBLE_DEVICES=7 nohup \
        python xtrain.py \
        --data_path="${data_path}" \
        --checkpoint_path="${checkpoint_path}" \
        --log_dir="${log_dir}" \
        --val_skip="${val_skip}" \
        --max_epoch="${max_epoch}" \
        --eval \
        > "${log_file}" 2>&1 &
else
    log_dir="./log/train"
    max_epoch=300
    data_path="/home/szj/HPNet/data/abc/h5"
    log_file="/home/szj/HPNet/log/train_$(date +%Y-%m-%d_%H:%M:%S).log" && touch "$log_file"
    nohup \
        python train.py \
        --data_path="${data_path}" \
        --log_dir="${log_dir}" \
        > "${log_file}" 2>&1 &
fi