#!/bin/bash

train_flag=0

data_path="/home/szj/data/h5"
checkpoint_path="/home/szj/HPNet/log/pretrained_models/abc_normal.tar"
log_dir="./log/tomoki"
max_epoch=300
val_skip=1

if [ $train_flag -eq 0 ]; then
    python xtrain.py \
        --data_path="${data_path}" \
        --checkpoint_path="${checkpoint_path}" \
        --log_dir="${log_dir}" \
        --val_skip="${val_skip}" \
        --max_epoch="${max_epoch}"
else
    checkpoint_path="/home/szj/HPNet/log/tomoki/checkpoint_eval299.tar"
    python xtrain.py \
        --data_path="${data_path}" \
        --checkpoint_path="${checkpoint_path}" \
        --log_dir="${log_dir}" \
        --val_skip="${val_skip}" \
        --eval
fi
