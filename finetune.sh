#!/bin/bash
LOG=./checkpoints/finetune`date +%Y-%m-%d-%H-%M-%S`.log
export PYTHONUNBUFFERED="True"
python finetune.py \
            --path /data/user/cephfs/arlencai/dataset/hymenoptera_data \
            --gpu-devices 0 \
            --model-zoo /data/user/cephfs/arlencai/pytorch_model \
            2>&1 | tee -i $LOG