#!/bin/bash
LOG=./checkpoints/finetune`date +%Y-%m-%d-%H-%M-%S`.log
export PYTHONUNBUFFERED="True"
python finetune.py \
            --path /data/user/cephfs/arlencai/dataset/hymenoptera_data \
            2>&1 | tee -i $LOG