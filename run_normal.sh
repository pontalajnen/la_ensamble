#!/bin/bash
python train.py \
    --model resnet20 \
    --dataset cifar10 \
    --val_split 0.1 \
    --packed \
    --epochs 5 \
    "$@"

    # --ensemble \