#!/bin/bash
cd "/Users/kathideckenbach/Documents/Machine Learning Master/Year 2/Master Thesis/laplace_approx_SAM"
export PYTHONPATH=$PWD
echo "!!Start training!!"
/Users/kathideckenbach/anaconda3/envs/thesis/bin/python train.py \
    --model ResNet18 \
    --dataset CIFAR10 \
    --batch_size 128 \
    --val_split 0.1 \
    --seed 1 \

echo "!!Training done!!"