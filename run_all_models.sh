#!/bin/bash
set -e

MODEL=${1:-resnet20}
DATASET=${2:-cifar10}
EPOCHS=${3:-200}

BASE_ARGS="--model $MODEL --dataset $DATASET --val_split 0.1 --epochs $EPOCHS"

run() {
    local name=$1; shift
    echo ""
    echo "=========================================="
    echo "  Running: $name"
    echo "=========================================="
    python train.py $BASE_ARGS "$@"
}

# run "Normal (SGD)"
# run "Ensemble"       --ensemble
run "Packed"         --packed
run "SAM"            --SAM
run "SGLD"           --base_optimizer SGLD

echo ""
echo "All models finished."
