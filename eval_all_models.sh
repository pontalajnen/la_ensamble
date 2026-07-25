#!/bin/bash
set -e

MODEL=${1:-resnet20}
DATASET=${2:-cifar10}
SEED=${3:-1}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

RESULTS_DIR="$SCRIPT_DIR/experiment_results"
PATH_FILES_DIR="$SCRIPT_DIR/eval_path_files"

NO_SAM_DIR="$RESULTS_DIR/${DATASET}_${MODEL}_no_SAM"
SAM_DIR="$RESULTS_DIR/${DATASET}_${MODEL}__SAM"

# model_name mirrors create_model_name() in train.py
BASE_NAME="${MODEL}_${DATASET}_SGD"
ENSEMBLE_NAME="${BASE_NAME}_ensemble"
PACKED_NAME="${BASE_NAME}_packed"
SAM_NAME="${BASE_NAME}"
SGLD_NAME="${MODEL}_${DATASET}_SGLD"

# Find the mn= checkpoint with the lowest val_loss for a given model_name and directory
find_best_checkpoint() {
    local dir=$1
    local model_name=$2
    python3 -c "
import glob, re, sys
files = glob.glob('$dir/mn=${model_name}-*.pth')
if not files:
    sys.exit(f'ERROR: no checkpoint found for ${model_name} in $dir')
best = min(files, key=lambda f: float(re.search(r'val_loss=([0-9.]+)', f).group(1)))
print(best)
"
}

make_path_file() {
    local filename=$1
    local dir=$2
    local model_name=$3
    local checkpoint
    checkpoint=$(find_best_checkpoint "$dir" "$model_name")
    echo "  [checkpoint]: $checkpoint"
    echo "$checkpoint" > "$PATH_FILES_DIR/$filename"
}

run_eval() {
    local name=$1
    local path_file=$2
    local model_type=$3
    local save_file=$4
    shift 4
    echo ""
    echo "=========================================="
    echo "  Evaluating: $name"
    echo "=========================================="
    python evaluate.py \
        --save_file_name "$save_file" \
        --model_path_file "$path_file" \
        --model_type "$model_type" \
        --dataset "$DATASET" \
        --batch_size 128 \
        "$@"
}

# Normal (SGD)
make_path_file "run_normal.txt" "$NO_SAM_DIR" "$BASE_NAME"
run_eval "Normal (SGD)" "run_normal.txt" "$MODEL" "${MODEL}_${DATASET}_normal.json"

# Ensemble
make_path_file "run_ensemble.txt" "$NO_SAM_DIR" "$ENSEMBLE_NAME"
run_eval "Ensemble" "run_ensemble.txt" "${MODEL}_ensemble" "${MODEL}_${DATASET}_ensemble.json"

# Packed
make_path_file "run_packed.txt" "$NO_SAM_DIR" "$PACKED_NAME"
run_eval "Packed" "run_packed.txt" "${MODEL}_packed" "${MODEL}_${DATASET}_packed.json"

# SAM
make_path_file "run_sam.txt" "$SAM_DIR" "$SAM_NAME"
run_eval "SAM" "run_sam.txt" "$MODEL" "${MODEL}_${DATASET}_sam.json"

# SGLD
SGLD_SAMPLES="$NO_SAM_DIR/sgld_samples_${SGLD_NAME}_seed${SEED}.txt"
if [ ! -f "$SGLD_SAMPLES" ]; then
    echo "ERROR: SGLD samples file not found: $SGLD_SAMPLES"
    exit 1
fi
cp "$SGLD_SAMPLES" "$PATH_FILES_DIR/run_sgld.txt"
run_eval "SGLD" "run_sgld.txt" "$MODEL" "${MODEL}_${DATASET}_sgld.json" --sgld_ensemble --max_sgld_samples 5

echo ""
echo "All evaluations finished."
