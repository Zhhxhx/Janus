#!/bin/bash

# --- Arguments ---
# 1:  OUTPUT_PATH
# 2:  MODEL_PATH
# 3:  THRESHOLD
# 4:  SPARSE_GSIZE
# 5:  ATTN_BACKEND
# 6:  VAE_VIT_SPARSE
# 7:  SELF_ATTN_SPARSE
# 8:  USE_CUSTOM_MLP
# 9:  MLP_USE_SIMILARITY
# 10: USE_QUANTIZED_MLP_W
# 11: REORDER_METHOD
# ... (add more as needed)

set -x

# --- 1. Parse Arguments ---
# All arguments passed to this script will be forwarded to the evaluate.sh script.
# The first argument is the output path, the rest are evaluation parameters.
LOG_PATH=$1
shift # Remove the first argument (LOG_PATH), the rest are now the parameters.
ARGS=("$@")

# --- 2. Static Configuration ---
# Define the list of datasets to evaluate on.
# DATASETS=("mme" "mmbench-dev-en" "mmvet" "mmmu-val" "mathvista-testmini" "mmvp")
# DATASETS=("mmbench-test-en")
DATASETS=("mmvet")
DATASETS_STR="${DATASETS[*]}"
export DATASETS_STR

# Environment settings
export GPUS=1
export ARNOLD_WORKER_NUM=1
export ARNOLD_ID=0
export ARNOLD_WORKER_0_HOST=localhost
export MASTER_PORT=10042 # Set a default port
export OPENAI_API_KEY=$openai_api_key # Assumes this is set in the parent environment

# --- 3. Execution Block ---
if [ ! -d "$LOG_PATH" ]; then
    mkdir -p "$LOG_PATH"
fi
echo "Starting VLM evaluation run. Results will be saved in: $LOG_PATH"
echo "Parameters passed to evaluate.sh: ${ARGS[@]}" > "$LOG_PATH/run_params.log"

# Loop through each dataset and call the evaluate.sh script
for DATASET in "${DATASETS[@]}"; do
    echo "--------------------------------------------------"
    echo "Evaluating on dataset: $DATASET"
    echo "--------------------------------------------------"
    
    # Call the core evaluation script, passing the dataset name, a dedicated output dir,
    # and all other parameters received by this script.
    bash ./scripts/eval/vlm/evaluate.sh \
        "$DATASET" \
        --out-dir "$LOG_PATH/$DATASET" \
        "${ARGS[@]}"
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Evaluation failed for dataset: $DATASET"
    fi
done

echo "======================================================================"
echo "VLM evaluation run finished for: $LOG_PATH"
echo "======================================================================"