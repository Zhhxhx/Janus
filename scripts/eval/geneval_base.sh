#!/bin/bash

# --- Arguments ---
# 1: OUTPUT_PATH
# 2: THRESHOLD
# 3: SPARSE_GSIZE
# 4: ATTN_BACKEND
# 5: VAE_VIT_SPARSE
# 6: SELF_ATTN_SPARSE
# 7: USE_CUSTOM_MLP
# 8: MLP_USE_SIMILARITY
# 9: USE_QUANTIZED_MLP_UND_W
# 10: USE_QUANTIZED_MLP_GEN_W
# 11: MLP_BACKEND

# Clean up function
cleanup() {
    echo -e "\n\nCtrl+C detected. Terminating all child processes..."
    pkill -P $$
    echo "All child processes terminated."
    exit 1
}
trap cleanup SIGINT SIGTERM

set -x

# --- 1. Parse Arguments ---
output_path=${1}
threshold=${2}
sparse_gsize=${3}
attn_backend=${4}
mlp_backend=${5}
GPU_START=${6:-1}
NUM_GPUS=${7:-4}

# --- 2. Static Configuration ---
# GPU_START=1
# NUM_GPUS=4
GPUS_TO_USE=(0 1 2 3 4 5 6 7)
export PYTHONPATH=.
model_path="./models/JanusFlow-1.3B"
reorder_method="None"
save_dir="maps/gedit_maps" # Can also be made a parameter if needed
metadata_file=./eval/geneval/prompts/evaluation_metadata_long.jsonl

# --- 3. Execution Block ---
mkdir -p "$output_path"
{
    date '+%Y-%m-%d %H:%M:%S'
    echo "Starting GenEval run for: $output_path"
    echo "Parameters:"
    echo "  - threshold: $threshold"
    echo "  - sparse_gsize: $sparse_gsize"
    echo "  - attn_backend: $attn_backend"
    echo "  - mlp_backend: $mlp_backend"

    # Launch processes in parallel for each GPU/chunk.
    for i in $(seq 0 $(($NUM_GPUS - 1))); do
        gpu_id=${GPUS_TO_USE[$i]}
        CUDA_VISIBLE_DEVICES=$gpu_id \
        python ./eval/geneval/evaluation/run_geneval_my.py \
            --group_id $i --group_num $NUM_GPUS --max_mem_per_gpu 80GiB --dtype bfloat16 \
            --model-path "$model_path" \
            --output_dir "$output_path/images" \
            --metadata_file "$metadata_file" \
						--batch_size 1 \
            --resolution 512 \
            --threshold "$threshold" --attn_backend "$attn_backend" \
            --sparse_gsize "$sparse_gsize" \
            --mlp_backend "$mlp_backend" &
    done

    # Wait for all background processes to finish.
    wait
    echo "All image generation processes finished."

    # Calculate scores
    torchrun \
        --nnodes=1 \
        --node_rank=0 \
        --nproc_per_node=1 \
        --master_addr=127.0.0.1 \
        --master_port=12345 \
        ./eval/geneval/evaluation/evaluate_images_mp.py \
        "$output_path/images" \
        --outfile "$output_path/results.jsonl" \
        --model-path ./eval/geneval/model

    # Summarize scores
    python ./eval/geneval/evaluation/summary_scores.py "$output_path/results.jsonl"

		python ./profile/geneval_summarize_metrics.py --root_dir $output_path/images --output_file $output_path/summary_metrics.json
    
    echo "Evaluation and summary finished."
    date '+%Y-%m-%d %H:%M:%S'

} 2>&1 | tee "$output_path/geneval_run.log"
