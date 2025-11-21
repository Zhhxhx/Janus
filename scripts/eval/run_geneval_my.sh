# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

# Clean up function
cleanup() {
    echo -e "\n\nCtrl+C detected. Terminating all child processes..."
    # Kill all the subprocess
    pkill -P $$
    echo "All child processes terminated."

    exit 1
}

# Set up the trap which will trigger our cleanup function
trap cleanup SIGINT SIGTERM

set -x
GPU_START=6
NUM_GPUS=2

# Set PYTHONPATH to include project root
export PYTHONPATH=.

model_path="./models/JanusFlow-1.3B"

output_path=./outputs/geneval/geneval_results
# output_path=$1
timestamp=$(date +"%Y%m%d_%H%M%S")
output_path=$output_path\_$timestamp

threshold=4e-5
sparse_gsize=10

attn_backend="naive_sparse_quant_cfg_self"
# attn_backend="flash"
vae_vit_sparse=--vae_vit_sparse
self_attn_sparse=--self_attn_sparse
use_custom_mlp=--use_custom_mlp
mlp_use_similarity=--mlp_use_similarity
use_full_head_similarity=--use_full_head_similarity
reorder_method="None"
save_dir="maps/gedit_maps"

mkdir -p $output_path
{
    date '+%Y-%m-%d %H:%M:%S'
    echo $output_path

    metadata_file=./eval/geneval/prompts/evaluation_metadata_long.jsonl
    # metadata_file=$2
    echo $metadata_file

    # Launch processes in parallel for each GPU/chunk.
    for i in $(seq 0 $(($NUM_GPUS - 1))); do
        CUDA_VISIBLE_DEVICES=$((GPU_START + i)) \
        python ./eval/geneval/evaluation/run_geneval_my.py \
            --group_id $i --group_num $NUM_GPUS --max_mem_per_gpu 80GiB --dtype bfloat16 \
            --model-path $model_path \
            --output_dir $output_path/images \
            --metadata_file $metadata_file \
						--resolution 512 \
						--threshold $threshold --attn_backend $attn_backend \
						--save_dir $save_dir \
						--sparse_gsize $sparse_gsize \
						$vae_vit_sparse $self_attn_sparse \
						$mlp_save $mlp_save_dir \
						--reorder_method $reorder_method \
						$use_custom_mlp $mlp_use_similarity $use_full_head_similarity &
    done

    # Wait for all background processes to finish.
    wait
    echo "All background processes finished."

    # calculate scorecle
    torchrun \
        --nnodes=1 \
        --node_rank=0 \
        --nproc_per_node=$NUM_GPUS \
        --master_addr=127.0.0.1 \
        --master_port=12345 \
        ./eval/geneval/evaluation/evaluate_images_mp.py \
        $output_path/images \
        --outfile $output_path/results.jsonl \
        --model-path ./eval/geneval/model

    python ./eval/geneval/evaluation/summary_scores.py $output_path/results.jsonl
    date '+%Y-%m-%d %H:%M:%S'
} 2>&1 | tee "$output_path/geneval.log"