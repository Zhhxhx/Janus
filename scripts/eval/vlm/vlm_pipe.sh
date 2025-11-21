#!/bin/bash

# --- 1. 定义预设的参数组合 ---
# 使用并行数组定义每个实验的参数。


THRESHOLDS=()
GSIZES=()
ATTN_BACKENDS=()
MLP_BACKENDS=()

# --- 在这里添加您的参数组合 ---

# Flash Attention
# THRESHOLDS+=("4e-3")
# GSIZES+=("16")
# ATTN_BACKENDS+=("flash")
# MLP_BACKENDS+=("default")

# Naive Attention
# THRESHOLDS+=("4e-3")
# GSIZES+=("16")
# ATTN_BACKENDS+=("naive")
# MLP_BACKENDS+=("default")

# ARGUS
# THRESHOLDS+=("4e-3")
# GSIZES+=("16")
# ATTN_BACKENDS+=("flash")
# MLP_BACKENDS+=("argus")

# S-DMA
# THRESHOLDS+=("4e-3")
# GSIZES+=("16")
# # ATTN_BACKENDS+=("naive_sdma")
# ATTN_BACKENDS+=("flash")
# MLP_BACKENDS+=("default")

# FIGNA
THRESHOLDS+=("8e-4")
GSIZES+=("16")
ATTN_BACKENDS+=("flash")
MLP_BACKENDS+=("figna")

# AxCore
# THRESHOLDS+=("8e-4")
# GSIZES+=("16")
# ATTN_BACKENDS+=("flash")
# MLP_BACKENDS+=("axcore")

# --- 2. 脚本执行设置 ---
MAIN_OUTPUT_DIR="outputs/vlm/pipe_runs"
MAIN_OUTPUT_DIR=$PWD/$MAIN_OUTPUT_DIR
timestamp=$(date +"%Y%m%d_%H%M%S")
MAIN_OUTPUT_DIR=${MAIN_OUTPUT_DIR}_${timestamp}
echo "Main Output Directory: $MAIN_OUTPUT_DIR"
mkdir -p "$MAIN_OUTPUT_DIR"

MODEL_PATH="./models/JanusFlow-1.3B"

total_start_time=$SECONDS

# --- 3. 按顺序执行所有组合 ---
num_runs=${#THRESHOLDS[@]}

for (( i=0; i<num_runs; i++ )); do
    run_start_time=$SECONDS
    run_id=$((i + 1))
    
    # 为当前参数组合定义一个唯一的输出目录
    RUN_OUTPUT_DIR="$MAIN_OUTPUT_DIR/run_${run_id}_${ATTN_BACKENDS[$i]}_thresh_${THRESHOLDS[$i]}"
    
    echo "======================================================================"
    echo "Starting VLM Pipe Run $run_id / $num_runs"
    echo "Outputting to: $RUN_OUTPUT_DIR"
    echo "======================================================================"

    # 调用基础脚本，传递所有参数
    # 注意参数的顺序和数量必须与 vlm_base.sh 匹配
    bash ./scripts/eval/vlm/vlm_base.sh \
        "$RUN_OUTPUT_DIR" \
        --model-path "$MODEL_PATH" \
        --threshold "${THRESHOLDS[$i]}" \
        --sparse_gsize "${GSIZES[$i]}" \
        --attn_backend "${ATTN_BACKENDS[$i]}" \
        --mlp_backend ${MLP_BACKENDS[$i]}

    if [ $? -ne 0 ]; then
        echo "ERROR: vlm_base.sh failed for Run $run_id"
    fi

    run_end_time=$SECONDS
    duration=$((run_end_time - run_start_time))
    echo "Run $run_id finished in $((duration / 60)) minutes and $((duration % 60)) seconds."
done

total_end_time=$SECONDS
total_duration=$((total_end_time - total_start_time))
echo "======================================================================"
echo "All VLM pipe runs completed in $((total_duration / 60)) minutes."
echo "All results are stored in subdirectories under: $MAIN_OUTPUT_DIR"
echo "======================================================================"