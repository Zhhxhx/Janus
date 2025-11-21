#!/bin/bash

# echo "Script will start in 1.25 hour. Current time: $(date)"
# sleep 4500
# echo "Starting script now. Current time: $(date)"

# --- 1. 定义预设的参数组合 ---
# 使用并行数组定义每个实验的参数。
# 每个数组的相同索引共同构成一个完整的参数组合。

THRESHOLDS=()
GSIZES=()
ATTN_BACKENDS=()
MLP_BACKENDS=()

# --- 在这里添加您的参数组合 ---

# Flash Attention
THRESHOLDS+=("4e-3")
GSIZES+=("16")
ATTN_BACKENDS+=("flash")
MLP_BACKENDS+=("default")

# # 组合 1: ARGUS
# THRESHOLDS+=("4e-3")
# GSIZES+=("16")
# ATTN_BACKENDS+=("naive_sparse_quant_cfg")
# MLP_BACKENDS+=("argus")

# # 组合 2: FLIGHTVGM
# THRESHOLDS+=("8e-4")
# GSIZES+=("16")
# ATTN_BACKENDS+=("naive")
# MLP_BACKENDS+=("flightvgm")

# # 组合 3: FIGNA
# THRESHOLDS+=("8e-4")
# GSIZES+=("16")
# ATTN_BACKENDS+=("naive")
# MLP_BACKENDS+=("figna")

# # 组合 4: AxCore
# THRESHOLDS+=("8e-4")
# GSIZES+=("16")
# ATTN_BACKENDS+=("naive")
# MLP_BACKENDS+=("axcore")

# 组合 5: S-DMA
# THRESHOLDS+=("4e-3")
# GSIZES+=("16")
# ATTN_BACKENDS+=("naive_sdma")
# MLP_BACKENDS+=("default")

# --- 2. 脚本执行设置 ---
MAIN_OUTPUT_DIR="outputs/geneval/pipe_runs"
MAIN_OUTPUT_DIR=$PWD/$MAIN_OUTPUT_DIR
timestamp=$(date +"%Y%m%d_%H%M%S")
MAIN_OUTPUT_DIR=${MAIN_OUTPUT_DIR}_${timestamp}
echo "Main Output Directory: $MAIN_OUTPUT_DIR"
mkdir -p "$MAIN_OUTPUT_DIR"

total_start_time=$SECONDS

# --- 3. 按顺序执行所有组合 ---
num_runs=${#THRESHOLDS[@]}

for (( i=0; i<num_runs; i++ )); do
    # 从数组中获取当前组合的参数
    threshold="${THRESHOLDS[$i]}"
    gsize="${GSIZES[$i]}"
    attn_backend="${ATTN_BACKENDS[$i]}"
    mlp_backend="${MLP_BACKENDS[$i]}"

    run_start_time=$SECONDS
    run_id=$((i + 1))
        
    echo "======================================================================"
    echo "Starting Run $run_id / $num_runs"
    echo "Parameters: threshold=$threshold, gsize=$gsize, attn_backend=$attn_backend, mlp_backend=$mlp_backend"
    echo "======================================================================"

    # 为当前参数组合定义一个唯一的输出目录
    RUN_OUTPUT_DIR="$MAIN_OUTPUT_DIR/run_${run_id}_${attn_backend}_${mlp_backend}"
		DEVICE_BASE=7
    N_GPU=1

    # 调用基础脚本执行实验
    bash ./scripts/eval/geneval_base.sh \
        "$RUN_OUTPUT_DIR" \
        "$threshold" "$gsize" "$attn_backend" \
        "$mlp_backend" "$DEVICE_BASE" "$N_GPU"

    if [ $? -ne 0 ]; then
        echo "ERROR: geneval_base.sh failed for Run $run_id"
        continue # 跳过当前失败的组合
    fi

    run_end_time=$SECONDS
    duration=$((run_end_time - run_start_time))
    echo "Run $run_id finished in $((duration / 60)) minutes and $((duration % 60)) seconds."
done

total_end_time=$SECONDS
total_duration=$((total_end_time - total_start_time))
echo "======================================================================"
echo "All preset runs completed in $((total_duration / 60)) minutes."
echo "All results are stored in subdirectories under: $MAIN_OUTPUT_DIR"
echo "======================================================================"
