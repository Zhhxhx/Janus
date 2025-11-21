# Copyright (c) 2023 OpenGVLab
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: MIT
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025-05-20.
#
# Original file was released under MIT, with the full license text
# available at https://github.com/OpenGVLab/InternVL/blob/main/LICENSE.
#
# This modified file is released under the same license.

set -x

export PYTHONPATH="$(pwd):${PYTHONPATH}"
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

# The first argument is the dataset name.
DATASET=${1}
# All arguments (including the dataset name) are saved.
# We will pass all arguments EXCEPT the first one to the python scripts.
ARGS=("$@")

# Default GPUS to 1, can be overridden by --auto flag.
# GPUS=1
# Parse for --auto to set GPUS, but don't consume other args.
for arg in "${ARGS[@]}"; do
  if [[ "$arg" == "--auto" ]]; then
    GPUS=1
    break
  fi
done
echo "GPUS: ${GPUS}"

# --- Dataset-specific execution ---
# Note: "${ARGS[@]:1}" passes all arguments except the first one (the dataset name).

if  [ ${DATASET} == "mme" ]; then
  python -m eval.vlm.eval.mme.eval "${ARGS[@]:1}"
fi

if [ ${DATASET} == "mmvet" ]; then
    python -m eval.vlm.eval.mmvet.evaluate_mmvet_my --datasets mmvet "${ARGS[@]:1}"
fi

if [[ ${DATASET} == "mmbench-dev-en" || ${DATASET} == "mmbench-dev-cn" || ${DATASET} == "mmbench-test-en" || ${DATASET} == "mmbench-test-cn" ]]; then
    DATASET_NAME=""
    if [ ${DATASET} == "mmbench-dev-en" ]; then DATASET_NAME="mmbench_dev_20230712"; fi
    if [ ${DATASET} == "mmbench-dev-cn" ]; then DATASET_NAME="mmbench_dev_cn_20231003"; fi
    if [ ${DATASET} == "mmbench-test-en" ]; then DATASET_NAME="mmbench_test_en_20231003"; fi
    if [ ${DATASET} == "mmbench-test-cn" ]; then DATASET_NAME="mmbench_test_cn_20231003"; fi
    
    torchrun \
      --nnodes=$ARNOLD_WORKER_NUM \
      --node_rank=$ARNOLD_ID \
      --master_addr=$ARNOLD_WORKER_0_HOST \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      -m eval.vlm.eval.mmbench.evaluate_mmbench_my --datasets "$DATASET_NAME" "${ARGS[@]:1}"
fi

# Add other if blocks for other datasets (mmmu, mathvista, pope, etc.) following the same pattern.
# Example for MMMU:
if [[ ${DATASET} == "mmmu-dev" || ${DATASET} == "mmmu-val" || ${DATASET} == "mmmu-test" ]]; then
    DATASET_NAME=""
    if [ ${DATASET} == "mmmu-dev" ]; then DATASET_NAME="MMMU_dev"; fi
    if [ ${DATASET} == "mmmu-val" ]; then DATASET_NAME="MMMU_validation"; fi
    if [ ${DATASET} == "mmmu-test" ]; then DATASET_NAME="MMMU_test"; fi

    torchrun \
      --nnodes=$ARNOLD_WORKER_NUM \
      --node_rank=$ARNOLD_ID \
      --master_addr=$ARNOLD_WORKER_0_HOST \
      --nproc_per_node=${GPUS} \
      --master_port=${MASTER_PORT} \
      -m eval.vlm.eval.mmmu.evaluate_mmmu --datasets "$DATASET_NAME" "${ARGS[@]:1}"
fi

# if [ ${DATASET} == "mmvet" ]; then
#     python -m eval.vlm.eval.mmvet.evaluate_mmvet --datasets mmvet "${ARGS[@]:1}"
# fi

# set -x

# export PYTHONPATH="$(pwd):${PYTHONPATH}"
# export TF_CPP_MIN_LOG_LEVEL=3
# export LAUNCHER=pytorch

# DATASET=${1}
# echo "CHECKPOINT: ${CHECKPOINT}"

# threshold=4e-5
# sparse_gsize=10

# # attn_backend="naive_sparse_quant_cfg"
# attn_backend="flash"
# vae_vit_sparse=--vae_vit_sparse
# self_attn_sparse=--self_attn_sparse
# # use_custom_mlp=--use_custom_mlp
# # mlp_use_similarity=--mlp_use_similarity
# # use_quantized_mlp_w=--quantized_mlp_w
# reorder_method="None"
# save_dir="maps/gedit_maps"

# # Save original arguments
# ARGS=("$@")

# # Parse options
# while [[ $# -gt 0 ]]; do
#   case "$1" in
#     --auto)
#       GPUS=1
#       shift
#       ;;
#     *)
#       shift
#       ;;
#   esac
# done
# echo "GPUS: ${GPUS}"

# if  [ ${DATASET} == "mme" ]; then
#   python -m eval.vlm.eval.mme.eval "${ARGS[@]:1}"
# fi

# if [ ${DATASET} == "mmvet" ]; then
#     python -m eval.vlm.eval.mmvet.evaluate_mmvet --datasets mmvet "${ARGS[@]:1}"
# fi

# if [ ${DATASET} == "mmbench-dev-en" ]; then
#     torchrun \
#       --nnodes=$ARNOLD_WORKER_NUM \
#       --node_rank=$ARNOLD_ID \
#       --master_addr=$ARNOLD_WORKER_0_HOST \
#       --nproc_per_node=${GPUS} \
#       --master_port=${MASTER_PORT} \
#       -m eval.vlm.eval.mmbench.evaluate_mmbench_my --datasets mmbench_dev_20230712 "${ARGS[@]:1}" \
# 			--threshold $threshold --attn_backend $attn_backend \
# 			--save_dir $save_dir \
# 			--sparse_gsize $sparse_gsize \
# 			$vae_vit_sparse $self_attn_sparse \
# 			$mlp_save $mlp_save_dir \
# 			--reorder_method $reorder_method \
# 			$use_custom_mlp $mlp_use_similarity $use_quantized_mlp_w
# fi

# if [ ${DATASET} == "mmbench-dev-cn" ]; then
#     torchrun \
#       --nnodes=$ARNOLD_WORKER_NUM \
#       --node_rank=$ARNOLD_ID \
#       --master_addr=$ARNOLD_WORKER_0_HOST \
#       --nproc_per_node=${GPUS} \
#       --master_port=${MASTER_PORT} \
#       -m eval.vlm.eval.mmbench.evaluate_mmbench_my --datasets mmbench_dev_cn_20231003 "${ARGS[@]:1}" \
# 			--threshold $threshold --attn_backend $attn_backend \
# 			--save_dir $save_dir \
# 			--sparse_gsize $sparse_gsize \
# 			$vae_vit_sparse $self_attn_sparse \
# 			$mlp_save $mlp_save_dir \
# 			--reorder_method $reorder_method \
# 			$use_custom_mlp $mlp_use_similarity $use_quantized_mlp_w
# fi

# if [ ${DATASET} == "mmbench-test-en" ]; then
#     torchrun \
#       --nnodes=$ARNOLD_WORKER_NUM \
#       --node_rank=$ARNOLD_ID \
#       --master_addr=$ARNOLD_WORKER_0_HOST \
#       --nproc_per_node=${GPUS} \
#       --master_port=${MASTER_PORT} \
#       -m eval.vlm.eval.mmbench.evaluate_mmbench_my --datasets mmbench_test_en_20231003 "${ARGS[@]:1}" \
# 			--threshold $threshold --attn_backend $attn_backend \
# 			--save_dir $save_dir \
# 			--sparse_gsize $sparse_gsize \
# 			$vae_vit_sparse $self_attn_sparse \
# 			$mlp_save $mlp_save_dir \
# 			--reorder_method $reorder_method \
# 			$use_custom_mlp $mlp_use_similarity $use_quantized_mlp_w
# fi

# if [ ${DATASET} == "mmbench-test-cn" ]; then
#     torchrun \
#       --nnodes=$ARNOLD_WORKER_NUM \
#       --node_rank=$ARNOLD_ID \
#       --master_addr=$ARNOLD_WORKER_0_HOST \
#       --nproc_per_node=${GPUS} \
#       --master_port=${MASTER_PORT} \
#       -m eval.vlm.eval.mmbench.evaluate_mmbench_my --datasets mmbench_test_cn_20231003 "${ARGS[@]:1}" \
# 			--threshold $threshold --attn_backend $attn_backend \
# 			--save_dir $save_dir \
# 			--sparse_gsize $sparse_gsize \
# 			$vae_vit_sparse $self_attn_sparse \
# 			$mlp_save $mlp_save_dir \
# 			--reorder_method $reorder_method \
# 			$use_custom_mlp $mlp_use_similarity $use_quantized_mlp_w
# fi

# if [ ${DATASET} == "mmmu-dev" ]; then
#     torchrun \
#       --nnodes=$ARNOLD_WORKER_NUM \
#       --node_rank=$ARNOLD_ID \
#       --master_addr=$ARNOLD_WORKER_0_HOST \
#       --nproc_per_node=${GPUS} \
#       --master_port=${MASTER_PORT} \
#       -m eval.vlm.eval.mmmu.evaluate_mmmu --datasets MMMU_dev "${ARGS[@]:1}"
# fi

# if [ ${DATASET} == "mmmu-val" ]; then
#     torchrun \
#       --nnodes=$ARNOLD_WORKER_NUM \
#       --node_rank=$ARNOLD_ID \
#       --master_addr=$ARNOLD_WORKER_0_HOST \
#       --nproc_per_node=${GPUS} \
#       --master_port=${MASTER_PORT} \
#       -m eval.vlm.eval.mmmu.evaluate_mmmu --datasets MMMU_validation "${ARGS[@]:1}"
# fi

# if [ ${DATASET} == "mmmu-val_cot" ]; then
#     torchrun \
#       --nnodes=$ARNOLD_WORKER_NUM \
#       --node_rank=$ARNOLD_ID \
#       --master_addr=$ARNOLD_WORKER_0_HOST \
#       --nproc_per_node=${GPUS} \
#       --master_port=${MASTER_PORT} \
#       -m eval.vlm.eval.mmmu.evaluate_mmmu_cot --datasets MMMU_validation_cot "${ARGS[@]:1}"
# fi

# if [ ${DATASET} == "mmmu-test" ]; then
#     torchrun \
#       --nnodes=$ARNOLD_WORKER_NUM \
#       --node_rank=$ARNOLD_ID \
#       --master_addr=$ARNOLD_WORKER_0_HOST \
#       --nproc_per_node=${GPUS} \
#       --master_port=${MASTER_PORT} \
#       -m eval.vlm.eval.mmmu.evaluate_mmmu --datasets MMMU_test "${ARGS[@]:1}"
# fi

# if [ ${DATASET} == "mathvista-testmini" ]; then
#     torchrun \
#       --nnodes=$ARNOLD_WORKER_NUM \
#       --node_rank=$ARNOLD_ID \
#       --master_addr=$ARNOLD_WORKER_0_HOST \
#       --nproc_per_node=${GPUS} \
#       --master_port=${MASTER_PORT} \
#       -m eval.vlm.eval.mathvista.evaluate_mathvista --datasets MathVista_testmini "${ARGS[@]:1}"
# fi

# if [ ${DATASET} == "mathvista-test" ]; then
#     torchrun \
#       --nnodes=$ARNOLD_WORKER_NUM \
#       --node_rank=$ARNOLD_ID \
#       --master_addr=$ARNOLD_WORKER_0_HOST \
#       --nproc_per_node=${GPUS} \
#       --master_port=${MASTER_PORT} \
#       -m eval.vlm.eval.mathvista.evaluate_mathvista --datasets MathVista_test "${ARGS[@]:1}"
# fi

# if [ ${DATASET} == "pope" ]; then
#     torchrun \
#     --nnodes=$ARNOLD_WORKER_NUM \
#     --node_rank=$ARNOLD_ID \
#     --master_addr=$ARNOLD_WORKER_0_HOST \
#     --nproc_per_node=${GPUS} \
#     --master_port=${MASTER_PORT} \
#     -m eval.vlm.eval.pope.evaluate_pope --datasets pope "${ARGS[@]:1}"
# fi

# if [ ${DATASET} == "pope_cot" ]; then
#     torchrun \
#     --nnodes=$ARNOLD_WORKER_NUM \
#     --node_rank=$ARNOLD_ID \
#     --master_addr=$ARNOLD_WORKER_0_HOST \
#     --nproc_per_node=${GPUS} \
#     --master_port=${MASTER_PORT} \
#     -m eval.vlm.eval.pope.evaluate_pope --datasets pope_cot --cot "${ARGS[@]:1}"
# fi

# if [ ${DATASET} == "vqa-gqa-testdev" ]; then
#     torchrun \
#     --nnodes=$ARNOLD_WORKER_NUM \
#     --node_rank=$ARNOLD_ID \
#     --master_addr=$ARNOLD_WORKER_0_HOST \
#     --nproc_per_node=${GPUS} \
#     --master_port=${MASTER_PORT} \
#     -m eval.vlm.eval.vqa.evaluate_vqa --datasets gqa_testdev_llava "${ARGS[@]:1}"
# fi

# if [ ${DATASET} == "mmvp" ]; then
#     torchrun \
#       --nnodes=$ARNOLD_WORKER_NUM \
#       --node_rank=$ARNOLD_ID \
#       --master_addr=$ARNOLD_WORKER_0_HOST \
#       --nproc_per_node=${GPUS} \
#       --master_port=${MASTER_PORT} \
#       -m eval.vlm.eval.mmvp.evaluate_mmvp --datasets MMVP "${ARGS[@]:1}"
# fi
