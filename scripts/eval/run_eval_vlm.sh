# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

set -x

# Set proxy and API key
export OPENAI_API_KEY=$openai_api_key

export GPUS=2

# DATASETS=("mme" "mmbench-dev-en" "mmvet" "mmmu-val" "mathvista-testmini" "mmvp")
# DATASETS=("mmmu-val_cot")
# DATASETS=("mmbench-dev-en")
DATASETS=("mmbench-test-en")

output_path="./outputs/vlm/normal"
model_path="./models/JanusFlow-1.3B"

export ARNOLD_WORKER_NUM=1
export ARNOLD_ID=0
export ARNOLD_WORKER_0_HOST=localhost


DATASETS_STR="${DATASETS[*]}"
export DATASETS_STR

bash scripts/eval/eval_vlm.sh \
    $output_path \
    --model-path $model_path