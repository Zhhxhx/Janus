#!/bin/bash

# 1. 定义一个清理函数
#    这个函数会在接收到信号时被调用
cleanup() {
    echo -e "\n\nCtrl+C detected. Terminating all child processes..."
    # 使用 pkill 来杀死所有由这个脚本启动的 Python 子进程
    # -P $$ 会查找父进程ID是当前脚本ID($$)的进程
    pkill -P $$
    echo "All child processes terminated."
    # 退出脚本
    exit 1
}

# 2. 设置陷阱 (trap)
#    当脚本接收到 SIGINT (Ctrl+C) 或 SIGTERM (来自 kill 命令) 信号时，
#    调用我们定义的 cleanup 函数。
trap cleanup SIGINT SIGTERM

export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=4

threshold=4e-3
# attn_backend="naive_sparse_quant_cfg"
attn_backend="flash"
# attn_backend="naive"
# attn_backend="naive_sdma"
sparse_gsize=16
# mlp_backend="argus"
# mlp_backend="flightvgm"
mlp_backend="figna"
# mlp_backend="axcore"
# mlp_backend="default"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
	python ./profile/und_flow_inference.py \
	--threshold $threshold \
	--attn_backend $attn_backend \
	--sparse_gsize $sparse_gsize \
	--mlp_backend $mlp_backend &

wait