#!/bin/bash
# GLM-5.2 性能模式启动脚本 (关探针 + CUDA graph, 用于准确评测)
# 用法: bash glm52_perf.sh
export USE_LIGHTOP_CONVERT_REQ_INDEX_TO_GLOBAL_INDEX=1
export VLLM_USE_LIGHTOP=1
export VLLM_USE_OPT_CAT=1
export USE_LIGHTOP_TOPK=1
export VLLM_USE_LIGHTOP_MOE_ALIGN=1
export VLLM_ROCM_USE_AITER_MOE=0

# 探针关闭 (零开销)
export PROBE_ON=0

vllm serve /models/GLM-5.2-Channel-INT4-w4a8 \
    --trust-remote-code --dtype bfloat16 --max-model-len 16384 \
    --max-num-batched-tokens 8192 -tp 8 \
    --speculative_config '{"method": "mtp", "num_speculative_tokens": 3}' \
    --disable-custom-all-reduce --gpu-memory-utilization 0.92 \
    --max-num-seqs 64 --block-size 64
