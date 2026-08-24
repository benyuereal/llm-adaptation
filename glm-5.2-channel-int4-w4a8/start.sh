#!/bin/bash
# GLM-5.2 性能模式启动脚本 (关探针 + CUDA graph, 用于准确评测)
# 用法: bash start.sh            # 前台运行 (日志同时输出到终端 + 日志文件)
#       bash start.sh --daemon   # 后台运行 (nohup, 日志写文件, 终端可关)
# 用法旧: bash glm52_perf.sh
export USE_LIGHTOP_CONVERT_REQ_INDEX_TO_GLOBAL_INDEX=1
export VLLM_USE_LIGHTOP=1
export VLLM_USE_OPT_CAT=1
export USE_LIGHTOP_TOPK=1
export VLLM_USE_LIGHTOP_MOE_ALIGN=1
export VLLM_ROCM_USE_AITER_MOE=0

# 探针关闭 (零开销)
export PROBE_ON=0

# 日志文件 (带启动时间戳, 方便多次启动各保留一份)
LOG_DIR="/data1/csy/llm-adaptation/glm-5.2-channel-int4-w4a8/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/vllm_$(date +%Y%m%d_%H%M%S).log"

VLLM_CMD=( vllm serve /models/GLM-5.2-Channel-INT4-w4a8 \
    --trust-remote-code --dtype bfloat16 --max-model-len 56384 \
    --max-num-batched-tokens 8192 -tp 8 \
    --speculative_config '{"method": "mtp", "num_speculative_tokens": 3}' \
    --disable-custom-all-reduce --gpu-memory-utilization 0.92 \
    --max-num-seqs 64 --block-size 64)

if [ "${1:-}" = "--daemon" ]; then
    # 后台运行: 日志写文件, 终端可关闭
    echo "[start.sh] 后台启动 vLLM, 日志: $LOG_FILE"
    nohup "${VLLM_CMD[@]}" > "$LOG_FILE" 2>&1 &
    echo "[start.sh] vLLM PID: $!  (查看日志: tail -f $LOG_FILE)"
else
    # 前台运行: 日志同时输出到终端和文件 (tee)
    echo "[start.sh] 前台启动 vLLM, 日志: $LOG_FILE"
    "${VLLM_CMD[@]}" 2>&1 | tee "$LOG_FILE"
fi
