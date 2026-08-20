#!/bin/bash
# GLM-5.2-Channel-INT4-w4a8 + MTP 投机解码 — DCU (gfx928) 启动脚本
# 端口 8000 (默认开启 MTP 投机解码, num_speculative_tokens=3)
#
# 要求: 已运行 apply_patch.sh 安装修复 patch.
# 用法: bash /workspace/llm-adaptation/glm-5.2-channel-int4-w4a8/start.sh
set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# 0. 停残留 vllm 进程 (kill -9 残留会泄漏显存, 需连 EngineCore/Worker 一起停)
# ──────────────────────────────────────────────────────────────────────────────
pkill -9 -f "vllm serve" 2>/dev/null || true
pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
pkill -9 -f "VLLM::Worker_TP" 2>/dev/null || true
sleep 3

# ──────────────────────────────────────────────────────────────────────────────
# 1. Library paths (DTK/HIP)
# ──────────────────────────────────────────────────────────────────────────────
export LD_LIBRARY_PATH="/opt/dtk/lib:/opt/dtk/hip/lib:/opt/dtk/dcc/lib:/opt/hyhal/lib:${LD_LIBRARY_PATH:-}"

# ──────────────────────────────────────────────────────────────────────────────
# 2. lightop / tilelang 算子开关 (gfx928 定制路径)
#    本分支无 vllm._C 原生 topk 算子, DSA indexer 走 lightop top_k_per_row +
#    tilelang mqa_logits / page_mqa_logits (apply_patch.sh 已修复其返回值 bug)
# ──────────────────────────────────────────────────────────────────────────────
export USE_LIGHTOP_CONVERT_REQ_INDEX_TO_GLOBAL_INDEX=1
export VLLM_USE_LIGHTOP=1
export VLLM_USE_OPT_CAT=1
export USE_LIGHTOP_TOPK=1
export VLLM_USE_LIGHTOP_MOE_ALIGN=1
export VLLM_ROCM_USE_AITER_MOE=0

# 日志实时落盘
export PYTHONUNBUFFERED=1

# ──────────────────────────────────────────────────────────────────────────────
# 3. 缓存目录 (改过 kernel 后必须清, 否则用旧编译结果)
# ──────────────────────────────────────────────────────────────────────────────
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/models/.triton_cache}"
export TORCHINDUCTOR_CACHE_DIR=/models/.torchinductor_cache
export TMPDIR=/models/tmp
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
rm -rf "$TRITON_CACHE_DIR"/* ~/.triton/* /tmp/torchinductor_root 2>/dev/null || true

# ──────────────────────────────────────────────────────────────────────────────
# 4. 日志
# ──────────────────────────────────────────────────────────────────────────────
LOG_DIR="${LOG_DIR:-/data1/csy/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/glm52_vllm.log"
echo "Starting GLM-5.2 at $(date)" > "$LOG_FILE"

# ──────────────────────────────────────────────────────────────────────────────
# 5. Launch vllm serve with MTP speculative decoding
#    --speculative_config: MTP 投机解码, num_speculative_tokens=3
#    --block-size 64: FlashMLA sparse backend 要求 block_size=64
#    --gpu-memory-utilization 0.92: 8卡 TP, 权重~47GB/卡
# ──────────────────────────────────────────────────────────────────────────────
exec vllm serve /models/GLM-5.2-Channel-INT4-w4a8 \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len 16384 \
    --max-num-batched-tokens 8192 \
    -tp 8 \
    --speculative_config '{"method": "mtp", "num_speculative_tokens": 3}' \
    --disable-custom-all-reduce \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 64 \
    --block-size 64 \
    2>&1 | tee -a "$LOG_FILE"
