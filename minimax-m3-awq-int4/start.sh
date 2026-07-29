#!/bin/bash
# MiniMax-M3 AWQ INT4 — DCU (gfx928/gfx936) launch script
# Requires: sglang dev 0.0.0.dev12695+, DTK 2604+, 8x K100_AI DCU
# First launch is slow (~10min) due to Triton MoE kernel autotuning.
# Subsequent launches use cached kernels from ~/.triton/cache.

set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# 1. Library paths (DTK/HIP/OMP)
# ──────────────────────────────────────────────────────────────────────────────
export LD_LIBRARY_PATH="/opt/dtk/lib:/opt/dtk/hip/lib:/opt/dtk/dcc/lib:/opt/hyhal/lib:${LD_LIBRARY_PATH:-}"

# ──────────────────────────────────────────────────────────────────────────────
# 2. Triton / HIP environment
# ──────────────────────────────────────────────────────────────────────────────
export TRITON_HIP_USE_BLOCK_PINGPONG=0
export HIP_FORCE_DEV_KERNARG=1
export CUDA_HOME=/opt/dtk

# ──────────────────────────────────────────────────────────────────────────────
# 3. sglang specific
# ──────────────────────────────────────────────────────────────────────────────
export SGLANG_USE_AITER=0
export SGLANG_MOE_TORCH_FALLBACK=0

# ──────────────────────────────────────────────────────────────────────────────
# 3b. MoE kernel fix for gfx936 (K100AI/BW100)
# fused MoE gate has no tuned config for this arch → precision loss
# Option 1: use lightop implementation
export VLLM_ENABLE_MOE_FUSED_GATE=1
export VLLM_USE_LIGHTOP=1
# Option 2 (alternative): disable fused gate entirely
# export VLLM_ENABLE_MOE_FUSED_GATE=0

# ──────────────────────────────────────────────────────────────────────────────
# 3c. Performance tuning
# ──────────────────────────────────────────────────────────────────────────────
# Reduce Triton autotuning overhead on repeated launches
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HOME/.triton/cache}"
# OMP threads for CPU-bound pre/post processing (one per NUMA node)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
# Reduce HIP memory fragmentation (note: expandable_segments not supported on DTK 2604)
export PYTORCH_HIP_ALLOC_CONF="garbage_collection_threshold:0.8"

# ──────────────────────────────────────────────────────────────────────────────
# 4. GPU memory check
# ──────────────────────────────────────────────────────────────────────────────
FREE_GB=$(python3 -c "
import torch
free, total = torch.cuda.mem_get_info(0)
print(int(free // (1024**3)))
" 2>/dev/null || echo "0")

if [ "$FREE_GB" -lt 40 ]; then
    echo "WARNING: GPU 0 only has ${FREE_GB}GB free (need ~50GB)."
    echo "Previous crashed runs may have leaked memory."
    echo "Please restart the container to reclaim GPU memory, then rerun."
    echo "Attempting launch with --mem-fraction-static 0.20 ..."
    MEM_FRAC=0.20
else
    MEM_FRAC=0.55
fi

# ──────────────────────────────────────────────────────────────────────────────
# 5. Launch sglang serve
# ──────────────────────────────────────────────────────────────────────────────
exec sglang serve \
    --model-path /models/MiniMax-M3-AWQ-INT4 \
    --mem-fraction-static "$MEM_FRAC" \
    --tp 8 \
    --dtype bfloat16 \
    --quantization compressed-tensors \
    --attention-backend triton \
    --mm-attention-backend triton_attn \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8080 \
    --cuda-graph-max-bs 8 \
    --chunked-prefill-size 4096 \
    --max-running-requests 64 \
    --schedule-policy fcfs
