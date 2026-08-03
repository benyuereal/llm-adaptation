#!/bin/bash
# MiniMax-M3 AWQ-INT4 + EAGLE3 投机解码启动脚本
# 端口 8082
# 用法: bash /workspace/quant-eagle3-hygon/start_eagle3.sh

set -ex

# 停残留 sglang 进程
pkill -f "sglang serve" || true
sleep 2

# 清 Triton/torchinductor 缓存（改过 kernel 后必须清）
rm -rf /models/.triton_cache/* /tmp/torchinductor_root/* /root/.triton/cache/* 2>/dev/null || true

# 环境变量
export SGLANG_USE_AITER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR=/models/tmp
export TORCHINDUCTOR_CACHE_DIR=/models/.torchinductor_cache
export TRITON_CACHE_DIR=/models/.triton_cache
mkdir -p $TMPDIR $TORCHINDUCTOR_CACHE_DIR $TRITON_CACHE_DIR

# 日志
LOG_FILE=/models/sglang_eagle3.log
echo "Starting EAGLE3 at $(date)" > "$LOG_FILE"

# ──────────────────────────────────────────────────────────────────────────────
# GPU memory check — 动态算 mem-fraction-static
# 模型: AWQ INT4 241GB, TP=8, 每卡权重 ~30.1GB
#       Draft 6.5GB, 每卡分摊 ~0.8GB
#       权重合计 ~30.9GB/卡
# ──────────────────────────────────────────────────────────────────────────────
FREE_GB=$(python3 -c "
import torch
free, total = torch.cuda.mem_get_info(0)
print(int(free // (1024**3)))
" 2>/dev/null || echo "0")

if [ "$FREE_GB" -lt 40 ]; then
    echo "WARNING: GPU 0 only has ${FREE_GB}GB free (need ~40GB+)."
    echo "Previous crashed runs may have leaked memory."
    echo "Attempting launch with tighter memory settings ..."
    # 显存紧张: 权重占 ~30.9GB, 只能留极少给 KV
    # frac 越高留给 KV 的越多: rest = after_free - before_free × (1-frac)
    # 32GB free → 30.9GB 权重 → 剩 1.1GB → 用 0.98 留给 KV 约 0.46GB
    MEM_FRAC=0.98
    CUDA_GRAPH_BS=2
else
    echo "GPU 0 has ${FREE_GB}GB free, memory looks healthy."
    MEM_FRAC=0.55
    CUDA_GRAPH_BS=8
fi

echo "Using --mem-fraction-static ${MEM_FRAC} --cuda-graph-max-bs ${CUDA_GRAPH_BS}"

sglang serve \
    --model-path /models/MiniMax-M3-AWQ-INT4 \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path /models/MiniMax-M3-EAGLE3 \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --tp-size 8 \
    --dtype bfloat16 \
    --mem-fraction-static "$MEM_FRAC" \
    --attention-backend triton \
    --mm-attention-backend triton_attn \
    --trust-remote-code \
    --skip-server-warmup \
    --cuda-graph-max-bs "$CUDA_GRAPH_BS" \
    --host 0.0.0.0 \
    --port 8082 \
    2>&1 | tee -a "$LOG_FILE"