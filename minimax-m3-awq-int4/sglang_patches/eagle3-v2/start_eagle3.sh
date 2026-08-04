#!/bin/bash
# MiniMax-M3 AWQ-INT4 + EAGLE3 投机解码 — DCU (gfx928/gfx936) 启动脚本
# 端口 8082
#
# 基于 start.sh (量化版,含 DTK/HIP/lightop 环境变量 + compressed-tensors 量化)
# 融合 start_eagle3.sh 的 EAGLE3 参数,修正草稿模型路径。
# 要求:已运行 sglang_patches/eagle3/install.sh 安装 EAGLE3 patch。
#
# 用法: bash /workspace/llm-adaptation/minimax-m3-awq-int4/start_eagle3.sh

set -euo pipefail

# ──────────────────────────────────────────────────────────────────────────────
# 0. 停残留 sglang 进程
# ──────────────────────────────────────────────────────────────────────────────
pkill -f "sglang serve" 2>/dev/null || true
sleep 2

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
# VMFault 根因: EAGLE3 是首个让 sparse prefill 进入 cuda graph 的场景。
#   两层问题(本版均已根治, 复用标准 prefill kernel, 无需 verify/ 子模块):
#   1) score buffer 动态尺寸: capture(dummy seq_lens=1)/replay(真实~2000) 不一致 → 越界写。
#      修复=utils.py get_cu_seqblocks 在 graph capture 下用静态上界 batch*max_seqblock。
#   2) forward_extend 临时 tensor: 旧版用临时 extend_seq_lens 算 cu_seqlens, capture 后
#      forward_batch 被 GC, replay 读失效地址 → 垃圾 → 越界 (bs=1 碰巧不炸, bs>=阈值必崩)。
#      修复=verify 分支直接 materialize extend_seq_lens=draft_token_num, K 长度重建为
#      prefix+draft, prefix_lens 用 forward_batch.seq_lens graph buffer 引用 (地址稳定)。
# HIP_LAUNCH_BLOCKING=1 已移除(定位专用, 会让推理极慢; 性能测试必须关).
# export HIP_LAUNCH_BLOCKING=1

# ──────────────────────────────────────────────────────────────────────────────
# 3. sglang specific
# ──────────────────────────────────────────────────────────────────────────────
export SGLANG_USE_AITER=0
export SGLANG_MOE_TORCH_FALLBACK=0
# 日志实时落盘:非 tty 下 Python 默认块缓冲,会憋着不 flush,看起来日志不更新
export PYTHONUNBUFFERED=1

# EAGLE3 verify/prefill VMFault 定位探针 (minimax_sparse_backend.py 顶部读此变量)
# 输出独立文件 /workspace/logs/eagle3_verify_probe.log, 不污染推理日志
export EAGLE3_VERIFY_PROBE=1

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
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/models/.triton_cache}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
# DTK 2604 不支持 expandable_segments;EAGLE3 verify 路径显存更碎,用 gc 阈值控制
export PYTORCH_HIP_ALLOC_CONF="garbage_collection_threshold:0.8"

# 清 Triton/torchinductor 缓存（改过 kernel 后必须清,否则用旧编译结果）
export TORCHINDUCTOR_CACHE_DIR=/models/.torchinductor_cache
export TMPDIR=/models/tmp
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
rm -rf "$TRITON_CACHE_DIR"/* ~/.triton/* /tmp/torchinductor_root 2>/dev/null || true

# ──────────────────────────────────────────────────────────────────────────────
# 4. 显存配置
# 模型: AWQ INT4 241GB, TP=8, 每卡权重 ~30.1GB
#       Draft 6.5GB, 每卡分摊 ~0.8GB
#       权重合计 ~30.9GB/卡,EAGLE3 verify 还需额外 KV/buffer
# mem-fraction-static=0.85: 0.90 会在权重加载(process_weights transpose/contiguous)
#   阶段 OOM,0.85 与之前 RSF 跑通配置一致,留足加载临时空间
# context-length=204800: 限制上下文 20万(>>HumanEval 最长 33K),省 score buffer
#   (方案C 上界 context_len:1M→20万,score 单卡 0.45GB→0.09GB)与 req_to_token pool
# cuda-graph-max-bs=16: verify 全 cuda graph(方案C),bs=16
# ──────────────────────────────────────────────────────────────────────────────
MEM_FRAC=0.85
CUDA_GRAPH_BS=16
CONTEXT_LEN=204800

# ──────────────────────────────────────────────────────────────────────────────
# 5. 日志(环境变量 LOG_DIR 可覆盖,默认 /workspace/logs;不放 /models,那是模型盘)
# ──────────────────────────────────────────────────────────────────────────────
LOG_DIR="${LOG_DIR:-/workspace/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/sglang_eagle3.log"
PROBE_FILE="$LOG_DIR/eagle3_verify_probe.log"
# 清空上次探针日志, 保证本次输出干净 (探针每次启动会 append "started" 行)
: > "$PROBE_FILE" 2>/dev/null || true
echo "Starting EAGLE3 at $(date)" > "$LOG_FILE"
echo "Using --mem-fraction-static ${MEM_FRAC} --cuda-graph-max-bs ${CUDA_GRAPH_BS}" | tee -a "$LOG_FILE"

# ──────────────────────────────────────────────────────────────────────────────
# 6. Launch sglang serve with EAGLE3 speculative decoding
# ──────────────────────────────────────────────────────────────────────────────
# 注意:不加 --quantization compressed-tensors。该参数会同时套用到草稿模型,
# 但草稿模型(EAGLE3 draft, BF16)无量化 config → CompressedTensorsConfig() 构造
# 缺参数崩溃。让 sglang 从主模型 config.json 的 quantization_config 自动识别即可
# (主模型 quant_method=compressed-tensors 会自动量化加载,草稿走 BF16)。
exec sglang serve \
    --model-path /models/MiniMax-M3-AWQ-INT4 \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path /models/Inferact/MiniMax-M3-EAGLE3 \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-attention-mode prefill \
    --mem-fraction-static "$MEM_FRAC" \
    --context-length "$CONTEXT_LEN" \
    --tp 8 \
    --dtype bfloat16 \
    --attention-backend triton \
    --mm-attention-backend triton_attn \
    --trust-remote-code \
    --skip-server-warmup \
    --cuda-graph-max-bs "$CUDA_GRAPH_BS" \
    --chunked-prefill-size 4096 \
    --max-running-requests 64 \
    --schedule-policy fcfs \
    --host 0.0.0.0 \
    --port 8082 \
    2>&1 | tee -a "$LOG_FILE"
