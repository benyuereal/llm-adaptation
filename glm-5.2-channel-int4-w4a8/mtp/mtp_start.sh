#!/bin/bash
# GLM-5.2 + MTP num_spec=4 + DEQUANT-ATTN: dequantize all attention int8
# linears to bf16 (the int8 GEMM is 60% of decode GPU time, 2.5-2.8x slower
# than bf16 at small M). MoE stays int4. Same config as champion baseline.
export VLLM_USE_V1=1
export VLLM_USE_FUSED_FILL_RMS_CAT=1
export LMSLIM_USE_LIGHTOP=1
export VLLM_DEQUANT_MTP_LAYER=-1   # MTP-layer dequant OFF (separate experiment)
export VLLM_DEQUANT_ATTN=1         # NEW: dequant all attention int8 linears -> bf16
exec vllm serve /models/GLM-5.2-Channel-INT4-w4a8 \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len 57344 \
  --max-num-batched-tokens 8192 \
  -tp 8 \
  --speculative_config '{"method": "mtp", "num_speculative_tokens": 4, "model": "/models/GLM-5.2-Channel-INT4-w4a8"}' \
  --disable-custom-all-reduce \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 64 \
  --block-size 64 \
  --disable-hybrid-kv-cache-manager
