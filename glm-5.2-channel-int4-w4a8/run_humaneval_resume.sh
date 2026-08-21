#!/bin/bash
# GLM-5.2 HumanEval 思考模式评测 — 断点续跑脚本
#
# 用途: 回家/断开后重新继续 173313 那次评测。evalscope --use-cache 会复用
#       outputs/20260821_173313 下已缓存的 predictions/reviews (按 sample_id 匹配),
#       只补跑没完成的题, 不必从头重跑 164 题。
#
# 前置: vLLM 服务已在跑 (bash start.sh), 且 evalscope 已应用提取修复 patch
#       (bash evalscope_apply_patch.sh)
#
# 用法:
#   bash glm-5.2-channel-int4-w4a8/run_humaneval_resume.sh          # 续跑 (复用缓存)
#   bash glm-5.2-channel-int4-w4a8/run_humaneval_resume.sh --fresh  # 从头全新跑 (新建目录)
set -euo pipefail

MODEL=/models/GLM-5.2-Channel-INT4-w4a8
API_URL=http://127.0.0.1:8000/v1/chat/completions
WORK_DIR=./outputs/
RESUME_DIR=/data1/csy/outputs/20260821_173313   # 要续跑的那次评测目录

# 检查 vLLM 是否在跑
if ! curl -s --max-time 3 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
    echo "[ERROR] vLLM 服务未启动, 请先: bash start.sh"
    exit 1
fi

if [ "${1:-}" = "--fresh" ]; then
    echo "=== 全新评测 (不复用缓存, 新建目录) ==="
    evalscope eval \
        --model "$MODEL" \
        --api-url "$API_URL" \
        --api-key EMPTY \
        --eval-type openai_api \
        --datasets humaneval \
        --generation-config '{"temperature": 0.2, "top_p": 0.95, "max_tokens": 15900, "repetition_penalty": 1.05}' \
        --eval-batch-size 32 \
        --work-dir "$WORK_DIR"
else
    if [ ! -d "$RESUME_DIR" ]; then
        echo "[ERROR] 续跑目录不存在: $RESUME_DIR"
        echo "        若要全新跑, 加 --fresh 参数"
        exit 1
    fi
    echo "=== 断点续跑: 复用 $RESUME_DIR 的缓存, 只补跑未完成题 ==="
    evalscope eval \
        --model "$MODEL" \
        --api-url "$API_URL" \
        --api-key EMPTY \
        --eval-type openai_api \
        --datasets humaneval \
        --generation-config '{"temperature": 0.2, "top_p": 0.95, "max_tokens": 15900, "repetition_penalty": 1.05}' \
        --eval-batch-size 32 \
        --work-dir "$WORK_DIR" \
        --use-cache "$RESUME_DIR"
fi
