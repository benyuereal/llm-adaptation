#!/bin/bash
# GLM-5.2-Channel-INT4-w4a8 (GlmMoeDsaForCausalLM) K100AI DCU 适配 — 一键应用 patch
#
# 修复两个独立根因 (长输入乱码):
#   1. shared DSA indexer 层跑零权重: 移植官方 skip_topk 机制
#      (index_topk_freq / index_skip_topk_offset), shared 层不建/不跑 indexer,
#      直接复用 full 层写入的共享 topk_indices_buffer
#   2. gfx928 prefill tilelang mqa_logits 返回值被丢弃: 该算子返回新 tensor
#      (非原地写入), 旧代码照抄 gfx938 原地写入模式丢弃返回值 -> topk 读全 0
#      -> 垃圾 indices -> 长输入乱码. 现直接用返回值传给 topk (与官方一致)
#
# 适用: 海光 K100AI DCU (gfx928), 定制 vllm 分支 (无 vllm._C 原生 topk 算子,
#       走 lightop top_k_per_row + tilelang mqa_logits/page_mqa_logits)
set -euo pipefail

VLLM_ROOT=$(python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))" 2>/dev/null)
if [ -z "$VLLM_ROOT" ]; then
    echo "[ERROR] 找不到 vllm 安装路径, 请确认 vllm 已安装且 python3 可用"
    exit 1
fi
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MOD_DIR="$SCRIPT_DIR/vllm_patches/modified"
ADD_DIR="$SCRIPT_DIR/vllm_patches/added"

echo "Target: $VLLM_ROOT"
echo ""

# ──────────────────────────────────────────────────────────────────────────────
# 1. Modified files (base 已有, 我们修改) — 先备份 .bak 再覆盖
# ──────────────────────────────────────────────────────────────────────────────
declare -A FILE_MAP=(
  ["deepseek_v2.py"]="$VLLM_ROOT/model_executor/models/deepseek_v2.py"
  ["mla.py"]="$VLLM_ROOT/model_executor/layers/mla.py"
  ["sparse_attn_indexer.py"]="$VLLM_ROOT/model_executor/layers/sparse_attn_indexer.py"
  ["flashmla_sparse.py"]="$VLLM_ROOT/v1/attention/backends/mla/flashmla_sparse.py"
)

for f in "${!FILE_MAP[@]}"; do
  target="${FILE_MAP[$f]}"
  if [ -f "$target" ]; then
    cp "$target" "${target}.bak"
    cp "$MOD_DIR/$f" "$target"
    echo "  ✓ [MODIFY] $f → $target"
  else
    echo "  ✗ [MODIFY] 目标不存在: $target"
    exit 1
  fi
done

# ──────────────────────────────────────────────────────────────────────────────
# 2. Added files (base 没有, 我们新增的 tilelang 算子)
# ──────────────────────────────────────────────────────────────────────────────
declare -A ADD_MAP=(
  ["mqa_logits.py"]="$VLLM_ROOT/model_executor/layers/mqa_logits.py"
  ["paged_mqa_logits.py"]="$VLLM_ROOT/model_executor/layers/paged_mqa_logits.py"
  ["sparse_mla_fwd.py"]="$VLLM_ROOT/v1/attention/backends/mla/sparse_mla_fwd.py"
)

for f in "${!ADD_MAP[@]}"; do
  target="${ADD_MAP[$f]}"
  mkdir -p "$(dirname "$target")"
  [ -f "$target" ] && cp "$target" "${target}.bak" || true
  cp "$ADD_DIR/$f" "$target"
  echo "  ✓ [ADD]     $f → $target"
done

# ──────────────────────────────────────────────────────────────────────────────
# 3. 语法检查
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== 语法检查 ==="
cd "$VLLM_ROOT"
ERR=0
for rel in \
  model_executor/models/deepseek_v2.py \
  model_executor/layers/mla.py \
  model_executor/layers/sparse_attn_indexer.py \
  model_executor/layers/mqa_logits.py \
  model_executor/layers/paged_mqa_logits.py \
  v1/attention/backends/mla/sparse_mla_fwd.py \
  v1/attention/backends/mla/flashmla_sparse.py; do
    if python3 -c "import ast; ast.parse(open('$rel').read())" 2>/dev/null; then
        echo "  OK  $rel"
    else
        echo "  FAIL $rel"; ERR=1
    fi
done

if [ $ERR -eq 0 ]; then
    echo ""
    echo "=== Patch applied successfully ==="
    echo "原文件已备份为 .bak. 回滚: for f in \$(find \$VLLM_ROOT -name '*.bak'); do mv \"\$f\" \"\${f%.bak}\"; done"
    echo "启动服务: bash $SCRIPT_DIR/start.sh"
else
    echo ""
    echo "=== [ERROR] 语法检查失败 ==="
    exit 1
fi
