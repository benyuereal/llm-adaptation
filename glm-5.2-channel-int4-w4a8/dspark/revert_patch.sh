#!/bin/bash
# GLM-5.2-Channel-INT4-w4a8 + DSpark 投机解码 — 回滚 patch
#
# 从 .bak 恢复被修改的原有文件，并删除纯新增的 DSpark 文件。
# 与 apply_patch.sh 的文件清单保持一致。
#
# 用法：
#   bash revert_patch.sh                 # 回滚真实 vllm
#   DRY_RUN=1 bash revert_patch.sh       # 只打印将要做的事
#   VLLM_ROOT=/path/to/vllm bash revert_patch.sh
set -euo pipefail

if [ -n "${VLLM_ROOT:-}" ]; then
  :
else
  VLLM_ROOT=$(python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))" 2>/dev/null)
fi
if [ -z "${VLLM_ROOT:-}" ] || [ ! -d "$VLLM_ROOT" ]; then
  echo "[ERROR] 找不到 vllm 安装路径 (可用 VLLM_ROOT=/path 指定)"
  exit 1
fi
DRY_RUN="${DRY_RUN:-0}"
echo "Target vllm : $VLLM_ROOT"
echo "Dry-run     : $DRY_RUN"
echo ""

MODIFIED_FILES=(
  "transformers_utils/configs/speculators/algos.py"
  "config/attention.py"
  "config/speculative.py"
  "config/vllm.py"
  "config/utils.py"
  "model_executor/models/registry.py"
  "model_executor/models/deepseek_v2.py"
  "model_executor/models/glm4_moe_lite.py"
  "attention/layer.py"
  "v1/worker/gpu/model_runner.py"
  "v1/worker/gpu/cudagraph_utils.py"
  "v1/worker/gpu/attn_utils.py"
  "v1/worker/gpu/spec_decode/__init__.py"
  "v1/worker/gpu_worker.py"
  "v1/core/sched/scheduler.py"
)
ADDED_FILES=(
  "model_executor/models/qwen3_dspark.py"
  "model_executor/models/qwen3_dflash.py"
  "v1/worker/gpu/spec_decode/dspark/__init__.py"
  "v1/worker/gpu/spec_decode/dspark/speculator.py"
  "v1/worker/gpu/spec_decode/dspark/utils.py"
  "v1/worker/gpu/spec_decode/dflash/__init__.py"
  "v1/worker/gpu/spec_decode/dflash/utils.py"
  "v1/worker/gpu/spec_decode/eagle3_utils.py"
  "v1/worker/gpu/spec_decode/eagle_utils.py"
)

# ── 1. 从 .bak 恢复被修改文件 ────────────────────────────────────────────────
echo "=== [1/2] 从 .bak 恢复被修改文件 ==="
for rel in "${MODIFIED_FILES[@]}"; do
  target="$VLLM_ROOT/$rel"
  bak="${target}.bak"
  if [ ! -f "$bak" ]; then
    echo "  ⊘ [SKIP]   $rel (无 .bak，可能从未应用)"
    continue
  fi
  if [ "$DRY_RUN" = "1" ]; then
    echo "  ~ [DRY]    $rel (将从 .bak 恢复)"
    continue
  fi
  cp "$bak" "$target"
  echo "  ✓ [REVERT] $rel"
done

# ── 2. 删除新增文件 ──────────────────────────────────────────────────────────
echo ""
echo "=== [2/2] 删除新增的 DSpark 文件 ==="
for rel in "${ADDED_FILES[@]}"; do
  target="$VLLM_ROOT/$rel"
  if [ ! -f "$target" ]; then
    echo "  ⊘ [SKIP]   $rel (不存在)"
    continue
  fi
  if [ "$DRY_RUN" = "1" ]; then
    echo "  ~ [DRY]    $rel (将删除)"
    continue
  fi
  rm -f "$target"
  echo "  ✓ [DEL]    $rel"
done

# 清理空目录（dspark / dflash 子目录）
if [ "$DRY_RUN" != "1" ]; then
  for d in \
    "$VLLM_ROOT/v1/worker/gpu/spec_decode/dspark" \
    "$VLLM_ROOT/v1/worker/gpu/spec_decode/dflash"; do
    if [ -d "$d" ]; then
      find "$d" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
      rmdir "$d" 2>/dev/null && echo "  ♻ [CLEAN] 删除空目录 ${d#$VLLM_ROOT/}" || true
    fi
  done
  # 清理相关 __pycache__
  for rel in "${MODIFIED_FILES[@]}"; do
    pycdir="$(dirname "$VLLM_ROOT/$rel")/__pycache__"
    [ -d "$pycdir" ] && find "$pycdir" -name "*.pyc" -newer "$VLLM_ROOT" -delete 2>/dev/null || true
  done
fi

echo ""
if [ "$DRY_RUN" = "1" ]; then
  echo "=== DRY-RUN 完成（未改动任何文件）==="
  exit 0
fi
echo "=== Revert 完成 ==="
echo "注意：.bak 备份文件仍保留在原地，如需彻底清理请手动删除 *.bak。"
