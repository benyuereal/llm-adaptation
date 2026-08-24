#!/bin/bash
# GLM-5.2-Channel-INT4-w4a8 + MTP — [DEQUANT-ATTN] 一键应用 patch
#
# 优化 (profiled 2026-08-24, AMD ROCm gfx928, TP=8, MTP num_spec=4):
#   w8a8 int8 GEMM (lmslim matmul_int8, Triton) 占 decode GPU 时间 ~60%,
#   在 decode 小 M (verify M=5 / draft M=1) 下卡在 ~90us 地板, 任何 Triton
#   config 都打不破; 而库 bf16 GEMM (rocBLAS) 没有这个地板, 在这些
#   attention shape 上快 2.5-2.8x.
#
# 改动 (整文件覆盖, 仅 2 处 hunk):
#   model_executor/layers/quantization/slimquant_w4a8.py
#     1. 新增 _dequant_attn_enabled() (读环境变量 VLLM_DEQUANT_ATTN)
#     2. get_quant_method() 的 LinearBase 分支: VLLM_DEQUANT_ATTN=1 时
#        强制 dequant=True -> 所有 attention int8 线性层加载时 dequant 成
#        bf16. MoE (int4 w4a8) 走 FusedMoE 分支, 不受影响.
#
# 效果: 19.68 -> 39.25 tok/s (~2x), 接受长度无损 (2.887->2.960),
#       显存 +3.59GB/rank.
#
# 注意: patch 本身不改变默认行为, 需启动时 export VLLM_DEQUANT_ATTN=1
#       (mtp_start.sh 已内置).
set -euo pipefail

VLLM_ROOT=$(python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))" 2>/dev/null)
if [ -z "$VLLM_ROOT" ]; then
    echo "[ERROR] 找不到 vllm 安装路径, 请确认 vllm 已安装且 python3 可用"
    exit 1
fi
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MOD_DIR="$SCRIPT_DIR/vllm_patches/modified"

echo "Target: $VLLM_ROOT"
echo ""

# ──────────────────────────────────────────────────────────────────────────────
# 1. Modified files (base 已有, 我们修改) — 先备份 .bak 再覆盖
# ──────────────────────────────────────────────────────────────────────────────
declare -A FILE_MAP=(
  ["slimquant_w4a8.py"]="$VLLM_ROOT/model_executor/layers/quantization/slimquant_w4a8.py"
)

for f in "${!FILE_MAP[@]}"; do
  target="${FILE_MAP[$f]}"
  if [ -f "$target" ]; then
    # 检测内容是否一致 (已经是最新版本则跳过)
    if diff -q "$MOD_DIR/$f" "$target" >/dev/null 2>&1; then
      echo "  ⊘ [SKIP]   $f → 已是最新版本, 跳过"
    else
      cp "$target" "${target}.bak"
      cp "$MOD_DIR/$f" "$target"
      echo "  ✓ [MODIFY] $f → $target"
    fi
  else
    echo "  ✗ [MODIFY] 目标不存在: $target"
    exit 1
  fi
done

# ──────────────────────────────────────────────────────────────────────────────
# 2. 语法检查
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== 语法检查 ==="
cd "$VLLM_ROOT"
ERR=0
for rel in \
  model_executor/layers/quantization/slimquant_w4a8.py; do
    if python3 -c "import ast; ast.parse(open('$rel').read())" 2>/dev/null; then
        echo "  OK  $rel"
    else
        echo "  FAIL $rel"; ERR=1
    fi
done

if [ $ERR -eq 0 ]; then
    echo ""
    echo "=== Patch applied successfully ==="
    echo "原文件已备份为 .bak. 回滚: bash $SCRIPT_DIR/revert_patch.sh"
    echo "启动服务: bash $SCRIPT_DIR/mtp_start.sh"
else
    echo ""
    echo "=== [ERROR] 语法检查失败 ==="
    exit 1
fi
