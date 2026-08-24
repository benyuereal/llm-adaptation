#!/bin/bash
# GLM-5.2-Channel-INT4-w4a8 + MTP — [DEQUANT-ATTN] 回滚 patch
# 从 .bak 恢复原始 slimquant_w4a8.py 并清理 __pycache__
set -euo pipefail

VLLM_ROOT=$(python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))" 2>/dev/null)
if [ -z "$VLLM_ROOT" ]; then
    echo "[ERROR] 找不到 vllm 安装路径"
    exit 1
fi

TARGET="$VLLM_ROOT/model_executor/layers/quantization/slimquant_w4a8.py"
BAK="${TARGET}.bak"

if [ ! -f "$BAK" ]; then
    echo "[WARN] 未找到备份 .bak ($BAK)"
    echo "       若从未应用过 patch 则无需回滚; 若是 .bak.prepatch, 请手动核对."
    exit 0
fi

cp "$BAK" "$TARGET"
echo "  ✓ [REVERT] slimquant_w4a8.py 已从 .bak 恢复"

PYC_DIR="$(dirname "$TARGET")/__pycache__"
if [ -d "$PYC_DIR" ]; then
    rm -f "$PYC_DIR"/slimquant_w4a8.*.pyc 2>/dev/null && \
        echo "  ♻ [CLEAN] 已清理 __pycache__"
fi

if python3 -c "import ast; ast.parse(open('$TARGET').read())" 2>/dev/null; then
    echo "=== Revert OK ==="
else
    echo "=== [ERROR] 回滚后语法检查失败, 请检查 $TARGET ==="
    exit 1
fi
