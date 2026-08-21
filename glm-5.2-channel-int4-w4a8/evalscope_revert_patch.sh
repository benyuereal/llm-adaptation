#!/bin/bash
# evalscope 1.8.1 — HumanEval 思考模式提取修复 — 回滚 patch
# 从 .bak 恢复原始 humaneval_adapter.py 并清理 __pycache__
set -euo pipefail

EVALSCOPE_ROOT=$(python3 -c "import evalscope, os; print(os.path.dirname(evalscope.__file__))" 2>/dev/null)
if [ -z "$EVALSCOPE_ROOT" ]; then
    echo "[ERROR] 找不到 evalscope 安装路径"
    exit 1
fi

TARGET="$EVALSCOPE_ROOT/benchmarks/humaneval/humaneval_adapter.py"
BAK="${TARGET}.bak"

if [ ! -f "$BAK" ]; then
    echo "[WARN] 未找到备份 .bak ($BAK)"
    echo "       若从未应用过 patch 则无需回滚; 若是 .bak.prepatch, 请手动核对."
    exit 0
fi

cp "$BAK" "$TARGET"
echo "  ✓ [REVERT] humaneval_adapter.py 已从 .bak 恢复"

PYC_DIR="$(dirname "$TARGET")/__pycache__"
if [ -d "$PYC_DIR" ]; then
    rm -f "$PYC_DIR"/humaneval_adapter.*.pyc 2>/dev/null && \
        echo "  ♻ [CLEAN] 已清理 __pycache__"
fi

if python3 -c "import ast; ast.parse(open('$TARGET').read())" 2>/dev/null; then
    echo "=== Revert OK ==="
else
    echo "=== [ERROR] 回滚后语法检查失败, 请检查 $TARGET ==="
    exit 1
fi
