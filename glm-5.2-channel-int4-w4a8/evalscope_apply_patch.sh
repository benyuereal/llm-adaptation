#!/bin/bash
# evalscope 1.8.1 — HumanEval 思考模式提取修复 — 一键应用 patch
#
# 背景: GLM-5.2 等模型在思考模式下, HumanEval 回答常包含多个 markdown 代码块
#   (探索性的草稿 + 最终实现, 偶尔混入相邻题目的代码块). evalscope 原版
#   _postprocess 一律取 blocks[0], 会取到草稿/错题代码 -> 误判 FAIL,
#   HumanEval pass@1 显著偏低.
#
# 修复 (仅 humaneval_adapter.py, 思考模式相关, 不影响其它 benchmark):
#   1. prompt_template 引导模型把最终实现放在最后一个 ```python 代码块
#   2. _postprocess 优先取「最后一个定义了目标 entry_point 函数」的代码块,
#      找不到则回退到最后一个块 (而非第一个)
#
# 验证: 思考模式 HumanEval 100% → 99%+ (真实模型错误除外, 非提取问题)
# 适用: evalscope==1.8.1 (其它版本需重新核对上下文)
#
# 非侵入: 不修改 evalscope 其它文件; 不应用也不影响主适配工作,
#         仅思考模式下 HumanEval 提取偶发错题.
set -euo pipefail

EVALSCOPE_ROOT=$(python3 -c "import evalscope, os; print(os.path.dirname(evalscope.__file__))" 2>/dev/null)
if [ -z "$EVALSCOPE_ROOT" ]; then
    echo "[ERROR] 找不到 evalscope 安装路径, 请确认 evalscope 已安装且 python3 可用"
    exit 1
fi
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PATCH_DIR="$SCRIPT_DIR/evalscope_patches"

TARGET="$EVALSCOPE_ROOT/benchmarks/humaneval/humaneval_adapter.py"
NEW="$PATCH_DIR/humaneval_adapter.py"

echo "Target: $TARGET"
echo ""

# ──────────────────────────────────────────────────────────────────────────────
# 应用修改: 备份 .bak 再覆盖; 已是最新版本则跳过 (幂等)
# ──────────────────────────────────────────────────────────────────────────────
if [ ! -f "$TARGET" ]; then
    echo "  ✗ [ERROR] 目标文件不存在: $TARGET"
    echo "           evalscope 版本可能不是 1.8.1, 请人工核对 humaneval_adapter.py 路径"
    exit 1
fi

if diff -q "$NEW" "$TARGET" >/dev/null 2>&1; then
    echo "  ⊘ [SKIP] humaneval_adapter.py → 已是最新版本, 跳过 (已应用过)"
else
    # 校验目标是否为未修改的 1.8.1 原版 (避免在已被其它改动污染的文件上盲目覆盖)
    if [ -f "$PATCH_DIR/humaneval_adapter.py.orig" ]; then
        if ! diff -q "$PATCH_DIR/humaneval_adapter.py.orig" "$TARGET" >/dev/null 2>&1; then
            echo "  ⚠ [WARN] 目标文件与 1.8.1 原版不一致 (可能已被手动改过或 evalscope 版本不同)"
            echo "          将备份当前版本为 .bak.prepatch 后覆盖. 请确认改动可接受."
            cp "$TARGET" "${TARGET}.bak.prepatch"
        else
            cp "$TARGET" "${TARGET}.bak"
        fi
    else
        cp "$TARGET" "${TARGET}.bak"
    fi
    cp "$NEW" "$TARGET"
    echo "  ✓ [MODIFY] humaneval_adapter.py → $TARGET"
fi

# 清理 __pycache__ 避免加载旧字节码
PYC_DIR="$(dirname "$TARGET")/__pycache__"
if [ -d "$PYC_DIR" ]; then
    rm -f "$PYC_DIR"/humaneval_adapter.*.pyc 2>/dev/null && \
        echo "  ♻ [CLEAN] 已清理 humaneval_adapter __pycache__"
fi

# ──────────────────────────────────────────────────────────────────────────────
# 语法检查
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== 语法检查 ==="
if python3 -c "import ast; ast.parse(open('$TARGET').read())" 2>/dev/null; then
    echo "  OK  benchmarks/humaneval/humaneval_adapter.py"
    echo ""
    echo "=== Patch applied successfully ==="
    echo "原文件已备份为 .bak. 回滚: cp \"${TARGET}.bak\" \"$TARGET\""
else
    echo "  FAIL benchmarks/humaneval/humaneval_adapter.py"
    echo ""
    echo "=== [ERROR] 语法检查失败, 已自动回滚 ==="
    if [ -f "${TARGET}.bak" ]; then cp "${TARGET}.bak" "$TARGET"; fi
    exit 1
fi
