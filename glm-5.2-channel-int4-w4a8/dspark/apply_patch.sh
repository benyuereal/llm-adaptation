#!/bin/bash
# GLM-5.2-Channel-INT4-w4a8 + DSpark 投机解码 — 一键应用 patch
#
# ⚠️ 当前状态：DSpark backport 部分完成、未启用、未测试。
#    本 patch 仅供以后有场景时应用，默认不要在生产环境启用。
#    详见同目录 README.md。
#
# 作用：把 DSpark（半自回归并行投机解码）的 backport 文件覆盖/新增到预装 vllm。
#   - 15 个"被修改的原有文件"：整文件覆盖（覆盖前备份 .bak）。
#   - 9 个"纯新增文件"：直接新增（qwen3_dspark.py、spec_decode/dspark/ 等）。
#
# 幂等：目标文件已是最新版本则 SKIP；.bak 只在首次覆盖时创建（保留最初原始版）。
# 语法：应用后对所有 .py 做 ast.parse 检查。
#
# 应用后 DSpark 仍不会自动启用 —— 需在启动时通过 speculative_config 指定
# method="dspark"（示例见 README.md「启用方式」）。
#
# 用法：
#   bash apply_patch.sh                 # 应用到真实 vllm
#   DRY_RUN=1 bash apply_patch.sh       # 只打印将要做的事，不改动任何文件
#   VLLM_ROOT=/path/to/vllm bash apply_patch.sh   # 指定 vllm 根目录（测试用）
set -euo pipefail

# ── 定位 vllm 安装路径（可用 VLLM_ROOT 覆盖，便于测试）─────────────────────
if [ -n "${VLLM_ROOT:-}" ]; then
  :
else
  VLLM_ROOT=$(python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))" 2>/dev/null)
fi
if [ -z "${VLLM_ROOT:-}" ] || [ ! -d "$VLLM_ROOT" ]; then
  echo "[ERROR] 找不到 vllm 安装路径 (可用 VLLM_ROOT=/path 指定)"
  exit 1
fi
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MOD_DIR="$SCRIPT_DIR/vllm_patches/modified"
ADD_DIR="$SCRIPT_DIR/vllm_patches/added"
DRY_RUN="${DRY_RUN:-0}"

echo "Target vllm : $VLLM_ROOT"
echo "Dry-run     : $DRY_RUN"
echo ""

# ── 文件清单（相对 vllm 根目录）────────────────────────────────────────────
# 被修改的原有文件（整文件覆盖，先 .bak 备份）
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
# 纯新增文件（直接新增）
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

# ──────────────────────────────────────────────────────────────────────────────
# 1. Modified files — 先备份 .bak 再整文件覆盖
# ──────────────────────────────────────────────────────────────────────────────
echo "=== [1/3] 覆盖被修改的原有文件 ==="
for rel in "${MODIFIED_FILES[@]}"; do
  src="$MOD_DIR/$rel"
  target="$VLLM_ROOT/$rel"
  if [ ! -f "$src" ]; then
    echo "  ✗ [ERROR] patch 源文件缺失: $src"; exit 1
  fi
  if [ ! -f "$target" ]; then
    echo "  ✗ [ERROR] 目标文件不存在: $target"; exit 1
  fi
  if diff -q "$src" "$target" >/dev/null 2>&1; then
    echo "  ⊘ [SKIP]   $rel (已是最新版本)"
    continue
  fi
  if [ "$DRY_RUN" = "1" ]; then
    echo "  ~ [DRY]    $rel (将覆盖, 备份 .bak)"
    continue
  fi
  # 只在首次覆盖时创建 .bak，保留最初原始版，避免二次覆盖冲掉原始备份
  if [ ! -f "${target}.bak" ]; then
    cp "$target" "${target}.bak"
  fi
  cp "$src" "$target"
  echo "  ✓ [MODIFY] $rel"
done

# ──────────────────────────────────────────────────────────────────────────────
# 2. Added files — 直接新增
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== [2/3] 新增 DSpark 文件 ==="
for rel in "${ADDED_FILES[@]}"; do
  src="$ADD_DIR/$rel"
  target="$VLLM_ROOT/$rel"
  if [ ! -f "$src" ]; then
    echo "  ✗ [ERROR] patch 源文件缺失: $src"; exit 1
  fi
  if [ -f "$target" ] && diff -q "$src" "$target" >/dev/null 2>&1; then
    echo "  ⊘ [SKIP]   $rel (已存在且一致)"
    continue
  fi
  if [ "$DRY_RUN" = "1" ]; then
    echo "  ~ [DRY]    $rel (将新增)"
    continue
  fi
  mkdir -p "$(dirname "$target")"
  cp "$src" "$target"
  echo "  ✓ [ADD]    $rel"
done

# ──────────────────────────────────────────────────────────────────────────────
# 3. 语法检查 (ast.parse)
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== [3/3] 语法检查 (ast.parse) ==="
ERR=0
ALL_FILES=( "${MODIFIED_FILES[@]}" "${ADDED_FILES[@]}" )
for rel in "${ALL_FILES[@]}"; do
  target="$VLLM_ROOT/$rel"
  # dry-run 时新增文件尚未落盘，改查 patch 源副本
  checkfile="$target"
  if [ ! -f "$checkfile" ]; then
    if [ "$DRY_RUN" = "1" ]; then
      checkfile="$ADD_DIR/$rel"
    else
      echo "  FAIL $rel (文件不存在)"; ERR=1; continue
    fi
  fi
  if python3 -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" "$checkfile" 2>/dev/null; then
    echo "  OK   $rel"
  else
    echo "  FAIL $rel"; ERR=1
  fi
done

echo ""
if [ "$DRY_RUN" = "1" ]; then
  echo "=== DRY-RUN 完成（未改动任何文件）==="
  exit 0
fi
if [ $ERR -eq 0 ]; then
  echo "=== Patch applied successfully ==="
  echo "被修改文件已备份为 .bak。回滚: bash $SCRIPT_DIR/revert_patch.sh"
  echo ""
  echo "⚠️  DSpark 尚未启用。启用需在启动参数里配置 speculative_config (method=dspark),"
  echo "    且当前 backport 未测试 —— 启用前请先跑通测试。详见 README.md。"
else
  echo "=== [ERROR] 语法检查失败，请检查上方 FAIL 项 ==="
  exit 1
fi
