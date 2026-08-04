#!/bin/bash
# EAGLE3 投机解码 patch 一键安装 (Path 1 Strategy B + graph-safety 修复)
#
# 把 sglang patch 应用到 site-packages,使 MiniMax-M3 (VL 类) 支持 EAGLE3
# 投机解码。安装后配合 start_eagle3.sh 即可启动 EAGLE3 服务。
#
# 本版改动 (替代已废弃的 Strategy A):
#   - 新增 verify/ 3 个 graph-safe verify kernel (固定上界 score buffer + OOB 双保险)
#   - minimax_sparse_backend.py: target_verify 走新 verify kernel;
#     forward_extend verify 分支 graph-safety 修复 (绕过临时 extend_seq_lens,
#     改用 seq_lens graph buffer + Python int + torch.arange)
#   - 修复 bs>=并发阈值 VMFault: 旧版用临时 extend_seq_lens 算 cu_seqlens,
#     capture 后 GC,replay 读失效地址 → 垃圾 → 越界
#
# 用法:
#   bash sglang_patches/eagle3/install.sh
#
# 会自动:
#   1. 备份原文件到 sglang_backup/
#   2. 覆盖 patch 文件到 sglang site-packages (含新增 verify/ 目录)
#   3. 清 triton 缓存
#   4. 验证 patch 生效
#
# 回滚:
#   bash sglang_patches/eagle3/install.sh --rollback

set -euo pipefail

SGLANG_ROOT=/usr/local/lib/python3.10/dist-packages/sglang/srt
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PATCH_DIR="$SCRIPT_DIR/modified"
BACKUP_DIR="${SGLANG_BACKUP_DIR:-/models/sglang_backup_eagle3}"

# patch 文件 → site-packages 目标路径
declare -A FILE_MAP=(
  ["minimax_m3_vl.py"]="$SGLANG_ROOT/models/minimax_m3_vl.py"
  ["minimax_sparse_backend.py"]="$SGLANG_ROOT/layers/attention/minimax_sparse_backend.py"
  ["utils.py"]="$SGLANG_ROOT/layers/attention/minimax_sparse_ops/common/utils.py"
)

# 每个 patch 文件的"已应用标记"(grep 不到 = 干净原始文件,可备份;grep 到 = 已打过,不覆盖备份)
declare -A PATCH_MARKER=(
  ["minimax_m3_vl.py"]="set_eagle3_layers_to_capture"
  ["minimax_sparse_backend.py"]="is_target_verify"
  ["utils.py"]="is_current_stream_capturing"
)

# Strategy B 新增 verify kernel 目录 (3 文件 + __init__.py)
VERIFY_SRC_DIR="$PATCH_DIR/verify"
VERIFY_DST_DIR="$SGLANG_ROOT/layers/attention/minimax_sparse_ops/verify"
VERIFY_FILES=("__init__.py" "flash_with_topk_idx.py" "topk_sparse.py" "verify_sparse.py")

# ---------- 回滚 ----------
if [ "${1:-}" = "--rollback" ]; then
  echo "=== 回滚 EAGLE3 patch (从 $BACKUP_DIR 恢复) ==="
  for name in "${!FILE_MAP[@]}"; do
    target="${FILE_MAP[$name]}"
    backup="$BACKUP_DIR/$name"
    if [ -f "$backup" ]; then
      cp "$backup" "$target"
      echo "  ✓ 恢复 $name"
    else
      echo "  ⚠ 无备份: $backup (跳过)"
    fi
  done
  # 删除新增的 verify/ 目录 (Strategy B 新增, 原始 sglang 无此目录)
  if [ -d "$VERIFY_DST_DIR" ]; then
    rm -rf "$VERIFY_DST_DIR"
    echo "  ✓ 删除 verify/ 目录 (Strategy B 新增)"
  fi
  rm -rf /models/.triton_cache/* ~/.triton/* /tmp/torchinductor_root 2>/dev/null || true
  echo "=== 回滚完成,已清 triton 缓存 ==="
  exit 0
fi

# ---------- 安装 ----------
echo "=== 安装 EAGLE3 patch (Strategy B + graph-safety) ==="
echo "Target: $SGLANG_ROOT"
echo "Backup: $BACKUP_DIR"
echo ""

mkdir -p "$BACKUP_DIR"

# 1) 覆盖已有文件 (minimax_m3_vl.py / minimax_sparse_backend.py / utils.py)
for name in "${!FILE_MAP[@]}"; do
  src="$PATCH_DIR/$name"
  target="${FILE_MAP[$name]}"
  backup="$BACKUP_DIR/$name"

  if [ ! -f "$src" ]; then
    echo "  ✗ 源文件缺失: $src"
    exit 1
  fi
  if [ ! -f "$target" ]; then
    echo "  ✗ 目标不存在: $target (sglang 版本不匹配?)"
    exit 1
  fi

  # 备份(仅当目标文件是"干净的原始版"时才备份,避免把已打 patch 的版本当原始版备份)
  marker="${PATCH_MARKER[$name]}"
  if [ ! -f "$backup" ]; then
    if grep -q "$marker" "$target" 2>/dev/null; then
      echo "  ⚠ $name 已含 patch 标记('$marker'),可能已安装过 — 不备份(避免把 patch 版当原始版)"
    else
      cp "$target" "$backup"
      echo "  ✓ 备份 $name → $backup"
    fi
  else
    echo "  · 已有备份 $name (跳过备份)"
  fi

  # 覆盖
  cp "$src" "$target"
  echo "  ✓ 应用 $name → $target"
done

# 2) 新增 verify/ 目录 (Strategy B graph-safe verify kernels)
echo ""
echo "=== 安装 verify/ kernel 目录 (Strategy B 新增) ==="
if [ ! -d "$VERIFY_SRC_DIR" ]; then
  echo "  ✗ verify 源目录缺失: $VERIFY_SRC_DIR"
  exit 1
fi
mkdir -p "$VERIFY_DST_DIR"
for vf in "${VERIFY_FILES[@]}"; do
  if [ ! -f "$VERIFY_SRC_DIR/$vf" ]; then
    echo "  ✗ 缺失: $VERIFY_SRC_DIR/$vf"
    exit 1
  fi
  cp "$VERIFY_SRC_DIR/$vf" "$VERIFY_DST_DIR/$vf"
  echo "  ✓ $vf → $VERIFY_DST_DIR/$vf"
done

# 清 triton 缓存(改过 kernel 后必须清,否则用旧编译结果)
echo ""
echo "=== 清理 triton 缓存 ==="
rm -rf /models/.triton_cache/* ~/.triton/* /tmp/torchinductor_root 2>/dev/null || true
echo "  ✓ 已清"

# 验证
echo ""
echo "=== 验证 patch 生效 ==="
python3 - <<'PYEOF'
import sys
SGLANG = "/usr/local/lib/python3.10/dist-packages/sglang/srt"
checks = [
    ("minimax_m3_vl.py", f"{SGLANG}/models/minimax_m3_vl.py", "set_eagle3_layers_to_capture"),
    ("minimax_sparse_backend.py", f"{SGLANG}/layers/attention/minimax_sparse_backend.py", "minimax_sparse_verify_prefill"),
    ("utils.py", f"{SGLANG}/layers/attention/minimax_sparse_ops/common/utils.py", "is_current_stream_capturing"),
]
# Strategy B 新增 verify kernel
verify_checks = [
    ("verify/__init__.py", f"{SGLANG}/layers/attention/minimax_sparse_ops/verify/__init__.py", "minimax_sparse_verify_prefill"),
    ("verify/flash_with_topk_idx.py", f"{SGLANG}/layers/attention/minimax_sparse_ops/verify/flash_with_topk_idx.py", "max_seqblock_k_upper"),
    ("verify/topk_sparse.py", f"{SGLANG}/layers/attention/minimax_sparse_ops/verify/topk_sparse.py", "max_seqblock_k_upper"),
    ("verify/verify_sparse.py", f"{SGLANG}/layers/attention/minimax_sparse_ops/verify/verify_sparse.py", "minimax_sparse_verify_prefill"),
]
all_ok = True
for name, path, marker in checks + verify_checks:
    try:
        with open(path) as f:
            content = f.read()
        if marker in content:
            print(f"  ✓ {name}: 含 '{marker}' (patch 已生效)")
        else:
            print(f"  ✗ {name}: 缺 '{marker}' (patch 未生效?)")
            all_ok = False
    except Exception as e:
        print(f"  ✗ {name}: 读取失败 {e}")
        all_ok = False

# 额外验证: graph-safety 修复标记 (forward_extend verify 分支用 arange, 不用 extend_seq_lens)
backend_path = f"{SGLANG}/layers/attention/minimax_sparse_backend.py"
with open(backend_path) as f:
    be = f.read()
if "torch.arange(" in be and "self.num_draft_tokens" in be and "is_verify = forward_batch.forward_mode.is_target_verify()" in be:
    print(f"  ✓ graph-safety 修复: forward_extend verify 分支用 arange + num_draft_tokens (绕过临时 extend_seq_lens)")
else:
    print(f"  ✗ graph-safety 修复标记缺失")
    all_ok = False

sys.exit(0 if all_ok else 1)
PYEOF

if [ $? -eq 0 ]; then
  echo ""
  echo "=== ✅ EAGLE3 patch 安装完成 (Strategy B + graph-safety) ==="
  echo "下一步: bash $(dirname "$0")/start_eagle3.sh"
  echo "回滚:  bash $(dirname "$0")/install.sh --rollback"
else
  echo ""
  echo "=== ⚠ 验证未全通过,请检查上面的输出 ==="
  exit 1
fi
