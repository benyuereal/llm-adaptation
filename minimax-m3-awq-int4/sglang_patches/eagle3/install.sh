#!/bin/bash
# EAGLE3 投机解码 patch 一键安装
#
# 把 3 个 sglang patch 应用到 site-packages,使 MiniMax-M3 (VL 类) 支持 EAGLE3
# 投机解码。安装后配合 start_eagle3.sh 即可启动 EAGLE3 服务。
#
# 用法:
#   bash sglang_patches/eagle3/install.sh
#
# 会自动:
#   1. 备份原文件到 sglang_backup/
#   2. 覆盖 3 个 patch 文件到 sglang site-packages
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
  rm -rf /models/.triton_cache/* ~/.triton/* /tmp/torchinductor_root 2>/dev/null || true
  echo "=== 回滚完成,已清 triton 缓存 ==="
  exit 0
fi

# ---------- 安装 ----------
echo "=== 安装 EAGLE3 patch ==="
echo "Target: $SGLANG_ROOT"
echo "Backup: $BACKUP_DIR"
echo ""

mkdir -p "$BACKUP_DIR"

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
checks = [
    ("minimax_m3_vl.py", "set_eagle3_layers_to_capture"),
    ("minimax_sparse_backend.py", "is_target_verify"),
    ("utils.py", "is_current_stream_capturing"),
]
SGLANG = "/usr/local/lib/python3.10/dist-packages/sglang/srt"
paths = {
    "minimax_m3_vl.py": f"{SGLANG}/models/minimax_m3_vl.py",
    "minimax_sparse_backend.py": f"{SGLANG}/layers/attention/minimax_sparse_backend.py",
    "utils.py": f"{SGLANG}/layers/attention/minimax_sparse_ops/common/utils.py",
}
all_ok = True
for name, marker in checks:
    try:
        with open(paths[name]) as f:
            content = f.read()
        if marker in content:
            print(f"  ✓ {name}: 含 '{marker}' (patch 已生效)")
        else:
            print(f"  ✗ {name}: 缺 '{marker}' (patch 未生效?)")
            all_ok = False
    except Exception as e:
        print(f"  ✗ {name}: 读取失败 {e}")
        all_ok = False
sys.exit(0 if all_ok else 1)
PYEOF

if [ $? -eq 0 ]; then
  echo ""
  echo "=== ✅ EAGLE3 patch 安装完成 ==="
  echo "下一步: bash $(dirname "$0")/start_eagle3.sh"
  echo "回滚:  bash $(dirname "$0")/install.sh --rollback"
else
  echo ""
  echo "=== ⚠ 验证未全通过,请检查上面的输出 ==="
  exit 1
fi
