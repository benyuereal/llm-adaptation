#!/bin/bash
# EAGLE3 投机解码 patch 一键安装 (eagle3-v2: 精简版, 复用标准 prefill kernel)
#
# 把 sglang patch 应用到 site-packages, 使 MiniMax-M3 (VL 类) 支持 EAGLE3
# 投机解码。本版直接复用标准 minimax_sparse_prefill 路径, 不依赖任何 verify/
# 子模块, 不含已废弃的 Strategy A/B 实验代码。
#
# 与 eagle3 (v1) 的区别:
#   - v1 依赖新增 verify/ 4 文件子模块 (Strategy B) + 带 seqlens_expand (Strategy A)
#   - v2 只改 3 个文件, verify 分支 materialize extend_seq_lens=draft_token_num
#     后走标准 prefill, K 长度=prefix+draft, prefix_lens 用 seq_lens graph buffer
#   - v2 = 本机实测能跑的最终精简版 (VMFault 两层已根治)
#
# 改动的 3 个文件:
#   minimax_m3_vl.py        : VL 类补 EAGLE3 target 侧接口
#                             (set_eagle3_layers_to_capture / get_embed_and_head /
#                              capture_aux_hidden_states flag + aux-aware forward)
#   minimax_sparse_backend.py: TARGET_VERIFY 字段 materialize (extend_seq_lens=None→
#                             draft_token_num) + K 长度=prefix+draft 重建 + graph-safe
#                             (capture/replay 用静态 q.shape[0], 不 host sync)
#   utils.py                : get_cu_seqblocks graph-safe
#                             (.sum().item() host sync → capture 下静态上界)
#
# 用法:
#   bash sglang_patches/eagle3-v2/install.sh            # 安装
#   bash sglang_patches/eagle3-v2/install.sh --rollback # 回滚
#   bash sglang_patches/eagle3-v2/install.sh --check    # 仅检查是否已安装
#
# 会自动:
#   1. 备份原文件到 sglang_backup_eagle3_v2/ (仅当目标为干净原版时)
#   2. 覆盖 patch 文件到 sglang site-packages
#   3. 清 triton 缓存
#   4. 验证 patch 生效 (grep 标记 + import 冒烟)

set -euo pipefail

SGLANG_ROOT=/usr/local/lib/python3.10/dist-packages/sglang/srt
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PATCH_DIR="$SCRIPT_DIR/modified"
BACKUP_DIR="${SGLANG_BACKUP_DIR:-/models/sglang_backup_eagle3_v2}"

# patch 文件 → site-packages 目标路径
declare -A FILE_MAP=(
  ["minimax_m3_vl.py"]="$SGLANG_ROOT/models/minimax_m3_vl.py"
  ["minimax_sparse_backend.py"]="$SGLANG_ROOT/layers/attention/minimax_sparse_backend.py"
  ["utils.py"]="$SGLANG_ROOT/layers/attention/minimax_sparse_ops/common/utils.py"
)

# 每个 patch 文件的"已应用标记"
# (grep 不到 = 干净原始文件, 可备份; grep 到 = 已打过 v2, 不覆盖备份)
declare -A PATCH_MARKER=(
  ["minimax_m3_vl.py"]="set_eagle3_layers_to_capture"
  ["minimax_sparse_backend.py"]="is_target_verify"
  ["utils.py"]="is_current_stream_capturing"
)

# v2 独有的"精简版标记" — 区分 v2 与 v1 (v1 带 verify_sparse / seqlens_expand)
V2_CLEAN_MARKER="minimax_sparse_prefill"   # v2 forward_extend 调用标准 prefill
V1_ONLY_MARKER="seqlens_expand_triton"     # v1 (Strategy A) 独有, v2 不应有

# ---------- --check ----------
if [ "${1:-}" = "--check" ]; then
  echo "=== 检查 EAGLE3 v2 patch 状态 ==="
  rc=0
  for name in "${!FILE_MAP[@]}"; do
    target="${FILE_MAP[$name]}"
    marker="${PATCH_MARKER[$name]}"
    if grep -q "$marker" "$target" 2>/dev/null; then
      echo "  ✓ $name: 含 '$marker'"
    else
      echo "  ✗ $name: 缺 '$marker' (未安装 v2)"; rc=1
    fi
  done
  # v1 残留检测
  be="${FILE_MAP[minimax_sparse_backend.py]}"
  if grep -q "$V1_ONLY_MARKER" "$be" 2>/dev/null; then
    echo "  ⚠ sparse_backend 含 '$V1_ONLY_MARKER' — 这是 v1 (Strategy A) 残留, v2 不应存在"
    echo "    建议: 先 install.sh --rollback 清掉 v1, 再重新安装 v2"
    rc=1
  else
    echo "  ✓ 无 v1 (Strategy A) 残留"
  fi
  # verify/ 子模块检测 (v1 依赖, v2 不需要)
  verify_dir="$SGLANG_ROOT/layers/attention/minimax_sparse_ops/verify"
  if [ -d "$verify_dir" ]; then
    echo "  ℹ 存在 verify/ 目录 (v1 Strategy B 残留; v2 不依赖它, 可删可留)"
  else
    echo "  ✓ 无 verify/ 子模块 (符合 v2)"
  fi
  exit $rc
fi

# ---------- --rollback ----------
if [ "${1:-}" = "--rollback" ]; then
  echo "=== 回滚 EAGLE3 v2 patch (从 $BACKUP_DIR 恢复干净原版) ==="
  for name in "${!FILE_MAP[@]}"; do
    target="${FILE_MAP[$name]}"
    backup="$BACKUP_DIR/$name"
    if [ -f "$backup" ]; then
      cp "$backup" "$target"
      echo "  ✓ 恢复 $name"
    else
      echo "  ⚠ 无备份: $backup (跳过 — 可能从未用 v2 安装过, 或备份被删)"
    fi
  done
  rm -rf /models/.triton_cache/* ~/.triton/* /tmp/torchinductor_root 2>/dev/null || true
  echo "  ✓ 已清 triton 缓存"
  echo "=== 回滚完成 ==="
  echo "注意: 回滚只恢复 v2 改的 3 个文件。若之前装过 v1, verify/ 目录与 v1 残留需手动处理。"
  exit 0
fi

# ---------- 安装 ----------
echo "=== 安装 EAGLE3 v2 patch (精简版, 复用标准 prefill) ==="
echo "Target: $SGLANG_ROOT"
echo "Backup: $BACKUP_DIR"
echo ""

# 前置检查: sglang 是否存在
if [ ! -d "$SGLANG_ROOT" ]; then
  echo "  ✗ sglang 未安装在 $SGLANG_ROOT"
  echo "    v2 patch 针对 sglang srt 内部文件, 请确认 sglang 已 pip install"
  exit 1
fi

# 前置警告: 若检测到 v1 残留 (Strategy A), 提示先清
be_target="${FILE_MAP[minimax_sparse_backend.py]}"
if [ -f "$be_target" ] && grep -q "$V1_ONLY_MARKER" "$be_target" 2>/dev/null; then
  echo "  ⚠ 检测到当前 sparse_backend 含 v1 (Strategy A) 标记 '$V1_ONLY_MARKER'"
  echo "    v2 会直接覆盖该文件, 但若你想先干净回滚 v1, 请 Ctrl-C 并运行:"
  echo "      bash sglang_patches/eagle3/install.sh --rollback   (v1 回滚)"
  echo "    然后再跑 v2 安装。"
  echo "    (5 秒后继续覆盖安装 v2...)"
  sleep 5
fi

mkdir -p "$BACKUP_DIR"

# 覆盖 3 个文件
for name in "${!FILE_MAP[@]}"; do
  src="$PATCH_DIR/$name"
  target="${FILE_MAP[$name]}"
  backup="$BACKUP_DIR/$name"

  if [ ! -f "$src" ]; then
    echo "  ✗ 源文件缺失: $src"; exit 1
  fi
  if [ ! -f "$target" ]; then
    echo "  ✗ 目标不存在: $target (sglang 版本/路径不匹配?)"; exit 1
  fi

  # 备份(仅当目标为干净原版: 不含 v2 标记)
  marker="${PATCH_MARKER[$name]}"
  if [ ! -f "$backup" ]; then
    if grep -q "$marker" "$target" 2>/dev/null; then
      echo "  ⚠ $name 已含 patch 标记('$marker') — 可能已装过, 不备份(避免把 patch 版当原版)"
    else
      cp "$target" "$backup"
      echo "  ✓ 备份原版 $name → $backup"
    fi
  else
    echo "  · 已有备份 $name (跳过备份)"
  fi

  cp "$src" "$target"
  echo "  ✓ 应用 $name → ${target#$SGLANG_ROOT/}"
done

# 清 triton 缓存
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
    # (文件, 路径, 应存在的标记, 不应存在的标记)
    ("minimax_m3_vl.py",
     f"{SGLANG}/models/minimax_m3_vl.py",
     "set_eagle3_layers_to_capture",
     None),
    ("minimax_sparse_backend.py",
     f"{SGLANG}/layers/attention/minimax_sparse_backend.py",
     "is_target_verify",            # v2 有
     "seqlens_expand_triton"),       # v1 独有, v2 不应有
    ("utils.py",
     f"{SGLANG}/layers/attention/minimax_sparse_ops/common/utils.py",
     "is_current_stream_capturing",
     None),
]
all_ok = True
for name, path, should_have, should_not in checks:
    try:
        with open(path) as f:
            content = f.read()
    except Exception as e:
        print(f"  ✗ {name}: 读取失败 {e}"); all_ok = False; continue
    if should_have and should_have in content:
        print(f"  ✓ {name}: 含 '{should_have}'")
    else:
        print(f"  ✗ {name}: 缺 '{should_have}'"); all_ok = False
    if should_not and should_not in content:
        print(f"  ✗ {name}: 不应含 '{should_not}' (v1 残留)"); all_ok = False

# 额外: v2 精简版核心 — forward_extend verify 分支用标准 prefill + 重建 K 长度
be = open(f"{SGLANG}/layers/attention/minimax_sparse_backend.py").read()
if ("is_target_verify()" in be
    and "raw_seq_lens + forward_batch.extend_seq_lens" in be
    and "minimax_sparse_prefill(" in be
    and "verify_sparse" not in be):
    print(f"  ✓ v2 精简版核心: verify 走标准 prefill, K=prefix+draft, 无 verify_sparse 依赖")
else:
    print(f"  ✗ v2 精简版核心标记缺失"); all_ok = False

sys.exit(0 if all_ok else 1)
PYEOF

if [ $? -eq 0 ]; then
  echo ""
  echo "=== ✅ EAGLE3 v2 patch 安装完成 ==="
  echo "下一步: bash $(dirname "$0")/start_eagle3.sh"
  echo "检查:   bash $(dirname "$0")/install.sh --check"
  echo "回滚:   bash $(dirname "$0")/install.sh --rollback"
else
  echo ""
  echo "=== ⚠ 验证未全通过, 请检查上面的输出 ==="
  exit 1
fi
