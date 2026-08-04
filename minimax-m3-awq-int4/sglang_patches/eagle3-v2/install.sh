#!/bin/bash
# EAGLE3 投机解码 patch 一键安装 (eagle3-v2: 专用 verify kernel + graph buffer 根治 VMFault)
#
# 把 sglang patch 应用到 site-packages, 使 MiniMax-M3 (VL 类) 支持 EAGLE3
# 投机解码。本版用专用 verify_prefill kernel + 预分配 graph buffer, 两层根治
# EAGLE3 TARGET_VERIFY 在 cuda graph 下的 KERNEL VMFault。详见 README.md。
#
# === VMFault 两层根因与修复 (详见 README.md) ===
#   阶段1 — score buffer 动态尺寸越界:
#     prefill kernel 的 score 第3维 = cdiv(max_seqlen_k, block_size_k),
#     capture 时 max_seqlen_k=dummy(5)→1, replay 时 =真实(~2000)→32,
#     graph 锁了 capture 的尺寸 1, replay 写 score[...,1..31] 越界 → VMFault。
#     修复: score 第3维用恒定上界 max_seqblock_k_upper = cdiv(context_len+D, bsk)
#           (capture/replay 形状恒定), kernel 内仍按真实 seq_lens 做 causal。
#   阶段2 — cu_seqlens/seq_lens 临时张量地址漂移 (真正根因):
#     旧 forward_extend 用 torch.cat / a+b 现场构造 cu_seqlens/seq_lens, 每次
#     new 临时张量 → data_ptr capture≠replay → graph 锁 capture 地址, replay
#     读到别处 → garbage/VMFault。(探针 EAGLE3_VERIFY_PROBE 已确认。)
#     修复: 新建 verify/ 专用 kernel (minimax_sparse_verify_prefill), 并在
#           init_cuda_graph_state 预分配 3 个 graph buffer (地址固定),
#           capture/replay 写入同一 buffer (data_ptr 不变) → graph-safe。
#
# 改动的 12 个文件:
#   --- EAGLE3 target 侧接口 (1) ---
#   minimax_m3_vl.py        : VL 类补 EAGLE3 target 侧接口
#                             (set_eagle3_layers_to_capture / get_embed_and_head /
#                              capture_aux_hidden_states flag + aux-aware forward)
#   --- verify 路由 + graph buffer 根治 (1) ---
#   minimax_sparse_backend.py: TARGET_VERIFY 路由到 _forward_verify (专用 verify
#                             kernel); init_cuda_graph_state 预分配 3 个 graph
#                             buffer (cu_seqlens/extend_seq_lens/seq_lens, int32);
#                             capture/replay 写同一 buffer (data_ptr 不变);
#                             _max_seqlen_k = max_seqblock_k_upper*bsk (恒定,
#                             仅过 score 断言); has_graph_buf eager 分支 (bs>
#                             cuda_graph_max_bs 时 torch.cat 构造临时张量)
#   --- 专用 verify kernel 子模块 (4) ---
#   verify/__init__.py      : export minimax_sparse_verify_prefill
#   verify/verify_sparse.py : verify 入口 (step1 score 固定上界 + step3 OOB 双保险)
#   verify/flash_with_topk_idx.py: verify step1 kernel, score 第3维用
#                             max_seqblock_k_upper (恒定), grid 用 max_seqlen_q(=D=4)
#   verify/topk_sparse.py  : verify step3 kernel (gqa share sparse, OOB 双保险)
#   --- score 上界透传 (阶段1 修复, 2) ---
#   minimax_sparse.py       : minimax_sparse_prefill 透传 max_seqblock_k_upper 到 step1
#   prefill_flash_with_topk_idx.py: score 第3维从动态 cdiv(max_seqlen_k,bsk) 改用
#                             恒定上界 max_seqblock_k_upper (capture/replay 形状恒定)
#   --- graph-safe 辅助 (1) ---
#   utils.py                : get_cu_seqblocks graph-safe
#                             (.sum().item() host sync → capture 下静态上界)
#   --- DCU/HIP + cuda-graph 兼容性 (3) ---
#   prefill_topk_sparse.py  : DCU 64KB shared-mem 限制 → num_stages=1 on HIP
#                             (原版写死 2,3 → ~68KB 超 65536, prefill 崩 OutOfResources)
#   decode_topk_sparse.py   : 同上, decode kernel num_stages=1 on HIP
#   decode_flash_with_topk_idx.py: cuda-graph capture 期间禁用 side_stream fork
#                             (原版无条件 fork 新 stream, capture 期间非法 → 多 batch 崩)
#
# 用法:
#   bash sglang_patches/eagle3-v2/install.sh            # 安装
#   bash sglang_patches/eagle3-v2/install.sh --rollback # 回滚
#   bash sglang_patches/eagle3-v2/install.sh --check    # 仅检查是否已安装
#
# 会自动:
#   1. 备份原文件到 sglang_backup_eagle3_v2/ (仅当目标为干净原版时)
#   2. 覆盖 patch 文件到 sglang site-packages (含新建 verify/ 子模块)
#   3. 清 triton 缓存
#   4. 验证 patch 生效 (grep 标记 + import 冒烟)

set -euo pipefail

SGLANG_ROOT=/usr/local/lib/python3.10/dist-packages/sglang/srt
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PATCH_DIR="$SCRIPT_DIR/modified"
BACKUP_DIR="${SGLANG_BACKUP_DIR:-/models/sglang_backup_eagle3_v2}"
VERIFY_TARGET="$SGLANG_ROOT/layers/attention/minimax_sparse_ops/verify"

# patch 文件 → site-packages 目标路径
declare -A FILE_MAP=(
  ["minimax_m3_vl.py"]="$SGLANG_ROOT/models/minimax_m3_vl.py"
  ["minimax_sparse_backend.py"]="$SGLANG_ROOT/layers/attention/minimax_sparse_backend.py"
  ["utils.py"]="$SGLANG_ROOT/layers/attention/minimax_sparse_ops/common/utils.py"
  ["minimax_sparse.py"]="$SGLANG_ROOT/layers/attention/minimax_sparse_ops/minimax_sparse.py"
  ["prefill_topk_sparse.py"]="$SGLANG_ROOT/layers/attention/minimax_sparse_ops/prefill/topk_sparse.py"
  ["prefill_flash_with_topk_idx.py"]="$SGLANG_ROOT/layers/attention/minimax_sparse_ops/prefill/flash_with_topk_idx.py"
  ["decode_topk_sparse.py"]="$SGLANG_ROOT/layers/attention/minimax_sparse_ops/decode/topk_sparse.py"
  ["decode_flash_with_topk_idx.py"]="$SGLANG_ROOT/layers/attention/minimax_sparse_ops/decode/flash_with_topk_idx.py"
)

# verify/ 子模块文件 → 目标路径 (新建目录, 需 mkdir)
declare -A VERIFY_MAP=(
  ["__init__.py"]="$VERIFY_TARGET/__init__.py"
  ["verify_sparse.py"]="$VERIFY_TARGET/verify_sparse.py"
  ["flash_with_topk_idx.py"]="$VERIFY_TARGET/flash_with_topk_idx.py"
  ["topk_sparse.py"]="$VERIFY_TARGET/topk_sparse.py"
)

# 每个 patch 文件的"已应用标记"
# (grep 不到 = 干净原始文件, 可备份; grep 到 = 已打过 v2, 不覆盖备份)
declare -A PATCH_MARKER=(
  ["minimax_m3_vl.py"]="set_eagle3_layers_to_capture"
  ["minimax_sparse_backend.py"]="_forward_verify"
  ["utils.py"]="is_current_stream_capturing"
  ["minimax_sparse.py"]="max_seqblock_k_upper"
  ["prefill_topk_sparse.py"]="_PREFILL_NUM_STAGES"
  ["prefill_flash_with_topk_idx.py"]="score_max_seqblock_k"
  ["decode_topk_sparse.py"]="_NUM_STAGES = [1] if _is_hip()"
  ["decode_flash_with_topk_idx.py"]="is_stream_capturing"
)

# verify/ 子模块文件标记 (新建文件, 存在即视为已装)
declare -A VERIFY_MARKER=(
  ["__init__.py"]="minimax_sparse_verify_prefill"
  ["verify_sparse.py"]="def minimax_sparse_verify_prefill"
  ["flash_with_topk_idx.py"]="max_seqblock_k_upper"
  ["topk_sparse.py"]="flash_verify_prefill_with_gqa_share_sparse"
)

# ---------- --check ----------
if [ "${1:-}" = "--check" ]; then
  echo "=== 检查 EAGLE3 v2 patch 状态 ==="
  rc=0
  for name in "${!FILE_MAP[@]}"; do
    target="${FILE_MAP[$name]}"
    marker="${PATCH_MARKER[$name]}"
    if grep -qF "$marker" "$target" 2>/dev/null; then
      echo "  ✓ $name: 含 '$marker'"
    else
      echo "  ✗ $name: 缺 '$marker' (未安装 v2)"; rc=1
    fi
  done
  # verify/ 子模块检查
  echo ""
  echo "  --- verify/ 子模块 (专用 verify kernel) ---"
  if [ ! -d "$VERIFY_TARGET" ]; then
    echo "  ✗ verify/ 目录不存在 (未安装 v2 verify kernel)"; rc=1
  else
    for name in "${!VERIFY_MAP[@]}"; do
      target="${VERIFY_MAP[$name]}"
      marker="${VERIFY_MARKER[$name]}"
      if grep -qF "$marker" "$target" 2>/dev/null; then
        echo "  ✓ verify/$name: 含 '$marker'"
      else
        echo "  ✗ verify/$name: 缺 '$marker'"; rc=1
      fi
    done
  fi
  # graph buffer 根治标记 (阶段2 核心)
  echo ""
  echo "  --- VMFault 阶段2 根治: graph buffer ---"
  be="${FILE_MAP[minimax_sparse_backend.py]}"
  for m in "_verify_cu_seqlens_buf" "_verify_extend_seq_lens_buf" \
           "_verify_seq_lens_buf" "has_graph_buf" \
           "init_cuda_graph_state"; do
    if grep -qF "$m" "$be" 2>/dev/null; then
      echo "  ✓ backend 含 '$m'"
    else
      echo "  ✗ backend 缺 '$m' (阶段2 根治缺失)"; rc=1
    fi
  done
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
  # verify/ 子模块: v2 新建, 原版无此目录 → 回滚即删除
  if [ -d "$VERIFY_TARGET" ]; then
    rm -rf "$VERIFY_TARGET"
    echo "  ✓ 删除 verify/ 子模块 (v2 新建, 原版无此目录)"
  fi
  rm -rf /models/.triton_cache/* ~/.triton/* /tmp/torchinductor_root 2>/dev/null || true
  echo "  ✓ 已清 triton 缓存"
  echo "=== 回滚完成 ==="
  exit 0
fi

# ---------- 安装 ----------
echo "=== 安装 EAGLE3 v2 patch (专用 verify kernel + graph buffer 根治 VMFault) ==="
echo "Target: $SGLANG_ROOT"
echo "Backup: $BACKUP_DIR"
echo ""

# 前置检查: sglang 是否存在
if [ ! -d "$SGLANG_ROOT" ]; then
  echo "  ✗ sglang 未安装在 $SGLANG_ROOT"
  echo "    v2 patch 针对 sglang srt 内部文件, 请确认 sglang 已 pip install"
  exit 1
fi

mkdir -p "$BACKUP_DIR"

# 覆盖 8 个既有文件
echo "--- 8 个既有文件 ---"
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
    if grep -qF "$marker" "$target" 2>/dev/null; then
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

# 新建 verify/ 子模块 (4 文件)
echo ""
echo "--- verify/ 子模块 (专用 verify kernel, 阶段2 根治) ---"
mkdir -p "$VERIFY_TARGET"
for name in "${!VERIFY_MAP[@]}"; do
  src="$PATCH_DIR/verify/$name"
  target="${VERIFY_MAP[$name]}"
  if [ ! -f "$src" ]; then
    echo "  ✗ 源文件缺失: $src"; exit 1
  fi
  cp "$src" "$target"
  echo "  ✓ 应用 verify/$name → ${target#$SGLANG_ROOT/}"
done
# 清 verify 目录可能残留的 __pycache__ (旧字节码)
rm -rf "$VERIFY_TARGET/__pycache__" 2>/dev/null || true

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
VERIFY = f"{SGLANG}/layers/attention/minimax_sparse_ops/verify"
checks = [
    # (文件, 路径, 应存在的标记, 不应存在的标记)
    ("minimax_m3_vl.py",
     f"{SGLANG}/models/minimax_m3_vl.py",
     "set_eagle3_layers_to_capture", None),
    # VMFault 阶段2 根治: graph buffer + verify routing
    ("minimax_sparse_backend.py",
     f"{SGLANG}/layers/attention/minimax_sparse_backend.py",
     "_forward_verify", None),
    ("utils.py",
     f"{SGLANG}/layers/attention/minimax_sparse_ops/common/utils.py",
     "is_current_stream_capturing", None),
    # VMFault 阶段1 修复: score buffer 恒定上界
    ("minimax_sparse.py",
     f"{SGLANG}/layers/attention/minimax_sparse_ops/minimax_sparse.py",
     "max_seqblock_k_upper", None),
    ("prefill_flash_with_topk_idx.py",
     f"{SGLANG}/layers/attention/minimax_sparse_ops/prefill/flash_with_topk_idx.py",
     "score_max_seqblock_k", None),
    # DCU/HIP + cuda-graph 兼容性 kernel 修复
    ("prefill_topk_sparse.py",
     f"{SGLANG}/layers/attention/minimax_sparse_ops/prefill/topk_sparse.py",
     "_PREFILL_NUM_STAGES", None),
    ("decode_topk_sparse.py",
     f"{SGLANG}/layers/attention/minimax_sparse_ops/decode/topk_sparse.py",
     "_NUM_STAGES = [1] if _is_hip()", None),
    ("decode_flash_with_topk_idx.py",
     f"{SGLANG}/layers/attention/minimax_sparse_ops/decode/flash_with_topk_idx.py",
     "is_stream_capturing", None),
    # verify/ 子模块 (阶段2 根治专用 kernel)
    ("verify/__init__.py",
     f"{VERIFY}/__init__.py",
     "minimax_sparse_verify_prefill", None),
    ("verify/verify_sparse.py",
     f"{VERIFY}/verify_sparse.py",
     "def minimax_sparse_verify_prefill", None),
    ("verify/flash_with_topk_idx.py",
     f"{VERIFY}/flash_with_topk_idx.py",
     "max_seqblock_k_upper", None),
    ("verify/topk_sparse.py",
     f"{VERIFY}/topk_sparse.py",
     "flash_verify_prefill_with_gqa_share_sparse", None),
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
        print(f"  ✗ {name}: 不应含 '{should_not}'"); all_ok = False

# 阶段2 核心: graph buffer 预分配 + capture/replay 写同一 buffer
be = open(f"{SGLANG}/layers/attention/minimax_sparse_backend.py").read()
graph_buf_markers = [
    "_verify_cu_seqlens_buf",          # cu_seqlens graph buffer
    "_verify_extend_seq_lens_buf",     # extend_seq_lens graph buffer
    "_verify_seq_lens_buf",            # seq_lens graph buffer
    "has_graph_buf",                   # eager 分支
    "minimax_sparse_verify_prefill",   # 路由到 verify kernel
]
missing = [m for m in graph_buf_markers if m not in be]
if not missing:
    print(f"  ✓ VMFault 阶段2 根治: 3 个 graph buffer + eager 分支 + verify routing 齐全")
else:
    print(f"  ✗ 阶段2 根治缺失: {missing}"); all_ok = False

# import 冒烟: verify 子模块能 import
try:
    from sglang.srt.layers.attention.minimax_sparse_ops.verify import (
        minimax_sparse_verify_prefill,
    )
    print(f"  ✓ import 冒烟: minimax_sparse_verify_prefill 可导入")
except Exception as e:
    print(f"  ✗ import 冒烟失败: {e}"); all_ok = False

sys.exit(0 if all_ok else 1)
PYEOF

if [ $? -eq 0 ]; then
  echo ""
  echo "=== ✅ EAGLE3 v2 patch 安装完成 (12 文件: 8 既有 + 4 verify 子模块) ==="
  echo "下一步: bash $(dirname "$0")/start_eagle3.sh"
  echo "检查:   bash $(dirname "$0")/install.sh --check"
  echo "回滚:   bash $(dirname "$0")/install.sh --rollback"
else
  echo ""
  echo "=== ⚠ 验证未全通过, 请检查上面的输出 ==="
  exit 1
fi
