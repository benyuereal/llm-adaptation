#!/bin/bash
# MiniMax-M3 AWQ INT4 K100AI DCU 适配 — 一键应用 patch
set -euo pipefail

SGLANG_ROOT=/usr/local/lib/python3.10/dist-packages/sglang/srt
TRANSFORMERS_ROOT=/usr/local/lib/python3.10/dist-packages/transformers
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PATCH_DIR="$SCRIPT_DIR/sglang_patches/modified"

echo "=== MiniMax-M3 AWQ INT4 DCU Patch ==="
echo "Target: $SGLANG_ROOT"
echo ""

# SGLang 文件
declare -A FILE_MAP=(
  ["compressed_tensors.py"]="$SGLANG_ROOT/layers/quantization/compressed_tensors/compressed_tensors.py"
  ["compressed_tensors_wNa16.py"]="$SGLANG_ROOT/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py"
  ["compressed_tensors_wNa16_moe.py"]="$SGLANG_ROOT/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py"
  ["minimax_m3.py"]="$SGLANG_ROOT/models/minimax_m3.py"
  ["minimax_m3_vl.py"]="$SGLANG_ROOT/models/minimax_m3_vl.py"
  ["fused_moe.py"]="$SGLANG_ROOT/layers/moe/moe_runner/triton_utils/fused_moe.py"
  ["configuration_utils.py"]="$TRANSFORMERS_ROOT/configuration_utils.py"
)

for f in "${!FILE_MAP[@]}"; do
  target="${FILE_MAP[$f]}"
  if [ -f "$target" ]; then
    cp "$target" "${target}.bak" 2>/dev/null || true
    cp "$PATCH_DIR/$f" "$target"
    echo "  ✓ $f → $target"
  else
    echo "  ✗ Target not found: $target"
    exit 1
  fi
done

# MoE kernel configs (K100_AI tuned)
CONFIG_DIR="$SGLANG_ROOT/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1"
ADDED_DIR="$SCRIPT_DIR/sglang_patches/added/configs"
mkdir -p "$CONFIG_DIR"
for cfg in "$ADDED_DIR"/*.json; do
  cp "$cfg" "$CONFIG_DIR/"
  echo "  ✓ $(basename $cfg) → configs/"
done

echo ""
echo "=== Patch applied successfully ==="
echo "Start service: bash $SCRIPT_DIR/start.sh"
