# MiniMax-M3 AWQ DCU Patch for SGLang

This patch contains modifications to SGLang (v0.0.0.dev12695+g1df793665) to support
MiniMax-M3 AWQ quantized model inference on DCU (HIP/ROCm) hardware.

## Files Modified

| File (relative to `sglang/srt/`) | Description |
|---|---|
| `layers/quantization/compressed_tensors/compressed_tensors.py` | Added `symmetric=weight_quant.symmetric` parameter |
| `layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py` | ZP reshape fix (`permute(0,2,1)`), `_process_weights_hip` and `_apply_weights_hip` modifications |
| `layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py` | Overflow fix, per-expert fill, diagnostics |
| `models/minimax_m3.py` | `mlp_layer_types` support, dense layer `quant_config=None`, shared_experts fix, attention quant_config logic, `_is_layer_sparse` helper |
| `models/minimax_m3_vl.py` | `out_proj->o_proj` mapping, `weight_packed` fallback, load diagnostics |
| `layers/moe/moe_runner/triton_utils/fused_moe.py` | HIP combine changed to `torch.sum`, diagnostics |

## Method 1: Apply Unified Patch

```bash
# Navigate to the sglang package directory
cd /usr/local/lib/python3.10/dist-packages/

# Apply the patch (dry-run first)
patch -p1 --dry-run < /path/to/minimax-m3-awq-dcu.patch

# Apply for real
patch -p1 < /path/to/minimax-m3-awq-dcu.patch
```

Note: The patch uses paths like `a/sglang/srt/...` and `b/sglang/srt/...`, so apply from
the `dist-packages/` (or `site-packages/`) directory with `-p1`.

## Method 2: Direct File Replacement

Copy the modified files directly over the installed package files:

```bash
SGLANG_SRT="/usr/local/lib/python3.10/dist-packages/sglang/srt"

cp files/layers/quantization/compressed_tensors/compressed_tensors.py \
   "$SGLANG_SRT/layers/quantization/compressed_tensors/compressed_tensors.py"

cp files/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py \
   "$SGLANG_SRT/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py"

cp files/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py \
   "$SGLANG_SRT/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py"

cp files/models/minimax_m3.py "$SGLANG_SRT/models/minimax_m3.py"
cp files/models/minimax_m3_vl.py "$SGLANG_SRT/models/minimax_m3_vl.py"

cp files/layers/moe/moe_runner/triton_utils/fused_moe.py \
   "$SGLANG_SRT/layers/moe/moe_runner/triton_utils/fused_moe.py"
```

## Compatibility

- **Base SGLang version**: 0.0.0.dev12695+g1df793665 (built 2026-06-05)
- **Target hardware**: DCU (HIP/ROCm)
- **Model**: MiniMax-M3 with AWQ quantization
