# MiniMax-M3 AWQ DCU Patch for SGLang

This patch contains modifications to SGLang (v0.0.0.dev12695+g1df793665) to support
MiniMax-M3 AWQ quantized model inference on DCU (HIP/ROCm) hardware.

## Files Modified

| File (relative to `sglang/srt/`) | Description |
|---|---|
| `layers/quantization/compressed_tensors/compressed_tensors.py` | Added `symmetric=weight_quant.symmetric` parameter |
| `layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py` | ZP reshape fix (`permute(0,2,1)`), `_process_weights_hip` and `_apply_weights_hip` modifications |
| `layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py` | Overflow fix, per-expert fill, diagnostics |
| `models/minimax_m3.py` | `mlp_layer_types` support, dense layer `quant_config=None`, shared_experts fix, attention quant_config logic, `_is_layer_sparse` helper, **lightning indexer fix Bug A (use `layer_types` for sparse-layer detection) + Bug B (use `get_minimax_sparse_attention_config` for `disable_index_value` so model/backend share the same `sparse_cfg`)** |
| `models/minimax_m3_vl.py` | `out_proj->o_proj` mapping, `weight_packed` fallback, load diagnostics |
| `configs/model_config.py` | **Lightning indexer fix Bug A**: `get_minimax_sparse_attention_config` adds `_get()` helper for dict/PreTrainedConfig `text_config` and injects `layer_types`; `get_minimax_sparse_layer_ids` prefers `layer_types` over all-zero `sparse_attention_freq`. **Bug B**: `get_minimax_sparse_disable_value_layer_ids` defaults to disabling `index_v/o_proj` on all sparse layers when `sparse_disable_index_value` is absent (checkpoint has no index_v/o weights) |
| `layers/moe/moe_runner/triton_utils/fused_moe.py` | HIP combine changed to `torch.sum`, diagnostics |

## Lightning Indexer Fix (Bug A + Bug B)

The lightning indexer fix is delivered as both modified files and a standalone
incremental patch:

- `modified/minimax_m3.py` / `modified/model_config.py` — full adapted files (used by `apply_patch.sh`)
- `modified/minimax_m3-indexer-fix.patch` — incremental patch for Bug A+B only,
  applies on top of a K100_AI-adapted sglang baseline (paths `a/sglang/srt/...`,
  apply from `dist-packages/` with `-p1`). Contains **no** trace probes.

Apply the incremental patch:

```bash
cd /usr/local/lib/python3.10/dist-packages/
patch -p1 --dry-run < /path/to/minimax-m3-indexer-fix.patch   # dry-run first
patch -p1 < /path/to/minimax-m3-indexer-fix.patch
```

See `docs/m3-indexer-fix-summary.md` for the full root-cause analysis.

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

## Method 2: Direct File Replacement (recommended)

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
