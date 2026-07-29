# Diagnostic code extracted from fused_moe.py
# These blocks were used to trace kernel outputs in the MoE pipeline.

# === In _fused_moe_kernel_sequence(), after first invoke_fused_moe_kernel ===
"""
    # Diagnostic: check if first kernel produced any output
    if not getattr(fused_experts_impl, '_kernel_diag_done', False):
        fused_experts_impl._kernel_diag_done = True
        import os
        rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "?"))
        torch.cuda.synchronize()
        c1 = intermediate_cache1.float()
        with open('/workspace/kernel_seq_diag.txt', 'a') as _f:
            _f.write(f"rank={rank} after_w13_kernel:\\n")
            _f.write(f"  cache1: shape={list(c1.shape)} mean={c1.mean().item():.6f} std={c1.std().item():.6f} "
                     f"min={c1.min().item():.4f} max={c1.max().item():.4f} "
                     f"nonzero={c1.abs().sum().item():.4f}\\n")
            _f.write(f"  hidden_states: shape={list(hidden_states.shape)} "
                     f"mean={hidden_states.float().mean().item():.6f}\\n")
            _f.write(f"  w1: shape={list(w1.shape)} dtype={w1.dtype}\\n")
            _f.write(f"  config={config}\\n")
            _f.write(f"  use_int4_w4a16={use_int4_w4a16} block_shape={block_shape}\\n")
            _f.write(f"  sorted_token_ids[:8]={sorted_token_ids[:8].tolist()}\\n")
            _f.write(f"  expert_ids[:8]={expert_ids[:8].tolist()}\\n")
            _f.write(f"  num_tokens_post_padded={num_tokens_post_padded.tolist()}\\n")
"""

# === In _fused_moe_kernel_sequence(), after second invoke_fused_moe_kernel ===
"""
    # Diagnostic: check w2 kernel output and combine
    if not getattr(fused_experts_impl, '_kernel_diag2_done', False):
        fused_experts_impl._kernel_diag2_done = True
        torch.cuda.synchronize()
        c2 = intermediate_cache2.float()
        c3 = intermediate_cache3.float()
        with open('/workspace/kernel_seq_diag.txt', 'a') as _f:
            _f.write(f"  after_silu cache2: mean={c2.mean().item():.6f} std={c2.std().item():.6f} nonzero={c2.abs().sum().item():.4f}\\n")
            _f.write(f"  after_w2_kernel cache3: shape={list(c3.shape)} mean={c3.mean().item():.6f} std={c3.std().item():.6f} nonzero={c3.abs().sum().item():.4f}\\n")
            _f.write(f"  out_hidden_states before combine: mean={out_hidden_states.float().mean().item():.6f}\\n")
"""

# === In _fused_moe_kernel_sequence(), HIP combine branch (elif _is_hip, else of _use_aiter) ===
"""
            if not getattr(fused_experts_impl, '_combine_diag_done', False):
                fused_experts_impl._combine_diag_done = True
                with open('/workspace/combine_diag.txt', 'a') as _f:
                    _f.write(f"BEFORE combine:\\n")
                    _f.write(f"  cache3: shape={list(intermediate_cache3.shape)} mean={intermediate_cache3.float().mean().item():.6f}\\n")
                    _f.write(f"  out_hs: shape={list(out_hidden_states.shape)} mean={out_hidden_states.float().mean().item():.6f} ptr={out_hidden_states.data_ptr()}\\n")
                    _f.write(f"  hidden_states ptr={hidden_states.data_ptr()}\\n")
                    _f.write(f"  inplace={inplace}, _use_intermediate={_use_intermediate}\\n")
"""
"""
            if not getattr(fused_experts_impl, '_combine_diag2_done', False):
                fused_experts_impl._combine_diag2_done = True
                with open('/workspace/combine_diag.txt', 'a') as _f:
                    _f.write(f"AFTER combine:\\n")
                    _f.write(f"  out_hs: mean={out_hidden_states.float().mean().item():.6f} ptr={out_hidden_states.data_ptr()}\\n")
                    _f.write(f"  hidden_states: mean={hidden_states.float().mean().item():.6f} ptr={hidden_states.data_ptr()}\\n")
"""
