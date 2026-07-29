# Diagnostic code extracted from minimax_m3.py
# These blocks were used for debugging layer outputs during initial bring-up.

# === In MiniMaxM3Model.forward(), after embedding ===
# Location: inside `if self.pp_group.is_first_rank:` block
"""
            # Diagnostic: embedding output
            if not getattr(MiniMaxM3Model, '_layer_diag_done', False) and hidden_states.numel() > 0:
                hs = hidden_states.float()
                with open('/workspace/layer_diag.txt', 'a') as _f:
                    _f.write(f"embed:   hs: mean={hs.mean().item():.6f} std={hs.std().item():.6f} "
                             f"min={hs.min().item():.4f} max={hs.max().item():.4f} "
                             f"shape={list(hs.shape)}\n")
"""

# === In MiniMaxM3Model.forward(), the for-loop over layers (else branch of can_run_tbo) ===
# Added a `_layer_diag_done` variable and per-layer hidden_states dump
"""
            _layer_diag_done = getattr(MiniMaxM3Model, '_layer_diag_done', False)
            # ... (for loop body unchanged) ...
                # Diagnostic: dump hidden_states stats for first forward pass
                if not _layer_diag_done and hidden_states is not None and hidden_states.numel() > 0:
                    hs = hidden_states.float()
                    rs = residual.float() if residual is not None else None
                    with open('/workspace/layer_diag.txt', 'a') as _f:
                        _f.write(f"layer={i:2d} hs: mean={hs.mean().item():.6f} std={hs.std().item():.6f} "
                                 f"min={hs.min().item():.4f} max={hs.max().item():.4f} "
                                 f"nan={hs.isnan().sum().item()} inf={hs.isinf().sum().item()}")
                        if rs is not None:
                            _f.write(f" | res: mean={rs.mean().item():.6f} std={rs.std().item():.6f} "
                                     f"min={rs.min().item():.4f} max={rs.max().item():.4f}")
                        _f.write(f" shape={list(hs.shape)}\n")
            if not _layer_diag_done:
                MiniMaxM3Model._layer_diag_done = True
"""

# === In MiniMaxM3DecoderLayer.forward(), after self_attn call ===
"""
        # Diagnostic: attention output for layers 0-5
        if self.layer_id <= 5 and not getattr(MiniMaxM3DecoderLayer, f'_attn_diag_{self.layer_id}', False):
            setattr(MiniMaxM3DecoderLayer, f'_attn_diag_{self.layer_id}', True)
            if hidden_states.numel() > 0:
                hs = hidden_states.float()
                with open('/workspace/layer_diag.txt', 'a') as _f:
                    _f.write(f"layer={self.layer_id} attn_out: mean={hs.mean().item():.6f} std={hs.std().item():.6f} min={hs.min().item():.4f} max={hs.max().item():.4f}\n")
"""
