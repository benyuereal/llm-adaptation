# Diagnostic code extracted from compressed_tensors_wNa16_moe.py
# These blocks were used to trace zero-point data during weight loading.

# === In CompressedTensorsWNA16TritonMoE.process_weights_after_loading(),
#     at the START (before weight conversion) ===
"""
        # Pre-conversion diagnostic: check which layers have zero zp
        if hasattr(layer, "w13_weight_zero_point") and layer.w13_weight_zero_point is not None:
            if not hasattr(CompressedTensorsWNA16TritonMoE, '_pre_conv_count'):
                CompressedTensorsWNA16TritonMoE._pre_conv_count = 0
            CompressedTensorsWNA16TritonMoE._pre_conv_count += 1
            cnt = CompressedTensorsWNA16TritonMoE._pre_conv_count
            raw = layer.w13_weight_zero_point.data
            all_zero = (raw == 0).all().item()
            nz_experts = sum(1 for ei in range(raw.shape[0]) if (raw[ei] != 0).any())
            if all_zero or cnt <= 5:
                import os
                rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "?"))
                with open('/workspace/zp_preconv_diag.txt', 'a') as _f:
                    _f.write(
                        f"layer_ord={cnt} rank={rank} all_zero={all_zero} "
                        f"nz_experts={nz_experts}/{raw.shape[0]} "
                        f"shape={list(raw.shape)} sample_e0={raw[0,0,:4].tolist()} "
                        f"data_ptr={raw.data_ptr()} param_id={id(layer.w13_weight_zero_point)}\\n"
                    )
"""

# === In CompressedTensorsWNA16TritonMoE.process_weights_after_loading(),
#     at the END (after `layer.is_triton_converted = True`) ===
"""
        # Debug: dump zp data for ALL layers
        if hasattr(layer, "w13_weight_zero_point"):
            if not hasattr(CompressedTensorsWNA16TritonMoE, '_zp_dump_count'):
                CompressedTensorsWNA16TritonMoE._zp_dump_count = 0
                CompressedTensorsWNA16TritonMoE._zp_nonzero_layers = 0
                CompressedTensorsWNA16TritonMoE._zp_zero_layers = 0
                CompressedTensorsWNA16TritonMoE._zp_zero_expert_info = []
            CompressedTensorsWNA16TritonMoE._zp_dump_count += 1
            raw = layer.w13_weight_zero_point.data
            nz = (raw != 0).sum().item()
            # Count per-expert zeros (after fill, should be 0x88 not zero)
            zero_experts = []
            for ei in range(raw.shape[0]):
                if (raw[ei] == 0).all():
                    zero_experts.append(ei)
            if nz > 0 and not zero_experts:
                CompressedTensorsWNA16TritonMoE._zp_nonzero_layers += 1
            else:
                CompressedTensorsWNA16TritonMoE._zp_zero_layers += 1
                CompressedTensorsWNA16TritonMoE._zp_zero_expert_info.append({
                    'layer_idx': CompressedTensorsWNA16TritonMoE._zp_dump_count - 1,
                    'zero_experts': zero_experts[:10],
                    'total_zero': len(zero_experts),
                })
            # Dump summary on last layer
            if CompressedTensorsWNA16TritonMoE._zp_dump_count == 57:
                torch.save({
                    'total_layers': 57,
                    'nonzero_layers': CompressedTensorsWNA16TritonMoE._zp_nonzero_layers,
                    'zero_layers': CompressedTensorsWNA16TritonMoE._zp_zero_layers,
                    'zero_expert_info': CompressedTensorsWNA16TritonMoE._zp_zero_expert_info,
                    'last_layer_nonzero': nz,
                    'last_layer_total': raw.numel(),
                    'last_layer_sample': raw[0, :4, :4].cpu().clone(),
                }, '/workspace/zp_all_layers.pt')
"""
