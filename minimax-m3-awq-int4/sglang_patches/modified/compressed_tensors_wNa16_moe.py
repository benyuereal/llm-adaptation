from __future__ import annotations

import enum
import logging
from enum import Enum
from typing import TYPE_CHECKING

import torch
from compressed_tensors import CompressionFormat

from sglang.srt.hardware_backend.npu.quantization.fused_moe_method_npu import (
    NPUW4A16Int4DynamicMoEMethod,
)
from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, MoeRunnerConfig
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    WNA16_SUPPORTED_BITS,
    CompressedTensorsMoEScheme,
)
from sglang.srt.layers.quantization.gptq import gptq_marlin_moe_repack
from sglang.srt.layers.quantization.marlin_utils import marlin_moe_permute_scales
from sglang.srt.layers.quantization.utils import replace_parameter
from sglang.srt.utils import get_bool_env_var, is_cuda, is_hip, set_weight_attrs

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )


__all__ = [
    "CompressedTensorsWNA16MoE",
    "CompressedTensorsWNA16TritonMoE",
    "NPUCompressedTensorsW4A16Int4DynamicMoE",
]

_is_hip = is_hip()
_is_cuda = is_cuda()

_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _use_aiter:
    pass


logger = logging.getLogger(__name__)


class GPTQMarlinState(Enum):
    REPACK = enum.auto()
    READY = enum.auto()


class CompressedTensorsWNA16MoE(CompressedTensorsMoEScheme):

    def __init__(self, quant_config: CompressedTensorsConfig, num_gpu_experts=-1):
        self.quant_config = quant_config
        config = self.quant_config.target_scheme_map["Linear"].get("weights")
        self.num_bits = config.num_bits
        self.packed_factor = 32 // config.num_bits
        self.strategy = config.strategy
        self.group_size = config.group_size
        self.actorder = config.actorder
        self.symmetric = config.symmetric

        if not (
            self.quant_config.quant_format == CompressionFormat.pack_quantized.value
            and self.num_bits in WNA16_SUPPORTED_BITS
        ):
            raise ValueError(
                "For Fused MoE layers, only ",
                f"{CompressionFormat.pack_quantized.value} ",
                "is supported for the following bits: ",
                f"{WNA16_SUPPORTED_BITS}",
            )
        self.num_gpu_experts = num_gpu_experts

    @classmethod
    def get_min_capability(cls) -> int:
        # ampere and up
        return 80

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # Will transpose the loaded weight along the
        # intermediate and hidden dim sizes. Will
        # shard for TP along the transposed dims
        extra_weight_attrs.update(
            {"is_transposed": True, "quant_method": self.strategy}
        )
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size // self.packed_factor,
                2 * intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_packed", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition // self.packed_factor,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # In the case where we have actorder/g_idx,
        # we do not partition the w2 scales
        load_full_w2 = self.actorder and self.group_size != -1

        if load_full_w2:
            w2_scales_size = intermediate_size_per_partition * layer.moe_tp_size
        else:
            w2_scales_size = intermediate_size_per_partition

        self.is_k_full = (not self.actorder) or layer.moe_tp_size == 1

        if self.strategy == "channel":
            num_groups_w2 = num_groups_w13 = 1
            self.group_size = -1
        else:
            num_groups_w2 = w2_scales_size // self.group_size
            num_groups_w13 = hidden_size // self.group_size

        w13_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                num_groups_w13,
                2 * intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_scale)
        set_weight_attrs(w13_scale, extra_weight_attrs)

        w2_scale = torch.nn.Parameter(
            torch.ones(num_experts, num_groups_w2, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_scale)
        set_weight_attrs(w2_scale, extra_weight_attrs)
        set_weight_attrs(w2_scale, {"load_full_w2": load_full_w2})

        # Zero points for asymmetric quantization
        # Checkpoint per-expert zp: [N//8, K//group_size] int32
        # Loader will transpose to [K//group_size, N//8] then TP-shard on dim 1
        # Final param shape: [E, K//group_size, N_tp//8] where N_tp = 2*intermediate_per_partition
        if not self.symmetric:
            zp_pack_factor = 8
            w13_zp = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    num_groups_w13,
                    2 * intermediate_size_per_partition // zp_pack_factor,
                    dtype=torch.int32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_zero_point", w13_zp)
            set_weight_attrs(w13_zp, extra_weight_attrs)

            w2_zp = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    num_groups_w2,
                    hidden_size // zp_pack_factor,
                    dtype=torch.int32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight_zero_point", w2_zp)
            set_weight_attrs(w2_zp, extra_weight_attrs)

        w2_weight_shape = torch.nn.Parameter(
            torch.empty(num_experts, 2), requires_grad=False
        )
        layer.register_parameter("w2_weight_shape", w2_weight_shape)
        set_weight_attrs(w2_weight_shape, extra_weight_attrs)
        w13_weight_shape = torch.nn.Parameter(
            torch.empty(num_experts, 2), requires_grad=False
        )

        layer.register_parameter("w13_weight_shape", w13_weight_shape)
        set_weight_attrs(w13_weight_shape, extra_weight_attrs)

        w13_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_g_idx", w13_g_idx)
        set_weight_attrs(w13_g_idx, extra_weight_attrs)

        w2_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_g_idx", w2_g_idx)
        set_weight_attrs(w2_g_idx, extra_weight_attrs)

        w13_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_g_idx_sort_indices", w13_g_idx_sort_indices)
        set_weight_attrs(w13_g_idx_sort_indices, extra_weight_attrs)

        w2_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_g_idx_sort_indices", w2_g_idx_sort_indices)
        set_weight_attrs(w2_g_idx_sort_indices, extra_weight_attrs)

        layer.a13_scale = None
        layer.a2_scale = None
        layer.marlin_state = GPTQMarlinState.REPACK

        if not hasattr(layer, "_original_shapes"):
            layer._original_shapes = {}

        # Force record: these are the target GPTQ shapes for rollback.
        layer._original_shapes["w13_weight_packed"] = tuple(w13_weight.shape)
        layer._original_shapes["w2_weight_packed"] = tuple(w2_weight.shape)

        # Also record the shapes of the scales.
        layer._original_shapes["w2_weight_scale"] = tuple(w2_scale.shape)
        layer._original_shapes["w13_weight_scale"] = tuple(w13_scale.shape)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:

        # Skip if the layer is already converted to Marlin format to prevent double-packing.
        if getattr(layer, "is_marlin_converted", False):
            return

        if not hasattr(layer, "_original_shapes"):
            layer._original_shapes = {}

        def replace_tensor(name, new_t):
            target_attr = getattr(layer, name)

            # Only save if the key doesn't exist to prevent overwriting with Marlin shapes.
            if name not in layer._original_shapes:
                # This is a safety check; `create_weights` usually handles this already.
                layer._original_shapes[name] = tuple(target_attr.shape)

            # It is important to use resize_() here since it ensures
            # the same buffer is reused
            target_attr.resize_(new_t.shape)
            target_attr.copy_(new_t)
            del new_t

        num_experts = layer.w13_weight_g_idx.shape[0]
        device = layer.w13_weight_g_idx.device

        # when running models with grouped act order,
        # resort to g_idx values provided in checkpoint
        if self.actorder == "group":
            w13_g_idx_sort_indices = torch.empty_like(layer.w13_weight_g_idx)
            w2_g_idx_sort_indices = torch.empty_like(layer.w2_weight_g_idx)
            w13_sorted_g_idx = torch.empty_like(layer.w13_weight_g_idx)
            w2_sorted_g_idx = torch.empty_like(layer.w2_weight_g_idx)

            for e in range(num_experts):
                w13_g_idx_sort_indices[e] = torch.argsort(layer.w13_weight_g_idx[e]).to(
                    torch.int32
                )
                w2_g_idx_sort_indices[e] = torch.argsort(layer.w2_weight_g_idx[e]).to(
                    torch.int32
                )
                w13_sorted_g_idx[e] = layer.w13_weight_g_idx[e][
                    w13_g_idx_sort_indices[e]
                ]
                w2_sorted_g_idx[e] = layer.w2_weight_g_idx[e][w2_g_idx_sort_indices[e]]

            replace_parameter(layer, "w13_weight_g_idx", w13_sorted_g_idx)
            replace_parameter(layer, "w2_weight_g_idx", w2_sorted_g_idx)
            replace_parameter(layer, "w13_g_idx_sort_indices", w13_g_idx_sort_indices)
            replace_parameter(layer, "w2_g_idx_sort_indices", w2_g_idx_sort_indices)

        else:
            layer.w13_weight_g_idx = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            )
            layer.w2_weight_g_idx = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            )
            layer.w13_g_idx_sort_indices = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            )
            layer.w2_g_idx_sort_indices = torch.nn.Parameter(
                torch.empty((num_experts, 0), dtype=torch.int32, device=device),
                requires_grad=False,
            )

        marlin_w13_qweight = gptq_marlin_moe_repack(
            layer.w13_weight_packed,
            layer.w13_g_idx_sort_indices,
            layer.w13_weight_packed.shape[1] * self.packed_factor,
            layer.w13_weight_packed.shape[2],
            self.num_bits,
        )
        replace_tensor("w13_weight_packed", marlin_w13_qweight)
        marlin_w2_qweight = gptq_marlin_moe_repack(
            layer.w2_weight_packed,
            layer.w2_g_idx_sort_indices,
            layer.w2_weight_packed.shape[1] * self.packed_factor,
            layer.w2_weight_packed.shape[2],
            self.num_bits,
        )
        replace_tensor("w2_weight_packed", marlin_w2_qweight)
        # Repack scales
        marlin_w13_scales = marlin_moe_permute_scales(
            layer.w13_weight_scale,
            layer.w13_weight_packed.shape[2],
            layer.w13_weight_scale.shape[2],
            self.group_size,
        )
        replace_tensor("w13_weight_scale", marlin_w13_scales)

        marlin_w2_scales = marlin_moe_permute_scales(
            layer.w2_weight_scale,
            layer.w2_weight_scale.shape[1]
            * (self.group_size if self.group_size != -1 else self.packed_factor),
            layer.w2_weight_scale.shape[2],
            self.group_size,
        )
        replace_tensor("w2_weight_scale", marlin_w2_scales)

        layer.is_marlin_converted = True

    def restore_weights_before_loading(self, layer: torch.nn.Module):
        """Forcibly resize parameters back to their original shapes (e.g., GPTQ format) before loading weights."""

        if not hasattr(layer, "_original_shapes"):
            return

        for name, orig_shape in layer._original_shapes.items():
            param = getattr(layer, name, None)

            if param is not None and param.shape != orig_shape:
                param.resize_(orig_shape)

        layer.is_marlin_converted = False

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        self.runner = MoeRunner(MoeRunnerBackend.MARLIN, moe_runner_config)

    def get_marlin_quant_info(self, layer):
        from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo

        return MarlinMoeQuantInfo(
            w13_qweight=layer.w13_weight_packed,
            w2_qweight=layer.w2_weight_packed,
            w13_scales=layer.w13_weight_scale,
            w2_scales=layer.w2_weight_scale,
            w13_g_idx_sort_indices=getattr(layer, "w13_g_idx_sort_indices", None),
            w2_g_idx_sort_indices=getattr(layer, "w2_g_idx_sort_indices", None),
            weight_bits=self.num_bits,
            w13_g_idx=getattr(layer, "w13_weight_g_idx", None),
            w2_g_idx=getattr(layer, "w2_weight_g_idx", None),
            is_k_full=self.is_k_full,
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
            fused_marlin_moe,
        )
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        assert (
            self.moe_runner_config.activation == "silu"
        ), "Only SiLU activation is supported."

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        topk_weights, topk_ids, router_logits = topk_output

        # Get expert_map for EP support
        expert_map = None
        global_num_experts = -1
        if hasattr(layer, "dispatcher") and hasattr(
            layer.dispatcher, "local_expert_mapping"
        ):
            expert_map = layer.dispatcher.local_expert_mapping
            if expert_map is not None:
                global_num_experts = self.moe_runner_config.num_experts

        output = fused_marlin_moe(
            x,
            layer.w13_weight_packed,
            layer.w2_weight_packed,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
            router_logits,
            topk_weights,
            topk_ids,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            g_idx1=layer.w13_weight_g_idx,
            g_idx2=layer.w2_weight_g_idx,
            sort_indices1=layer.w13_g_idx_sort_indices,
            sort_indices2=layer.w2_g_idx_sort_indices,
            num_bits=self.num_bits,
            is_k_full=self.is_k_full,
            routed_scaling_factor=self.moe_runner_config.routed_scaling_factor,
        )
        return StandardCombineInput(hidden_states=output)


class CompressedTensorsWNA16TritonMoE(CompressedTensorsWNA16MoE):
    """ROCm/HIP-compatible W4A16 MoE method using Triton kernels instead of Marlin.

    Keeps INT4 packed weights in VRAM. At inference time, passes packed weights
    + scales directly to the Triton fused MoE kernel which performs dequant
    inside the kernel (use_int4_w4a16=True).
    """

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "is_triton_converted", False):
            return

        # Convert w13 weights: [E, K//8, N] int32 -> [E, N, K//2] uint8
        w13 = layer.w13_weight_packed.data
        w13 = w13.transpose(1, 2).contiguous().view(torch.uint8)
        layer.w13_weight_packed = torch.nn.Parameter(w13, requires_grad=False)

        # Convert w2 weights: [E, K//8, N] int32 -> [E, N, K//2] uint8
        w2 = layer.w2_weight_packed.data
        w2 = w2.transpose(1, 2).contiguous().view(torch.uint8)
        layer.w2_weight_packed = torch.nn.Parameter(w2, requires_grad=False)

        # Convert w13 scales: [E, K//group_size, N] -> [E, N, K//group_size]
        w13_scale = layer.w13_weight_scale.data
        w13_scale = w13_scale.transpose(1, 2).contiguous()
        layer.w13_weight_scale = torch.nn.Parameter(w13_scale, requires_grad=False)

        # Convert w2 scales: [E, K//group_size, N] -> [E, N, K//group_size]
        w2_scale = layer.w2_weight_scale.data
        w2_scale = w2_scale.transpose(1, 2).contiguous()
        layer.w2_weight_scale = torch.nn.Parameter(w2_scale, requires_grad=False)

        # Convert zero points: [E, K_groups, N_tp//8] int32 → [E, N_tp//2, K_groups] uint8
        # Each int32 has 8 nibbles for 8 N positions; each byte has 2 nibbles
        if hasattr(layer, "w13_weight_zero_point") and layer.w13_weight_zero_point is not None:
            w13_zp = layer.w13_weight_zero_point.data  # [E, K_groups, N_tp//8] int32
            # Fill zero experts with default zp=8 (each nibble = 8 → each byte = 0x88)
            # Check per-expert to avoid masking partially-loaded layers
            num_zero_experts = 0
            for ei in range(w13_zp.shape[0]):
                if (w13_zp[ei] == 0).all():
                    w13_zp[ei].view(torch.uint8).fill_(0x88)
                    num_zero_experts += 1
            if num_zero_experts > 0:
                logger.warning(
                    f"w13_zp: {num_zero_experts}/{w13_zp.shape[0]} experts had zero zp, "
                    f"filled with default zp=8. Shape={w13_zp.shape}"
                )
            w13_zp = w13_zp.transpose(1, 2).contiguous()  # [E, N_tp//8, K_groups]
            E_zp, Npack, Kgroups = w13_zp.shape
            zp_u8 = w13_zp.view(torch.uint8).reshape(E_zp, Npack, Kgroups, 4)
            zp_u8 = zp_u8.permute(0, 1, 3, 2).contiguous().reshape(E_zp, Npack * 4, Kgroups)
            layer.w13_weight_zero_point = torch.nn.Parameter(zp_u8, requires_grad=False)

        if hasattr(layer, "w2_weight_zero_point") and layer.w2_weight_zero_point is not None:
            w2_zp = layer.w2_weight_zero_point.data
            num_zero_experts = 0
            for ei in range(w2_zp.shape[0]):
                if (w2_zp[ei] == 0).all():
                    w2_zp[ei].view(torch.uint8).fill_(0x88)
                    num_zero_experts += 1
            if num_zero_experts > 0:
                logger.warning(
                    f"w2_zp: {num_zero_experts}/{w2_zp.shape[0]} experts had zero zp, "
                    f"filled with default zp=8. Shape={w2_zp.shape}"
                )
            w2_zp = w2_zp.transpose(1, 2).contiguous()
            E_zp, Npack, Kgroups = w2_zp.shape
            zp_u8 = w2_zp.view(torch.uint8).reshape(E_zp, Npack, Kgroups, 4)
            zp_u8 = zp_u8.permute(0, 1, 3, 2).contiguous().reshape(E_zp, Npack * 4, Kgroups)
            layer.w2_weight_zero_point = torch.nn.Parameter(zp_u8, requires_grad=False)

        layer.is_triton_converted = True

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config
        self.runner = MoeRunner(MoeRunnerBackend.TRITON, moe_runner_config)

    def get_triton_quant_info(self, layer):
        from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo

        w13_zp = getattr(layer, "w13_weight_zero_point", None)
        w2_zp = getattr(layer, "w2_weight_zero_point", None)

        return TritonMoeQuantInfo(
            w13_weight=layer.w13_weight_packed,
            w2_weight=layer.w2_weight_packed,
            use_int4_w4a16=True,
            w13_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            w13_zp=w13_zp,
            w2_zp=w2_zp,
            block_shape=[0, self.group_size],
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        """Dynamic dequant: pass INT4 packed weights to Triton fused kernel."""
        if get_bool_env_var("SGLANG_MOE_TORCH_FALLBACK"):
            return self._apply_weights_torch_fallback(layer, dispatch_output)
        quant_info = self.get_triton_quant_info(layer)
        return self.runner.run(dispatch_output, quant_info)

    def _apply_weights_torch_fallback(self, layer, dispatch_output):
        """Pure PyTorch dequant fallback for debugging. Slow but correct."""
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states  # [num_tokens, K]
        topk_output = dispatch_output.topk_output
        topk_weights, topk_ids, _ = topk_output
        num_tokens, K = x.shape
        top_k = topk_ids.shape[1]

        # w13: [E, N_w13, K//2] uint8, scale: [E, N_w13, K_groups], zp: [E, N_w13//2, K_groups]
        w13 = layer.w13_weight_packed  # uint8
        s13 = layer.w13_weight_scale   # bf16
        zp13 = getattr(layer, "w13_weight_zero_point", None)  # uint8
        w2 = layer.w2_weight_packed
        s2 = layer.w2_weight_scale
        zp2 = getattr(layer, "w2_weight_zero_point", None)

        E, N_w13, K_half = w13.shape
        N_w2 = w2.shape[1]
        group_size = self.group_size

        # Dequant one expert's weight on the fly
        def dequant_expert(w, s, zp, expert_id):
            # w: [E, N, K//2] uint8 -> [N, K] float
            we = w[expert_id]  # [N, K//2]
            se = s[expert_id].float()  # [N, K_groups]
            N_dim = we.shape[0]
            # Unpack nibbles
            lo = (we & 0x0F).to(torch.float32)  # even K
            hi = ((we >> 4) & 0x0F).to(torch.float32)  # odd K
            # Interleave: [N, K]
            w_full = torch.stack([lo, hi], dim=-1).reshape(N_dim, -1)  # [N, K]
            # ZP
            if zp is not None:
                zpe = zp[expert_id]  # [N//2, K_groups]
                zp_lo = (zpe & 0x0F).to(torch.float32)  # even N
                zp_hi = ((zpe >> 4) & 0x0F).to(torch.float32)  # odd N
                zp_full = torch.stack([zp_lo, zp_hi], dim=1).reshape(N_dim, -1)  # [N, K_groups]... 
                # Actually zp is [N//2, K_groups], need to expand to [N, K_groups]
                zp_full2 = torch.zeros(N_dim, zpe.shape[1], device=zpe.device, dtype=torch.float32)
                zp_full2[0::2, :] = zp_lo
                zp_full2[1::2, :] = zp_hi
                # Expand zp to [N, K]
                zp_expanded = zp_full2.repeat_interleave(group_size, dim=1)  # [N, K]
            else:
                zp_expanded = torch.full_like(w_full, 8.0)
            # Scale expand to [N, K]
            s_expanded = se.repeat_interleave(group_size, dim=1)  # [N, K]
            # Dequant
            return ((w_full - zp_expanded) * s_expanded).to(x.dtype)

        # Per-token per-expert matmul
        N_w2 = w2.shape[1]  # hidden_size (output dim of down proj)
        output = torch.zeros(num_tokens, N_w2, dtype=x.dtype, device=x.device)
        
        for k in range(top_k):
            expert_ids_k = topk_ids[:, k]  # [num_tokens]
            weights_k = topk_weights[:, k]  # [num_tokens]
            
            # Group tokens by expert
            unique_experts = expert_ids_k.unique()
            for eid in unique_experts:
                mask = expert_ids_k == eid
                x_expert = x[mask]  # [n, K]
                
                # w13 = gate + up fused: first N_w13//2 is gate, second is up
                w13_dequant = dequant_expert(w13, s13, zp13, eid)  # [N_w13, K]
                gate_up = x_expert @ w13_dequant.T  # [n, N_w13]
                
                # SiLU gate
                N_half = N_w13 // 2
                gate = gate_up[:, :N_half]
                up = gate_up[:, N_half:]
                hidden = torch.nn.functional.silu(gate) * up  # [n, N_half]
                
                # w2: down projection
                w2_dequant = dequant_expert(w2, s2, zp2, eid)  # [N_w2, K_w2]
                down = hidden @ w2_dequant.T  # [n, N_w2]
                
                # Weighted accumulate
                output[mask] += down * weights_k[mask, None]
        
        return StandardCombineInput(hidden_states=output)


class NPUCompressedTensorsW4A16Int4DynamicMoE(CompressedTensorsMoEScheme):

    def __init__(self, quantization_config) -> None:
        self.pack_factor = 8  # weight dtype is int4,  but use int32 to create
        target = (
            "MoEGMM" if "MoEGMM" in quantization_config.target_scheme_map else "Linear"
        )
        if target in quantization_config.target_scheme_map:
            self.group_size = quantization_config.target_scheme_map[target][
                "weights"
            ].group_size
        else:
            self.group_size = 128

        self.kernel = NPUW4A16Int4DynamicMoEMethod()

    # TODO: See if we can merge this method's logic
    # with CompressedTensorsWNA16MoE. Need more models and tests.
    # @OrangeRedeng @TamirBaydasov
    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        self.num_experts = num_experts
        if (
            extra_weight_attrs.get(
                "moe_intermediate_size", intermediate_size_per_partition
            )
            // intermediate_size_per_partition
            > 1
        ):
            quant_method = FusedMoeWeightScaleSupported.GROUP.value
        else:
            quant_method = FusedMoeWeightScaleSupported.CHANNEL.value
        extra_weight_attrs.update({"quant_method": quant_method})
        # weight
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # scale
        weight_scale_dtype = torch.bfloat16
        w13_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.group_size,
                dtype=weight_scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        w2_weight_scale = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.group_size,
                dtype=weight_scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # offset
        w13_weight_offset = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.group_size,
                dtype=weight_scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_offset", w13_weight_offset)
        set_weight_attrs(w13_weight_offset, extra_weight_attrs)

        w2_weight_offset = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.group_size,
                dtype=weight_scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_offset", w2_weight_offset)
        set_weight_attrs(w2_weight_offset, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.kernel.process_weights_after_loading(layer)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:

        return self.kernel.apply(layer, dispatch_output)

    def apply_without_routing_weights(
        self,
        layer,
        hidden_states,
        hidden_states_scale,
        group_list_type,
        group_list,
        output_dtype,
    ):
        return self.kernel.apply_without_routing_weights(
            layer,
            hidden_states,
            hidden_states_scale,
            group_list_type,
            group_list,
            output_dtype,
        )
