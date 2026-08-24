from typing import Any, Callable, Dict, List, Optional

import torch
from vllm.model_executor.utils import set_weight_attrs
from vllm.distributed import get_tensor_model_parallel_world_size
from torch.nn.parameter import Parameter
from vllm.model_executor.layers.linear import (LinearBase,LinearMethodBase)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.fused_moe import (FusedMoE, FusedMoEMethodBase,
                                                  FusedMoeWeightScaleSupported)
from vllm.model_executor.parameter import (BasevLLMParameter,
                                           ChannelQuantScaleParameter,
                                           ModelWeightParameter,
                                           PerTensorScaleParameter)
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEModularKernel)
from lmslim.layers.gemm.int8_utils import (
    per_token_group_quant_int8,
    per_token_quant_int8)
from vllm.utils import W8a8GetCacheJSON
from vllm.model_executor.layers.quantization.utils.w8a8_utils import apply_int8_linear
import os
from vllm import _custom_ops as ops
from vllm import envs

from vllm.logger import init_logger
logger = init_logger(__name__)

try:
    from aiter.ops.shuffle import w4a8_moe_layout_shuffle_gemm1,w4a8_moe_layout_shuffle_gemm2
    from aiter.moe import (
    get_aiter_moe_config,
    aiter_moe,
    MoeSolutionType,
    MoeQuantType,
    )
    from aiter import dtypes, ActivationType
except ImportError as e:
    print("Import error msg: import aiter")

W8A8_TRITONJSON=W8a8GetCacheJSON()


# =============================================================================
# [DEQUANT] MTP draft-layer dequantization helpers
# =============================================================================
# The MTP draft reuses a SINGLE w4a8 layer serially for every spec position.
# Per-position quantization noise accumulates and produces the acceptance
# cliff (pos0 ~0.95 -> pos3 ~0.03). Dequantizing this one layer to bf16 at
# load time removes the *weight* quantization noise. Verified dequant formulas:
#   - int4-packed MoE expert (K dim halved in checkpoint):
#       w[n,k] = (nibble*16).to(int8).to(float) * scale[n]
#       k even -> hi nibble (byte>>4), k odd -> lo nibble (byte & 0xF)
#   - true int8 (attention / indexer / shared_experts, K NOT halved):
#       w[n,k] = weight[n,k].to(float) * scale[n]
# Output cast to bf16.
# =============================================================================

# Env override: set VLLM_DEQUANT_MTP_LAYER to the layer index (default: read
# from model config num_hidden_layers). Set to -1 to disable.
_DEQUANT_LAYER_IDX = None


def _get_dequant_layer_idx() -> int:
    """Return the MTP draft layer index to dequantize, or -1 to disable."""
    global _DEQUANT_LAYER_IDX
    if _DEQUANT_LAYER_IDX is not None:
        return _DEQUANT_LAYER_IDX
    env_val = os.environ.get("VLLM_DEQUANT_MTP_LAYER", "")
    # explicit disable / not set => dequant OFF by default. Set
    # VLLM_DEQUANT_MTP_LAYER=<layer_idx> (e.g. 78) to enable.
    if env_val == "" or env_val == "-1" or env_val.lower() in (
            "off", "disable", "false"):
        _DEQUANT_LAYER_IDX = -1
        return _DEQUANT_LAYER_IDX
    _DEQUANT_LAYER_IDX = int(env_val)
    return _DEQUANT_LAYER_IDX


def _is_dequant_layer(prefix: str) -> bool:
    """True if prefix belongs to the MTP draft layer to be dequantized."""
    if prefix is None:
        return False
    idx = _get_dequant_layer_idx()
    if idx < 0:
        return False
    # prefix like "model.layers.78.mtp_block.self_attn.q_a_proj"
    # or "...experts..." etc. Match the layer index as a dotted segment.
    import re
    return bool(re.search(rf"layers\.{idx}\.", prefix))


def _dequant_attn_enabled() -> bool:
    """[DEQUANT-ATTN] Dequantize ALL attention int8 (w8a8) linears to bf16.

    Rationale (profiled 2026-08-24): the w8a8 int8 GEMM (lmslim matmul_int8,
    Triton) is 60% of GPU time in MTP decode and sits on a ~90us floor at
    small M (verify M=5 / draft M=1) that no Triton config breaks. The
    library bf16 GEMM (rocBLAS) has no such floor and is 2.5-2.8x faster on
    these attention shapes. Dequantizing the attention linears to bf16 at
    load time removes the int8 GEMM entirely (and the activation quant).
    MoE (int4 w4a8) is left untouched -- it is already fast.
    Set VLLM_DEQUANT_ATTN=1 to enable.
    """
    return os.environ.get("VLLM_DEQUANT_ATTN", "0") == "1"


def _dequant_int4_packed(weight_int8: torch.Tensor,
                         scale: torch.Tensor) -> torch.Tensor:
    """Dequantize int4-packed (2-per-int8) MoE weights to bf16.

    Args:
        weight_int8: [..., K//2] int8 tensor, each byte holds 2 nibbles.
            k even -> hi nibble (byte >> 4), k odd -> lo nibble (byte & 0xF).
        scale: [..., 1] float scale per output channel (broadcasts on N).
    Returns:
        [..., K] bf16 tensor = (nibble*16).to(int8).float() * scale.
    """
    # unpack: last dim K//2 -> K
    u8 = weight_int8.to(torch.uint8)
    hi = (u8 >> 4) & 0x0F          # [..., K//2]  (k even)
    lo = u8 & 0x0F                 # [..., K//2]  (k odd)
    # interleave [hi, lo] along last dim -> [..., K]  (k=0:hi, k=1:lo, ...)
    nib = torch.stack([hi, lo], dim=-1).reshape(*u8.shape[:-1], -1)
    # (nib*16).to(int8): reinterpret 0..240 unsigned as signed -128..-1/0..127
    val = (nib * 16).to(torch.int8).to(torch.float32)
    # scale broadcasts: val is [..., K], scale is [..., 1]
    val = val * scale.to(torch.float32)
    return val.to(torch.bfloat16)


def _dequant_int8(weight_int8: torch.Tensor,
                  scale: torch.Tensor) -> torch.Tensor:
    """Dequantize true int8 (w8a8) weights to bf16.

    Args:
        weight_int8: [..., K] int8 tensor.
        scale: [..., 1] float per-output-channel scale.
    Returns:
        [..., K] bf16 = weight.float() * scale.
    """
    val = weight_int8.to(torch.float32) * scale.to(torch.float32)
    return val.to(torch.bfloat16)

def baseline_scaled_mm(a: torch.Tensor,
                      b: torch.Tensor,
                      scale_a: torch.Tensor,
                      scale_b: torch.Tensor,
                      out_dtype: torch.dtype,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:

    scales= scale_a* scale_b.T
    gemmout= torch.mm(
        a.to(dtype=torch.float32), b.to(dtype=torch.float32))
    output = (scales *gemmout).to(out_dtype)
    if bias is not None:
        output = output + bias
    return output.to(out_dtype)


class SlimQuantW4A8Int8Config(QuantizationConfig):
    """Config class for W8A8 Int8 Quantization.

    - Weight: static, per-channel, symmetric
    - Activation: dynamic, per-token, symmetric
    """

    def __init__(self):
        pass

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def get_name(self) -> str:
        return "slimquant_w4a8"

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SlimQuantW4A8Int8Config":
        return cls()

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional["QuantizeMethodBase"]:
        # === [DEQUANT] MTP draft layer (layer == num_hidden_layers) is
        # dequantized to bf16 at load time to remove the per-position
        # quantization-noise accumulation that tanks MTP acceptance.
        # Recognized via the layer index in the prefix, e.g.
        # "model.layers.78.mtp_block.self_attn.q_a_proj".
        dequant = _is_dequant_layer(prefix)

        if isinstance(layer, LinearBase):
            # [DEQUANT-ATTN] VLLM_DEQUANT_ATTN=1 -> dequant ALL attention
            # int8 linears to bf16 (the int8 GEMM is 60% of decode GPU time
            # and 2.5-2.8x slower than the bf16 GEMM at small M). MoE is
            # handled in the FusedMoE branch below and is NOT affected.
            if _dequant_attn_enabled():
                dequant = True
            return SlimQuantW4A8Int8LinearMethod(self, dequant=dequant)
        elif isinstance(layer, FusedMoE):
            if envs.VLLM_ROCM_USE_AITER_MOE:
                return SlimQuantW4A8Int8AiterMoEMethod(self, layer.moe_config,
                                                       dequant=dequant)
            else:
                return SlimQuantW4A8Int8MoEMethod(self, layer.moe_config,
                                                  dequant=dequant)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class SlimQuantW4A8Int8LinearMethod(LinearMethodBase):

    def __init__(self, quantization_config: SlimQuantW4A8Int8Config,
                 dequant: bool = False):
        self.quantization_config = quantization_config
        self.tritonsingleton= W8a8GetCacheJSON()
        self.w8a8_strategy = envs.VLLM_W8A8_BACKEND
        self.dequant = dequant

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.dequant:
            # Dequantize int8 weight -> bf16, then drop the scale.
            # weight is [N, K] int8, weight_scale is [N, 1] f32 (per-output-ch).
            w = _dequant_int8(layer.weight.data, layer.weight_scale.data)
            # UnquantizedLinearMethod expects weight stored as [N, K] and
            # applies x @ weight.t(); keep this convention.
            layer.weight = Parameter(w, requires_grad=False)
            del layer.weight_scale
            return
        n=layer.weight.shape[0]
        k=layer.weight.shape[1]
        
        if self.w8a8_strategy==1:
            if {n,k} not in self.tritonsingleton.weight_shapes:
                self.tritonsingleton.weight_shapes.append({n,k})
                json_file=self.tritonsingleton.get_w8a8json_name(n,k)
                configs_dict=self.tritonsingleton.get_triton_cache(json_file,n,k)
                
                if configs_dict:
                    self.tritonsingleton.triton_json_dict.update(configs_dict)
                    
                    for key, value in configs_dict.items():
                        m=int(key.split('_')[0])
                        ops.triton_int8_gemm_helper(m=m,n=n,k=k,per_token_act_quant=True,per_out_channel_weight_quant=True,use_bias=False,device=layer.weight.device,best_config=value)
        elif self.w8a8_strategy == 3:
            layer.weight.data = layer.weight.data.T
        else:
            weight_data=layer.weight.data
            _weight=weight_data.T.contiguous().reshape(n,-1)
            layer.weight.data=_weight
            
        layer.weight = Parameter(layer.weight.t(), requires_grad=False)
        layer.weight_scale = Parameter(layer.weight_scale.data, requires_grad=False)

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):

        weight_loader = extra_weight_attrs.get("weight_loader")
        self.logical_widths = output_partition_sizes

        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes), input_size_per_partition, dtype=torch.int8
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        weight_scale = ChannelQuantScaleParameter(
            data=torch.empty((sum(output_partition_sizes), 1), dtype=torch.float32),
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        input_quant_args: Optional[list[torch.Tensor]] = None,
        silu_quant_args: Optional[list[torch.Tensor]] = None
    ):
        if self.dequant:
            # bf16 weight [N, K]; standard linear: x @ weight.t()
            w = layer.weight
            if bias is not None:
                if x.dim() == 2:
                    return torch.addmm(bias, x, w.t())
                return torch.matmul(x, w.t()) + bias
            return torch.matmul(x, w.t())
        return apply_int8_linear(input=x,
                                 weight=layer.weight,
                                 weight_scale=layer.weight_scale,
                                 bias=bias,
                                 w8a8_strategy=self.w8a8_strategy,
                                 input_quant_args=input_quant_args,
                                 silu_quant_args=silu_quant_args)



class SlimQuantW4A8Int8MoEMethod:
    """MoE method for W4A8INT8.
    Supports loading INT8 checkpoints with static weight scale and
    dynamic/static activation scale.
    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.
    Args:
        quant_config: The quantization config.
    """

    def __new__(cls, *args, **kwargs):

        if not hasattr(cls, "_initialized"):
            original_init = cls.__init__
            new_cls = type(
                cls.__name__,
                (FusedMoEMethodBase,),
                {
                    "__init__": original_init,
                    **{k: v for k, v in cls.__dict__.items() if k != "__dict__"},
                },
            )
            obj = super(new_cls, new_cls).__new__(new_cls)
            obj.__init__(*args, **kwargs)
            return obj
        return super().__new__(cls)

    def __init__(self, quant_config, moe, dequant: bool = False):
        self.moe = moe
        self.quant_config = quant_config
        self.tritonsingleton= W8a8GetCacheJSON()
        self.moe_quant_config: Optional[FusedMoEQuantConfig] = None
        self.moe_mk: Optional[FusedMoEModularKernel] = None
        self.dequant = dequant

    def get_fused_moe_quant_config(
            self, layer: torch.nn.Module)-> Optional[FusedMoEQuantConfig]:
        if self.dequant:
            from vllm.model_executor.layers.fused_moe.config import (
                FUSED_MOE_UNQUANTIZED_CONFIG)
            self.moe_quant_config = FUSED_MOE_UNQUANTIZED_CONFIG
            return self.moe_quant_config
        self.moe_quant_config = FusedMoEQuantConfig.make(
            torch.int8,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            per_act_token_quant=True,
            per_out_ch_quant=False,
            block_shape=None,
            weight_dtype='int4'
        )
        return self.moe_quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        tp_size = get_tensor_model_parallel_world_size()

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts, 2 * intermediate_size, hidden_size//2, dtype=torch.int8
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size//2, dtype=torch.int8),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, 2 * intermediate_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, hidden_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )

        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        w13_input_scale = None
        layer.register_parameter("w13_input_scale", w13_input_scale)

        w2_input_scale = None
        layer.register_parameter("w2_input_scale", w2_input_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.dequant:
            # Dequantize int4-packed MoE expert weights to bf16.
            # w13_weight: [E, 2*intermediate, hidden//2] int8 (packed)
            #   -> [E, 2*intermediate, hidden] bf16
            # w2_weight:  [E, hidden, intermediate//2] int8 (packed)
            #   -> [E, hidden, intermediate] bf16
            # scales: [E, N, 1] f32 per-output-channel
            w13 = _dequant_int4_packed(layer.w13_weight.data,
                                       layer.w13_weight_scale.data)
            w2 = _dequant_int4_packed(layer.w2_weight.data,
                                      layer.w2_weight_scale.data)
            layer.w13_weight = Parameter(w13.contiguous(), requires_grad=False)
            layer.w2_weight = Parameter(w2.contiguous(), requires_grad=False)
            # scales no longer needed for bf16 path
            del layer.w13_weight_scale
            del layer.w2_weight_scale
            logger.info("[DEQUANT] MoE layer dequantized to bf16: "
                        "w13=%s w2=%s",
                        tuple(w13.shape), tuple(w2.shape))
            return
        E=layer.w13_weight.shape[0]
        N1=layer.w13_weight.shape[1]
        N2=layer.w2_weight.shape[1]
        K=N1//2
        if [E,N1,N2,K] not in self.tritonsingleton.moe_weight_shapes:
            self.tritonsingleton.moe_weight_shapes.append([E,N1,N2,K])
            
        TOPK= self.tritonsingleton.topk

        json_file=self.tritonsingleton.get_moeint8json_name(E,N1,N2,K,TOPK,use_int4_w4a8=True)
        configs_dict=self.tritonsingleton.get_moeint8_triton_cache(json_file,E,N1,N2,K,TOPK)
        
        #warmup
        if configs_dict:
            self.tritonsingleton.triton_moejson_dict.update(configs_dict)

        layer.w13_weight = Parameter(layer.w13_weight, requires_grad=False)
        layer.w2_weight = Parameter(layer.w2_weight, requires_grad=False)
        layer.w13_weight_scale = Parameter(
            layer.w13_weight_scale.data, requires_grad=False
        )
        layer.w2_weight_scale = Parameter(
            layer.w2_weight_scale.data, requires_grad=False
        )

    def apply(
            self,
            layer: FusedMoE,
            x: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            use_nn_moe: bool | None = False,
            use_fused_gate: bool | None = False,
            i_q: torch.Tensor | None = None,
            i_s: torch.Tensor | None = None,
            shared_output: torch.Tensor | None = None,
            routed_scaling_factor: float = 1.0,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        from vllm.model_executor.layers.fused_moe import fused_experts
        if self.dequant:
            # bf16 unquantized path. quant_config == FUSED_MOE_UNQUANTIZED.
            return fused_experts(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                inplace=True,
                activation=layer.activation,
                expert_map=layer.expert_map,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                global_num_experts=layer.global_num_experts,
                quant_config=self.moe_quant_config,
                use_nn_moe=use_nn_moe,
                shared_output=shared_output,
                routed_scaling_factor=routed_scaling_factor,
            )
        return fused_experts(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            activation=layer.activation,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            global_num_experts=layer.global_num_experts,
            quant_config=self.moe_quant_config,
            use_nn_moe=use_nn_moe,
            shared_output=shared_output,
            routed_scaling_factor=routed_scaling_factor,
        ) 

class SlimQuantW4A8Int8AiterMoEMethod:
    """MoE method for W4A8INT8.
    Supports loading INT8 checkpoints with static weight scale and
    dynamic/static activation scale.
    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.
    Args:
        quant_config: The quantization config.
    """

    def __new__(cls, *args, **kwargs):

        if not hasattr(cls, "_initialized"):
            original_init = cls.__init__
            new_cls = type(
                cls.__name__,
                (FusedMoEMethodBase,),
                {
                    "__init__": original_init,
                    **{k: v for k, v in cls.__dict__.items() if k != "__dict__"},
                },
            )
            obj = super(new_cls, new_cls).__new__(new_cls)
            obj.__init__(*args, **kwargs)
            return obj
        return super().__new__(cls)

    def __init__(self, quant_config, moe, dequant: bool = False):
        self.moe = moe
        self.quant_config = quant_config
        self.tritonsingleton= W8a8GetCacheJSON()
        self.moe_quant_config: Optional[FusedMoEQuantConfig] = None
        self.moe_mk: Optional[FusedMoEModularKernel] = None
        self.dequant = dequant

    def get_fused_moe_quant_config(
            self, layer: torch.nn.Module)-> Optional[FusedMoEQuantConfig]:
        if self.dequant:
            from vllm.model_executor.layers.fused_moe.config import (
                FUSED_MOE_UNQUANTIZED_CONFIG)
            self.moe_quant_config = FUSED_MOE_UNQUANTIZED_CONFIG
            return self.moe_quant_config
        self.moe_quant_config = FusedMoEQuantConfig.make(
            torch.int8,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            per_act_token_quant=True,
            per_out_ch_quant=False,
            block_shape=None,
            weight_dtype='int4'
        )
        return self.moe_quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        tp_size = get_tensor_model_parallel_world_size()

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts, 2 * intermediate_size, hidden_size//2, dtype=torch.int8
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size//2, dtype=torch.int8),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, 2 * intermediate_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, hidden_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )

        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        w13_input_scale = None
        layer.register_parameter("w13_input_scale", w13_input_scale)

        w2_input_scale = None
        layer.register_parameter("w2_input_scale", w2_input_scale)

    def repack_and_shuffle_w4a8(self, weight_data, E):
        """
        逐 expert 处理 [n, k_half]
        处理完直接写回 weight_data[i]
        """
        # 原始 shape: [E, n, k_half]
        for i in range(E):
            # 1. 取当前 expert [n, k_half]
            expert = weight_data[i]
            n, k_half = expert.shape

            # 2. repack 逻辑（连续 → blocked）
            w_u8 = expert.to(torch.uint8)
            
            # 解包 1byte → 2个4bit
            w_unpacked = torch.stack([
                (w_u8 >> 4) & 0x0F,
                w_u8 & 0x0F
            ], dim=-1).view(n, -1)

            # 8个4bit分块重排
            blocks = w_unpacked.view(n, -1, 8)
            w_low = blocks[..., :4]
            w_high = blocks[..., 4:]
            packed = (w_low << 4) | w_high
            packed = packed.view(n, k_half)

            # 3. shuffle
            w_marlin_in = w4a8_moe_layout_shuffle_gemm2(packed)
            w_marlin_in = w_marlin_in.reshape(n, k_half)
            # 4. 直接写回
            weight_data[i] = w_marlin_in

        return weight_data
    
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.dequant:
            w13 = _dequant_int4_packed(layer.w13_weight.data,
                                       layer.w13_weight_scale.data)
            w2 = _dequant_int4_packed(layer.w2_weight.data,
                                      layer.w2_weight_scale.data)
            layer.w13_weight = Parameter(w13.contiguous(), requires_grad=False)
            layer.w2_weight = Parameter(w2.contiguous(), requires_grad=False)
            del layer.w13_weight_scale
            del layer.w2_weight_scale
            logger.info("[DEQUANT] AITER MoE layer dequantized to bf16: "
                        "w13=%s w2=%s", tuple(w13.shape), tuple(w2.shape))
            return
        E=layer.w13_weight.shape[0]
        layer.w13_weight_scale = Parameter(
            layer.w13_weight_scale.data, requires_grad=False
        )
        layer.w2_weight_scale = Parameter(
            layer.w2_weight_scale.data, requires_grad=False
        )
        layer.w13_weight = Parameter(self.repack_and_shuffle_w4a8(layer.w13_weight.data, E), requires_grad=False)
        layer.w2_weight = Parameter(self.repack_and_shuffle_w4a8(layer.w2_weight.data, E), requires_grad=False)
  
    def apply(
            self,
            layer: FusedMoE,
            x: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            use_nn_moe: bool | None = False,
            use_fused_gate: bool | None = False,
            i_q: torch.Tensor | None = None,
            i_s: torch.Tensor | None = None,
            shared_output: torch.Tensor | None = None,
            routed_scaling_factor: float = 1.0,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        from vllm.model_executor.layers.fused_moe import fused_experts

        if self.dequant:
            return fused_experts(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                inplace=True,
                activation=layer.activation,
                expert_map=layer.expert_map,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                global_num_experts=layer.global_num_experts,
                quant_config=self.moe_quant_config,
                use_nn_moe=use_nn_moe,
                shared_output=shared_output,
                routed_scaling_factor=routed_scaling_factor,
            )

        E = layer.w13_weight.size(0)
        K = x.size(-1)
        N1 = layer.w13_weight.size(1)

        if x.dim() == 2:
            # Make sure we are using the correct a1 (pre-permute).
            M = x.size(0)
        else:
            assert x.dim() == 3
            assert x.size(0) == E, f"{x.size(0)} == {E}"
            M = x.size(1)
        topk = topk_ids.size(1)
        status, moe_cfg = get_aiter_moe_config(
            M=M,
            E=E,
            N1=N1,
            N2=N1//2,
            K=K,
            top_k=topk,
            block_size=None,
            dtype=dtypes.bf16,
            quant_type=MoeQuantType.W4A8,
        )
        if not status:
            assert moe_cfg.solution_type is None
            assert moe_cfg.config is None
            logger.info(f"[get_config_w4a8] {M=}, no solution found")

        return aiter_moe(
            x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            moe_config=moe_cfg,
            activation="silu",
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            global_num_experts=E,
            expert_map=None,
        )