# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer."""

from typing import cast, Optional

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.attention.utils.kv_sharing_utils import validate_kv_sharing_target
from vllm.attention.utils.kv_transfer_utils import maybe_transfer_kv_layer
from vllm.config import CacheConfig, get_current_vllm_config
from vllm.config.vllm import VllmConfig
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.batch_invariant import vllm_is_batch_invariant
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    direct_register_custom_op,
    kv_cache_dtype_str_to_dtype,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionType,
    MLAAttentionImpl,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.selector import get_attn_backend
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)

logger = init_logger(__name__)


def should_load_quant_weights(quant_method: QuantizeMethodBase | None) -> bool:
    """Returns whether the quantization method should load quantized weights."""
    return quant_method is not None and not isinstance(
        quant_method, UnquantizedLinearMethod
    )


def set_default_quant_scales(layer: nn.Module, register_buffer: bool = False) -> None:
    """Sets default quantization scales for the layer."""
    if register_buffer:
        layer.register_buffer("_k_scale", torch.tensor(1.0, dtype=torch.float32))
        layer.register_buffer("_v_scale", torch.tensor(1.0, dtype=torch.float32))
        layer.register_buffer("_q_scale", torch.tensor(1.0, dtype=torch.float32))
        layer.register_buffer("_prob_scale", torch.tensor(1.0, dtype=torch.float32))
    else:
        layer._k_scale.fill_(1.0)
        layer._v_scale.fill_(1.0)
        layer._q_scale.fill_(1.0)
        layer._prob_scale.fill_(1.0)

    # We also keep q/k/v_scale on host (cpu) memory for attention
    # backends that require the scales to be on host instead of on device.
    # e.g. Flashinfer
    layer._q_scale_float = 1.0
    layer._k_scale_float = 1.0
    layer._v_scale_float = 1.0
    layer._prob_scale_float = 1.0

    # Initialize q/k/v range constants used by calc_kv_scales
    layer.q_range = torch.tensor(envs.Q_SCALE_CONSTANT, dtype=torch.float32)
    layer.k_range = torch.tensor(envs.K_SCALE_CONSTANT, dtype=torch.float32)
    layer.v_range = torch.tensor(envs.V_SCALE_CONSTANT, dtype=torch.float32)


def _init_kv_cache_quant(
    layer: nn.Module,
    quant_config: QuantizationConfig | None,
    prefix: str,
) -> None:
    """Initializes KV cache scaling factors and quantization method.

    This helper function sets up the KV cache quantization attributes that are
    shared between Attention and MLAAttention layers. It initializes scale
    tensors for query, key, value, and probability, and configures the
    quantization method if applicable.

    Args:
        layer: The attention layer instance to initialize.
        quant_config: Optional quantization configuration.
        prefix: Layer name prefix for quantization method lookup.
    """
    quant_method = (
        quant_config.get_quant_method(layer, prefix=prefix) if quant_config else None
    )

    # Note [Register q/k/v/prob scales in state dict]
    # When calling model.to(device), only parameters/buffers in state dict are
    # moved. If not registering q/k/v/prob scales in state dict, there would
    # be an IMA error when a cuda kernel (e.g., quant_fp8) accesses the tensor
    # on cpu.
    # Registering in state dict means it interacts with weight loading. One edge
    # case is when quant_method is None, or quant_method is UnquantizedLinearMethod
    # (i.e., should_load_quant_weights(quant_method) == False).
    # In this case, the checkpoint does not have the scales. We need to
    # initialize the scales to 1.0 and update the scales after weight loading.
    # This is espectially important when we load dummy weights first (providing
    # wrong scales) and then load real weights (which misses scales and keeps the
    # wrong scales from dummy load).
    set_default_quant_scales(layer, register_buffer=True)

    # The output scale on host memory. This should be the input scale of
    # the quant op after this attention layer.
    layer._o_scale_float = None

    quant_method = (
        quant_config.get_quant_method(layer, prefix=prefix) if quant_config else None
    )

    # See [Note: Register q/k/v/prob scales in state dict]
    if should_load_quant_weights(quant_method):
        assert isinstance(quant_method, BaseKVCacheMethod)
        # TODO (mgoin): kv cache dtype should be specified in the FP8
        # checkpoint config and become the "auto" behavior
        # if layer.kv_cache_dtype == "fp8_e5m2":
        #     raise ValueError("fp8_e5m2 kv-cache is not supported with fp8 checkpoints.")
        # If quantization is enabled, we make "k_scale" and "v_scale"
        # parameters so that it can be loaded from the model checkpoint.
        # The k/v_scale will then be converted back to native float32
        # values after weight loading.
        layer.quant_method = quant_method
        layer.quant_method.create_weights(layer)


class Attention(nn.Module, AttentionLayerBase):
    """Attention layer.

    This class takes query, key, and value tensors as input. The input tensors
    can either contain prompt tokens or generation tokens.
    The class does the following:

    1. Store the input key and value tensors in the KV cache.
    2. Perform (multi-head/multi-query/grouped-query) attention.
    3. Return the output tensor.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        use_alibi_sqrt: bool | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        logits_soft_cap: float | None = None,
        per_layer_sliding_window: int | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        attn_backend: type[AttentionBackend] | None = None,
        head_size_v: int | None = None,
        **extra_impl_args,
    ) -> None:
        """
        The KV cache is stored inside this class and is accessed via
        `self.kv_cache`.
        """
        super().__init__()
        if per_layer_sliding_window is not None:
            # per-layer sliding window
            sliding_window = per_layer_sliding_window
        elif cache_config is not None:
            # model-level sliding window
            sliding_window = cache_config.sliding_window
        else:
            sliding_window = None

        vllm_config = get_current_vllm_config()
        if cache_config is not None:
            kv_cache_dtype = cache_config.cache_dtype
            block_size = cache_config.block_size
            calculate_kv_scales = cache_config.calculate_kv_scales
        else:
            kv_cache_dtype = "auto"
            block_size = 64 if envs.VLLM_USE_FLASH_ATTN_PA and envs.VLLM_USE_FLASH_MLA else 16
            calculate_kv_scales = False

        self.block_size = block_size

        # llm-compressor mdls need to set cache_dtype to "fp8" manually.
        if getattr(quant_config, "kv_cache_scheme", None) is not None:
            kv_cache_dtype = "fp8"
            calculate_kv_scales = False
            if cache_config is not None:
                cache_config.cache_dtype = "fp8"
                cache_config.calculate_kv_scales = False
                
        # Skip quantization for specified layers
        if cache_config is not None and cache_config.kv_cache_dtype_skip_layers:
            from vllm.model_executor.models.utils import extract_layer_index

            skip = False
            # Check attention type
            if (
                sliding_window is not None
                and "sliding_window" in cache_config.kv_cache_dtype_skip_layers
            ):
                skip = True
            # Check layer index
            layer_idx = extract_layer_index(prefix)
            if str(layer_idx) in cache_config.kv_cache_dtype_skip_layers:
                skip = True
            if skip:
                kv_cache_dtype = "auto"
                calculate_kv_scales = False
            logger.info(
                "Layer %s: kv_cache_dtype=%s, sliding_window=%s",
                prefix,
                kv_cache_dtype,
                sliding_window,
            )

        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            kv_cache_dtype, vllm_config.model_config
        )
        self.kv_cache_dtype = kv_cache_dtype
        self.calculate_kv_scales = calculate_kv_scales
        if self.kv_cache_dtype in {"fp8", "fp8_e4m3","fp8_e5m2"} :
            self.check_fp8_overflow = True
        else:
            self.check_fp8_overflow = False
        if num_kv_heads is None:
            num_kv_heads = num_heads
        assert num_heads % num_kv_heads == 0, (
            f"num_heads ({num_heads}) is not divisible by num_kv_heads ({num_kv_heads})"
        )
        self.quant_config = quant_config
        self.layer_name = prefix

        self.num_heads = num_heads
        self.head_size = head_size
        self.head_size_v = self.head_size if head_size_v is None else head_size_v
        self.num_kv_heads = num_kv_heads
        self.sliding_window = sliding_window
        self.has_sink = extra_impl_args.get("sinks") is not None

        # NOTE: model_config may be None during certain tests
        model_config = vllm_config.model_config
        self.use_mm_prefix = model_config is not None and model_config.is_mm_prefix_lm

        # During model initialization, the default dtype is set as the model
        # weight and activation dtype.
        dtype = torch.get_default_dtype()
        if attn_backend is None:
            self.attn_backend = get_attn_backend(
                head_size,
                dtype,
                kv_cache_dtype,
                block_size,
                use_mla=False,
                has_sink=self.has_sink,
                use_mm_prefix=self.use_mm_prefix,
                attn_type=attn_type,
            )
        else:
            self.attn_backend = attn_backend
        backend_supports_alibi_sqrt = self.attn_backend.supports_alibi_sqrt()
        use_alibi_sqrt = use_alibi_sqrt if use_alibi_sqrt else False
        if use_alibi_sqrt and not backend_supports_alibi_sqrt:
            raise ValueError(
                f"use_alibi_sqrt is not supported by backend "
                f"{self.attn_backend.get_name()}."
            )
        self.use_alibi_sqrt = bool(use_alibi_sqrt)
        if backend_supports_alibi_sqrt:
            extra_impl_args["use_alibi_sqrt"] = self.use_alibi_sqrt
        # prefix caching + batch invariance is currently not supported for
        # FLASHINFER and TRITON_MLA.
        if (
            cache_config is not None
            and cache_config.enable_prefix_caching
            and vllm_is_batch_invariant()
            and (
                self.attn_backend.get_name() == "FLASHINFER"
                or self.attn_backend.get_name() == "TRITON_MLA"
            )
        ):
            logger.warning_once(
                "Disabling prefix caching for FLASHINFER/TRITON_MLA "
                "with batch invariance, as it is not yet supported.",
                scope="local",
            )
            cache_config.enable_prefix_caching = False

        impl_cls = self.attn_backend.get_impl_cls()
        self.impl = impl_cls(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **extra_impl_args,
        )
        self.backend = AttentionBackendEnum[self.attn_backend.get_name()]
        self.dtype = dtype

        # For cuda-alike (CUDA and ROCM) and cpu platforms, we control how
        # torch.compile works by registering the attention as one giant
        # opaque custom op. For other platforms, we directly call them
        # and let torch.compile handle them.
        self.use_direct_call = not current_platform.opaque_attention_op()

        self.use_output = self.attn_backend.accept_output_buffer
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        self.attn_type = attn_type

        if kv_sharing_target_layer_name is not None:
            validate_kv_sharing_target(
                prefix,
                kv_sharing_target_layer_name,
                compilation_config.static_forward_context,
            )
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        # use a placeholder kv cache tensor during init, which will be replaced
        # by bind_kv_cache
        # this variable will not be accessed if use_direct_call is True
        self.kv_cache = [
            torch.tensor([])
            for _ in range(vllm_config.parallel_config.pipeline_parallel_size)
        ]

        # Initialize KV cache quantization attributes
        _init_kv_cache_quant(self, quant_config, prefix)

        # Initialize TurboQuant buffers (Pi, S, centroids) if tq cache dtype
        if kv_cache_dtype.startswith("turboquant_"):
            self._init_turboquant_buffers(kv_cache_dtype, head_size, prefix)

        # for attn backends supporting query quantization
        self.query_quant = None
        # @TODO
        if envs.VLLM_USE_QUERY_QUANT:
            if self.impl.supports_quant_query_input and self.kv_cache_dtype.startswith(
                "fp8"
            ):
                is_per_head = (
                    hasattr(self, "q_scale") and self.q_scale.numel() == self.num_kv_heads
                )
                block_size = self.head_size * self.num_heads // self.num_kv_heads
                self.query_quant = QuantFP8(
                    static=True,
                    group_shape=GroupShape(-1, block_size)
                    if is_per_head
                    else GroupShape.PER_TENSOR,
                )

    def _init_turboquant_buffers(
        self, cache_dtype: str, head_size: int, prefix: str
    ) -> None:
        """Initialize TurboQuant rotation/projection matrices and centroids."""
        from vllm.model_executor.layers.quantization.turboquant.centroids import (
            get_centroids,
        )
        from vllm.model_executor.layers.quantization.turboquant.config import (
            TurboQuantConfig,
        )
        from vllm.model_executor.layers.quantization.turboquant.quantizer import (
            generate_wht_signs,
        )

        tq_config = TurboQuantConfig.from_cache_dtype(cache_dtype, head_size)

        # Each layer needs a unique rotation matrix so quantization errors
        # don't correlate across layers. Stride must exceed max head_dim to
        # ensure non-overlapping RNG streams between adjacent layers.
        _TQ_LAYER_SEED_STRIDE = 1337

        from vllm.model_executor.models.utils import extract_layer_index

        layer_idx = extract_layer_index(prefix)
        seed = tq_config.seed + layer_idx * _TQ_LAYER_SEED_STRIDE

        self.register_buffer(
            "_tq_signs",
            generate_wht_signs(head_size, seed=seed),
        )
        self.register_buffer(
            "_tq_centroids",
            get_centroids(head_size, tq_config.centroid_bits),
        )
        self._tq_config = tq_config

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        # For some alternate attention backends like MLA the attention output
        # shape does not match the query shape, so we optionally let the model
        # definition specify the output tensor shape.
        output_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        """
        The KV cache is stored inside this class and is accessed via
        `self.kv_cache`.

        Attention metadata (`attn_metadata`) is set using a context manager in
        the model runner's `execute_model` method. It is accessed via forward
        context using
        `vllm.forward_context.get_forward_context().attn_metadata`.
        """
        if self.calculate_kv_scales or self.check_fp8_overflow:
            torch.ops.vllm.maybe_calc_kv_scales(query, key, value, self.layer_name)
            self.check_fp8_overflow = False
        output_dtype = query.dtype
        if self.query_quant is not None:
            # quantizing with a simple torch operation enables
            # torch.compile to fuse this into previous ops
            # which reduces overheads during decoding.
            # Otherwise queries are quantized using custom ops
            # which causes decoding overheads
            assert self.kv_cache_dtype in {"fp8", "fp8_e4m3"}

            # check if query quantization is supported
            if self.impl.supports_quant_query_input:
                query, _ = self.query_quant(query, self._q_scale)

        if self.use_output:
            if output_shape is None:
                # Handle both 2D [num_tokens, hidden] and
                # 3D [num_tokens, heads, head_dim] query
                num_tokens = query.shape[0]
                output_shape = torch.Size(
                    (num_tokens, self.num_heads * self.head_size_v)
                )
            output = torch.empty(output_shape, dtype=output_dtype, device=query.device)
            hidden_size = output_shape[-1]
            # Reshape the query, key, and value tensors.
            # NOTE(woosuk): We do this outside the custom op to minimize the
            # CPU overheads from the non-CUDA-graph regions.
            query = query.view(-1, self.num_heads, self.head_size)
            output = output.view(-1, self.num_heads, self.head_size_v)
            if key is not None:
                key = key.view(-1, self.num_kv_heads, self.head_size)
            if value is not None:
                value = value.view(-1, self.num_kv_heads, self.head_size_v)
            if self.use_direct_call:
                kv_cache_dummy_dep = None
                if not self.attn_backend.forward_includes_kv_cache_update:
                    kv_cache_dummy_dep = unified_kv_cache_update(
                        key, value, self.layer_name
                    )
                unified_attention_with_output(
                    query,
                    key,
                    value,
                    output,
                    self.layer_name,
                    kv_cache_dummy_dep=kv_cache_dummy_dep,
                )
            else:
                kv_cache_dummy_dep = None
                if not self.attn_backend.forward_includes_kv_cache_update and (
                    # torch can only dispatch custom op if a tensor is passed
                    key is not None or value is not None
                ):
                    kv_cache_dummy_dep = torch.ops.vllm.unified_kv_cache_update(
                        key, value, self.layer_name
                    )
                torch.ops.vllm.unified_attention_with_output(
                    query,
                    key,
                    value,
                    output,
                    self.layer_name,
                    kv_cache_dummy_dep=kv_cache_dummy_dep,
                )
            return output.view(-1, hidden_size)
        else:
            assert self.attn_backend.forward_includes_kv_cache_update, (
                "Split KV cache update not supported when output tensor not provided."
            )
            if self.use_direct_call:
                return unified_attention(query, key, value, self.layer_name)
            else:
                return torch.ops.vllm.unified_attention(
                    query, key, value, self.layer_name
                )

    def calc_kv_scales(self, query, key, value):
        bias=0.0 # add bias to avoid q values are too small(or zeros) and scales are not correct
        if torch.abs(query).max().item() < 0.01:
            if self.kv_cache_dtype in {"fp8_e5m2"}:
                bias = 0.1
            else :
                bias = 1.0
        self._q_scale.copy_(torch.abs(query).max() / self.q_range+bias)
        self._k_scale.copy_(torch.abs(key).max() / self.k_range+bias)
        self._v_scale.copy_(torch.abs(value).max() / self.v_range+bias)
        
        self._q_scale_float = self._q_scale.item()
        self._k_scale_float = self._k_scale.item()
        self._v_scale_float = self._v_scale.item()
        # We only calculate the scales once
        self.calculate_kv_scales = False

    def extra_repr(self) -> str:
        s = f"head_size={self.impl.head_size}"  # type: ignore
        s += f", num_heads={self.impl.num_heads}"  # type: ignore
        s += f", num_kv_heads={self.impl.num_kv_heads}"  # type: ignore
        s += f", scale={self.impl.scale}"  # type: ignore
        s += f", backend={self.impl.__class__.__name__}"
        return s

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        self.impl.process_weights_after_loading(act_dtype)

        # If we should not load quant weights, we initialize the scales to 1.0
        # as the default value. See [Note: Register q/k/v/prob scales in state dict]
        # for more details.
        quant_method = (
            self.quant_config.get_quant_method(self, prefix=self.layer_name)
            if self.quant_config
            else None
        )
        if not should_load_quant_weights(quant_method):
            set_default_quant_scales(self, register_buffer=False)

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # Block size may get updated after model loading, refresh it
        block_size = vllm_config.cache_config.block_size
        # Should not be called for enc-dec or encoder-only attention.
        assert self.attn_type == AttentionType.DECODER
        if self.sliding_window is not None:
            # Backported: speculative-decoding draft heads (e.g. DSpark,
            # DFlash, EAGLE) are independent non-MLA models that legitimately
            # use sliding-window attention even when the *target* model uses
            # MLA. The draft's layers are appended after the target's layers
            # (layer index >= target layer count, e.g. target 78 layers =>
            # draft layers indexed 78, 79, 80) and share the target's
            # `static_forward_context`, so the global `use_mla` flag (which
            # reflects the target) must not be applied to them. Only enforce
            # the MLA/sliding-window exclusivity for the target's own layers.
            is_draft_layer = False
            try:
                import re as _re
                _m = _re.search(r"layers\.(\d+)", self.layer_name)
                if _m is not None:
                    _layer_idx = int(_m.group(1))
                    _target_num_layers = vllm_config.model_config.get_num_layers(
                        vllm_config.parallel_config
                    )
                    is_draft_layer = _layer_idx >= _target_num_layers
            except Exception:
                pass
            assert is_draft_layer or not vllm_config.model_config.use_mla, (
                "MLA is not supported for slidingwindow"
            )
            return SlidingWindowSpec(
                block_size=block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                dtype=self.kv_cache_torch_dtype,
                sliding_window=self.sliding_window,
            )
        elif self.kv_cache_dtype.startswith("turboquant_"):
            from vllm.model_executor.layers.quantization.turboquant.config import (
                TurboQuantConfig,
            )
            from vllm.v1.kv_cache_interface import TQFullAttentionSpec

            tq_config = TurboQuantConfig.from_cache_dtype(
                self.kv_cache_dtype, self.head_size
            )
            return TQFullAttentionSpec(
                block_size=block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                head_size_v=self.head_size,
                dtype=self.kv_cache_torch_dtype,
                tq_slot_size=tq_config.slot_size_aligned,
            )
        else:
            return FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                head_size_v=self.head_size_v,
                dtype=self.kv_cache_torch_dtype,
            )

class FusedQkvSplitRmsNormRopeAttention(Attention):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        use_alibi_sqrt: bool | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        logits_soft_cap: float | None = None,
        per_layer_sliding_window: int | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        attn_backend: type[AttentionBackend] | None = None,
        head_size_v: int | None = None,
        **extra_impl_args,
    ) -> None:
        super().__init__(num_heads, head_size, scale,
                       num_kv_heads, alibi_slopes,
                       use_alibi_sqrt, cache_config,
                       quant_config, logits_soft_cap,
                       per_layer_sliding_window,
                       prefix, attn_type,
                       kv_sharing_target_layer_name,
                       attn_backend,
                       head_size_v,
                       **extra_impl_args)
        
    def forward(
        self,
        qkv: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        weight_q_norm: torch.Tensor,
        weight_k_norm: torch.Tensor,
        epsilon: float,
        # For some alternate attention backends like MLA the attention output
        # shape does not match the query shape, so we optionally let the model
        # definition specify the output tensor shape.
        output_shape: torch.Size | None = None,
        is_neox: bool = False,
    ) -> torch.Tensor:
        """
        The KV cache is stored inside this class and is accessed via
        `self.kv_cache`.

        Attention metadata (`attn_metadata`) is set using a context manager in
        the model runner's `execute_model` method. It is accessed via forward
        context using
        `vllm.forward_context.get_forward_context().attn_metadata`.
        """
        output_dtype = qkv.dtype
        num_tokens = qkv.shape[0]

        if output_shape is None:
            # Handle both 2D [num_tokens, hidden] and
            # 3D [num_tokens, heads, head_dim] query
            output_shape = torch.Size(
                (num_tokens, self.num_heads * self.head_size_v)
            )
        output = torch.empty(output_shape, dtype=output_dtype, device=qkv.device)
        output = output.view(-1, self.num_heads, self.head_size_v)
        hidden_size = output_shape[-1]

        q_size = self.num_heads * self.head_size
        kv_size = self.num_kv_heads * self.head_size
        query, key, value = torch.ops.vllm.fused_qkv_split_rmsnorm_rope_kv_store(qkv=qkv,
                                                 positions=positions,
                                                 layer_name=self.layer_name,
                                                 kv_cache_dtype=self.kv_cache_dtype,
                                                 cos_sin_cache=cos_sin_cache,
                                                 weight_q_norm=weight_q_norm,
                                                 weight_k_norm=weight_k_norm,
                                                 epsilon=epsilon,
                                                 head_size=self.head_size,
                                                 head_size_v=self.head_size_v,
                                                 q_size=q_size,
                                                 kv_size=kv_size,
                                                 block_size=self.block_size,
                                                 is_neox=is_neox)

        kv_cache_dummy_dep = None
        torch.ops.vllm.unified_attention_with_output(
            query,
            key,
            value,
            output,
            self.layer_name,
            kv_cache_dummy_dep=kv_cache_dummy_dep,
        )

        return output.view(-1, hidden_size)

class MLAAttention(nn.Module, AttentionLayerBase):
    """Multi-Head Latent Attention layer.

    This class takes query, and compressed key/value tensors as input.
    The class does the following:

    1. Store the input key and value tensors in the KV cache.
    2. Perform (multi-head/multi-query/grouped-query) attention.
    3. Return the output tensor.
    """

    def __init__(
        self,
        num_heads: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        kv_b_proj: ColumnParallelLinear,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        use_sparse: bool = False,
        indexer: object | None = None,
        **extra_impl_args,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = scale
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.head_size = kv_lora_rank + qk_rope_head_dim
        self.layer_name = prefix

        if cache_config is not None:
            kv_cache_dtype = cache_config.cache_dtype
            block_size = cache_config.block_size
            calculate_kv_scales = cache_config.calculate_kv_scales
        else:
            kv_cache_dtype = "auto"
            block_size = 16
            calculate_kv_scales = False
        self.quant_config = quant_config

        # Initialize KV cache quantization attributes
        self.kv_cache_dtype = kv_cache_dtype
        self.calculate_kv_scales = calculate_kv_scales
        _init_kv_cache_quant(self, quant_config, prefix)

        dtype = torch.get_default_dtype()
        self.attn_backend = get_attn_backend(
            self.head_size,
            dtype,
            kv_cache_dtype,
            block_size,
            use_mla=True,
            use_sparse=use_sparse,
        )
        if (
            cache_config is not None
            and cache_config.enable_prefix_caching
            and vllm_is_batch_invariant()
            and (
                self.attn_backend.get_name() == "TRITON_MLA"
                or self.attn_backend.get_name() == "FLASHINFER"
            )
        ):
            logger.warning_once(
                "Disabling prefix caching for TRITON_MLA / FLASHINFER "
                "with batch invariance, as it is not yet supported.",
                scope="local",
            )
            cache_config.enable_prefix_caching = False

        impl_cls = cast(type[MLAAttentionImpl], self.attn_backend.get_impl_cls())
        self.impl = impl_cls(
            num_heads=self.num_heads,
            head_size=self.head_size,
            scale=self.scale,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype=self.kv_cache_dtype,
            logits_soft_cap=None,
            attn_type=AttentionType.DECODER,
            kv_sharing_target_layer_name=None,
            # MLA Args
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            qk_head_dim=self.qk_nope_head_dim + self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            kv_b_proj=kv_b_proj,
            indexer=indexer,
            **extra_impl_args,
        )

        self.use_direct_call = not current_platform.opaque_attention_op()

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        self.kv_cache = [
            torch.tensor([])
            for _ in range(
                get_current_vllm_config().parallel_config.pipeline_parallel_size
            )
        ]

        self.use_sparse = use_sparse

        # Initialize q/k/v range constants.
        self.q_range = torch.tensor(envs.Q_SCALE_CONSTANT, dtype=torch.float32)
        self.k_range = torch.tensor(envs.K_SCALE_CONSTANT, dtype=torch.float32)
        self.v_range = torch.tensor(envs.V_SCALE_CONSTANT, dtype=torch.float32)

    def forward(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        output_shape: torch.Size | None = None,
        q_ori: torch.Tensor | None = None,
        key_normed: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        weight: torch.Tensor | None = None,
        cos_sin_cache: torch.Tensor | None = None,
        epsilon: float | None = None,
    ) -> torch.Tensor:
        # NOTE: fused path computes kv_c_normed inside the attention impl.
        if self.calculate_kv_scales and q_ori is None:
            torch.ops.vllm.maybe_calc_kv_scales(q, kv_c_normed, k_pe, self.layer_name)

        extra_kwargs: dict[str, object] = {}
        if q_ori is not None:
            extra_kwargs["q_ori"] = q_ori
        if key_normed is not None:
            extra_kwargs["key_normed"] = key_normed
        if positions is not None:
            extra_kwargs["positions"] = positions
        if weight is not None:
            extra_kwargs["weight"] = weight
        if cos_sin_cache is not None:
            extra_kwargs["cos_sin_cache"] = cos_sin_cache
        if epsilon is not None:
            extra_kwargs["epsilon"] = epsilon

        if self.use_direct_call:
            forward_context: ForwardContext = get_forward_context()
            attn_metadata = forward_context.attn_metadata
            if isinstance(attn_metadata, dict):
                attn_metadata = attn_metadata[self.layer_name]
            self_kv_cache = self.kv_cache[forward_context.virtual_engine]

            if self.attn_backend.accept_output_buffer:
                output = torch.empty(output_shape, dtype=q.dtype, device=q.device)
                self.impl.forward(
                    self,
                    q,
                    kv_c_normed,
                    k_pe,
                    self_kv_cache,
                    attn_metadata,
                    output=output,
                )
                return output
            else:
                return self.impl.forward(
                    self, q, kv_c_normed, k_pe, self_kv_cache, attn_metadata
                )
        else:
            if self.attn_backend.accept_output_buffer:
                output = torch.empty(output_shape, dtype=q.dtype, device=q.device)
                if not extra_kwargs:
                    torch.ops.vllm.unified_mla_attention_with_output(
                        q, kv_c_normed, k_pe, output, self.layer_name
                    )
                else:
                    torch.ops.vllm.unified_mla_attention_with_output(
                        q,
                        kv_c_normed,
                        k_pe,
                        output,
                        self.layer_name,
                        None,
                        None,
                        q_ori,
                        key_normed,
                        positions,
                        weight,
                        cos_sin_cache,
                        epsilon,
                    )
                return output
            else:
                return torch.ops.vllm.unified_mla_attention(
                    q,
                    kv_c_normed,
                    k_pe,
                    self.layer_name,
                )

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        if hasattr(self.impl, "process_weights_after_loading"):
            self.impl.process_weights_after_loading(act_dtype)

        # If we should not load quant weights, we initialize the scales to 1.0
        # as the default value. See [Note: Register q/k/v/prob scales in state dict]
        # for more details.
        quant_method = (
            self.quant_config.get_quant_method(self, prefix=self.layer_name)
            if self.quant_config
            else None
        )
        if not should_load_quant_weights(quant_method):
            set_default_quant_scales(self, register_buffer=False)

    def calc_kv_scales(
        self, q: torch.Tensor, kv_c_normed: torch.Tensor, k_pe: torch.Tensor
    ) -> None:
        """Optional scale calculation for MLA inputs.

        Mirrors Attention.calc_kv_scales. Not all MLA backends require this
        """
        # Use safe defaults if ranges are not present
        q_range = getattr(self, "q_range", torch.tensor(1.0))
        k_range = getattr(self, "k_range", torch.tensor(1.0))
        v_range = getattr(self, "v_range", torch.tensor(1.0))

        self._q_scale.copy_(torch.abs(q).max() / q_range)
        # kv_c_normed is the compressed KV representation; use it for k/v
        kv_abs_max = torch.abs(kv_c_normed).max()
        self._k_scale.copy_(kv_abs_max / k_range)
        self._v_scale.copy_(kv_abs_max / v_range)
        self._q_scale_float = self._q_scale.item()
        self._k_scale_float = self._k_scale.item()
        self._v_scale_float = self._v_scale.item()
        self.calculate_kv_scales = False

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        kv_cache_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, vllm_config.model_config
        )
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_size,
            dtype=kv_cache_dtype,
            cache_dtype_str=vllm_config.cache_config.cache_dtype,
        )


def maybe_calc_kv_scales(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]

    # Only calculate if the layer's calculate_kv_scales flag is True
    # This flag gets set to False after the first forward pass
    if self.check_fp8_overflow :
        if self.kv_cache_dtype in {"fp8", "fp8_e4m3"} and torch.abs(query).max().item()>200 : #check fp8  overflow
            self.calculate_kv_scales = True
        if self.kv_cache_dtype in {"fp8_e5m2"} and torch.abs(query).max().item()<0.01 : #check fp8 too small
            self.calculate_kv_scales = True
    if not self.calculate_kv_scales:
        return

    self.calc_kv_scales(query, key, value)


def maybe_calc_kv_scales_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="maybe_calc_kv_scales",
    op_func=maybe_calc_kv_scales,
    mutates_args=[],
    fake_impl=maybe_calc_kv_scales_fake,
)


def get_attention_context(
    layer_name: str,
) -> tuple[dict | object | None, Attention | MLAAttention, torch.Tensor]:
    """Extract attention context for a given layer.

    This helper function extracts the attention metadata, attention layer
    instance, and KV cache tensor for a specific layer.

    Args:
        layer_name: The name/identifier of the attention layer.

    Returns:
        A tuple containing:
        - attn_metadata: Attention metadata for this specific layer, or None if
            no metadata available
        - attn_layer: The attention layer instance (Attention or MLAAttention)
        - kv_cache: The KV cache tensor for current virtual engine

        Note: attn_metadata may be None, but attn_layer and kv_cache are always
        extracted from the forward context.
    """
    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if isinstance(attn_metadata, dict):
        attn_metadata = attn_metadata[layer_name]
    attn_layer: Attention | MLAAttention = forward_context.no_compile_layers[layer_name]
    kv_cache = attn_layer.kv_cache[forward_context.virtual_engine]
    return attn_metadata, attn_layer, kv_cache


@maybe_transfer_kv_layer
def unified_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    attn_metadata, self, kv_cache = get_attention_context(layer_name)
    output = self.impl.forward(self, query, key, value, kv_cache, attn_metadata)

    return output


def unified_attention_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    return torch.empty_like(query).contiguous()


direct_register_custom_op(
    op_name="unified_attention",
    op_func=unified_attention,
    fake_impl=unified_attention_fake,
)


def unified_kv_cache_update(
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    """
    Returns a dummy that is passed to unified_attention to signal a side effect and
    the data dependency between them to ensure torch.compile preserves ordering.
    """
    forward_context = get_forward_context()
    attn_layer = forward_context.no_compile_layers[layer_name]
    kv_cache = attn_layer.kv_cache[forward_context.virtual_engine]

    slot_mapping = forward_context.slot_mapping
    assert isinstance(slot_mapping, dict), (
        f"Expected slot_mapping to be a dict, got {type(slot_mapping)}. "
    )
    layer_slot_mapping = slot_mapping.get(layer_name)
    if layer_slot_mapping is not None:
        assert hasattr(attn_layer.impl, "do_kv_cache_update"), (
            f"{attn_layer.impl.__class__.__name__} does not support kv cache update"
        )
        attn_layer.impl.do_kv_cache_update(
            attn_layer,
            key,
            value,
            kv_cache,
            layer_slot_mapping,
        )

    if current_platform.is_rocm():
        return torch.empty(0, device=key.device, dtype=key.dtype)
    else:
        return torch.empty(0, device=kv_cache.device, dtype=kv_cache.dtype)


def unified_kv_cache_update_fake(
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    return torch.empty(0, device=key.device, dtype=key.dtype)


direct_register_custom_op(
    op_name="unified_kv_cache_update",
    op_func=unified_kv_cache_update,
    fake_impl=unified_kv_cache_update_fake,
    mutates_args=[],
)


@maybe_transfer_kv_layer
def unified_attention_with_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    output_scale: torch.Tensor | None = None,
    output_block_scale: torch.Tensor | None = None,
    kv_cache_dummy_dep: torch.Tensor | None = None,
) -> None:
    # kv_cache_dummy_dep is not used but accepting it creates a data dependency
    # that ensures torch.compile preserves ordering between KV cache update and
    # attention forward.
    del kv_cache_dummy_dep
    attn_metadata, self, kv_cache = get_attention_context(layer_name)

    self.impl.forward(
        self,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=output,
        output_scale=output_scale,
        output_block_scale=output_block_scale,
    )


def unified_attention_with_output_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    output_scale: torch.Tensor | None = None,
    output_block_scale: torch.Tensor | None = None,
    kv_cache_dummy_dep: torch.Tensor | None = None,
) -> None:
    return


direct_register_custom_op(
    op_name="unified_attention_with_output",
    op_func=unified_attention_with_output,
    mutates_args=["output", "output_block_scale"],
    fake_impl=unified_attention_with_output_fake,
)


@maybe_transfer_kv_layer
def unified_mla_attention(
    q: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    attn_metadata, self, kv_cache = get_attention_context(layer_name)
    output = self.impl.forward(self, q, kv_c_normed, k_pe, kv_cache, attn_metadata)

    return output


def unified_mla_attention_fake(
    q: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    return torch.empty_like(q).contiguous()


direct_register_custom_op(
    op_name="unified_mla_attention",
    op_func=unified_mla_attention,
    mutates_args=[],
    fake_impl=unified_mla_attention_fake,
    dispatch_key=current_platform.dispatch_key,
)


@maybe_transfer_kv_layer
def unified_mla_attention_with_output(
    q: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    output_scale: torch.Tensor | None = None,
    output_block_scale: torch.Tensor | None = None,
    q_ori: torch.Tensor | None = None,
    key_normed: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    weight: torch.Tensor | None = None,
    cos_sin_cache: torch.Tensor | None = None,
    epsilon: float | None = None,
) -> None:
    attn_metadata, self, kv_cache = get_attention_context(layer_name)

    extra_kwargs: dict[str, object] = {}
    if q_ori is not None:
        extra_kwargs["q_ori"] = q_ori
    if key_normed is not None:
        extra_kwargs["key_normed"] = key_normed
    if positions is not None:
        extra_kwargs["positions"] = positions
    if weight is not None:
        extra_kwargs["weight"] = weight
    if cos_sin_cache is not None:
        extra_kwargs["cos_sin_cache"] = cos_sin_cache
    if epsilon is not None:
        extra_kwargs["epsilon"] = epsilon

    self.impl.forward(
        self,
        q,
        kv_c_normed,
        k_pe,
        kv_cache,
        attn_metadata,
        output=output,
        output_scale=output_scale,
        output_block_scale=output_block_scale,
        **extra_kwargs,
    )


def unified_mla_attention_with_output_fake(
    q: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    output_scale: torch.Tensor | None = None,
    output_block_scale: torch.Tensor | None = None,
    q_ori: torch.Tensor | None = None,
    key_normed: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    weight: torch.Tensor | None = None,
    cos_sin_cache: torch.Tensor | None = None,
    epsilon: float | None = None,
) -> None:
    return


direct_register_custom_op(
    op_name="unified_mla_attention_with_output",
    op_func=unified_mla_attention_with_output,
    mutates_args=["output", "output_block_scale"],
    fake_impl=unified_mla_attention_with_output_fake,
    dispatch_key=current_platform.dispatch_key,
)

def fused_qkv_split_rmsnorm_rope_kv_store_impl(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    layer_name: str,
    kv_cache_dtype: str,
    cos_sin_cache: torch.Tensor,
    weight_q_norm: torch.Tensor,
    weight_k_norm: torch.Tensor,
    epsilon: float,
    head_size: int,
    head_size_v: int,
    q_size: int,
    kv_size: int,
    block_size: int,
    is_neox: bool = False)-> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    num_tokens = qkv.shape[0]
    forward_context = get_forward_context()
    
    slot_mapping = forward_context.slot_mapping
    layer_slot_mapping = slot_mapping.get(layer_name)
    assert isinstance(slot_mapping, dict), (
        f"Expected slot_mapping to be a dict, got {type(slot_mapping)}. "
    )
    
    attn_layer = forward_context.no_compile_layers[layer_name]
    kv_cache = attn_layer.kv_cache[forward_context.virtual_engine]

    if layer_slot_mapping is not None:
        if current_platform.is_rocm():
            key_cache, value_cache = kv_cache
        else:
            key_cache, value_cache = kv_cache.unbind(0)

        if kv_cache_dtype.startswith("fp8"):
            # queries are quantized in the attention layer
            from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
            kv_cache_dtype = FlashAttentionBackend.get_fp8_dtype_for_flashattn(
                kv_cache_dtype
            )
            key_cache = key_cache.view(kv_cache_dtype)
            value_cache = value_cache.view(kv_cache_dtype)
    else:
        key_cache = torch.empty([0], device=qkv.device, dtype=qkv.dtype)
        value_cache = torch.empty([0], device=qkv.device, dtype=qkv.dtype)

    from lightop import split_qkv_rms_rotary_embedding_fuse_with_kv_store_quant
    q, k, v = split_qkv_rms_rotary_embedding_fuse_with_kv_store_quant(positions,
                                                        qkv.contiguous(),
                                                        q_size,
                                                        kv_size,
                                                        cos_sin_cache,
                                                        head_dim=head_size,
                                                        page_size=block_size,
                                                        k_buffer=key_cache,
                                                        v_buffer=value_cache,
                                                        kv_cache_loc=layer_slot_mapping,
                                                        is_neox=is_neox,
                                                        weight_q=weight_q_norm,
                                                        weight_k=weight_k_norm,
                                                        output_dtype=qkv.dtype,
                                                        kv_cache_dtype=kv_cache_dtype,
                                                        epsilon=epsilon,
                                                        residual_q=None,
                                                        residual_k=None,
                                                        k_scale=None,
                                                        v_scale=None,
                                                        )
    q = q.contiguous().view(num_tokens, q_size//head_size, head_size)
    k = k.contiguous().view(num_tokens, kv_size//head_size_v, head_size_v)
    v = v.contiguous().view(num_tokens, kv_size//head_size_v, head_size_v)
    return q, k ,v


def fused_qkv_split_rmsnorm_rope_kv_store_fake(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    layer_name: str,
    kv_cache_dtype: str,
    cos_sin_cache: torch.Tensor,
    weight_q_norm: torch.Tensor,
    weight_k_norm: torch.Tensor,
    epsilon: float,
    head_size: int,
    head_size_v: int,
    q_size: int,
    kv_size: int,
    block_size: int,
    is_neox: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_token = qkv.shape[0]
    q = torch.empty((num_token, q_size//head_size, head_size), device=qkv.device, dtype=qkv.dtype)
    k = torch.empty((num_token, kv_size//head_size_v, head_size_v), device=qkv.device, dtype=qkv.dtype)
    v = torch.empty((num_token, kv_size//head_size_v, head_size_v), device=qkv.device, dtype=qkv.dtype)
    return q, k, v

direct_register_custom_op(
    op_name="fused_qkv_split_rmsnorm_rope_kv_store",
    op_func=fused_qkv_split_rmsnorm_rope_kv_store_impl,
    mutates_args=["qkv", "positions"],
    fake_impl=fused_qkv_split_rmsnorm_rope_kv_store_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)
