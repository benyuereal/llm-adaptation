# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import os
import torch

from vllm.attention.layer import MLAAttention
from vllm.config import CacheConfig
import vllm.envs as envs
from vllm.forward_context import get_forward_context

from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.distributed import (
    tensor_model_parallel_all_gather,
)


@dataclass
class MLAModules:
    """Modules used in MLA."""

    kv_a_layernorm: torch.nn.Module
    kv_b_proj: torch.nn.Module
    rotary_emb: torch.nn.Module
    o_proj: torch.nn.Module
    fused_qkv_a_proj: torch.nn.Module | None
    kv_a_proj_with_mqa: torch.nn.Module | None
    q_a_layernorm: torch.nn.Module | None
    q_b_proj: torch.nn.Module | None
    q_proj: torch.nn.Module | None
    indexer: torch.nn.Module | None
    is_sparse: bool
    topk_indices_buffer: torch.Tensor | None
    indexer_rotary_emb: torch.nn.Module | None = None


# --8<-- [start:multi_head_latent_attention]
@PluggableLayer.register("multi_head_latent_attention")
class MultiHeadLatentAttentionWrapper(PluggableLayer):
    """Pluggable MLA layer which allows OOT backends to add
    custom implementations of the outer MLA layer (including rope & o_proj).
    Note that currently oot platforms can still use CustomOp.register_oot to
    replace MLA layer entirly, although we use PluggableLayer to register
    this layer now.

    This class takes positions and hidden_states as input.
    The input tensors can either contain prefill tokens or decode tokens.
    The class does the following:

    1. MLA Preprocess.
    2. Perform multi-head attention to prefill tokens and
       multi-query attention to decode tokens separately.
    3. Return the output tensor.
    """

    # --8<-- [end:multi_head_latent_attention]

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        mla_modules: MLAModules,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        skip_topk: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        self.fused_qkv_a_proj = mla_modules.fused_qkv_a_proj
        self.kv_a_proj_with_mqa = mla_modules.kv_a_proj_with_mqa
        self.q_a_layernorm = mla_modules.q_a_layernorm
        self.q_b_proj = mla_modules.q_b_proj
        self.q_proj = mla_modules.q_proj
        self.kv_a_layernorm = mla_modules.kv_a_layernorm
        self.kv_b_proj = mla_modules.kv_b_proj
        self.rotary_emb = mla_modules.rotary_emb
        self.o_proj = mla_modules.o_proj
        self.indexer = mla_modules.indexer
        self.indexer_rope_emb = mla_modules.indexer_rotary_emb
        self.is_sparse = mla_modules.is_sparse
        # GLM-5.2: shared 层 skip_topk=True, 不跑 indexer, 直接复用 full 层
        # 写入的共享 topk_indices_buffer.
        self.skip_topk = skip_topk

        if self.indexer is not None:
            assert hasattr(self.indexer, "topk_tokens")
            self.topk_tokens = self.indexer.topk_tokens
        # shared 层虽无 indexer, 仍需读取共享 buffer.
        self.topk_indices_buffer = mla_modules.topk_indices_buffer

        self.mla_attn = MLAAttention(
            num_heads=self.num_heads,
            scale=scale,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            kv_b_proj=self.kv_b_proj,
            use_sparse=self.is_sparse,
            indexer=self.indexer,
            topk_indice_buffer=self.topk_indices_buffer,
        )

        self.prefix = prefix

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
        *, iqis: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> torch.Tensor:
        q_c = None
        kv_lora = None

        if self.q_lora_rank is not None:
            assert self.fused_qkv_a_proj is not None, (
                "fused_qkv_a_proj is required when q_lora_rank is not None"
            )
            assert self.q_a_layernorm is not None, (
                "q_a_layernorm is required when q_lora_rank is not None"
            )
            assert self.q_b_proj is not None, (
                "q_b_proj is required when q_lora_rank is not None"
            )
            if envs.USE_FUSED_RMS_QUANT and iqis is not None:
                qkv_lora = self.fused_qkv_a_proj(hidden_states, iqis=iqis)[0]
            else:
                qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
            q_c, kv_lora = qkv_lora.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
            if envs.USE_FUSED_RMS_QUANT:
                qa_iq, qa_is, _ = self.q_a_layernorm(x=q_c,
                                                     residual=None, 
                                                     quant_dtype=torch.int8,
                                                     update_input=False)
                q = self.q_b_proj(q_c, iqis=(qa_iq, qa_is))[0]
                
            else:
                q_c = self.q_a_layernorm(q_c)
                q = self.q_b_proj(q_c)[0]
        else:
            assert self.kv_a_proj_with_mqa is not None, (
                "kv_a_proj_with_mqa is required when q_lora_rank is None"
            )
            assert self.q_proj is not None, (
                "q_proj is required when q_lora_rank is None"
            )
            kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
            q = self.q_proj(hidden_states)[0]

        kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_cache_dtype = getattr(self.mla_attn, "kv_cache_dtype", "auto")
        calculate_kv_scales = getattr(self.mla_attn, "calculate_kv_scales", False)

        if not envs.VLLM_USE_LIGHTOP_RMS_ROPE_CONCAT:
            kv_c_normed = self.kv_a_layernorm(kv_c)

        q = q.view(-1, self.num_heads, self.qk_head_dim)
        # Add head dim of 1 to k_pe
        k_pe = k_pe.unsqueeze(1)

        # if not use_fused_rms_rope_concat and self.rotary_emb is not None:
        if not envs.VLLM_USE_LIGHTOP_RMS_ROPE_CONCAT and self.rotary_emb is not None:
            q[..., self.qk_nope_head_dim:], k_pe = self.rotary_emb(
                positions, q[..., self.qk_nope_head_dim:], k_pe
            )

        if self.indexer is not None and self.is_sparse and not self.skip_topk:
            if envs.USE_FUSED_RMS_QUANT and iqis is not None:
                _topk_indices = self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb, iqis=iqis)
            else:
                _topk_indices = self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb)
        # GLM-5.2: shared 层 (skip_topk=True) 不跑 indexer, 直接复用 full 层
        # 写入的共享 topk_indices_buffer, 此处无需任何操作.

        if llama_4_scaling is not None:
            q *= llama_4_scaling

        enable_lightly_cp = get_forward_context().enable_lightly_cp
        enable_lightly_cplb = get_forward_context().enable_lightly_cplb

        # if not use_fused_rms_rope_concat:
        if not envs.VLLM_USE_LIGHTOP_RMS_ROPE_CONCAT:
            if enable_lightly_cp:
                kv_c_normed = tensor_model_parallel_all_gather(
                    kv_c_normed.contiguous(), 0
                )
                k_pe = tensor_model_parallel_all_gather(
                    k_pe.contiguous(), 0
                )

                gather_indexes_tensor = get_forward_context().gather_indexes_tensor
                if enable_lightly_cplb and gather_indexes_tensor is not None:
                    # Reorder kv after pcp allgather.
                    kv_c_normed = torch.index_select(kv_c_normed, 0, gather_indexes_tensor)
                    k_pe = torch.index_select(k_pe, 0, gather_indexes_tensor)

            attn_out = self.mla_attn(
                q,
                kv_c_normed,
                k_pe,
                output_shape=(hidden_states.shape[0],
                              self.num_heads * self.v_head_dim),
            )
        else:
            # Lightop fused path:
            # - kv_c is passed as "unnormed" and written to kv_cache by the backend.
            # - key_normed is an output buffer filled by the fused op and then
            #   used for the prefill path.
            # Keep kv_c/k_pe as views into the original kv_lora buffer so they
            # share the same row stride. The lightop fused op requires
            # `kv_c.stride(0) == k_pe.stride(0)`, which is not true if we make
            # kv_c individually contiguous.
            key_normed = torch.empty_like(kv_c,
                                          memory_format=torch.contiguous_format)
            weight = getattr(self.kv_a_layernorm, "weight", None)
            epsilon = getattr(self.kv_a_layernorm, "variance_epsilon", 1e-6)
            if weight is None:
                raise RuntimeError(
                    "VLLM_USE_LIGHTOP_RMS_ROPE_CONCAT requires kv_a_layernorm "
                    "to have a 'weight' parameter."
                )
            # Keep cos_sin_cache on the same device/dtype as q.
            if hasattr(self.rotary_emb, "_match_cos_sin_cache_dtype"):
                # type: ignore[attr-defined]
                self.rotary_emb._match_cos_sin_cache_dtype(q)
            cos_sin_cache = getattr(self.rotary_emb, "cos_sin_cache", None)
            if cos_sin_cache is None:
                raise RuntimeError(
                    "VLLM_USE_LIGHTOP_RMS_ROPE_CONCAT requires rotary_emb to "
                    "expose 'cos_sin_cache'."
                )

            if enable_lightly_cp:
                kv_c = tensor_model_parallel_all_gather(
                    kv_c.contiguous(), 0
                )
                k_pe = tensor_model_parallel_all_gather(
                    k_pe.contiguous(), 0
                )
                gather_indexes_tensor = get_forward_context().gather_indexes_tensor
                if enable_lightly_cplb and gather_indexes_tensor is not None:
                    # Reorder kv after pcp allgather.
                    kv_c = torch.index_select(kv_c, 0, gather_indexes_tensor)
                    k_pe = torch.index_select(k_pe, 0, gather_indexes_tensor)

            attn_out = self.mla_attn(
                q[..., self.qk_nope_head_dim:],
                kv_c,
                k_pe,
                output_shape=(hidden_states.shape[0],
                              self.num_heads * self.v_head_dim),
                q_ori=q,
                key_normed=key_normed,
                positions=positions,
                weight=weight,
                cos_sin_cache=cos_sin_cache,
                epsilon=epsilon,
            )

        _oout = self.o_proj(attn_out)[0]
        return _oout
