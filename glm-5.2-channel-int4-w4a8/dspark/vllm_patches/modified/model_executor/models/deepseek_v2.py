# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only DeepseekV2/DeepseekV3 model."""
import os
import re
from vllm import envs

import typing
from collections.abc import Callable, Iterable
from itertools import islice

import torch
from torch import nn
from transformers import DeepseekV2Config, DeepseekV3Config

from vllm._aiter_ops import rocm_aiter_ops
from vllm.attention.layer import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ParallelConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.forward_context import get_forward_context
from vllm.distributed.communication_op import tensor_model_parallel_all_reduce, tensor_model_parallel_reduce_scatter
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.fused_moe import SharedFusedMoE
from vllm.model_executor.layers.layernorm import LayerNorm, RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mla import MLAModules, MultiHeadLatentAttentionWrapper
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerBackend,
)
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec

from .interfaces import (
    MixtureOfExperts,
    SupportsEagle,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
)
from .utils import (
    PPMissingLayer,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm import _custom_ops as ops
from vllm.utils import W8a8GetCacheJSON

from vllm.model_executor.layers.layernorm import FusedRMSNormQuant

logger = init_logger(__name__)


class DeepseekAttention(nn.Module):
    """Normal MHA implementation used by Deepseek v1."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config,
        hidden_size: int,
        num_heads: int,
        max_position_embeddings: int = 8192,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        **kwargs,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=config.rope_parameters,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        *, iqis: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


def iqis_all_gather(
    iqis: tuple[torch.Tensor, torch.Tensor],
    tp_size: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    assert iqis is not None
    iq_tensor, is_tensor = iqis
    assert isinstance(iq_tensor, torch.Tensor)
    assert isinstance(is_tensor, torch.Tensor)
    assert iq_tensor.dtype == torch.int8, f"iq_tensor dtype is {iq_tensor.dtype}"
    assert is_tensor.dtype == torch.float32, f"is_tensor dtype is {is_tensor.dtype}"
    assert iq_tensor.dim() == 2
    assert is_tensor.dim() == 2
    
    m_local, n = iq_tensor.shape
    assert is_tensor.shape[0] == m_local, f"{is_tensor.shape[0]} != {iq_tensor.shape[0]}"
    assert is_tensor.shape[1] == 1, f"is_tensor dim 1 ={is_tensor.shape[1]}"
    
    iq_int8_2d = iq_tensor.view(torch.int8)
    is_int8_2d = is_tensor.view(torch.int8)
    combined_2d = torch.cat([iq_int8_2d, is_int8_2d], dim=1) # [m_local, n + 4]
    
    if not combined_2d.is_contiguous():
        combined_2d = combined_2d.contiguous()
    
    combined_gathered = tensor_model_parallel_all_gather(combined_2d, dim=0)
    split_idx = n
    iq_gathered_int8 = combined_gathered[:, :split_idx].contiguous()
    is_gathered_int8 = combined_gathered[:, split_idx:].contiguous()
    
    iq_gathered = iq_gathered_int8.view(torch.int8)
    assert iq_gathered.shape[0] == m_local * tp_size, f"iq_gathered dim0= {iq_gathered.shape[0]}, expected {m_local * tp_size}"
    # is_gathered_int8 should be [m_local*tp_size, 4]
    assert is_gathered_int8.shape[0] == m_local * tp_size, f"is_gathered_int8 dim0= {is_gathered_int8.shape[0]}, expected {m_local * tp_size}"
    assert is_gathered_int8.shape[1] == 4, f"is_gathered_int8 dim1= {is_gathered_int8.shape[1]}"
    is_gathered = is_gathered_int8.view(torch.float32)
    return (iq_gathered, is_gathered)


class DeepseekV2MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        is_sequence_parallel=False,
        prefix: str = "",
    ) -> None:
        super().__init__()

        # If is_sequence_parallel, the input and output tensors are sharded
        # across the ranks within the tp_group. In this case the weights are
        # replicated and no collective ops are needed.
        # Otherwise we use standard TP with an allreduce at the end.
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            #reduce_results=reduce_results,
            reduce_results=False,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.down_proj",
        )
        self.tp_size = get_tensor_model_parallel_world_size()
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, 
                x,
                *, iqis: tuple[torch.Tensor, torch.Tensor] | None = None
                ):
        enable_lightly_cp = get_forward_context().enable_lightly_cp
        if enable_lightly_cp: 
            if iqis is not None and iqis[0] is not None and iqis[1] is not None:
                iqis = iqis_all_gather(iqis, tp_size=self.tp_size)
            else:
                x = tensor_model_parallel_all_gather(x.contiguous(), 0)

        if envs.USE_FUSED_RMS_QUANT:
            gate_up, _ = self.gate_up_proj(x, iqis=iqis)
            if envs.USE_FUSED_SILU_MUL_QUANT:
                from lmslim.quantize.quant_ops import lm_fuse_silu_mul_quant
                xq, xs = lm_fuse_silu_mul_quant(gate_up)
                x, _ = self.down_proj(gate_up, iqis=(xq, xs))
            else:
                x = self.act_fn(gate_up)
                x, _ = self.down_proj(x)
        else:
            gate_up, _ = self.gate_up_proj(x)
            x = self.act_fn(gate_up)
            x, _ = self.down_proj(x)

        if enable_lightly_cp:
            x = tensor_model_parallel_reduce_scatter(x.contiguous(), dim=0)
            return x
        elif self.tp_size > 1:
            x = tensor_model_parallel_all_reduce(x)
        return x
    

class DeepseekV2SharedMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        is_sequence_parallel=False,
        prefix: str = "",
    ) -> None:
        super().__init__()

        # If is_sequence_parallel, the input and output tensors are sharded
        # across the ranks within the tp_group. In this case the weights are
        # replicated and no collective ops are needed.
        # Otherwise we use standard TP with an allreduce at the end.
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.down_proj"
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, 
                x,
                *, iqis: tuple[torch.Tensor, torch.Tensor] | None = None
                ):
        if envs.USE_FUSED_RMS_QUANT:
            gate_up, _ = self.gate_up_proj(x, iqis=iqis)
            if envs.USE_FUSED_SILU_MUL_QUANT:
                from lmslim.quantize.quant_ops import lm_fuse_silu_mul_quant
                xq, xs = lm_fuse_silu_mul_quant(gate_up)
                x, _ = self.down_proj(gate_up, iqis=(xq, xs))
            else:
                x = self.act_fn(gate_up)
                x, _ = self.down_proj(x)
        else:
            gate_up, _ = self.gate_up_proj(x)
            x = self.act_fn(gate_up)
            x, _ = self.down_proj(x)
        return x


class DeepseekV2MoE(nn.Module):
    def __init__(
        self,
        config: DeepseekV2Config | DeepseekV3Config,
        parallel_config: ParallelConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)

        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts: int = config.n_routed_experts
        self.n_shared_experts: int = config.n_shared_experts

        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.n_routed_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        if getattr(config, "topk_method", None) == "noaux_tc":
            if envs.VLLM_ENABLE_MOE_FUSED_GATE:
                # avoid moe_fused_gate precision error
                self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts))
            else:
                self.gate.e_score_correction_bias = nn.Parameter(
                    torch.empty(config.n_routed_experts, dtype=torch.float32)
                )
        else:
            self.gate.e_score_correction_bias = None

        # Load balancing settings.
        eplb_config = parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb

        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_logical_experts = self.n_routed_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        self.is_rocm_aiter_moe_enabled = rocm_aiter_ops.is_fused_moe_enabled()
        self.is_fusion_moe_shared_experts_enabled = (
            rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
        )
        if config.n_shared_experts is None or self.is_fusion_moe_shared_experts_enabled:
            self.shared_experts = None
        else:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts

            self.shared_experts = DeepseekV2SharedMLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                is_sequence_parallel=self.is_sequence_parallel,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )

        self.experts = SharedFusedMoE(
            shared_experts=self.shared_experts,
            gate=self.gate,
            # num_experts=config.n_routed_experts,
            # top_k=config.num_experts_per_tok,
            num_experts=config.n_routed_experts 
                    + (config.n_shared_experts if self.is_fusion_moe_shared_experts_enabled else 0),
            top_k = config.num_experts_per_tok
                    + (config.n_shared_experts if self.is_fusion_moe_shared_experts_enabled else 0),
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            use_grouped_topk=True,
            num_expert_group=getattr(config, "n_group", 1),
            topk_group=getattr(config, "topk_group", 1),
            prefix=f"{prefix}.experts",
            scoring_func=getattr(config, "scoring_func", "softmax"),
            # we do scaling outside, set factor to 1.0 to avoid double mul
            # aiter applies routed_scaling_factor internally
            routed_scaling_factor=1.0
            if not self.is_rocm_aiter_moe_enabled
            else self.routed_scaling_factor,
            e_score_correction_bias=self.gate.e_score_correction_bias,
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
            n_shared_experts=config.n_shared_experts
            if self.is_fusion_moe_shared_experts_enabled
            else None,
        )

    def forward(self, hidden_states: torch.Tensor,
                *, iqis: tuple[torch.Tensor, torch.Tensor] | None = None
                ) -> torch.Tensor:
        enable_lightly_cp = get_forward_context().enable_lightly_cp
        if enable_lightly_cp:
            hidden_states = tensor_model_parallel_all_gather(
                hidden_states.contiguous(), 0
            )
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # Chunk the hidden states so they aren't replicated across TP ranks.
        # This avoids duplicate computation in self.experts.
        # TODO: We can replace the all_reduce at the end of attn with a
        # reduce_scatter instead of chunking here.
        if self.is_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        needs_post_moe_combine = (
            getattr(self.experts, "dp_size", 1) > 1
            or getattr(self.experts, "pcp_size", 1) > 1
        )

        if (
            envs.VLLM_USE_LIGHTOP_MOE_SUM_MUL_ADD
            and self.shared_experts is not None
            and not needs_post_moe_combine
        ):
            shared_output = self.shared_experts(hidden_states, iqis=iqis)
            if self.experts.is_internal_router:
                # In this case, the gate/router runs inside the FusedMoE class.
                router_logits = hidden_states
            else:
                router_logits, _ = self.gate(hidden_states)
            routed_scaling_factor = (
                1.0 if self.is_rocm_aiter_moe_enabled
                else self.routed_scaling_factor
            )
            # Keep shared-expert path intact and only fuse routed scale + add
            # in the downstream MoE kernel.
            _, final_hidden_states = self.experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                iqis=iqis,
                shared_output=shared_output,
                routed_scaling_factor=routed_scaling_factor,
            )
        else:
            if self.experts.is_internal_router:
                # In this case, the gate/router runs inside the FusedMoE class
                fused_moe_out = self.experts(
                    hidden_states=hidden_states,
                    router_logits=hidden_states,
                    iqis=iqis,
                )
            else:
                # router_logits: (num_tokens, n_experts)
                router_logits, _ = self.gate(hidden_states)
                fused_moe_out = self.experts(
                    hidden_states=hidden_states, router_logits=router_logits
                )

            shared_output, final_hidden_states = fused_moe_out
            if self.shared_experts is None:
                assert shared_output is None

            # Fix FP16 overflow
            # See DeepseekV2DecoderLayer for more details.
            if hidden_states.dtype != torch.float16:
                if not self.is_rocm_aiter_moe_enabled:
                    final_hidden_states *= self.routed_scaling_factor
            elif self.shared_experts is not None:
                assert shared_output is not None
                shared_output *= 1.0 / self.routed_scaling_factor

            if self.shared_experts is not None:
                assert shared_output is not None
                final_hidden_states += shared_output

        if enable_lightly_cp:
            final_hidden_states = tensor_model_parallel_reduce_scatter(
                final_hidden_states.contiguous(), 0
            )
            return final_hidden_states
        elif self.is_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0
            )
            final_hidden_states = final_hidden_states[:num_tokens]
        elif self.tp_size > 1:
            final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(
                final_hidden_states
            )

        return final_hidden_states.view(num_tokens, hidden_dim)


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    import math

    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def _get_llama_4_scaling(
    original_max_position_embeddings: int, scaling_beta: float, positions: torch.Tensor
) -> torch.Tensor:
    scaling = 1 + scaling_beta * torch.log(
        1 + torch.floor(positions / original_max_position_embeddings)
    )
    # Broadcast over num_heads and head_dim
    return scaling[..., None, None]


class DeepseekV2Attention(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        max_position_embeddings: int = 8192,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        topk_indices_buffer: torch.Tensor | None = None,
        prefix: str = "",
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
        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        self.num_local_heads = num_heads // tp_size
        self.scaling = self.qk_head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings
        assert topk_indices_buffer is None, (
            "topk_indices_buffer is not \
        supported for DeepseekV2Attention"
        )

        if self.q_lora_rank is not None:
            self.q_a_proj = ReplicatedLinear(
                self.hidden_size,
                self.q_lora_rank,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_a_proj",
            )
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_proj",
            )
        else:
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
            )

        self.kv_a_proj_with_mqa = ReplicatedLinear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_a_proj_with_mqa",
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        # O projection.
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        if config.rope_parameters["rope_type"] != "default":
            config.rope_parameters["rope_type"] = (
                "deepseek_yarn"
                if config.rope_parameters.get("apply_yarn_scaling", True)
                else "deepseek_llama_scaling"
            )

        self.rotary_emb = get_rope(
            qk_rope_head_dim,
            max_position=max_position_embeddings,
            rope_parameters=config.rope_parameters,
            is_neox_style=False,
        )

        if (
            config.rope_parameters["rope_type"] != "default"
            and config.rope_parameters["rope_type"] == "deepseek_yarn"
        ):
            mscale_all_dim = config.rope_parameters.get("mscale_all_dim", False)
            scaling_factor = config.rope_parameters["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

        self.attn = Attention(
            self.num_local_heads,
            self.qk_head_dim,
            self.scaling,
            num_kv_heads=self.num_local_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None,
        *, iqis: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> torch.Tensor:
        if self.q_lora_rank is not None:
            q = self.q_a_proj(hidden_states)[0]
            q = self.q_a_layernorm(q)
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
        kv_a, _ = latent_cache.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        latent_cache = latent_cache.unsqueeze(1)
        kv_a = self.kv_a_layernorm(kv_a)
        kv = self.kv_b_proj(kv_a)[0]
        kv = kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_pe = latent_cache[:, :, self.kv_lora_rank :]

        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)

        q[..., self.qk_nope_head_dim :] = q_pe
        k = torch.empty_like(q)
        k[..., : self.qk_nope_head_dim] = k_nope
        k[..., self.qk_nope_head_dim :] = k_pe

        # Apply llama 4 scaling if provided
        if llama_4_scaling is not None:
            q *= llama_4_scaling

        # padding value to qk_head_dim for alignment
        v = torch.nn.functional.pad(
            v, [0, self.qk_head_dim - self.v_head_dim], value=0
        ).view(-1, self.num_local_heads * self.qk_head_dim)
        attn_output = self.attn(q, k, v)
        attn_output = attn_output.view(-1, self.num_local_heads, self.qk_head_dim)[
            ..., : self.v_head_dim
        ].reshape(-1, self.num_local_heads * self.v_head_dim)
        output, _ = self.o_proj(attn_output)
        return output


class DeepseekV32IndexerCache(torch.nn.Module, AttentionLayerBase):
    def __init__(
        self, head_dim: int, dtype: torch.dtype, prefix: str, cache_config: CacheConfig
    ):
        super().__init__()
        self.kv_cache = [torch.tensor([])]
        self.head_dim = head_dim
        self.prefix = prefix
        self.cache_config = cache_config
        self.dtype = dtype
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return MLAAttentionSpec(  # Only has one vector instead of K + V
            block_size=self.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
        )

    def forward(self): ...

    def get_attn_backend(self) -> AttentionBackend:
        return DeepseekV32IndexerBackend


class Indexer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config,
        hidden_size: int,
        q_lora_rank: int,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        topk_indices_buffer: torch.Tensor | None,
        prefix: str = "",
    ):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = config
        # self.indexer_cfg = config.attn_module_list_cfg[0]["attn_index"]
        self.topk_tokens = config.index_topk
        self.n_head = config.index_n_heads  # 64
        self.head_dim = config.index_head_dim  # 128
        self.rope_dim = config.qk_rope_head_dim  # 64
        self.q_lora_rank = q_lora_rank  # 1536
        # no tensor parallel, just replicated
        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.head_dim * self.n_head,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        self.wk = ReplicatedLinear(
            hidden_size,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wk",
        )
        self.k_norm = LayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = ReplicatedLinear(
            hidden_size,
            self.n_head,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.weights_proj",
        )
        self.softmax_scale = self.head_dim**-0.5

        self.scale_fmt = "ue8m0"
        self.quant_block_size = 128  # TODO: get from config
        self.topk_indices_buffer = topk_indices_buffer

        # NOTE: (zyongye) we use fp8 naive cache,
        #       where we store value in fp8 and scale in fp32
        #       per self.quant_block_size element
        self.k_cache = DeepseekV32IndexerCache(
            head_dim=self.head_dim + self.head_dim // self.quant_block_size * 4 if torch.cuda.get_device_properties("cuda").gcnArchName.split(':')[0] == "gfx938" else self.head_dim,
            dtype=torch.uint8  if torch.cuda.get_device_properties("cuda").gcnArchName.split(':')[0] == "gfx938" else torch.bfloat16,
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config,
        )
        self.max_model_len = vllm_config.model_config.max_model_len
        self.prefix = prefix
        # GLM-5.2 DSA: shared 类型层不自己算 indexer, 复用同组 full 层写入的
        # topk_indices_buffer (层顺序执行, full 层先写, shared 层后读).
        # checkpoint 里 shared 层无 indexer 权重, 跑零权重会算出全零 query
        # -> topk 选无意义 token -> 长输入乱码. 故 shared 层 forward 直接返回 buffer.
        indexer_types = getattr(config, "indexer_types", None)
        try:
            self.layer_idx = int(prefix.split(".")[-2])
        except (ValueError, IndexError):
            self.layer_idx = -1
        self.is_shared = (
            indexer_types is not None
            and 0 <= self.layer_idx < len(indexer_types)
            and indexer_types[self.layer_idx] == "shared"
        )
        from vllm.v1.attention.backends.mla.indexer import get_max_prefill_buffer_size

        self.max_total_seq_len = get_max_prefill_buffer_size(vllm_config)
        self.indexer_op = SparseAttnIndexer(
            self.k_cache,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
        )

    def forward(
        self, hidden_states: torch.Tensor, qr: torch.Tensor, positions, rotary_emb,
        *, iqis: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> torch.Tensor:
        q, _ = self.wq_b(qr)
        q = q.view(-1, self.n_head, self.head_dim)
        q_pe, q_nope = torch.split(
            q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )
        if envs.USE_FUSED_RMS_QUANT and self.wk.weight.dtype == torch.int8 and iqis is not None:
            k, _ = self.wk(hidden_states, iqis=iqis)
        else:
            k, _ = self.wk(hidden_states)
        k = self.k_norm(k)
        k_pe, k_nope = torch.split(
            k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
        )

        q_pe, k_pe = rotary_emb(positions, q_pe, k_pe.unsqueeze(1))
        # Note: RoPE (NeoX) can introduce extra leading dimensions during compilation
        # so we need to reshape back to token-flattened shapes
        q_pe = q_pe.reshape(-1, self.n_head, self.rope_dim)
        k_pe = k_pe.reshape(-1, 1, self.rope_dim)

        # `rotary_emb` is shape-preserving; `q_pe` is already
        # [num_tokens, n_head, rope_dim].
        q = torch.cat([q_pe, q_nope], dim=-1)
        # `k_pe` is [num_tokens, 1, rope_dim] (MQA).
        k = torch.cat([k_pe.squeeze(-2), k_nope], dim=-1)

        enable_lightly_cp = get_forward_context().enable_lightly_cp
        if enable_lightly_cp:
            k = tensor_model_parallel_all_gather(
                k.contiguous(), 0
            )
            gather_indexes_tensor = get_forward_context().gather_indexes_tensor
            enable_lightly_cplb = get_forward_context().enable_lightly_cplb
            if enable_lightly_cplb and gather_indexes_tensor is not None:
                k = torch.index_select(k, 0, gather_indexes_tensor)                

        # we only quant q here since k quant is fused with cache insertion
        if not current_platform.is_rocm() or torch.cuda.get_device_properties("cuda").gcnArchName.split(':')[0] == "gfx938":
            q = q.view(-1, self.head_dim)
            q_fp8, q_scale = per_token_group_quant_fp8(
                q,
                self.quant_block_size,
                column_major_scales=False,
                use_ue8m0=self.scale_fmt is not None,
            )
            q_fp8 = q_fp8.view(-1, self.n_head, self.head_dim)
            q_scale = q_scale.view(-1, self.n_head, 1)
        else:
            q_fp8 = q

        if envs.USE_FUSED_RMS_QUANT and self.weights_proj.weight.dtype == torch.int8 and iqis is not None:
            weights, _ = self.weights_proj(hidden_states, iqis=iqis)
        else:
            weights, _ = self.weights_proj(hidden_states)
        if not current_platform.is_rocm() or torch.cuda.get_device_properties("cuda").gcnArchName.split(':')[0] == "gfx938":
            weights = (
                weights.unsqueeze(-1) * q_scale * self.softmax_scale * self.n_head**-0.5
            )
            weights = weights.squeeze(-1)


        _topk = self.indexer_op(hidden_states, q_fp8, k, weights)

        return _topk


class DeepseekV2MLAAttention(nn.Module):
    """
    Main reference: DeepseekV2 paper, and FlashInfer Implementation
    (https://arxiv.org/abs/2405.04434 and https://github.com/flashinfer-ai/flashinfer/pull/551).

        For more info see MLACommonImpl in:
        vllm/v1/attention/backends/mla/utils.py
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        max_position_embeddings: int = 8192,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
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
        tp_size = get_tensor_model_parallel_world_size()
        assert num_heads % tp_size == 0
        self.num_local_heads = num_heads // tp_size if not \
            vllm_config.parallel_config.enable_lightly_cp else self.num_heads

        self.scaling = self.qk_head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings

        if self.q_lora_rank is not None:
            self.fused_qkv_a_proj = MergedColumnParallelLinear(
                self.hidden_size,
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.fused_qkv_a_proj",
                disable_tp=True,
            )
        else:
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_a_proj_with_mqa",
            )

        if self.q_lora_rank is not None:
            if envs.USE_FUSED_RMS_QUANT:
                self.q_a_layernorm = FusedRMSNormQuant(self.q_lora_rank, eps=config.rms_norm_eps)
            else:
                self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_proj",
                disable_tp=vllm_config.parallel_config.enable_lightly_cp
            )
        else:
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
                disable_tp=vllm_config.parallel_config.enable_lightly_cp,
            )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
            disable_tp=vllm_config.parallel_config.enable_lightly_cp,
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
            disable_tp=vllm_config.parallel_config.enable_lightly_cp,
        )

        if config.rope_parameters["rope_type"] != "default":
            config.rope_parameters["rope_type"] = (
                "deepseek_yarn"
                if config.rope_parameters.get("apply_yarn_scaling", True)
                else "deepseek_llama_scaling"
            )

        self.rotary_emb = get_rope(
            qk_rope_head_dim,
            max_position=max_position_embeddings,
            rope_parameters=config.rope_parameters,
            is_neox_style=False,
        )

        if (
            config.rope_parameters["rope_type"] != "default"
            and config.rope_parameters["rope_type"] == "deepseek_yarn"
        ):
            mscale_all_dim = config.rope_parameters.get("mscale_all_dim", False)
            scaling_factor = config.rope_parameters["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale
        #添加判断，默认开启DSA
        force_disable_dsa = envs.VLLM_DISABLE_DSA
        self.is_v32 = hasattr(config, "index_topk") and not force_disable_dsa

        # GLM-5.2 DSA IndexCache: 跨层共享 topk_indices.
        # 参考 vllm 官方实现 (index_topk_freq / index_skip_topk_offset).
        # full 层(layer 0,1,2,6,10,...) 自己建 indexer 算 topk 写入共享 buffer;
        # shared 层(其余) skip_topk, 不建 indexer, 直接复用同组 full 层写入的 buffer.
        # checkpoint 里 shared 层无 indexer 权重, 若仍跑零权重 indexer 会清空 buffer
        # 并写入垃圾 indices -> 覆盖 full 层结果 -> 长输入乱码.
        # MTP/nextn 层(layer_id >= num_hidden_layers) 永远建 full indexer.
        _skip_topk = False
        is_mtp_layer = False
        if self.is_v32:
            _index_topk_freq = getattr(config, "index_topk_freq", 1)
            _index_topk_pattern = getattr(config, "index_topk_pattern", None)
            _index_skip_topk_offset = getattr(
                config, "index_skip_topk_offset", 2)
            layer_id = extract_layer_index(prefix)

            if _index_topk_pattern is None:
                _skip_topk = (
                    max(layer_id - _index_skip_topk_offset + 1, 0)
                    % _index_topk_freq != 0
                )
            elif 0 <= layer_id < len(_index_topk_pattern):
                _skip_topk = _index_topk_pattern[layer_id] == "S"

            _num_hidden_layers = getattr(config, "num_hidden_layers", None)
            is_mtp_layer = (
                _num_hidden_layers is not None
                and layer_id >= _num_hidden_layers
            )

        if self.is_v32 and (not _skip_topk or is_mtp_layer):
            self.indexer_rope_emb = get_rope(
                qk_rope_head_dim,
                max_position=max_position_embeddings,
                rope_parameters=config.rope_parameters,
                is_neox_style=not getattr(config, "indexer_rope_interleave", True),
            )
            self.indexer = Indexer(
                vllm_config,
                config,
                hidden_size,
                q_lora_rank,
                quant_config,
                cache_config,
                topk_indices_buffer,
                f"{prefix}.indexer",
            )
        else:
            self.indexer_rope_emb = None
            self.indexer = None
        # 骨干 shared 层标记 skip_topk; MTP 层不 skip (运行时由 spec 框架控制).
        self.skip_topk = _skip_topk and not is_mtp_layer

        mla_modules = MLAModules(
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            rotary_emb=self.rotary_emb,
            o_proj=self.o_proj,
            fused_qkv_a_proj=self.fused_qkv_a_proj
            if self.q_lora_rank is not None
            else None,
            kv_a_proj_with_mqa=self.kv_a_proj_with_mqa
            if self.q_lora_rank is None
            else None,
            q_a_layernorm=self.q_a_layernorm if self.q_lora_rank is not None else None,
            q_b_proj=self.q_b_proj if self.q_lora_rank is not None else None,
            q_proj=self.q_proj if self.q_lora_rank is None else None,
            indexer=self.indexer,
            indexer_rotary_emb=self.indexer_rope_emb,
            is_sparse=self.is_v32,
            topk_indices_buffer=topk_indices_buffer,
        )

        self.mla_attn = MultiHeadLatentAttentionWrapper(
            self.hidden_size,
            self.num_local_heads,
            self.scaling,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
            self.q_lora_rank,
            self.kv_lora_rank,
            mla_modules,
            cache_config,
            quant_config,
            prefix,
            skip_topk=self.skip_topk,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None,
        *, iqis: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> torch.Tensor:
        return self.mla_attn(positions, hidden_states, llama_4_scaling, iqis=iqis)


class DeepseekV2DecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        config: DeepseekV2Config | None = None,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__()

        if config is None:
            config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        moe_layer_freq = getattr(config, "moe_layer_freq", 1)
        # DecoderLayers are created with `make_layers` which passes the prefix
        # with the layer's index.
        layer_idx = int(prefix.split(sep=".")[-1])
        self.layer_idx = layer_idx

        # verify MLA attention specific fields
        qk_nope_head_dim = getattr(config, "qk_nope_head_dim", 0)
        qk_rope_head_dim = getattr(config, "qk_rope_head_dim", 0)
        v_head_dim = getattr(config, "v_head_dim", 0)
        kv_lora_rank = getattr(config, "kv_lora_rank", 0)
        use_mha = config.model_type == "deepseek" or all(
            dim == 0 for dim in (qk_nope_head_dim, qk_rope_head_dim)
        )

        self.use_mha = use_mha

        if use_mha:
            attn_cls = DeepseekAttention
        elif model_config.use_mla:
            attn_cls = DeepseekV2MLAAttention
        else:
            attn_cls = DeepseekV2Attention
        self.self_attn = attn_cls(
            vllm_config=vllm_config,
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=config.q_lora_rank if hasattr(config, "q_lora_rank") else None,
            kv_lora_rank=kv_lora_rank,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            topk_indices_buffer=topk_indices_buffer,
        )

        if (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % moe_layer_freq == 0
        ):
            self.mlp = DeepseekV2MoE(
                config=config,
                parallel_config=parallel_config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        
        if not envs.USE_FUSED_RMS_QUANT:
            self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
        else:
            self.input_layernorm = FusedRMSNormQuant(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = FusedRMSNormQuant(
                config.hidden_size, eps=config.rms_norm_eps
            )

        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)

    def forward_RQ(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Self Attention
        # Fix residual FP16 overflow
        residual_fix_overflow = False
        assert self.input_layernorm.has_weight is True
        # DSA should set update_input True
        _dsa_flag = hasattr(self.self_attn, "indexer") and self.self_attn.indexer is not None
        if residual is None:
            residual = hidden_states.clone()
            i_q, i_s, _ = self.input_layernorm(x=hidden_states, 
                                               residual=None, 
                                               quant_dtype=torch.int8,
                                               update_input=_dsa_flag
                                               )
            residual_fix_overflow = True
        else:
            i_q, i_s, residual = self.input_layernorm(x=hidden_states, 
                                                      residual=residual, 
                                                      quant_dtype=torch.int8,
                                                      update_input=_dsa_flag
                                                      )
        attn_kwargs = {
            "positions": positions,
            "hidden_states": hidden_states,
            "iqis": (i_q, i_s)
        }
        if not self.use_mha:
            attn_kwargs["llama_4_scaling"] = llama_4_scaling
        hidden_states = self.self_attn(**attn_kwargs)

        if (
            not isinstance(self.self_attn, DeepseekAttention)
            and hidden_states.dtype == torch.float16
        ):
            # Fix FP16 overflow
            # We scale both hidden_states and residual before
            # rmsnorm, and rmsnorm result would not affect by scale.
            hidden_states *= 1.0 / self.routed_scaling_factor
            if self.layer_idx == 0:
                # The residual is shared by all layers, we only scale it on
                # first layer.
                residual *= 1.0 / self.routed_scaling_factor

        # Fully Connected
        enable_lightly_cp = get_forward_context().enable_lightly_cp
        update_hs = True if isinstance(self.mlp, DeepseekV2MoE) else False
        assert self.post_attention_layernorm.has_weight is True
        _i_q, _i_s, residual = self.post_attention_layernorm(x=hidden_states,
                                                             residual=residual,
                                                             quant_dtype=torch.int8,
                                                             update_input=update_hs
                                                             )
        new_resi = residual
        if enable_lightly_cp and isinstance(self.mlp, DeepseekV2MoE):
            hidden_states = self.mlp(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states, iqis=(_i_q, _i_s))


        if isinstance(self.mlp, DeepseekV2MLP) and hidden_states.dtype == torch.float16:
            # Fix FP16 overflow
            # Scaling the DeepseekV2MLP output, it is the input of
            # input_layernorm of next decoder layer.
            # The scaling of DeepseekV2MOE output would be done in the forward
            # of DeepseekV2MOE
            hidden_states *= 1.0 / self.routed_scaling_factor

        return hidden_states, new_resi

    def forward_default(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Self Attention
        # Fix residual FP16 overflow
        residual_fix_overflow = False
        if residual is None:
            residual = hidden_states.clone()
            hidden_states = self.input_layernorm(hidden_states)
            residual_fix_overflow = True
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)


        attn_kwargs = {
            "positions": positions,
            "hidden_states": hidden_states,
        }
        if not self.use_mha:
            attn_kwargs["llama_4_scaling"] = llama_4_scaling
        hidden_states = self.self_attn(**attn_kwargs)

        if (
            not isinstance(self.self_attn, DeepseekAttention)
            and hidden_states.dtype == torch.float16
        ):
            # Fix FP16 overflow
            # We scale both hidden_states and residual before
            # rmsnorm, and rmsnorm result would not affect by scale.
            hidden_states *= 1.0 / self.routed_scaling_factor
            if self.layer_idx == 0:
                # The residual is shared by all layers, we only scale it on
                # first layer.
                residual *= 1.0 / self.routed_scaling_factor

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        if isinstance(self.mlp, DeepseekV2MLP) and hidden_states.dtype == torch.float16:
            # Fix FP16 overflow
            # Scaling the DeepseekV2MLP output, it is the input of
            # input_layernorm of next decoder layer.
            # The scaling of DeepseekV2MOE output would be done in the forward
            # of DeepseekV2MOE
            hidden_states *= 1.0 / self.routed_scaling_factor

        return hidden_states, residual

    def choose_forward(self):
        if envs.USE_FUSED_RMS_QUANT:
            return self.forward_RQ
        else:
            return self.forward_default

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        forward_func = self.choose_forward()
        return forward_func(positions=positions,
                            hidden_states=hidden_states,
                            residual=residual,
                            llama_4_scaling=llama_4_scaling)


@support_torch_compile
class DeepseekV2Model(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.device = current_platform.device_type

        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        self.vocab_size = config.vocab_size
        #添加判断，默认开启DSA
        force_disable_dsa = envs.VLLM_DISABLE_DSA
        self.is_v32 = hasattr(config, "index_topk") and not force_disable_dsa        

        if self.is_v32:
            topk_tokens = config.index_topk
            topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                topk_tokens,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            topk_indices_buffer = None

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: DeepseekV2DecoderLayer(
                vllm_config, prefix, topk_indices_buffer=topk_indices_buffer
            ),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        # Backported: auxiliary hidden state collection for speculative
        # decoding drafters (EAGLE3 / DFlash / DSpark) that consume intermediate
        # target layer outputs. Populated by set_aux_hidden_state_layers().
        self.aux_hidden_state_layers: tuple[int, ...] = ()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        enable_lightly_cp = get_forward_context().enable_lightly_cp
        if enable_lightly_cp:
            scatter_indexes_tensor = get_forward_context().scatter_indexes_tensor
            if scatter_indexes_tensor is None:
                hidden_states_per_rank = torch.chunk(hidden_states, chunks=self.tp_size, dim=0)
                hidden_states = hidden_states_per_rank[self.tp_rank].contiguous()

                if residual is not None:
                    residual_per_rank = torch.chunk(residual, chunks=self.tp_size, dim=0)
                    residual = residual_per_rank[self.tp_rank].contiguous()

                if positions is not None:
                    positions_per_rank = torch.chunk(positions, chunks=self.tp_size, dim=0)
                    positions = positions_per_rank[self.tp_rank].contiguous()
            else:
                scatter_indexes_tensor = torch.where(scatter_indexes_tensor == -1, 0, scatter_indexes_tensor)
                hidden_states = torch.index_select(hidden_states, 0, scatter_indexes_tensor)

                if residual is not None:
                    residual = torch.index_select(residual, 0, scatter_indexes_tensor)

                if positions is not None:
                    positions = torch.index_select(positions, 0, scatter_indexes_tensor)

        # Compute llama 4 scaling once per forward pass if enabled
        llama_4_scaling_config = getattr(self.config, "llama_4_scaling", None)
        llama_4_scaling: torch.Tensor | None
        if llama_4_scaling_config is not None:
            llama_4_scaling = _get_llama_4_scaling(
                original_max_position_embeddings=llama_4_scaling_config[
                    "original_max_position_embeddings"
                ],
                scaling_beta=llama_4_scaling_config["beta"],
                positions=positions,
            )
        else:
            llama_4_scaling = None

        aux_hidden_states: list[torch.Tensor] = []
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            if self.aux_hidden_state_layers and idx in self.aux_hidden_state_layers:
                # Collect the pre-layer residual stream (hidden_states + residual)
                # as the auxiliary input for the drafter, matching EAGLE3/DFlash.
                if residual is None:
                    aux_hidden_state = hidden_states
                else:
                    aux_hidden_state = hidden_states + residual
                aux_hidden_states.append(aux_hidden_state)
            hidden_states, residual = layer(
                positions, hidden_states, residual, llama_4_scaling
            )

        if not get_pp_group().is_last_rank:
            if enable_lightly_cp:
                hidden_states = tensor_model_parallel_all_gather(hidden_states.contiguous(), dim=0)
                residual = tensor_model_parallel_all_gather(residual.contiguous(), dim=0)
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        # The final layer's output is collected after the loop if requested.
        if self.aux_hidden_state_layers and self.end_layer - 1 in \
                self.aux_hidden_state_layers:
            aux_hidden_states.append(hidden_states + residual)

        hidden_states, _ = self.norm(hidden_states, residual)

        if enable_lightly_cp:
            hidden_states = tensor_model_parallel_all_gather(hidden_states.contiguous(), dim=0)
            gather_indexes_tensor = get_forward_context().gather_indexes_tensor
            if gather_indexes_tensor is not None:
                hidden_states = torch.index_select(hidden_states, 0, gather_indexes_tensor)

        if self.aux_hidden_state_layers and len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states


class DeepseekV2MixtureOfExperts(MixtureOfExperts):
    moe_mlp_layers: list[DeepseekV2MoE]
    """
    List of MoE MLP layers in the model.
    """

    def extract_moe_parameters(self, example_moe: DeepseekV2MoE | None):
        if example_moe is None:
            self.num_moe_layers = 0
            self.num_expert_groups = 0
            self.num_logical_experts = 0
            self.num_physical_experts = 0
            self.num_local_physical_experts = 0
            self.num_routed_experts = 0
            self.num_shared_experts = 0
            self.num_redundant_experts = 0
            logger.warning("DeepSeekV2: No DeepseekV2MoE layer found in model.layers.")
        else:
            self.num_logical_experts = example_moe.n_logical_experts
            self.num_physical_experts = example_moe.n_physical_experts
            self.num_local_physical_experts = example_moe.n_local_physical_experts
            self.num_routed_experts = example_moe.n_routed_experts
            self.num_shared_experts = example_moe.n_shared_experts
            self.num_redundant_experts = example_moe.n_redundant_experts

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for moe in self.moe_mlp_layers:
            moe.n_local_physical_experts = num_local_physical_experts
            moe.n_physical_experts = num_physical_experts
            moe.n_redundant_experts = self.num_redundant_experts
            moe.experts.update_expert_map()


class DeepseekV2ForCausalLM(
    nn.Module,
    SupportsPP,
    DeepseekV2MixtureOfExperts,
    SupportsLoRA,
    SupportsEagle,
    SupportsEagle3,
):
    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    model_cls = DeepseekV2Model

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.quant_method = None
        if quant_config is not None:
            self.quant_method = quant_config.get_name()
            os.environ['LLAMA_NN'] = '0'
            os.environ['LM_NN'] = '0'

        self.config = config
        self.quant_config = quant_config

        qk_nope_head_dim = getattr(config, "qk_nope_head_dim", 0)
        qk_rope_head_dim = getattr(config, "qk_rope_head_dim", 0)
        self.use_mha = config.model_type == "deepseek" or all(
            dim == 0 for dim in (qk_nope_head_dim, qk_rope_head_dim)
        )

        if self.use_mha:
            self.packed_modules_mapping["qkv_proj"] = ["q_proj", "k_proj", "v_proj"]

        # `packed_modules_mapping` needs to be modified before
        # initializing DeepseekV2Model, as it is passed inplace to
        # quantization config init and may be used to select the
        # quant_method for relevant layers during initialization.
        self.fuse_qkv_a_proj = (
            hasattr(config, "q_lora_rank") and config.q_lora_rank is not None
        )
        if self.fuse_qkv_a_proj:
            self.packed_modules_mapping["fused_qkv_a_proj"] = [
                "q_a_proj",
                "kv_a_proj_with_mqa",
            ]

        self.model = self.model_cls(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        # Set MoE hyperparameters
        self.num_moe_layers = (
            self.config.num_hidden_layers - self.config.first_k_dense_replace
        )
        self.set_moe_parameters()

    def set_moe_parameters(self):
        self.expert_weights = []

        self.num_expert_groups = getattr(self.config, "n_group", 1)

        self.moe_layers = []
        self.moe_mlp_layers = []
        example_moe = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue

            assert isinstance(layer, DeepseekV2DecoderLayer)
            if isinstance(layer.mlp, DeepseekV2MoE):
                # Pick last one layer since the first ones may be dense layers.
                example_moe = layer.mlp
                self.moe_mlp_layers.append(layer.mlp)
                self.moe_layers.append(layer.mlp.experts)

        self.extract_moe_parameters(example_moe)
        self.use_llama_nn = os.environ.get('LLAMA_NN') == '1'
        self.use_awq_pad = os.environ.get('AWQ_PAD') == '1'
        self.tritonsingleton= W8a8GetCacheJSON() 
        self.tritonsingleton.topk = self.config.num_experts_per_tok
        self.tritonsingleton.quant_method=self.quant_method 

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = layers

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        num_layers = len(self.model.layers)
        return (2, num_layers // 2, num_layers - 3)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return SharedFusedMoE.make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
            num_redundant_experts=0,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        rocm_aiter_moe_shared_expert_enabled = (
            rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
        )
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        mla_params_mapping = [
            ("fused_qkv_a_proj", "q_a_proj", 0),
            ("fused_qkv_a_proj", "kv_a_proj_with_mqa", 1),
        ]
        mha_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        if self.use_mha:
            stacked_params_mapping.extend(mha_params_mapping)
        else:
            stacked_params_mapping.extend(mla_params_mapping)

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = SharedFusedMoE.make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts
            + (
                self.config.n_shared_experts
                if rocm_aiter_moe_shared_expert_enabled
                else 0
            ),
            num_redundant_experts=self.num_redundant_experts,
        )

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        # 判断是否加载"indexer"权重
        model_has_indexer = any("indexer" in param_name for param_name in params_dict.keys())
        
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            #  跳过加载"indexer"权重
            if "indexer" in name and not model_has_indexer:
                logger.info(f"Skipping indexer weight (DSA disabled): {name}")
                continue

            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is not None:
                continue  # skip spec decode layers for main model

            is_fusion_moe_shared_experts_layer = (
                rocm_aiter_moe_shared_expert_enabled and ("mlp.shared_experts" in name)
            )

            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                if is_fusion_moe_shared_experts_layer:
                    continue
                name_mapped = name.replace(weight_name, param_name)

                # QKV fusion is optional, fall back to normal
                # weight loading if it's not enabled
                # if go with fusion option, then update name
                if (
                    param_name == "fused_qkv_a_proj"
                ) and name_mapped not in params_dict:
                    continue
                else:
                    name = name_mapped
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False

                # Special handling: when AITER fusion_shared_experts is enabled,
                # checkpoints may provide a single widened shared_experts tensor
                # without explicit expert indices
                # (e.g. ...mlp.shared_experts.gate_proj.weight).
                # For models with multiple shared experts, split that tensor
                # evenly into per-shared-expert slices and load them into
                # appended expert slots mlp.experts.{n_routed_experts + j}.*
                # accordingly.
                num_chunks = 1
                if is_fusion_moe_shared_experts_layer:
                    num_chunks = getattr(self.config, "n_shared_experts", 1) or 1
                    # Determine split axis based on op type
                    # gate/up: ColumnParallel → split along dim 0
                    # down: RowParallel → split along dim 1
                    split_dim = (
                        1
                        if ("down_proj.weight" in name and loaded_weight.ndim > 1)
                        else 0
                    )
                    total = loaded_weight.shape[split_dim]
                    assert total % num_chunks == 0, (
                        f"Shared expert weight dim {total} "
                        f"not divisible by num_chunks {num_chunks}"
                    )
                    chunk_size = total // num_chunks

                for j in range(num_chunks):
                    chunk_name = name
                    weight_to_load = loaded_weight

                    if is_fusion_moe_shared_experts_layer:
                        chunk_slice = slice(j * chunk_size, (j + 1) * chunk_size)
                        if loaded_weight.ndim == 1:
                            weight_to_load = loaded_weight[chunk_slice]
                        elif split_dim == 0:
                            weight_to_load = loaded_weight[chunk_slice, :]
                        else:
                            weight_to_load = loaded_weight[:, chunk_slice]
                        # Synthesize an expert-style name so expert mapping
                        # can route it
                        chunk_name = name.replace(
                            "mlp.shared_experts",
                            f"mlp.experts.{self.config.n_routed_experts + j}",
                        )

                    # Use expert_params_mapping to locate the destination
                    # param and delegate to its expert-aware weight_loader
                    # with expert_id.
                    for mapping in expert_params_mapping:
                        param_name, weight_name, expert_id, shard_id = mapping
                        if weight_name not in chunk_name:
                            continue

                        # Anyway, this is an expert weight and should not be
                        # attempted to load as other weights later
                        is_expert_weight = True

                        # Do not modify `name` since the loop may continue here
                        # Instead, create a new variable
                        name_mapped = chunk_name.replace(weight_name, param_name)

                        if is_pp_missing_parameter(name_mapped, self):
                            continue

                        param = params_dict[name_mapped]
                        # We should ask the weight loader to return success or
                        # not here since otherwise we may skip experts with
                        # other available replicas.
                        weight_loader = typing.cast(
                            Callable[..., bool], param.weight_loader
                        )
                        success = weight_loader(
                            param,
                            weight_to_load,
                            name_mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                        if success:
                            if not is_fusion_moe_shared_experts_layer:
                                name = name_mapped
                            else:
                                loaded_params.add(name_mapped)
                            break
                    else:
                        if is_expert_weight:
                            # We've checked that this is an expert weight
                            # However it's not mapped locally to this rank
                            # So we simply skip it
                            continue

                        # Skip loading extra bias for GPTQ models.
                        if name.endswith(".bias") and name not in params_dict:
                            continue

                        # Remapping the name of FP8 kv-scale.
                        name = maybe_remap_kv_scale_name(name, params_dict)
                        if name is None:
                            continue

                        if is_pp_missing_parameter(name, self):
                            continue

                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
            if not is_fusion_moe_shared_experts_layer:
                loaded_params.add(name)
                
        if self.use_llama_nn and self.quant_method is None:
            lay_key_words = [
                "self_attn.q_proj.weight",
                "self_attn.q_a_proj.weight",
                "self_attn.q_b_proj.weight",
                "self_attn.kv_a_proj_with_mqa.weight",
                "self_attn.kv_b_proj.weight",
                "self_attn.o_proj.weight",
                "mlp.gate_up_proj.weight",
                "mlp.down_proj.weight",
                "mlp.gate.weight",
                "shared_experts.gate_up_proj.weight",
                "shared_experts.down_proj.weight",
                "lm_head.weight",
            ]

            combined_words = "|".join(lay_key_words)
            
            for layername in loaded_params:
                weight = params_dict[layername]
                matches = re.findall(combined_words, layername)
                if matches:
                    _weight = torch.zeros_like(weight.data)
                    ori_shape =_weight.shape
                    
                    ops.trans_w16_gemm(_weight, weight.data, _weight.shape[0], _weight.shape[1])
                    weight.data.copy_(_weight)
                    
                    weight.data=weight.data.reshape(ori_shape[1],-1)
            
        return loaded_params


class DeepseekForCausalLM(DeepseekV2ForCausalLM):
    pass


class DeepseekV3ForCausalLM(DeepseekV2ForCausalLM):
    pass


class GlmMoeDsaForCausalLM(DeepseekV2ForCausalLM):
    pass


# Compatibility with
# https://huggingface.co/deepseek-ai/DeepSeek-V3-Base/blob/main/configuration_deepseek.py
def get_spec_layer_idx_from_weight_name(
    config: DeepseekV2Config | DeepseekV3Config, weight_name: str
) -> int | None:
    if (
        hasattr(config, "num_nextn_predict_layers")
        and config.num_nextn_predict_layers > 0
    ):
        layer_idx = config.num_hidden_layers
        for i in range(config.num_nextn_predict_layers):
            if weight_name.startswith(f"model.layers.{layer_idx + i}."):
                return layer_idx + i
    return None
