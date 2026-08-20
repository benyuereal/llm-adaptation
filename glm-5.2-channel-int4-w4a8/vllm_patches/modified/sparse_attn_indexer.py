# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom Sparse Attention Indexer layers."""

import torch
import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.platforms import current_platform
from vllm.platforms.rocm import get_gcn_arch_name
from vllm.utils.deep_gemm import fp8_mqa_logits, fp8_paged_mqa_logits
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
)
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import indexer_k_bf16_cache_triton, cp_gather_indexer_k_bf16_cache_triton
from vllm.v1.worker.workspace import current_workspace_manager
from lightop import op, gemmopt

from vllm.attention.utils.kv_transfer_utils import (
    maybe_transfer_kv_layer,
)

if current_platform.is_cuda_alike():
    from vllm import _custom_ops as ops
elif current_platform.is_xpu():
    from vllm._ipex_ops import ipex_ops as ops

logger = init_logger(__name__)
_GLOBAL_LOGITS_BUFFERS = {}

@maybe_transfer_kv_layer
def sparse_attn_indexer(
    hidden_states: torch.Tensor,
    layer_name:str,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
) -> torch.Tensor:
    # careful! this will be None in dummy run
    attn_metadata = get_forward_context().attn_metadata
    fp8_dtype = current_platform.fp8_dtype()
    if q_fp8.dtype == fp8_dtype:
        MAX_ELEMENTS = 65536 * 65536
    elif q_fp8.dtype in (torch.bfloat16, torch.float16):
        MAX_ELEMENTS = 16384 * 32768
    else:
        MAX_ELEMENTS = 16384 * 32768 

    device = q_fp8.device
    if device not in _GLOBAL_LOGITS_BUFFERS or _GLOBAL_LOGITS_BUFFERS[device].numel() < MAX_ELEMENTS:
        _GLOBAL_LOGITS_BUFFERS[device] = torch.empty(
            MAX_ELEMENTS, 
            dtype=torch.float32, 
            device=device
        )
    logits_buffer = _GLOBAL_LOGITS_BUFFERS[device]
    # assert isinstance(attn_metadata, dict)
    if not isinstance(attn_metadata, dict):
        # Reserve workspace for indexer during profiling run
        current_workspace_manager().get_simultaneous(
            ((total_seq_lens, head_dim), fp8_dtype if not current_platform.is_rocm() or torch.cuda.get_device_properties("cuda").gcnArchName.split(':')[0] == "gfx938" else k.dtype,),
            ((total_seq_lens, 4), torch.uint8),
        )
        return sparse_attn_indexer_fake(
            hidden_states,
            layer_name,
            kv_cache,
            q_fp8,
            k,
            weights,
            quant_block_size,
            scale_fmt,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
        )
    attn_metadata = attn_metadata[layer_name]
    assert isinstance(attn_metadata, DeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata.slot_mapping[:attn_metadata.num_kv_actual_tokens]
    has_decode = attn_metadata.num_decodes > 0
    has_prefill = attn_metadata.num_prefills > 0
    num_decode_tokens = attn_metadata.num_decode_tokens
    num_tokens = slot_mapping.shape[0]
    k = k[:num_tokens]
    if not current_platform.is_rocm() or torch.cuda.get_device_properties("cuda").gcnArchName.split(':')[0] == "gfx938":
        ops.indexer_k_quant_and_cache(
            k,
            kv_cache,
            slot_mapping,
            quant_block_size,
            scale_fmt,
        )
    else:
        indexer_k_bf16_cache_triton(
            k,
            kv_cache,
            slot_mapping,
        )
    topk_indices_buffer[: hidden_states.shape[0]] = -1
    if has_prefill:
        prefill_metadata = attn_metadata.prefill

        # Get the full shared workspace buffers once (will allocate on first use)
        workspace_manager = current_workspace_manager()
        k_fp8_full, k_scale_full = workspace_manager.get_simultaneous(
                ((total_seq_lens, head_dim), fp8_dtype if not current_platform.is_rocm() or get_gcn_arch_name() == "gfx938" else k.dtype,),
                ((total_seq_lens, 4), torch.uint8),
            )
        for chunk in prefill_metadata.chunks:
            if not current_platform.is_rocm(): # or torch.cuda.get_device_properties("cuda").gcnArchName.split(':')[0] == "gfx938":                       
                k_fp8 = k_fp8_full[: chunk.total_seq_lens]
                k_scale = k_scale_full[: chunk.total_seq_lens]
                ops.cp_gather_indexer_k_quant_cache(
                    kv_cache,
                    k_fp8,
                    k_scale,
                    chunk.block_table,
                    chunk.cu_seq_lens,
                )
            elif get_gcn_arch_name() == "gfx938":
                k_fp8 = k_fp8_full[: chunk.total_seq_lens]
                k_scale = k_scale_full[: chunk.total_seq_lens]
                ops.cp_gather_indexer_k_quant_cache(
                    kv_cache,
                    k_fp8,
                    k_scale,
                    chunk.block_table,
                    chunk.cu_seq_lens,
                )
            else:
                k_fp8 = k_fp8_full[: chunk.total_seq_lens]
                k_scale = k_scale_full[: chunk.total_seq_lens]   
                cp_gather_indexer_k_bf16_cache_triton(
                    kv_cache,
                    k_fp8,
                    chunk.block_table,
                    chunk.cu_seq_lens,
                )        

            q_all = q_fp8[chunk.token_start:chunk.token_end]
            weights_all = weights[chunk.token_start:chunk.token_end]
            ks_all = chunk.cu_seqlen_ks
            ke_all = chunk.cu_seqlen_ke
            
            num_q = q_all.shape[0]
            num_k = k_fp8.shape[0]

            is_q_fp16_bf16 = q_all.dtype in (torch.float16, torch.bfloat16)
            align_size = 128 if is_q_fp16_bf16 else 1
            
            kv_seq_len_aligned = (num_k + align_size - 1) // align_size * align_size

            current_capacity = logits_buffer.numel()
            MAX_Q_CHUNK = current_capacity // max(1, kv_seq_len_aligned)
            if align_size > 1:
                MAX_Q_CHUNK = (MAX_Q_CHUNK // align_size) * align_size
            MAX_Q_CHUNK = max(1, MAX_Q_CHUNK)

            slices = []

            for start_idx in range(0, num_q, MAX_Q_CHUNK):
                end_idx = min(start_idx + MAX_Q_CHUNK, num_q)
                slices.append((start_idx, end_idx))

            for q_start, q_end in slices:
                if q_end <= q_start:
                    continue
                    
                q_slice = q_all[q_start:q_end]
                weights_slice = weights_all[q_start:q_end]

                ks_slice = ks_all[q_start:q_end]
                ke_slice = ke_all[q_start:q_end]

                q_len = q_end - q_start
                q_seq_len_aligned = (q_len + align_size - 1) // align_size * align_size

                required_size = q_seq_len_aligned * kv_seq_len_aligned
                logits_slice_view = logits_buffer[:required_size].view(q_seq_len_aligned, kv_seq_len_aligned)

                if not current_platform.is_rocm():
                    logits_slice = fp8_mqa_logits(
                        q_slice,
                        (k_fp8, k_scale.view(torch.float32).flatten()),
                        weights_slice,
                        ks_slice,
                        ke_slice,
                    )
                elif get_gcn_arch_name() == "gfx938":
                    op.mqa_logits(
                        q_slice,  
                        k_fp8, 
                        weights_slice, 
                        ks_slice, 
                        ke_slice,
                        q_slice.shape[0], # logical lengths
                        k_fp8.shape[0],
                        q_slice.shape[1],
                        q_slice.shape[2],
                        k_scale.view(torch.float32).flatten(),
                        True,
                        logits_slice_view # padded properly out of box for hardware requirements
                    )
                    # Extract the exact logical valid window for downstream topk
                    logits_slice = logits_slice_view[:q_len, :num_k]
                else:
                    from vllm.model_executor.layers.mqa_logits import mqa_logits

                    # tilelang mqa_logits 返回新 tensor [q_len, num_k] (非原地写入),
                    # 与官方 fp8_fp4_mqa_logits 一致: 直接用返回值传给 topk,
                    # 不要 copy_ 到 padded logits_slice_view (会引入 padding -inf
                    # 与非连续 stride, 导致 topk 退化/乱码).
                    logits_slice = mqa_logits(
                        q_slice,
                        k_fp8,
                        weights_slice.to(torch.float32),
                        ks_slice,
                        ke_slice
                    )

                num_rows_slice = logits_slice.shape[0]

                topk_indices_slice = topk_indices_buffer[
                    chunk.token_start + q_start : chunk.token_start + q_end, :topk_tokens
                ]

                if not envs.USE_LIGHTOP_TOPK:
                    torch.ops._C.top_k_per_row_prefill(
                        logits_slice,
                        ks_slice,
                        ke_slice,
                        topk_indices_slice,
                        num_rows_slice,
                        logits_slice.stride(0), # Automatically fetches kv_seq_len_aligned stride
                        logits_slice.stride(1),
                        topk_tokens,
                    )
                else:
                    op.top_k_per_row_prefill(
                        logits_slice,
                        ks_slice,
                        ke_slice,
                        topk_indices_slice,
                        num_rows_slice,
                        logits_slice.stride(0),
                        logits_slice.stride(1),
                        topk_tokens,
                    )

    if has_decode:
        decode_metadata = attn_metadata.decode
        # kv_cache size requirement [num_block, block_size, n_head, head_dim],
        # we only have [num_block, block_size, head_dim],
        kv_cache = kv_cache.unsqueeze(-2)
        decode_lens = decode_metadata.decode_lens
        if decode_metadata.requires_padding:
            # pad in edge case where we have short chunked prefill length <
            # decode_threshold since we unstrictly split
            # prefill and decode by decode_threshold
            # (currently set to 1 + speculative tokens)
            padded_q_fp8_decode_tokens = pack_seq_triton(
                q_fp8[:num_decode_tokens], decode_lens
            )
        else:
            padded_q_fp8_decode_tokens = q_fp8[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_fp8.shape[1:]
            )
        # TODO: move and optimize below logic with triton kernels
        batch_size = padded_q_fp8_decode_tokens.shape[0]
        next_n = padded_q_fp8_decode_tokens.shape[1]
        assert batch_size == decode_metadata.seq_lens.shape[0]
        num_padded_tokens = batch_size * next_n

        if not current_platform.is_rocm():
            logits = fp8_paged_mqa_logits(
                padded_q_fp8_decode_tokens,
                kv_cache,
                weights[:num_padded_tokens],
                decode_metadata.seq_lens,
                decode_metadata.block_table,
                decode_metadata.schedule_metadata,
                max_model_len=max_model_len,
            )
        else:
            from vllm.model_executor.layers.paged_mqa_logits import page_mqa_logits
            logits = page_mqa_logits(
                padded_q_fp8_decode_tokens,
                kv_cache,
                weights[:num_padded_tokens] if torch.cuda.get_device_properties("cuda").gcnArchName.split(':')[
                                                   0] == "gfx938" else weights[:num_padded_tokens].to(torch.float32),
                decode_metadata.seq_lens,
                decode_metadata.block_table,
                max_model_len
            )

        num_rows = logits.shape[0]

        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]
        # if torch.distributed.get_rank() == 0:
        #     print(f"====[DEBUG] logits shape: {logits.shape}, next_n: {next_n}, topk_tokens size: {topk_tokens}")
        if not envs.USE_LIGHTOP_TOPK:
            torch.ops._C.top_k_per_row_decode(
                logits,
                next_n,
                decode_metadata.seq_lens,
                topk_indices,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                topk_tokens,
            )
        else:
            op.top_k_per_row_decode(
                logits,
                next_n,
                decode_metadata.seq_lens,
                topk_indices,
                num_rows,
                logits.stride(0),
                logits.stride(1),
                topk_tokens,
            )
        if decode_metadata.requires_padding:
            # if padded, we need to unpack
            # the topk indices removing padded tokens
            topk_indices = unpack_seq_triton(
                topk_indices.reshape(batch_size, -1, topk_indices.shape[-1]),
                decode_lens,
            )
            topk_indices_buffer[:num_decode_tokens, : topk_indices.shape[-1]] = (
                topk_indices
            )

    return topk_indices_buffer


def sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    layer_name: str,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
) -> torch.Tensor:
    return topk_indices_buffer


direct_register_custom_op(
    op_name="sparse_attn_indexer",
    op_func=sparse_attn_indexer,
    mutates_args=["topk_indices_buffer"],
    fake_impl=sparse_attn_indexer_fake,
    dispatch_key=current_platform.dispatch_key,
)


@CustomOp.register("sparse_attn_indexer")
class SparseAttnIndexer(CustomOp):
    """Sparse Attention Indexer Custom Op Layer. This layer is extracted as a
    separate custom op since it involves heavy custom kernels like `mqa_logits`,
    `paged_mqa_logits` and `top_k_per_row`, etc. Those kernels maybe requires
    specific memory layout or implementation for different hardware backends to
    achieve optimal performance.

    For now, the default native path will use CUDA backend path. Other platform
    may requires add the corresponding Custom Op name `sparse_attn_indexer` to
    `custom_ops` in `CompilationConfig` to enable the platform specific path.
    """

    def __init__(
        self,
        k_cache,
        quant_block_size: int,
        scale_fmt: str,
        topk_tokens: int,
        head_dim: int,
        max_model_len: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor,
    ):
        super().__init__()
        self.k_cache = k_cache
        self.quant_block_size = quant_block_size
        self.scale_fmt = scale_fmt
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.max_model_len = max_model_len
        self.max_total_seq_len = max_total_seq_len
        self.topk_indices_buffer = topk_indices_buffer

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        if current_platform.is_cuda():
            return self.forward_cuda(hidden_states, q_fp8, k, weights)
        elif current_platform.is_rocm():
            return self.forward_hip(hidden_states, q_fp8, k, weights)
        else:
            raise NotImplementedError(
                "SparseAttnIndexer native forward is only implemented for "
                "CUDA and ROCm platform."
            )

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        return torch.ops.vllm.sparse_attn_indexer(
            hidden_states,
            self.k_cache.prefix,
            self.k_cache.kv_cache[0],
            q_fp8,
            k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
        )

    def forward_hip(
        self,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        if rocm_aiter_ops.is_enabled():
            return torch.ops.vllm.rocm_aiter_sparse_attn_indexer(
                hidden_states,
                self.k_cache.prefix,
                self.k_cache.kv_cache[0],
                q_fp8,
                k,
                weights,
                self.quant_block_size,
                self.scale_fmt,
                self.topk_tokens,
                self.head_dim,
                self.max_model_len,
                self.max_total_seq_len,
                self.topk_indices_buffer,
            )        
        else:
            return torch.ops.vllm.sparse_attn_indexer(
                hidden_states,
                self.k_cache.prefix,
                self.k_cache.kv_cache[0],
                q_fp8,
                k,
                weights,
                self.quant_block_size,
                self.scale_fmt,
                self.topk_tokens,
                self.head_dim,
                self.max_model_len,
                self.max_total_seq_len,
                self.topk_indices_buffer,
            )
