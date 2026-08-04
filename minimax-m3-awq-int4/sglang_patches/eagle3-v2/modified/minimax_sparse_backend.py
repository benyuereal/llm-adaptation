from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.configs.model_config import (
    get_minimax_sparse_attention_config,
    get_minimax_sparse_disable_value_layer_ids,
    get_minimax_sparse_layer_ids,
    get_minimax_sparse_score_type,
)
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.minimax_sparse_ops.minimax_sparse import (
    minimax_sparse_decode,
    minimax_sparse_prefill,
)
from sglang.srt.mem_cache.memory_pool import MiniMaxSparseKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class MiniMaxSparseAttnBackend(AttentionBackend):
    def __init__(self, runner: "ModelRunner"):

        assert isinstance(runner.token_to_kv_pool, MiniMaxSparseKVPool)
        self.kv_pool = runner.token_to_kv_pool
        self.req_to_token = runner.req_to_token_pool.req_to_token

        hf_config = runner.model_config.hf_config
        sparse_cfg = get_minimax_sparse_attention_config(hf_config)
        self.idx_head_dim = sparse_cfg["sparse_index_dim"]
        self.dense_layer_ids, self.sparse_layer_ids = get_minimax_sparse_layer_ids(
            sparse_cfg
        )
        self.disable_value_layer_ids: set[int] = set(
            get_minimax_sparse_disable_value_layer_ids(sparse_cfg)
        )
        self.score_type: str = get_minimax_sparse_score_type(sparse_cfg)
        # assert self.idx_head_dim == head_dim

        # max_seqlen for the current forward pass, stored as a plain Python int
        # so that it is safe to use inside CUDA graphs (no .item() at graph time).
        # Populated by init_forward_metadata* before each forward.
        self._max_seqlen_q: int = 1
        self._max_seqlen_k: int = 1

        self.block_size_q = 1
        self.block_size_k = sparse_cfg["sparse_block_size"]
        if "sparse_init_block" in sparse_cfg:
            self.init_blocks = sparse_cfg["sparse_init_block"]
        else:
            init_tokens = sparse_cfg["sparse_init_tokens"]
            self.init_blocks = (
                init_tokens + self.block_size_k - 1
            ) // self.block_size_k
        if "sparse_local_block" in sparse_cfg:
            self.local_blocks = sparse_cfg["sparse_local_block"]
        else:
            local_tokens = sparse_cfg["sparse_local_tokens"]
            self.local_blocks = (
                local_tokens + self.block_size_k - 1
            ) // self.block_size_k + 1
        self.topk_blocks = sparse_cfg["sparse_topk_blocks"]

        logger.info(
            f"[MiniMaxSparse] Backend initialized "
            f"(score_type={self.score_type!r}, "
            f"disable_value_layers={sorted(self.disable_value_layer_ids)})"
        )

    # ------------------------------------------------------------------
    # Delegation helpers
    # ------------------------------------------------------------------

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        mode = forward_batch.forward_mode

        # Compute max_seqlen from CPU-side data so it is safe inside CUDA graphs.
        if mode.is_extend() and not mode.is_decode():
            # EAGLE3 TARGET_VERIFY is is_extend()==True, so this backend runs its
            # forward_extend path. But ForwardBatch.init_new routes target_verify
            # through its *decode* branch, which leaves extend_seq_lens /
            # extend_seq_lens_cpu / extend_prefix_lens as None. The base
            # AttentionBackend.forward also dispatches TARGET_VERIFY to
            # forward_extend (is_decode()==False), so every backend hits this —
            # most just don't read these fields directly (they use
            # spec_info.positions). The MiniMax sparse backend reads them
            # directly, so materialise them here.
            #
            # IMPORTANT: in the verify path batch.seq_lens is the PREFIX length
            # (NOT prefix+draft) — prepare_for_v2_verify does not add draft to
            # seq_lens, and after verify the scheduler updates seq_lens via
            # `seq_lens + accept_lens`. So:
            #   extend_prefix_lens = seq_lens          (the prefix)
            #   extend_seq_lens     = draft_token_num  (the proposed tokens)
            #   kernel's seq_lens   = prefix + draft   (built in forward_extend)
            if forward_batch.extend_seq_lens is None:
                spec_info = getattr(forward_batch, "spec_info", None)
                draft_token_num = getattr(spec_info, "draft_token_num", None)
                num_reqs = len(forward_batch.seq_lens)
                if draft_token_num is None:
                    draft_token_num = (
                        forward_batch.input_ids.shape[0] // max(num_reqs, 1)
                    )
                device = forward_batch.seq_lens.device
                forward_batch.extend_seq_lens = torch.full(
                    (num_reqs,), int(draft_token_num),
                    dtype=torch.int32, device=device,
                )
                forward_batch.extend_seq_lens_cpu = [
                    int(draft_token_num)
                ] * num_reqs
                if forward_batch.extend_prefix_lens is None:
                    # seq_lens is the prefix here (verify did not add draft).
                    forward_batch.extend_prefix_lens = (
                        forward_batch.seq_lens.to(torch.int32)
                    )
                    forward_batch.extend_prefix_lens_cpu = (
                        forward_batch.extend_prefix_lens.cpu().tolist()
                    )

            self._max_seqlen_q = int(max(forward_batch.extend_seq_lens_cpu))
            # K length = prefix + draft. For verify, seq_lens is the prefix and
            # K length = prefix + draft. For verify, seq_lens is the prefix
            # (scheduler adds accept_lens after); for normal extend, seq_lens
            # is already prefix+extend. Use forward_mode (clear) instead of a
            # torch.all() comparison.
            if forward_batch.forward_mode.is_target_verify():
                prefix_plus_extend = (
                    forward_batch.extend_prefix_lens + forward_batch.extend_seq_lens
                )
                self._max_seqlen_k = int(prefix_plus_extend.max().item())
            else:
                # Normal extend: seq_lens already = prefix + extend.
                self._max_seqlen_k = int(forward_batch.seq_lens.max().item())
        else:
            # seq_lens_cpu is a CPU tensor – .max().item() is fine here
            self._max_seqlen_q = 1
            self._max_seqlen_k = int(forward_batch.seq_lens_cpu.max().item())

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        pass

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs,
        num_tokens,
        req_pool_indices,
        seq_lens,
        encoder_lens,
        forward_mode,
        spec_info,
    ):
        # EAGLE3 TARGET_VERIFY: q length = draft_token_num (not 1, which is the
        # decode value), K length = prefix + draft (seq_lens here is the prefix,
        # the scheduler adds accept_lens after). Same semantics as the eager
        # init_forward_metadata path. Without this, sparse prefill computes wrong
        # block counts and output garbles under cuda graph.
        draft_token_num = getattr(spec_info, "draft_token_num", None)
        if forward_mode.is_target_verify() and draft_token_num is not None:
            self._max_seqlen_q = int(draft_token_num)
            self._max_seqlen_k = int(seq_lens[:bs].max().item()) + int(draft_token_num)
        else:
            self._max_seqlen_q = 1
            self._max_seqlen_k = int(seq_lens[:bs].max().item())

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs,
        req_pool_indices,
        seq_lens,
        seq_lens_sum,
        encoder_lens,
        forward_mode,
        spec_info,
        seq_lens_cpu,
    ):
        # seq_lens_cpu is a CPU tensor – safe to call .max().item() here.
        # Same TARGET_VERIFY fix as capture path above.
        draft_token_num = getattr(spec_info, "draft_token_num", None)
        if forward_mode.is_target_verify() and draft_token_num is not None:
            self._max_seqlen_q = int(draft_token_num)
            self._max_seqlen_k = int(seq_lens_cpu[:bs].max().item()) + int(draft_token_num)
        else:
            self._max_seqlen_q = 1
            self._max_seqlen_k = int(seq_lens_cpu[:bs].max().item())

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def forward(
        self,
        q,
        k,
        v,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        if forward_batch.forward_mode.is_idle():
            idx_q = kwargs.get("idx_q")
            num_idx_heads = idx_q.shape[1]
            disable_value = layer.layer_id in self.disable_value_layer_ids
            idx_out: Optional[torch.Tensor] = (
                None
                if disable_value
                else q.new_zeros(q.shape[0], num_idx_heads * self.idx_head_dim)
            )
            out = q.new_zeros(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
            return idx_out, out
        else:
            return super().forward(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        *,
        idx_q: torch.Tensor,
        idx_k: torch.Tensor,
        idx_v: Optional[torch.Tensor],
    ):
        disable_value = layer.layer_id in self.disable_value_layer_ids

        # EAGLE3 TARGET_VERIFY: extend_seq_lens may be None here.
        # - eager: init_forward_metadata already materialised it.
        # - cuda graph capture/replay: init_forward_metadata_capture_cuda_graph
        #   does NOT set it (the captured ForwardBatch leaves it None), so the
        #   cu_seqlens build below would crash on `.device`. Materialise it here
        #   from spec_info.draft_token_num with a fixed shape (graph-safe: no
        #   host sync, no dynamic shape). Uniform draft_token_num per request.
        if forward_batch.extend_seq_lens is None:
            spec_info = getattr(forward_batch, "spec_info", None)
            draft_token_num = getattr(spec_info, "draft_token_num", None)
            num_reqs = forward_batch.seq_lens.shape[0]
            if draft_token_num is None:
                draft_token_num = (
                    forward_batch.input_ids.shape[0] // max(num_reqs, 1)
                )
            forward_batch.extend_seq_lens = torch.full(
                (num_reqs,), int(draft_token_num),
                dtype=torch.int32, device=forward_batch.seq_lens.device,
            )
            # _cpu list fields use no host sync to build, but only set them
            # outside graph capture (forward_extend only needs the tensors;
            # the _cpu lists are consumed by init_forward_metadata in eager).
            if not torch.cuda.is_current_stream_capturing():
                if forward_batch.extend_seq_lens_cpu is None:
                    forward_batch.extend_seq_lens_cpu = [int(draft_token_num)] * num_reqs
                if forward_batch.extend_prefix_lens is None:
                    # verify: seq_lens is the prefix (scheduler adds accept_lens later)
                    forward_batch.extend_prefix_lens = forward_batch.seq_lens.to(torch.int32)
                    forward_batch.extend_prefix_lens_cpu = (
                        forward_batch.extend_prefix_lens.cpu().tolist()
                    )
            else:
                # Under graph capture: only materialise the tensors (graph-safe,
                # fixed shape). _cpu lists are not needed in the captured path.
                if forward_batch.extend_prefix_lens is None:
                    forward_batch.extend_prefix_lens = forward_batch.seq_lens.to(torch.int32)

        self.kv_pool.set_kv_buffer(
            layer,
            forward_batch.out_cache_loc,
            k,
            v,
        )
        if disable_value:
            self.kv_pool.set_index_k_buffer(
                layer,
                forward_batch.out_cache_loc,
                idx_k,
            )
        else:
            self.kv_pool.set_index_kv_buffer(
                layer,
                forward_batch.out_cache_loc,
                idx_k,
                idx_v,
            )
        k_cache, v_cache = self.kv_pool.get_kv_buffer(layer.layer_id)
        if disable_value:
            idx_k_cache = self.kv_pool.get_index_k_buffer(layer.layer_id)
            idx_v_cache = None
        else:
            idx_k_cache, idx_v_cache = self.kv_pool.get_index_kv_buffer(
                layer.layer_id
            )

        cu_seqlens = torch.cat(
            [
                torch.zeros(
                    1, dtype=torch.int32, device=forward_batch.extend_seq_lens.device
                ),
                forward_batch.extend_seq_lens.to(torch.int32).cumsum(0).to(torch.int32),
            ]
        )
        # Kernel expects seq_lens = prefix + extend. For normal prefill,
        # forward_batch.seq_lens already is prefix+extend. For EAGLE3
        # TARGET_VERIFY, prepare_for_v2_verify leaves seq_lens as the prefix
        # (the scheduler adds accept_lens after verify), so rebuild it as
        # prefix + extend here.
        raw_seq_lens = forward_batch.seq_lens.to(torch.int32)
        if forward_batch.forward_mode.is_target_verify():
            # EAGLE3 verify: seq_lens is the PREFIX (scheduler adds accept_lens
            # after). Use raw_seq_lens directly as prefix_lens — it is a buffer
            # reference that holds the REAL per-request prefix at replay time.
            # Do NOT use forward_batch.extend_prefix_lens: under cuda graph it
            # was materialised once at capture time with a fixed value and is
            # not updated on replay, so it would be stale and garble output.
            prefix_lens = raw_seq_lens
            seq_lens = raw_seq_lens + forward_batch.extend_seq_lens.to(torch.int32)
        elif forward_batch.extend_prefix_lens is not None:
            prefix_lens = forward_batch.extend_prefix_lens.to(torch.int32)
            seq_lens = raw_seq_lens  # normal extend: already prefix + extend
        else:
            prefix_lens = torch.zeros_like(raw_seq_lens)
            seq_lens = raw_seq_lens

        # In DP attention mode, q may be padded beyond the actual token count
        # for collective communication alignment. Trim to actual tokens so
        # the sparse attention kernel sees consistent shapes.
        # NOTE: .item() is a host sync and is illegal inside CUDA graph capture.
        # Under capture (EAGLE3 TARGET_VERIFY graph), shapes are fixed and
        # non-DP, so cu_seqlens[-1] == q.shape[0] == num_tokens; use the static
        # q.shape[0] instead of .item(). .item() is only safe in eager mode.
        if torch.cuda.is_current_stream_capturing():
            actual_num_tokens = q.shape[0]
        else:
            actual_num_tokens = int(cu_seqlens[-1].item())
        original_num_tokens = q.shape[0]
        if actual_num_tokens < original_num_tokens:
            q = q[:actual_num_tokens]
            idx_q = idx_q[:actual_num_tokens]

        idx_o, o = minimax_sparse_prefill(
            q,
            k_cache,
            v_cache,
            None,
            idx_q,
            idx_k_cache,
            idx_v_cache,
            None,
            self.req_to_token,
            forward_batch.req_pool_indices,
            cu_seqlens,
            seq_lens,
            prefix_lens,
            self._max_seqlen_q,
            self._max_seqlen_k,
            self.block_size_q,
            self.block_size_k,
            self.topk_blocks,
            self.init_blocks,
            self.local_blocks,
            score_type=self.score_type,
            disable_index_value=disable_value,
        )

        # Pad output back to original size for DP communication
        if actual_num_tokens < original_num_tokens:
            pad_len = original_num_tokens - actual_num_tokens
            o = torch.cat(
                [o, o.new_zeros(pad_len, *o.shape[1:])], dim=0
            )
            if idx_o is not None:
                idx_o = torch.cat(
                    [idx_o, idx_o.new_zeros(pad_len, *idx_o.shape[1:])], dim=0
                )

        return (
            None if idx_o is None else idx_o.reshape(original_num_tokens, -1).contiguous(),
            o.reshape(original_num_tokens, -1).contiguous(),
        )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        *,
        idx_q: torch.Tensor,
        idx_k: torch.Tensor,
        idx_v: Optional[torch.Tensor],
    ):
        disable_value = layer.layer_id in self.disable_value_layer_ids
        self.kv_pool.set_kv_buffer(
            layer,
            forward_batch.out_cache_loc,
            k,
            v,
        )
        if disable_value:
            self.kv_pool.set_index_k_buffer(
                layer,
                forward_batch.out_cache_loc,
                idx_k,
            )
        else:
            self.kv_pool.set_index_kv_buffer(
                layer,
                forward_batch.out_cache_loc,
                idx_k,
                idx_v,
            )
        k_cache, v_cache = self.kv_pool.get_kv_buffer(layer.layer_id)
        if disable_value:
            idx_k_cache = self.kv_pool.get_index_k_buffer(layer.layer_id)
            idx_v_cache = None
        else:
            idx_k_cache, idx_v_cache = self.kv_pool.get_index_kv_buffer(
                layer.layer_id
            )

        idx_o, o = minimax_sparse_decode(
            q,
            None,
            k_cache,
            v_cache,
            idx_q,
            None,
            idx_k_cache,
            idx_v_cache,
            self.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            self._max_seqlen_k,
            1,
            self.block_size_k,
            self.topk_blocks,
            self.init_blocks,
            self.local_blocks,
            score_type=self.score_type,
            disable_index_value=disable_value,
        )
        return (
            None if idx_o is None else idx_o.reshape(q.shape[0], -1).contiguous(),
            o.reshape(q.shape[0], -1).contiguous(),
        )


class MiniMaxHybridAttnBackend(AttentionBackend):
    """Combines a dense backend and a sparse backend, routing by call site."""

    def __init__(
        self,
        dense_backend: AttentionBackend,
        sparse_backend: MiniMaxSparseAttnBackend,
        sparse_layer_ids: list[int],
    ):
        self.dense = dense_backend
        self.sparse = sparse_backend
        self.sparse_layer_ids = sparse_layer_ids

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        self.sparse.init_forward_metadata(forward_batch)
        self.dense.init_forward_metadata(forward_batch)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        self.dense.init_cuda_graph_state(max_bs, max_num_tokens)
        self.sparse.init_cuda_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs,
        num_tokens,
        req_pool_indices,
        seq_lens,
        encoder_lens,
        forward_mode,
        spec_info,
    ):
        self.sparse.init_forward_metadata_capture_cuda_graph(
            bs,
            num_tokens,
            req_pool_indices,
            seq_lens,
            encoder_lens,
            forward_mode,
            spec_info,
        )
        self.dense.init_forward_metadata_capture_cuda_graph(
            bs,
            num_tokens,
            req_pool_indices,
            seq_lens,
            encoder_lens,
            forward_mode,
            spec_info,
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs,
        req_pool_indices,
        seq_lens,
        seq_lens_sum,
        encoder_lens,
        forward_mode,
        spec_info,
        seq_lens_cpu,
    ):
        self.sparse.init_forward_metadata_replay_cuda_graph(
            bs,
            req_pool_indices,
            seq_lens,
            seq_lens_sum,
            encoder_lens,
            forward_mode,
            spec_info,
            seq_lens_cpu,
        )
        self.dense.init_forward_metadata_replay_cuda_graph(
            bs,
            req_pool_indices,
            seq_lens,
            seq_lens_sum,
            encoder_lens,
            forward_mode,
            spec_info,
            seq_lens_cpu,
        )

    def get_cuda_graph_seq_len_fill_value(self):
        return self.sparse.get_cuda_graph_seq_len_fill_value()

    def forward(
        self,
        q,
        k,
        v,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        if layer.layer_id in self.sparse_layer_ids:
            return self.sparse.forward(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
        else:
            return self.dense.forward(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )

    def forward_extend(
        self,
        q,
        k,
        v,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        if layer.layer_id in self.sparse_layer_ids:
            return self.sparse.forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
        else:
            return self.dense.forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )

    def forward_decode(
        self,
        q,
        k,
        v,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        if layer.layer_id in self.sparse_layer_ids:
            return self.sparse.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
        else:
            return self.dense.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
