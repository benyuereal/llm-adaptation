from __future__ import annotations

import logging
import os
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
from sglang.srt.layers.attention.minimax_sparse_ops.verify import (
    minimax_sparse_verify_prefill,
)
from sglang.srt.mem_cache.memory_pool import MiniMaxSparseKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

# ============================================================================
# EAGLE3 verify VMFault 定位探针
# 开关: EAGLE3_VERIFY_PROBE=1 启用(默认关). 仅 TP rank0 输出.
# 输出: /workspace/logs/eagle3_verify_probe.log (独立文件, 不污染推理日志)
# graph-safe: capture 阶段(is_current_stream_capturing)只打静态属性, 绝不同步.
# 目标: 定位 verify 走 prefill kernel 时, 哪个张量在 capture/replay 间地址/形状
#       变化导致越界写 VMFault.
# ============================================================================
_VERIFY_PROBE = os.environ.get("EAGLE3_VERIFY_PROBE", "0") == "1"
_VERIFY_PROBE_RANK = 0
try:
    if _VERIFY_PROBE:
        from sglang.srt.distributed import get_tensor_model_parallel_rank
        _VERIFY_PROBE_RANK = get_tensor_model_parallel_rank()
except Exception:
    pass
_PROBE_FH = None
if _VERIFY_PROBE and _VERIFY_PROBE_RANK == 0:
    try:
        _PROBE_FH = open("/workspace/logs/eagle3_verify_probe.log", "a", buffering=1)
        _PROBE_FH.write(f"\n===== EAGLE3 verify probe started pid={os.getpid()} =====\n")
    except Exception:
        _PROBE_FH = None


def _probe_log(msg: str):
    if _PROBE_FH is not None:
        try:
            _PROBE_FH.write(msg + "\n")
            _PROBE_FH.flush()
        except Exception:
            pass


def _probe_tensor(tag: str, t, max_vals=32):
    """graph-safe 打印 tensor. capture: 只打静态属性(ptr/shape/stride); replay/eager: 打值.
    关键: capture 和 replay 都打 ptr —— graph buffer 的 ptr 在 capture/replay 间应恒定,
    临时张量的 ptr 会变. 这是定位越界张量的核心信号."""
    if not _VERIFY_PROBE or _VERIFY_PROBE_RANK != 0:
        return
    if t is None:
        _probe_log(f"  {tag}: None")
        return
    try:
        if not isinstance(t, torch.Tensor):
            _probe_log(f"  {tag}: {t}")
            return
        if t.numel() == 0:
            _probe_log(f"  {tag}: empty shape={tuple(t.shape)} dtype={t.dtype} ptr={t.data_ptr()}")
            return
        capturing = torch.cuda.is_current_stream_capturing()
        ptr = t.data_ptr()
        shape = tuple(t.shape)
        stride = tuple(t.stride())
        if capturing:
            # capture: 绝不同步, 只打静态属性
            _probe_log(f"  {tag}: CAPTURE shape={shape} stride={stride} dtype={t.dtype} ptr={ptr:#x}")
        else:
            # replay/eager: 可同步, 打 ptr + 少量值
            flat = t.detach()
            if flat.numel() <= max_vals:
                vals = flat.tolist()
            else:
                vals = (f"shape={shape} dtype={flat.dtype} "
                        f"min={flat.min().item()} max={flat.max().item()}")
            _probe_log(f"  {tag}: REPLAY shape={shape} stride={stride} dtype={t.dtype} ptr={ptr:#x} vals={vals}")
    except Exception as e:
        _probe_log(f"  {tag}: probe-err {e}")



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
        # Graph-safe upper bound for the score buffer's K-block dimension.
        # The prefill score tensor is allocated as [num_heads, total_q,
        # cdiv(max_seqlen_k, block_size_k)]. Under EAGLE3 TARGET_VERIFY this
        # prefill path runs inside a cuda graph: capture uses a dummy
        # seq_lens=1 (see get_cuda_graph_seq_len_fill_value) so _max_seqlen_k
        # is tiny (~draft_token_num), while replay uses the real seq_lens
        # (~thousands). cuda graph locks the allocation shape at capture time,
        # so replay writes score[..., real_block_id] past the captured 3rd dim
        # -> out-of-bounds write -> VMFault.
        # Fix: allocate the score 3rd dim against a CONSTANT upper bound
        # (same shape at capture & replay). Kernels index score via
        # cu_seqblocks_k / real seq_lens with boundary_check, so the extra slots
        # are never accessed; causal is unchanged. Memory cost is modest (v1
        # Strategy B measured ~0.09GB/card at context_len=204800). This mirrors
        # the sglang triton backend's pattern of sizing graph buffers off
        # max_context_len, not live seqlen.
        # The bound must cover the worst case _max_seqlen_k = seq_lens.max() +
        # draft_token_num. seq_lens.max() <= context_len, and the largest
        # draft_token_num seen is speculative_num_draft_tokens; add one extra
        # block of slack to be safe against off-by-one at block boundaries.
        context_len = int(getattr(runner.model_config, "context_len", 0) or 0)
        if context_len <= 0:
            context_len = 1
        max_draft = int(getattr(runner.server_args, "speculative_num_draft_tokens", 0) or 0)
        # fall back to a generous constant if the spec arg is absent/zero
        if max_draft <= 0:
            max_draft = 8
        max_seqlen_k_bound = context_len + max_draft
        self.max_seqblock_k_upper = (
            max_seqlen_k_bound + self.block_size_k - 1
        ) // self.block_size_k
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
        # Pre-allocate graph-stable buffers for EAGLE3 TARGET_VERIFY.
        #
        # Probe (EAGLE3_VERIFY_PROBE) confirmed that the OLD verify path built
        # `cu_seqlens` / `seq_lens` / `extend_seq_lens` with torch.cat / torch.full
        # / `a + b` INSIDE forward_extend. Those are fresh temporaries each call
        # -> their data_ptr differs between capture and replay -> cuda graph
        # captured the capture-time ptr, but replay reads a different address
        # -> garbage / VMFault. This is the REAL root cause the upper-bound
        # score fix alone did not solve.
        #
        # Fix: pre-allocate these as graph buffers here (one-shot, address fixed
        # for the backend's lifetime), and in forward_extend's verify branch
        # WRITE INTO them (same address at capture & replay). This mirrors the
        # sglang triton backend's `self.qo_indptr` pattern (triton_backend.py
        # init_cuda_graph_state + capture/replay arange).
        #
        # verify invariant: every request has the SAME draft_token_num D, so:
        #   extend_seq_lens = [D, D, ..., D]            (constant per graph)
        #   cu_seqlens      = [0, D, 2D, ..., bs*D]     (constant per graph)
        # Both depend only on (bs, D) which are fixed for a given captured
        # graph -> truly graph-safe (no seq_len dependency).
        device = self.kv_pool.device if hasattr(self.kv_pool, "device") else "cuda"
        try:
            device = self.req_to_token.device
        except Exception:
            pass
        self._verify_max_bs = int(max_bs)
        # extend_seq_lens buffer: [max_bs] of D. Filled per-capture with the
        # actual D (constant). Reused at replay (content stays [D]*bs).
        self._verify_extend_seq_lens_buf = torch.zeros(
            (max_bs,), dtype=torch.int32, device=device
        )
        # cu_seqlens buffer: [max_bs + 1]. Filled with arange(0, (bs+1)*D, D).
        self._verify_cu_seqlens_buf = torch.zeros(
            (max_bs + 1,), dtype=torch.int32, device=device
        )
        # seq_lens buffer: [max_bs] of (prefix + D). At replay this must hold
        # the REAL per-request prefix+D. We write `forward_batch.seq_lens + D`
        # into it at replay (forward_batch.seq_lens is the graph seq_lens buffer,
        # address-stable, filled by replay_prepare).
        self._verify_seq_lens_buf = torch.zeros(
            (max_bs,), dtype=torch.int32, device=device
        )

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
            D = int(draft_token_num)
            self._max_seqlen_q = D
            # _max_seqlen_k only sizes the score buffer's K-block dim; the
            # verify kernel uses max_seqblock_k_upper (constant) for that, so
            # the live value here is only a fallback. Keep it at the constant
            # upper bound to avoid any dynamic-shape leak.
            self._max_seqlen_k = self.max_seqblock_k_upper * self.block_size_k
            # Pre-fill the constant verify buffers (graph-safe: depend only on
            # bs & D, both fixed for this captured graph).
            self._verify_extend_seq_lens_buf[:bs].fill_(D)
            self._verify_cu_seqlens_buf[: bs + 1] = torch.arange(
                0, (bs + 1) * D, step=D, dtype=torch.int32,
                device=self._verify_cu_seqlens_buf.device,
            )
            # seq_lens (K length = prefix + D). At capture seq_lens is the dummy
            # fill value (=1), so prefix+D = 1+D. This is just a safe dummy so
            # the captured kernel run does not crash; replay overwrites it with
            # the real prefix+D in init_forward_metadata_replay_cuda_graph.
            self._verify_seq_lens_buf[:bs] = seq_lens[:bs].to(torch.int32) + D
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
            D = int(draft_token_num)
            self._max_seqlen_q = D
            self._max_seqlen_k = self.max_seqblock_k_upper * self.block_size_k
            # Re-fill constant buffers (same content as capture, same address).
            # cu_seqlens / extend_seq_lens are constant [0,D,2D,..] / [D]*bs.
            self._verify_extend_seq_lens_buf[:bs].fill_(D)
            self._verify_cu_seqlens_buf[: bs + 1] = torch.arange(
                0, (bs + 1) * D, step=D, dtype=torch.int32,
                device=self._verify_cu_seqlens_buf.device,
            )
            # seq_lens = real prefix (from the graph seq_lens buffer) + D.
            # `seq_lens` here is buffers.seq_lens[:bs] (address-stable graph
            # buffer filled by replay_prepare with the real per-request prefix).
            # Write prefix+D into our pre-allocated buffer (address-stable).
            self._verify_seq_lens_buf[:bs] = seq_lens[:bs].to(torch.int32) + D
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

        is_verify = forward_batch.forward_mode.is_target_verify()

        if is_verify:
            # ---- EAGLE3 TARGET_VERIFY: route to the graph-safe verify kernel ----
            # The OLD path built cu_seqlens/seq_lens/extend_seq_lens as temporaries
            # inside forward_extend (torch.cat / a+b / torch.full). Probe confirmed
            # their data_ptr DIFFERS between capture and replay -> cuda graph
            # captured the capture-time ptr but replay read a different address ->
            # garbage / VMFault. (The score upper-bound fix alone was not enough.)
            #
            # NEW path: use the PRE-ALLOCATED graph buffers from
            # init_cuda_graph_state (address-stable for the backend's lifetime),
            # filled in init_forward_metadata_{capture,replay}_cuda_graph. These
            # depend only on (bs, D) which are constant per captured graph.
            return self._forward_verify(q, k, v, layer, forward_batch,
                                        save_kv_cache, disable_value,
                                        idx_q=idx_q, idx_k=idx_k, idx_v=idx_v)

        # ---- normal extend (prefill): unchanged dynamic path ----
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
            if not torch.cuda.is_current_stream_capturing():
                if forward_batch.extend_seq_lens_cpu is None:
                    forward_batch.extend_seq_lens_cpu = [int(draft_token_num)] * num_reqs
                if forward_batch.extend_prefix_lens is None:
                    forward_batch.extend_prefix_lens = forward_batch.seq_lens.to(torch.int32)
                    forward_batch.extend_prefix_lens_cpu = (
                        forward_batch.extend_prefix_lens.cpu().tolist()
                    )
            else:
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
        raw_seq_lens = forward_batch.seq_lens.to(torch.int32)
        if forward_batch.extend_prefix_lens is not None:
            prefix_lens = forward_batch.extend_prefix_lens.to(torch.int32)
            seq_lens = raw_seq_lens  # normal extend: already prefix + extend
        else:
            prefix_lens = torch.zeros_like(raw_seq_lens)
            seq_lens = raw_seq_lens

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
            max_seqblock_k_upper=self.max_seqblock_k_upper,
        )

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

    def _forward_verify(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache: bool,
        disable_value: bool,
        *,
        idx_q: torch.Tensor,
        idx_k: torch.Tensor,
        idx_v: Optional[torch.Tensor],
    ):
        """EAGLE3 TARGET_VERIFY via the graph-safe verify kernel.

        Uses pre-allocated graph buffers (init_cuda_graph_state) for
        cu_seqlens / extend_seq_lens / seq_lens so their addresses are
        identical at capture and replay. All shapes depend only on (bs, D)
        and the constant max_seqblock_k_upper — no live seq_len dependency
        in any allocation or grid. The kernel still reads real seq_lens at
        runtime for causal / KV indexing (via the seq_lens buffer)."""
        bs = forward_batch.seq_lens.shape[0]

        # write k/v + idx into paged caches (same as forward_extend/decode)
        self.kv_pool.set_kv_buffer(
            layer, forward_batch.out_cache_loc, k, v,
        )
        if disable_value:
            self.kv_pool.set_index_k_buffer(
                layer, forward_batch.out_cache_loc, idx_k,
            )
        else:
            self.kv_pool.set_index_kv_buffer(
                layer, forward_batch.out_cache_loc, idx_k, idx_v,
            )
        k_cache, v_cache = self.kv_pool.get_kv_buffer(layer.layer_id)
        if disable_value:
            idx_k_cache = self.kv_pool.get_index_k_buffer(layer.layer_id)
            idx_v_cache = None
        else:
            idx_k_cache, idx_v_cache = self.kv_pool.get_index_kv_buffer(
                layer.layer_id
            )

        # ---- graph-stable buffers (pre-allocated, address fixed) ----
        # cu_seqlens = [0, D, 2D, ..., bs*D]  (constant, filled in capture/replay)
        # extend_seq_lens = [D]*bs            (constant)
        # seq_lens = real_prefix + D          (filled in replay from graph buffer)
        # prefix_lens = forward_batch.seq_lens (the graph seq_lens buffer = real
        #   prefix at replay; address-stable). verify's seq_lens is the PREFIX,
        #   so prefix_lens == seq_lens buffer (NOT prefix+D).
        #
        # Graph path: buffers were pre-allocated in init_cuda_graph_state and
        # filled in init_forward_metadata_{capture,replay}_cuda_graph. Their
        # data_ptr is identical at capture & replay -> graph-safe.
        # Eager path (bs > cuda_graph_max_bs, or cuda graph disabled): buffers
        # do NOT exist (init_cuda_graph_state is never called). Fall back to
        # constructing temporaries here — eager does not need graph-safety.
        has_graph_buf = hasattr(self, "_verify_cu_seqlens_buf")
        prefix_lens = forward_batch.seq_lens.to(torch.int32)
        if has_graph_buf:
            cu_seqlens = self._verify_cu_seqlens_buf[: bs + 1]
            extend_seq_lens = self._verify_extend_seq_lens_buf[:bs]
            seq_lens = self._verify_seq_lens_buf[:bs]
            if not torch.cuda.is_current_stream_capturing():
                # eager-with-graph-buf (rare): fill seq_lens now. Under capture
                # the buffer was filled in init_capture (dummy); under replay
                # it was filled in init_replay (real prefix+D).
                seq_lens.copy_(prefix_lens + extend_seq_lens)
        else:
            # pure eager (no graph buffers): build temporaries, graph-safety N/A.
            spec_info = getattr(forward_batch, "spec_info", None)
            D = int(getattr(spec_info, "draft_token_num", self._max_seqlen_q))
            extend_seq_lens = torch.full(
                (bs,), D, dtype=torch.int32, device=prefix_lens.device,
            )
            cu_seqlens = torch.cat(
                [
                    torch.zeros(1, dtype=torch.int32, device=prefix_lens.device),
                    extend_seq_lens.cumsum(0).to(torch.int32),
                ]
            )
            seq_lens = prefix_lens + extend_seq_lens

        original_num_tokens = q.shape[0]

        # ---- probe: confirm buffer addresses are stable capture vs replay ----
        if _VERIFY_PROBE and _VERIFY_PROBE_RANK == 0:
            try:
                _cap = torch.cuda.is_current_stream_capturing()
                _tag = "CAPTURE" if _cap else "REPLAY"
                _probe_log(f"[V {_tag}] _forward_verify layer={layer.layer_id} "
                           f"bs={bs} D={int(self._max_seqlen_q)} "
                           f"max_seqblock_k_upper={self.max_seqblock_k_upper} "
                           f"disable_value={disable_value}")
                _probe_tensor("  q", q)
                _probe_tensor("  cu_seqlens(buf)", cu_seqlens)
                _probe_tensor("  extend_seq_lens(buf)", extend_seq_lens)
                _probe_tensor("  seq_lens(buf)", seq_lens)
                _probe_tensor("  prefix_lens(fb.seq_lens)", prefix_lens)
                _probe_tensor("  req_pool_indices", forward_batch.req_pool_indices)
                _probe_tensor("  out_cache_loc", forward_batch.out_cache_loc)
            except Exception as _e:
                _probe_log(f"[V] probe-err: {_e}")

        idx_o, o = minimax_sparse_verify_prefill(
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
            self.max_seqblock_k_upper,
            self.block_size_q,
            self.block_size_k,
            self.topk_blocks,
            self.init_blocks,
            self.local_blocks,
            score_type=self.score_type,
            disable_index_value=disable_value,
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
