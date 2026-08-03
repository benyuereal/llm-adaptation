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
from sglang.srt.layers.attention.minimax_sparse_ops.common.utils import (
    seqlens_expand_triton,
)
from sglang.srt.layers.attention.minimax_sparse_ops.minimax_sparse import (
    minimax_sparse_decode,
    minimax_sparse_prefill,
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
# 定位目标: 并发 verify 时哪个动态张量越界(单并发不崩, 无EAGLE3不崩)
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
    """探针输出到独立文件, 不影响推理日志/性能路径外的 stdout."""
    if _PROBE_FH is not None:
        try:
            _PROBE_FH.write(msg + "\n")
            _PROBE_FH.flush()
        except Exception:
            pass


def _probe_tensor(tag: str, t, max_vals=32):
    """graph-safe 打印 tensor: capture 时打 shape/dtype/ptr, replay/eager 时打值."""
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
            _probe_log(f"  {tag}: empty shape={tuple(t.shape)} dtype={t.dtype}")
            return
        if torch.cuda.is_current_stream_capturing():
            # capture: 绝不同步, 只打静态属性
            _probe_log(f"  {tag}: CAPTURE shape={tuple(t.shape)} dtype={t.dtype} ptr={t.data_ptr()}")
        else:
            # replay/eager: 可同步, 打真实值/范围
            flat = t.detach()
            if flat.numel() <= max_vals:
                vals = flat.tolist()
            else:
                vals = (f"shape={tuple(flat.shape)} dtype={flat.dtype} "
                        f"min={flat.min().item()} max={flat.max().item()}")
            _probe_log(f"  {tag}: {vals}")
    except Exception as e:
        _probe_log(f"  {tag}: probe-err {e}")


class MiniMaxSparseAttnBackend(AttentionBackend):
    def __init__(self, runner: "ModelRunner"):

        assert isinstance(runner.token_to_kv_pool, MiniMaxSparseKVPool)
        self.kv_pool = runner.token_to_kv_pool
        self.req_to_token = runner.req_to_token_pool.req_to_token
        # EAGLE3 Strategy A: keep a handle to the runner so forward_extend can
        # read server_args.speculative_attention_mode ("decode" routes verify
        # through the graph-safe decode kernel; "prefill" keeps the existing
        # crashing-under-graph prefill path as a fallback). Falls back to the
        # EAGLE3_VERIFY_DECODE env var if server_args is unavailable.
        self.runner = runner

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
        # Upper bound on KV length, used as fixed max_seqlen_k under cuda graph.
        # The sparse prefill kernel allocates a `score` tensor of shape
        # (num_heads, total_q, ceil(max_seqlen_k / block_size_k)). Under cuda
        # graph, capture uses dummy seq_lens=1 -> max_seqlen_k~5 -> tiny score
        # tensor; replay has real seq_lens~190 -> kernel writes past the score
        # tensor -> HIP VMFault. Using the context-length upper bound makes the
        # score tensor a fixed size across capture/replay (graph-safe). The
        # extra memory is ~0.5MB/layer for verify's small total_q, and the
        # kernel only loops over real valid_blocks (no compute waste).
        self._max_seqlen_k_upper = int(runner.model_config.context_len)

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
        #
        # IMPORTANT (cuda graph): the sparse prefill kernel allocates a `score`
        # tensor of shape (num_heads, total_q, ceil(max_seqlen_k / block_size_k)).
        # Under cuda graph, capture runs with dummy seq_lens=1 (fill_value), so
        # max_seqlen_k~5 -> tiny score tensor; replay has real seq_lens~190 ->
        # kernel writes past the score tensor -> HIP VMFault. Use the
        # context-length upper bound so the score tensor has a fixed size across
        # capture/replay (graph-safe). Extra memory ~0.5MB/layer for verify's
        # small total_q; the kernel only loops over real valid_blocks (no waste).
        draft_token_num = getattr(spec_info, "draft_token_num", None)
        if forward_mode.is_target_verify() and draft_token_num is not None:
            self._max_seqlen_q = int(draft_token_num)
            self._max_seqlen_k = self._max_seqlen_k_upper
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
        # Same TARGET_VERIFY fix as capture path above. Under replay, real
        # seq_lens are restored, but we keep max_seqlen_k at the upper bound so
        # the score tensor (allocated at capture time with this size) is not
        # overflowed. The kernel uses real seq_lens for actual KV access; the
        # upper-bound max_seqlen_k only sizes the score scratch buffer.
        draft_token_num = getattr(spec_info, "draft_token_num", None)
        if forward_mode.is_target_verify() and draft_token_num is not None:
            self._max_seqlen_q = int(draft_token_num)
            self._max_seqlen_k = self._max_seqlen_k_upper
        else:
            self._max_seqlen_q = 1
            self._max_seqlen_k = int(seq_lens_cpu[:bs].max().item())

        # ---- 探针 A: REPLAY 时点(replay 前唯一执行 Python 的地方)----
        # 重点: forward_extend 里的探针 B/C/D 在 replay 时不执行(被 graph 跳过),
        # 只有这里(replay_prepare 调的)能在 replay 时拿到真实值.
        # 通过 self._replay_forward_batch(cuda_graph_runner line 64 设置)拿 out_cache_loc.
        if (_VERIFY_PROBE and _VERIFY_PROBE_RANK == 0
                and forward_mode.is_target_verify() and draft_token_num is not None):
            try:
                _probe_log(f"[A REPLAY] target_verify bs={bs} draft_token_num={draft_token_num}")
                _probe_log(f"  max_seqlen_q={self._max_seqlen_q} max_seqlen_k={self._max_seqlen_k} "
                           f"max_seqlen_k_upper={self._max_seqlen_k_upper} "
                           f"block_size_k={self.block_size_k} topk_blocks={self.topk_blocks}")
                _probe_tensor("  seq_lens", seq_lens[:bs])
                _probe_tensor("  seq_lens_cpu", seq_lens_cpu[:bs] if seq_lens_cpu is not None else None)
                _probe_tensor("  req_pool_indices", req_pool_indices[:bs] if req_pool_indices is not None else None)
                # out_cache_loc: set_kv_buffer 用它写 k_cache, 越界则 VMFault
                _rfb = getattr(self, "_replay_forward_batch", None)
                if _rfb is not None and getattr(_rfb, "out_cache_loc", None) is not None:
                    _probe_tensor("  out_cache_loc", _rfb.out_cache_loc)
                # OOB 检查: kernel 访问 req_to_token[sid, pos]
                if req_pool_indices is not None:
                    _max_slots = self.kv_pool.size
                    _r2t = self.req_to_token
                    _sl_cpu = seq_lens[:bs].detach().to(torch.int32)
                    _rp_cpu = req_pool_indices[:bs].detach().to(torch.int64)
                    # topk_idx 值理论上界 = max_seqblock_k = ceil(max_seqlen_k_upper / block_size_k)
                    # kernel 用 pos = topk_idx * block_size_k + off_n, pos 最大可达 max_seqblock_k*block_size_k
                    # 所以 req_to_token 的完整检查范围 = max_seqblock_k * block_size_k (= max_seqlen_k_upper)
                    _max_pos = min(self._max_seqlen_k_upper, _r2t.shape[1])
                    _bad = []
                    for _i in range(int(_sl_cpu.shape[0])):
                        _plen = int(_sl_cpu[_i].item())
                        _ridx = int(_rp_cpu[_i].item())
                        if _ridx < 0 or _ridx >= _r2t.shape[0]:
                            _bad.append(f"req {_i}: req_pool_idx={_ridx} >= r2t_rows={_r2t.shape[0]} (行越界!)")
                            continue
                        # 1) draft 范围 (set_kv_buffer 写): req_to_token[req, prefix:prefix+draft]
                        _rlen = _plen + int(draft_token_num)
                        _row = _r2t[_ridx, :_rlen]
                        _mx = int(_row.max().item())
                        _mn = int(_row.min().item())
                        if _mx >= _max_slots or _mn < 0:
                            _bad.append(f"req {_i} pool={_ridx} prefix={_plen} rlen={_rlen}: "
                                        f"draft slot [{_mn},{_mx}] >= max_slots={_max_slots} (写k_cache越界!)")
                        # 2) topk 完整范围 (sparse kernel 读): req_to_token[req, 0:max_seqblock_k*block_size_k]
                        #    覆盖 topk_idx 值异常(>=valid_blocks)时 pos 越界的情况
                        _krow = _r2t[_ridx, :_max_pos]
                        _kmx = int(_krow.max().item())
                        _kmn = int(_krow.min().item())
                        if _kmx >= _max_slots or _kmn < 0:
                            _bad.append(f"req {_i} pool={_ridx}: topk slot [{_kmn},{_kmx}] "
                                        f">= max_slots={_max_slots} (max_pos={_max_pos}, 读k_cache越界!)")
                    # 3) out_cache_loc 范围检查 (set_kv_buffer 写 k_cache 的位置)
                    if _rfb is not None and getattr(_rfb, "out_cache_loc", None) is not None:
                        _ocl = _rfb.out_cache_loc.detach()
                        _omn, _omx = int(_ocl.min().item()), int(_ocl.max().item())
                        _ooob = int(((_ocl >= _max_slots) | (_ocl < 0)).sum().item())
                        if _ooob:
                            _bad.append(f"out_cache_loc: min={_omn} max={_omx} oob_cnt={_ooob} "
                                        f">= max_slots={_max_slots} (写k_cache越界!)")
                    if _bad:
                        _probe_log(f"[A REPLAY] !!! OOB DETECTED ({len(_bad)}) !!!")
                        for _b in _bad:
                            _probe_log(f"  {_b}")
                    else:
                        _probe_log(f"[A REPLAY] slots OK (max_slots={_max_slots}, r2t={tuple(_r2t.shape)}, "
                                   f"max_pos_checked={_max_pos})")
            except Exception as _e:
                _probe_log(f"[A REPLAY] probe-err: {_e}")

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

    def _verify_decode_mode(self) -> str:
        """Determine the EAGLE3 verify attention mode.

        Returns "decode" (route verify through the graph-safe decode kernel,
        Strategy A) or "prefill" (keep the existing prefill path). Reads
        server_args.speculative_attention_mode via self.runner; falls back to
        the EAGLE3_VERIFY_DECODE env var (set by the start script) if
        server_args is not reachable. Default "prefill" preserves prior
        behavior when neither is set.
        """
        # Env var takes precedence as an explicit override (1=decode, 0=prefill).
        env = os.environ.get("EAGLE3_VERIFY_DECODE", "")
        if env == "1":
            return "decode"
        if env == "0":
            return "prefill"
        # Otherwise consult server_args.speculative_attention_mode.
        try:
            sa = getattr(self.runner, "server_args", None)
            mode = getattr(sa, "speculative_attention_mode", None)
            if mode == "decode":
                return "decode"
            if mode == "prefill":
                return "prefill"
        except Exception:
            pass
        return "prefill"

    def _forward_verify_via_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache: bool,
        *,
        idx_q: torch.Tensor,
        idx_k: torch.Tensor,
        idx_v: Optional[torch.Tensor],
    ):
        """EAGLE3 Strategy A: route target_verify through the graph-safe
        DECODE kernel instead of the crashing prefill kernel.

        verify has q_len = draft_token_num D per request. The decode kernel
        assumes q_len = 1 per request and is fundamentally graph-safe (its
        `score` scratch tensor is sized by req_to_token.shape[1] which is
        STATIC, and its grid is bs*NUM_KV_CHUNKS independent of seq_len). So
        we EXPAND the verify batch to look like bs*D independent single-token
        requests and reuse forward_decode's kernel path verbatim.

        Token layout: verify's q / idx_q / idx_k / idx_v / out_cache_loc
        arrive as flat [bs*D, ...] tensors in interleaved-by-request order
        [r0_t0..r0_t{D-1}, r1_t0, ...] (see assign_extend_cache_locs in
        eagle_info_v2.py). So q is ALREADY [bs*D, num_heads, head_dim] — no
        reshape from [bs, D, ...] is needed; we only build the expanded
        per-(request,draft-token) seq_lens and req_pool_indices.
        """
        disable_value = layer.layer_id in self.disable_value_layer_ids

        # ---- draft_token_num D and per-request KV length ----
        spec_info = getattr(forward_batch, "spec_info", None)
        D = getattr(spec_info, "draft_token_num", None)
        bs = forward_batch.seq_lens.shape[0]
        if D is None:
            # Fallback: infer from token count (uniform D per request).
            D = forward_batch.input_ids.shape[0] // max(bs, 1)
        D = int(D)

        # For verify, forward_batch.seq_lens is the PREFIX (the scheduler adds
        # accept_lens AFTER verify). The real KV length after the D draft
        # tokens are written = prefix + D. Rebuild it here, mirroring the
        # existing forward_extend logic (lines ~494-495). extend_seq_lens is
        # materialised to [bs] of D by init_forward_metadata / the forward_extend
        # prelude; if it is still None under graph capture, build a fixed-shape
        # tensor (graph-safe, no host sync).
        device = forward_batch.seq_lens.device
        prefix_lens = forward_batch.seq_lens.to(torch.int32)
        # extend_seq_lens for EAGLE3 verify is uniform D per request, but its
        # TYPE varies by path:
        #   - eager (forward_mode.is_target_verify via eagle_info_v2.py:236):
        #     a Python list [D, D, ...]
        #   - cuda graph capture/replay: None (the captured ForwardBatch does
        #     not carry it; forward_extend materialises it, but we route here
        #     BEFORE that materialisation). Some paths may also hand a tensor.
        # Build a uniform int32 [bs] tensor of D in all cases (graph-safe: the
        # output shape [bs] is fixed, and D is a python int constexpr). Never
        # call .to() on a list.
        ext_raw = forward_batch.extend_seq_lens
        if isinstance(ext_raw, torch.Tensor):
            ext = ext_raw.to(torch.int32)
        elif ext_raw is not None and len(ext_raw) > 0:
            # Python list (eager eagle_info path). All entries equal D for verify.
            ext = torch.full((bs,), int(ext_raw[0]), dtype=torch.int32, device=device)
        else:
            ext = torch.full((bs,), D, dtype=torch.int32, device=device)
        # kv_len per request = prefix + D (the D draft tokens are written to
        # the cache by set_kv_buffer below before the kernel reads them).
        seq_lens_kv = prefix_lens + ext  # [bs] int32

        # ---- expanded tensors: bs*D independent single-token requests ----
        # seq_lens_expanded[r*D + j] = kv_len_r - D + 1 + j = prefix_r + j + 1
        # (causal: draft token j attends to prefix + earlier draft tokens).
        seq_lens_expanded = seqlens_expand_triton(seq_lens_kv, D)
        # Each of the D expanded rows for request r maps to the same request r.
        # Match forward_decode: pass req_pool_indices as-is (the decode kernel
        # loads slot_ids and casts to int64 internally, so any int dtype works).
        # Use view-based expand+reshape (NO kernel launch, fully graph-safe)
        # instead of torch.repeat_interleave (which launches a kernel that may
        # not capture cleanly on HIP). [bs] -> [bs, D] -> [bs*D] interleaved.
        req_pool_indices_expanded = (
            forward_batch.req_pool_indices.unsqueeze(1)
            .expand(bs, D)
            .reshape(bs * D)
        )
        # out_cache_loc is already [bs*D] flat (interleaved-by-request).
        out_cache_loc = forward_batch.out_cache_loc
        assert out_cache_loc.shape[0] == bs * D, (
            f"verify out_cache_loc len {out_cache_loc.shape[0]} != bs*D={bs * D}"
        )

        # ---- write k/v and idx_k/idx_v into the paged caches ----
        # Uses the SAME buffers/calls as forward_decode, just with the verify
        # out_cache_loc (which already covers all D draft slots per request).
        self.kv_pool.set_kv_buffer(layer, out_cache_loc, k, v)
        if disable_value:
            self.kv_pool.set_index_k_buffer(layer, out_cache_loc, idx_k)
        else:
            self.kv_pool.set_index_kv_buffer(layer, out_cache_loc, idx_k, idx_v)
        k_cache, v_cache = self.kv_pool.get_kv_buffer(layer.layer_id)
        if disable_value:
            idx_k_cache = self.kv_pool.get_index_k_buffer(layer.layer_id)
            idx_v_cache = None
        else:
            idx_k_cache, idx_v_cache = self.kv_pool.get_index_kv_buffer(
                layer.layer_id
            )

        # ---- idx_q expansion (indexer query for the sparse topk step) ----
        # idx_q arrives as [bs*D, num_idx_heads, idx_head_dim] (flat, same
        # interleaved-by-request layout as q). The decode kernel's idx_q arg
        # is [batch_size, num_idx_heads, idx_head_dim] with batch_size = bs*D
        # here, so NO expansion is needed — it already has the right shape.
        idx_q_expanded = idx_q

        # ---- 探针: 记录 verify-decode 路径的关键张量 (capture/replay 都生效) ----
        if (_VERIFY_PROBE and _VERIFY_PROBE_RANK == 0):
            try:
                _cap = torch.cuda.is_current_stream_capturing()
                _tag = "CAPTURE" if _cap else "EAGER/REPLAY"
                _probe_log(f"[V {_tag}] verify->decode layer={layer.layer_id} "
                           f"bs={bs} D={D} bsD={bs * D} "
                           f"max_seqlen_k={self._max_seqlen_k} "
                           f"block_size_k={self.block_size_k} "
                           f"topk_blocks={self.topk_blocks} "
                           f"disable_value={disable_value}")
                _probe_tensor("  q", q)
                _probe_tensor("  idx_q", idx_q_expanded)
                _probe_tensor("  seq_lens_kv", seq_lens_kv)
                _probe_tensor("  seq_lens_expanded", seq_lens_expanded)
                _probe_tensor("  req_pool_indices_expanded", req_pool_indices_expanded)
                _probe_tensor("  out_cache_loc", out_cache_loc)
            except Exception as _e:
                _probe_log(f"[V] probe-err layer={layer.layer_id}: {_e}")

        # ---- call the SAME decode kernel path as forward_decode ----
        # Args mirror forward_decode exactly, with the expanded tensors and
        # max_seqlen = self._max_seqlen_k (the kernel only uses it as an upper
        # bound for chunking; for verify we keep _max_seqlen_k_upper set by
        # init_forward_metadata_capture/replay, which is graph-safe).
        idx_o, o = minimax_sparse_decode(
            q,                                  # [bs*D, num_heads, head_dim]
            None,                               # sink (unused, None like decode)
            k_cache,
            v_cache,
            idx_q_expanded,                     # [bs*D, num_idx_heads, idx_head_dim]
            None,                               # idx_sink
            idx_k_cache,
            idx_v_cache,
            self.req_to_token,
            req_pool_indices_expanded,          # [bs*D]
            seq_lens_expanded,                  # [bs*D]
            self._max_seqlen_k,
            1,                                  # block_size_q (unused, =1)
            self.block_size_k,
            self.topk_blocks,
            self.init_blocks,
            self.local_blocks,
            score_type=self.score_type,
            disable_index_value=disable_value,
        )

        # ---- reshape output to match forward_extend's verify return shape ----
        # forward_extend returns (idx_o, o) each reshaped to
        # [original_num_tokens, -1].contiguous() where original_num_tokens =
        # q.shape[0] = bs*D. The decode kernel already returns o as
        # [bs*D, num_heads, head_dim] and idx_o as [bs*D, num_idx_heads, idx_dim]
        # (or None when disable_value), so the reshape is identical.
        original_num_tokens = q.shape[0]
        return (
            None if idx_o is None
            else idx_o.reshape(original_num_tokens, -1).contiguous(),
            o.reshape(original_num_tokens, -1).contiguous(),
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
        # EAGLE3 Strategy A: route target_verify through the graph-safe DECODE
        # kernel when speculative_attention_mode == "decode". This avoids the
        # prefill sparse kernel's VMFault under cuda graph (its score/topk
        # tensors are sized by a dynamic max_seqlen_k that differs between
        # capture and replay). The decode kernel is graph-safe (score sized by
        # the static req_to_token.shape[1]). Falls through to the existing
        # prefill path for mode == "prefill" (kept as a fallback) or for any
        # non-verify extend.
        if forward_batch.forward_mode.is_target_verify():
            if self._verify_decode_mode() == "decode":
                return self._forward_verify_via_decode(
                    q, k, v, layer, forward_batch, save_kv_cache,
                    idx_q=idx_q, idx_k=idx_k, idx_v=idx_v,
                )
            # else: mode == "prefill" -> fall through to the existing prefill
            # path below (the crashing-under-graph one, kept as fallback).

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

        # ---- 探针 B: capture 时记录 out_cache_loc 静态属性 ----
        # NOTE: forward_extend 在 replay 时不执行 Python(被 graph 跳过), 此探针仅 capture 时生效.
        # replay 时的 out_cache_loc 真实值由探针 A(通过 self._replay_forward_batch)检查.
        if (_VERIFY_PROBE and _VERIFY_PROBE_RANK == 0
                and forward_batch.forward_mode.is_target_verify()
                and torch.cuda.is_current_stream_capturing()):
            try:
                _ocl = forward_batch.out_cache_loc
                _probe_log(f"[B CAPTURE] layer={layer.layer_id} out_cache_loc "
                           f"shape={tuple(_ocl.shape)} dtype={_ocl.dtype} ptr={_ocl.data_ptr()}")
            except Exception as _e:
                _probe_log(f"[B CAPTURE] probe-err: {_e}")

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

        # ---- 探针 E: EAGER prefill 路径 (非 capture) 越界检查 ----
        # 崩溃定位: 并发 prefill (cuda graph: False) 时 _flash_attn_fwd_with_block_score_kernel
        # 写 score[h, seq_start+q_off, seqblock_k] 越界. score 形状 = (num_heads, total_q, cdiv(max_seqlen_k, block_size_k)).
        # 三类越界可能: (1) seq_lens 某值 > max_seqlen_k → score 第3维越界
        #              (2) cu_seqlens 累计 > total_q → score 第2维越界 (seq_start 越界)
        #              (3) req_pool_indices / out_cache_loc 越界 → k_cache 读写越界
        # 只在 eager (非 capture) 触发: capture 路径由探针 B/C 覆盖, replay 由探针 A 覆盖.
        # 越界量是 batch 级 (seq_lens/cu_seqlens/max_seqlen_k 所有 layer 共用), 故:
        #   - 第一个 sparse layer (min sparse_layer_ids) 打详细行 + OOB
        #   - 后续 layer 只在检出 OOB 时打 (崩溃前最后一行 OOB 即定位点)
        if (_VERIFY_PROBE and _VERIFY_PROBE_RANK == 0
                and not torch.cuda.is_current_stream_capturing()):
            try:
                _first_sparse = min(self.sparse_layer_ids) if self.sparse_layer_ids else layer.layer_id
                _is_first = (layer.layer_id == _first_sparse)
                _is_verify = forward_batch.forward_mode.is_target_verify()
                _total_q = q.shape[0]
                _num_heads = q.shape[1]
                _max_sblk_k = (self._max_seqlen_k + self.block_size_k - 1) // self.block_size_k
                # seq_lens / prefix_lens 是 GPU tensor, eager 下可同步
                _sl = seq_lens.detach()
                _pl = prefix_lens.detach()
                _cu = cu_seqlens.detach()
                _sl_max = int(_sl.max().item())
                _sl_min = int(_sl.min().item())
                _cu_last = int(_cu[-1].item())
                _rp = forward_batch.req_pool_indices.detach()
                _rp_max = int(_rp.max().item())
                _rp_min = int(_rp.min().item())
                _max_slots = self.kv_pool.size
                _r2t_rows = self.req_to_token.shape[0]
                _r2t_cols = self.req_to_token.shape[1]
                _ocl = forward_batch.out_cache_loc.detach()
                _ocl_min = int(_ocl.min().item())
                _ocl_max = int(_ocl.max().item())
                _ocl_oob = int(((_ocl >= _max_slots) | (_ocl < 0)).sum().item())

                _bad = []
                # (1) score 第3维: 任一 seq_len > max_seqlen_k → block_num 越界写 score
                if _sl_max > self._max_seqlen_k:
                    _bad.append(f"score-dim3: seq_lens.max={_sl_max} > max_seqlen_k={self._max_seqlen_k} "
                                f"(score第3维={_max_sblk_k} 不够, kernel写越界!)")
                # (2) score 第2维: cu_seqlens 累计 > total_q → seq_start 越界
                if _cu_last > _total_q:
                    _bad.append(f"score-dim2: cu_seqlens[-1]={_cu_last} > total_q={_total_q} "
                                f"(score第2维不够, seq_start越界!)")
                # (3a) req_pool_indices 行越界 → req_to_token[req, ...] 越界
                if _rp_max >= _r2t_rows or _rp_min < 0:
                    _bad.append(f"req_pool_indices: [{_rp_min},{_rp_max}] 越界 req_to_token rows={_r2t_rows}")
                # (3b) seq_lens 列越界: req_to_token[req, seq_len] 需 < r2t_cols
                if _sl_max > _r2t_cols:
                    _bad.append(f"req_to_token列: seq_lens.max={_sl_max} > r2t_cols={_r2t_cols}")
                # (3c) out_cache_loc 越界 → set_kv_buffer 写 k_cache 越界
                if _ocl_oob:
                    _bad.append(f"out_cache_loc: [{_ocl_min},{_ocl_max}] oob_cnt={_ocl_oob} >= max_slots={_max_slots}")
                # (3d) 【最关键】 req_to_token[req, 0:seq_len] 里的 slot 值是否 >= max_slots
                # 崩溃 kernel _flash_attn_fwd_with_block_score_kernel 读 req_to_token[sid, pos] 得 slots,
                # 再 k_cache[slots, ...]. 若 slots >= max_slots (未初始化/过期/被回收) → k_cache 越界 VMFault.
                # kernel 的 (slots+max_slots)%max_slots 只防负数, 不防 >= max_slots.
                # 检查每个 req 的 [0, seq_len) 范围 (kernel 实际读取范围, pos_mask=pos<seq_len):
                _r2t = self.req_to_token
                _sl_cpu = _sl.detach().to(torch.int64).cpu()
                _rp_cpu = _rp.detach().to(torch.int64).cpu()
                for _i in range(int(_sl_cpu.shape[0])):
                    _slen = int(_sl_cpu[_i].item())
                    _ridx = int(_rp_cpu[_i].item())
                    if _slen <= 0 or _ridx < 0 or _ridx >= _r2t_rows:
                        continue
                    _chk_cols = min(_slen, _r2t_cols)
                    _row_vals = _r2t[_ridx, :_chk_cols].detach()
                    _vmx = int(_row_vals.max().item())
                    _vmn = int(_row_vals.min().item())
                    if _vmx >= _max_slots or _vmn < -1:
                        _oob_cnt = int(((_row_vals >= _max_slots) | (_row_vals < -1)).sum().item())
                        _bad.append(f"req_to_token[req={_i},pool={_ridx},:seq_len={_slen}]: "
                                    f"slot [{_vmn},{_vmx}] oob_cnt={_oob_cnt} >= max_slots={_max_slots} "
                                    f"(k_cache读越界! 这就是VMFault源)")

                _tag = "VERIFY" if _is_verify else "PREFILL"
                # 第一个 sparse layer: 总是打详细行 (日志干净, 1次prefill=1行)
                # 后续 layer: 仅 OOB 时打 (崩溃前定位点)
                if _is_first or _bad:
                    _probe_log(f"[E EAGER {_tag}] layer={layer.layer_id} bs={_sl.shape[0]} total_q={_total_q} "
                               f"num_heads={_num_heads} max_seqlen_q={self._max_seqlen_q} "
                               f"max_seqlen_k={self._max_seqlen_k} max_seqblock_k={_max_sblk_k} "
                               f"block_size_k={self.block_size_k}")
                    _probe_log(f"  seq_lens: min={_sl_min} max={_sl_max} (vs max_seqlen_k={self._max_seqlen_k})")
                    _probe_log(f"  prefix_lens: min={int(_pl.min().item())} max={int(_pl.max().item())}")
                    _probe_log(f"  cu_seqlens[-1]={_cu_last} (vs total_q={_total_q})")
                    _probe_log(f"  req_pool_indices: [{_rp_min},{_rp_max}] (vs r2t_rows={_r2t_rows})")
                    _probe_log(f"  out_cache_loc: [{_ocl_min},{_ocl_max}] oob={_ocl_oob} (vs max_slots={_max_slots})")
                    _probe_log(f"  r2t_cols={_r2t_cols} kv_pool.size={_max_slots}")
                if _bad:
                    _probe_log(f"[E EAGER {_tag}] !!! OOB DETECTED ({len(_bad)}) layer={layer.layer_id} !!!")
                    for _b in _bad:
                        _probe_log(f"  {_b}")
                elif _is_first:
                    _probe_log(f"[E EAGER {_tag}] OK layer={layer.layer_id}")
            except Exception as _e:
                _probe_log(f"[E EAGER] probe-err layer={layer.layer_id}: {_e}")

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
        # ---- 探针 C: capture 时记录传给 kernel 的张量 shape(用于和 replay 对比)----
        # NOTE: forward_extend 在 replay 时不执行 Python, 此探针仅 capture 时生效.
        # capture 时 seq_lens=[1,1,..](dummy), 这里记录 shape/ptr; replay 真实值由探针 A 检查.
        if (_VERIFY_PROBE and _VERIFY_PROBE_RANK == 0
                and forward_batch.forward_mode.is_target_verify()
                and torch.cuda.is_current_stream_capturing()):
            try:
                _probe_log(f"[C CAPTURE] layer={layer.layer_id} q.shape={tuple(q.shape)} "
                           f"cu_seqlens.shape={tuple(cu_seqlens.shape)} "
                           f"seq_lens.shape={tuple(seq_lens.shape)} "
                           f"req_pool_indices.shape={tuple(forward_batch.req_pool_indices.shape)} "
                           f"max_seqlen_q={self._max_seqlen_q} max_seqlen_k={self._max_seqlen_k}")
            except Exception as _e:
                _probe_log(f"[C CAPTURE] probe-err: {_e}")

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
        # 探针用: cuda_graph_runner 把 _replay_forward_batch 设在 hybrid 上,
        # 这里转发给 sparse 子 backend, 让探针 A 能拿 out_cache_loc
        if hasattr(self, "_replay_forward_batch"):
            self.sparse._replay_forward_batch = self._replay_forward_batch
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
