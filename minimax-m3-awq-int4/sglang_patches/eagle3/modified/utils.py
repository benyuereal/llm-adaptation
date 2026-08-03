# Copyright 2025 XunhaoLai. All rights reserved.

import functools
from collections import deque
from typing import Any, Callable, Tuple

import torch
import triton
import triton.language as tl

_tma_keep_alive_buf = deque(maxlen=200)

try:
    make_tensor_descriptor = tl.make_tensor_descriptor
except Exception:
    make_tensor_descriptor = tl._experimental_make_tensor_descriptor


def robust_allocator(size: int, alignment: int, stream: int = None):
    """Allocator for Triton TMA descriptors.

    We keep reference in deque to prevent GC from collecting the buffer.
    """
    tensor = torch.empty(size, device="cuda", dtype=torch.uint8)
    _tma_keep_alive_buf.append(tensor)
    return tensor


def tensor_cache(maxsize: int = 8):
    """
    Cache function results using identity comparison.
    Zero-overhead cache hit: no hash, no DtoH, just pointer comparison.

    Args:
        maxsize: Maximum number of cached entries. Supports multi-GPU scenarios
                 where different devices have different tensor arguments.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        # LRU-style cache: list of (args, kwargs, result) tuples
        # Most recently used at the end
        _cache: list = []

        def _args_match(args: tuple, cached_args: tuple) -> bool:
            if len(args) != len(cached_args):
                return False
            for i in range(len(args)):
                if args[i] is not cached_args[i]:
                    return False
            return True

        def _kwargs_match(kwargs: dict, cached_kwargs: dict) -> bool:
            if not kwargs and not cached_kwargs:
                return True
            if kwargs.keys() != cached_kwargs.keys():
                return False
            for k, v in kwargs.items():
                if v is not cached_kwargs[k]:
                    return False
            return True

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Search cache (most recent first for better hit rate)
            for i in range(len(_cache) - 1, -1, -1):
                cached_args, cached_kwargs, cached_result = _cache[i]
                if _args_match(args, cached_args) and _kwargs_match(
                    kwargs, cached_kwargs
                ):
                    # Move to end (most recently used)
                    if i != len(_cache) - 1:
                        _cache.append(_cache.pop(i))
                    return cached_result

            # Cache miss
            result = fn(*args, **kwargs)

            # Add to cache
            if len(_cache) >= maxsize:
                _cache.pop(0)  # Remove oldest
            _cache.append((args, kwargs, result))
            return result

        # Expose cache for manual clearing if needed
        wrapper.cache_clear = lambda: _cache.clear()
        wrapper.cache_info = lambda: {"size": len(_cache), "maxsize": maxsize}

        return wrapper

    return decorator


@tensor_cache(maxsize=8)
def get_cu_seqblocks(
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    block_size_q: int,
    block_size_k: int,
) -> Tuple[torch.Tensor, int, int, torch.Tensor, int, int]:
    """Compute cumulative sequence block indices for blocked sparse attention.

    Converts token-level cumulative sequence lengths to block-level indices,
    which are needed for block-sparse attention kernels.

    Note:
        Results are cached (maxsize=8) based on input arguments. Repeated calls
        with the same cu_seqlens, max_seqlen, and block sizes will return cached
        results without recomputation.

    Args:
        cu_seqlens: Cumulative sequence lengths. Shape: [batch_size + 1], dtype: int32.
        max_seqlen: Maximum sequence length in the batch.
        block_size_q: Query block size.
        block_size_k: Key-value block size.

    Returns:
        A tuple of 6 values:
            - cu_seqblocks_q: Cumulative query block indices. Shape: [batch_size + 1]
            - max_seqblock_q: Maximum number of query blocks per sequence.
            - all_seqblock_q: Total number of query blocks across all sequences.
            - cu_seqblocks_k: Cumulative key block indices. Shape: [batch_size + 1]
            - max_seqblock_k: Maximum number of key blocks per sequence.
            - all_seqblock_k: Total number of key blocks across all sequences.
    """
    cu_seqblocks_q = torch.zeros_like(cu_seqlens)
    cu_seqblocks_k = torch.zeros_like(cu_seqlens)
    seq_lens = torch.diff(cu_seqlens)
    seqblocks_q = (seq_lens + block_size_q - 1) // block_size_q
    seqblocks_k = (seq_lens + block_size_k - 1) // block_size_k
    max_seqblock_q = (max_seqlen + block_size_q - 1) // block_size_q
    max_seqblock_k = (max_seqlen + block_size_k - 1) // block_size_k
    cu_seqblocks_q[1:] = seqblocks_q
    cu_seqblocks_k[1:] = seqblocks_k
    cu_seqblocks_q.cumsum_(0)
    cu_seqblocks_k.cumsum_(0)
    # all_seqblock_q/k = total blocks across all sequences. Used as a tensor
    # shape (torch.full) downstream, so it must be a host int. .sum().item()
    # is a host sync and is illegal under CUDA graph capture. Under capture,
    # use the static upper bound batch_size * max_seqblock (over-allocates the
    # score/topk_idx tensors slightly, but kernels index via cu_seqblocks so
    # the extra slots are never accessed). Eager keeps the exact value.
    if torch.cuda.is_current_stream_capturing():
        batch_size = cu_seqblocks_q.shape[0] - 1
        all_seqblock_q = batch_size * int(max_seqblock_q)
        all_seqblock_k = batch_size * int(max_seqblock_k)
    else:
        all_seqblock_q = seqblocks_q.sum().item()
        all_seqblock_k = seqblocks_k.sum().item()
    return (
        cu_seqblocks_q,
        max_seqblock_q,
        all_seqblock_q,
        cu_seqblocks_k,
        max_seqblock_k,
        all_seqblock_k,
    )


# ============================================================================
# EAGLE3 verify -> decode causal-expand (Strategy A)
# ============================================================================
# Ported from the fork NSA reference (sglang/srt/layers/attention/utils.py
# `seqlens_expand_kernel` / `seqlens_expand_triton`). The NSA version handles
# per-request variable extend_seq_lens via an offsets array; here EAGLE3
# target_verify has a UNIFORM draft_token_num D per request, so we simplify to
# a single qo_len argument and a regular `out[r*D + j]` layout.
#
# Semantics (matches NSA exactly): for request r with KV length kv_len_r
# (= prefix_r + D after the draft tokens are written), the D draft tokens
# should see KV lengths kv_len_r - D + 1, kv_len_r - D + 2, ..., kv_len_r
# (causal: draft token j attends to prefix + earlier draft tokens, i.e. the
# first P_r+j+1 KV positions). With kv_len_r = prefix_r + D this gives
# prefix_r+1 .. prefix_r+D, which is exactly the prefill causal pattern.
#
# Graph-safe: the output tensor is sized by bs*qo_len (a cuda-graph constant
# for the verify batch size), and the kernel grid is (bs,) independent of any
# dynamic seq_len. All host values (qo_len) are passed as kernel args /
# constexpr, never read back from the GPU.
# ============================================================================


@triton.jit
def seqlens_expand_kernel(
    seq_lens_ptr,  # [N] int32, KV length per request (= prefix + D for verify)
    output_ptr,  # [N * QO_LEN] int32, expanded seqlens
    N,
    QO_LEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Expand per-request seq_lens into per-(request,draft-token) seq_lens.

    For request r (pid), writes output[r*QO_LEN + j] = seq_lens[r] - QO_LEN + 1 + j
    for j in [0, QO_LEN). This is the causal increment needed to route a
    q_len=D verify batch through the q_len=1 decode kernel.
    """
    pid = tl.program_id(0)
    if pid >= N:
        return
    kv_len = tl.load(seq_lens_ptr + pid)
    start = kv_len - QO_LEN + 1
    offs = tl.arange(0, BLOCK)
    mask = offs < QO_LEN
    values = start + offs
    tl.store(output_ptr + pid * QO_LEN + offs, values, mask=mask)


def seqlens_expand_triton(
    seq_lens: torch.Tensor,
    qo_len: int,
) -> torch.Tensor:
    """Expand seq_lens [bs] into [bs * qo_len] for verify->decode routing.

    Args:
        seq_lens: [bs] int32 tensor on device. For EAGLE3 target_verify this is
            the real KV length per request AFTER the draft tokens are written,
            i.e. prefix + draft_token_num. Pass forward_batch.seq_lens rebuilt
            as prefix + extend (NOT the raw prefix-only seq_lens the scheduler
            leaves for verify).
        qo_len: uniform query length per request (= draft_token_num for verify).

    Returns:
        seq_lens_expanded: [bs * qo_len] int32 on seq_lens.device, where the
        entry for request r's j-th draft token is seq_lens[r] - qo_len + 1 + j.
        Layout is interleaved-by-request: [r0_t0, r0_t1, ..., r0_t{D-1},
        r1_t0, ...] matching the flat [bs*D] token ordering of verify's q /
        out_cache_loc (see assign_extend_cache_locs in eagle_info_v2.py).
    """
    assert seq_lens.dtype == torch.int32, (
        f"seq_lens must be int32, got {seq_lens.dtype}"
    )
    assert qo_len >= 1
    N = seq_lens.numel()
    output = torch.empty(
        N * qo_len, device=seq_lens.device, dtype=torch.int32
    )
    if N == 0:
        return output
    BLOCK = triton.next_power_of_2(qo_len)
    grid = (N,)
    seqlens_expand_kernel[grid](
        seq_lens,
        output,
        N,
        QO_LEN=qo_len,
        BLOCK=BLOCK,
    )
    return output
