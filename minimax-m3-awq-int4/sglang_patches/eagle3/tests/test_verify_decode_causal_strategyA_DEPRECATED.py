#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Causal-correctness unit test for EAGLE3 Strategy A (verify -> decode expand).

PROVES: routing target_verify (q_len = draft_token_num D per request) through
the graph-safe DECODE kernel, by EXPANDING q to look like bs*D independent
single-token requests with per-(request,draft-token) causal seq_lens
(P_r+1 .. P_r+D), produces the SAME attention output as a full causal
reference. This is the same pattern sglang's NSA backend uses
(seqlens_expand_triton + repeat_interleave page_table) and the same pattern
vllm uses (UNIFORM_BATCH verify->decode).

The decode kernel does block-sparse attention (selects topk blocks via an
indexer head). To make sparse == full causal attention (so we can compare
against a simple numpy reference), we set:
  - block_size_k large enough that each request's KV fits in ONE block, and
  - topk_blocks >= 1 (selects the single block), init_blocks/local_blocks
    large enough that the block is always selected.
Then the kernel degenerates to full causal attention over [0, seq_len) per
expanded row, which is exactly the prefill causal pattern for the D draft
tokens.

This test does NOT need the M3 model loaded. It calls the low-level decode
ops directly (flash_decode_with_topk_idx + flash_decode_with_gqa_share_sparse)
with manually-constructed small caches, mirroring what
MiniMaxSparseAttnBackend._forward_verify_via_decode does at runtime.

Run: python /workspace/verify/test_verify_decode_causal.py
"""

from __future__ import annotations

import os
import sys

import torch

# Ensure the installed sglang package is importable.
sys.path.insert(0, "/usr/local/lib/python3.10/dist-packages")

from sglang.srt.layers.attention.minimax_sparse_ops.common.utils import (
    seqlens_expand_triton,
)
from sglang.srt.layers.attention.minimax_sparse_ops.decode.flash_with_topk_idx import (
    flash_decode_with_topk_idx,
)
from sglang.srt.layers.attention.minimax_sparse_ops.decode.topk_sparse import (
    flash_decode_with_gqa_share_sparse,
)


def _to_np(t: torch.Tensor) -> "torch.Tensor":
    return t.detach().to(torch.float32).cpu()


def numpy_full_causal_attn(
    q_all: torch.Tensor,        # [bs, D, num_heads, head_dim]
    k_cache: torch.Tensor,      # [max_slots, num_kv_heads, head_dim]
    v_cache: torch.Tensor,      # [max_slots, num_kv_heads, head_dim]
    req_to_token: torch.Tensor, # [max_reqs, max_kv_len]
    req_pool_indices: torch.Tensor,  # [bs]
    prefix_lens: list[int],
    D: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    sm_scale: float,
) -> torch.Tensor:
    """Reference: for each request r and draft token j, full causal attention
    over KV positions [0, P_r + j + 1) using the SAME q/k/v the kernel sees.

    Returns o_ref [bs, D, num_heads, head_dim] in float32.
    """
    bs = len(prefix_lens)
    gqa = num_heads // num_kv_heads
    o_ref = torch.zeros(bs, D, num_heads, head_dim, dtype=torch.float32)
    k_np = k_cache.to(torch.float32).cpu()
    v_np = v_cache.to(torch.float32).cpu()
    r2t = req_to_token.to(torch.int64).cpu()
    rp = req_pool_indices.to(torch.int64).cpu().tolist()
    q_np = q_all.to(torch.float32).cpu()

    for r in range(bs):
        P = prefix_lens[r]
        sid = rp[r]
        # KV positions [0, P+D) (prefix + D draft tokens, all written to cache)
        slots = r2t[sid, : P + D].tolist()
        K = k_np[slots]  # [P+D, num_kv_heads, head_dim]
        V = v_np[slots]  # [P+D, num_kv_heads, head_dim]
        for j in range(D):
            kv_len = P + j + 1  # causal: draft token j sees prefix + 0..j
            q_j = q_np[r, j]    # [num_heads, head_dim]
            # GQA: each kv_head serves gqa q-heads
            o_j = torch.zeros(num_heads, head_dim, dtype=torch.float32)
            for h in range(num_heads):
                kh = h % num_kv_heads
                qh = q_j[h]                    # [head_dim]
                kk = K[:kv_len, kh]            # [kv_len, head_dim]
                vv = V[:kv_len, kh]            # [kv_len, head_dim]
                scores = qh @ kk.T * sm_scale  # [kv_len]
                # numeric-stable softmax
                scores = scores - scores.max()
                p = torch.exp(scores)
                p = p / p.sum()
                o_j[h] = p @ vv
            o_ref[r, j] = o_j
    return o_ref


def run_test(device: str = "cuda"):
    torch.manual_seed(0)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("WARNING: CUDA not available, falling back to CPU (triton may fail)")

    # ---- scenario ----
    bs = 2
    D = 4
    num_heads = 2
    num_kv_heads = 1
    head_dim = 8
    idx_head_dim = 8
    num_idx_heads = 1
    block_size_k = 128          # large: each request's KV fits in ONE block
    topk_blocks = 1             # select the single block (sparse == full causal)
    init_blocks = 1             # force-select the first block
    local_blocks = 1            # force-select the local block
    prefix_lens = [16, 23]
    max_kv_len = max(prefix_lens) + D + 8  # req_to_token cols (static, graph-safe)
    max_slots = 512              # k_cache rows
    max_reqs = 8
    score_type = "max"
    sm_scale = head_dim ** -0.5
    idx_sm_scale = idx_head_dim ** -0.5
    dtype = torch.bfloat16

    # ---- build caches & req_to_token ----
    # Assign contiguous slots per request: req r uses slots [r*max_kv_len, ...].
    req_to_token = torch.full(
        (max_reqs, max_kv_len), -1, dtype=torch.int32, device=device
    )
    for r in range(bs):
        base = r * (max(prefix_lens) + D + 4)
        for pos in range(prefix_lens[r] + D):
            req_to_token[r, pos] = base + pos

    k_cache = torch.randn(max_slots, num_kv_heads, head_dim, dtype=dtype, device=device)
    v_cache = torch.randn(max_slots, num_kv_heads, head_dim, dtype=dtype, device=device)
    idx_k_cache = torch.randn(
        max_slots, num_idx_heads, idx_head_dim, dtype=dtype, device=device
    )
    # idx_v_cache (indexer value) - used when disable_index_value=False
    idx_v_cache = torch.randn(
        max_slots, num_idx_heads, idx_head_dim, dtype=dtype, device=device
    )

    # ---- q / idx_q for the D draft tokens per request ----
    # Layout: [bs*D, ...] interleaved-by-request (matches assign_extend_cache_locs).
    q_all = torch.randn(bs, D, num_heads, head_dim, dtype=dtype, device=device)
    idx_q_all = torch.randn(
        bs, D, num_idx_heads, idx_head_dim, dtype=dtype, device=device
    )
    q_flat = q_all.reshape(bs * D, num_heads, head_dim).contiguous()
    idx_q_flat = idx_q_all.reshape(
        bs * D, num_idx_heads, idx_head_dim
    ).contiguous()

    # ---- EXPANDED tensors (the Strategy A transform) ----
    # kv_len per request = prefix + D (draft tokens written to cache first)
    prefix_lens_t = torch.tensor(prefix_lens, dtype=torch.int32, device=device)
    seq_lens_kv = prefix_lens_t + D  # [bs]
    seq_lens_expanded = seqlens_expand_triton(seq_lens_kv, D)  # [bs*D]
    req_pool_indices = torch.arange(bs, dtype=torch.int32, device=device)
    req_pool_indices_expanded = torch.repeat_interleave(req_pool_indices, D).to(
        torch.int32
    )

    print(f"[setup] bs={bs} D={D} bs*D={bs * D} prefixes={prefix_lens}")
    print(f"[setup] seq_lens_kv={seq_lens_kv.tolist()}")
    print(f"[setup] seq_lens_expanded={seq_lens_expanded.tolist()}")
    # sanity: seq_lens_expanded[r*D+j] = prefix_r + j + 1
    for r in range(bs):
        for j in range(D):
            expect = prefix_lens[r] + j + 1
            got = int(seq_lens_expanded[r * D + j].item())
            assert got == expect, (
                f"seqlens_expand r={r} j={j}: got {got} != expect {expect}"
            )
    print("[sanity] seqlens_expand_triton causal values OK")

    # ---- DECODE-EXPAND path (Strategy A): low-level decode ops ----
    # Step 1: indexer head -> idx_o + topk_idx (using expanded batch = bs*D)
    idx_o, topk_idx = flash_decode_with_topk_idx(
        q=idx_q_flat,
        sink=None,
        k_cache=idx_k_cache,
        v_cache=idx_v_cache,
        req_to_token=req_to_token,
        seq_lens=seq_lens_expanded,
        max_seqlen=int(seq_lens_kv.max().item()),
        slot_ids=req_pool_indices_expanded,
        block_size=block_size_k,
        topk=topk_blocks,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        sm_scale=idx_sm_scale,
        score_type=score_type,
        disable_index_value=False,
    )
    # Step 2: main head sparse attn using topk_idx
    o_decode = flash_decode_with_gqa_share_sparse(
        q=q_flat,
        sink=None,
        k_cache=k_cache,
        v_cache=v_cache,
        req_to_token=req_to_token,
        seq_lens=seq_lens_expanded,
        slot_ids=req_pool_indices_expanded,
        block_size=block_size_k,
        topk_idx=topk_idx,
        sm_scale=sm_scale,
    )
    # o_decode: [bs*D, num_heads, head_dim] -> [bs, D, num_heads, head_dim]
    o_decode = o_decode.reshape(bs, D, num_heads, head_dim)

    # ---- numpy full-causal reference ----
    o_ref = numpy_full_causal_attn(
        q_all, k_cache, v_cache, req_to_token, req_pool_indices,
        prefix_lens, D, num_heads, num_kv_heads, head_dim, sm_scale,
    )

    # ---- compare ----
    o_decode_np = _to_np(o_decode)
    o_ref_np = _to_np(o_ref)
    max_abs_diff = (o_decode_np - o_ref_np).abs().max().item()
    max_ref = o_ref_np.abs().max().item()
    atol, rtol = 1e-2, 1e-2
    ok = torch.allclose(o_decode_np, o_ref_np, atol=atol, rtol=rtol)

    print(f"[compare] max_abs_diff={max_abs_diff:.6f}  max_ref={max_ref:.4f}")
    print(f"[compare] o_decode[0,0,:3]={o_decode_np[0,0,:3].tolist()}")
    print(f"[compare] o_ref[0,0,:3]   ={o_ref_np[0,0,:3].tolist()}")
    print(f"[compare] o_decode[1,3,:3]={o_decode_np[1,3,:3].tolist()}")
    print(f"[compare] o_ref[1,3,:3]   ={o_ref_np[1,3,:3].tolist()}")

    # Per-(request,draft-token) causal check: each draft token's output must
    # match the reference for THAT token's causal KV window.
    for r in range(bs):
        for j in range(D):
            d = (o_decode_np[r, j] - o_ref_np[r, j]).abs().max().item()
            if d > atol + rtol * o_ref_np[r, j].abs().max().item():
                print(
                    f"[FAIL] r={r} j={j} max_diff={d:.6f} "
                    f"decode={o_decode_np[r,j,:3].tolist()} "
                    f"ref={o_ref_np[r,j,:3].tolist()}"
                )
                ok = False

    if ok:
        print("\nPASS: verify->decode expand matches full causal reference "
              f"(atol={atol}, rtol={rtol})")
        return True
    else:
        print("\nFAIL: verify->decode expand does NOT match full causal reference")
        return False


if __name__ == "__main__":
    try:
        ok = run_test()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nFAIL with exception: {e}")
        ok = False
    sys.exit(0 if ok else 1)
