#!/usr/bin/env python3
"""端到端 graph 复现: 模拟 sglang 的 cuda graph 机制, 验证修复后 verify 在 bs=16 不崩.

模拟 sglang 的两个关键机制:
  1. 真实 graph 内存池 (torch.cuda.graph_pool_handle) — capture 期间所有分配进私有池,
     replay 地址固定
  2. graph buffer 预分配 + replay 时改值 (不重新分配):
     - buffers.seq_lens = (max_bs,) 预分配, capture 全填 dummy=1
     - replay 时 populate: seq_lens[:raw_bs].copy_(真实值), 其余填 fill_value

本测试复现 sglang forward_extend 的 verify 分支构建逻辑:
  - capture: forward_batch.seq_lens = dummy [1,1,...,1] (max_bs 个)
  - replay:  forward_batch.seq_lens[:bs].copy_(真实 prefix), 其余 padding
  - forward_extend verify 分支用修复后的逻辑构建 cu_seqlens/seq_lens/prefix_lens:
      cu_seqlens = torch.arange(0, (bs+1)*D, D)        # Python int, graph-safe
      prefix_lens = raw_seq_lens (= seq_lens buffer)   # graph buffer view
      seq_lens = raw_seq_lens + D                       # graph buffer + Python int

关键: capture 用 dummy seq_lens=1, replay 改 buffer 值为真实 prefix (~500),
验证 kernel 输出正确 (与 eager 基线一致) 且不崩.

运行: python test_verify_sglang_graph_e2e.py
"""
import os
import sys
import torch

sys.path.insert(0, "/usr/local/lib/python3.10/dist-packages")

from sglang.srt.layers.attention.minimax_sparse_ops.verify.verify_sparse import (
    minimax_sparse_verify_prefill,
)

torch.manual_seed(0)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# M3 真实配置 (TP=8 per-rank)
NUM_Q_HEADS = 8
NUM_KV_HEADS = 1
NUM_IDX_HEADS = 1
HEAD_DIM = 128
IDX_HEAD_DIM = 128
BLOCK_SIZE_K = 128
BLOCK_SIZE_Q = 1
TOPK = 16
INIT_BLOCKS = 0
LOCAL_BLOCKS = 1
D = 4  # num_draft_tokens
DTYPE = torch.bfloat16
CONTEXT_LEN = 204800
MAX_SEQBLOCK_K_UPPER = (CONTEXT_LEN + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K  # 1600
MAX_BS = 16  # cuda-graph-max-bs
SEQ_LEN_FILL_VALUE = 1  # sglang get_cuda_graph_seq_len_fill_value 返回 1


def build_paged_cache(max_bs, max_prefix):
    """建 paged KV cache, 预分配 max_bs 个请求的 slot (sglang kv_pool 是预分配的)."""
    max_kv_len = max_prefix + D
    max_slots = max_bs * max_kv_len + 256
    max_reqs = max_bs + 8

    k_cache = torch.zeros(max_slots, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    v_cache = torch.zeros(max_slots, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    idx_k_cache = torch.zeros(max_slots, NUM_IDX_HEADS, IDX_HEAD_DIM, dtype=DTYPE, device=DEVICE)
    idx_v_cache = torch.zeros(max_slots, NUM_IDX_HEADS, IDX_HEAD_DIM, dtype=DTYPE, device=DEVICE)

    # req_to_token: 预分配, 每个请求连续 slot
    req_to_token = torch.full((max_reqs, max_kv_len), -1, dtype=torch.int32, device=DEVICE)
    for r in range(max_bs):
        start = r * max_kv_len
        for p in range(max_kv_len):
            req_to_token[r, p] = start + p
            k_cache[start + p] = torch.randn(NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE) * 0.3
            v_cache[start + p] = torch.randn(NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE) * 0.3
            idx_k_cache[start + p] = torch.randn(NUM_IDX_HEADS, IDX_HEAD_DIM, dtype=DTYPE, device=DEVICE) * 0.3
            idx_v_cache[start + p] = torch.randn(NUM_IDX_HEADS, IDX_HEAD_DIM, dtype=DTYPE, device=DEVICE) * 0.3
    return k_cache, v_cache, idx_k_cache, idx_v_cache, req_to_token, max_slots


def build_verify_inputs_eager(bs, prefix_len, k_cache, v_cache, idx_k_cache, idx_v_cache, req_to_token):
    """eager 基线: 真实 shape, 不进 graph."""
    total_q = bs * D
    q = torch.randn(total_q, NUM_Q_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE) * 0.3
    idx_q = torch.randn(total_q, NUM_IDX_HEADS, IDX_HEAD_DIM, dtype=DTYPE, device=DEVICE) * 0.3
    cu_seqlens = torch.arange(0, (bs + 1) * D, step=D, dtype=torch.int32, device=DEVICE)
    seq_lens = torch.full((bs,), prefix_len + D, dtype=torch.int32, device=DEVICE)
    prefix_lens = torch.full((bs,), prefix_len, dtype=torch.int32, device=DEVICE)
    req_pool_indices = torch.arange(bs, dtype=torch.int32, device=DEVICE)
    return q, idx_q, cu_seqlens, seq_lens, prefix_lens, req_pool_indices


def sglang_forward_extend_verify_branch(seq_lens_buffer, bs, D_const):
    """复现 sglang forward_extend 的 verify 分支 (修复后逻辑).

    seq_lens_buffer: graph buffer (max_bs,), capture dummy=1, replay populate 真实 prefix.
    bs: capture 时固定的 batch size (Python int).
    返回 (cu_seqlens, seq_lens, prefix_lens) — 都是 graph-safe 构建.
    """
    raw_seq_lens = seq_lens_buffer[:bs].to(torch.int32)  # graph buffer 切片, 同地址
    # 修复后逻辑 (minimax_sparse_backend.py forward_extend verify 分支):
    cu_seqlens = torch.arange(0, (bs + 1) * D_const, step=D_const, dtype=torch.int32, device=raw_seq_lens.device)
    prefix_lens = raw_seq_lens  # = forward_batch.seq_lens (prefix), graph buffer view
    seq_lens = raw_seq_lens + D_const  # prefix + D, graph buffer + Python int
    return cu_seqlens, seq_lens, prefix_lens


def test_sglang_graph_e2e():
    print("=" * 70)
    print("端到端 graph 复现: 模拟 sglang 机制 (graph pool + buffer populate)")
    print(f"  MAX_BS={MAX_BS}, D={D}, CONTEXT_LEN={CONTEXT_LEN}, max_seqblock_k_upper={MAX_SEQBLOCK_K_UPPER}")
    print("=" * 70)

    real_prefix = 500  # replay 真实 prefix
    k_cache, v_cache, idx_k_cache, idx_v_cache, req_to_token, max_slots = build_paged_cache(MAX_BS, real_prefix)

    # ---- sglang graph buffer: 预分配, capture 全填 dummy ----
    # buffers.seq_lens = (max_bs,), capture fill_value=1
    seq_lens_buffer = torch.full((MAX_BS,), SEQ_LEN_FILL_VALUE, dtype=torch.int32, device=DEVICE)
    # buffers.req_pool_indices = (max_bs,), capture dummy
    req_pool_indices_buffer = torch.arange(MAX_BS, dtype=torch.int32, device=DEVICE)
    # q/idx_q 是 model forward 输出, 在 graph 内分配 (这里预分配 capture 用)
    total_q = MAX_BS * D
    q_buffer = torch.randn(total_q, NUM_Q_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE) * 0.3
    idx_q_buffer = torch.randn(total_q, NUM_IDX_HEADS, IDX_HEAD_DIM, dtype=DTYPE, device=DEVICE) * 0.3

    bs = MAX_BS  # 测 bs=16 (崩溃点)

    # ---- 1. eager 基线 (真实值, 不进 graph) ----
    print(f"\n[1] eager 基线 (bs={bs}, prefix={real_prefix}):")
    q_e, idx_q_e, cu_e, sl_e, pl_e, rp_e = build_verify_inputs_eager(
        bs, real_prefix, k_cache, v_cache, idx_k_cache, idx_v_cache, req_to_token)
    try:
        torch.cuda.synchronize()
        _, o_eager = minimax_sparse_verify_prefill(
            q=q_e, k_cache=k_cache, v_cache=v_cache, sink=None,
            idx_q=idx_q_e, idx_k_cache=idx_k_cache, idx_v_cache=idx_v_cache, idx_sink=None,
            req_to_token=req_to_token, slot_ids=rp_e,
            cu_seqlens=cu_e, seq_lens=sl_e, prefix_lens=pl_e,
            max_seqlen_q=D, max_seqlen_k=real_prefix + D,
            max_seqblock_k_upper=MAX_SEQBLOCK_K_UPPER,
            block_size_q=BLOCK_SIZE_Q, block_size_k=BLOCK_SIZE_K,
            topk=TOPK, init_blocks=INIT_BLOCKS, local_blocks=LOCAL_BLOCKS,
            score_type="max", disable_index_value=False,
        )
        torch.cuda.synchronize()
        print(f"  eager OK: o.shape={tuple(o_eager.shape)}")
    except Exception as e:
        print(f"  eager FAIL: {type(e).__name__}: {e}")
        return False

    # ---- 2. graph: capture (dummy seq_lens=1) + replay (populate 真实 prefix) ----
    print(f"\n[2] graph capture (dummy seq_lens={SEQ_LEN_FILL_VALUE}) + replay (populate prefix={real_prefix}):")

    # capture 前: buffer 全填 dummy (sglang capture 时 fill_value=1)
    seq_lens_buffer.fill_(SEQ_LEN_FILL_VALUE)
    req_pool_indices_buffer[:bs] = torch.arange(bs, dtype=torch.int32, device=DEVICE)

    # warmup (triton autotune, 不进 graph)
    for _ in range(3):
        cu_w, sl_w, pl_w = sglang_forward_extend_verify_branch(seq_lens_buffer, bs, D)
        _ = minimax_sparse_verify_prefill(
            q=q_buffer, k_cache=k_cache, v_cache=v_cache, sink=None,
            idx_q=idx_q_buffer, idx_k_cache=idx_k_cache, idx_v_cache=idx_v_cache, idx_sink=None,
            req_to_token=req_to_token, slot_ids=req_pool_indices_buffer[:bs],
            cu_seqlens=cu_w, seq_lens=sl_w, prefix_lens=pl_w,
            max_seqlen_q=D, max_seqlen_k=SEQ_LEN_FILL_VALUE + D,  # capture max_seqlen_k 小
            max_seqblock_k_upper=MAX_SEQBLOCK_K_UPPER,
            block_size_q=BLOCK_SIZE_Q, block_size_k=BLOCK_SIZE_K,
            topk=TOPK, init_blocks=INIT_BLOCKS, local_blocks=LOCAL_BLOCKS,
            score_type="max", disable_index_value=False,
        )
    torch.cuda.synchronize()

    # capture (用真实 graph pool, 同 sglang)
    pool = torch.cuda.graph_pool_handle()
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g, pool=pool):
            # forward_extend verify 分支 (修复后逻辑)
            cu_g, sl_g, pl_g = sglang_forward_extend_verify_branch(seq_lens_buffer, bs, D)
            # kernel 调用
            _, o_g = minimax_sparse_verify_prefill(
                q=q_buffer, k_cache=k_cache, v_cache=v_cache, sink=None,
                idx_q=idx_q_buffer, idx_k_cache=idx_k_cache, idx_v_cache=idx_v_cache, idx_sink=None,
                req_to_token=req_to_token, slot_ids=req_pool_indices_buffer[:bs],
                cu_seqlens=cu_g, seq_lens=sl_g, prefix_lens=pl_g,
                max_seqlen_q=D, max_seqlen_k=SEQ_LEN_FILL_VALUE + D,  # capture 小值
                max_seqblock_k_upper=MAX_SEQBLOCK_K_UPPER,
                block_size_q=BLOCK_SIZE_Q, block_size_k=BLOCK_SIZE_K,
                topk=TOPK, init_blocks=INIT_BLOCKS, local_blocks=LOCAL_BLOCKS,
                score_type="max", disable_index_value=False,
            )
        torch.cuda.synchronize()
        print(f"  capture OK: o_g.shape={tuple(o_g.shape)}, cu_g.shape={tuple(cu_g.shape)}, sl_g.shape={tuple(sl_g.shape)}")
    except Exception as e:
        print(f"  capture FAIL: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        return False

    # ---- replay: 模拟 sglang populate (改 buffer 值, 不重新分配) ----
    print(f"\n[3] replay: populate buffer (sglang 机制: 改值不重新分配)")
    # sglang populate: bs==raw_bs 时不 fill, 直接 copy_ 真实值
    # 这里 raw_bs=bs=MAX_BS, 所以直接 copy_ 真实 prefix
    real_seq_lens = torch.full((bs,), real_prefix, dtype=torch.int32, device=DEVICE)
    seq_lens_buffer[:bs].copy_(real_seq_lens)  # populate: 真实 prefix
    # q/idx_q 是 model forward 输出, replay 时 graph 重放 qkv_proj 写同地址 (这里模拟: 改值)
    q_buffer.copy_(q_e)  # 模拟 replay 时 q 被重新计算
    idx_q_buffer.copy_(idx_q_e)
    print(f"  populate: seq_lens_buffer[:{bs}] = {real_prefix}, q/idx_q 已更新")

    try:
        g.replay()
        torch.cuda.synchronize()
        print(f"  replay OK: 不崩! o_g has_nan={torch.isnan(o_g.float()).any().item()}, has_inf={torch.isinf(o_g.float()).any().item()}")

        # 对比 eager vs graph replay (应该一致)
        diff = (o_eager.float() - o_g.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        print(f"  eager vs graph diff: max={max_diff:.6f} mean={mean_diff:.6f}")
        match = max_diff < 1e-2
        print(f"  输出一致 (max_diff<1e-2)? {match}")
        return match
    except Exception as e:
        print(f"  replay FAIL (VMFault?): {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        return False


def test_padding_case():
    """额外: 测 padding 情况 (raw_bs < capture_bs, sglang 会 fill padding 位)."""
    print("\n" + "=" * 70)
    print(f"[padding] raw_bs=10 < capture_bs={MAX_BS} (sglang padding 到 capture_bs)")
    print("=" * 70)

    real_prefix = 300
    raw_bs = 10
    k_cache, v_cache, idx_k_cache, idx_v_cache, req_to_token, max_slots = build_paged_cache(MAX_BS, real_prefix)

    seq_lens_buffer = torch.full((MAX_BS,), SEQ_LEN_FILL_VALUE, dtype=torch.int32, device=DEVICE)
    req_pool_indices_buffer = torch.arange(MAX_BS, dtype=torch.int32, device=DEVICE)
    total_q = MAX_BS * D
    q_buffer = torch.randn(total_q, NUM_Q_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE) * 0.3
    idx_q_buffer = torch.randn(total_q, NUM_IDX_HEADS, IDX_HEAD_DIM, dtype=DTYPE, device=DEVICE) * 0.3

    bs = MAX_BS  # graph 按 capture_bs=16 固定

    # warmup + capture (同上)
    for _ in range(3):
        cu_w, sl_w, pl_w = sglang_forward_extend_verify_branch(seq_lens_buffer, bs, D)
        _ = minimax_sparse_verify_prefill(
            q=q_buffer, k_cache=k_cache, v_cache=v_cache, sink=None,
            idx_q=idx_q_buffer, idx_k_cache=idx_k_cache, idx_v_cache=idx_v_cache, idx_sink=None,
            req_to_token=req_to_token, slot_ids=req_pool_indices_buffer[:bs],
            cu_seqlens=cu_w, seq_lens=sl_w, prefix_lens=pl_w,
            max_seqlen_q=D, max_seqlen_k=SEQ_LEN_FILL_VALUE + D,
            max_seqblock_k_upper=MAX_SEQBLOCK_K_UPPER,
            block_size_q=BLOCK_SIZE_Q, block_size_k=BLOCK_SIZE_K,
            topk=TOPK, init_blocks=INIT_BLOCKS, local_blocks=LOCAL_BLOCKS,
            score_type="max", disable_index_value=False,
        )
    torch.cuda.synchronize()
    pool = torch.cuda.graph_pool_handle()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, pool=pool):
        cu_g, sl_g, pl_g = sglang_forward_extend_verify_branch(seq_lens_buffer, bs, D)
        _, o_g = minimax_sparse_verify_prefill(
            q=q_buffer, k_cache=k_cache, v_cache=v_cache, sink=None,
            idx_q=idx_q_buffer, idx_k_cache=idx_k_cache, idx_v_cache=idx_v_cache, idx_sink=None,
            req_to_token=req_to_token, slot_ids=req_pool_indices_buffer[:bs],
            cu_seqlens=cu_g, seq_lens=sl_g, prefix_lens=pl_g,
            max_seqlen_q=D, max_seqlen_k=SEQ_LEN_FILL_VALUE + D,
            max_seqblock_k_upper=MAX_SEQBLOCK_K_UPPER,
            block_size_q=BLOCK_SIZE_Q, block_size_k=BLOCK_SIZE_K,
            topk=TOPK, init_blocks=INIT_BLOCKS, local_blocks=LOCAL_BLOCKS,
            score_type="max", disable_index_value=False,
        )
    torch.cuda.synchronize()

    # replay: sglang populate padding 机制
    # bs != raw_bs: 先 fill_(fill_value), 再 copy_ 真实值到 [:raw_bs]
    seq_lens_buffer.fill_(SEQ_LEN_FILL_VALUE)  # padding 位填 dummy
    real_seq_lens = torch.full((raw_bs,), real_prefix, dtype=torch.int32, device=DEVICE)
    seq_lens_buffer[:raw_bs].copy_(real_seq_lens)  # 真实请求
    print(f"  populate: seq_lens[:{raw_bs}]={real_prefix}, [{raw_bs}:{bs}]={SEQ_LEN_FILL_VALUE} (padding)")

    try:
        g.replay()
        torch.cuda.synchronize()
        # 只比前 raw_bs 个请求的输出 (padding 位的输出无意义)
        o_real = o_g[:raw_bs * D]
        has_nan = torch.isnan(o_real.float()).any().item()
        print(f"  replay OK (padding): 前{raw_bs}请求 o has_nan={has_nan}")
        return not has_nan
    except Exception as e:
        print(f"  replay FAIL: {type(e).__name__}: {e}")
        return False


def main():
    if not torch.cuda.is_available():
        print("SKIP: 需要 CUDA/HIP")
        return
    r1 = test_sglang_graph_e2e()
    r2 = test_padding_case()
    print("\n" + "=" * 70)
    print(f"  bs=16 graph e2e:      {'PASS' if r1 else 'FAIL'}")
    print(f"  bs=10 padding graph:  {'PASS' if r2 else 'FAIL'}")
    all_pass = r1 and r2
    print("=" * 70)
    if all_pass:
        print("ALL PASS: 修复后 verify 在 sglang 式 graph (pool + buffer populate) 下不崩, 输出正确")
    else:
        print("FAILED: 仍有问题")
    print("=" * 70)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
