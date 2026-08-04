#!/usr/bin/env python3
"""复现测试: verify kernel 在 cuda graph capture/replay 下的崩溃.

端到端崩溃现象:
  - 单请求 (bs=1) verify graph replay: OK
  - 并发 16 (bs=16) verify graph replay: VMFault (崩溃 kernel =
    _flash_attn_fwd_with_block_score_kernel_verify / _topk_index_kernel_verify /
    _gqa_share_sparse_fwd_kernel_verify)

核心假设: sglang target_verify 进 cuda graph, capture 时 dummy seq_lens=1 (由
  get_cuda_graph_seq_len_fill_value 返回 1), replay 时真实 seq_lens≈2000.
  离线测试 (eager, 每次真实 shape) 全过, 但 graph replay 崩 → 问题在
  capture/replay shape 错配.

本测试精确复现 graph 流程:
  1. 用 capture 时的 dummy 输入 (seq_lens=1, prefix_lens=1, extend=D) 调一次
     minimax_sparse_verify_prefill, 在 cuda graph capture 模式下.
  2. 用 replay 时的真实输入 (seq_lens=2000, prefix_lens=2000, extend=D) 调同一
     graph replay.
  3. 看是否崩溃 / 输出是否正确.

关键: capture 和 replay 用 *同一个 graph*, 但 seq_lens/prefix_lens/cu_seqlens 等
graph buffer 的 *值* 变化 (shape 不变). 这正是 sglang 的机制.

注意: 完整复现 sglang 的 graph buffer 机制很复杂, 这里用简化版: 直接用
torch.cuda.graph 捕获 verify kernel 调用, capture 用 dummy, replay 改 buffer 值.

运行: python /workspace/verify/test_verify_graph_capture_replay.py
"""
import os
import sys
import torch

sys.path.insert(0, "/usr/local/lib/python3.10/dist-packages")

from sglang.srt.layers.attention.minimax_sparse_ops.verify.verify_sparse import (
    minimax_sparse_verify_prefill,
)

torch.manual_seed(0)
DEVICE = "cuda:0"

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
D = 4  # draft_token_num
DTYPE = torch.bfloat16
CONTEXT_LEN = 204800
MAX_BS = 16  # capture 上界
MAX_SEQBLOCK_K_UPPER = (CONTEXT_LEN + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K  # 1600


def build_paged_cache(bs, max_prefix):
    """建 paged KV cache, 每个请求 max_prefix+D 个 slot 连续."""
    max_kv_len = max_prefix + D
    max_slots = bs * max_kv_len + 256
    max_reqs = bs + 8

    k_cache = torch.zeros(max_slots, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    v_cache = torch.zeros(max_slots, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    idx_k_cache = torch.zeros(max_slots, NUM_IDX_HEADS, IDX_HEAD_DIM, dtype=DTYPE, device=DEVICE)
    idx_v_cache = torch.zeros(max_slots, NUM_IDX_HEADS, IDX_HEAD_DIM, dtype=DTYPE, device=DEVICE)

    req_to_token = torch.full((max_reqs, max_kv_len), -1, dtype=torch.int32, device=DEVICE)
    for r in range(bs):
        start = r * max_kv_len
        for p in range(max_kv_len):
            req_to_token[r, p] = start + p
            k_cache[start + p] = torch.randn(NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE) * 0.3
            v_cache[start + p] = torch.randn(NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE) * 0.3
            idx_k_cache[start + p] = torch.randn(NUM_IDX_HEADS, IDX_HEAD_DIM, dtype=DTYPE, device=DEVICE) * 0.3
            idx_v_cache[start + p] = torch.randn(NUM_IDX_HEADS, IDX_HEAD_DIM, dtype=DTYPE, device=DEVICE) * 0.3
    return k_cache, v_cache, idx_k_cache, idx_v_cache, req_to_token, max_slots


def make_buffers(bs):
    """建 graph buffer: q, idx_q, cu_seqlens, seq_lens, prefix_lens, slot_ids.
    shape 按 capture bs (=MAX_BS) 固定, replay 时只改值."""
    total_q = bs * D
    q = torch.randn(total_q, NUM_Q_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE) * 0.3
    idx_q = torch.randn(total_q, NUM_IDX_HEADS, IDX_HEAD_DIM, dtype=DTYPE, device=DEVICE) * 0.3
    # cu_seqlens: [bs+1], capture 时 dummy (每请求 D), replay 时不变 (每请求还是 D)
    cu_seqlens = torch.zeros(bs + 1, dtype=torch.int32, device=DEVICE)
    for r in range(bs):
        cu_seqlens[r + 1] = cu_seqlens[r] + D
    # seq_lens / prefix_lens: [bs], capture dummy=1, replay 真实. 用 buffer 模拟.
    seq_lens = torch.full((bs,), 1, dtype=torch.int32, device=DEVICE)  # capture: 1
    prefix_lens = torch.full((bs,), 1, dtype=torch.int32, device=DEVICE)  # capture: 1
    slot_ids = torch.arange(bs, dtype=torch.int32, device=DEVICE)
    return q, idx_q, cu_seqlens, seq_lens, prefix_lens, slot_ids


def test_eager_vs_graph():
    """对比: eager (真实 shape) vs graph (capture dummy + replay 真实)."""
    print("=" * 70)
    print("复现: verify kernel graph capture/replay (bs=16)")
    print("=" * 70)

    bs = MAX_BS
    max_prefix = 2000  # replay 真实 prefix
    k_cache, v_cache, idx_k_cache, idx_v_cache, req_to_token, max_slots = build_paged_cache(bs, max_prefix)
    q, idx_q, cu_seqlens, seq_lens, prefix_lens, slot_ids = make_buffers(bs)

    # ---- 1. eager 基线: 真实 seq_lens=2000, 不进 graph ----
    print("\n[1] eager 基线 (真实 seq_lens=2000, 无 graph):")
    seq_lens_real = torch.full((bs,), max_prefix + D, dtype=torch.int32, device=DEVICE)
    prefix_lens_real = torch.full((bs,), max_prefix, dtype=torch.int32, device=DEVICE)
    try:
        torch.cuda.synchronize()
        idx_o_eager, o_eager = minimax_sparse_verify_prefill(
            q=q, k_cache=k_cache, v_cache=v_cache, sink=None,
            idx_q=idx_q, idx_k_cache=idx_k_cache, idx_v_cache=idx_v_cache, idx_sink=None,
            req_to_token=req_to_token, slot_ids=slot_ids,
            cu_seqlens=cu_seqlens, seq_lens=seq_lens_real, prefix_lens=prefix_lens_real,
            max_seqlen_q=D, max_seqlen_k=max_prefix + D,
            max_seqblock_k_upper=MAX_SEQBLOCK_K_UPPER,
            block_size_q=BLOCK_SIZE_Q, block_size_k=BLOCK_SIZE_K,
            topk=TOPK, init_blocks=INIT_BLOCKS, local_blocks=LOCAL_BLOCKS,
            score_type="max", disable_index_value=False,
        )
        torch.cuda.synchronize()
        print(f"  eager OK: o.shape={tuple(o_eager.shape)}, has_nan={torch.isnan(o_eager.float()).any().item()}")
    except Exception as e:
        print(f"  eager FAIL: {type(e).__name__}: {e}")
        return False

    # ---- 2. graph: capture (dummy seq_lens=1) + replay (真实 seq_lens=2000) ----
    print("\n[2] graph capture (dummy seq_lens=1) + replay (真实 seq_lens=2000):")
    # capture: seq_lens=1 (dummy, 对应 sglang seq_len_fill_value=1)
    seq_lens.fill_(1)
    prefix_lens.fill_(1)  # capture: prefix=1, seq_len=prefix+extend=1+4=5
    try:
        torch.cuda.synchronize()
        # warmup (triton autotune)
        for _ in range(3):
            _ = minimax_sparse_verify_prefill(
                q=q, k_cache=k_cache, v_cache=v_cache, sink=None,
                idx_q=idx_q, idx_k_cache=idx_k_cache, idx_v_cache=idx_v_cache, idx_sink=None,
                req_to_token=req_to_token, slot_ids=slot_ids,
                cu_seqlens=cu_seqlens, seq_lens=seq_lens, prefix_lens=prefix_lens,
                max_seqlen_q=D, max_seqlen_k=5,  # capture max_seqlen_k 小
                max_seqblock_k_upper=MAX_SEQBLOCK_K_UPPER,
                block_size_q=BLOCK_SIZE_Q, block_size_k=BLOCK_SIZE_K,
                topk=TOPK, init_blocks=INIT_BLOCKS, local_blocks=LOCAL_BLOCKS,
                score_type="max", disable_index_value=False,
            )
        torch.cuda.synchronize()

        # capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            idx_o_g, o_g = minimax_sparse_verify_prefill(
                q=q, k_cache=k_cache, v_cache=v_cache, sink=None,
                idx_q=idx_q, idx_k_cache=idx_k_cache, idx_v_cache=idx_v_cache, idx_sink=None,
                req_to_token=req_to_token, slot_ids=slot_ids,
                cu_seqlens=cu_seqlens, seq_lens=seq_lens, prefix_lens=prefix_lens,
                max_seqlen_q=D, max_seqlen_k=5,
                max_seqblock_k_upper=MAX_SEQBLOCK_K_UPPER,
                block_size_q=BLOCK_SIZE_Q, block_size_k=BLOCK_SIZE_K,
                topk=TOPK, init_blocks=INIT_BLOCKS, local_blocks=LOCAL_BLOCKS,
                score_type="max", disable_index_value=False,
            )
        torch.cuda.synchronize()
        print(f"  capture OK: o_g.shape={tuple(o_g.shape)}")

        # replay: 改 seq_lens/prefix_lens 为真实值 (shape 不变, 只改值)
        seq_lens.copy_(seq_lens_real)
        prefix_lens.copy_(prefix_lens_real)
        print(f"  replay: seq_lens[0]={seq_lens[0].item()} prefix_lens[0]={prefix_lens[0].item()}")
        g.replay()
        torch.cuda.synchronize()
        print(f"  replay OK: o_g has_nan={torch.isnan(o_g.float()).any().item()}, has_inf={torch.isinf(o_g.float()).any().item()}")

        # 对比 eager vs graph replay (应该一致, 因为计算相同)
        diff = (o_eager.float() - o_g.float()).abs()
        print(f"  eager vs graph diff: max={diff.max().item():.6f} mean={diff.mean().item():.6f}")
        return True
    except Exception as e:
        print(f"  graph FAIL: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        return False


def main():
    if not torch.cuda.is_available():
        print("SKIP: 需要 CUDA/HIP")
        return
    ok = test_eager_vs_graph()
    print("\n" + "=" * 70)
    print(f"结果: {'PASS (graph capture/replay 不崩)' if ok else 'FAIL (复现崩溃)'}")
    print("=" * 70)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
