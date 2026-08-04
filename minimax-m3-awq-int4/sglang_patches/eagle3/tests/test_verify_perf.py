#!/usr/bin/env python3
"""性能测试: verify kernel (固定上界) vs 原 prefill kernel (动态尺寸) 耗时对比.

【graph 边界说明】
  sglang 标准: prefill(变长)=eager 不进 graph; decode(固定1token)=进 graph.
  EAGLE3 例外: verify(target_verify)虽走 forward_extend 路径, 但 draft token 数 D
  固定, 故 sglang 专门给它一个 cuda graph bucket (cuda_graph_runner.py:641
  capture_forward_mode=TARGET_VERIFY). 所以 verify *进 graph*.

  原 prefill kernel 的 score 第3维 = cdiv(max_seqlen_k, block_size_k) 是 *运行时
  动态值*. eager prefill 每次按真实 max_seqlen_k 分配, 无错配. 但 verify 进 graph
  后, capture 时 seq_lens=[1,1..](dummy)→max_seqlen_k≈1+D→score第3维=1; replay 时
  真实 seq_lens≈2000→kernel 按真实 seq_len 循环写 score 越界→VMFault. 这就是根因.

  所以本测试的对照组 = 原 prefill kernel 在 *eager* 模式跑同样的 verify 输入
  (即每次按真实 max_seqlen_k 分配, 不会越界). 测的是纯 kernel 计算开销差异:
    verify kernel score 第3维固定 = cdiv(context_len, block_size_k) (大, 如 1600)
    prefill  kernel score 第3维动态 = cdiv(max_seqlen_k, block_size_k) (小, 如 16)
  kernel 内部循环仍只按真实 seq_len, 计算量相同. 差异主要在:
    1. score buffer 分配更大 (1600 vs 16 个 float, 每 (num_heads, total_q) 行)
    2. topk kernel 扫描 score 的范围 (有 causal_mask 限制, 只扫 valid_blocks)

  注意: 真实端到端里原 prefill kernel 在 verify 进 graph 时 *根本不能用* (VMFault),
  所以这个开销对比是 "graph-safe 版 vs eager 版的纯计算代价", 不是 "能否用" 的对比.

期望: verify kernel 开销 < 10% (主要来自大 score 分配 + topk 扫描范围).

测试场景 (verify 典型):
  - bs=16, prefix=2000, D=4 (16并发 verify)
  - bs=4, prefix=2000, D=4
  - bs=1, prefix=2000, D=4 (单请求)
  - bs=16, prefix=500, D=4 (短 prefix)

方法: 各 warmup 5 次, 测 20 次取中位数 (避免 autotune 干扰, triton cache 已 warm).

运行: python /workspace/verify/test_verify_perf.py
"""
import os
import sys
import statistics
import torch

sys.path.insert(0, "/usr/local/lib/python3.10/dist-packages")

from sglang.srt.layers.attention.minimax_sparse_ops.minimax_sparse import (
    minimax_sparse_prefill,
)
from sglang.srt.layers.attention.minimax_sparse_ops.verify.verify_sparse import (
    minimax_sparse_verify_prefill,
)

torch.manual_seed(0)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

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
D = 4
DTYPE = torch.bfloat16
CONTEXT_LEN = 204800


def build_inputs(batch_size, prefix_len):
    total_q = batch_size * D
    max_kv_len = prefix_len + D
    max_slots = batch_size * max_kv_len + 256
    max_reqs = batch_size + 8

    k_cache = torch.zeros(max_slots, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    v_cache = torch.zeros(max_slots, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    idx_k_cache = torch.zeros(max_slots, NUM_IDX_HEADS, IDX_HEAD_DIM, dtype=DTYPE, device=DEVICE)
    idx_v_cache = torch.zeros(max_slots, NUM_IDX_HEADS, IDX_HEAD_DIM, dtype=DTYPE, device=DEVICE)

    req_to_token = torch.full((max_reqs, max_kv_len), -1, dtype=torch.int32, device=DEVICE)
    for r in range(batch_size):
        start = r * max_kv_len
        for p in range(max_kv_len):
            req_to_token[r, p] = start + p
            k_cache[start + p] = torch.randn(NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE) * 0.3
            v_cache[start + p] = torch.randn(NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE) * 0.3
            idx_k_cache[start + p] = torch.randn(NUM_IDX_HEADS, IDX_HEAD_DIM, dtype=DTYPE, device=DEVICE) * 0.3
            idx_v_cache[start + p] = torch.randn(NUM_IDX_HEADS, IDX_HEAD_DIM, dtype=DTYPE, device=DEVICE) * 0.3

    q = torch.randn(total_q, NUM_Q_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE) * 0.3
    idx_q = torch.randn(total_q, NUM_IDX_HEADS, IDX_HEAD_DIM, dtype=DTYPE, device=DEVICE) * 0.3
    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=DEVICE)
    for r in range(batch_size):
        cu_seqlens[r + 1] = cu_seqlens[r] + D
    seq_lens = torch.full((batch_size,), prefix_len + D, dtype=torch.int32, device=DEVICE)
    prefix_lens = torch.full((batch_size,), prefix_len, dtype=torch.int32, device=DEVICE)
    slot_ids = torch.arange(batch_size, dtype=torch.int32, device=DEVICE)

    return dict(
        q=q, idx_q=idx_q, k_cache=k_cache, v_cache=v_cache,
        idx_k_cache=idx_k_cache, idx_v_cache=idx_v_cache,
        req_to_token=req_to_token, slot_ids=slot_ids,
        cu_seqlens=cu_seqlens, seq_lens=seq_lens, prefix_lens=prefix_lens,
    )


def common_kwargs(inp, prefix_len):
    max_seqlen_k = prefix_len + D
    return dict(
        q=inp["q"], k_cache=inp["k_cache"], v_cache=inp["v_cache"], sink=None,
        idx_q=inp["idx_q"], idx_k_cache=inp["idx_k_cache"], idx_v_cache=inp["idx_v_cache"],
        idx_sink=None, req_to_token=inp["req_to_token"], slot_ids=inp["slot_ids"],
        cu_seqlens=inp["cu_seqlens"], seq_lens=inp["seq_lens"], prefix_lens=inp["prefix_lens"],
        max_seqlen_q=D, max_seqlen_k=max_seqlen_k,
        block_size_q=BLOCK_SIZE_Q, block_size_k=BLOCK_SIZE_K,
        topk=TOPK, init_blocks=INIT_BLOCKS, local_blocks=LOCAL_BLOCKS,
        score_type="max", disable_index_value=False,
    )


def bench(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # ms
    return statistics.median(times), min(times), max(times)


def main():
    if not torch.cuda.is_available():
        print("SKIP: 需要 CUDA/HIP")
        return
    print("=" * 80)
    print("性能测试: verify kernel (固定上界) vs 原 prefill kernel (动态尺寸)")
    print(f"  score 第3维: prefill=cdiv(max_seqlen_k,128) 动态; verify=cdiv({CONTEXT_LEN},128)={(CONTEXT_LEN+127)//128} 固定")
    print("=" * 80)
    print(f"{'场景':<28} {'prefill(ms)':>12} {'verify(ms)':>12} {'开销':>8} {'判定':>6}")
    print("-" * 80)

    cases = [
        (1, 2000, "bs=1  prefix=2000"),
        (4, 2000, "bs=4  prefix=2000"),
        (16, 2000, "bs=16 prefix=2000 (典型)"),
        (16, 500, "bs=16 prefix=500 (短)"),
        (16, 8000, "bs=16 prefix=8000 (长)"),
    ]

    all_pass = True
    for bs, pl, desc in cases:
        inp = build_inputs(bs, pl)
        kw = common_kwargs(inp, pl)
        max_seqblock_k_upper = (CONTEXT_LEN + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K

        # prefill
        def fn_prefill():
            minimax_sparse_prefill(**kw)
        p_med, p_min, p_max = bench(fn_prefill)

        # verify (固定上界)
        def fn_verify():
            minimax_sparse_verify_prefill(**kw, max_seqblock_k_upper=max_seqblock_k_upper)
        v_med, v_min, v_max = bench(fn_verify)

        overhead = (v_med - p_med) / p_med * 100 if p_med > 0 else 0
        ok = overhead < 10.0
        if not ok:
            all_pass = False
        print(f"{desc:<28} {p_med:>12.4f} {v_med:>12.4f} {overhead:>7.2f}% {'PASS' if ok else 'FAIL':>6}")
        print(f"{'  (min/max)':<28} {f'{p_min:.4f}/{p_max:.4f}':>12} {f'{v_min:.4f}/{v_max:.4f}':>12}")

    print("-" * 80)
    print("=" * 80)
    if all_pass:
        print("ALL PASS: verify kernel 开销 < 10%, 固定上界改造性能可接受")
    else:
        print("FAILED: verify kernel 开销 >= 10%, 需优化 (可能 score 分配过大)")
    print("=" * 80)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
