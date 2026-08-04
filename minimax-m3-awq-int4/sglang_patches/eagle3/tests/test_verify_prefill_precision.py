#!/usr/bin/env python3
"""精度测试: minimax_sparse_verify_prefill (新kernel, 固定上界) vs
            minimax_sparse_prefill       (原kernel, 动态尺寸)

证明 Strategy B 的固定上界改造不改变计算结果 (o 和 topk_idx 完全一致,
除 bf16 数值噪声外). 覆盖不同 prefix_len / batch / disable_value.

运行: python /workspace/verify/test_verify_prefill_precision.py
"""
import os
import sys
import math
import torch
import triton

# 让 sglang 包可 import
sys.path.insert(0, "/usr/local/lib/python3.10/dist-packages")

from sglang.srt.layers.attention.minimax_sparse_ops.minimax_sparse import (
    minimax_sparse_prefill,
)
from sglang.srt.layers.attention.minimax_sparse_ops.verify.verify_sparse import (
    minimax_sparse_verify_prefill,
)

torch.manual_seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 用 cuda:0 (DCU 上 torch.cuda 即 hip)
if torch.cuda.is_available():
    DEVICE = "cuda:0"


def build_test_inputs(
    batch_size: int,
    draft_token_num: int,
    prefix_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    idx_head_dim: int,
    block_size_k: int,
    disable_value: bool,
):
    """构造 verify 场景的输入: bs 个请求, 每个有 prefix_len 个已存 KV + D 个新 draft.

    真实 verify: q_len = D, seq_len = prefix + D, prefix_len = prefix.
    """
    D = draft_token_num
    total_q = batch_size * D
    gqa_group_size = num_q_heads // num_kv_heads
    # max_slots: 足够容纳所有请求的 KV
    max_kv_len = prefix_len + D
    max_slots = batch_size * max_kv_len + 128  # 留余量
    max_reqs = batch_size + 4

    dtype = torch.bfloat16

    # paged KV cache (main)
    k_cache = torch.zeros(max_slots, num_kv_heads, head_dim, dtype=dtype, device=DEVICE)
    v_cache = torch.zeros(max_slots, num_kv_heads, head_dim, dtype=dtype, device=DEVICE)
    # index cache: num_idx_heads = 1 (M3 实际配置)
    idx_k_cache = torch.zeros(max_slots, 1, idx_head_dim, dtype=dtype, device=DEVICE)
    idx_v_cache = (
        None
        if disable_value
        else torch.zeros(max_slots, 1, idx_head_dim, dtype=dtype, device=DEVICE)
    )

    # req_to_token: [max_reqs, max_kv_len] 映射 req -> slot 列表.
    # 每个请求分配连续 prefix+D 个 slot.
    req_to_token = torch.full(
        (max_reqs, max_kv_len), -1, dtype=torch.int32, device=DEVICE
    )
    for r in range(batch_size):
        start_slot = r * max_kv_len
        for p in range(max_kv_len):
            req_to_token[r, p] = start_slot + p
        # 填充 prefix 的 KV (draft 的 KV 由 caller 通过 set_kv_buffer 写, 这里测试直接全填)
        for p in range(max_kv_len):
            slot = start_slot + p
            k_cache[slot] = torch.randn(num_kv_heads, head_dim, dtype=dtype, device=DEVICE) * 0.5
            v_cache[slot] = torch.randn(num_kv_heads, head_dim, dtype=dtype, device=DEVICE) * 0.5
            idx_k_cache[slot] = torch.randn(1, idx_head_dim, dtype=dtype, device=DEVICE) * 0.5
            if idx_v_cache is not None:
                idx_v_cache[slot] = torch.randn(1, idx_head_dim, dtype=dtype, device=DEVICE) * 0.5

    # q / idx_q: [total_q, num_heads, head_dim], 按 [r0_t0..r0_t{D-1}, r1_t0, ...] 交错
    q = torch.randn(total_q, num_q_heads, head_dim, dtype=dtype, device=DEVICE) * 0.5
    idx_q = torch.randn(total_q, 1, idx_head_dim, dtype=dtype, device=DEVICE) * 0.5

    # cu_seqlens: [bs+1], 每请求 D 个 token
    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=DEVICE)
    for r in range(batch_size):
        cu_seqlens[r + 1] = cu_seqlens[r] + D

    # seq_lens: prefix + D (verify 真实 KV 长度)
    seq_lens = torch.full((batch_size,), prefix_len + D, dtype=torch.int32, device=DEVICE)
    # prefix_lens: prefix
    prefix_lens = torch.full((batch_size,), prefix_len, dtype=torch.int32, device=DEVICE)
    # slot_ids = req_pool_indices (M3 约定)
    slot_ids = torch.arange(batch_size, dtype=torch.int32, device=DEVICE)

    return {
        "q": q, "idx_q": idx_q,
        "k_cache": k_cache, "v_cache": v_cache,
        "idx_k_cache": idx_k_cache, "idx_v_cache": idx_v_cache,
        "req_to_token": req_to_token, "slot_ids": slot_ids,
        "cu_seqlens": cu_seqlens, "seq_lens": seq_lens, "prefix_lens": prefix_lens,
        "max_kv_len": max_kv_len,
    }


def run_case(batch_size, prefix_len, disable_value, topk=16, init_blocks=0, local_blocks=1,
             num_q_heads=8, num_kv_heads=1, head_dim=128, idx_head_dim=128,
             block_size_k=128, draft_token_num=4):
    D = draft_token_num
    inp = build_test_inputs(
        batch_size, D, prefix_len, num_q_heads, num_kv_heads, head_dim,
        idx_head_dim, block_size_k, disable_value,
    )
    max_seqlen_q = D
    max_seqlen_k = prefix_len + D
    max_seqblock_k_upper = (max_seqlen_k + block_size_k - 1) // block_size_k  # 真实上界 (测试用)

    common_kwargs = dict(
        q=inp["q"], k_cache=inp["k_cache"], v_cache=inp["v_cache"], sink=None,
        idx_q=inp["idx_q"], idx_k_cache=inp["idx_k_cache"], idx_v_cache=inp["idx_v_cache"],
        idx_sink=None, req_to_token=inp["req_to_token"], slot_ids=inp["slot_ids"],
        cu_seqlens=inp["cu_seqlens"], seq_lens=inp["seq_lens"], prefix_lens=inp["prefix_lens"],
        max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
        block_size_q=1, block_size_k=block_size_k, topk=topk,
        init_blocks=init_blocks, local_blocks=local_blocks,
        score_type="max", disable_index_value=disable_value,
    )

    # 关键: 用同一份输入 inp 对比两个 kernel, 才能证明固定上界改造不改变计算.
    # 两个 kernel 都只读 k_cache/v_cache/req_to_token, 输出 o/topk_idx 是新分配的张量,
    # 不会污染 inp, 所以同一份 inp 复用安全.

    # 原 prefill kernel
    o_orig, topk_idx_orig = minimax_sparse_prefill(**common_kwargs)

    # 新 verify kernel (固定上界, 同一 inp)
    o_new, topk_idx_new = minimax_sparse_verify_prefill(
        **common_kwargs, max_seqblock_k_upper=max_seqblock_k_upper,
    )

    # 对比 topk_idx (最关键: 选块必须一致)
    # topk_idx 形状: [num_kv_heads, all_seqblock_q, topk]
    topk_match = torch.equal(topk_idx_orig, topk_idx_new)
    if not topk_match:
        diff = (topk_idx_orig != topk_idx_new).sum().item()
        total = topk_idx_orig.numel()
        print(f"    topk_idx diff: {diff}/{total} elements differ")
        print(f"    orig[0,0,:8]: {topk_idx_orig[0,0,:8].tolist()}")
        print(f"    new [0,0,:8]: {topk_idx_new[0,0,:8].tolist()}")

    # 对比 o (输出): 应该完全一致 (除 bf16 数值噪声, max_abs_diff < 1e-2)
    # idx_o (第一个返回) 是 index head 的 o, 第二个是 main head 的 o.
    # 两 kernel 都返回 (idx_o, o); 这里比 main o.
    if o_orig is not None and o_new is not None:
        o_diff = (o_orig.float() - o_new.float()).abs()
        o_max = o_diff.max().item()
        o_mean = o_diff.mean().item()
    else:
        o_max = -1.0
        o_mean = -1.0
    o_match = o_max < 1e-2  # bf16 噪声阈值

    if not o_match:
        print(f"    o max_abs_diff={o_max:.6f} mean={o_mean:.6f} (阈值 1e-2)")

    return topk_match and o_match


def main():
    if not torch.cuda.is_available():
        print("SKIP: 需要 CUDA/HIP")
        return

    print("=" * 70)
    print("精度测试: minimax_sparse_verify_prefill vs minimax_sparse_prefill")
    print("目标: 固定上界改造不改变 topk_idx (选块一致)")
    print("=" * 70)

    cases = [
        # (batch_size, prefix_len, disable_value, 描述)
        (1, 16, False, "bs=1 prefix=16 (短)"),
        (1, 128, False, "bs=1 prefix=128 (1 block)"),
        (1, 1000, False, "bs=1 prefix=1000 (长)"),
        (4, 16, False, "bs=4 prefix=16"),
        (4, 500, False, "bs=4 prefix=500"),
        (16, 2000, False, "bs=16 prefix=2000 (verify 典型)"),
        (1, 16, True, "bs=1 prefix=16 disable_value=True"),
        (4, 500, True, "bs=4 prefix=500 disable_value=True"),
        (16, 2000, True, "bs=16 prefix=2000 disable_value=True"),
    ]

    all_pass = True
    for bs, pl, dv, desc in cases:
        try:
            ok = run_case(bs, pl, dv)
            status = "PASS" if ok else "FAIL"
            if not ok:
                all_pass = False
            print(f"  [{status}] {desc}  (disable_value={dv})")
        except Exception as e:
            all_pass = False
            print(f"  [ERROR] {desc}: {e}")
            import traceback
            traceback.print_exc()

    print("=" * 70)
    if all_pass:
        print("ALL PASS: verify kernel 与原 prefill kernel topk_idx 完全一致")
    else:
        print("FAILED: 存在不一致, 固定上界改造改变了计算结果!")
    print("=" * 70)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
