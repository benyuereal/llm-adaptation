#!/usr/bin/env python3
"""对照测试: 旧代码 (临时 extend_seq_lens) vs 新代码 (arange + buffer) 在 graph 下行为.

目的: 验证 "旧代码复现崩, 新代码不崩" 的假设是否成立.

关键认知 (前置测试已证):
  - graph pool 保护 capture 期间所有分配的地址 (不被 GC)
  - torch.full 在 capture 时 lazy (不执行), replay 时重放 op 才填值
  - 所以旧代码 (临时 extend_seq_lens) 在 graph 下地址和值都正确

预期结论:
  - 若旧代码在离线 graph 下也 PASS → 旧代码不是因 "临时 tensor" 崩,
    端到端崩溃根因在别处 (sglang 特定机制, 离线无法复现)
  - 若旧代码在离线 graph 下 FAIL → 复现成功, 证明修复有效

本测试同时跑旧/新逻辑, 对比结果.

运行: python test_verify_old_vs_new_graph.py
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
MAX_SEQBLOCK_K_UPPER = (CONTEXT_LEN + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
MAX_BS = 16
SEQ_LEN_FILL_VALUE = 1


def build_paged_cache(max_bs, max_prefix):
    max_kv_len = max_prefix + D
    max_slots = max_bs * max_kv_len + 256
    max_reqs = max_bs + 8
    k_cache = torch.zeros(max_slots, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    v_cache = torch.zeros(max_slots, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    idx_k_cache = torch.zeros(max_slots, NUM_IDX_HEADS, IDX_HEAD_DIM, dtype=DTYPE, device=DEVICE)
    idx_v_cache = torch.zeros(max_slots, NUM_IDX_HEADS, IDX_HEAD_DIM, dtype=DTYPE, device=DEVICE)
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


def old_verify_branch(seq_lens_buffer, bs, D_const):
    """旧代码 (修复前): 用临时 extend_seq_lens 算 cu_seqlens/seq_lens.

    模拟 minimax_sparse_backend.py 修复前的 forward_extend verify 分支:
      if forward_batch.extend_seq_lens is None:
          forward_batch.extend_seq_lens = torch.full((num_reqs,), D, ...)  # 临时新建
      cu_seqlens = torch.cat([zeros(1), extend_seq_lens.cumsum(0)])
      prefix_lens = raw_seq_lens
      seq_lens = raw_seq_lens + extend_seq_lens
    """
    raw_seq_lens = seq_lens_buffer[:bs].to(torch.int32)
    # 临时 extend_seq_lens (capture 内新建, 进 graph 私有池)
    extend_seq_lens = torch.full((bs,), D_const, dtype=torch.int32, device=raw_seq_lens.device)
    cu_seqlens = torch.cat([
        torch.zeros(1, dtype=torch.int32, device=raw_seq_lens.device),
        extend_seq_lens.to(torch.int32).cumsum(0).to(torch.int32),
    ])
    prefix_lens = raw_seq_lens
    seq_lens = raw_seq_lens + extend_seq_lens.to(torch.int32)
    return cu_seqlens, seq_lens, prefix_lens


def new_verify_branch(seq_lens_buffer, bs, D_const):
    """新代码 (修复后): 用 arange + buffer + Python int.

    模拟修复后的 forward_extend verify 分支:
      cu_seqlens = torch.arange(0, (bs+1)*D, D)
      prefix_lens = raw_seq_lens
      seq_lens = raw_seq_lens + D
    """
    raw_seq_lens = seq_lens_buffer[:bs].to(torch.int32)
    cu_seqlens = torch.arange(0, (bs + 1) * D_const, step=D_const, dtype=torch.int32, device=raw_seq_lens.device)
    prefix_lens = raw_seq_lens
    seq_lens = raw_seq_lens + D_const
    return cu_seqlens, seq_lens, prefix_lens


def run_graph(branch_fn, seq_lens_buffer, q_buffer, idx_q_buffer, req_pool_indices_buffer,
              k_cache, v_cache, idx_k_cache, idx_v_cache, req_to_token, bs, capture_max_k):
    """capture (dummy) + replay (populate), 返回 graph 输出 o_g. 失败返回 None."""
    pool = torch.cuda.graph_pool_handle()
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g, pool=pool):
            cu_g, sl_g, pl_g = branch_fn(seq_lens_buffer, bs, D)
            _, o_g = minimax_sparse_verify_prefill(
                q=q_buffer, k_cache=k_cache, v_cache=v_cache, sink=None,
                idx_q=idx_q_buffer, idx_k_cache=idx_k_cache, idx_v_cache=idx_v_cache, idx_sink=None,
                req_to_token=req_to_token, slot_ids=req_pool_indices_buffer[:bs],
                cu_seqlens=cu_g, seq_lens=sl_g, prefix_lens=pl_g,
                max_seqlen_q=D, max_seqlen_k=capture_max_k,
                max_seqblock_k_upper=MAX_SEQBLOCK_K_UPPER,
                block_size_q=BLOCK_SIZE_Q, block_size_k=BLOCK_SIZE_K,
                topk=TOPK, init_blocks=INIT_BLOCKS, local_blocks=LOCAL_BLOCKS,
                score_type="max", disable_index_value=False,
            )
        torch.cuda.synchronize()
        return g, o_g
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def test_old_vs_new():
    print("=" * 70)
    print("对照: 旧代码 (临时 extend_seq_lens) vs 新代码 (arange+buffer) 在 graph 下")
    print("=" * 70)

    real_prefix = 500
    bs = MAX_BS
    k_cache, v_cache, idx_k_cache, idx_v_cache, req_to_token, max_slots = build_paged_cache(MAX_BS, real_prefix)

    # graph buffers (预分配)
    seq_lens_buffer = torch.full((MAX_BS,), SEQ_LEN_FILL_VALUE, dtype=torch.int32, device=DEVICE)
    req_pool_indices_buffer = torch.arange(MAX_BS, dtype=torch.int32, device=DEVICE)
    total_q = MAX_BS * D
    q_buffer = torch.randn(total_q, NUM_Q_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE) * 0.3
    idx_q_buffer = torch.randn(total_q, NUM_IDX_HEADS, IDX_HEAD_DIM, dtype=DTYPE, device=DEVICE) * 0.3

    # eager 基线 (真实值)
    q_e = q_buffer.clone()
    idx_q_e = idx_q_buffer.clone()
    cu_e = torch.arange(0, (bs+1)*D, step=D, dtype=torch.int32, device=DEVICE)
    sl_e = torch.full((bs,), real_prefix + D, dtype=torch.int32, device=DEVICE)
    pl_e = torch.full((bs,), real_prefix, dtype=torch.int32, device=DEVICE)
    rp_e = torch.arange(bs, dtype=torch.int32, device=DEVICE)
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
    print(f"\n[基线] eager OK: o.shape={tuple(o_eager.shape)}")

    # warmup (两个 branch 各 warmup, 触发 autotune)
    for fn in [old_verify_branch, new_verify_branch]:
        for _ in range(3):
            cu_w, sl_w, pl_w = fn(seq_lens_buffer, bs, D)
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

    capture_max_k = SEQ_LEN_FILL_VALUE + D  # capture 时 max_seqlen_k 小

    # ---- 旧代码 ----
    print("\n[旧代码] 临时 extend_seq_lens, graph capture + replay:")
    seq_lens_buffer.fill_(SEQ_LEN_FILL_VALUE)
    g_old, res_old = run_graph(old_verify_branch, seq_lens_buffer, q_buffer, idx_q_buffer,
                               req_pool_indices_buffer, k_cache, v_cache, idx_k_cache, idx_v_cache,
                               req_to_token, bs, capture_max_k)
    if g_old is None:
        print(f"  capture FAIL: {res_old}")
        old_pass = False
    else:
        print(f"  capture OK")
        # replay: populate
        seq_lens_buffer[:bs].copy_(torch.full((bs,), real_prefix, dtype=torch.int32, device=DEVICE))
        q_buffer.copy_(q_e); idx_q_buffer.copy_(idx_q_e)
        try:
            g_old.replay(); torch.cuda.synchronize()
            o_old = res_old
            diff_old = (o_eager.float() - o_old.float()).abs().max().item()
            print(f"  replay OK: diff vs eager = {diff_old:.6f}")
            old_pass = diff_old < 1e-2
        except Exception as e:
            print(f"  replay FAIL (复现崩?): {type(e).__name__}: {e}")
            old_pass = False

    # ---- 新代码 ----
    print("\n[新代码] arange + buffer + Python int, graph capture + replay:")
    seq_lens_buffer.fill_(SEQ_LEN_FILL_VALUE)
    g_new, res_new = run_graph(new_verify_branch, seq_lens_buffer, q_buffer, idx_q_buffer,
                               req_pool_indices_buffer, k_cache, v_cache, idx_k_cache, idx_v_cache,
                               req_to_token, bs, capture_max_k)
    if g_new is None:
        print(f"  capture FAIL: {res_new}")
        new_pass = False
    else:
        print(f"  capture OK")
        seq_lens_buffer[:bs].copy_(torch.full((bs,), real_prefix, dtype=torch.int32, device=DEVICE))
        q_buffer.copy_(q_e); idx_q_buffer.copy_(idx_q_e)
        try:
            g_new.replay(); torch.cuda.synchronize()
            o_new = res_new
            diff_new = (o_eager.float() - o_new.float()).abs().max().item()
            print(f"  replay OK: diff vs eager = {diff_new:.6f}")
            new_pass = diff_new < 1e-2
        except Exception as e:
            print(f"  replay FAIL: {type(e).__name__}: {e}")
            new_pass = False

    print("\n" + "=" * 70)
    print(f"  旧代码 (临时 extend_seq_lens): {'PASS' if old_pass else 'FAIL/崩'}")
    print(f"  新代码 (arange + buffer):      {'PASS' if new_pass else 'FAIL/崩'}")
    print("=" * 70)
    if old_pass and new_pass:
        print("结论: 旧代码在离线 graph 下也 PASS → '临时 tensor' 不是崩因,")
        print("      端到端崩溃根因在 sglang 特定机制 (离线无法复现), 需端到端探针定位.")
        print("      新代码与旧代码结果一致, 写法更稳妥 (对齐 triton backend 模式).")
    elif not old_pass and new_pass:
        print("结论: 复现成功! 旧代码崩, 新代码不崩 → 修复有效.")
    elif not old_pass and not new_pass:
        print("结论: 两者都崩 → 根因不在 extend_seq_lens, 需另查.")
    else:
        print("结论: 旧崩新不崩的反向? 异常, 需检查.")
    sys.exit(0 if (new_pass) else 1)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("SKIP: 需要 CUDA/HIP"); sys.exit(0)
    test_old_vs_new()
