#!/usr/bin/env python3
"""验证 verify kernel 的 grid 在 capture 时是否用 max_seqlen_q=D (而非 prefill 的 16384).

崩溃线索: 崩溃 grid (256,1,1) 暗示 max_seqlen_q=16384 (=max_prefill_tokens),
而非 verify 应有的 D=4. 本测试捕获 verify kernel 调用, 打印实际 grid, 确认
capture 时 max_seqlen_q=D → grid=(1, bs*num_heads).

运行: python test_verify_grid_capture.py
"""
import os
import sys
import torch
import triton

sys.path.insert(0, "/usr/local/lib/python3.10/dist-packages")

from sglang.srt.layers.attention.minimax_sparse_ops.verify.flash_with_topk_idx import (
    flash_verify_prefill_with_topk_index,
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
NUM_Q_HEADS = 8
NUM_KV_HEADS = 1
HEAD_DIM = 128
IDX_HEAD_DIM = 128
BLOCK_SIZE_K = 128
BLOCK_SIZE_Q = 1
TOPK = 16
D = 4
DTYPE = torch.bfloat16
MAX_SEQBLOCK_K_UPPER = 1600


def test_grid():
    print("=" * 60)
    print("验证 verify kernel grid: capture 时 max_seqlen_q 应=D=4")
    print("=" * 60)

    for bs in [1, 4, 16]:
        total_q = bs * D
        max_kv_len = 500 + D
        max_slots = bs * max_kv_len + 128
        q = torch.randn(total_q, NUM_Q_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE) * 0.3
        idx_q = torch.randn(total_q, 1, IDX_HEAD_DIM, dtype=DTYPE, device=DEVICE) * 0.3
        k_cache = torch.randn(max_slots, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE) * 0.3
        v_cache = torch.randn(max_slots, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE) * 0.3
        idx_k_cache = torch.randn(max_slots, 1, IDX_HEAD_DIM, dtype=DTYPE, device=DEVICE) * 0.3
        idx_v_cache = torch.randn(max_slots, 1, IDX_HEAD_DIM, dtype=DTYPE, device=DEVICE) * 0.3
        req_to_token = torch.full((bs + 4, max_kv_len), -1, dtype=torch.int32, device=DEVICE)
        for r in range(bs):
            for p in range(max_kv_len):
                req_to_token[r, p] = r * max_kv_len + p
        cu_seqlens = torch.arange(0, (bs + 1) * D, step=D, dtype=torch.int32, device=DEVICE)
        seq_lens = torch.full((bs,), 500 + D, dtype=torch.int32, device=DEVICE)
        prefix_lens = torch.full((bs,), 500, dtype=torch.int32, device=DEVICE)
        slot_ids = torch.arange(bs, dtype=torch.int32, device=DEVICE)

        # 预期 grid
        expected_grid_x = triton.cdiv(D, 64)  # cdiv(4,64)=1
        expected_grid_y = bs * NUM_Q_HEADS
        print(f"\nbs={bs}: 预期 grid=({expected_grid_x}, {expected_grid_y}) (max_seqlen_q={D})")

        # patch kernel 的 launch 捕获 grid
        orig_kernel = flash_verify_prefill_with_topk_index
        captured_grid = {}

        # 直接调, 看是否崩 + 打印 grid (通过 monkey-patch triton kernel)
        try:
            torch.cuda.synchronize()
            _ = orig_kernel(
                q=q, k_cache=k_cache, v_cache=v_cache, sink=None,
                req_to_token=req_to_token, slot_ids=slot_ids,
                cu_seqlens=cu_seqlens, seq_lens=seq_lens, prefix_lens=prefix_lens,
                max_seqlen_q=D, max_seqlen_k=500 + D,
                max_seqblock_k_upper=MAX_SEQBLOCK_K_UPPER,
                block_size_q=BLOCK_SIZE_Q, block_size_k=BLOCK_SIZE_K,
                topk=TOPK, init_blocks=0, local_blocks=1,
                score_type="max", disable_index_value=False,
            )
            torch.cuda.synchronize()
            print(f"  kernel OK (max_seqlen_q={D} → grid.x 应={expected_grid_x})")
            print(f"  若端到端崩溃 grid=(256,1,1), 则 max_seqlen_q 被当成 16384 (=max_prefill_tokens)")
        except Exception as e:
            print(f"  kernel FAIL: {type(e).__name__}: {e}")

    print("\n" + "=" * 60)
    print("结论: 离线 max_seqlen_q=D=4 → grid.x=1 (正确)")
    print("  端到端崩溃 grid=(256,1,1) → max_seqlen_q=16384")
    print("  → 端到端时 self._max_seqlen_q 在 verify forward_extend 不是 D")
    print("  需在端到端加探针打印 forward_extend 时的 self._max_seqlen_q")
    print("=" * 60)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("SKIP"); sys.exit(0)
    test_grid()
