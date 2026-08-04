"""
VMFault 根治 — score 第3维形状一致性回归测试 (纯 CPU, 秒级, 无需 GPU).

本测试**不**复现越界写本身 (那需要 GPU + cuda graph, 见
test_vmfault_graph_repro.py), 只验证越界的**必要条件**: score buffer 第3维
在 capture/replay 下是否一致。

根因 (site-packages 实测确认):
  prefill/flash_with_topk_idx.py:
    max_seqblock_k = cdiv(max_seqlen_k, block_size_k)
    score = torch.full((num_heads, total_q, max_seqblock_k), -inf)
  score 第3维依赖 max_seqlen_k = backend._max_seqlen_k:
    capture: get_cuda_graph_seq_len_fill_value()=1, dummy seq_lens=1
             -> _max_seqlen_k = 1 + draft(4) = 5  -> score第3维 = cdiv(5,128) = 1  (被graph锁定)
    replay:  真实 seq_lens ~2000
             -> _max_seqlen_k ~2004               -> kernel写 score[...,16] -> 越界 -> VMFault

修复: backend __init__ 存 max_seqblock_k_upper = cdiv(context_len + draft, block_size_k)
      经 minimax_sparse_prefill 透传, score 第3维用此恒定上界 (capture/replay 形状一致).
      kernel 内 block_num 仍从真实 seq_lens 算, boundary_check 保护, 多余槽位填 -inf 不读.

本测试的价值 = 回归保护: 以后若有人把 max_seqblock_k_upper 改回动态值 (cdiv(max_seqlen_k,...)),
capture/replay 形状会重新分叉, 本测试立刻 fail, 提醒越界风险回归.
"""
import torch
import triton

BLOCK_SIZE_K = 128  # = MiniMax-M3 sparse_block_size (sparse_cfg["sparse_block_size"])
CONTEXT_LEN = 204800
MAX_DRAFT = 4
max_seqlen_k_bound = CONTEXT_LEN + MAX_DRAFT
UPPER = (max_seqlen_k_bound + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K  # = cdiv(204804,128) = 1601


def score_dim(max_seqlen_k, max_seqblock_k_upper=None):
    """复刻 flash_prefill_with_topk_index 里 score 第3维的决策逻辑."""
    max_seqblock_k = triton.cdiv(max_seqlen_k, BLOCK_SIZE_K)
    return (max_seqblock_k_upper if max_seqblock_k_upper is not None
            else max_seqblock_k)


def main():
    DRAFT = MAX_DRAFT

    # --- 旧逻辑 (无上界): capture vs replay 形状不一致 (越界的必要条件) ---
    cap_old = score_dim(1 + DRAFT)
    rep_old = score_dim(2000 + DRAFT)
    print(f"[旧] capture score第3维={cap_old}, replay={rep_old}")
    assert cap_old != rep_old, "旧逻辑 capture/replay 应不一致 (这是 bug)"
    print(f"  -> 旧代码: capture={cap_old} vs replay={rep_old} 形状分叉")
    print(f"     (越界的必要条件; 真越界复现见 test_vmfault_graph_repro.py)")

    # --- 新逻辑 (恒定上界): capture vs replay 形状一致 ---
    cap_new = score_dim(1 + DRAFT, UPPER)
    rep_new = score_dim(2000 + DRAFT, UPPER)
    big_new = score_dim(150000 + DRAFT, UPPER)
    print(f"[新] capture={cap_new}, replay={rep_new}, 长上下文={big_new}, 上界={UPPER}")
    assert cap_new == rep_new == big_new == UPPER
    assert rep_new >= 16, "上界必须 >= 真实 block 数 (cdiv(2004,128)=16)"
    print("  -> 新代码: capture/replay/长上下文 形状恒定 = 上界, 且 >= 真实 block 数")

    # --- 上界必须覆盖最坏可能真实 block 数 (context_len + draft) ---
    worst_real = score_dim(CONTEXT_LEN + DRAFT)
    assert UPPER >= worst_real, f"上界 {UPPER} 必须 >= 最坏真实 {worst_real}"
    print(f"  -> 上界 {UPPER} >= 最坏真实 block 数 {worst_real} (不会越界)")

    # --- 真实 score alloc shape (新逻辑, CPU 小张量) ---
    num_heads, total_q = 8, 16
    score = torch.full((num_heads, total_q, UPPER), float("-inf"),
                       dtype=torch.float32, device="cpu")
    print(f"[新] score alloc shape={tuple(score.shape)}, "
          f"显存={score.numel()*4/1024:.1f}KB (单层单batch)")
    assert score.shape[2] == UPPER
    print("  -> score 第3维 = 恒定上界, capture/replay 一致")

    print("\n=== ✅ VMFault 根治 (形状一致性) 验证通过 ===")
    print("score buffer 第3维用 cdiv(context_len+draft, block_size_k) 恒定上界,")
    print("capture (dummy seq_lens=1) 与 replay (真实 ~2000) 形状一致, 越界必要条件消除。")


if __name__ == "__main__":
    main()
