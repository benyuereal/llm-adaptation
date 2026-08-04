"""
真正的 cuda graph 越界写复现 (不依赖 sglang/模型, 只用 torch+triton+1张DCU).

复现 EAGLE3 TARGET_VERIFY 的 VMFault 机制。根因在
prefill/flash_with_topk_idx.py:
    score = torch.full((H, total_q, cdiv(max_seqlen_k, block_size_k)), -inf)
    # kernel 内:
    block_num = (seq_len + block_size - 1) // block_size   # 从真实 seq_lens 算
    s_ptrs = tl.make_block_ptr(shape=(q_len, block_num), ...)
    tl.store(s_ptrs, val, boundary_check=(0,1))
score 第3维 = cdiv(max_seqlen_k, block_size_k), 而 max_seqlen_k = backend._max_seqlen_k:
  - capture: get_cuda_graph_seq_len_fill_value()=1, dummy seq_lens=1
             -> _max_seqlen_k = 1 + draft(4) = 5  -> score第3维 = cdiv(5,64) = 1  (被graph锁定)
  - replay:  真实 seq_lens ~2000 -> _max_seqlen_k ~2004
             -> kernel block_num = cdiv(2004,64) = 32 > score第3维(1)
             -> store 写 score[..., 1..31] 落到 score 分配之外 -> 越界写

真实环境里 score 之后是同一 allocator 池里相邻的张量, 越界写破坏它们 -> garbage
输出, 越界足够远跨页/未映射页时 ROCr 报 KERNEL VMFault: Invalid address access.

本测试用最小 triton kernel 复刻该 store 模式, 用一个连续大 buffer 切片出 score,
紧贴其后是 guard 区. 这样越界写一定能被检测到 (不依赖 GPU 页布局):
  - 旧逻辑 (score 第3维 = cdiv(capture_seq, block_size) = 1): replay 越界写 guard 区
  - 新逻辑 (score 第3维 = 恒定上界 3201): replay block_num(32) << 3201, 全在 score 内

注: 测试不真正触发 ROCr VMFault (那会崩进程). 它演示的是越界写本身 ——
"写到 score 分配范围之外的内存" —— 即 VMFault 的直接前因.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _store_score_kernel(
    score_ptr,          # 连续 buffer 的起始 (score 切片自此)
    seq_lens_ptr,       # [batch]  真实 seq_len
    stride_s_h, stride_s_q, stride_s_k,
    block_size,
    H: tl.constexpr,
    BATCH: tl.constexpr,
):
    """复刻 flash_with_topk_idx 的 score store: block_num 从真实 seq_len 算,
    mask = off_k < block_num, 写入地址 base + off_k*stride_s_k."""
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    seq_len = tl.load(seq_lens_ptr + pid_b)
    block_num = (seq_len + block_size - 1) // block_size
    off_k = tl.arange(0, 256)
    mask = off_k < block_num
    base = score_ptr + pid_b * stride_s_q + pid_h * stride_s_h
    # 写入值 = off_k (便于检测哪些 k 位置被写)
    tl.store(base + off_k * stride_s_k, off_k.to(tl.float32), mask=mask)


def _alloc_score_with_guard(score_dim3, H=2, BATCH=1, guard_len=512):
    """从一个连续大 buffer 切出 score, 紧贴其后是 guard 区.
    返回 (score, guard, big). score 和 guard 共享底层连续内存,
    所以越界写 score 一定会污染 guard (不受 GPU 页布局影响)."""
    device = "cuda"
    total_score_elems = H * BATCH * score_dim3
    big = torch.full((total_score_elems + guard_len,), -99.0,
                     dtype=torch.float32, device=device)
    score = big[:total_score_elems].view(H, BATCH, score_dim3)
    guard = big[total_score_elems:]
    assert score.data_ptr() + score.numel() * 4 == guard.data_ptr(), "guard 必须紧贴 score"
    return score, guard, big


def run_one(score_dim3, capture_seq, replay_seq, block_size=64, H=2, BATCH=1):
    """模拟 cuda graph capture/replay 一次. 返回 (越界写到的最远 big 索引, score 第3维)."""
    score, guard, big = _alloc_score_with_guard(score_dim3, H, BATCH)
    seq_lens = torch.tensor([capture_seq], dtype=torch.int32, device=score.device)

    g = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        _store_score_kernel[(BATCH, H)](
            score, seq_lens,
            score.stride(0), score.stride(1), score.stride(2),
            block_size, H=H, BATCH=BATCH,
        )
    # replay: 换成真实 seq_lens (模拟 replay_prepare 填 graph buffer)
    seq_lens.copy_(torch.tensor([replay_seq], dtype=torch.int32, device=score.device))
    g.replay()
    torch.cuda.synchronize()

    # 检测越界: big 里被写成非 -99 的最远索引 (写入值=off_k, 都是 >=0 的整数)
    written = (big != -99.0).nonzero(as_tuple=False).reshape(-1)
    if written.numel() == 0:
        return -1, score_dim3, big
    farthest = int(written.max().item())
    return farthest, score_dim3, big


def main():
    BLOCK_SIZE = 64
    H, BATCH = 2, 1
    capture_seq = 1 + 4      # dummy seq_lens=1 + draft 4 (同真实 capture)
    replay_seq = 2000 + 4    # 真实 seq_lens ~2000 + draft 4

    print("=" * 72)
    print("真正的 cuda graph 越界写复现 (capture 锁小 score 形状 -> replay 越界)")
    print("=" * 72)
    print(f"capture seq_lens={capture_seq}, replay seq_lens={replay_seq}, "
          f"block_size={BLOCK_SIZE}")
    print(f"replay block_num = cdiv({replay_seq},{BLOCK_SIZE}) = "
          f"{triton.cdiv(replay_seq, BLOCK_SIZE)} (kernel 按此写 score[..., 0:block_num])")

    # ---- 旧逻辑: score 第3维 = cdiv(capture_seq_len, block_size) ----
    old_dim3 = triton.cdiv(capture_seq, BLOCK_SIZE)  # = 1
    print(f"\n[旧逻辑] score 第3维 = cdiv(capture_seq={capture_seq}, {BLOCK_SIZE}) = {old_dim3}")
    farthest, _, big = run_one(old_dim3, capture_seq, replay_seq, BLOCK_SIZE, H, BATCH)
    score_elems = H * BATCH * old_dim3  # = 2
    print(f"         score 合法范围 = {score_elems} 个 float")
    print(f"         被写的最远 big 索引 = {farthest}  (写入值如下)")
    print(f"         big[0:36] = {big[:36].tolist()}")
    oob = farthest >= score_elems
    if oob:
        print(f"         ✗ 越界写复现! 写到 big[{farthest}] >= score 末尾({score_elems})")
        print(f"           真实 sglang 里这就是 score 之后的相邻张量被破坏 ->")
        print(f"           garbage 输出 / 越界跨页时 ROCr 报 KERNEL VMFault")
        old_repro = True
    else:
        print(f"         (未检测到越界)")
        old_repro = False

    # ---- 新逻辑: score 第3维 = 恒定上界 ----
    CONTEXT_LEN = 204800
    MAX_DRAFT = 4
    new_dim3 = triton.cdiv(CONTEXT_LEN + MAX_DRAFT, BLOCK_SIZE)  # = 3201
    print(f"\n[新逻辑] score 第3维 = cdiv(context_len+draft={CONTEXT_LEN+MAX_DRAFT}, {BLOCK_SIZE}) = {new_dim3}")
    farthest, _, big = run_one(new_dim3, capture_seq, replay_seq, BLOCK_SIZE, H, BATCH)
    score_elems = H * BATCH * new_dim3
    print(f"         score 合法范围 = {score_elems} 个 float")
    print(f"         被写的最远 big 索引 = {farthest}")
    print(f"         block_num(replay)={triton.cdiv(replay_seq, BLOCK_SIZE)} << score第3维={new_dim3}")
    safe = farthest < score_elems
    if safe:
        print(f"         ✓ 安全: 所有写落在 score 分配范围内 (capture/replay 形状恒定={new_dim3})")
        new_safe = True
    else:
        print(f"         ✗ 意外越界")
        new_safe = False

    print("\n" + "=" * 72)
    if old_repro and new_safe:
        print("=== ✅ 真复现成功 ===")
        print("旧逻辑: capture 锁 score 第3维=1, replay 写 block_num=32 -> 越界写 30 个 float")
        print("        到 score 之后的内存 (真实环境即 VMFault 的直接前因).")
        print("新逻辑: score 第3维恒定=3201, replay block_num=32 全在范围内, 不越界.")
    else:
        print(f"=== 结果: 旧越界={old_repro}, 新安全={new_safe} ===")


if __name__ == "__main__":
    main()
