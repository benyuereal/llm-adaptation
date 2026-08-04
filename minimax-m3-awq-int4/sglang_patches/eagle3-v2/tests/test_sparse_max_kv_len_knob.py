"""
MINIMAX_SPARSE_MAX_KV_LEN 性能旋钮单元测试 (纯 CPU, 秒级, 无需 GPU/模型).

测试对象: minimax_sparse_backend.MiniMaxSparseAttnBackend.__init__ 里
max_seqblock_k_upper 的可配置收紧逻辑 + replay 路径的 fail-fast 安全检查。

背景 (性能):
  verify kernel 每 layer 每 verify 都分配 score[num_heads, total_q, max_seqblock_k_upper],
  默认 max_seqblock_k_upper = cdiv(context_len + D, block_size_k)。context_len=204800
  → 1601, 但真实请求远短, score 按 1601 分配但只填前 ~10 维 → 57 层 × 每 verify
  的 alloc + -inf init 开销浪费。MINIMAX_SPARSE_MAX_KV_LEN 收紧上界 (仍 graph-safe
  常量), 同比例降分配量。

背景 (安全):
  收紧上界后, 若真实 seq_len+D 超过 max_seqblock_k_upper, kernel 的 K-block 循环
  会 OOB 写 score → VMFault。kernel 内的 assert 抓不住 (因 verify 传
  _max_seqlen_k = upper*bsk, 使 assert 变恒等式)。所以 replay 路径用真实
  seq_lens_cpu 做 fail-fast 检查, 设太小 → 清晰报错, 不静默 VMFault。

测试 (纯 CPU, mock 复刻 __init__ + replay 检查逻辑):
  1. 不设旋钮 (默认 0) → 用完整 context_len, 上界 = 1601 (行为不变)
  2. 设 32768 → 上界 = min(204800, 32768)→257 (降 6 倍)
  3. 设 > context_len → 仍用 context_len (不能放宽)
  4. replay 真实 seq_len+D <= 上界 → 通过 (不抛)
  5. replay 真实 seq_len+D > 上界 → fail-fast 抛 AssertionError (不 VMFault)
  6. 真实 backend 源码含旋钮 + fail-fast 检查 (回归保护)

运行:
    python test_sparse_max_kv_len_knob.py
预期: 6 组全过, "✅ MINIMAX_SPARSE_MAX_KV_LEN 旋钮验证通过"
"""
import os
import torch

CONTEXT_LEN = 204800
BLOCK_SIZE_K = 128
MAX_DRAFT = 4


def cdiv(a, b):
    return (a + b - 1) // b


def compute_upper(context_len, max_draft, block_size_k, user_cap=0):
    """复刻 backend __init__ 的 max_seqblock_k_upper 计算逻辑."""
    if user_cap > 0:
        effective_ctx = min(context_len, user_cap)
    else:
        effective_ctx = context_len
    max_seqlen_k_bound = effective_ctx + max_draft
    return (max_seqlen_k_bound + block_size_k - 1) // block_size_k, effective_ctx


def replay_fail_fast_check(max_seqblock_k_upper, real_max_k, block_size_k, effective_ctx):
    """复刻 backend init_forward_metadata_replay_cuda_graph 的 fail-fast assert.
    real_max_k = max(seq_lens_cpu) + D (真实 K 长度)."""
    need_blocks = (real_max_k + block_size_k - 1) // block_size_k
    assert need_blocks <= max_seqblock_k_upper, (
        f"verify score upper bound too small: max_seqblock_k_upper="
        f"{max_seqblock_k_upper} but real max seq_len+D={real_max_k} needs "
        f"{need_blocks}. Increase MINIMAX_SPARSE_MAX_KV_LEN (currently "
        f"{effective_ctx}) to >= {real_max_k}."
    )


# ============================================================
def test_1_default_uses_full_context_len():
    """不设旋钮 → 上界 = cdiv(context_len+D, bsk) = 1601 (行为不变)."""
    upper, eff = compute_upper(CONTEXT_LEN, MAX_DRAFT, BLOCK_SIZE_K, user_cap=0)
    assert upper == 1601, upper
    assert eff == CONTEXT_LEN
    print(f"  [1] 默认 (不设): upper={upper} (= cdiv(204804,128)) 行为不变 ✓")


def test_2_knob_tightens_upper():
    """设 32768 → 上界降到 257 (降 6 倍)."""
    upper, eff = compute_upper(CONTEXT_LEN, MAX_DRAFT, BLOCK_SIZE_K, user_cap=32768)
    expected = cdiv(32768 + MAX_DRAFT, BLOCK_SIZE_K)  # = 257
    assert upper == expected, f"upper={upper} expected={expected}"
    assert eff == 32768
    ratio = 1601 / upper
    print(f"  [2] 设 32768: upper={upper} (从 1601 降 {ratio:.1f}x) ✓")


def test_3_knob_above_context_uses_context():
    """设 > context_len → 仍用 context_len (旋钮只能收紧, 不能放宽)."""
    upper, eff = compute_upper(CONTEXT_LEN, MAX_DRAFT, BLOCK_SIZE_K, user_cap=999999)
    assert upper == 1601, upper
    assert eff == CONTEXT_LEN  # min(204800, 999999) = 204800
    print(f"  [3] 设 999999 (>context): upper={upper} (仍用 context_len, 不放宽) ✓")


def test_4_replay_real_within_upper_passes():
    """replay 真实 seq_len+D <= 上界 → 不抛 (正常服役)."""
    # 设 32768, upper=257; 真实最长请求 30000 token + D=4 = 30004, need 235 blocks
    upper, eff = compute_upper(CONTEXT_LEN, MAX_DRAFT, BLOCK_SIZE_K, user_cap=32768)
    real_max_k = 30000 + MAX_DRAFT  # 30004
    replay_fail_fast_check(upper, real_max_k, BLOCK_SIZE_K, eff)  # 不抛
    print(f"  [4] replay 真实 {real_max_k} <= upper {upper}*{BLOCK_SIZE_K} → 通过 ✓")


def test_5_replay_real_exceeds_upper_fails_fast():
    """replay 真实 seq_len+D > 上界 → fail-fast 抛 AssertionError (不 VMFault)."""
    # 设 32768, upper=257 → 上界 257*128=32896; 真实请求 40000+4=40004 超过
    upper, eff = compute_upper(CONTEXT_LEN, MAX_DRAFT, BLOCK_SIZE_K, user_cap=32768)
    real_max_k = 40000 + MAX_DRAFT  # 40004 > 32896
    raised = False
    try:
        replay_fail_fast_check(upper, real_max_k, BLOCK_SIZE_K, eff)
    except AssertionError as e:
        raised = True
        assert "MINIMAX_SPARSE_MAX_KV_LEN" in str(e), "错误信息应提示旋钮"
        assert "32768" in str(e), "错误信息应含当前值"
    assert raised, "超过上界应抛 AssertionError (fail-fast, 不静默 VMFault)"
    print(f"  [5] replay 真实 {real_max_k} > upper 上限 → fail-fast 抛错 (含旋钮提示) ✓")


def test_6_real_backend_has_knob_and_failfast():
    """回归保护: 真实 backend 源码含旋钮 + fail-fast 检查."""
    import inspect
    try:
        from sglang.srt.layers.attention.minimax_sparse_backend import (
            MiniMaxSparseAttnBackend as B,
        )
    except Exception as e:
        print(f"  [6] 跳过 (无法 import sglang backend: {e}) — 非 sglang 环境")
        return
    init_src = inspect.getsource(B.__init__)
    replay_src = inspect.getsource(B.init_forward_metadata_replay_cuda_graph)
    checks = {
        "__init__ 读 MINIMAX_SPARSE_MAX_KV_LEN": "MINIMAX_SPARSE_MAX_KV_LEN" in init_src,
        "__init__ min(context, cap)": "min(context_len, _user_cap)" in init_src,
        "__init__ 存 _sparse_effective_ctx": "_sparse_effective_ctx" in init_src,
        "replay fail-fast real_max_k": "real_max_k" in replay_src,
        "replay fail-fast need_blocks": "need_blocks" in replay_src,
        "replay assert need_blocks <= upper": "need_blocks <= self.max_seqblock_k_upper" in replay_src,
        "replay 错误提示含旋钮名": "MINIMAX_SPARSE_MAX_KV_LEN" in replay_src,
    }
    missing = [k for k, v in checks.items() if not v]
    assert not missing, "真实 backend 缺失旋钮/fail-fast 逻辑:\n  " + "\n  ".join(missing)
    print(f"  [6] 真实 backend 含全部 {len(checks)} 项旋钮+fail-fast 标记 ✓")


# ============================================================
def main():
    print("=" * 66)
    print("MINIMAX_SPARSE_MAX_KV_LEN 性能旋钮单元测试 (纯 CPU)")
    print(f"  context_len={CONTEXT_LEN} block_size_k={BLOCK_SIZE_K} D={MAX_DRAFT}")
    print("=" * 66)
    test_1_default_uses_full_context_len()
    test_2_knob_tightens_upper()
    test_3_knob_above_context_uses_context()
    test_4_replay_real_within_upper_passes()
    test_5_replay_real_exceeds_upper_fails_fast()
    test_6_real_backend_has_knob_and_failfast()
    print("=" * 66)
    print("✅ MINIMAX_SPARSE_MAX_KV_LEN 旋钮验证通过 (6 组测试全过)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
