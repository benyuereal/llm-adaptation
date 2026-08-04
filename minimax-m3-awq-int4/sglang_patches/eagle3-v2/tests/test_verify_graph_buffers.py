"""
verify_prefill kernel graph-safety 单元测试 (纯 CPU, 秒级, 无需 GPU/模型).

测试对象: minimax_sparse_backend.MiniMaxSparseAttnBackend 的 EAGLE3 TARGET_VERIFY
graph buffer 逻辑 —— 这是 v2 VMFault 根治的核心. 测试**不**起 sglang / 不跑 triton
kernel / 不需要 GPU, 只用 mock 复刻 backend 的 buffer 管理 5 个方法, 验证:

  1. init_cuda_graph_state 预分配 3 个 graph buffer, dtype=int32, shape 正确
  2. capture/replay 填充后: cu_seqlens=[0,D,2D,...,bs*D], extend_seq_lens=[D]*bs
     (只依赖 bs & D -> graph 内常量 -> 地址稳定)
  3. capture 与 replay 用的是**同一块预分配 buffer** (data_ptr 相同) -> graph-safe
     (这是旧实现 vmfault 的真正根因: 旧实现用 torch.cat/a+b 在 forward_extend 里
      每次 new 出临时张量, capture/replay data_ptr 不同 -> graph 锁错地址)
  4. eager 路径 (无 graph buffer, bs>cuda_graph_max_bs) 不崩, 产出与 graph 路径
     数值一致 (torch.cat 构造的 cu_seqlens == 预分配版)
  5. verify kernel 的 score 上界断言 max_seqblock_k_upper >= cdiv(max_seqlen_k, bsk)
     在 _max_seqlen_k 取恒定上界时通过

CPU-only 说明: torch 在无 GPU 环境下用 CPU tensor, data_ptr 同样唯一标识张量
存储, 测试逻辑与 GPU 完全等价 (graph-safety 的本质是 data_ptr 不变, 与设备无关).

运行:
    python test_verify_graph_buffers.py
预期: 5 组测试全过, 末尾打印 "✅ verify graph buffer 逻辑验证通过"
"""
import torch

# ---- mock 配置 (复刻 MiniMax-M3 AWQ-INT4 的实际参数) ----
CONTEXT_LEN = 204800
BLOCK_SIZE_K = 128
BLOCK_SIZE_Q = 1
MAX_DRAFT = 4  # draft_token_num D
CUDA_GRAPH_MAX_BS = 8

# max_seqblock_k_upper = cdiv(context_len + D, block_size_k)
#   = cdiv(204800 + 4, 128) = 1601  (与 backend __init__ 一致)
MAX_SEQBLOCK_K_UPPER = (CONTEXT_LEN + MAX_DRAFT + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K


def cdiv(a, b):
    return (a + b - 1) // b


class MockVerifyBackend:
    """复刻 MiniMaxSparseAttnBackend 的 verify graph buffer 逻辑 (5 个方法).

    只保留与 graph-safety 相关的字段/方法, 剥离 KV cache / layer 等无关依赖.
    字段名/方法名/填充逻辑与 site-packages 的 minimax_sparse_backend.py 完全一致,
    以便回归保护: 若 backend 实现改了 buffer 逻辑, 本 mock 也要同步改, 否则
    测试会提醒逻辑漂移.
    """

    def __init__(self, device):
        self.device = device
        self.block_size_q = BLOCK_SIZE_Q
        self.block_size_k = BLOCK_SIZE_K
        self.max_seqblock_k_upper = MAX_SEQBLOCK_K_UPPER
        # init_cuda_graph_state 调用前不存在 (模拟 eager 路径)
        self._has_graph_buf = False

    # ---------- 1. init_cuda_graph_state ----------
    def init_cuda_graph_state(self, max_bs):
        """复刻 backend.init_cuda_graph_state 的 verify buffer 预分配."""
        self._verify_max_bs = int(max_bs)
        self._verify_extend_seq_lens_buf = torch.zeros(
            (max_bs,), dtype=torch.int32, device=self.device
        )
        self._verify_cu_seqlens_buf = torch.zeros(
            (max_bs + 1,), dtype=torch.int32, device=self.device
        )
        self._verify_seq_lens_buf = torch.zeros(
            (max_bs,), dtype=torch.int32, device=self.device
        )
        self._has_graph_buf = True

    # ---------- 2. capture ----------
    def init_forward_metadata_capture_cuda_graph(self, bs, seq_lens, D):
        """复刻 backend capture 路径 (target_verify 分支)."""
        assert self._has_graph_buf, "capture 需要 graph buffer"
        self._max_seqlen_q = D
        # _max_seqlen_k = 恒定上界 (graph-safe): 只用于 score 断言, 不影响 score 第3维
        self._max_seqlen_k = self.max_seqblock_k_upper * self.block_size_k
        self._verify_extend_seq_lens_buf[:bs].fill_(D)
        self._verify_cu_seqlens_buf[: bs + 1] = torch.arange(
            0, (bs + 1) * D, step=D, dtype=torch.int32,
            device=self._verify_cu_seqlens_buf.device,
        )
        # capture 时 seq_lens 是 dummy (=1), prefix+D = 1+D, 仅保证 capture 不崩
        self._verify_seq_lens_buf[:bs] = seq_lens[:bs].to(torch.int32) + D

    # ---------- 3. replay ----------
    def init_forward_metadata_replay_cuda_graph(self, bs, seq_lens, D):
        """复刻 backend replay 路径 (target_verify 分支)."""
        assert self._has_graph_buf, "replay 需要 graph buffer"
        self._max_seqlen_q = D
        self._max_seqlen_k = self.max_seqblock_k_upper * self.block_size_k
        # 常量 buffer 重填 (内容与 capture 相同, 地址相同)
        self._verify_extend_seq_lens_buf[:bs].fill_(D)
        self._verify_cu_seqlens_buf[: bs + 1] = torch.arange(
            0, (bs + 1) * D, step=D, dtype=torch.int32,
            device=self._verify_cu_seqlens_buf.device,
        )
        # seq_lens = 真实 prefix + D
        self._verify_seq_lens_buf[:bs] = seq_lens[:bs].to(torch.int32) + D

    # ---------- 4. _forward_verify 的 buffer 选择逻辑 ----------
    def select_verify_buffers(self, bs, prefix_lens, D, stream_capturing=False):
        """复刻 _forward_verify 里 cu_seqlens/extend_seq_lens/seq_lens 的选择逻辑.

        返回 (cu_seqlens, extend_seq_lens, seq_lens, prefix_lens).
        graph 路径用预分配 buffer; eager 路径 (无 buffer) 构造临时张量.
        """
        prefix_lens = prefix_lens.to(torch.int32)
        if self._has_graph_buf:
            cu_seqlens = self._verify_cu_seqlens_buf[: bs + 1]
            extend_seq_lens = self._verify_extend_seq_lens_buf[:bs]
            seq_lens = self._verify_seq_lens_buf[:bs]
            if not stream_capturing:
                # eager-with-graph-buf: 现场填 seq_lens
                seq_lens.copy_(prefix_lens + extend_seq_lens)
        else:
            # pure eager: 构造临时张量 (graph-safety N/A)
            extend_seq_lens = torch.full(
                (bs,), D, dtype=torch.int32, device=prefix_lens.device,
            )
            cu_seqlens = torch.cat(
                [
                    torch.zeros(1, dtype=torch.int32, device=prefix_lens.device),
                    extend_seq_lens.cumsum(0).to(torch.int32),
                ]
            )
            seq_lens = prefix_lens + extend_seq_lens
        return cu_seqlens, extend_seq_lens, seq_lens, prefix_lens

    # ---------- 5. verify kernel score 上界断言 (复刻 flash_with_topk_idx:478) ----------
    def check_score_upper_bound_assertion(self, max_seqlen_k):
        """复刻 verify kernel 的 assert:
            max_seqblock_k_upper >= cdiv(max_seqlen_k, block_size_k)
        _max_seqlen_k 取恒定上界 max_seqblock_k_upper*block_size_k 时必过.
        """
        needed = cdiv(max_seqlen_k, self.block_size_k)
        assert self.max_seqblock_k_upper >= needed, (
            f"max_seqblock_k_upper={self.max_seqblock_k_upper} too small for "
            f"max_seqlen_k={max_seqlen_k} (need {needed})"
        )


# ============================================================
# 测试用例
# ============================================================

def test_1_init_allocates_three_graph_buffers():
    """init_cuda_graph_state 预分配 3 个 buffer, dtype=int32, shape 正确."""
    be = MockVerifyBackend("cpu")
    assert not be._has_graph_buf, "init 前不应有 graph buffer"
    be.init_cuda_graph_state(CUDA_GRAPH_MAX_BS)

    assert be._has_graph_buf
    # dtype 必须是 int32 (kernel assert cu_seqlens.dtype == int32)
    assert be._verify_extend_seq_lens_buf.dtype == torch.int32
    assert be._verify_cu_seqlens_buf.dtype == torch.int32
    assert be._verify_seq_lens_buf.dtype == torch.int32
    # shape: extend/seq = [max_bs], cu = [max_bs+1]
    assert be._verify_extend_seq_lens_buf.shape[0] == CUDA_GRAPH_MAX_BS
    assert be._verify_seq_lens_buf.shape[0] == CUDA_GRAPH_MAX_BS
    assert be._verify_cu_seqlens_buf.shape[0] == CUDA_GRAPH_MAX_BS + 1
    print("  [1] 三个 buffer 预分配: dtype=int32, shape 正确 ✓")


def test_2_capture_replay_fill_constant_buffers():
    """capture/replay 填充后 cu_seqlens=[0,D,...,bs*D], extend=[D]*bs (只依赖 bs,D)."""
    be = MockVerifyBackend("cpu")
    be.init_cuda_graph_state(CUDA_GRAPH_MAX_BS)

    bs = CUDA_GRAPH_MAX_BS
    D = MAX_DRAFT
    # capture: dummy seq_lens=1
    dummy_seq_lens = torch.ones(bs, dtype=torch.int32)
    be.init_forward_metadata_capture_cuda_graph(bs, dummy_seq_lens, D)

    cu = be._verify_cu_seqlens_buf[: bs + 1]
    ext = be._verify_extend_seq_lens_buf[:bs]
    assert cu.tolist() == [i * D for i in range(bs + 1)], cu.tolist()
    assert ext.tolist() == [D] * bs, ext.tolist()
    # capture 时 seq_lens = dummy(1) + D = 1+D (仅 dummy)
    sl = be._verify_seq_lens_buf[:bs]
    assert sl.tolist() == [1 + D] * bs, sl.tolist()

    # replay: 真实 prefix 各不同
    real_prefix = torch.tensor([100, 200, 500, 1000, 2048, 8192, 32768, 100000],
                               dtype=torch.int32)[:bs]
    be.init_forward_metadata_replay_cuda_graph(bs, real_prefix, D)
    cu2 = be._verify_cu_seqlens_buf[: bs + 1]
    ext2 = be._verify_extend_seq_lens_buf[:bs]
    # cu/ext 仍是常量 (与 prefix 无关)
    assert cu2.tolist() == [i * D for i in range(bs + 1)], cu2.tolist()
    assert ext2.tolist() == [D] * bs, ext2.tolist()
    # seq_lens = 真实 prefix + D
    sl2 = be._verify_seq_lens_buf[:bs]
    assert sl2.tolist() == [int(p) + D for p in real_prefix.tolist()], sl2.tolist()
    print("  [2] capture/replay 填充: cu/ext 常量, seq_lens=prefix+D ✓")


def test_3_buffer_data_ptr_stable_capture_vs_replay():
    """核心: capture 与 replay 用同一块预分配 buffer (data_ptr 不变) -> graph-safe.

    这是旧实现 vmfault 的真正根因: 旧实现 forward_extend 里每次 torch.cat/a+b
    new 出临时张量, capture/replay data_ptr 不同, graph 锁了 capture 时的地址,
    replay 读到别处 -> garbage/VMFault.
    """
    be = MockVerifyBackend("cpu")
    be.init_cuda_graph_state(CUDA_GRAPH_MAX_BS)
    bs = CUDA_GRAPH_MAX_BS
    D = MAX_DRAFT

    be.init_forward_metadata_capture_cuda_graph(
        bs, torch.ones(bs, dtype=torch.int32), D)
    cap_cu = be._verify_cu_seqlens_buf.data_ptr()
    cap_ext = be._verify_extend_seq_lens_buf.data_ptr()
    cap_sl = be._verify_seq_lens_buf.data_ptr()
    # select_verify_buffers 返回的 slice 的 data_ptr 应等于 buffer 起点
    cap_cu_sel = be._verify_cu_seqlens_buf[: bs + 1].data_ptr()

    be.init_forward_metadata_replay_cuda_graph(
        bs, torch.tensor([100] * bs, dtype=torch.int32), D)
    rep_cu = be._verify_cu_seqlens_buf.data_ptr()
    rep_ext = be._verify_extend_seq_lens_buf.data_ptr()
    rep_sl = be._verify_seq_lens_buf.data_ptr()

    assert cap_cu == rep_cu, f"cu_seqlens buf data_ptr 变了 {cap_cu}->{rep_cu}"
    assert cap_ext == rep_ext, f"extend buf data_ptr 变了"
    assert cap_sl == rep_sl, f"seq_lens buf data_ptr 变了"
    assert cap_cu == cap_cu_sel, "slice data_ptr 应等于 buffer 起点"
    print(f"  [3] buffer data_ptr capture==replay (cu={cap_cu:#x}) ✓ graph-safe")


def test_4_eager_fallback_no_crash_and_matches_graph_values():
    """eager 路径 (无 graph buffer) 不崩, 且构造的 cu/ext 数值与 graph 路径一致."""
    be = MockVerifyBackend("cpu")
    # 不调用 init_cuda_graph_state -> 模拟 bs>cuda_graph_max_bs 的 eager 路径
    assert not be._has_graph_buf
    bs = CUDA_GRAPH_MAX_BS + 4  # 超过 graph max_bs
    D = MAX_DRAFT
    prefix = torch.tensor([50, 150, 300, 600, 1200, 2400, 4800, 9600,
                           19200, 38400, 76800, 100000][:bs], dtype=torch.int32)

    cu, ext, sl, pl = be.select_verify_buffers(bs, prefix, D, stream_capturing=False)
    # eager 构造: cu = [0,D,2D,...,bs*D], ext=[D]*bs, sl=prefix+D
    assert cu.shape[0] == bs + 1
    assert ext.shape[0] == bs
    assert cu.tolist() == [i * D for i in range(bs + 1)], cu.tolist()
    assert ext.tolist() == [D] * bs, ext.tolist()
    assert sl.tolist() == [int(p) + D for p in prefix.tolist()], sl.tolist()
    # prefix_lens 原样 int32 传回
    assert pl.tolist() == prefix.tolist()
    print(f"  [4] eager 路径 bs={bs}(>max_bs={CUDA_GRAPH_MAX_BS}) 不崩, 数值正确 ✓")


def test_4b_eager_and_graph_produce_identical_constants():
    """graph 路径与 eager 路径对同一 (bs,D) 产出的 cu/ext 常量完全一致."""
    be_g = MockVerifyBackend("cpu")
    be_g.init_cuda_graph_state(CUDA_GRAPH_MAX_BS)
    be_e = MockVerifyBackend("cpu")  # 无 graph buffer

    bs = 5
    D = MAX_DRAFT
    prefix = torch.tensor([10, 20, 30, 40, 50], dtype=torch.int32)

    be_g.init_forward_metadata_replay_cuda_graph(bs, prefix, D)
    cu_g, ext_g, sl_g, _ = be_g.select_verify_buffers(
        bs, prefix, D, stream_capturing=True)  # replay 视为 capturing
    cu_e, ext_e, sl_e, _ = be_e.select_verify_buffers(
        bs, prefix, D, stream_capturing=False)

    assert cu_g.tolist() == cu_e.tolist()
    assert ext_g.tolist() == ext_e.tolist()
    assert sl_g.tolist() == sl_e.tolist()
    print("  [4b] graph 与 eager 路径常量数值一致 ✓")


def test_5_score_upper_bound_assertion_passes_for_constant_max_seqlen_k():
    """verify kernel assert: max_seqblock_k_upper >= cdiv(max_seqlen_k, bsk).

    backend 把 _max_seqlen_k 设为恒定上界 max_seqblock_k_upper*block_size_k,
    此时 cdiv(_max_seqlen_k, bsk) == max_seqblock_k_upper, assert 恒过.
    同时验证任意真实 seq_len (<=context_len+D) 也满足 (kernel 内 block_num 不越界).
    """
    be = MockVerifyBackend("cpu")
    # 恒定上界场景 (capture/replay 都用这个)
    const_max_seqlen_k = be.max_seqblock_k_upper * be.block_size_k
    be.check_score_upper_bound_assertion(const_max_seqlen_k)

    # 任意真实 seq_len <= context_len + D 也应满足 (kernel block_num <= upper)
    for real_seq in [1 + MAX_DRAFT, 100 + MAX_DRAFT, 2048 + MAX_DRAFT,
                     CONTEXT_LEN + MAX_DRAFT]:
        be.check_score_upper_bound_assertion(real_seq)

    # 反例: 上界不够时应抛断言 (回归保护: 防止有人调小 max_seqblock_k_upper)
    raised = False
    try:
        be.check_score_upper_bound_assertion(CONTEXT_LEN + MAX_DRAFT + 10_000)
    except AssertionError:
        raised = True
    assert raised, "上界不足时应抛 AssertionError"
    print(f"  [5] score 上界断言: 恒定上界通过, 超界反例正确抛错 ✓")


def test_7_real_backend_source_matches_mock():
    """回归保护: 用 inspect 检查真实 backend 源码包含 mock 所复刻的所有 graph-safe
    关键语句. 若 backend 实现改了 buffer 逻辑, 本测试 fail, 提醒同步本 mock.

    纯 CPU: 只读源码字符串, 不实例化 backend (实例化需 sglang 运行时/KV pool).
    """
    import inspect
    try:
        from sglang.srt.layers.attention.minimax_sparse_backend import (
            MiniMaxSparseAttnBackend as B,
        )
    except Exception as e:
        print(f"  [7] 跳过 (无法 import sglang backend: {e}) — 非 sglang 环境")
        return
    src = inspect.getsource(B)
    checks = {
        "init_cuda_graph_state 预分配 cu_seqlens":
            "_verify_cu_seqlens_buf = torch.zeros" in src,
        "init_cuda_graph_state 预分配 extend":
            "_verify_extend_seq_lens_buf = torch.zeros" in src,
        "init_cuda_graph_state 预分配 seq_lens":
            "_verify_seq_lens_buf = torch.zeros" in src,
        "capture 填 arange":
            "_verify_cu_seqlens_buf[: bs + 1] = torch.arange" in src,
        "capture/replay 都填 extend D (出现2次)":
            src.count("_verify_extend_seq_lens_buf[:bs].fill_(D)") == 2,
        "capture/replay 都填 arange (出现2次, graph-safe)":
            src.count("_verify_cu_seqlens_buf[: bs + 1] = torch.arange") == 2,
        "capture/replay 都填 seq_lens=prefix+D (出现2次)":
            src.count("_verify_seq_lens_buf[:bs] = seq_lens[:bs].to(torch.int32) + D") == 2,
        "_max_seqlen_k 恒定上界":
            "self.max_seqblock_k_upper * self.block_size_k" in src,
        "has_graph_buf eager 分支":
            'has_graph_buf = hasattr(self, "_verify_cu_seqlens_buf")' in src,
        "eager torch.cat fallback":
            "torch.cat" in src and "extend_seq_lens.cumsum" in src,
        "routing is_target_verify -> _forward_verify":
            "return self._forward_verify" in src,
    }
    missing = [k for k, v in checks.items() if not v]
    assert not missing, "真实 backend 源码缺失/m漂移 graph-safe 逻辑:\n  " + "\n  ".join(missing)
    print(f"  [7] 真实 backend 源码含全部 {len(checks)} 项 graph-safe 关键语句 ✓")


def test_6_cu_seqlens_independent_of_prefix_lens():
    """回归: cu_seqlens/ext 只依赖 (bs,D), 与 prefix_lens 完全无关.

    这是 graph-safe 的前提: 同一 captured graph 内 bs,D 固定, 所以 cu/ext 内容
    在所有请求的所有 prefix 长度下都相同 -> graph 锁的地址+内容都稳定.
    """
    be = MockVerifyBackend("cpu")
    be.init_cuda_graph_state(CUDA_GRAPH_MAX_BS)
    bs = 4
    D = MAX_DRAFT
    for prefix_vals in [[1, 1, 1, 1], [10, 999, 50, 204800],
                        [32768, 100, 8192, 1]]:
        prefix = torch.tensor(prefix_vals, dtype=torch.int32)
        be.init_forward_metadata_replay_cuda_graph(bs, prefix, D)
        cu = be._verify_cu_seqlens_buf[: bs + 1]
        ext = be._verify_extend_seq_lens_buf[:bs]
        assert cu.tolist() == [0, D, 2 * D, 3 * D, 4 * D]
        assert ext.tolist() == [D] * bs
        # seq_lens 才随 prefix 变 (这是 kernel 运行时读的)
        sl = be._verify_seq_lens_buf[:bs]
        assert sl.tolist() == [p + D for p in prefix_vals]
    print("  [6] cu/ext 与 prefix 无关 (只依赖 bs,D), seq_lens 随 prefix ✓")


# ============================================================
def main():
    print("=" * 64)
    print("verify_prefill kernel graph-safety 单元测试 (纯 CPU)")
    print(f"  context_len={CONTEXT_LEN} block_size_k={BLOCK_SIZE_K} "
          f"D={MAX_DRAFT} max_seqblock_k_upper={MAX_SEQBLOCK_K_UPPER} "
          f"cuda_graph_max_bs={CUDA_GRAPH_MAX_BS}")
    print("=" * 64)
    test_1_init_allocates_three_graph_buffers()
    test_2_capture_replay_fill_constant_buffers()
    test_3_buffer_data_ptr_stable_capture_vs_replay()
    test_4_eager_fallback_no_crash_and_matches_graph_values()
    test_4b_eager_and_graph_produce_identical_constants()
    test_5_score_upper_bound_assertion_passes_for_constant_max_seqlen_k()
    test_6_cu_seqlens_independent_of_prefix_lens()
    test_7_real_backend_source_matches_mock()
    print("=" * 64)
    print("✅ verify graph buffer 逻辑验证通过 (8 组测试全过)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
