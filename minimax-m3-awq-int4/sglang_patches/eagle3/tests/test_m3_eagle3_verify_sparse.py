#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""最小测试: 验证 EAGLE3 TARGET_VERIFY 路径在 M3 sparse backend 上的字段补全逻辑.

不启动 sglang 服务, 不加载 225G 模型, 不做真实 attention 计算.
只验证我们 patch 的两个关键修复在 verify batch (extend 字段全 None) 下不崩:

1. minimax_m3_vl.py: VL 类有 set_eagle3_layers_to_capture / get_embed_and_head,
   set_eagle3_layers_to_capture 能正确标记 layer._is_layer_to_capture 且不崩
   (坑 A: self.config.text_config.num_hidden_layers).
2. minimax_sparse_backend.py: init_forward_metadata 在 TARGET_VERIFY (extend 字段
   全 None, 模拟 ForwardBatch.init_new decode 分支产物) 下, 能用 spec_info.draft_token_num
   补全 extend_seq_lens / extend_seq_lens_cpu / extend_prefix_lens / extend_prefix_lens_cpu,
   且 _max_seqlen_q 正确 (坑 B/C: max(None) / .device 崩).

跑法:
    cd /models/quant-eagle3-hygon/sglang_patches/tests
    python3 test_m3_eagle3_verify_sparse.py
"""

import sys
import types
from dataclasses import dataclass

import torch


def ok(msg):
    print(f"  ✅ {msg}")


def fail(msg):
    print(f"  ❌ {msg}")
    raise SystemExit(1)


# --------------------------------------------------------------------------
# 测试 1: VL 类接口 + set_eagle3_layers_to_capture 不崩
# --------------------------------------------------------------------------
def test_vl_class_eagle3_interface():
    print("[测试 1] M3 VL 类 EAGLE3 接口")
    from sglang.srt.models.minimax_m3_vl import (
        MiniMaxM3SparseForConditionalGeneration as VL,
    )

    if not hasattr(VL, "set_eagle3_layers_to_capture"):
        fail("VL 类缺 set_eagle3_layers_to_capture (patch 没生效?)")
    if not hasattr(VL, "get_embed_and_head"):
        fail("VL 类缺 get_embed_and_head")
    ok("VL 类有 set_eagle3_layers_to_capture / get_embed_and_head")

    # 构造 mock self (绕过 nn.Module.__init__ 的限制, 用 namespace)
    class FakePP:
        def is_last_rank(self):
            return True

    class FakeLayer(torch.nn.Module):
        pass

    class FakeModel(torch.nn.Module):
        def __init__(self, num_layers):
            super().__init__()
            self.layers = torch.nn.ModuleList([FakeLayer() for _ in range(num_layers)])

    class TextConfig:
        num_hidden_layers = 60

    class VLConfig:
        text_config = TextConfig()

    ns = types.SimpleNamespace()
    ns.pp_group = FakePP()
    ns.model = FakeModel(60)
    # lm_head / embed_tokens 用占位, get_embed_and_head 只读 .weight
    ns.lm_head = types.SimpleNamespace(weight="HEAD_W")
    ns.model.embed_tokens = types.SimpleNamespace(weight="EMBED_W")
    ns.is_mrope_enabled = False
    ns.config = VLConfig()

    # 调 set_eagle3_layers_to_capture(None) — 坑 A 就崩在这里
    try:
        VL.set_eagle3_layers_to_capture(ns)
    except AttributeError as e:
        fail(f"set_eagle3_layers_to_capture 崩 (坑 A 未修?): {e}")

    expected = [2, 30, 57]  # [2, N//2, N-3] for N=60
    if ns.model.layers_to_capture != expected:
        fail(f"layers_to_capture={ns.model.layers_to_capture}, 期望 {expected}")
    ok(f"layers_to_capture = {expected}")

    for lid in expected:
        if not getattr(ns.model.layers[lid], "_is_layer_to_capture", False):
            fail(f"layer {lid} 未被标记 _is_layer_to_capture")
    if getattr(ns.model.layers[0], "_is_layer_to_capture", False):
        fail("layer 0 不该被标记")
    ok("layer 2/30/57 标记 _is_layer_to_capture=True, 其他 False")

    if not ns.capture_aux_hidden_states:
        fail("capture_aux_hidden_states 没设 True")
    ok("capture_aux_hidden_states = True")

    embed, head = VL.get_embed_and_head(ns)
    if embed != "EMBED_W" or head != "HEAD_W":
        fail(f"get_embed_and_head 返回错: {embed}, {head}")
    ok("get_embed_and_head 返回 (embed_tokens.weight, lm_head.weight)")


# --------------------------------------------------------------------------
# 测试 2: sparse backend init_forward_metadata 在 verify batch (extend 全 None) 下补全
# --------------------------------------------------------------------------
@dataclass
class FakeSpecInfo:
    """模拟 EagleVerifyInput 的 spec_info."""
    draft_token_num: int


class FakeForwardBatch:
    """模拟 ForwardBatch.init_new 对 TARGET_VERIFY 的产物:
    forward_mode=TARGET_VERIFY, extend_* 字段全 None (decode 分支不填)."""


def test_sparse_backend_verify_field_materialization():
    print("[测试 2] sparse backend TARGET_VERIFY 字段补全 (坑 B/C)")
    from sglang.srt.layers.attention.minimax_sparse_backend import (
        MiniMaxSparseAttnBackend,
    )
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    # 构造一个 backend 实例 — 只用 init_forward_metadata, 不需要完整 __init__
    backend = MiniMaxSparseAttnBackend.__new__(MiniMaxSparseAttnBackend)
    backend._max_seqlen_q = 1
    backend._max_seqlen_k = 1

    num_reqs = 4
    draft_token_num = 4  # = num_steps(3) + 1, topk=1
    # verify 时 seq_lens 是 PREFIX (不含 draft) — prepare_for_v2_verify 不加 draft,
    # scheduler 在 verify 后用 seq_lens + accept_lens 更新。
    prefix_lens = [10, 20, 5, 33]
    seq_lens = prefix_lens  # 注意: verify 时 seq_lens 就是 prefix, 不含 draft

    batch = FakeForwardBatch()
    batch.forward_mode = ForwardMode.TARGET_VERIFY
    batch.spec_info = FakeSpecInfo(draft_token_num=draft_token_num)
    batch.seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device="cpu")
    batch.seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int32, device="cpu")
    batch.input_ids = torch.zeros(num_reqs * draft_token_num, dtype=torch.int32)
    # 模拟 init_new decode 分支: extend 字段全 None
    batch.extend_seq_lens = None
    batch.extend_seq_lens_cpu = None
    batch.extend_prefix_lens = None
    batch.extend_prefix_lens_cpu = None

    try:
        backend.init_forward_metadata(batch)
    except Exception as e:
        fail(f"init_forward_metadata 崩 (坑 B/C 未修?): {type(e).__name__}: {e}")

    # 验证补全的字段
    if batch.extend_seq_lens is None:
        fail("extend_seq_lens 没补全")
    if tuple(batch.extend_seq_lens.shape) != (num_reqs,):
        fail(f"extend_seq_lens shape={batch.extend_seq_lens.shape}, 期望 ({num_reqs},)")
    if not torch.all(batch.extend_seq_lens == draft_token_num):
        fail(f"extend_seq_lens 值={batch.extend_seq_lens}, 期望全 {draft_token_num}")
    ok(f"extend_seq_lens 补全为 [{draft_token_num}] * {num_reqs} (tensor)")

    if batch.extend_seq_lens_cpu != [draft_token_num] * num_reqs:
        fail(f"extend_seq_lens_cpu={batch.extend_seq_lens_cpu}")
    ok(f"extend_seq_lens_cpu 补全为 [{draft_token_num}] * {num_reqs} (list)")

    if batch.extend_prefix_lens is None:
        fail("extend_prefix_lens 没补全")
    # verify 时 seq_lens 就是 prefix, extend_prefix_lens 应 = seq_lens (不减 draft)
    exp_prefix = torch.tensor(prefix_lens, dtype=torch.int32)
    if not torch.equal(batch.extend_prefix_lens, exp_prefix):
        fail(f"extend_prefix_lens={batch.extend_prefix_lens}, 期望 {exp_prefix} (= seq_lens, 不减 draft)")
    ok(f"extend_prefix_lens = seq_lens = {prefix_lens} (verify 时 seq_lens 即 prefix, 不减 draft)")

    if backend._max_seqlen_q != draft_token_num:
        fail(f"_max_seqlen_q={backend._max_seqlen_q}, 期望 {draft_token_num}")
    ok(f"_max_seqlen_q = {draft_token_num} (= draft_token_num)")

    # K 长度 = prefix + draft
    exp_max_k = max(p + draft_token_num for p in prefix_lens)
    if backend._max_seqlen_k != exp_max_k:
        fail(f"_max_seqlen_k={backend._max_seqlen_k}, 期望 {exp_max_k} (prefix+draft)")
    ok(f"_max_seqlen_k = {exp_max_k} (= max(prefix + draft))")

    # 关键: 模拟 forward_extend 取 extend_seq_lens.device — 坑 C 就崩在这
    try:
        _ = batch.extend_seq_lens.device
        _ = batch.extend_seq_lens.to(torch.int32).cumsum(0).to(torch.int32)
    except AttributeError as e:
        fail(f"forward_extend 会崩 .device (坑 C 未修?): {e}")
    ok("forward_extend 取 extend_seq_lens.device / cumsum 不再崩 (坑 C 已修)")

    # 模拟 cu_seqlens 构造 (forward_extend line 213-219)
    cu_seqlens = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=batch.extend_seq_lens.device),
            batch.extend_seq_lens.to(torch.int32).cumsum(0).to(torch.int32),
        ]
    )
    if int(cu_seqlens[-1].item()) != num_reqs * draft_token_num:
        fail(f"cu_seqlens[-1]={cu_seqlens[-1]}, 期望 {num_reqs * draft_token_num}")
    ok(f"cu_seqlens 构造正确, 总 token = {num_reqs * draft_token_num}")


# --------------------------------------------------------------------------
# 测试 3: 正常 extend (非 verify) 不受影响 — 回归保护
# --------------------------------------------------------------------------
def test_normal_extend_unchanged():
    print("[测试 3] 正常 extend (非 verify) 不受 patch 影响 (回归)")
    from sglang.srt.layers.attention.minimax_sparse_backend import (
        MiniMaxSparseAttnBackend,
    )
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    backend = MiniMaxSparseAttnBackend.__new__(MiniMaxSparseAttnBackend)
    backend._max_seqlen_q = 1
    backend._max_seqlen_k = 1

    # 正常 prefill: extend_seq_lens_cpu 有值
    batch = FakeForwardBatch()
    batch.forward_mode = ForwardMode.EXTEND
    batch.spec_info = None
    batch.seq_lens = torch.tensor([15, 25], dtype=torch.int32)
    batch.seq_lens_cpu = torch.tensor([15, 25], dtype=torch.int32)
    batch.input_ids = torch.zeros(20, dtype=torch.int32)
    batch.extend_seq_lens = torch.tensor([5, 10], dtype=torch.int32)
    batch.extend_seq_lens_cpu = [5, 10]
    batch.extend_prefix_lens = torch.tensor([10, 15], dtype=torch.int32)
    batch.extend_prefix_lens_cpu = [10, 15]

    backend.init_forward_metadata(batch)
    if backend._max_seqlen_q != 10:
        fail(f"正常 extend _max_seqlen_q={backend._max_seqlen_q}, 期望 10 (max of [5,10])")
    ok(f"正常 extend _max_seqlen_q = 10 (max([5,10])), 未走兜底")


if __name__ == "__main__":
    print("=" * 60)
    print("M3 EAGLE3 verify 路径 patch 测试 (不启动服务)")
    print("=" * 60)
    test_vl_class_eagle3_interface()
    test_sparse_backend_verify_field_materialization()
    test_normal_extend_unchanged()
    print("=" * 60)
    print("🎉 全部通过 — patch 逻辑 OK, 可起服务实测 verify 阶段")
    print("=" * 60)
