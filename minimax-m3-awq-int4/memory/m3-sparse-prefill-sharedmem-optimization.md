---
name: m3-sparse-prefill-sharedmem-optimization
description: "DCU sparse prefill kernel OutOfResources(shared mem 69632>65536)根因+num_stages=1临时修复;待跑通后评估性能影响并更新patch"
metadata:
  node_type: memory
  type: project
  originSessionId: 2026-08-04-sharedmem
  modified: 2026-08-04T13:30:00.000Z
---

## 现象

`start_eagle3.sh` 启动 sglang serve 时, CUDA graph capture 阶段(`init_device_graphs` → `capture`, bs=16)抛:

```
triton.runtime.errors.OutOfResources: out of resource: shared memory,
Required: 69632, Hardware limit: 65536. Reducing block sizes or `num_stages` may help.
```

栈底: `_gqa_share_sparse_fwd_kernel` (minimax sparse prefill kernel), 在 EAGLE3 verify/prefill 路径被 capture.

## 环境

- 硬件: 8× K100AI DCU (gfx928, compute cap 9.2), HIP/DTK 后端(torch 2.9.0), 非 NVIDIA CUDA
- 模型: MiniMax-M3-AWQ-INT4, TP=8, bf16, attention-backend triton, EAGLE3 draft=3 steps
- 关键: DCU 这代 shared memory 硬上限 = 64KB (65536B), 报错里的 Hardware limit 即此. **不像 NV 卡可以 cudaFuncSetAttribute opt-in 到 96/164KB**, DCU 上基本是架构固定上限, 没法申请更多.

## 根因(已测算确认)

kernel: `sglang/srt/layers/attention/minimax_sparse_ops/prefill/topk_sparse.py` 的 `_gqa_share_sparse_fwd_kernel`, 原 autotune config:

```python
configs=[
    triton.Config({}, num_warps=4, num_stages=2),
    triton.Config({}, num_warps=4, num_stages=3),
    triton.Config({}, num_warps=8, num_stages=2),
    triton.Config({}, num_warps=8, num_stages=3),
]
```

**最小档 num_stages=2 就已经超限**, 所以 4 个 config 没一个能过, autotune 必抛 OutOfResources.

### shared memory 测算

模型参数(MiniMax-M3-AWQ-INT4/text_config):
- head_dim=128, num_attention_heads=64, num_key_value_heads=4 → gqa_group_size=16
- sparse_block_size=128, sparse_topk_blocks=16
- block_size_q=1(固定), block_size_k=128 → BLOCK_SIZE_Q=1, BLOCK_SIZE_K=128
- dtype=bf16 (2B)

heuristics 推导 tile:
- q = [QH=16, KD=128] = 16*128*2 = **4096B**
- k = [KD=128, K_eff] bf16
- v = [K_eff, VD=128] bf16

公式: `shared = q + num_stages * (k + v)`

**关键反推**: 报错值 69632 = 4096 + 2*(16384+16384) = 4096 + 65536 = 69632, 精确吻合 **K_eff=64, num_stages=2**. 即 autotuner 选到最小档 ns=2 时实际 K tile 是 64(非满 128), 占用 69632, 超 4096B.

| num_stages | shared (K_eff=64) | vs 65536 |
|---|---|---|
| 1 | 36864B | ✅ 安全 |
| 2 | **69632B** | ❌ 超 4096(=报错值) |
| 3 | 102400B | ❌ 超 36864 |

## 临时修复(已 apply, 待验证)

topk_sparse.py autotune config 改为只保留 num_stages=1:

```python
configs=[
    triton.Config({}, num_warps=4, num_stages=1),
    triton.Config({}, num_warps=8, num_stages=1),
]
```

占用 36864B, 远低于 64KB. **已 DCU 实测**: 离线构造 M3 维度输入(head_dim=128, kv_heads=4, gqa=16, bs=2, total_q=8)跑通, 输出 shape (8,64,128) 正确, isfinite=True, 无 OutOfResources.

## 性能影响(待跑通后实测, 当前为判断)

**会影响 prefill 吞吐, 但对 EAGLE3 整体影响有限**:
- num_stages 1→2 损失的是 k/v 访存与 tl.dot 计算的 overlap. 但此 kernel 是 sparse + paged gather(非连续大块 FA), 且 BLOCK_SIZE_Q=1 计算量小, pipeline 能掩盖的延迟比例本就不大.
- 这是 **prefill kernel**, EAGLE3 高频热路径是 draft/verify 的 decode, prefill 频率低, 影响次线性.
- 经验估计 prefill 慢 10–25%, 具体看 CONTEXT_LEN.

**为何不选更优解(降 K tile 保 ns=2)**: ns=2 要 ≤65536 需 k+v≤30720, 得把 K_eff 从 64 降到 32(k=v=8192, shared=36864). 但 BLOCK_SIZE_K 和 sparse_block_size=128 绑定, topk_idx 选的是 128-token 块, 改 K tile 破坏 sparse 语义 → 正确性风险, 不碰.

## 后续动作(待跑通后)

1. [ ] 清 Triton 缓存重启: `rm -rf ~/.triton/cache && bash start_eagle3.sh`
2. [ ] 跑通后, 用代表性长上下文请求测 prefill tokens/s, 对比预期
3. [ ] 若 prefill 慢到不可接受 → 评估给 kernel 加运行时分支(按上下文长度/卡型选 config)
4. [ ] 确认无误后, 把此临时修复正式合入 `sglang_patches/` 的 patch(当前只改了 site-packages, 未进 patch 体系)

## 相关

- [[m3-sglang-verify-vmfault-rootcause]] — 同一 sparse prefill 路径在 graph capture 下的另一类问题(VMFault)
- [[m3-verify-graph-temp-tensor-rootcause]] — verify graph temp tensor 根因, 同为 graph capture 阶段暴露
