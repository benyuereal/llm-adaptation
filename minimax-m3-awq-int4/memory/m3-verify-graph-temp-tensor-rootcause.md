---
name: m3-verify-graph-temp-tensor-rootcause
description: "EAGLE3 verify bs>=并发阈值 VMFault 根因: forward_extend 用临时 extend_seq_lens 算 cu_seqlens, graph replay 读失效地址"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2b4cf476-1e6e-4434-abc0-b3a2c3a36a95
  modified: 2026-08-04T02:25:00.376Z
---

EAGLE3 verify 在 cuda graph replay 时 bs=16 VMFault(单请求 bs=1 不崩)的真正根因, 不同于 [[m3-sglang-verify-vmfault-rootcause]] 里的 score buffer 动态尺寸问题(那个由 Strategy B 固定上界 max_seqblock_k_upper 解决了).

**根因**: minimax_sparse_backend.forward_extend 的 verify 分支用 `forward_batch.extend_seq_lens`(临时 tensor)算 cu_seqlens/seq_lens:
- verify 时 `ForwardBatch.init_new`(forward_batch_info.py:561 is_target_verify 分支)**不设 extend_seq_lens**(保持 None)
- 旧代码在 forward_extend 里 `torch.full((bs,), D)` 临时新建 extend_seq_lens
- capture 的 forward_batch 是局部变量(cuda_graph_runner.py:1072), 捕获后被 GC → extend_seq_lens 内存释放/复用
- graph 记录了 `cu_seqlens = torch.cat([zeros, extend_seq_lens.cumsum(0)])` 和 `seq_lens = raw_seq_lens + extend_seq_lens` 这些依赖 extend_seq_lens 地址的 op
- replay 时 graph 读 extend_seq_lens 老地址 → 内存已复用(垃圾值) → cu_seqlens/seq_lens 错 → kernel 越界 → VMFault
- bs=1 临时 tensor 小, 碰巧内存没被复用, 不炸; bs=16 必崩

**为什么标准 triton backend 不崩**: 它的 `qo_indptr = torch.arange(0, (bs+1)*D, D)`(triton_backend.py:363)用 Python int 常量, 不依赖临时 tensor; kv_indptr 写入预分配 graph buffer.

**修复**(minimax_sparse_backend.py forward_extend verify 分支): 完全绕过 extend_seq_lens, 改用
- `prefix_lens = forward_batch.seq_lens`(graph buffer, replay_prepare 填充, 地址稳定)
- `seq_lens = forward_batch.seq_lens + D`(graph buffer + Python int, graph-safe)
- `cu_seqlens = torch.arange(0, (bs+1)*D, D)`(Python int 输入, 同 triton qo_indptr 模式, graph-safe)
- `__init__` 存 `self.num_draft_tokens = server_args.speculative_num_draft_tokens`

**验证**: /workspace/verify/test_verify_graph_buffer.py 复现旧实现 bug(extend_seq_lens 复用成 999 → seq_lens=500+999=1499 错) + 验证新实现 graph-safe(replay seq_lens=504 正确). 离线 4 类测试(精度/越界/性能)仍 PASS.

**关键认知**: cuda graph 捕获 tensor 地址, forward_extend 里任何 `torch.full/cat/+` 新建的临时 tensor, capture 后若被 GC, replay 时 graph 读老地址失效. verify 进 graph(target_verify 有专门 graph bucket), 必须只用 graph buffer + Python 常量. [[m3-sglang-verify-vmfault-rootcause]] 里的 score buffer 问题是真的, 但这个 temp tensor 问题独立且是 bs>=阈值崩的真正原因.
