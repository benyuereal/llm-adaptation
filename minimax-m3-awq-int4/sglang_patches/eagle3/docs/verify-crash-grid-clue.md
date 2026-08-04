# bs=16 崩溃新线索: grid 异常 + 诚实结论 (2026-08-04)

## 离线测试结论

`test_verify_old_vs_new_graph.py` 证实: **旧代码(临时 extend_seq_lens)在离线 graph 下也 PASS**
(diff=0)。所以 "临时 tensor 被 GC" 的假设是错的 —— graph pool 保护地址,`torch.full` replay
重放填值。**我的 graph-safety 修复(绕过 extend_seq_lens)没有真正修复端到端崩溃**,只是写法
更稳妥(对齐 triton backend 模式)。

## 崩溃日志新线索: grid 异常

崩溃 kernel 的 grid(从 ROCm HOSTQUEUE AQL dump):

| kernel | 崩溃 grid | verify bs=16 应有 grid |
|--------|----------|----------------------|
| `_flash_attn_fwd_with_block_score_kernel_verify` | `(256,1,1)` / `(512,16,1)` | `(1, 128, 1)` |
| `_topk_index_kernel_verify` | `(1024,1,16)` | `(4, 16, 8)` |
| `_gqa_share_sparse_fwd_kernel_verify` | `(12288,2,1)` | `(4, 1, 16)` |

**Step1 block_score grid (256,1,1)**: `grid.x = cdiv(max_seqlen_q, BLOCK_SIZE_Q=64)`。
若 grid.x=256 → max_seqlen_q=16384。**16384 = sglang max_prefill_tokens**!
verify 的 max_seqlen_q 应 = D = 4 → grid.x 应 = 1。

**但 probe A 显示 replay 时 `max_seqlen_q=4`**。矛盾。

## 可能解释

1. **grid 是 capture 时算的, capture 时 max_seqlen_q 可能不是 4**
   - capture 走 `init_forward_metadata_capture_cuda_graph`, target_verify 分支设 `_max_seqlen_q=D=4`
   - 但若 capture 时 `forward_mode.is_target_verify()` 为 False (走 else), `_max_seqlen_q=1`
   - 16384 既不是 4 也不是 1, 来源待查

2. **grid 大不直接导致越界**: kernel 内 `if BLOCK_SIZE_Q * pid_q >= q_len: return`
   (q_len=cu_seqlens 算出=4), 多余 program 立即 return。所以 grid (256,1,1) 本身不越界。
   **VMFault 在 kernel 内部某个 tl.store/tl.load, 不是 grid 大导致**。

3. **grid 线索可能误导**: HOSTQUEUE dump 的 grid 可能含 capture 时其他 layer 的 launch,
   或 ROCm AQL grid 维度映射与 triton 不完全一致。需谨慎。

## 诚实结论

- **离线无法复现崩**。所有离线测试(含 graph pool + buffer populate)全 PASS。
- **端到端崩溃根因未确认**。临时 extend_seq_lens 不是根因(graph pool 保护)。
- **graph-safety 修复未必有效**, 但写法更稳妥, 不会更差。
- **grid 异常是新线索**, 但不直接导致越界(kernel 有 q_len 检查)。

## 下一步: 端到端探针定位

已增强探针 F(capture 时打印 grid 计算用的 max_seqlen_q + grid 值 + 警告)。
新容器需:

1. `EAGLE3_VERIFY_PROBE=1 bash start_eagle3.sh`
2. 跑 HumanEval 触发 bs=16 崩溃
3. 看 `/workspace/logs/eagle3_verify_probe.log`:
   - `[F CAPTURE]` 含 `Step1 grid=(x,y)` 和 `max_seqlen_q=` —— 确认 capture 时 grid 是否异常
   - `[F 警告] max_seqlen_q != D` —— 若出现, 说明 capture 时 max_seqlen_q 错
   - `[A REPLAY]` 含 replay 时 max_seqlen_q/seq_lens/req_pool_indices —— 确认 replay 输入

4. 若 capture grid 正常但仍崩, 需在 kernel 内部加越界断言(triton printf)定位具体 store/load

## 待排查的其他可能根因 (离线 vs 端到端差异)

1. **set_kv_buffer 写 k_cache**: 端到端 graph 捕获 set_kv_buffer, replay 用 graph buffer
   out_cache_loc 写 k_cache。若 out_cache_loc 与 req_to_token 不一致, verify kernel 读 k_cache
   位置错。离线测试没 set_kv_buffer (预填 k_cache)。
2. **多层叠加**: 端到端 60 层, 离线单层。某层输出异常累积。
3. **TP 通信**: 端到端 8 卡, 离线单卡。all-reduce 时序问题。
4. **scheduler 时序**: req_to_token 的 slot 分配/回收与 graph replay 的时序。

最可疑: #1 (set_kv_buffer), 因离线测试跳过了它。
