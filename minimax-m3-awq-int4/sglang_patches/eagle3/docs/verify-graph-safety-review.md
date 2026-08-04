# verify kernel graph-safety 复查 (2026-08-04)

复查 `minimax_sparse_verify_prefill` 的每个参数:地址在 capture/replay 间是否稳定,
值能否在 replay 时正确更新。核心原则:**cuda graph 捕获 tensor 地址,replay 时地址不能变,
但值要能更新**。

## 关键认知:cuda graph 私有内存池

sglang 用全局 graph 内存池(`cuda_graph_runner.py:1181 set_global_graph_memory_pool`)。
**capture 期间所有 GPU 内存分配(含 torch.full/cat/+/linear 输出)都从这个池分配,
replay 时复用同一地址,不会被 Python GC 回收**。

验证(`/workspace/verify/` 临时测试):graph pool 下 `torch.full` 临时 tensor replay 后
地址不变、值正确。**所以"临时 tensor 被 GC 导致地址失效"的假设是错的**——graph 池保护地址。

→ 这意味着:旧代码(用临时 extend_seq_lens)在 graph 下地址也稳定,不该因地址失效崩。
**bs=16 崩溃的真正根因仍需在新机器用修复后代码重跑端到端确认**。本修复(绕过 extend_seq_lens
改用 arange)至少与标准 triton backend 的 qo_indptr 模式对齐,是更稳妥的写法,不会更差。

## 参数复查表

### 1. 数据 tensor (kernel 读写的数据)

| 参数 | 来源 | 地址稳定性 | 值更新机制 | 判定 |
|------|------|-----------|-----------|------|
| `q` | model.forward 的 qkv_proj 输出 | ✅ graph 私有池,replay 重放 qkv_proj 写同地址 | ✅ replay 时 qkv_proj(hidden_states) 重新计算 | OK |
| `idx_q` | index_q_proj 输出 | ✅ 同上 | ✅ 同上 | OK |
| `k_cache`/`v_cache` | kv_pool.get_kv_buffer (paged pool) | ✅ paged pool 地址固定 | ✅ set_kv_buffer 用 out_cache_loc 写新 KV | OK |
| `idx_k_cache`/`idx_v_cache` | kv_pool.get_index_kv_buffer | ✅ 同上 | ✅ set_index_kv_buffer | OK |
| `req_to_token` | self.req_to_token (runner.req_to_token_pool) | ✅ pool 内地址固定 | ✅ scheduler 分配 slot 时更新 | OK |
| `forward_batch.req_pool_indices` | graph buffer `buffers.req_pool_indices[:bs]` | ✅ graph buffer 固定 | ✅ replay_prepare populate copy_ 真实值 | OK |
| `forward_batch.out_cache_loc` | graph buffer `buffers.out_cache_loc[:raw_num_token]` | ✅ graph buffer 固定 | ✅ replay_prepare populate copy_ 真实值 | OK |

### 2. seqlens 类 tensor (forward_extend 里构建)

| 参数 | 修复前(旧) | 修复后(新) | 判定 |
|------|-----------|-----------|------|
| `cu_seqlens` | `torch.cat([zeros, extend_seq_lens.cumsum(0)])` 依赖临时 extend_seq_lens | `torch.arange(0,(bs+1)*D,D)` Python int 输入,同 triton qo_indptr | 修复后更稳 |
| `seq_lens` | `raw_seq_lens + extend_seq_lens` 依赖临时 | `raw_seq_lens + D` (graph buffer + Python int) | 修复后更稳 |
| `prefix_lens` | `= raw_seq_lens` (graph buffer view) | `= raw_seq_lens` (不变) | OK |
| `raw_seq_lens` | `forward_batch.seq_lens.to(int32)` | 同左,seq_lens 是 graph buffer,.to(int32) no-op 同地址 | OK |

### 3. Python int 参数 (graph-safe 天然)

| 参数 | 值 | 判定 |
|------|---|------|
| `self._max_seqlen_q` | D (capture/replay 一致) | OK |
| `self._max_seqlen_k` | _max_seqlen_k_upper (上界,一致) | OK |
| `max_seqblock_k_upper` | cdiv(context_len, block_size_k) (静态) | OK |
| `block_size_q/k`, `topk_blocks`, `init/local_blocks` | config 常量 | OK |

### 4. score / topk_idx / o 输出 (kernel 内分配)

| 参数 | 分配点 | 地址稳定性 | 判定 |
|------|--------|-----------|------|
| `score` | verify kernel wrapper `torch.full((num_heads,total_q,max_seqblock_k_upper),-inf)` | ✅ graph 私有池,第3维固定上界 1600 | OK (Strategy B 核心修复) |
| `topk_idx` | `torch.full((num_heads,all_seqblock_q,topk),-1)` | ✅ graph 私有池,all_seqblock_q=bs*D 固定 | OK |
| `o`/`idx_o` | `torch.empty(total_q,...)` | ✅ graph 私有池,total_q=bs*D 固定 | OK |

## 仍存疑的点 (需新机器端到端验证)

1. **bs=16 崩溃的真正根因未完全确认**。graph pool 测试证明临时 tensor 地址稳定,所以
   "extend_seq_lens 被 GC"的假设可能不是真因。真因可能是:
   - kernel 内部某个依赖 seq_lens 的计算在 bs=16 时触发越界(但 probe A 检查的输入都 OK)
   - capture/replay 间某个 buffer 的 shape 或值 sglang 没正确填充
   - DCU/ROCm 特定的 graph replay 问题

2. **修复后的代码从未在端到端跑过**(本机显存被僵尸上下文占,无法重启服务)。

3. **离线测试全过**(graph-safety/精度/越界/性能/graph复现),但离线测试的 graph 环境
   与 sglang 端到端的 graph 环境有差异(离线是我自己分配 buffer,sglang 是 populate 填充)。

## 结论

修复后的代码在**写法上更稳妥**(对齐 triton backend 的 graph-safe 模式),离线测试全过。
但**端到端 bs=16 是否真不崩,必须在新机器重跑 `verify_humaneval_eagle3.py` 确认**。
若仍崩,需在 sglang 加更精细的探针(我加的探针 F 已在 backend,但崩溃那次没启用)定位
真正的越界 tensor。

## 建议:新机器验证步骤

1. `bash apply_patch.sh && bash sglang_patches/eagle3/install.sh`
2. 跑 5 个离线测试(应全过)
3. `EAGLE3_VERIFY_PROBE=1 bash start_eagle3.sh`(开探针)
4. 跑 `verify_humaneval_eagle3.py`
5. 若崩:看 `/workspace/logs/eagle3_verify_probe.log` 的 `[F CAPTURE]`(capture bs=16 shape)
   + `[A REPLAY]`(replay bs=16 输入值),定位真正越界点
6. 若不崩:确认 graph-safety 修复有效
