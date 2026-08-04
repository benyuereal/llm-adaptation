# eagle3-v2 — MiniMax-M3 EAGLE3 投机解码 patch (专用 verify kernel + graph buffer 根治 VMFault)

> 本机实测稳定可跑的最终版本 (并发 8、每请求 1000 tokens 压测无 VMFault,
> accept rate ~0.59)。用专用 `verify_prefill` kernel + 预分配 graph buffer,
> 两层根治 EAGLE3 TARGET_VERIFY 在 cuda graph 下的 KERNEL VMFault。

## 这是什么

把当前容器里**实际在跑**的 sglang EAGLE3 适配代码打成 patch,可在别的容器一键安装,
验证 EAGLE3 投机解码是否生效。覆盖 12 个文件 (8 个既有 + 4 个 verify 子模块)。

---

## VMFault 根因与修复 (核心, 必读)

EAGLE3 是首个让 sparse **prefill** (verify 步) 进入 cuda graph 的场景。无 EAGLE3 时
prefill=eager、decode=graph-safe, 所以这些 graph 问题之前从未暴露。cuda graph 的本质:
**capture 时锁定所有张量的分配形状与地址, replay 时复用**。任何在 capture/replay 之间
**形状变化**或**地址变化**的张量 → 越界 / 读垃圾 → KERNEL VMFault。

VMFault 有**两层根因**, 阶段1 的修复是必要但不充分的, 阶段2 才是真正根治。

### 阶段1 — score buffer 第3维动态尺寸 (越界写)

`prefill/flash_with_topk_idx.py` 里:
```python
max_seqblock_k = cdiv(max_seqlen_k, block_size_k)
score = torch.full((num_heads, total_q, max_seqblock_k), -inf)   # 第3维依赖 max_seqlen_k
# kernel 内:
block_num = cdiv(seq_len, block_size_k)                          # 从真实 seq_lens 算
tl.store(s_ptrs, val, boundary_check=(0,1))                      # 写 score[..., 0:block_num]
```

`max_seqlen_k` 来自 backend 的 `_max_seqlen_k`, 它在 capture/replay 之间变化:

| 阶段 | `_max_seqlen_k` | score 第3维 | 结果 |
|---|---|---|---|
| **capture** | dummy `seq_lens=1` + draft(4) = 5 | `cdiv(5,128) = 1` | 被 graph 锁定 |
| **replay** | 真实 `seq_lens ~2000` + 4 = 2004 | kernel `block_num = 16` | 写 `score[...,1..15]` 越界 |

越界写破坏 score 之后的相邻张量 → garbage 输出; 越界足够远跨页时 ROCr 报
`KERNEL VMFault: Invalid address access`。

**修复 (阶段1)**: backend `__init__` 存恒定上界
`max_seqblock_k_upper = cdiv(context_len + draft, block_size_k) = cdiv(204800+4, 128) = 1601`,
经 `minimax_sparse.py:minimax_sparse_prefill` 透传, score 第3维用此恒定上界。
capture/replay 形状恒定; kernel 内 `block_num` 仍从真实 `seq_lens` 算, `boundary_check`
保护, 多余槽位填 `-inf` 不被读, causal 语义零变化。显存代价 ~0.09GB/卡, 可接受。

### 阶段2 — cu_seqlens/seq_lens 临时张量地址漂移 (真正根因)

阶段1 修复后**仍崩 VMFault**。加探针 (`EAGLE3_VERIFY_PROBE`) 记录 capture/replay 时
传给 verify kernel 的所有张量 data_ptr, 发现:

| 张量 | capture data_ptr | replay data_ptr | 稳定? |
|---|---|---|---|
| `cu_seqlens` | `0x7f3460200200` | `0x7f34c60ec400` | ❌ **变了** |
| `seq_lens` | (变化) | (变化) | ❌ **变了** |
| `q` | (变化) | (变化) | ❌ (但 q 是 sglang 既有 graph buffer, 见下表) |
| `extend_seq_lens` | 稳定 | 稳定 | ✓ |
| `prefix_lens` | 稳定 | 稳定 | ✓ |

根因: 旧 `forward_extend` 用 `torch.cat([zeros, cumsum])` / `a + b` / `torch.full`
**现场临时构造** `cu_seqlens` / `seq_lens` / `extend_seq_lens`。每次调用都 new 一块新内存
→ **data_ptr 在 capture 与 replay 不同** → graph 锁了 capture 时的地址, replay 读到别处
→ garbage / VMFault。`cu_seqlens` 内容是常量 `[0,D,2D,...,bs*D]`, 但因为每次重建, 地址
还是漂。

**修复 (阶段2, 真正根治)**: 新建专用 verify kernel 子模块 `verify/`
(`minimax_sparse_verify_prefill`), 并在 backend 预分配 graph buffer:

1. `init_cuda_graph_state`: 一次性预分配 3 个 graph buffer (地址锁死 backend 生命周期):
   - `_verify_cu_seqlens_buf[max_bs+1]` (int32)
   - `_verify_extend_seq_lens_buf[max_bs]` (int32)
   - `_verify_seq_lens_buf[max_bs]` (int32)

   (镜像 sglang triton backend 的 `qo_indptr` 模式: 一次性分配, 地址固定。)

2. `init_forward_metadata_{capture,replay}_cuda_graph`: **写入同一块 buffer**
   (capture/replay data_ptr 相同 → graph-safe):
   - `cu_seqlens = arange(0, (bs+1)*D, D)` = `[0,D,2D,...,bs*D]` — 只依赖 `(bs,D)`, graph 内常量
   - `extend_seq_lens = [D]*bs` — 只依赖 `(bs,D)`, graph 内常量
   - `seq_lens = prefix + D` — 随真实 prefix 变 (kernel 运行时读, 这是正确的)

3. `forward_extend`: `is_target_verify()` 路由到 `_forward_verify`, 用预分配 buffer;
   `_max_seqlen_k = max_seqblock_k_upper * block_size_k` (恒定, 仅过 score 断言, 不参与分配)

4. `_forward_verify` 有 `has_graph_buf` 分支: eager 路径 (`bs > cuda_graph_max_bs` 或
   cuda graph 关闭) 无 buffer → 用 `torch.cat` 构造临时张量 (eager 不需要 graph-safety)

### 传给 verify kernel 的数据分类 (为何只需固化 3 个 buffer)

不是所有输入都要固化 —— 只固化「地址会变的临时张量」, 其余用既有稳定地址:

| 类型 | 数据 | 是否固化 | 原因 |
|---|---|---|---|
| **graph buffer (新加)** | `cu_seqlens` / `seq_lens` | ✅ 固化 | 旧实现临时构造, 地址漂移 → vmfault 真因 |
| sglang 既有 graph buffer | `q` / `idx_q` / `req_pool_indices` / `prefix_lens`(=seq_lens) | ❌ 本来稳定 | sglang `DecodeInputBuffers` 已预分配, 地址固定 |
| paged cache 池 | `k_cache` / `v_cache` / `idx_k_cache` / `idx_v_cache` | ❌ 本来稳定 | 池地址固定, 内容随 slot 更新 |
| 全局表 | `self.req_to_token` | ❌ 本来稳定 | 一次性分配 |
| 静态标量 | `max_seqlen_q`(=D) / `max_seqblock_k_upper` / `block_size_k` / `topk_blocks`... | ❌ Python int | 编译期常量 |

> 注: `prefix_lens = forward_batch.seq_lens.to(int32)`, 而 `seq_lens` buffer 是 int32
> (`cuda_graph_runner.py` 确认), `.to(int32)` 是 no-op 返回 self, 地址稳定, graph-safe。

### probe 如何定位根因

`EAGLE3_VERIFY_PROBE=1` 环境变量开启探针, `_forward_verify` 每次 capture/replay 都把
所有输入张量的 `data_ptr` / `shape` / `dtype` 写到 `/workspace/logs/eagle3_verify_probe.log`。
对比 `[V CAPTURE]` 与 `[V REPLAY]` 行即可看出哪个张量地址漂了。正是这个探针确认了
`cu_seqlens`/`seq_lens` 地址变化 (而非 score 上界) 才是阶段1 修复后仍崩的真正根因。

---

## 改动的 12 个文件

### A. EAGLE3 target 侧接口 (1)

| 文件 | site-packages 目标 | 作用 |
|---|---|---|
| `modified/minimax_m3_vl.py` | `sglang/srt/models/minimax_m3_vl.py` | VL 类补 EAGLE3 target 侧接口: `set_eagle3_layers_to_capture` / `get_embed_and_head` / `capture_aux_hidden_states` flag + aux-aware forward |

### B. verify 路由 + graph buffer 根治 (1, 阶段2 核心)

| 文件 | site-packages 目标 | 作用 |
|---|---|---|
| `modified/minimax_sparse_backend.py` | `…/layers/attention/minimax_sparse_backend.py` | TARGET_VERIFY 路由到 `_forward_verify`; `init_cuda_graph_state` 预分配 3 个 graph buffer; capture/replay 写同一 buffer (data_ptr 不变); `_max_seqlen_k` 取恒定上界; `has_graph_buf` eager 分支 |

### C. 专用 verify kernel 子模块 (4, 阶段2 核心)

| 文件 | site-packages 目标 | 作用 |
|---|---|---|
| `modified/verify/__init__.py` | `…/minimax_sparse_ops/verify/__init__.py` | export `minimax_sparse_verify_prefill` |
| `modified/verify/verify_sparse.py` | `…/minimax_sparse_ops/verify/verify_sparse.py` | verify 入口 (step1 score 固定上界 + step3 OOB 双保险) |
| `modified/verify/flash_with_topk_idx.py` | `…/minimax_sparse_ops/verify/flash_with_topk_idx.py` | verify step1 kernel, score 第3维用 `max_seqblock_k_upper` (恒定), grid 用 `max_seqlen_q`(=D=4, 恒定) |
| `modified/verify/topk_sparse.py` | `…/minimax_sparse_ops/verify/topk_sparse.py` | verify step3 kernel (gqa share sparse, OOB 双保险) |

### D. score 上界透传 (2, 阶段1 修复)

| 文件 | site-packages 目标 | 作用 |
|---|---|---|
| `modified/minimax_sparse.py` | `…/minimax_sparse_ops/minimax_sparse.py` | `minimax_sparse_prefill` 透传 `max_seqblock_k_upper` 到 step1 (供普通 prefill 路径, verify 走专用 kernel) |
| `modified/prefill_flash_with_topk_idx.py` | `…/minimax_sparse_ops/prefill/flash_with_topk_idx.py` | score 第3维从动态 `cdiv(max_seqlen_k,bsk)` 改用恒定上界 `max_seqblock_k_upper` (capture/replay 形状恒定) |

### E. graph-safe 辅助 (1)

| 文件 | site-packages 目标 | 作用 |
|---|---|---|
| `modified/utils.py` | `…/minimax_sparse_ops/common/utils.py` | `get_cu_seqblocks` graph-safe (`.sum().item()` host sync → capture 下静态上界 `batch*max_seqblock`) |

### F. DCU/HIP + cuda-graph 兼容性 (3)

这 3 个是 EAGLE3 让 sparse **prefill** 首次进入 cuda graph 后才暴露的 DCU 兼容问题。
**不加这 3 个, 别机必崩** (本机因已 in-place 改过 site-packages 才不崩):

| 文件 | site-packages 目标 | 作用 |
|---|---|---|
| `modified/prefill_topk_sparse.py` | `…/minimax_sparse_ops/prefill/topk_sparse.py` | DCU 64KB shared-mem 限制 → `_PREFILL_NUM_STAGES=[1] if is_hip() else [2,3]`。原版写死 `num_stages=2,3` (~68KB) 超 65536 → prefill 崩 `OutOfResources: shared memory 69632 > 65536` |
| `modified/decode_topk_sparse.py` | `…/minimax_sparse_ops/decode/topk_sparse.py` | 同上, decode kernel `_NUM_STAGES=[1] if is_hip() else [2,3,4,5]` |
| `modified/decode_flash_with_topk_idx.py` | `…/minimax_sparse_ops/decode/flash_with_topk_idx.py` | cuda-graph capture 期间禁用 side_stream fork (`is_stream_capturing()` → `side_stream=None`)。原版无条件 fork 新 stream, capture 期间非法 → 多 batch cuda graph 崩 |

---

## 用法

```bash
# 安装 (自动备份原版 → 覆盖 12 文件 → 清 triton 缓存 → 验证)
bash sglang_patches/eagle3-v2/install.sh

# 仅检查是否已安装 v2 (8 既有 + 4 verify + 5 graph buffer 标记)
bash sglang_patches/eagle3-v2/install.sh --check

# 回滚 (从备份恢复 8 既有文件 + 删除 verify/ 子模块)
bash sglang_patches/eagle3-v2/install.sh --rollback

# 安装后启动 EAGLE3 服务 (端口 8082)
bash sglang_patches/eagle3-v2/start_eagle3.sh
```

## 前置条件 (目标容器)

- sglang 已 `pip install`, 路径 `/usr/local/lib/python3.10/dist-packages/sglang/srt`
- sglang 版本与本机一致 (`0.0.0.dev12695+g1df793665.d20260605`), 否则 patch 可能 apply 失败
- Hygon DCU 环境 (gfx936/gfx928), `hip_moe_w4a16` backend (W4A16 moe-only 量化)
- EAGLE3 draft 模型: `/models/Inferact/MiniMax-M3-EAGLE3`
- 目标模型: MiniMax-M3-AWQ-INT4

## 文件清单

```
eagle3-v2/
├── README.md                              # 本文件
├── install.sh                             # 一键安装/检查/回滚 (12 文件)
├── start_eagle3.sh                        # 启动 EAGLE3 sglang 服务 (端口 8082)
├── modified/                              # 12 个 patch 文件 (整文件覆盖)
│   ├── minimax_m3_vl.py                   # EAGLE3 target 侧接口
│   ├── minimax_sparse_backend.py          # verify 路由 + graph buffer 根治 (阶段2)
│   ├── utils.py                           # get_cu_seqblocks graph-safe
│   ├── minimax_sparse.py                  # 透传 max_seqblock_k_upper (阶段1)
│   ├── prefill_flash_with_topk_idx.py     # score 第3维恒定上界 (阶段1)
│   ├── prefill_topk_sparse.py             # DCU shared-mem 修复
│   ├── decode_topk_sparse.py              # DCU shared-mem 修复
│   ├── decode_flash_with_topk_idx.py      # cuda-graph side_stream 修复
│   └── verify/                            # 专用 verify kernel 子模块 (阶段2)
│       ├── __init__.py
│       ├── verify_sparse.py
│       ├── flash_with_topk_idx.py
│       └── topk_sparse.py
└── tests/                                 # VMFault 根治测试 (见 tests/README.md)
    ├── README.md                          # 测试说明 + 运行顺序
    ├── test_vmfault_score_upper_bound.py  # 阶段1: score 形状一致性回归 (纯CPU, 秒级)
    ├── test_vmfault_graph_repro.py        # 阶段1: 真越界写复现 (需GPU, cuda graph)
    └── test_verify_graph_buffers.py       # 阶段2: graph buffer 逻辑单元测试 (纯CPU, 8 组)
```

## 验证 EAGLE3 是否生效

启动后看日志:
- `[EAGLE3]` / `speculative` 相关行, accept rate ~0.59 (并发 8 实测), accept len ~2.78
- 吞吐: 纯 W4A16 eager ~5 tok/s → EAGLE3 + cuda graph 显著提升
- 输出正确: 对话/代码不乱码不复读
- **无 `KERNEL VMFault` / `Invalid address access` / `SIGSEGV`** (并发 8 × 1000 tokens 压测通过)

## 排查指南

若崩 VMFault:
1. 跑 `install.sh --check` 确认 12 文件标记全 ✓ (尤其 5 个 graph buffer 标记 + 4 个 verify 标记)
2. **先跑离线测试** (不起服务, 见 `tests/README.md`):
   ```bash
   cd sglang_patches/eagle3-v2/tests
   python test_vmfault_score_upper_bound.py   # 阶段1: 形状一致性 (纯CPU, 秒级)
   python test_verify_graph_buffers.py        # 阶段2: graph buffer 逻辑 (纯CPU, 8 组)
   python test_vmfault_graph_repro.py         # 阶段1: 真越界复现 (需GPU, ~10秒)
   ```
3. 若需复现探针定位: `export EAGLE3_VERIFY_PROBE=1` 后重启, 看
   `/workspace/logs/eagle3_verify_probe.log` 里 `[V CAPTURE]` vs `[V REPLAY]` 的
   `cu_seqlens(buf)` / `seq_lens(buf)` data_ptr 是否相同 (相同 = 阶段2 修复生效)
4. 确认 sglang 版本一致 (版本不一致是 patch apply 后行为异常的最常见原因)

若崩 `OutOfResources: shared memory 69632 > 65536`:
- DCU 64KB shared-mem 限制, 说明 `prefill_topk_sparse.py` / `decode_topk_sparse.py` 的
  `_NUM_STAGES` 修复没生效 → 跑 `install.sh --check` 确认这 2 个标记 ✓
- 同时确认 `python3 -c "import torch; print(torch.version.hip)"` 非 None
  (is_hip()=False 时 num_stages 仍会选 2,3 — 那是 PyTorch 装错成 CUDA 版, 不是 patch 问题)

若多 batch cuda graph 崩 (stream capture 相关):
- 说明 `decode_flash_with_topk_idx.py` 的 `is_stream_capturing` 修复没生效
  → 跑 `install.sh --check` 确认该标记 ✓
