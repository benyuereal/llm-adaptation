# EAGLE3 v2 VMFault 根治测试套件

本目录验证 **eagle3-v2** 的 VMFault 根治修复。**不起 sglang 服务**, 直接测根因逻辑。
运行顺序 = 先纯 CPU 形状一致性 / graph buffer 逻辑, 再 GPU 真越界复现。

## 修复演进 (两阶段)

### 阶段 1 — score buffer 恒定上界 (test 1, 2)

根因在 `prefill/flash_with_topk_idx.py`:

```python
max_seqblock_k = cdiv(max_seqlen_k, block_size_k)
score = torch.full((num_heads, total_q, max_seqblock_k), -inf)   # 第3维依赖 max_seqlen_k
# kernel 内:
block_num = (seq_len + block_size - 1) // block_size            # 从真实 seq_lens 算
tl.store(s_ptrs, val, boundary_check=(0,1))                     # 写 score[..., 0:block_num]
```

`max_seqlen_k` 来自 backend 的 `_max_seqlen_k`:

| 阶段 | `_max_seqlen_k` | score 第3维 | 结果 |
|---|---|---|---|
| **capture** | dummy `seq_lens=1` + draft(4) = 5 | `cdiv(5,128)=1` | 被 graph 锁定 |
| **replay** | 真实 `seq_lens~2000` + 4 = 2004 | kernel `block_num=16` | 写 `score[...,1..15]` 越界 |

越界写破坏 score 之后的相邻张量 → garbage 输出; 越界足够远跨页时 ROCr 报
`KERNEL VMFault: Invalid address access`。**单请求也崩**(不是并发问题),
崩在第 2 个请求 decode 阶段(replay 时 seq_lens 已增长到 ~2000)。

**修复**: backend `__init__` 存 `max_seqblock_k_upper = cdiv(context_len + draft, block_size_k)`
= `cdiv(204800+4, 128)` = **1601**, 经 `minimax_sparse_prefill` 透传, score 第3维用此
恒定上界。capture/replay 形状恒定, kernel 内 `block_num` 仍从真实 seq_lens 算,
`boundary_check` 保护, 多余槽位填 `-inf` 不被读, causal 语义零变化。

### 阶段 2 — verify_prefill kernel + graph buffer (test 3, 根治)

阶段 1 的 score 上界修复**不够**: 探针(EAGLE3_VERIFY_PROBE)确认 `cu_seqlens` /
`seq_lens` / `extend_seq_lens` 在旧 `forward_extend` 里用 `torch.cat` / `a+b` /
`torch.full` 现场构造, 每次 new 出临时张量 → **data_ptr 在 capture 与 replay 不同**
→ graph 锁了 capture 时地址, replay 读到别处 → garbage/VMFault。这才是真正根因。

**修复 (v2 架构)**: 新建 `minimax_sparse_ops/verify/` 专用 verify_prefill kernel
(`minimax_sparse_verify_prefill`), 并在 `minimax_sparse_backend.py`:

- `init_cuda_graph_state`: 预分配 3 个 graph buffer (一次性, 地址固定):
  `_verify_cu_seqlens_buf[max_bs+1]` / `_verify_extend_seq_lens_buf[max_bs]` /
  `_verify_seq_lens_buf[max_bs]`, 全 int32 (镜像 triton backend 的 `qo_indptr` 模式)
- `init_forward_metadata_{capture,replay}_cuda_graph`: 写入同一块 buffer
  (capture/replay data_ptr 相同 → graph-safe)。`cu_seqlens=[0,D,2D,...,bs*D]` /
  `extend_seq_lens=[D]*bs` 只依赖 `(bs,D)` (graph 内常量); `seq_lens=prefix+D` 随真实 prefix
- `forward_extend` 对 `is_target_verify()` 路由到 `_forward_verify`, 用预分配 buffer;
  `_max_seqlen_k = max_seqblock_k_upper * block_size_k` (恒定, 仅过 score 断言)
- `_forward_verify` 有 `has_graph_buf` 分支: eager 路径 (bs>cuda_graph_max_bs) 无 buffer
  → 用 `torch.cat` 构造临时张量 (eager 不需要 graph-safety)

## 前置

- 已装 eagle3-v2 patch (`bash ../install.sh`, `--check` 8/8 ✓)。
  本容器若已直接改 site-packages, 则无需再装。
- 测试2 需 GPU(DCU/CUDA), `torch.cuda.is_available()` 为 True, 至少 1 张卡。
- 测试1 纯 CPU, 无需 GPU。
- triton 缓存已清(`install.sh` 会清; 手测可 `rm -rf /models/.triton_cache/*`)。

## 测试清单与运行顺序

### 1. `test_vmfault_score_upper_bound.py` — 形状一致性回归测试(**最先跑, 纯 CPU 秒级**)

验证越界的**必要条件**: score 第3维在 capture/replay 下是否一致。
**不复现越界写本身**(那需 GPU, 见测试2), 只复刻 `flash_with_topk_idx.py:523` 那行算术,
证明旧逻辑 capture=1 / replay=16 形状分叉, 新逻辑恒定=1601。

- 旧逻辑: `cdiv(1+4, 128)=1` vs `cdiv(2000+4, 128)=16` → 不一致(越界必要条件)
- 新逻辑: 任何 seq_len 都返回上界 1601 → 一致
- 上界 1601 ≥ 最坏真实 `cdiv(context_len+draft, 128)=1601` → 不会越界

**价值 = 回归保护**: 以后若有人把 `max_seqblock_k_upper` 改回动态值, 本测试立刻 fail。

```bash
python test_vmfault_score_upper_bound.py
# 预期: ✅ VMFault 根治 (形状一致性) 验证通过
```

### 2. `test_vmfault_graph_repro.py` — 真越界写复现(**需 GPU, ~10 秒**)

用最小 triton kernel **真正复刻** `flash_with_topk_idx` 的 score store 模式, 在真实
`torch.cuda.CUDAGraph` capture/replay 下演示越界写:

- 用连续大 buffer 切片出 score, 紧贴 guard 区 → 越界写一定能被检测(不依赖 GPU 页布局)
- capture: `seq_lens=5`(同真实 dummy), replay: `seq_lens=2004`(同真实)
- **旧逻辑**(score第3维=1): replay `block_num=32` → 越界写 30 个 float 到 guard 区 ✗
- **新逻辑**(score第3维=3201): `block_num=32 << 3201` → 全在 score 内 ✓

> 注: 本脚本用 `block_size_k=64` 作**演示参数** (让 capture=1/replay block_num=32 的
> 越界效果明显, score 张量小跑得快), 非真实配置。真实配置 `block_size_k=128`
> (见 test 1 / test 3)。两者演示的机制相同: score 第3维用恒定上界 → capture/replay
> 形状一致 → 不越界。

注: 测试**不**真正触发 ROCr VMFault(那会崩进程)。它演示的是越界写本身 ——
"写到 score 分配范围之外的内存" —— 即 VMFault 的直接前因。

```bash
python test_vmfault_graph_repro.py
# 预期: ✅ 真复现成功 (旧逻辑越界写, 新逻辑恒定上界安全)
```

### 3. `test_verify_graph_buffers.py` — graph buffer 逻辑单元测试(**纯 CPU 秒级, 阶段2 根治**)

验证阶段 2 的 verify_prefill kernel + graph buffer 方案 (VMFault 真正根因的修复)。
用 mock 复刻 backend 的 5 个方法 (`init_cuda_graph_state` /
`init_forward_metadata_{capture,replay}_cuda_graph` / `_forward_verify` 的 buffer 选择 /
score 上界断言), **不起 sglang / 不跑 triton / 不需 GPU**, 8 组测试:

1. `init_cuda_graph_state` 预分配 3 个 buffer, dtype=int32, shape 正确
2. capture/replay 填充: `cu_seqlens=[0,D,2D,...,bs*D]`, `extend=[D]*bs` (只依赖 bs,D)
3. **核心**: capture 与 replay 用同一块预分配 buffer (data_ptr 相同) → graph-safe
   (旧实现 vmfault 真因: `torch.cat`/`a+b` 现场临时张量, data_ptr capture≠replay)
4. eager 路径 (bs>cuda_graph_max_bs, 无 buffer) 不崩, 数值正确
4b. graph 与 eager 路径对同 (bs,D) 产出常量数值一致
5. verify kernel score 上界断言 `max_seqblock_k_upper >= cdiv(max_seqlen_k, bsk)` 通过
6. `cu_seqlens`/`extend` 与 prefix 无关 (只依赖 bs,D), `seq_lens` 随 prefix
7. **回归保护**: `inspect` 检查真实 backend 源码含全部 11 项 graph-safe 关键语句
   (若 backend 改了 buffer 逻辑, 本测试 fail, 提醒同步 mock)

```bash
python test_verify_graph_buffers.py
# 预期: ✅ verify graph buffer 逻辑验证通过 (8 组测试全过)
```

## 三个测试的关系

| | 测试1 (score_upper_bound) | 测试2 (graph_repro) | 测试3 (verify_graph_buffers) |
|---|---|---|---|
| 阶段 | 1 (score 上界) | 1 (score 上界) | 2 (graph buffer, 根治) |
| 需 GPU | 否(纯 CPU) | 是(1 张 DCU) | 否(纯 CPU) |
| 用 cuda graph | 否 | 是(真 capture/replay) | 否(mock) |
| 跑 triton kernel | 否 | 是 | 否 |
| 测什么 | score 形状分叉 | 越界写机制 | buffer 地址稳定性 + 填充逻辑 |
| 用途 | 快速回归 | 根因机制演示 | 根治方案回归保护 |
| 耗时 | <1 秒 | ~10 秒 | <1 秒 |

测试1+2 针对阶段 1 (score 上界, 必要但不充分); 测试3 针对阶段 2 (graph buffer, 真正根治)。
三者互补: 1/3 适合 CI 秒级跑, 2 适合改动 kernel store 逻辑后做机制验证。

## 端到端验证(本目录之外)

离线测试全过后, 起服务做端到端:

```bash
bash ../start.sh                   # 等待 "server is fired up" (默认 EAGLE3)
# 跑 ≥2 个请求 (上次崩在第 2 个请求 decode 阶段, 单请求 #running-req:1)
```

重点看:
- 无 `KERNEL VMFault` / `Invalid address access`
- `accept rate` 正常 (0.4–0.9), 输出不乱码不复读

## 与 v1 (`eagle3/tests/`) 的区别

v1 的测试套件(`test_verify_*.py`)针对 Strategy B 的独立 verify kernel(`verify/` 子模块)。
v2 **重新启用** verify 子模块 (`minimax_sparse_ops/verify/`, 新 `minimax_sparse_verify_prefill`),
但加了 graph buffer 管理 (阶段 2), 所以 v1 那套测试不完全适用 —— 本目录测试3 专门覆盖
v2 的 graph buffer 逻辑, v1 测试覆盖 kernel 数值正确性。
