# eagle3-v2 — MiniMax-M3 EAGLE3 投机解码 patch (精简版)

> 本机实测可跑的最终版本。直接复用 sglang 标准 `minimax_sparse_prefill` 路径,
> **不依赖任何 `verify/` 子模块**,不含已废弃的 Strategy A/B 实验代码。

## 这是什么

把当前容器里**实际在跑**的 sglang EAGLE3 适配代码打成 patch,可在别的容器一键安装,
验证 EAGLE3 投机解码是否生效。

### 与 `eagle3` (v1) 的区别

| | eagle3 (v1) | **eagle3-v2** (本目录) |
|---|---|---|
| 改动文件数 | 3 + 新增 `verify/` 4 文件 | **6 个文件** (3 EAGLE3 接口 + 3 DCU/cuda-graph kernel) |
| verify 路径 | Strategy B 新 kernel (`verify_sparse`) | **标准 `minimax_sparse_prefill`** |
| 是否带 Strategy A | 是 (`seqlens_expand_triton`, 已废弃) | **否** |
| VMFault 修复 | score 固定上界 + 临时 tensor 绕过 | **同根因, 更精简的实现** |
| DCU shared-mem / cuda-graph 兼容 | ❌ 未含 (别机必崩) | **✅ 含 3 个 kernel 修复** |
| 来源 | git 仓库提交 (8-4 上午, 端到端根因当时未确认) | **本机 site-packages 在跑版** |

v1 在 git 历史里 (commit `5417311`) 自己承认"端到端根因仍未确认"。
v2 = 本机后来演化出、实际在跑的精简版 (绕开整个 verify/ 子模块)。

## 改动的 6 个文件

### A. EAGLE3 接口与 graph-safe 修复 (3 个)

| 文件 | site-packages 目标 | 作用 |
|---|---|---|
| `modified/minimax_m3_vl.py` | `sglang/srt/models/minimax_m3_vl.py` | VL 类补 EAGLE3 target 侧接口: `set_eagle3_layers_to_capture` / `get_embed_and_head` / `capture_aux_hidden_states` flag + aux-aware forward |
| `modified/minimax_sparse_backend.py` | `sglang/srt/layers/attention/minimax_sparse_backend.py` | TARGET_VERIFY 字段 materialize (`extend_seq_lens=None→draft_token_num`) + K 长度=`prefix+draft` 重建 + graph-safe (capture/replay 用静态 `q.shape[0]`, 不 host sync) |
| `modified/utils.py` | `sglang/srt/layers/attention/minimax_sparse_ops/common/utils.py` | `get_cu_seqblocks` graph-safe (`.sum().item()` host sync → capture 下静态上界 `batch*max_seqblock`) |

### B. DCU/HIP + cuda-graph 兼容性修复 (3 个 sparse_ops kernel)

这 3 个是 EAGLE3 让 sparse **prefill** 首次进入 cuda graph 后才暴露的 DCU 兼容问题。
**不加这 3 个, 别机必崩** (本机因已 in-place 改过 site-packages 才不崩):

| 文件 | site-packages 目标 | 作用 |
|---|---|---|
| `modified/prefill_topk_sparse.py` | `…/minimax_sparse_ops/prefill/topk_sparse.py` | DCU 64KB shared-mem 限制 → `_PREFILL_NUM_STAGES=[1] if is_hip() else [2,3]`。原版写死 `num_stages=2,3` (~68KB) 超 65536 → prefill 崩 `OutOfResources: shared memory 69632 > 65536` |
| `modified/decode_topk_sparse.py` | `…/minimax_sparse_ops/decode/topk_sparse.py` | 同上, decode kernel `_NUM_STAGES=[1] if is_hip() else [2,3,4,5]`, decode 路径同样会超 shared-mem |
| `modified/decode_flash_with_topk_idx.py` | `…/minimax_sparse_ops/decode/flash_with_topk_idx.py` | cuda-graph capture 期间禁用 side_stream fork (`torch.cuda.is_stream_capturing()` → `side_stream=None`)。原版无条件 `torch.cuda.Stream()` fork 新 stream, capture 期间非法 → 多 batch cuda graph 崩 |

每个文件同时附 `.patch` (基于 sglang 干净原版的 unified diff), 供审计/手工 apply。

> ⚠ **为何之前别机装了 v2 仍崩**: 早期 v2 只含 A 组 3 个文件, B 组 3 个 kernel 是本机直接改
> site-packages 的, 一直没进 patch 体系。别机装完 v2 后这 3 个 kernel 仍是原版 →
> prefill/decode 超 shared-mem 崩 + cuda-graph fork stream 崩。本版补齐 B 组后根治。

## VMFault 修复 (两层, 本版均已根治)

EAGLE3 是首个让 sparse **prefill** 进入 cuda graph 的场景 (无 EAGLE3 时
prefill=eager, decode=graph-safe)。两层问题:

1. **score buffer 动态尺寸** — prefill 的 `score`/`topk` 按 `cdiv(max_seqlen_k, block_size_k)`
   动态分配, capture (dummy `seq_lens=1`) 与 replay (真实 ~2000) 尺寸不一致 → 越界写。
   **修复**: `utils.py get_cu_seqblocks` 在 `is_current_stream_capturing()` 下用静态上界
   `batch_size * max_seqblock`, 张量形状 capture/replay 恒定; kernel 内部仍按真实 seq_len 索引。

2. **forward_extend 临时 tensor** (bs≥并发阈值才崩的真正根因) — 旧版用临时 `extend_seq_lens`
   (`forward_extend` 里 `torch.full` 新建) 算 cu_seqlens。capture 的 `forward_batch` 是局部变量,
   捕获后被 GC, `extend_seq_lens` 内存释放/复用; replay 读老地址 → 垃圾 → 越界。
   bs=1 碰巧不炸, bs=16 必崩。
   **修复**: verify 分支 materialize `extend_seq_lens = draft_token_num` (固定形状, graph-safe),
   K 长度重建为 `prefix + draft` (`raw_seq_lens + extend_seq_lens`),
   `prefix_lens` 用 `forward_batch.seq_lens` **graph buffer 引用** (地址稳定, replay 时是真实值),
   不用 capture 时物化的 `extend_prefix_lens` (会过期)。

## 用法

```bash
# 安装 (自动备份原版 → 覆盖 → 清 triton 缓存 → 验证)
bash sglang_patches/eagle3-v2/install.sh

# 仅检查是否已安装 v2
bash sglang_patches/eagle3-v2/install.sh --check

# 回滚 (从备份恢复干净原版)
bash sglang_patches/eagle3-v2/install.sh --rollback

# 安装后启动 EAGLE3 服务 (端口 8082)
bash sglang_patches/eagle3-v2/start_eagle3.sh
```

### 从 v1 切到 v2

若容器之前装过 v1 (会残留 `verify/` 目录 + `seqlens_expand_triton` 标记):

```bash
bash sglang_patches/eagle3/install.sh --rollback   # 先回滚 v1
# 可选: 删 v1 新增的 verify/ 目录
rm -rf /usr/local/lib/python3.10/dist-packages/sglang/srt/layers/attention/minimax_sparse_ops/verify
bash sglang_patches/eagle3-v2/install.sh           # 再装 v2
bash sglang_patches/eagle3-v2/install.sh --check   # 确认无 v1 残留
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
├── install.sh                             # 一键安装/检查/回滚
├── start_eagle3.sh                        # 启动 EAGLE3 sglang 服务 (端口 8082)
├── modified/
│   ├── minimax_m3_vl.py                   # 在跑版 (完整文件)
│   ├── minimax_m3_vl.py.patch             # unified diff (原版→在跑版)
│   ├── minimax_sparse_backend.py
│   ├── minimax_sparse_backend.py.patch
│   ├── utils.py
│   ├── utils.py.patch
│   ├── prefill_topk_sparse.py             # DCU shared-mem 修复
│   ├── prefill_topk_sparse.py.patch
│   ├── decode_topk_sparse.py              # DCU shared-mem 修复
│   ├── decode_topk_sparse.py.patch
│   ├── decode_flash_with_topk_idx.py      # cuda-graph side_stream 修复
│   └── decode_flash_with_topk_idx.py.patch
└── docs/
    └── (可放验证记录)
```

## 验证 EAGLE3 是否生效

启动后看日志:
- `[EAGLE3]` / `speculative` 相关行, accept rate ~0.7-0.8
- 吞吐: 纯 W4A16 eager ~5 tok/s → EAGLE3 + cuda graph 16-22 tok/s
- 输出正确: 对话/代码不乱码不复读

若崩 VMFault:
- 确认 `--check` 全 ✓ 且无 v1 残留 (尤其 6 个文件标记都要 ✓)
- `start_eagle3.sh` 里 `EAGLE3_VERIFY_PROBE=1` 已开, 看 `/workspace/logs/eagle3_verify_probe.log`
- 确认 sglang 版本一致 (版本不一致是 patch apply 后行为异常的最常见原因)

若崩 `OutOfResources: shared memory 69632 > 65536`:
- 这是 DCU 64KB shared-mem 限制, 说明 `prefill_topk_sparse.py` / `decode_topk_sparse.py`
  的 `_NUM_STAGES` 修复没生效 → 跑 `install.sh --check` 确认这 2 个标记 ✓
- 同时确认 `python3 -c "import torch; print(torch.version.hip)"` 非 None
  (is_hip()=False 时 num_stages 仍会选 2,3 — 那是 PyTorch 装错成 CUDA 版, 不是 patch 问题)

若多 batch cuda graph 崩 (stream capture 相关):
- 说明 `decode_flash_with_topk_idx.py` 的 `is_stream_capturing` 修复没生效
  → 跑 `install.sh --check` 确认该标记 ✓

## 与 v1 (`eagle3/`) 残留共存说明

v2 **完全不引用** v1 的 `verify/` 子模块 (v2 的 verify 路径走标准 `minimax_sparse_prefill`,
不依赖 `verify_sparse`)。已验证:

- v2 的 3 个文件零 `verify_sparse` / `minimax_sparse_ops.verify` 引用
- 全 sglang 代码只有 v1 自己的 sparse_backend import 过 verify (装 v2 后被覆盖, 该 import 消失)
- `minimax_sparse_ops/__init__.py` 不自动加载 verify 子包 → 残留的 verify/ 是"孤儿", 无人触发

因此 **v1 残留的 `verify/` 目录对 v2 执行零影响**, 三种场景:

| 场景 | 结果 |
|---|---|
| 全新机器 (没装过 v1) | 最干净, 无 verify/ 目录, 直接装 v2 |
| 装过 v1 直接装 v2 | v1 sparse_backend 被覆盖, verify/ 成孤儿残留, **v2 照常运行** (~40KB 占磁盘, 无害) |
| 装过 v1 先回滚再装 v2 | v1 `--rollback` 会删 verify/, 然后装 v2, 纯净 |

**v2 install.sh 有意不删 v1 的 verify/ 残留** (只管自己改的 3 个文件, 不碰别人的东西)。
若想彻底清理残留:

```bash
# 方式1: 先回滚 v1 (会删 verify/), 再装 v2
bash sglang_patches/eagle3/install.sh --rollback
bash sglang_patches/eagle3-v2/install.sh

# 方式2: 手动删孤儿目录
rm -rf /usr/local/lib/python3.10/dist-packages/sglang/srt/layers/attention/minimax_sparse_ops/verify
```
