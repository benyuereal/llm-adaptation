# EAGLE3 投机解码 patch — MiniMax-M3 (Hygon DCU)

给 MiniMax-M3 AWQ-INT4 (W4A16 moe-only) 接 EAGLE3 投机解码的 sglang 补丁。
适配海光 DCU (gfx936/gfx928),搭配 `Inferact/MiniMax-M3-EAGLE3` draft head。

> **当前版本:Path 1 Strategy B + graph-safety 修复**(替代已废弃的 Strategy A)。
> Strategy A(verify→decode kernel)虽能不崩,但 decode kernel 无 causal mask、用启发式
> 选块,导致对话输出 garbage(HumanEval 代码补全因宽容而误判通过)。Strategy B 复制
> prefill kernel、保持严格 causal 语义,并修复 graph-safety 使并发不崩。

## 成果

- 纯 W4A16 eager 5 tok/s → EAGLE3 + cuda graph 目标 16-22 tok/s,accept ~0.78
- 输出正确(对话/代码均不乱码不复读 —— Strategy A 的 garbage 问题已解决)
- **VMFault 根治(两层)**:
  1. **score buffer 动态尺寸**(Strategy B):原 prefill 的 `score` 张量按动态
     `max_seqlen_k` 分配,capture(dummy seq_lens=1)/replay(真实 seq_lens~2000)尺寸不一致
     → 越界写。Strategy B 把 score 第3维固定为 `max_seqblock_k_upper=cdiv(context_len, block_size_k)`。
  2. **forward_extend 临时 tensor**(graph-safety 修复):verify 分支用临时 `extend_seq_lens`
     算 cu_seqlens/seq_lens,capture 后 forward_batch 被 GC,replay 读失效地址 → 垃圾 → 越界
     (bs=1 碰巧不炸,bs>=并发阈值必崩)。修复:改用 `forward_batch.seq_lens`(graph buffer)
     + Python int D + `torch.arange`(同 triton backend 的 qo_indptr 模式)。

## 文件清单

| 文件 | 作用 |
|---|---|
| `install.sh` | **一键安装** patch 到 site-packages(含 verify/ 新增目录、备份、验证、回滚) |
| `start_eagle3.sh` | 启动 EAGLE3 sglang 服务(端口 8082,`--speculative-attention-mode prefill`) |
| `modified/minimax_m3_vl.py` | VL 类补 EAGLE3 target 侧接口:`set_eagle3_layers_to_capture` / `get_embed_and_head` / aux-aware forward |
| `modified/minimax_sparse_backend.py` | sparse backend:target_verify 路由到新 verify kernel + **forward_extend verify 分支 graph-safety 修复**(绕过临时 extend_seq_lens,用 seq_lens buffer + arange) |
| `modified/utils.py` | `get_cu_seqblocks` graph-safe(host sync → 静态上界) + `seqlens_expand_triton` |
| `modified/verify/` | **Strategy B 新增:3 个 graph-safe verify kernel**(`flash_with_topk_idx.py` / `topk_sparse.py` / `verify_sparse.py` + `__init__.py`)。复制 prefill kernel,score buffer 固定上界 + Step3 OOB 双保险,causal 语义与 prefill 完全一致 |
| `tests/` | 测试套件(见 `tests/README.md`):graph-safety 验证 / 精度 / 越界碰撞 / 性能 / graph 复现 / HumanEval 端到端 |
| `docs/MiniMax-M3-EAGLE3-work-log.md` | 完整工作记录(背景/踩坑/修复/选型/待办) |

## 一键使用

```bash
# 1. 安装 patch(备份原文件 → 覆盖 backend/utils → 新增 verify/ → 清缓存 → 验证)
bash sglang_patches/eagle3/install.sh

# 2. 跑离线测试(不起服务,验证 kernel + graph-safety,前 5 个全过再起服务)
cd sglang_patches/eagle3/tests
python test_verify_graph_buffer.py        # graph-safety 修复验证(最先跑)
python test_verify_prefill_precision.py   # 精度 (9/9)
python test_verify_oob_collision.py       # 越界碰撞 (3/3)
python test_verify_perf.py                # 性能 (5/5, 开销<10%)
python test_verify_graph_capture_replay.py # kernel 级 graph 复现

# 3. 起服务 + 端到端 HumanEval(离线测试全过后)
bash sglang_patches/eagle3/start_eagle3.sh  # 等 "server is fired up"
python sglang_patches/eagle3/tests/verify_humaneval_eagle3.py  # 164题并发16

# 回滚 patch
bash sglang_patches/eagle3/install.sh --rollback
```

## 前置依赖

- sglang (dev,DCU dtk2604 build)
- MiniMax-M3 AWQ-INT4 模型权重(moe-only W4A16 量化产物)
- `Inferact/MiniMax-M3-EAGLE3` draft head(BF16,6.5GB)
- 量化 patch(见上级目录 `sglang_patches/modified/`,EAGLE3 依赖模型能正确加载量化权重)
- `sitecustomize.py` 注册 `minimax_m3_sparse` layer type(见上级目录)

## 解决的上游缺口

EAGLE3 + MiniMax-M3 + sparse attention 三者组合在 sglang 上游有缺口:

1. **VL 类缺 EAGLE3 接口** — `MiniMaxM3SparseForConditionalGeneration` 没有
   `set_eagle3_layers_to_capture` / `get_embed_and_head` / aux-aware forward → 启动即 AttributeError
2. **aux 捕获链路断裂** — `MiniMaxM3Model.forward` 读 `_is_layer_to_capture` 但无代码 setattr → 补在 `set_eagle3_layers_to_capture`
3. **TARGET_VERIFY 字段全 None** — `ForwardBatch.init_new` 把 target_verify 归 decode 分支不填
   extend 字段,sparse backend 的 forward_extend 直接读 → 崩
4. **verify 时 seq_lens 语义** — verify 路径 `batch.seq_lens` 是 prefix(非 prefix+draft),
   sparse backend 误当 prefix+draft → attention 位置错 → 输出乱码。正确:
   `prefix_lens=seq_lens`,`seq_lens=prefix+D`
5. **verify 走 cuda graph 必崩 VMFault(两层根治)** — EAGLE3 是首个让 sparse **prefill** 进入
   cuda graph 的场景(无 EAGLE3 时 prefill=eager,decode=graph-safe decode kernel)。两层问题:
   - **score buffer 动态尺寸**:prefill 的 `score`/`topk` 张量按动态 `max_seqlen_k` 分配,
     capture(dummy seq_lens=1)与 replay(真实 seq_lens~2000)尺寸不一致 → 越界写 → VMFault。
     **Strategy B 修复**:复制 prefill kernel 为 verify kernel,score 第3维固定为
     `max_seqblock_k_upper=cdiv(context_len, block_size_k)`,capture/replay 形状恒定。
     kernel 内部仍按真实 seq_len 做 causal(无算力浪费)。Step3 加 `pos < pos_upper` OOB 双保险。
   - **forward_extend 临时 tensor**(bs>=并发阈值才崩的真正根因):verify 分支用临时
     `extend_seq_lens`(forward_extend 里 `torch.full` 新建)算 cu_seqlens/seq_lens。capture 的
     forward_batch 是局部变量,捕获后被 GC,`extend_seq_lens` 内存释放/复用;graph 记录了依赖
     该地址的 op,replay 时读老地址 → 垃圾值 → 越界。bs=1 临时 tensor 小碰巧不炸,bs=16 必崩。
     **graph-safety 修复**:verify 分支完全绕过 `extend_seq_lens`,改用 `forward_batch.seq_lens`
     (graph buffer,replay_prepare 填充,地址稳定)+ Python int D + `torch.arange(0,(bs+1)*D,D)`
     (Python int 输入,同 triton backend 的 qo_indptr,sglang 标准 graph-safe 模式)。

   标准 sglang 的 triton backend 不崩,是因为它的 `qo_indptr=torch.arange(...)` 用 Python int
   常量、`kv_indptr` 写入预分配 graph buffer,不依赖临时 tensor。本 patch 的 verify 路径对齐这个模式。

## 为什么放弃 Strategy A

Strategy A(`--speculative-attention-mode decode`,verify 路由到 decode kernel)能不崩,
但 **decode kernel 无 causal mask**(用启发式 init_blocks/local_blocks 选块),而 prefill 有
严格 causal(`off_q >= pos_k`)。verify 的 draft token 在 prefix..prefix+D 位置,decode kernel
假设 q 在 seq 末尾 → 位置错。HumanEval(代码补全,宽容)误判通过 96.95%,但**自然语言对话输出
garbage**(复读、混模板)。Strategy B 复制 prefill kernel 保持严格 causal,根治此问题。

Strategy A 代码(`_forward_verify_via_decode` / `_verify_decode_mode`)在 backend 里保留未调用,
可回滚参考。

## 与量化 patch 的关系

本目录只含 **EAGLE3 投机解码** 专属改动。量化适配(W4A16 MoE kernel、compressed_tensors、
sparse attention 共享内存等)在上级 `sglang_patches/modified/`,需先应用量化 patch 让模型能加载,
再应用本 EAGLE3 patch。

详见 `docs/MiniMax-M3-EAGLE3-work-log.md`。
