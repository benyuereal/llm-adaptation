# MiniMax-M3 EAGLE3 投机解码工作记录
> 创建日期: 2026-07-23 ｜ 更新: 2026-07-23 (cuda graph 适配完成, 4x 加速)
> 目标: 在海光 DCU (gfx936 / BW100) 上给 W4A16 moe-only 量化的 MiniMax-M3 接 EAGLE3 投机解码
> 关联: `MiniMax-M3-量化工作记录.md` (量化本体) ｜ sglang dev12695 (DCU dtk2604 build)
>
> **最终成果**: 纯 W4A16 eager 5 tok/s → EAGLE3 + cuda graph **16-22 tok/s (峰 21.7)**, accept 0.78,
> 输出正确 (纯文本, 不乱码不复读). = **4x over eager, 1.7x over 纯 W4A16 cuda graph (13 tok/s)**.
> Draft: `Inferact/MiniMax-M3-EAGLE3` (BF16, MXFP8 target 训, 与我们 W4A16 不一致但实测 accept 仍 0.78).

---

## 一、背景: MTP vs EAGLE3

最初需求是"给 MiniMax-M3 开 MTP"。澄清后明确:

| | MTP | EAGLE3 |
|---|---|---|
| Draft 来源 | 模型自带的 MTP head 权重 | 独立训练的 draft head checkpoint |
| 是否需训练 | 否(权重随模型发布) | 是(SpecForge/TorchSpec) |
| M3 是否有 | **没有** — config 有 `num_mtp_modules:7` 但 safetensors 里 0 个 `model.mtp.layers.*` 权重 | 需外部 draft |

**结论**: M3 这个 checkpoint 没有自带 MTP 权重(config 里 `num_mtp_modules:7` 只是架构能力声明),原生 MTP 路线走不通。只能用 EAGLE3(独立训练的 draft head)。参考的 MiniMax-M2.5 文档(`thoughtworks/MiniMax-M2.5-Eagle3`)用的也是 EAGLE3,不是 MTP。

---

## 二、Draft head 来源与结构

### 2.1 模型
- **仓库**: `Inferact/MiniMax-M3-EAGLE3` (modelscope), 已下到 `/models/Inferact/MiniMax-M3-EAGLE3`
- **文件**: `model.safetensors` (6.5GB), `config.json`
- **训练目标**: `MiniMaxAI/MiniMax-M3-MXFP8` (FP8 精度!) — 用 TorchSpec 训
- **aux hidden layers**: (2, 30, 57) + final layer

### 2.2 架构 (`LlamaForCausalLMEagle3`)

| 维度 | 值 | 与 M3 target 对齐? |
|---|---|---|
| hidden_size | 6144 | ✅ 一致 |
| vocab_size | 200064 | ✅ 一致 |
| num_hidden_layers | 1 (单层 draft) | — |
| num_attention_heads | 64 | — |
| intermediate_size | 18432 | — |
| fc_norm | true (3 个 aux 各一个 RMSNorm) | — |
| norm_output | true | — |
| draft_vocab_size | 200064 (与 target 同, 不裁剪) | — |
| tie_word_embeddings | false (运行时复用 target 的 embed/lm_head) | ✅ |

### 2.3 实际权重键 (17 个)
```
embed_tokens.weight        # 运行时被 target 覆盖
fc.weight                  # 3 个 aux hidden 拼接后投影
fc_norm.0/1/2.weight       # 3 个 aux 各一个 norm
layers.0.{input_layernorm, post_attention_layernorm, hidden_norm}.weight
layers.0.self_attn.{q,k,v,o}_proj.weight
layers.0.mlp.{gate,up,down}_proj.weight
lm_head.weight             # 运行时被 target 覆盖
norm.weight
```
结构与 sglang `srt/models/llama_eagle3.py` 的 `LlamaForCausalLMEagle3` 完全匹配(`fc_norm`/`norm_output`/`draft_vocab_size` 字段均覆盖)。

### 2.4 README 性能 (target=MXFP8, vLLM, TP=4, num_spec_tokens=3, greedy topk=1)
- GSM8K/HumanEval/MATH500: 平均接受长度 ~3.5, 接受率 ~84%
- MT-Bench/低熵合成数据: 接受长度 ~2.7, 接受率 ~57%
- **注意**: 这些是 MXFP8 target 的数字。我们用 W4A16 moe-only target, 精度不同, 接受率会偏低。

---

## 三、量化模型与 sglang 的 EAGLE3 支持核查

### 3.1 量化模型 (W4A16 moe-only) 与 draft 对齐
| 检查项 | draft 期望 | W4A16 产物 | 结论 |
|---|---|---|---|
| hidden_size | 6144 | 6144 | ✅ |
| vocab_size | 200064 | 200064 | ✅ |
| architectures | M3 | `MiniMaxM3SparseForConditionalGeneration` (VL 类) | ✅ |
| tie_word_embeddings | false | false | ✅ |
| MoE 量化范围 | — | moe-only (embed/lm_head 原精度) | ✅ 共享无精度错配 |

### 3.2 sglang EAGLE3 target 侧接口 (核查 minimax_m3.py / minimax_m3_vl.py)

sglang 里 M3 有**两个模型类**:

| 类 | 文件 | EAGLE3 接口 |
|---|---|---|
| `MiniMaxM3SparseForCausalLM` (text-only) | minimax_m3.py | **有** `set_eagle3_layers_to_capture` / `get_embed_and_head` / aux-aware forward |
| `MiniMaxM3SparseForConditionalGeneration` (VL) | minimax_m3_vl.py | **没有** — 这就是量化模型加载的类 |

draft 侧: sglang `llama_eagle3.py` 能加载该 draft; spec_registry 里 EAGLE3/NEXTN 均为保留算法; server_args 有完整 `--speculative-*` flag。

---

## 四、踩坑: sglang M3 EAGLE3 两处缺口 (上游 bug)

### 缺口 1: VL 类缺 EAGLE3 接口 (直接崩溃)
首次启动报错:
```
AttributeError: 'MiniMaxM3SparseForConditionalGeneration' object
has no attribute 'set_eagle3_layers_to_capture'
```
原因: `model_runner.init_aux_hidden_state_capture()` (model_runner.py:904) 调
`model.set_eagle3_layers_to_capture()`, 但 VL 类没这个方法 (只在 text-only 类上)。
量化模型 config.architectures = VL 类, 所以加载 VL 类 → 崩溃。

### 缺口 2: M3 aux 捕获链路本身是断的 (更隐蔽)
即便补上接口, `MiniMaxM3Model.forward` (minimax_m3.py:1016-1020) 用
`getattr(layer, "_is_layer_to_capture", False)` 判断要不要把 `aux_hidden_states`
列表传进 decoder layer。但**全 sglang 没有任何代码把 `layers_to_capture` 转成
layer 上的 `_is_layer_to_capture=True`**。

对比: `qwen3_moe.py:939` 的 `set_eagle3_layers_to_capture` 里有
`setattr(self.layers[layer_id], "_is_layer_to_capture", True)` —— M3 缺这一步。
即 text-only 类的 EAGLE3 路径也是断的 (设了 `layers_to_capture` 但 layer 没被标记 →
`aux_hidden_states` 永远空 → 返回纯 hidden, 不是元组)。

> 这说明 sglang 上游对 M3 的 EAGLE3 支持是半成品: 接口写了, 但 aux 捕获链路没接通。

---

## 五、修复: 直接改 site-packages (沿用 W4A16 patch 方式)

### 5.1 备份与改动位置
- **原文件备份**: `/models/sglang_backup/minimax_m3_vl.py` (与 W4A16 patch 的 backup 同目录)
- **改动文件**: `/usr/local/lib/python3.10/dist-packages/sglang/srt/models/minimax_m3_vl.py`
- **曾尝试 monkey-patch** (`sglang_patches/added/minimax_m3_vl_eagle3.py`), 但 sglang TP=4 起 4 个 scheduler 子进程, monkey-patch 只在主进程生效, 子进程仍是原类 → 报错不变。故改为直接改文件 (子进程 fork 后继承改动)。

### 5.2 改动 (全部针对 VL 类 `MiniMaxM3SparseForConditionalGeneration`)

| 改动 | 内容 | 依据 |
|---|---|---|
| 补 `set_eagle3_layers_to_capture` | 转发给 `self.model`, 设 `self.model.layers_to_capture`, 并对每个待捕获 layer `setattr(_is_layer_to_capture=True)` (修复缺口 2) | 仿 qwen3_moe.py:939 |
| 补 `get_embed_and_head` | `return self.model.embed_tokens.weight, self.lm_head.weight` | 仿 text-only 类 minimax_m3.py:1130 |
| 补 `capture_aux_hidden_states` 属性 | set_eagle3 时设 True, 默认 False | — |
| 重写 `forward` | capture 模式下解包 `general_mm_embed_routine` 返回的 `(hidden, aux)` 元组, 把 aux 传给 logits_processor | 仿 text-only 类 minimax_m3.py:1147-1154 |

**默认 aux layers**: `set_eagle3_layers_to_capture(None)` → `[2, num_layers//2, num_layers-3]`
= `[2, 30, 57]` (N=60), 与 draft 训练用的 (2,30,57) 一致。draft config 无 `eagle_config`
字段 → model_runner 走 except → `layer_ids=None` → 用默认 → 自动对齐, 无需改 draft config。

### 5.3 影响范围
- 仅 `capture_aux_hidden_states=True` (开了 EAGLE3) 时走新 forward 路径
- 平时 (普通 W4A16 推理) `capture_aux_hidden_states=False`, 新 forward 行为与原 forward 完全一致 (都调 `general_mm_embed_routine` + `logits_processor`, 只是多一个 `getattr` 判断)
- **不改 site-packages 原文件**, sglang 升级不丢 patch; 去掉 import 即完全回滚

### 5.4 已验证 (不实际起服务)
- ✅ patch import 生效, VL 类有了 `set_eagle3_layers_to_capture` / `get_embed_and_head`
- ✅ mock 验证: layer 2/30/57 被标记 `_is_layer_to_capture=True`, layer 0 未标记; 显式 layer_ids 走 +1 偏移
- ✅ sglang 参数解析: EAGLE3 归一化 num_steps=3 / topk=1 → num_draft_tokens=4 (=steps+1) 正确
- ✅ 幂等: 重复 import 不报错

### 5.5 实测验证结果 (已全部验证)
- ✅ DCU 上 draft worker 加载成功 (`LlamaForCausalLMEagle3`, 2.2GB/卡)
- ✅ verify 阶段执行正常, 输出正确 (纯文本不乱码不复读)
- ✅ 接受率 accept 0.78 (eager 0.4 → cuda graph 0.78, 接近 README 的 0.84)
- ✅ 加速比: eager 1.5x (5→7.5), cuda graph **4x (5→21.7)** — 见 5.6
- 注: W4A16 瓶颈是 MoE int4 GEMM, 但 cuda graph 消掉 launch 开销后 spec decode 收益放大,
  实际加速远超之前 "MoE 瓶颈可能打折" 的预期. draft 训练 target (MXFP8) 与我们 (W4A16)
  不一致, 但实测 accept 仍 0.78, 说明 hidden 分布差异影响有限.

### 5.6 cuda graph 适配 (最终性能突破)

eager 模式 (disable-cuda-graph) 下 1.5x (5→7.5 tok/s), accept~0.4. 开 cuda graph 后需修一连串 graph-unsafe 的 host 同步, 最终 4x (5→21 tok/s), accept~0.78.

**根因**: sglang 的 M3 sparse attention **decode kernel 专为 cuda graph 设计** (grid 不依赖 seq_len, 注释 "grid independent of seq_len for cuda graph"), 但 **prefill kernel 不是** — EAGLE3 verify 走 forward_extend → sparse prefill, 撞上 prefill 的 host 同步. 纯 W4A16 (无 EAGLE3) 捕获 DECODE mode 走 decode kernel, 所以能 graph; EAGLE3 捕获 TARGET_VERIFY mode 走 prefill, 所以崩.

**修的 host 同步 (全部 `is_current_stream_capturing()` 判断或静态计算)**:
1. `forward_extend` 的 `extend_seq_lens.device` None → capture 路径 `init_forward_metadata_capture_cuda_graph` 不补 extend 字段, 在 forward_extend 开头兜底 (graph-safe: `torch.full` 固定 shape).
2. `forward_extend` 的 `torch.all(raw_seq_lens >= prefix_plus_extend)` → host 同步, 改用 `forward_mode.is_target_verify()` 静态判断.
3. `forward_extend` 的 `extend_prefix_lens.cpu().tolist()` → capture 时跳过 _cpu 字段.
4. `forward_extend` 的 `cu_seqlens[-1].item()` → capture 时用静态 `q.shape[0]`.
5. `get_cu_seqblocks` (utils.py) 的 `seqblocks_q.sum().item()` ×2 → capture 时用静态上界 `batch_size * max_seqblock` (graph-safe, 不读 tensor 值).

**修的字段语义 (graph 路径漏了 eager 路径的坑 D 修复)**:
6. `init_forward_metadata_capture_cuda_graph` / `replay` 的 `_max_seqlen_q=1, _max_seqlen_k=max(seq_lens)` 是 decode 语义 → verify 下应为 `_max_seqlen_q=draft_token_num, _max_seqlen_k=max(seq_lens)+draft_token_num` (seq_lens 是 prefix).
7. `forward_extend` 的 `prefix_lens = forward_batch.extend_prefix_lens` → graph 下是 capture 时的固定 tensor (stale), 改用 `raw_seq_lens` (= `forward_batch.seq_lens`, buffer 引用, replay 时是真实 prefix).

**改动文件**: `minimax_sparse_backend.py` (1-7 全部) + `utils.py` (5). 备份 `sglang_backup/`, patch 在 `sglang_patches/modified/`.

**最终性能** (W4A16 moe-only target + Inferact BF16 draft, TP=4, B=1, temp=0.7, cuda-graph-max-bs=8):
- 纯 W4A16 eager: 5 tok/s
- 纯 W4A16 cuda graph: 13 tok/s
- EAGLE3 eager: 7.5 tok/s (accept~0.4)
- **EAGLE3 cuda graph: 16-22 tok/s (峰值 21.7), accept 0.58-0.87 (平均 ~0.78), accept len 3.4**
- = 4x over eager, 1.7x over 纯 graph, accept 接近 README 的 0.84

> accept 从 eager 的 0.4 涨到 graph 的 0.78, 可能因 eager 下 attention 不稳定; graph 路径更稳定反而拉高 accept. 输出经 chat.py 验证正常 (不乱码不复读, 思考+代码连贯).

### 5.7 实测踩坑 (启动后)

跑通 patch 后, 启动又依次撞到两个 sglang 上游与 EAGLE3 的兼容 bug, 均已修 (直接改 site-packages + 备份):

**坑 A: `MiniMaxM3Model` 无 `.config` 属性**
`set_eagle3_layers_to_capture` 里取 `self.model.config.num_hidden_layers` 报 `AttributeError`.
原因: text-only 类 `self.config` 是 text_config, 但 VL 类 `self.config` 是顶层 VL config (text 在 `self.config.text_config`), 且 `MiniMaxM3Model` 构造时不保留 `.config` 引用.
修法: 改成 `self.config.text_config.num_hidden_layers` (VL 类自身代码也是这么取的).

**坑 B: EAGLE3 TARGET_VERIFY 时 sparse backend 取 `max(None)` 崩**
`minimax_sparse_backend.py:init_forward_metadata` 用 `mode.is_extend()` 判断, TARGET_VERIFY 的 `is_extend()==True`, 故进 extend 分支取 `forward_batch.extend_seq_lens_cpu`. 但 `ForwardBatch.init_new` 把 TARGET_VERIFY 归到 **decode 分支** (`if is_decode() or is_target_verify()`), 不填 `extend_seq_lens_cpu` → 保持 None → `max(None)` 报 `TypeError: 'NoneType' object is not iterable`.
根因: sparse backend 用 `is_extend()` 分类, init_new 用 `is_target_verify()` 分类, 两者对 TARGET_VERIFY 的归类不一致 (sglang 上游 bug, 非 MiniMax 专属, 但只有 sparse backend 的模型触发).

**坑 C: 同根 — `forward_extend` 用 `extend_seq_lens.device` 崩**
修完坑 B 后, `forward_extend` (line 216) 又取 `forward_batch.extend_seq_lens.device` → None 崩. 同根: TARGET_VERIFY 走 decode 分支, **所有** extend 字段 (`extend_seq_lens` tensor / `extend_seq_lens_cpu` list / `extend_prefix_lens`) 都没填. 且基类 `AttentionBackend.forward` 也用 `is_decode()` 分发, TARGET_VERIFY → `forward_extend`, 所以所有 backend 在 verify 都走 extend 路径, 只是多数 backend 不直接读这些字段所以没崩.
修法 (坑 B+C 合一): 在 `init_forward_metadata` 里, 若 `extend_seq_lens is None`, 用 `spec_info.draft_token_num` 一次性补全 `extend_seq_lens` (tensor) / `extend_seq_lens_cpu` (list) / `extend_prefix_lens` / `extend_prefix_lens_cpu`. 改动文件: `minimax_sparse_backend.py`, 备份 `sglang_backup/minimax_sparse_backend.py`.

**坑 D: verify 时 seq_lens 语义 — 输出乱码/复读的真正根因**
修完坑 B/C 服务能跑, accept rate 正常 (0.3~1.0), 但**输出乱码 + 复读** (`<mm:think>...user has been thinking...`), 而纯 W4A16 同 prompt 正常.
根因: sparse backend `forward_extend` 给 kernel 的契约是 `seq_lens = prefix + extend`, `prefix_lens = prefix`. 但 verify 路径里:
- `prepare_for_v2_verify` **不给 `batch.seq_lens` 加 draft** (对比 `prepare_for_extend_to_fill_draft_kvcache` 会 `seq_lens += draft`). verify 后 scheduler 用 `seq_lens + accept_lens` 更新, 所以 verify 时 `seq_lens` 必须保持 **prefix** (不含 draft).
- 标准 backend (triton/flashinfer) 不直接读 `extend_seq_lens`, 用 `spec_info.positions`, 所以没暴露此问题. M3 sparse backend 直接读, 就错.
- 我第一版补 `extend_prefix_lens = seq_lens - draft` 是**错的** (基于"seq_lens 已加 draft"的错误假设), 导致 prefix 少算 draft, attention 位置错 → accept 的 token 写错位置 → 乱码/复读.

正确修法 (对照 `prepare_for_extend_to_fill_draft_kvcache` 的自洽三元组):
- `extend_prefix_lens = seq_lens` (verify 时 seq_lens 就是 prefix, 不减 draft)
- `extend_seq_lens = draft_token_num`
- kernel 的 `seq_lens` = `prefix + draft` (在 `forward_extend` 里用 `prefix_lens + extend_seq_lens` 重建, 不改 `forward_batch.seq_lens` 以免破坏后续 `+accept_lens`)
- `_max_seqlen_k` = `max(prefix + draft)`
- 用 `seq_lens >= prefix + extend` 判断是 normal extend (seq_lens 已含 extend, 不重建) 还是 verify (重建), 两种情况都对.

> 这几个坑 fork (tails-mpt) 也没修 — fork 做的是 M2 (text-only, 无 VL 类问题) 且不用 M3 的 sparse backend. 属于 M3 + EAGLE3 + sparse attention 三者组合的独有缺口.

---
参考 `https://github.com/tails-mpt/sglang` (MiniMax-M2.5 Eagle3 的 fork) 的相关 commit:

| fork commit | 改动 | 对我们 W4A16 场景 | 我们状态 |
|---|---|---|---|
| `b5927f9a7c` feat: MiniMax-M2.5 Eagle3 support | minimax_m2.py 默认层 [2,30,57]→[1,30,58] + logits_processor aux→bf16 cast | 默认层不适用 (M3=60层, draft 用 (2,30,57)); aux cast 见下行 | 默认层用 [2,30,57] ✅ |
| `b5927f9a7c` / `9049265fb2` logits_processor aux→bf16 cast | aux_hidden_states 拼接后 cast 到 bf16 (FP8 target 产生 float32 aux) | **可能需要** — W4A16 MoE dequant 可能产生 float32 aux, 跑起来若报 dtype mismatch 再加 | ⏳ 暂未加, 待实测 |
| `9049265fb2` llama.py set_embed→bf16 cast | 共享 embed/lm_head 时 cast 到 bf16 (FP8 target embed fp16/f32) | 不触发 — 我们 embed/lm_head 本就是 bf16 (moe-only 不量化) | 不需要 |
| `ea6c44888b` / `b3e73aba38` llama_eagle3 FC/embeds dtype cast | FP8 aux float32 vs bf16 FC 的 dtype 修复 | 不触发 — 我们 aux 是 bf16 | 不需要 |
| `0675f9531c` final-layer aux capture | 循环外捕获 final layer hidden | **不该加** — draft FC in_features=18432=3×6144, 只吃 3 个 aux (2,30,57); 加 final 会变 4 个 → FC 维度不匹配报错 | 正确未加 ✅ |
| `b3e73aba38` multi-layer EAGLE3 (raise→warning) | 允许 num_hidden_layers>1 | 不触发 — 我们 draft num_hidden_layers=1 | 不需要 |
| **M3 VL 类缺接口** (fork 也没做 M3) | set_eagle3_layers_to_capture / get_embed_and_head / aux-aware forward + `_is_layer_to_capture` setattr | **必须** — M3 VL 类缺这些, fork 只做了 M2 (text-only 已有接口) | ✅ 已补 (5.2) |

**结论**: fork 的 EAGLE3 修复大部分针对 **FP8 target** 的 dtype 问题 (float32 aux/embed), 我们 W4A16 (bf16) 多数不触发。唯一可能需要的 `logits_processor aux→bf16 cast` 留待实测 — 若启动后 draft FC 报 `expected mat1 and mat2 to have the same dtype` 再加 (条件触发, 零风险)。M3 VL 类的接口缺失是 fork 也没覆盖的 (fork 只做 M2), 我们已补齐。

---

## 六、启动脚本

**位置**: `/models/minimax_w4a16_eagle3.sh` (端口 8081)

### 6.1 关键设计
- **用 `python3 -c` 包装启动**, 不用 `sglang serve` CLI:
  ```bash
  python3 -c "
  import minimax_m3_vl_eagle3  # EAGLE3 patch (import 即生效)
  from sglang.launch_server import main
  main()
  " --model-path ... --speculative-algorithm EAGLE3 ...
  ```
  确保 patch 在 sglang 加载模型前生效。
  - 注意: `prepare_server_args` (server_args.py:7683) 的 parser 是 `prog="sglang serve"`,
    直接 `parser.parse_args(argv)`, **不需要 `serve` 子命令前缀** (那是 `sglang` CLI 的子命令,
    不是 launch_server 的)。所以 `--model-path` 直接起头。
- `PYTHONPATH` 加上 patch 目录: `/models/quant-eagle3-hygon/sglang_patches/added`

### 6.2 EAGLE3 配置 (README 验证过的 greedy 配置)
| 参数 | 值 | 说明 |
|---|---|---|
| `--speculative-algorithm` | EAGLE3 | |
| `--speculative-draft-model-path` | `/models/Inferact/MiniMax-M3-EAGLE3` | M3 不在 DeepSeek MTP 自动复用 model_path 的列表, 必须显式给 |
| `--speculative-num-steps` | 3 | README `num_speculative_tokens=3` (树深度) |
| `--speculative-eagle-topk` | 1 | greedy; topk=1 在 DCU 上最稳 |
| `--speculative-num-draft-tokens` | 不设 | topk=1 时 sglang 自动改成 steps+1=4 (server_args.py:3802) |
| `--mem-fraction-static` | 0.93 | 比纯 W4A16 (0.97) 降, 给 draft (~1.6G/卡) + spec 调度留余量 |
| `--disable-cuda-graph` | 是 | DCU 上 EAGLE draft graph 未验证, 先 eager 跑通 |

### 6.3 与纯 W4A16 脚本 (`minimax_w4a16.sh`) 的差异
- 加 EAGLE3 三参数 + draft path
- mem-fraction 0.97 → 0.93
- 启动方式 `sglang serve` → `python3 -c` 包装 (为加载 patch)
- 端口 8081 (与原脚本同, 互斥)

---

## 七、待办 / 风险 (现状)

1. ~~实测 DCU 加载~~ ✅ 已验证: draft worker 加载 + spec 调度 + verify 全部正常.
2. ~~接受率风险~~ ✅ 实测 accept 0.78 (eager 0.4, cuda graph 0.78), 远超 "<60%" 预期.
   draft (MXFP8 训) 与 W4A16 target 不一致的影响有限, 无需重训.
3. ~~加速比风险~~ ✅ 实测 4x (5→21.7 tok/s), 远超 "可能远低于 1.7-2.1x" 预期.
   cuda graph 消掉 MoE launch 开销后, spec decode 收益放大.
4. **text-only 类 EAGLE3 仍断**: 本 patch 只修 VL 类. 若将来用 text-only M3
   (`MiniMaxM3SparseForCausalLM`), 其 `set_eagle3_layers_to_capture` 同样缺 setattr layer
   那步, 需另补 (当前量化产物用 VL 类, 不需要).
5. **视觉请求 + EAGLE3 未测**: 纯文本已验证正常. 图片/视频请求走同一 `MiniMaxM3Model.forward`,
   aux 捕获逻辑一致, 理论上能跑, 但 draft 训练时未见视觉 hidden → accept 可能掉.
   不会硬报错 (大概率), 但 spec 收益打折. 若需视觉+EAGLE3, 要单独验证 accept.
6. **2x+ 进一步提升**: 现成 draft 已是最佳 (无适配 moe-only W4A16 的现成货, 见九).
   要更高只有重训 draft (TorchSpec, 成本高, DCU 跑训练流程有障碍) — 非当前优先级.

---

## 八、文件清单

| 文件 | 作用 |
|---|---|
| `/models/Inferact/MiniMax-M3-EAGLE3/` | draft head (下载, BF16, 6.5GB) |
| `/models/minimax_w4a16_eagle3.sh` | 启动脚本 (端口 8081, cuda-graph-max-bs=8) |
| `/models/sglang_w4a16_eagle3.log` | 运行日志 |

**sglang patch** (直接改 site-packages, 备份 + modified/.patch 规范):

| 文件 (site-packages) | 备份 | modified/.patch | 改动 |
|---|---|---|---|
| `srt/models/minimax_m3_vl.py` | `sglang_backup/minimax_m3_vl.py` | `sglang_patches/modified/minimax_m3_vl.py{,.patch}` | VL 类 EAGLE3 接口 (5.2) |
| `srt/layers/attention/minimax_sparse_backend.py` | `sglang_backup/minimax_sparse_backend.py` | `sglang_patches/modified/minimax_sparse_backend.py{,.patch}` | verify 字段补全 + seq_lens 语义 + cuda graph 7 处 (5.6/5.7) |
| `srt/layers/attention/minimax_sparse_ops/common/utils.py` | `sglang_backup/utils.py` | `sglang_patches/modified/utils.py{,.patch}` | get_cu_seqblocks graph-safe (5.6-5) |

**测试**: `sglang_patches/tests/test_m3_eagle3_verify_sparse.py` (不起服务, 验证 verify 字段补全逻辑).

**回滚**: `cp /models/sglang_backup/<file> /usr/local/lib/python3.10/dist-packages/sglang/<原路径>`.

---

## 九、Draft 选型排查 (确认现成 draft 无适配货)

为提升 accept (eager 阶段 0.4), 排查了 HuggingFace 所有 MiniMax-M3 EAGLE3 draft (7 个):

| Draft | 训练 target | 对我们 moe-only W4A16 | 结论 |
|---|---|---|---|
| `Inferact/MiniMax-M3-EAGLE3` (采用) | MXFP8 (FP8 全模型) | 不一致, 但实测 accept 0.78 | ✅ 最佳现成选择 |
| `Sebesky/MiniMax-M3-EAGLE3-RTN-INT4` | GPTQ (int4 全模型) | 更不一致 + draft 自身 int4 | ❌ DCU 加载崩 (无 `gptq_marlin_repack`), 已删 |
| `Inferact/MiniMax-M3-EAGLE3-GQA` | MXFP8 (GQA 架构对齐) | target 同 MXFP8, 提升不确定 | 未测 (MHA 版已够) |
| `Inferact/MiniMax-M3-EAGLE3-GQA-NVFP4` | MXFP8 + NVFP4 (NVIDIA) | 硬件不匹配 | ❌ |
| `amd/MiniMax-M3-EAGLE3.1` | MXFP4, MI350X 专用 | 硬件不匹配 (gfx936) | ❌ |
| `tonjum/...-GGUF` | GGUF (llama.cpp) | 格式不对 | ❌ |

**结论**: 没有适配 moe-only W4A16 RTN 的现成 draft (我们自己量化的, 没人专门训).
现有 `Inferact/MiniMax-M3-EAGLE3` 已是最佳, cuda graph 下 accept 0.78 接近 README 的 0.84,
无需换. 2x+ 进一步提升只有重训 draft (见七-6).

**INT4 draft 适配性**: DCU (gfx936) 无 NVIDIA `gptq_marlin_repack`, 任何 int4 dense draft
(走 compressed-tensors W4A16 GPTQ Marlin 路径) 都加载崩. 只能用 BF16 draft.

---

## 十、待办: AWQ INT4 target 配 BF16 draft (未实测)

### 背景
另有一份 AWQ INT4 量化 target (`/models/MiniMax-M3-AWQ-INT4`), 全模型量化 (compressed-tensors
pack-quantized, observer=mse, group=32, 非对称 int8 zp, W4). 评估能否复用现有 BF16 draft
(`Inferact/MiniMax-M3-EAGLE3`) 跑 EAGLE3 投机解码. **DCU int4 dense 加载已单独解决 (target 可独立推理).**

### 可行性预估 (耦合点 3 项)
| 耦合点 | AWQ INT4 target 情况 | 与 BF16 draft | 风险 |
|---|---|---|---|
| hidden_size / vocab / 60层 / VL类 | 6144 / 200064 / 60 / `MiniMaxM3SparseForConditionalGeneration` | 完全一致 | ✅ 无 |
| DCU int4 dense 加载 | 已解决 (target 单独可跑) | — | ✅ 无 |
| **aux 层 dtype** | 第2层在 ignore → bf16; 第30/57层 dense 被量化 int4 → 反量化后**可能 float32** | draft FC 是 bf16 | ⚠️ **唯一风险** |

aux 层号仍为 `[2,30,57]` (N=60). 关键区别:
- **现在 W4A16 moe-only**: dense 全 bf16 → 3 个 aux 全 bf16 → draft FC 无 dtype 问题.
- **AWQ INT4**: 前3层(lay 0-2)在 ignore 列表保持 bf16, 其余 dense 层(lay 3-59)int4.
  所以 aux 中**第2层 bf16, 第30/57层可能 float32**.

### 风险点 (记录, 不预先改)
若 AWQ 反量化路径输出 float32 aux, 会在 `llama_eagle3.py` 的 `self.fc(hidden_states)`
(line 225, hidden 来自 `forward_batch.spec_info.hidden_states` line 217) 报
`expected mat1 and mat2 to have the same dtype`. 概率估约 50/50 (sglang 部分反量化路径会
cast 回模型 dtype, 不一定留 float32, 需实测确认).

**待实测确认后再决定是否打 patch**, 不预先改 (避免对已验证通过的 W4A16 路径引入风险).
若实测报 dtype mismatch, 需加的 cast (沿用 fork tails-mpt 验证过的方案, 3 处, patch 规范同 5.1):
1. `srt/models/llama_eagle3.py` line ~217: `forward_batch.spec_info.hidden_states` 喂 `self.fc` 前 cast 到 draft dtype (`self.embed_tokens.weight.dtype`).
2. `srt/models/llama_eagle3.py` line ~212: embeds cast 到 draft dtype (embed 共享自 target, 本应 bf16, 保险).
3. `srt/models/minimax_m3_vl.py` line ~244: aux unpack 后立即 cast 到 `self.lm_head.weight.dtype` (lm_head 在 ignore → bf16), 从源头统一 aux 为 bf16, 下游 logits_processor / draft 全 bf16.

### accept 预估
AWQ 有校准 (mse observer), hidden 分布比 RTN 接近原模型; 但 draft 训练于 MXFP8 target,
与 AWQ int4 hidden 分布不完全一致. 预估 accept **0.6~0.8**, 吞吐净收益取决于 accept × draft 开销,
需实测. (现 W4A16 + cuda graph: accept 0.78, 21.7 tok/s.)

### 行动项 (待办)
- [ ] 用现有 `minimax_w4a16_eagle3.sh` 改 target 路径指向 AWQ INT4, 实测能否起服务 + 推理正常
- [ ] 若起服务即报 dtype mismatch → 打上述 3 处 cast patch (先备份 `llama_eagle3.py` 原始版到 `sglang_backup/`)
- [ ] 实测 accept / 吞吐, 与 W4A16 (0.78 / 21.7 tok/s) 对比
- [ ] 对比 AWQ INT4 target 单独推理精度 (gpqa 等) 与 W4A16 moe-only, 确认精度不降 (用户硬约束: 保持精度)
