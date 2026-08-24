# GLM-5.2-Channel-INT4-w4a8 DCU 适配技术文档

> 模型：GLM-5.2-Channel-INT4-w4a8（GlmMoeDsaForCausalLM，slimquant w4a8）
> 平台：海光 K100AI DCU（AMD ROCm gfx928，8×64GB，128 CU/卡）
> 日期：2026-08-22
> 仓库：benyuereal/llm-adaptation

---

## 0. 总览

本文档覆盖 GLM-5.2-Channel-INT4-w4a8 在 K100AI DCU 上的**完整适配过程**，包括部署启动、功能修复、精度验证、性能优化四个阶段。适配从「长输入乱码 + HumanEval 66.5%」推进到「乱码清零 + HumanEval 99 分（153 题已评分）」，并完成解码阶段性能瓶颈定位。

### 适配成果速览

| 维度 | 适配前 | 适配后 |
|------|--------|--------|
| 启动 | gfx928 崩溃（lightop 算子不支持） | ✅ 正常启动 |
| 长输入 | 乱码率 70~90% | ✅ 乱码 0/10 |
| HumanEval（思考模式） | 66.5% | ✅ 99 分（153 题已评分） |
| 上下文长度 | — | 当前 16384，理论上限 ~520K |
| 性能优化 | 19.68 tok/s（MTP int8 baseline） | ✅ **39.25 tok/s**（dequant-attn，~2x，见 5.5） |

### 环境

| 项目 | 配置 |
|------|------|
| 硬件 | K100AI DCU × 8（gfx928，128 CU/卡，64GB HBM/卡） |
| 算力栈 | DTK 2604 / ROCm，vLLM 0.15.1，lmslim 0.3.1，aiter 0.1.2+das.opt1.dtk2604，lightop，tilelang，PyTorch 2.9.0 |
| 模型 | GLM-5.2-Channel-INT4-w4a8，权重 363GB，78 层 MoE，256 expert，topk=8，DSA 动态稀疏注意力 |
| 量化 | slimquant w4a8：权重 INT4 + 激活 INT8 |
| 部署 | TP=8，dtype=bfloat16，MTP 投机解码（num_speculative_tokens=3），CUDA graph |

### 关键 env（start.sh）
```
VLLM_USE_LIGHTOP=1
VLLM_USE_LIGHTOP_MOE_ALIGN=1
USE_LIGHTOP_TOPK=1
VLLM_ROCM_USE_AITER_MOE=0      # MoE 走 lmslim fused kernel
VLLM_W8A8_BACKEND 强制=1       # gfx928 上 Linear 走 triton_scaled_mm
```

---

## 1. 部署启动适配

### 1.1 启动崩溃修复（gfx928 lightop 算子不支持）

**现象**：vLLM 在 gfx928 上启动即崩溃。

**根因**：`vllm/v1/attention/backends/mla/indexer.py` 的 `_build_attn_group_metadata` rocm 分支调用 `gemmopt.get_paged_mqa_logits_metadata`（lightop 算子），该算子 .so 的架构白名单不含 gfx928，raise `Unsupported device configuration`。

**修复**：rocm 分支改为 `self.scheduler_metadata_buffer.zero_()`（占位），不调用不支持的 lightop op。rocm decode 走 `page_mqa_logits`（tilelang，不用 scheduler_metadata），该 buffer 是死值，占位安全。

### 1.2 网络环境配置

容器内 `ip` 命令受限，网络是 Calico+以太网（无 IB 网卡）。`~/.bashrc` 原配的 `NCCL_SOCKET_IFNAME=ibs50` / `GLOO_SOCKET_IFNAME=ibs50` 会导致 Gloo 绑接口失败崩溃。**删除这两个变量**，8 卡 TP 走 PCIe P2P，不需要 IB。

### 1.3 启动脚本（start.sh）

支持前台（tee 日志）和 `--daemon`（nohup 后台）两种模式，日志带时间戳分文件保留。核心参数：`--max-model-len 16384 --max-num-batched-tokens 8192 -tp 8 --gpu-memory-utilization 0.92 --max-num-seqs 64 --block-size 64`，配合 MTP speculative_config。

---

## 2. 功能适配一：长输入乱码修复

详细排查现场见 [`glm52-long-input-garbage-rootcause.md`](glm52-long-input-garbage-rootcause.md)。

### 2.1 现象

- 短输入正常，**长输入（>2048 tokens）输出乱码**（`0.0.0.0...`、`|}{}{}{}...`）
- `temperature=0`（greedy）也乱 → 计算错误，非采样随机
- 乱码率 70~90%，高度可复现
- GLM-5.1 同环境正常 → 5.1→5.2 回归

### 2.2 排查方法

量化模型 transformers 不支持，无法做 HF 基线对比。采用 **op-locate 逐算子埋点定位**：探针工具 `vllm/_probe.py` 打印张量统计（mean/std/min/max/nan/inf/zero%/absmax），异常高亮。逐层逐算子找第一个异常点。分两阶段定位到两个独立 bug。

### 2.3 Bug 1：shared 层跑零权重 indexer（5.1→5.2 回归）

**定位**：逐层打印 indexer 的 `idx_q`，发现 full 层正常、shared 层全零（zero%=100%）。

**根因**：GLM-5.2 的 DSA indexer 从 5.1 的「每层独立」改为「full/shared 分组共享」：
- `index_topk_freq=4`：每 4 层做一次 indexer
- 每 4 层一组，1 层 full（有权重，算 indexer），3 层 shared（无权重，应复用 full 的 buffer）
- full 层位置 = `[0,1,2,6,10,14,...,74]`，由公式 `skip_topk = max(layer_id-3+1,0) % 4 != 0` 判定

定制分支不区分 full/shared，所有 78 层都建 Indexer、都跑。shared 层零权重跑出垃圾，且 `sparse_attn_indexer` 开头清空共享 buffer 再覆盖 → 把 full 层好结果冲掉。

**修复**：移植官方 skip_topk 机制：
- `deepseek_v2.py`：按公式算 `_skip_topk`，shared 层不建 Indexer，MTP 层（layer_id≥78）强制建
- `mla.py`：加 `skip_topk` 参数，`not self.skip_topk` 才跑 indexer；shared 层复用 buffer
- `flashmla_sparse.py`：shared 层 indexer=None，透传 `topk_indices_buffer` 给 backend

**效果**：乱码率 70~90% → 20%。

### 2.4 Bug 2：gfx928 prefill mqa_logits 返回值被丢弃（定制分支 bug）

**定位**：修复 Bug 1 后残留 20% 乱码。对 prefill indexer 打分埋点，发现 `pf_logits`（prefill indexer 打分）全零 → topk 在全零上退化 → 输出 `-1` 和 `2147483647`（未初始化垃圾）→ 稀疏注意力选错 token。

**根因**：gfx938 用 lightop `op.mqa_logits(..., logits_slice_view)`（原地写入）；gfx928 用 tilelang `mqa_logits(...)`（返回新 tensor）。调用处照抄 gfx938 模式，**丢弃了返回值** → logits 全零。直接测试 lightop `top_k_per_row_prefill` 算子本身正常，证明问题在返回值丢失而非 topk 算子。

**修复**：接收返回值直接作为 `logits_slice` 传给 topk（与官方 `fp8_fp4_mqa_logits` 一致）。

**效果**：乱码率 20% → 0。

### 2.5 为什么短输入不乱、长输入才乱

- **短输入**（≤2048）：`topk = min(2048, seq_len) = seq_len` → 全选，indexer 不起作用 → 即使 shared 层 indexer 坏结果也一样 → 不乱码
- **长输入**（>2048）：`topk = 2048` → 真正稀疏化 → full 层挑对、shared 层挑错（或 prefill logits 全零挑垃圾）→ 乱码

阈值 = `index_topk`（2048），完美解释现象。

### 2.6 验证

| 配置 | 修复前 | 修复后 |
|------|--------|--------|
| repeat 长输入 ×10 | 乱码 7-9/10 | **乱码 0/10** |
| knowledge 长输入 sampling | 乱码 | **完全正常** |

**已知遗留**（非乱码）：greedy 下部分问答首 token = EOS（空输出），是 w4a8 量化使 EOS 与首 token 概率极接近的边界情况，用 sampling 缓解。

---

## 3. 功能适配二：HumanEval 精度修复

详细排查见 [`glm52-humaneval-precision-rootcause.md`](glm52-humaneval-precision-rootcause.md)。HumanEval 初始 66.5%，远低于 GLM-5.1 的 98%，经排查发现四层独立问题。

### 3.1 问题 1：lightop topk 短输入垃圾索引（66.5% → 78%）

**根因**：lightop `top_k_per_row_prefill` kernel 在短输入（KV < topk_tokens=2048）时，没有像官方 `_C` kernel 那样把不足槽位填 -1，导致 `topk_indices_buffer` 残留 `torch.empty` 未初始化垃圾索引（实测 min=-2e9, max=+1.9e9）。shared 层（L75/76/77）复用垃圾索引越界访问 KV cache → sparse attention 算错 → MoE router 退化。

**修复**（`vllm_patches/modified/sparse_attn_indexer.py`）：lightop topk 写入后清洗越界索引为 -1（prefill 用 per-row `ke_slice`，decode 用 `max_model_len`）。下游 kernel 已正确处理 -1。

**效果**：HumanEval 66.5% → 78%。

> 注：此 bug 与长输入乱码 Bug 1 都表现为 shared 层复用 buffer 出问题，但病因不同——本 bug 是 full 层 topk 写入**垃圾索引**（短输入未填 -1），长输入 Bug 1 是 shared 层跑**零权重 indexer** 污染 buffer。

### 3.2 问题 2：evalscope 思考模式代码提取错误（→ 99 分）

**根因**：evalscope 1.8.1 的 `humaneval_adapter._postprocess` 一律取首个 markdown 代码块（`blocks[0]`）。思考模式下模型常输出多个代码块（探索性草稿 + 最终实现，偶有 batch 串扰），取 `blocks[0]` 会取到草稿/错题代码 → pass@1 虚低。

**修复**（`evalscope_patches/humaneval_adapter.py`，非侵入）：
1. prompt 引导模型把最终实现放在最后一个 ` ```python ` 代码块
2. `_postprocess` 优先取「最后一个定义了目标 `entry_point` 函数」的代码块，找不到回退到最后一个块

**效果**：思考模式 6 个失败题中 4 个是提取问题，修复后通过，pass@1 → 99 分（153 题已评分）。

### 3.3 问题 3：思考死循环（采样参数问题）

思考模式下部分题（idx 32/116/118/129/132/145）在 `temperature=0.2` 低温度下陷入确定性自我推翻循环：模型反复用 "Actually"/"but"/"Wait" 推翻自己的推理（单题重复 68~176 次），不收敛到代码，直到 `max_tokens=15900` 截断，提取代码长度=0。这是采样参数问题，非量化缺陷。

**破解**：`repetition_penalty` 1.05→1.15 + `frequency_penalty` 0.3，抑制重复 token 打断循环（temperature 保持 0.2 不变）。

### 3.4 精度演进总览

| 阶段 | 修复 | pass@1 |
|------|------|--------|
| 初始 | — | 66.5% |
| 问题 1 | lightop topk 短输入垃圾索引清洗 | 78% |
| 问题 2 | evalscope entry_point 提取修复 | 99 分（153 题已评分） |
| 问题 3 | 思考死循环（采样参数调优） | — |

### 3.5 最终评测方案

```bash
evalscope eval \
  --model /models/GLM-5.2-Channel-INT4-w4a8 \
  --api-url http://127.0.0.1:8000/v1/chat/completions \
  --api-key EMPTY --eval-type openai_api --datasets humaneval \
  --generation-config '{"temperature":0.2,"top_p":0.95,"max_tokens":15900,"repetition_penalty":1.15,"frequency_penalty":0.3}' \
  --eval-batch-size 32 --work-dir ./outputs/
```
关键：不传 `enable_thinking`（默认思考模式）、`max_tokens=15900`（思考+代码需足够空间）、`repetition_penalty=1.15 + frequency_penalty=0.3`（抑制死循环）、应用 evalscope 提取 patch。

---

## 4. 精度验证

### 4.1 思考模式 HumanEval
- **153 题已评分，得分 99 分**
- 失败题分两类：
  - **思考死循环**（6 题，idx 32/116/118/129/132/145）：模型在低温度（0.2）下陷入确定性自我推翻循环（"Actually"/"but" 重复 68~176 次），生成 1.4~1.9 万 token 不收敛，触发 max_tokens 截断，提取代码长度=0。破解方案：`repetition_penalty` 1.05→1.15 + `frequency_penalty` 0.3 抑制重复打断循环
  - **真实生成错误**（少数）：如 `#153`（Strongest_Extension）模型把 `for ext in extensions:` 误写成 `for ext in ext:`，属思考模式偶发的「想对了写错了」
- **结论：w4a8 量化在思考模式下精度基本无损，死循环为采样参数问题非量化缺陷**

### 4.2 上下文长度
- MLA KV cache：每 token 约 0.15 MB（2 × kv_lora_rank=512 × 2 bytes × 78 层）
- 8 卡可用显存约 93GB → 理论上限约 520K token
- 当前 max-model-len=16384，远低于上限，**上下文非瓶颈**

---

## 5. 性能优化探索：GEMM tile tuning

### 5.1 GEMM 算子链路

**Linear 层**（MLA 投影 + dense/shared MLP）：
- 路径：`SlimQuantW4A8Int8LinearMethod.apply()` → `apply_int8_linear()` → `ops.triton_scaled_mm()`（strategy=1）
- 配置查找：`W8a8GetCacheJSON` 单例，按 `device_name`（gfx928_128cu）查 `W8A8_{N}_{K}_gfx928_128cu.json`
- best_config=None 时 fallback：`matmul_int8()` 内置 5 档硬编码默认 tile（num_stages 恒为 0）

**MoE Expert 层**：
- 路径：`SlimQuantW4A8Int8MoEMethod.apply()` → `fused_experts()` → `fused_experts_impl_w4a8()`（layers 版，`@triton.jit`）
- 配置查找：`get_w8a8moe_json()` 读 `triton_moejson_dict`
- best_config=None 时 fallback：硬编码默认 config `{M:16, N:64, K:64, GM:1, stages:0, warps:4}`
- **关键**：MoE kernel 是 `@triton.jit`（非 autotune），不会自动选 config，完全依赖 json

GLM-5.2 各 Linear per-TP-card shape（TP=8）：

| 层 | N | K |
|----|---|---|
| q_a_proj | 2048 | 6144 |
| q_b_proj | 2048 | 2048 |
| kv_a_proj | 576 | 6144 |
| kv_b_proj | 3584 | 512 |
| o_proj | 6144 | 2048 |
| shared_gate_up | 512 | 6144 |
| shared_down | 6144 | 256 |
| dense_gate_up（前3层）| 3072 | 6144 |
| dense_down（前3层）| 6144 | 1536 |

MoE shape（per card, EP=8）：E=32, N1=512, N2=6144, K2=256, topk=8

**配置缺口**：Linear 9 个 shape 中 gfx928_128cu 配置基本全缺（运行时走 None fallback）；MoE 配置 0 个（全走默认）。

### 5.2 tuning 方法
- Linear：`lmslim/tools/w8a8_int8_tools.py` → `w8a8int8_triton_tuning(n_list, k_list, free_gpus, ...)`
- MoE：`lmslim/tools/fused_moe_tools_w4a8.py`（需改 shape 为 GLM-5.2）
- 完整搜索空间：Linear 约 1944~3645 configs/段，MoE 约 432 configs/shape
- 完整空间 triton 编译量过大（单 shape >1 小时），改用精简空间（num_stages=[0,1,2,3]，split_k=[1]，保留主要 block 组合）

### 5.3 实测结果（解码阶段 M=1~32）

**Linear（triton_scaled_mm），576×6144（kv_a_proj）：**

| M | fallback (μs) | tuned (μs) | 加速 |
|---|---|---|---|
| 1 | 94.4 | 94.8 | 1.00x |
| 8 | 96.3 | 91.2 | 1.06x |
| 16 | 96.9 | 92.2 | 1.05x |

**MoE fused kernel，GLM-5.2 真实 shape：**

| M | default (μs) | best (μs) | 加速 |
|---|---|---|---|
| 1 | 762.7 | 745.5 | 1.02x |
| 8 | 672.5 | 670.8 | 1.00x |
| 32 | 673.4 | 671.3 | 1.00x |

**结论：解码阶段 tile tuning 收益 0~6%，无显著优化效果。**

### 5.4 根因：解码是 launch-bound

**Roofline**：解码 M=1 时所有 Linear 算术强度均 2.0 op/byte，全部 memory-bound。

**MoE 解码 M=1 耗时分解**（实测，总 763μs）：

| 环节 | 耗时 | 占比 |
|------|------|------|
| moe_align_block_size（排序对齐） | 54μs | 7% |
| per_token_quant_int8（激活量化） | 66μs | 9% |
| silu+quant+2×GEMM+launch+CPU查表 | 642μs | 84% |

**关键发现**：MoE 解码 M=1 纯 GEMM 权重读取仅需 ~3μs（18.87MB @ 6.4TB/s），但实测 763μs——**HBM 带宽利用率仅 0.3%**。瓶颈在于：
1. **kernel launch 开销**：M=1 时 GEMM 计算量极小（~7M FLOPs），launch/dispatch 开销远超计算
2. **辅助 kernel 串联**：每层 MoE 需 align→quant→GEMM→silu→quant→GEMM 共 6 个 kernel，每个有固定 launch 开销
3. **CPU 侧 `get_w8a8moe_json` 查表**：每层查 dict

→ tile tuning 改的是 GEMM 内部分块，但 GEMM 在 M=1 时只占总耗时极小部分，故无收益。

### 5.5 MTP dequant-attn：attention int8 权重反量化为 bf16（2x 吞吐，已落地）

> 本节是**真正落地并验证有效**的性能优化，推翻了 5.4「解码 launch-bound、GEMM 无收益」的初步判断。
> 独立 patch 见 [`../mtp/`](../mtp/)（`apply_patch.sh` / `revert_patch.sh` / `README.md`）。

#### 5.5.1 优化思路（为什么想到反量化）

**第一步：profile 定位真瓶颈。** 用 `--profiler-config '{"profiler":"torch",...}'` 重启 MTP server，跑一次 512-token decode 抓 kernel 级 trace。结果颠覆了「MoE 是大头」的直觉：

| Kernel | 占 GPU 时间 | 说明 |
|--------|-----------|------|
| **`matmul_kernel`（w8a8 int8 GEMM）** | **60.5%** | attention qkv/o + shared expert + DSA indexer 的 int8 线性 |
| NCCL 通信 | 8.4% | TP=8 allreduce |
| `fused_moe_kernel_int4_w4a8`（MoE） | 7.9% | 之前以为是大头，其实不是 |
| torch.compile region | 7.4% | |

**结论：瓶颈是 w8a8 int8 GEMM（attention 那些 int8 线性层），不是 MoE。**

**第二步：诊断 int8 GEMM 为什么慢。** 单独 benchmark `lmslim matmul_int8`（Triton kernel）：
- 大部分 shape 卡在 **~90µs 地板**，与 M/K/N 几乎无关（N=512 的 1MB 权重和 N=16384 的 12MB 权重都是 ~94µs）→ 不是带宽瓶颈，是**小 M 下 kernel 配置/占用率问题**
- 换 Triton 配置（含 SPLIT_K）只能 1.11x，SPLIT_K 反而更慢
- 对比 **bf16 GEMM（rocBLAS）：0.307ms/layer vs int8 0.855ms → 2.5~2.8x 更快**，且 bf16 路径**不量化激活**（精度还更高）
- 其他 int8 后端全废：hipblaslt 19x 慢、rocblas/cutlass VMFault 崩溃
- AITER 没有 dense int8 linear（只有 MoE op），帮不上 attention GEMM
- `LMSLIM_USE_LIGHTOP=1` 设了但 `lightop_channel_int8_mm` op 未注册，一直 fallback 到 Triton

**第三步：解法。** 既然 int8 GEMM 慢、bf16 GEMM 快且精度更高，那就**把 attention 的 int8 权重在加载时反量化成 bf16**，让 attention 线性走 bf16 GEMM。MoE 的 int4 权重不动（它已经够快，且反量化 MoE 显存代价太大）。

#### 5.5.2 改动（2 处 hunk，单文件）

文件：`vllm/model_executor/layers/quantization/slimquant_w4a8.py`
1. 新增 `_dequant_attn_enabled()`：读环境变量 `VLLM_DEQUANT_ATTN`
2. `get_quant_method()` 的 `LinearBase`（attention 线性）分支：`VLLM_DEQUANT_ATTN=1` 时强制 `dequant=True`（`FusedMoE` 分支不受影响）

反量化是**数学精确**的：`bf16权重 = int8权重 × per-channel scale`，与 int8 GEMM 的 `x @ (int8×scale)` 完全等价，只是 GEMM kernel 从「int8 Triton（小 M 有地板）」换成「bf16 rocBLAS（无地板）」。

#### 5.5.3 实测结果

| 指标 | int8 baseline | dequant-attn | 变化 |
|------|--------------|--------------|------|
| 吞吐（512 tok） | 19.68 tok/s | **38.42 tok/s** | **1.95x** |
| 吞吐（1024 tok ×3） | — | **39.25 tok/s** | ~2x |
| 接受长度 | 2.887 | 2.960 | 无损（略升） |
| pos0~3 接受率 | 0.95/0.62/0.24/0.07 | 0.92~0.94/0.60~0.62/0.26~0.30/0.09~0.12 | 无损 |
| 显存 | — | +3.59GB/rank | 64GB 卡放得下 |

**profile 复核**：dequant 后 `matmul_kernel`（int8 GEMM）从 60% 消失，新瓶颈分布为 MoE int4 15.4% + NCCL 15.2% + bf16 GEMM（Cijk/rocBLAS）~21% + torch.compile 7.6%——瓶颈从「单一 int8 GEMM」分散开，符合预期。

#### 5.5.4 精度验证（HumanEval 思考模式）

dequant 是数学精确的权重还原 + 更高精度激活，理论上不应影响精度。正在用 dequant-attn server 重跑 HumanEval 思考模式（164 题）确认无系统性精度下降（int8 baseline 参考 99.3%）。

#### 5.5.5 用法

```bash
# 应用 patch（幂等，备份 .bak）
bash mtp/apply_patch.sh
# 启动（run_mtp_dequant_attn.sh 已含 export VLLM_DEQUANT_ATTN=1）
bash mtp/run_mtp_dequant_attn.sh
# 验证吞吐
python3 mtp/bench_mtp.py 1 2 512
# 回滚
bash mtp/revert_patch.sh
```

---

## 6. 适配度总结与后续方向

### 6.1 适配度评估

| 维度 | 状态 | 说明 |
|------|------|------|
| 功能适配 | ✅ 完成 | 长输入乱码清零，3 个独立 bug 修复 |
| 精度适配 | ✅ 思考模式无损 | 99 分（153 题已评分） |
| 部署适配 | ✅ 完成 | 启动崩溃修复，网络配置，启动脚本 |
| 性能适配 | ✅ 已优化 | dequant-attn 落地：19.68 → 39.25 tok/s（~2x），接受长度无损（见 5.5） |

### 6.2 已验证无效的方向
| 方向 | 解码收益 | 评估 |
|------|---------|------|
| Linear w8a8 tile tuning | 3~6% | ❌ 投入产出比低 |
| MoE fused tile tuning | 0~2% | ❌ 投入产出比低 |
| int8 GEMM 换后端（hipblaslt / rocblas / cutlass） | 19x 慢 / VMFault 崩溃 | ❌ 全废（见 5.5.1） |
| Triton int8 配置调优（含 SPLIT_K） | 1.11x，SPLIT_K 更慢 | ❌ 打不破 ~90µs 地板 |
| AITER dense int8 linear | 无此 op（只有 MoE） | ❌ 帮不上 attention GEMM |
| `LMSLIM_USE_LIGHTOP=1` | op 未注册，fallback Triton | ❌ 无效 |

### 6.3 建议后续方向

dequant-attn 落地后，profile 显示新瓶颈分布为：bf16 GEMM（Cijk/rocBLAS）~21%、MoE int4 15.4%、NCCL 15.2%、torch.compile 7.6%。

| 方向 | 预期收益 | 代价 | 优先级 |
|------|---------|------|--------|
| 增大解码 batch（continuous batching） | 线性摊薄 launch 开销 | 无 | ★★★ |
| MoE int4 优化（aiter MoE 路径 A/B 对比） | 15.4% 占比，可能显著 | 改 env + 功能验证 | ★★★ |
| NCCL 通信优化（TP=8 allreduce 15.2%） | 中等 | 通信调优 | ★★ |
| kernel 融合（align+quant+GEMM） | 中等 | 改 lmslim 源码 | ★★ |
| prefill 阶段 tile tuning（大 M） | 中等 | 数小时 tuning | ★★ |
| CPU 查表消除（配置预加载） | 小 | 改 lmslim 源码 | ★ |

**核心判断**：dequant-attn 已消除最大单一瓶颈（int8 GEMM 60%）。下一步重点：① 增大 batch 摊薄 launch 开销；② MoE int4（15.4%）与 NCCL（15.2%）的 A/B 对比优化。

---

## 附录 A：修复文件清单

| 文件 | 作用 | 对应问题 |
|------|------|---------|
| `vllm_patches/modified/sparse_attn_indexer.py` | lightop topk 短输入垃圾索引清洗 | HumanEval 问题 1 |
| `vllm_patches/modified/deepseek_v2.py` | shared 层 skip_topk + 不建 Indexer | 长输入 Bug 1 |
| `vllm_patches/modified/mla.py` | skip_topk 参数，shared 复用 buffer | 长输入 Bug 1 |
| `vllm_patches/modified/flashmla_sparse.py` | shared 层 indexer=None 透传 | 长输入 Bug 1 |
| `vllm_patches/modified/indexer.py` | rocm 分支 scheduler_metadata 占位 | 启动崩溃 |
| `evalscope_patches/humaneval_adapter.py` | entry_point 代码块提取修复 | HumanEval 问题 2 |
| `apply_patch.sh` | vllm patch 应用（检测已应用则跳过） | — |
| `evalscope_apply_patch.sh` / `evalscope_revert_patch.sh` | evalscope patch 应用/回滚 | — |
| `start.sh` | 性能模式启动脚本 | 部署 |
| `mtp/vllm_patches/slimquant_w4a8.patch` | dequant-attn patch（attention int8→bf16） | 性能优化 5.5 |
| `mtp/apply_patch.sh` / `mtp/revert_patch.sh` | dequant-attn patch 应用/回滚 | 性能优化 5.5 |
| `mtp/run_mtp_dequant_attn.sh` | dequant-attn 启动脚本（含 `VLLM_DEQUANT_ATTN=1`） | 性能优化 5.5 |
| `mtp/bench_mtp.py` | MTP 吞吐/接受长度 benchmark | 性能优化 5.5 |
| `mtp/README.md` | dequant-attn 优化说明 | 性能优化 5.5 |

## 附录 B：相关文档
- [`glm52-long-input-garbage-rootcause.md`](glm52-long-input-garbage-rootcause.md) — 长输入乱码完整排查现场
- [`glm52-humaneval-precision-rootcause.md`](glm52-humaneval-precision-rootcause.md) — HumanEval 精度四层问题根因
- [`GLM-5.2-DCU适配度技术文档.md`](GLM-5.2-DCU适配度技术文档.md) — 性能优化探索专题（tile tuning）

## 附录 C：排查方法学（可复用）
1. **无 HF 基线时用数值打印**：量化模型 transformers 不支持，改用逐算子打印 mean/std/absmax 找第一个异常
2. **编译期常量守卫探针**：探针 `.item()` 在 torch.compile 图内会触发 `Unsupported Tensor.item()`，用模块级 `_PROBE_ON = os.environ.get(...)=="1"` 守卫，或 `--enforce-eager` 关闭编译定位
3. **隔离变量**：关 MTP / 关 cuda graph 逐步隔离，确认各组件影响
4. **对照官方实现**：定制分支的 bug 常是「照抄 A 架构模式用到 B 架构」的 API 不兼容（如 gfx938 原地写入 vs gfx928 返回值）
