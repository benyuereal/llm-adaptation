# GLM-5.2-Channel-INT4-w4a8 DCU 适配度技术文档

> 报告周期周报配套技术文档
> 日期：2026-08-22
> 平台：AMD ROCm gfx928（K100AI DCU，8×64GB）
> 模型：GLM-5.2-Channel-INT4-w4a8（GlmMoeDsaForCausalLM，slimquant w4a8）

---

## 1. 适配环境概述

| 项目 | 配置 |
|------|------|
| 硬件 | K100AI DCU × 8（gfx928，128 CU/卡，64GB HBM/卡） |
| 算力栈 | ROCm / DTK，vLLM 0.15.1，lmslim 0.3.1，aiter 0.1.2+das.opt1.dtk2604，lightop |
| 模型 | GLM-5.2-Channel-INT4-w4a8，权重 363GB |
| 量化 | slimquant w4a8：权重 INT4 + 激活 INT8 |
| 部署 | TP=8，dtype=bfloat16，max-model-len=36384 |
| 架构 | MLA（Multi-head Latent Attention）+ MoE，78 层，256 expert，topk=8 |

### 关键 env 配置（start.sh）
```
VLLM_USE_LIGHTOP=1
VLLM_USE_LIGHTOP_MOE_ALIGN=1
VLLM_ROCM_USE_AITER_MOE=0      # MoE 走 lmslim fused kernel，不走 aiter
VLLM_W8A8_BACKEND 强制=1       # gfx928 上 Linear 走 triton_scaled_mm（envs.py:1911）
```

---

## 2. 适配状态：已完成验证

### 2.1 推理功能验证（HumanEval，思考模式）
- 164 题全量跑通，PASS 157 / FAIL 7，**正确率 95.7%**
- 7 个 FAIL 根因：4 个真实错误（量化噪声 + 题目难度），3 个思考死循环（生成 15900 token 不收敛 → finish_reason=length）
- **结论：w4a8 量化在思考模式下精度基本无损**，死循环问题通过 max_tokens / 上下文长度调节可控

### 2.2 非思考模式验证
- 164 题正确率 82.3%（PASS 135）
- 降级主因：w4a8 量化噪声导致结构性 token 错误（括号、缩进、函数名），非随机乱码
- **结论：非思考模式不适合作为主推理路径**，w4a8 降级明显

### 2.3 上下文长度
- MLA KV cache：每 token 约 0.15 MB（2 × kv_lora_rank=512 × 2 bytes × 78 层）
- 8 卡可用显存约 93GB → 理论上限约 520K token
- 当前 max-model-len=36384，远低于上限，**上下文非瓶颈**

---

## 3. GEMM 算子链路梳理

### 3.1 Linear 层（MLA 投影 + dense/shared MLP）
- 路径：`SlimQuantW4A8Int8LinearMethod.apply()` → `apply_int8_linear()` → `ops.triton_scaled_mm()`（strategy=1）
- 配置查找：`W8a8GetCacheJSON` 单例，按 `device_name`（gfx928_128cu）查 `W8A8_{N}_{K}_gfx928_128cu.json`
- **best_config=None 时的 fallback**：`matmul_int8()` 内置 5 档硬编码默认 tile（按 M 分桶，num_stages 恒为 0）

GLM-5.2 各 Linear 的 per-TP-card shape（TP=8）：

| 层 | N | K | 解码每 token 调用 |
|----|---|---|---|
| q_a_proj | 2048 | 6144 | 每层 |
| q_b_proj | 2048 | 2048 | 每层 |
| kv_a_proj | 576 | 6144 | 每层 |
| kv_b_proj | 3584 | 512 | 每层 |
| o_proj | 6144 | 2048 | 每层 |
| shared_gate_up | 512 | 6144 | 每个 MoE 层 |
| shared_down | 6144 | 256 | 每个 MoE 层 |
| dense_gate_up（前3层）| 3072 | 6144 | 前 3 层 |
| dense_down（前3层）| 6144 | 1536 | 前 3 层 |

**适配缺口**：上述 9 个 shape 中，除 576×6144 外，gfx928_128cu 配置全部缺失（连 gfx928_120cu 回退也基本没有）→ 运行时全部走 None fallback。

### 3.2 MoE Expert 层
- 路径：`SlimQuantW4A8Int8MoEMethod.apply()` → `fused_experts()` → `fused_experts_impl_w4a8()`（layers 版，`@triton.jit`）
- 配置查找：`get_w8a8moe_json()` 读 `triton_moejson_dict`，按 `MOE_W4A8INT8_E=.._N1=.._N2=.._K=.._TOPK8_gfx928_128cu.json`
- **best_config=None 时的 fallback**：硬编码默认 config `{M:16, N:64, K:64, GM:1, stages:0, warps:4}`
- GLM-5.2 MoE shape（per card, EP=8）：E=32, N1=512, N2=6144, K2=256, topk=8
- **适配缺口**：gfx928_128cu 的 MoE 配置 = 0 个 → 全部走默认 config

**关键区别**：MoE kernel 是 `@triton.jit`（非 autotune），不会自动选 config，完全依赖 json；Linear 的 triton_scaled_mm 同样依赖 json，None 时走硬编码默认。

---

## 4. 性能优化探索：tile tuning

### 4.1 优化目标
针对 GLM-5.2 的真实 shape，在 gfx928_128cu 上生成 tuned config，替换 None fallback，提升 GEMM 性能。

### 4.2 工具与方法
- Linear：`lmslim/tools/w8a8_int8_tools.py` → `w8a8int8_triton_tuning(n_list, k_list, free_gpus, ...)`
- MoE：`lmslim/tools/fused_moe_tools_w4a8.py` → `main()`（需改 shape 为 GLM-5.2）
- 完整搜索空间：Linear 约 1944~3645 configs/段（low/mid/high），MoE 约 432 configs/shape
- 因完整空间 triton 编译量过大（单 shape >1 小时），改用精简空间（num_stages=[0,1,2,3]，split_k=[1]，保留主要 block 组合）

### 4.3 实测结果（解码阶段，M=1~32）

**Linear（triton_scaled_mm），576×6144（kv_a_proj）：**

| M | fallback (μs) | tuned (μs) | 加速 |
|---|---|---|---|
| 1 | 94.4 | 94.8 | 1.00x |
| 8 | 96.3 | 91.2 | 1.06x |
| 16 | 96.9 | 92.2 | 1.05x |

**MoE fused kernel，GLM-5.2 真实 shape（E=32, N1=512, N2=6144, K2=256, topk=8）：**

| M | default (μs) | best (μs) | 加速 |
|---|---|---|---|
| 1 | 762.7 | 745.5 | 1.02x |
| 8 | 672.5 | 670.8 | 1.00x |
| 32 | 673.4 | 671.3 | 1.00x |

**结论：解码阶段 tile tuning 收益 0~6%，无显著优化效果。**

### 4.4 根因分析：解码阶段是 launch-bound，非 compute/bandwidth-bound

**Roofline 分析（解码 M=1）**：所有 7 个 Linear 的算术强度均为 2.0 op/byte，全部 memory-bound。

**MoE 解码 M=1 耗时分解（实测，总 763μs）**：

| 环节 | 耗时 | 占比 |
|------|------|------|
| moe_align_block_size（排序对齐） | 54μs | 7% |
| per_token_quant_int8（激活量化） | 66μs | 9% |
| silu+quant+2×GEMM+launch+CPU查表 | 642μs | 84% |

**关键发现**：MoE 解码 M=1 的纯 GEMM 权重读取仅需 ~3μs（18.87MB @ 6.4TB/s），但实测 763μs —— **HBM 带宽利用率仅 0.3%**。瓶颈在于：
1. **kernel launch 开销**：M=1 时 GEMM 计算量极小（~7M FLOPs），launch/dispatch 开销远超计算
2. **辅助 kernel 串联**：每层 MoE 需 align→quant→GEMM→silu→quant→GEMM 共 6 个 kernel，每个有固定 launch 开销
3. **CPU 侧 `get_w8a8moe_json` 查表**：每层查 dict

→ tile tuning 改的是 GEMM 内部分块，但 GEMM 在 M=1 时只占总耗时极小部分，故无收益。

---

## 5. 适配度结论与后续方向

### 5.1 当前适配度
- **功能适配**：✅ 完成。模型正常加载、推理、思考模式 95.7% 正确率
- **精度适配**：✅ 思考模式基本无损；⚠️ 非思考模式有 w4a8 降级（82.3%）
- **性能适配**：⚠️ 基础可用，但 GEMM 配置缺失（Linear 9 个 shape 缺 8 个，MoE 全缺），解码阶段 launch-bound 待优化

### 5.2 已验证无效的方向
| 方向 | 解码收益 | 评估 |
|------|---------|------|
| Linear w8a8 tile tuning | 3~6% | ❌ 投入产出比低 |
| MoE fused tile tuning | 0~2% | ❌ 投入产出比低 |

### 5.3 建议后续方向
| 方向 | 预期收益 | 代价 | 优先级 |
|------|---------|------|--------|
| 增大解码 batch（continuous batching） | 线性摊薄 launch 开销 | 无 | ★★★ |
| 开 aiter MoE 路径（VLLM_ROCM_USE_AITER_MOE=1）对比 | 未知，可能显著 | 改 env + 功能验证 | ★★★ |
| kernel 融合（align+quant+GEMM） | 中等 | 改 lmslim 源码 | ★★ |
| CPU 查表消除（配置预加载） | 小 | 改 lmslim 源码 | ★ |
| prefill 阶段 tile tuning（大 M） | 中等 | 数小时 tuning | ★★ |

**核心判断**：解码阶段优化重点不在 GEMM tile，而在减少 kernel 数量 / 增大 batch 摊薄 launch 开销。aiter MoE 路径是最值得优先验证的方向。

---

## 附录 A：关键文件位置
- vLLM w4a8 量化：`vllm/model_executor/layers/quantization/slimquant_w4a8.py`
- Linear GEMM：`vllm/model_executor/layers/quantization/utils/w8a8_utils.py`（apply_int8_linear）
- MoE fused kernel：`lmslim/layers/fused_moe/fuse_moe_w4a8.py`（运行时）/ `lmslim/tools/fused_moe_tools_w4a8.py`（tuning）
- 配置缓存目录：`lmslim/configs/w8a8/`（gfx928_128cu 配置 67 个，GLM-5.2 专用 shape 大部分缺失）
- 配置查找单例：`vllm/utils/__init__.py` → `W8a8GetCacheJSON`
- 启动脚本：`/data1/csy/llm-adaptation/glm-5.2-channel-int4-w4a8/start.sh`

## 附录 B：已生成的 tuned 配置
- `W8A8_576_6144_gfx928_128cu.json`：解码小 M 段（M=1~17）已 tune，最优 tile 稳定在 32×32×256，num_stages=2~3
- 其余 shape 的 tuning 脚本已就绪（`tune_w8a8_fast.py`），按需可补全 prefill 大 M 配置
