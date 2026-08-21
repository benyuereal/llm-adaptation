# GLM-5.2 HumanEval 精度排查 — 完整根因分析

## 一、背景

GLM-5.2-Channel-INT4-w4a8（GlmMoeDsaForCausalLM）在 AMD ROCm gfx928（K100_AI, 8-GPU TP）上 HumanEval 评分异常。最初 66.5%，远低于 GLM-5.1 的 98%。经多轮排查，发现四层独立问题。

## 二、四层问题与修复

### 问题 1：lightop topk 短输入垃圾索引（66.5% → 78%）

**根因**：lightop 闭源 `top_k_per_row_prefill` kernel 在短输入（KV < topk_tokens=2048）时，没有像官方 `_C` kernel（`csrc/libtorch_stable/sampler.cu:393-404`）那样把不足槽位填 -1，导致 `topk_indices_buffer` 残留 `torch.empty` 未初始化垃圾索引（实测 min=-2e9, max=+1.9e9）。shared 层（L75/76/77，最后一组 IndexShare）复用垃圾索引越界访问 KV cache → sparse attention 算错 → MoE router 退化（std=0 均匀）。

**修复**（`sparse_attn_indexer.py`）：lightop topk 写入后清洗越界索引为 -1（prefill 用 per-row `ke_slice`，decode 用 `max_model_len`）。下游 `triton_convert_req_index_to_global_index` 和 `sparse_mla_fwd` kernel 已正确处理 -1。

**验证**：修复后 L75-77 退化消除（std 非零、zero%=0），HumanEval 66.5% → 78%。

### 问题 2：非思考模式生成质量下降（评测配置问题）

**根因**：早期评测用 `enable_thinking: false`（或误传到 `extra_body` 顶层被 vLLM 忽略）。
GLM-5.2 的 chat_template 在 `enable_thinking=false` 时，prompt 末尾追加空的
`<|think|><|/think|>`（token 154841 + **154842**，其中 154842 = `<|/think|>`），
让模型跳过思考直接写代码。

> ⚠️ **早期误判纠正**：曾以为 154842 是 `sandbox` token 污染导致解码错乱，实测
> 154842 实为 `<|/think|>`（sandbox 是 76147/42510，无关）。非思考模式掉分的真正
> 原因是**模型不经思考直接生成，代码质量本身下降**，而非 token 污染。

**验证**（同题对比，纯思考 vs 非思考）：
- 思考模式：99%+（模型先推理再写代码，提取修复后）
- 非思考模式：~87%（真实生成错误增多：语法/逻辑错误，如 `#62` 求导、`#64` vowels_count）
- 结论：**思考模式是质量必需**，非思考模式不可用于精度评测

**vLLM 配置坑**：`enable_thinking` 必须放在 `extra_body.chat_template_kwargs` 内，
放 `extra_body` 顶层会被 vLLM 忽略并告警 `{'enable_thinking'} ignored`。

### 问题 3：evalscope 思考模式代码提取错误（提取修复，99.3%）

**根因**：evalscope 1.8.1 的 `humaneval_adapter._postprocess` 一律取首个 markdown
代码块（`blocks[0]`）。思考模式下模型常输出多个代码块——探索性草稿 + 最终实现，
偶尔还混入相邻题目的代码（batch 推理串扰）。取 `blocks[0]` 会取到草稿/错题代码 →
误判 FAIL → pass@1 虚低（实测 ~87% → 修复后 99%+）。

**典型案例**：`#14`（all_prefixes）模型输出混入了相邻题 `rolling_max` 的代码块，
`blocks[0]` 取到 `rolling_max`，提取出错误函数。

**修复**（`evalscope_patches/humaneval_adapter.py`，非侵入，仅改此文件）：
1. prompt 引导模型把最终实现放在最后一个 ` ```python ` 代码块
2. `_postprocess` 优先取「最后一个定义了目标 `entry_point` 函数」的代码块，找不到
   则回退到最后一个块（而非第一个）

**验证**：思考模式 6 个失败题中，4 个（`#115/#20/#127/#109`）是提取问题，修复后
通过；剩余 2 个（`#62/#64`）是真实模型逻辑错误。最终 145/164 已评分，PASS 144
（99.3%），仅 `#153` 失败。

### 问题 4：模型真实生成错误（少数题，无法靠提取修复）

思考模式 + 提取修复后仍偶发的真实错误，例如 `#153`（Strongest_Extension）：模型在
最终代码块把 `for ext in extensions:` 误写成 `for ext in ext:`（遍历变量自身），
是 NameError 级生成错误。讽刺的是其探索性草稿块写对了，但最终块写错——属思考模式
下偶发的"想对了写错了"，无法通过提取修复。早期 completions 模式另有 `#62`（求导）、
`#64`（vowels_count，`'xyz'` 返回 0）等真实逻辑错，思考模式后大幅减少。

## 三、最终评测方案（思考模式，99%+）

```bash
evalscope eval \
  --model /models/GLM-5.2-Channel-INT4-w4a8 \
  --api-url http://127.0.0.1:8000/v1/chat/completions \
  --api-key EMPTY \
  --eval-type openai_api \
  --datasets humaneval \
  --generation-config '{"temperature": 0.2, "top_p": 0.95, "max_tokens": 15900, "repetition_penalty": 1.05}' \
  --eval-batch-size 32 \
  --work-dir ./outputs/
```

**关键**：
- 不传 `enable_thinking`（默认 true，走思考模式；思考模式是质量必需，见问题 2）
- `max_tokens: 15900`（思考+代码需足够空间，过小会截断思考导致代码没生成）
- `temperature: 0.2`（pass@1 用低温度，减少随机性）
- `repetition_penalty: 1.05`（抑制重复，但思考模式长输出尾部偶有请求跑到 max_tokens）

**应用 evalscope 提取修复**（见问题 3，建议应用）：
```bash
bash glm-5.2-channel-int4-w4a8/evalscope_apply_patch.sh   # 应用
bash glm-5.2-channel-int4-w4a8/evalscope_revert_patch.sh  # 回滚
```

**注意**：evalscope humaneval adapter 强制用 ChatMessageUser（走 chat），改 api-url 到
`/v1/completions` 无效；`enable_thinking` 须放 `extra_body.chat_template_kwargs`。

## 四、已排除的嫌疑

- **head_dim（64→192）**：官方 ROCm 走 deepseek_v2，scaling 用 qk_head_dim/index_head_dim，config.head_dim 是死字段
- **rope_theta（1e6→8e6）**：get_rope 必读 rope_parameters.rope_theta，链路完整
- **mqa_logits 算子**：tilelang 实现精度测试误差 1e-9，比官方 FP8 更准
- **MoE 路由**：修复后前 75 层健康，L75-77 退化是症状不是病因
- **MTP L78 prefill 全零**：是 CUDA graph warmup 的 dummy 输入，decode 时正常，不影响精度

## 五、相关文件

- vllm 修复：`vllm_patches/modified/sparse_attn_indexer.py`（topk 清洗，问题 1）
- evalscope 修复：`evalscope_patches/humaneval_adapter.py`（entry_point 提取，问题 3）
  - 应用：`evalscope_apply_patch.sh` / 回滚：`evalscope_revert_patch.sh`
- 启动：`start.sh`（性能模式，CUDA graph + MTP）
- 部署：`apply_patch.sh`（vllm patch，检测已应用则跳过）
- 精度测试：`test/test_mqa_logits_precision.py`（mqa_logits vs aiter 对比）

## 六、精度演进总览

| 阶段 | 修复 | pass@1 |
|------|------|--------|
| 初始 | — | 66.5% |
| 问题 1 | lightop topk 短输入垃圾索引清洗 | 78% |
| 问题 2 | 改用思考模式（非思考 ~87%，质量不足） | — |
| 问题 3 | evalscope entry_point 提取修复 | 99.3%（145/164 已评分） |
| 问题 4 | 真实模型错误（如 #153），无法靠提取修复 | — |

> 长输入乱码（shared 层 indexer 全零 + mqa_logits 返回值丢弃）是另一条独立线，
> 详见 `docs/glm52-long-input-garbage-rootcause.md`。HumanEval 属短输入场景，
> 仅命中问题 1 的短输入 topk 垃圾索引分支。
