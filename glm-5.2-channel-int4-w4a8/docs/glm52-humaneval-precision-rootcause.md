# GLM-5.2 HumanEval 精度排查 — 完整根因分析

## 一、背景

GLM-5.2-Channel-INT4-w4a8（GlmMoeDsaForCausalLM）在 AMD ROCm gfx928（K100_AI, 8-GPU TP）上 HumanEval 评分异常。最初 66.5%，远低于 GLM-5.1 的 98%。经多轮排查，发现三层独立问题。

## 二、三层问题与修复

### 问题 1：lightop topk 短输入垃圾索引（66.5% → 78%）

**根因**：lightop 闭源 `top_k_per_row_prefill` kernel 在短输入（KV < topk_tokens=2048）时，没有像官方 `_C` kernel（`csrc/libtorch_stable/sampler.cu:393-404`）那样把不足槽位填 -1，导致 `topk_indices_buffer` 残留 `torch.empty` 未初始化垃圾索引（实测 min=-2e9, max=+1.9e9）。shared 层（L75/76/77，最后一组 IndexShare）复用垃圾索引越界访问 KV cache → sparse attention 算错 → MoE router 退化（std=0 均匀）。

**修复**（`sparse_attn_indexer.py`）：lightop topk 写入后清洗越界索引为 -1（prefill 用 per-row `ke_slice`，decode 用 `max_model_len`）。下游 `triton_convert_req_index_to_global_index` 和 `sparse_mla_fwd` kernel 已正确处理 -1。

**验证**：修复后 L75-77 退化消除（std 非零、zero%=0），HumanEval 66.5% → 78%。

### 问题 2：chat 模式 enable_thinking=false 注入 sandbox token（78% → 77% 误判）

**根因**：evalscope 用 chat 接口 + `enable_thinking: false` 评测。chat_template 在 `enable_thinking=false` 时，生成 prompt 末尾注入 ` sandbox\n` token（token id 154842）。这个 sandbox 被当成代码开头，导致后续解码错乱——生成内容出现 token 乱序/丢失/多空格（如 `as a        input`、`if char == >:`、引号丢失）。

**验证**：
- completions API（不走 chat_template）生成完全正确
- chat 默认（enable_thinking=true，不传 extra_body）无乱码
- 只有 `enable_thinking=false` 才乱码

**这不是模型精度问题，是评测配置问题。** 35 个失败题中约 19 题是这种乱码误判。

### 问题 3：模型真实不会做（少数题）

completions 模式下仍有约 14 题 FAIL，其中 5 题重测能 PASS（MTP 非确定性），3 题辅助函数未定义（is_prime/factorial），真实逻辑错约 5 题。思考模式后大幅减少。

## 三、最终评测方案（思考模式，95%+）

```bash
evalscope eval \
  --model /models/GLM-5.2-Channel-INT4-w4a8 \
  --api-url http://127.0.0.1:8000/v1/chat/completions \
  --api-key EMPTY \
  --eval-type openai_api \
  --datasets humaneval \
  --generation-config '{
    "temperature": 0.2,
    "top_p": 0.95,
    "max_tokens": 16384
  }' \
  --eval-batch-size 32 \
  --work-dir ./outputs/
```

**关键**：
- 去掉 `extra_body` 里的 `enable_thinking: false`（默认 true，思考模式）
- `max_tokens: 16384`（思考+代码需要足够空间，4096 会截断思考导致代码没生成）
- `temperature: 0.2`（pass@1 用低温度，减少随机性）

**注意**：evalscope humaneval adapter 强制用 ChatMessageUser（走 chat），改 api-url 到 `/v1/completions` 无效。

## 四、已排除的嫌疑

- **head_dim（64→192）**：官方 ROCm 走 deepseek_v2，scaling 用 qk_head_dim/index_head_dim，config.head_dim 是死字段
- **rope_theta（1e6→8e6）**：get_rope 必读 rope_parameters.rope_theta，链路完整
- **mqa_logits 算子**：tilelang 实现精度测试误差 1e-9，比官方 FP8 更准
- **MoE 路由**：修复后前 75 层健康，L75-77 退化是症状不是病因
- **MTP L78 prefill 全零**：是 CUDA graph warmup 的 dummy 输入，decode 时正常，不影响精度

## 五、相关文件

- 修复：`vllm_patches/modified/sparse_attn_indexer.py`（topk 清洗）
- 启动：`start.sh`（性能模式，CUDA graph + MTP）
- 部署：`apply_patch.sh`（检测已应用则跳过）
- 精度测试：`test/test_mqa_logits_precision.py`（mqa_logits vs aiter 对比）
