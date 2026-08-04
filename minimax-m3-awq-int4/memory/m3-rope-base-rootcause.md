---
name: m3-rope-base-rootcause
description: "M3 RoPE base排查结论:主路径正确(5000000),非根因;仅边界情况会错"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2b4cf476-1e6e-4434-abc0-b3a2c3a36a95
  modified: 2026-08-03T07:30:34.721Z
---

# M3 RoPE base 排查结论(2026-08-03,已闭合)

## 结论:RoPE base 在 sglang 主加载路径下是正确的(5000000),**不是**"不聪明"的根因。
之前怀疑"sglang 传顶层 VL config → base=10000"是**假阳性**,被运行时实验推翻。

## 排查过程与证据
- config.json:顶层无 rope_theta/rope_parameters;text_config.rope_parameters.rope_theta=5000000;architectures=["MiniMaxM3SparseForConditionalGeneration"]。
- sglang arch 解析(运行时验证):`MiniMaxM3SparseForConditionalGeneration` → `minimax_m3_vl.py` 的 VL 类(__file__ 确认),不是 minimax_m3.py 的纯文本类。
- VL 类 minimax_m3_vl.py:108 `text_config = config.text_config` 传给 MiniMaxM3Model → 一路传到 MiniMaxM3Attention。
- 运行时验证(sglang 正常加载,无 sys.path 污染):`top.text_config` 是 PreTrainedConfig(不是 dict),有 rope_parameters 属性 → `get_rope_config(text_config)` = **5000000** ✓。
- minimax_m3.py:414 `get_rope_config(config)` 收到的是 text_config(PreTrainedConfig),返回 5000000。

## 假阳性来源(记录防重蹈)
- 一次实验我手动 `sys.path.insert(0, '/models/...')`,导致 configuration_minimax_m3_vl 被 import 两次、`MiniMaxM3VLConfig` 成两个不同类、`__post_init__` 的 isinstance 检查失败 → text_config 没被 coerce 成对象,保留为 dict → `get_rope_config(dict)` = 10000。这是**实验污染**,非生产路径。
- model_config.py:138-141 注释提到"sglang's get_config loads MiniMaxM3VLConfig with text_config as dict"——指某种边界情况,非当前主路径。

## 真实但非触发的脆弱点(待观察)
- sglang **没有** vllm 的 `patch_rope_parameters`(vllm transformers_utils/config.py:529)——后者在 get_config 时把 rope_parameters.rope_theta 提升为顶层 `config.rope_theta` 属性,所以 vllm 模型类无论收到顶层还是 text_config 都能用 `config.rope_theta`。sglang 完全依赖模型类自己用 text_config + rope_parameters。如果未来某模型类误读顶层 config 的 rope_theta,会落到 10000。当前 M3 VL 类没踩这个坑。
- 若 M3 改用纯文本 arch(MiniMaxM3SparseForCausalLM,minimax_m3.py)且 loader 传顶层 config,则会触发 base=10000。但当前 checkpoint arch 是 VL,不走这条路。

## 状态
RoPE 排除。根因转向 MoE(routed_scaling_factor=2.0 / scoring_func=sigmoid / norm_topk_prob)、Sparse indexer、Norm/scale。4 个对比 agent 进行中。关联 [[m3-lightning-indexer-fix]] [[humaneval-det-eval-result]] [[m3-precision-nondeterminism]]。
