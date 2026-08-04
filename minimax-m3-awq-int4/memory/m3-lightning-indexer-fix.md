---
name: m3-lightning-indexer-fix
description: "M3\"流畅但不聪明\"根因:indexer权重未加载,已用layer_types修复,精度回升但吞吐降半"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2b4cf476-1e6e-4434-abc0-b3a2c3a36a95
  modified: 2026-08-02T16:52:23.903Z
---

M3-AWQ-INT4 在 sglang 上分数被锁死低于官方10-20%(BF16/W8A8/INT4都一样)、输出"流畅但不聪明"的**根因和修复**。

**根因**:config 里 `sparse_attention_freq` 字段是全零占位符。`get_minimax_sparse_layer_ids()` 用它 → 返回空 sparse_layer_ids → 57个sparse层全被判成dense → lightning indexer 的 228 个权重(index_q_proj/index_k_proj/index_q_norm/index_k_norm)从未创建 → checkpoint 里这些权重被丢弃(not found in params_dict,244+条) → indexer 用零权重运行 → 长序列 KV block 选择全错。短任务(<2048 token,HumanEval/MMMU)退化成full attention不触发indexer所以只小幅掉分;长任务(GPQA ~2900 token)触发坏indexer→灾难性掉分。所有量化格式受影响(权重加载是公共路径,与量化无关)。

**修复**:改用 config 里正确的 `layer_types` 字段(layer_id 3-59 是 "minimax_m3_sparse")识别 sparse 层,而非全零的 sparse_attention_freq。改了两处:
1. `/usr/local/lib/python3.10/dist-packages/sglang/srt/models/minimax_m3.py` ~779行 `is_sparse_attention_layer` 判定
2. `/usr/local/lib/python3.10/dist-packages/sglang/srt/configs/model_config.py` 的 `get_minimax_sparse_attention_config`(注入layer_types)和 `get_minimax_sparse_layer_ids`(优先用layer_types)
两处都要改,只改一处会崩(backend选择也用get_minimax_sparse_layer_ids,或model直接读config.sparse_attention_config拿不到注入的layer_types)。备份:.bak/.bak2。

**验证**:修复后 index not found 从 244+ 降到 0;HumanEval 之前全错的46题,修复后前8题6通过(75%,修复前0/8)。启动日志出现 `[MiniMaxSparse] Backend initialized`(修复前没有,证明之前全dense)。

**代价**:修复前57层sparse全退化成dense(短序列反而快),修复后sparse/indexer真正生效,8并发decode吞吐从 56.7 tok/s 降到 28 tok/s(同口径running-req=8 cuda graph=True)。这是预期代价,sparse短序列吃亏长序列收益。另有4分钟卡顿(见 [[m3-perf-disk-full]] 可能相关)。

详见 [[humaneval-det-eval-result]] [[m3-trace-findings]]。修复 patch 整理进 llm-adaptation(见 /workspace/patch/)。
