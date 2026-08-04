---
name: m3-indexer-voproj-bug
description: M3 lightning indexer的index_v/o_proj假支路Bug B根因与修复方向
metadata: 
  node_type: memory
  type: project
  originSessionId: 2b4cf476-1e6e-4434-abc0-b3a2c3a36a95
  modified: 2026-08-03T03:24:51.277Z
---

M3 "流畅但不聪明" 的第二个 bug(Bug B),与 [[m3-lightning-indexer-fix]] 的 Bug A 叠加。

## 根因(铁证)
MiniMax-M3 模型**设计上只有 index_q_proj / index_k_proj**(各57,层3-59)用于选块,**没有 index_v_proj / index_o_proj**。BF16 原始权重和 AWQ-INT4 都确认 index_v/o_proj = 0 条。这是"设计甲":index head 只对 Q/K 算 score 决定 topk 选块,V/O 不参与输出。

checkpoint 的 quantization_config.ignore 只 ignore `re:.*self_attn[.]index_[qk]_proj`(只 qk,不含 v/o),因为作者根本没考虑不存在的 v/o。

## Bug B 机制
sglang `minimax_m3.py` 第872行 `disable_index_value = layer_id in disable_value_layer_ids`,而 `get_minimax_sparse_disable_value_layer_ids` 在 config 无 `sparse_disable_index_value` 字段时返回空 list → `disable_index_value` 恒 False → **无条件构造 index_v_proj(ReplicatedLinear)+ index_o_proj(RowParallelLinear)**,套 quant_config。

因 index_v/o_proj 不在 ignore 列表 → 走 W4A16 量化 → 创建 weight_packed(torch.empty int32)/weight_scale(torch.empty)/weight_shape/weight_zero_point(torch.zeros)。checkpoint 无这俩权重 → load_weights 不清零未加载 param(源码 line 1398-1403 直接 return)→ 保持 torch.empty 初始值。

forward: idx_v = dequant(weight_packed, weight_scale, 0) @ hidden; idx_o = sparse_attn(idx_q,idx_k,idx_v); idx_output = index_o_proj(idx_o) + 8卡all-reduce; return output + idx_output。

## 实测(2026-08-03)
- 单独构造 ReplicatedLinear+量化: weight_packed/weight_scale = absmax 0 std 0 nz 0(CUDA零页), weight_shape=垃圾(7.7e18,小张量从碎片分配)
- sglang verify4 探针: iv.weight_shape absmax=0(sglang处理过), idx_v=0.0000(idx_o=0, idx_output=0)
- **当前运行=情况A(idx_v=0,数值碰巧无害),但依赖CUDA零页,是未定义行为**

## 修复(已实施,静态验证全过,运行时验证中)
两处改动,关键: model层和backend层必须用同一份带layer_types的sparse_cfg, 否则model disable=True传idx_v=None但backend disable_value=False试图set_index_kv_buffer(idx_v=None)会崩溃(第一次修复只改model层就崩在line 795 idx_o/idx_replica_size NoneType)。

1. model_config.py get_minimax_sparse_disable_value_layer_ids: config无sparse_disable_index_value字段时, 返回get_minimax_sparse_layer_ids(sparse_cfg)的sparse_layer_ids(要求sparse_cfg含layer_types, 由get_minimax_sparse_attention_config注入)。
2. minimax_m3.py decoder layer: 改用get_minimax_sparse_attention_config(config)取带layer_types的sparse_cfg(原getattr(config,sparse_attention_config)是无layer_types的dict, 会让disable函数fallback到全0的sparse_attention_freq得空)。import加get_minimax_sparse_attention_config。

backend(minimax_sparse_backend.py)本就用get_minimax_sparse_attention_config, 自动受益, 无需改。

静态验证(sglang真实config路径): model层57 sparse+57 disable, backend层57+57, 两层disable集合完全一致, 0崩溃层。语法OK importOK。

## 之前崩溃教训(重要,避免重蹈)
- 第一次: 只改model_config.py的disable函数用sparse_layer_ids, 但decoder layer传的sparse_cfg是text_config.sparse_attention_config(无layer_types的dict) -> fallback全0freq -> 返回空 -> disable=False -> 崩溃line 795 idx_o/idx_replica_size NoneType
- 第二次: 只改minimax_m3.py用is_sparse_attention_layer兜底, 但backend没改 -> model传idx_v=None backend disable=False -> set_index_kv_buffer(None)崩溃
- 必须两层一致: model和backend都用get_minimax_sparse_attention_config拿带layer_types的sparse_cfg

## 状态
- 根因+机制查证完成(weight_dequantized实测全零=情况A, 依赖CUDA零页未定义行为)
- 修复代码已写(model_config.py + minimax_m3.py)
- 静态验证全过(model/backend一致, 0崩溃层, 语法import OK)
- sglang_fix_b3运行时验证中(IDX-REAL应消失, iv=None, 不崩)
- HumanEval长题对比待跑
- patch未更新push(当前9eed7d5只有Bug A)
