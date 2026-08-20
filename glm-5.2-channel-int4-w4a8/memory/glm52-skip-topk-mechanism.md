---
name: glm52-skip-topk-mechanism
description: 官方 vllm 用 index_topk_freq/index_skip_topk_offset 公式判定 full/shared,与 indexer_types 一致
metadata:
  type: reference
---

官方 vllm `deepseek_v2.py` 的 skip_topk 判定(本修复照此移植):

```python
_index_topk_freq = getattr(config, "index_topk_freq", 1)        # GLM-5.2: 4
_index_skip_topk_offset = getattr(config, "index_skip_topk_offset", 2)  # GLM-5.2: 3
layer_id = extract_layer_index(prefix)
if _index_topk_pattern is None:
    _skip_topk = max(layer_id - _index_skip_topk_offset + 1, 0) % _index_topk_freq != 0
elif 0 <= layer_id < len(_index_topk_pattern):
    _skip_topk = _index_topk_pattern[layer_id] == "S"
# MTP/nextn 层 (layer_id >= num_hidden_layers) 永远建 full indexer
is_mtp_layer = layer_id >= num_hidden_layers
if self.is_v32 and (not _skip_topk or is_mtp_layer):
    self.indexer = Indexer(...)   # full 层建
else:
    self.indexer = None            # shared 层不建
```

公式算出的 build 层 = `[0,1,2,6,10,14,...,74]`,与 config `indexer_types` 的 full 层
**完全一致**(0/78 不匹配)。`indexer_types` 只是 `index_topk_pattern`(F/S 字符串)的
另一种表达。MLAAttention wrapper 加 `skip_topk` 参数,`not self.skip_topk` 才跑 indexer。
相关:[[glm52-shared-indexer-zero-weight]]
