---
name: glm52-mqa-logits-return-dropped
description: gfx928 tilelang mqa_logits 返回新tensor但调用处丢弃→prefill logits全0→乱码
metadata:
  type: project
---

gfx928 prefill indexer 打分用 tilelang `mqa_logits(q,k,weights,ks,ke)`,它**返回新 tensor**
(非原地写入)。但 `sparse_attn_indexer.py` gfx928 分支照抄 gfx938 的原地写入模式
(gfx938 用 lightop `op.mqa_logits(..., logits_slice_view)` 原地写),**丢弃了返回值**,
导致 `logits_slice_view` 从未被写入 → topk 读全 0 → 输出垃圾 indices → 长输入乱码。

**Why:** API 不兼容:gfx938 原地写入 vs gfx928 返回值,照抄模式导致返回值丢失。
探针证据:`pf_logits mean=0 std=0 zero%=100`,`pf_topk_out min=-1 max=2147483647`(垃圾)。
直接测 lightop topk 算子本身正常,证明问题在 logits 全零(上游返回值丢失),非 topk 算子。

**How to apply:** 接收 `mqa_logits` 返回值直接作为 `logits_slice` 传给 topk,不要 copy 到
padded view(避免 padding -inf 与非连续 stride)。与官方 `fp8_fp4_mqa_logits` 用法一致。
修复后乱码率 20%→0。这是 [[glm52-shared-indexer-zero-weight]] 修复后的残留根因。
同类风险:`paged_mqa_logits`(decode)已正确接收返回值,无需改。
