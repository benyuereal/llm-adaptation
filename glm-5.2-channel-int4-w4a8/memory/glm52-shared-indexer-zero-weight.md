---
name: glm52-shared-indexer-zero-weight
description: GLM-5.2 shared DSA indexer 层零权重跑出垃圾并覆盖共享 buffer,长输入乱码主因
metadata:
  type: project
---

GLM-5.2 的 DSA indexer 从 5.1 的"每层独立"改为 full/shared 分组共享。`indexer_types`
里 full 层=[0,1,2,6,10,...,74](21个,有 checkpoint 权重),shared 层=其余57个(无权重,
应复用 full 层的 topk_indices_buffer)。定制分支不区分 full/shared,所有层都建 Indexer
都跑,shared 层零权重跑出垃圾,且 `sparse_attn_indexer` 开头 `topk_indices_buffer[:n]=-1`
清空共享 buffer 再覆盖,把 full 层好结果冲掉。

**Why:** 5.1 无 indexer_types(全 full),5.2 引入共享机制但定制分支未跟进,是回归 bug。
探针证据:shared 层 idx_q std=0 zero%=100,full 层正常。

**How to apply:** 移植官方 skip_topk 机制(`deepseek_v2.py` 按公式算 _skip_topk,shared
层不建 Indexer;`mla.py` 加 skip_topk 参数,not self.skip_topk 才跑 indexer)。修复后乱码率
70-90%→20%。详见 [[glm52-skip-topk-mechanism]]。残留 20% 见 [[glm52-mqa-logits-return-dropped]]。
