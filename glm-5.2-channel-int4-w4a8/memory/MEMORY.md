# GLM-5.2 适配工作记忆索引

- [shared indexer 零权重根因](glm52-shared-indexer-zero-weight.md) — 5.1→5.2 回归:shared 层无权重却跑 indexer,清空覆盖共享 buffer
- [prefill mqa_logits 返回值丢弃](glm52-mqa-logits-return-dropped.md) — gfx928 tilelang 算子返回新tensor,调用处照抄gfx938原地写入模式丢弃返回值→logits全0
- [skip_topk 官方机制](glm52-skip-topk-mechanism.md) — 官方 index_topk_freq/index_skip_topk_offset 公式判定 full/shared,与 indexer_types 一致
- [探针定位方法](glm52-probe-methodology.md) — 无HF基线时逐算子打印数值定位;探针.item()需编译期常量守卫或enforce-eager
- [greedy EOS 量化边界](glm52-greedy-eos-boundary.md) — 非乱码:sampling正常greedy空,w4a8精度使EOS概率边界
