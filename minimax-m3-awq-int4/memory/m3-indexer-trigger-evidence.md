---
name: m3-indexer-trigger-evidence
description: "铁证:indexer触发(>2048token)的题通过率40% vs 短题78.5%,锁定精度问题在sparse路径"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2b4cf476-1e6e-4434-abc0-b3a2c3a36a95
  modified: 2026-08-03T01:27:22.901Z
---

定位"模型被锁死低于官方20%"的决定性证据。用户确认:**BF16/W8A8/INT4 都用同一套 sglang 代码跑,GPQA/MMMU 得分也低** → bug 在 sglang 代码,与量化无关。

**HumanEval 修复后全量(73.78%)按输出长度分层**:
- 输出 ≤2048 token(不触发 indexer,走全注意力):144题,通过率 **78.5%**
- 输出 >2048 token(thinking长输出触发 indexer,走稀疏):20题,通过率 **40.0%**
- 差近40个点!长题通过率暴跌 = indexer 触发后精度崩塌的特征。

这解释了:GPQA(~2900token全程触发)低、MMMU(长题触发)低、HumanEval整体被20%长题拖累。也解释"流畅但不聪明":短序列语言流利(78.5%),长序列推理崩(40%)。

**修复后长题仍只有40%**,说明三种可能:
- (A) 修复没真生效(indexer权重加载但没起作用)→ IDX-CHECK探针验证中
- (B) 修复生效但sparse实现还有bug(权重对但topk选块/sparse attn算法有问题)
- (C) 40%已是修复后改善(需修复前长题对比,但修复前没分层统计)

关键:HumanEval thinking_mode=adaptive 下,即使输入短,长思考输出>2048token 也会触发 indexer。所以"HumanEval是短题不触发indexer"的早期判断不完全对 —— thinking 长输出会触发。

IDX-CHECK探针:在 forward_core sparse分支打印 idx_o 的 mean/absmax/nonzero。若全零→(A);若有正常值但长题仍错→(B) sparse实现bug,需查 minimax_sparse_backend.py 的 topk_blocks 选块逻辑。

相关:[[m3-lightning-indexer-fix]] [[m3-perf-disk-full]]
