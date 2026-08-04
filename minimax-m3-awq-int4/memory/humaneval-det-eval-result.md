---
name: humaneval-det-eval-result
description: HumanEval result on deterministic config (single request) and comparison with v3
metadata: 
  node_type: memory
  type: project
  originSessionId: 2b4cf476-1e6e-4434-abc0-b3a2c3a36a95
  modified: 2026-07-30T19:47:16.255Z
---

**HumanEval 确定配置评测结果 (2026-07-31)**
- 配置: 单请求(max-running-requests=1) + cuda-graph-max-bs=1 + radix cache 开 + 埋点关
- prompt: v3 的 4步(RESTATE/EDGE/IMPLEMENT/SELF-TEST)模板
- 参数: temp=0, top_p=0.95, max_tokens=16384, thinking_mode=adaptive, 去 mm:think
- 结果: **pass@1 = 71.34% (117/164)**, 耗时 ~7.5 小时(单请求串行,长生成题拖慢)
- 输出: outputs/humaneval_det/20260730_192904

**与 v3 对比: v3 = 79.27% (130/164)**
- 确定配置反而低 ~8 个点, 主要原因是 **长生成题被 max_tokens=16384 截断**
- 47 道失败题中: 16 道输出>8000字符, 9 道明显截断(think未闭合/代码块不完整)
- v3 是并发配置(max-running-requests=64), 速度快, 且 v3 当时部分题用了 use-cache

**逐题差异:**
- 确定配置失败但v3通过: 10道 (39,67,96,97,103,104,109,126,134,144) — 多为长生成截断
- v3失败但确定配置通过: 23道 — 确定配置在这些题反而做对(但整体仍低, 因截断题更多)
- 两者都失败: 24道 — 真正的能力不足题

**关键结论:**
1. 71.34% 是"确定配置 + max_tokens=16384 + 长生成截断"的综合结果, 不是模型纯能力上限
2. 长生成是最大问题: 模型对某些题陷入冗长思考, 挤占 max_tokens 导致代码被截断
3. 要拿真实能力分数, 建议: 降 max_tokens 到能覆盖正常题的上限(如4096)并加 stop 提前终止, 或用 v3 的并发配置(更快, 但有非确定性)
4. 确定 vs 并发的分数差(71→79)主要来自截断, 而非并发"运气好"

相关: [[m3-trace-findings]] [[m3-precision-nondeterminism]]
