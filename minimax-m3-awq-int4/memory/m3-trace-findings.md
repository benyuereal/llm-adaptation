---
name: m3-trace-findings
description: Key findings from MiniMax-M3 per-layer tracing about non-determinism source
metadata: 
  node_type: memory
  type: project
  originSessionId: 2b4cf476-1e6e-4434-abc0-b3a2c3a36a95
  modified: 2026-07-30T10:48:19.606Z
---

逐层埋点对比实验结论(2026-07-30):

**结论1: prefill 阶段数值完全确定**
- 两次 HumanEval/26 的 prefill forward 逐层逐算子 7568 个事件对比,**0 处差异**(mean/std/min/max/l2/nan/inf 全一致)
- 说明 AWQ INT4 量化 + MoE fused gate + Triton 在相同输入的 prefill 上**不产生数值非确定性**
- 启动脚本注释 "fused MoE gate → precision loss" 在相同输入下没有体现为数值差异

**结论2: 非确定性在低并发配置下消失**
- 当前配置(max-running-requests=1, cuda-graph-max-bs=1, top_p=1.0, --disable-radix-cache)下,同一请求连跑3次输出**完全相同**
- 之前方案C不一致时用的是 max-running-requests=64, cuda-graph-max-bs=8, top_p=0.95
- **非确定性来源高度怀疑: 高并发batch调度 / cuda graph多batch路径 / top_p采样边界, 而非量化本身**

**结论3: cuda graph 会绕过 Python 埋点**
- decode 阶段走 cuda graph 重放,不经过 MiniMaxM3Model.forward 的 Python 代码,所以 decode 中间张量不被记录
- 要埋点 decode 必须 --disable-cuda-graph(但推理会变慢很多,eager 模式 ~3s/token)

**待完成: 对比 decode 阶段**
- 关 cuda graph 后 decode 走 eager,能被埋点
- 策略: run1 prefill(fwd1) vs run2 prefill(预期一致), 然后逐个 decode forward 对比找首个差异 decode step
- 当前重启中(max_forwards调到120), 等ready

**实验配置要点:**
- M3_TRACE_MAX_FORWARDS 要足够大容纳 prefill+N*decode(一个请求~20 forward)
- --disable-radix-cache 必须,否则run2跳过prefill无法对比
- --disable-cuda-graph 必须,否则decode不被埋点

相关: [[m3-trace-instrumentation]] [[m3-precision-nondeterminism]]
