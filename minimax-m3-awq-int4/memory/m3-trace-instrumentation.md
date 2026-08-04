---
name: m3-trace-instrumentation
description: MiniMax-M3 per-layer/operator runtime tracing setup for locating non-determinism
metadata: 
  node_type: memory
  type: project
  originSessionId: 2b4cf476-1e6e-4434-abc0-b3a2c3a36a95
  modified: 2026-07-30T09:22:23.205Z
---

为定位 MiniMax-M3-AWQ-INT4 在 K100_AI DCU + sglang + TP8 上的推理非确定性(方案C已证实 temp=0 下 HumanEval/26 输出不一致),加了逐层/逐算子埋点。

**实现:**
- `/workspace/patch/diagnostics/runtime_trace.py` — 追踪工具(M3_TRACE=1 启用,默认关;每 rank 独立 JSONL;只存统计不存张量)
- `minimax_m3.py` 加了 20 处埋点:model/layer/attention/mlp/moe 各阶段
- `/workspace/start_awq_trace.sh` — 诊断启动脚本(M3_TRACE=1, max-running-requests=1, cuda-graph-max-bs=1)
- `/workspace/analyze_m3_trace.py` — 离线对比两次 trace,定位首个数值差异节点
- `apply_patch.sh` 会把 runtime_trace.py 装为 sglang_m3_runtime_trace.py

**已安装到 site-packages 并重启服务**(2026-07-30 17:22)。`/workspace/sglang_trace.log` 是诊断启动日志,trace 输出在 `/workspace/trace/`。

**复现脚本:** `/workspace/run_trace_repro.py`(MODE=short 或 humaneval,跑两次相同请求分别存 trace_run1/trace_run2)

**注意:**
- MLP 的 layer_id 标签这次启动为 None(MLP.__init__ 原没存 layer_id,已修源文件但当前进程已加载旧版);dense层0-2才用MLP,MoE层用MiniMaxM3MoE(layer_id正常),影响小
- 埋点默认关闭,不影响正常评测;诊断时用 start_awq_trace.sh
- 相关: [[m3-precision-nondeterminism]]
