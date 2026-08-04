---
name: m3-precision-nondeterminism
description: Evidence that MiniMax-M3 AWQ INT4 DCU deployment is non-deterministic at temp=0
metadata: 
  node_type: memory
  type: project
  originSessionId: 2b4cf476-1e6e-4434-abc0-b3a2c3a36a95
  modified: 2026-07-30T09:22:30.051Z
---

**方案C实测(2026-07-30):** 同一 HumanEval/26 请求,temperature=0.0,连续推理3次,完整输出 unique=2,最终代码 unique=2(一次用了 `from collections import Counter`,两次没用)。证明当前部署非确定性。

**根因高度怀疑:**
1. AWQ INT4 量化(num_bits=4, group_size=32)
2. MoE fused gate 在 gfx936 无 tuned config(启动脚本注释明确写 `→ precision loss`),用 VLLM_ENABLE_MOE_FUSED_GATE=1 + VLLM_USE_LIGHTOP=1
3. Triton attention/MoE kernel 的 DCU 适配(sglang dev 0.0.0.dev12695)
4. TP=8 多卡 MoE 路由 + 浮点归约顺序

**待验证实验(优先级):**
- A. 关闭 fused MoE gate (VLLM_ENABLE_MOE_FUSED_GATE=0) 看是否变确定 + 正确率是否升
- B. seed=42, top_p=1.0, max-running-requests=1
- 逐层埋点对比(见 [[m3-trace-instrumentation]])定位首个偏差算子

**注意:** temp=0 不等于硬件计算完全确定;分析器要把"输出不同"和"首次数值差异节点"分开报告,别直接全归因量化。
