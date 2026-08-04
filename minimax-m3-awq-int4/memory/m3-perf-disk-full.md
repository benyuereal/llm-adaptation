---
name: m3-perf-disk-full
description: "M3评测卡顿4分钟+吞吐降半,sparse生效所致;磁盘已清理够用不再追"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2b4cf476-1e6e-4434-abc0-b3a2c3a36a95
  modified: 2026-08-02T16:53:08.153Z
---

M3-AWQ-INT4 修复后(sparse/indexer生效)的两个性能现象:

**1. 吞吐降半**(已确认是修复预期代价,非bug):同口径 running-req=8 cuda graph=True,修复前全dense 56.7 tok/s(per-seq 7.1) → 修复后sparse生效 28 tok/s(per-seq 3.5)。修复前57层sparse全退化成dense(短序列反而快),修复后indexer+block选择真正生效,短序列吃亏长序列收益。prod日志(/workspace/logs/sglang_prod.log)decode几乎全running-req=1(18tok/s),86-94tok/s那批是running-req=16且cuda graph:False。fix3日志(/workspace/logs/sglang_fix3.log)。

**2. 4分钟卡顿**:sglang scheduler_TP0 持续高CPU(80-186%)、8个running-req不出新token、然后突然恢复(token数从11395暴跌到2348,batch重组)。无traceback无API error。排查时发现根盘100%满,清了 /tmp/torchinductor_root(5G)+comgr-*腾出2.8G。**用户指示磁盘不再追,清理够用就行**。卡顿根因待查(排除磁盘后查indexer kernel/JIT fallback/radix cache命中/cuda graph重建)。详细分析见 /workspace/docs/m3-perf-bottleneck.md(子agent产出)。

**显存利用率低**:每卡 65520 MiB,mem-fraction-static=0.55 下只用 38214 MiB(41%空闲)。正在调到 0.90 增大 KV cache 提升并发。

**清理命令备忘**:`rm -rf /tmp/torchinductor_root /tmp/comgr-* /tmp/tmp*.s`。根盘是 /etc/hosts 挂的 2.6T overlay 盘。

修复见 [[m3-lightning-indexer-fix]]。
