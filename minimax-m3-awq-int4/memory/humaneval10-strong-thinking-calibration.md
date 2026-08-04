---
name: humaneval10-strong-thinking-calibration
description: "HumanEval/10强思考校准靶心配置,换容器先复现此题确认能力没退化"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2b4cf476-1e6e-4434-abc0-b3a2c3a36a95
  modified: 2026-07-31T13:43:58.642Z
---

HumanEval/10 (make_palindrome) 是高灵敏度能力探针:短思考下思路对但代码索引错位,反复失败;强思考下能稳定通过。换容器/换框架版本后用它做**校准靶心**:复现配置跑通 = 框架+模型能力正常。

**通过的三要素**:`thinking_mode=enabled` + `max_tokens=32768` + prompt 强制"逐个 docstring 例子手算 trace"。实测 640s、思考 37628 字符、5/5 用例全过、`finish_reason=stop` 无截断。

**关键配置细节**(完整文档见 `/workspace/docs/humaneval-strong-thinking-calibration.md`):
- `chat_template_kwargs.thinking_mode` 必须放 `extra_body` 下,sglang 从那里读(evalscope 会把顶层挪进 extra_body,直接放顶层对 sglang 不生效)。见 [[m3-trace-findings]]。
- 客户端 timeout ≥ 1800s(强思考单题可 640s+)。
- 服务端无需指定 `--context-length`(模型原生 1M,KV 池 72K 够单题);不要 `allow_auto_truncate`(截断是失败主因)。
- 非确定性来自并发/cuda-graph 而非量化,要完全确定用单请求+cuda-graph bs=1,见 [[m3-precision-nondeterminism]]。
- `VLLM_ENABLE_MOE_FUSED_GATE` 对 sglang 无效(vllm 的变量)。

校准脚本:`/workspace/test_humaneval10.py`;单题输出留存样例:`/workspace/humaneval10_output.json`。全量 164 题强思考串行约 29h,需并发+按难度分流控成本。
