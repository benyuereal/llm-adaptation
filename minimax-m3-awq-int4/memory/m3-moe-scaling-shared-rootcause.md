---
name: m3-moe-scaling-shared-rootcause
description: "M3 MoE routed_scaling_factor=2.0丢失:静态证据链完整确认bug成立,待运行时验证"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2b4cf476-1e6e-4434-abc0-b3a2c3a36a95
  modified: 2026-08-03T07:49:35.019Z
---

# M3 MoE routed_scaling_factor bug(2026-08-03,静态证据链完整,待运行时验证)

## 结论(修正前次过早推翻):routed_scaling_factor=2.0 大概率丢失,bug 1 成立。
我之前因看到 `fused_moe.py:797` HIP combine 有 `out.mul_(routed_scaling_factor)` 就推翻了 bug1,**错了**——因为上游传给 runner 的 routed_scaling_factor 是 None,那个分支永远不进。重新追完整链后确认 bug 成立。

## 完整证据链(全部静态,已逐行核实)
1. `minimax_m3.py:246-262` `self.experts = get_moe_impl_class(quant_config)(...)` 构造 FusedMoE 时**没传 routed_scaling_factor**。
2. `minimax_m3.py:264-273` `self.topk = TopK(routed_scaling_factor=2.0, apply_routed_scaling_factor_on_output=True)` —— 2.0 只传给了 TopK,没给 experts。
3. `fused_moe_triton/layer.py:179` FusedMoE.__init__ 默认 `routed_scaling_factor=None`;line 267 存进 `moe_runner_config.routed_scaling_factor=None`。
4. `topk.py:683-689` fused_topk sigmoid 分支调 5 参数 `topk_sigmoid(无scaling)` 后 line 693 直接 return;`apply_routed_scaling_factor_on_output` 在此分支仅用于 line 670 assert,不乘 scaling。→ topk_weights 和=1.0(应2.0)。
5. WNA16 `compressed_tensors_wNa16_moe.py:451` `routed_scaling_factor=self.moe_runner_config.routed_scaling_factor` 即 None 传给底层。
6. `fused_moe.py:748` `if routed_scaling_factor is None: routed_scaling_factor=1.0`;line 797 `if !=1.0` 不成立 → combine 不乘 2.0。
7. line 313 `should_fuse_routed_scaling_factor_in_topk` 仅对 Fp8/NvFp4/Unquantized+特定runner=True;M3 是 CompressedTensorsWNA16TritonMoE(INT4)→ False。

## 数学影响
- 当前 sglang 每层 MoE: `routed_sum(和=1.0) + shared(独立MLP,和=1.0)` = 总~2.0 量级
- 正确 vllm: `2.0*routed_sum(和=2.0) + shared(1.0)` = 总~3.0 量级
- 每 MoE 层 routed 贡献少一半 → 所有层、所有序列、所有评估集系统性偏差 → "流畅但笨",不崩。

## 仍需运行时验证(因我两次假阳性,不再仅凭静态下结论)
验证方法(下次重启时一次性确认):
- 在 `fused_topk`(topk.py:683)后加 `print("RSF topk_weights sum:", topk_weights.sum(-1).mean())` —— 若≈1.0 则 topk 没乘 scaling(确认bug)。
- 在 `fused_moe.py:748` 附近加 `print("RSF combine rsf=", routed_scaling_factor)` —— 若 None/1.0 则 combine 没乘(确认bug)。
- 修复后在同一处打印,确认 sum≈2.0。

## 其他两个 MoE 点的状态(已确认非bug)
- shared expert:DCU/HIP 下 fusion 被禁(日志 line263),走独立 MLP(minimax_m3.py:275-284, forward_normal:329+339),正常激活。combine_diag.txt cache3 shape=[1,4,6144] 证实 topk=4。**非bug**。
- 激活 SwiGLU-OAI:M3 走 triton runner `swiglu_no_interleaved_with_alpha_and_limit`(fused_moe.py:341-345):`gate*sigmoid(gate*1.702)*(up+1)` 带 clamp,与 vllm 文档公式一致。**非bug**。

## 修复方向
`minimax_m3.py:246-262` 给 `get_moe_impl_class(quant_config)(...)` 加 `routed_scaling_factor=self.routed_scaling_factor`。这样 runner config 拿到 2.0,combine 阶段 line 797 会乘。但需确认:topk 阶段没乘 + combine 乘 = 数学等价于 vllm(topk_weights 和=1.0, combine sum*2.0)。要避免 double-apply(topk 和 combine 都乘)。当前 topk 没乘,所以只在 combine 乘是对的。

## 状态
静态证据链完整指向 bug1 成立(routed_scaling_factor=2.0 丢失)。待重启运行时验证。这是继 [[m3-lightning-indexer-fix]] [[m3-indexer-voprojs-bug]] 后最可能的全局根因(那两个只影响>2048)。关联 [[m3-rope-base-rootcause]](已排除) [[humaneval-det-eval-result]]。
