---
name: glm52-probe-methodology
description: 无HF基线的量化模型用逐算子打印数值定位;探针.item()需编译期守卫或enforce-eager
metadata:
  type: feedback
---

w4a8 量化模型 transformers 不支持(slimquant),无法做 HF 基线对比。定位改用 op-locate-agent
风格的逐算子打印 mean/std/absmax/nan/inf,找第一个数值异常的算子。

**Why:** 探针里的 `.item()`/`.tolist()` 在 torch.compile 图内会触发
"Unsupported Tensor.item() call" / "Failed to trace builtin operator"。
`torch.compiler.disable` 在本老版本 torch 会 graph break 导致 vllm 崩,不能用。

**How to apply:**
1. 探针函数用模块级常量守卫 `_PROBE_ON = os.environ.get("PROBE_ON")=="1"`,同模块内
   编译器识别为常量跳过整块;跨模块(从 deepseek_v2.py 调 _probe.py)不识别常量,
   需在调用处也用本模块 `_PROBE_ON` 守卫
2. 或定位阶段用 `--enforce-eager` 关闭编译,探针 .item() 不进图,最快
3. 探针打印 int 张量(如 topk_indices)的 min/max/neg/unique + 前 N 元素,能直接看出
   全零(zero%=100)、垃圾(max=2147483647)、-1(无效标记)等异常
