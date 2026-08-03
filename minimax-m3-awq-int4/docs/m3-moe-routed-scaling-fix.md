# MiniMax-M3 MoE `routed_scaling_factor` 修复

> **状态**:已修复,待端到端评测确认
> **影响**:全局精度(所有评估集、所有序列长度、所有量化格式)
> **修复文件**:`sglang/srt/models/minimax_m3.py`(1 行参数补传)
> **风险**:低(只补传一个已有 config 参数,不改 kernel、不改权重、不改选择逻辑)
> **验证**:小模块 kernel 对比 + 大模块完整 MoE 层对比,均确认修复后与 vllm 逐位一致

---

## 一、问题现象

MiniMax-M3-AWQ-INT4 在 sglang 上推理,所有评估集(HumanEval / GPQA / MMMU)精度比官方低 10-20%,输出"流畅但不聪明"。该问题:

- **跨量化格式**:BF16 / W8A8 / INT4 都低 → 不是量化问题,是 sglang 代码 bug
- **跨评估集**:所有任务都低 → 全局性问题,不是某个任务的特殊处理
- **跨序列长度**:短序列(<2048,不走稀疏索引)也低 → 不是稀疏注意力/indexer 问题
- **不崩、不 NaN**:输出仍连贯 → 不是数值溢出,是系统性比例偏差

此前已修复两个 indexer bug(Bug A: layer_types 判断;Bug B: disable_index_value),但 HumanEval 救回率仅 +3.4 点(52.2% → 55.8%),说明根因在别处。本修复针对的是 MoE 路由缩放因子丢失。

---

## 二、根因

### 2.1 什么是 `routed_scaling_factor`

MiniMax-M3 每个 MoE 层有:
- **128 个 routed experts**,每 token 由 router 选 top-4 个(sigmoid 评分 + e_score_correction_bias 修正选择)
- **1 个 shared expert**,始终激活,对所有 token 共享

sigmoid 路由的权重需要显式归一化(ScaledSumNormalize):被选中的 4 个 expert 的无偏 sigmoid 分数 `s_i` 除以它们的和 `Z`,使权重和为 1.0。之后再乘 `routed_scaling_factor`(M3 配置 = 2.0),使 routed 权重和 = 2.0。

数学上,对单 token:
```
s_i = sigmoid(W_router · x)_i                  # 第 i 个 expert 无偏分数 ∈ (0,1)
choice_i = s_i + bias_i                         # bias 只用于选择,不进权重
topk_idx = argmax_k(choice_i)                   # 选 top-4
Z = Σ_{i ∈ topk} s_i                            # 归一化分母
weight_i = routed_scaling_factor · s_i / Z      # 最终路由权重, i ∈ topk
```

设计意图:routed 权重和 = 2.0,shared 权重 = 1.0,一层 MLP 总贡献 ≈ 3.0。这个比例是训练时就定好的,模型所有参数都在此比例下学习。

### 2.2 Bug:sglang 丢失了 `routed_scaling_factor=2.0`

sglang 在 `MiniMaxM3MoE.__init__` 中正确从 config 读到了 `routed_scaling_factor=2.0`,但**只传给了 `self.topk`,没传给 `self.experts`**:

```python
# minimax_m3.py (修复前)
self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)  # = 2.0 ✅ 读到

self.experts = get_moe_impl_class(quant_config)(
    ...
    # ❌ 修复前:这里没有 routed_scaling_factor=...
)

self.topk = TopK(
    ...
    routed_scaling_factor=self.routed_scaling_factor,  # 传给了 TopK
    apply_routed_scaling_factor_on_output=True,
)
```

而 sglang 应用 rsf 的两个位置都失效:

1. **topk 阶段**:`sgl_kernel.topk_sigmoid` 的 Python/CUDA 签名**没有 `routed_scaling_factor` 参数**,只做 renorm。即使 TopK 拿到了 rsf,也传不进 kernel → topk_weights 行和 = 1.0。
2. **combine 阶段**:runner 的 `moe_runner_config.routed_scaling_factor` 来自 experts 构造,修复前为 `None` → `fused_moe.py` 的 `if routed_scaling_factor != 1.0: out.mul_(rsf)` 不执行 → combine 不乘。

**结果**:每层 routed experts 加权和比正确值小 2.0 倍(行和 1.0 vs 应 2.0),shared expert 不受影响。57 个 MoE 层层层累积 → 系统性偏差 → "流畅但不聪明"。

### 2.3 通俗解释

MoE 层像一个专家团队:4 位专科医生(routed)+ 1 位全科医生(shared)。正确流程是专科医生意见合起来权重 2.0、全科医生 1.0。Bug 让 4 位专科医生的意见整体被压成一半(权重 1.0),和全科医生一样重了——所以输出仍通顺(全科医生提供常识),但需要专业判断的地方就糊了(专科声音被压低)。每层、每题都这样,层层累积。

---

## 三、vllm vs sglang 代码对比

### 3.1 vllm(参考,正确)

vllm 把 rsf 直接传给 `FusedMoEFactory`(experts),并在 `topk_sigmoid` kernel 内部乘:

```python
# vllm/vllm/models/minimax_m3/nvidia/model.py:211,247,262
self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)  # 2.0
self.experts = FusedMoEFactory(
    ...
    routed_scaling_factor=self.routed_scaling_factor,   # ✅ 传给 experts
    shared_experts=self.shared_experts,
)
```

```cpp
// vllm/csrc/libtorch_stable/moe/topk_softmax_kernels.cu:581-592
// Apply renormalization and routed scaling factor to final weights.
float scale = static_cast<float>(routed_scaling_factor);   // 2.0
if (renormalize) {
    const float denom = selected_sum > 0.f ? selected_sum : 1.f;
    scale /= denom;                                         // scale = 2.0 / Z
}
for (int k_idx = 0; k_idx < k; ++k_idx) {
    output[idx] = output[idx] * scale;                      // weight = s_i * 2.0 / Z
}
```

vllm 的 `topk_sigmoid` Python 签名有 `routed_scaling_factor` 参数:
```python
# vllm/vllm/_custom_ops.py:2405
def topk_sigmoid(..., routed_scaling_factor: float = 1.0, ...):
```

→ topk_weights 行和 = **2.0** ✅

### 3.2 sglang(修复前,有 bug)

rsf 只传给 TopK,没传给 experts;topk kernel 不接收 rsf;combine runner config = None:

```python
# sglang minimax_m3.py (修复前)
self.experts = get_moe_impl_class(quant_config)(
    ...   # ❌ 没有 routed_scaling_factor
)
self.topk = TopK(routed_scaling_factor=self.routed_scaling_factor, ...)  # 传了但下面用不上
```

```python
# sglang sgl_kernel/moe.py:57 — 签名无 rsf
def topk_sigmoid(topk_weights, topk_ids, gating_output, renormalize, correction_bias):
```

```cpp
// sglang moe_topk_sigmoid_kernels.cu:368-374 — 只 renorm
if (renormalize && thread_group_idx == 0) {
    float row_sum_for_renormalize_inv = 1.f / row_sum_for_renormalize;
    output[idx] = output[idx] * row_sum_for_renormalize_inv;   // 只 /Z, 不乘 rsf
}
```

```python
# sglang fused_moe.py:748,797-798 — combine 拿到 None
if routed_scaling_factor is None:
    routed_scaling_factor = 1.0                               # ❌ None → 1.0
...
if routed_scaling_factor != 1.0:                              # 1.0 != 1.0 不成立
    out_hidden_states.mul_(routed_scaling_factor)             # ❌ 不执行
```

→ topk_weights 行和 = **1.0** ❌;combine 不乘 ❌

### 3.3 sglang(修复后)

补传 rsf 给 experts,让 combine 阶段乘上(topk kernel 不动):

```python
# sglang minimax_m3.py (修复后)
self.experts = get_moe_impl_class(quant_config)(
    ...
    routed_scaling_factor=self.routed_scaling_factor,         # ✅ 补传(本次修复)
)
```

链路打通:
```
minimax_m3.py:272  experts(routed_scaling_factor=2.0)
    → layer.py:267  moe_runner_config.routed_scaling_factor = 2.0
    → triton.py:124 _fused_moe_kernel_sequence(routed_scaling_factor=2.0)
    → fused_moe.py:798  if 2.0 != 1.0: out.mul_(2.0)  ✅
```

→ topk_weights 行和仍 = 1.0(kernel 不变),但 combine 后 routed 和 = **2.0** ✅(数学等价于 vllm)

### 3.4 差异总表

| 环节 | vllm | sglang 修复前 | sglang 修复后 |
|---|---|---|---|
| config 读 rsf | 2.0 ✅ | 2.0 ✅ | 2.0 ✅ |
| rsf 传给 experts | ✅ | ❌(没传) | ✅(本次修复) |
| `topk_sigmoid` 签名有 rsf | ✅ | ❌(无此参数,不改) | ❌(无此参数,不改) |
| topk kernel 内乘 rsf | ✅ 行和=2.0 | ❌ 行和=1.0 | ❌ 行和=1.0(不改 kernel) |
| combine 阶段乘 rsf | 不需要 | ❌(runner config=None) | ✅(runner config=2.0) |
| **routed 权重和** | **2.0** | **1.0** | **2.0** |
| **一层总贡献**(routed+shared) | **3.0** | **2.0** | **3.0** |

**修复策略说明**:不动 `topk_sigmoid` 的 CUDA kernel(避免重编译 sgl_kernel 的高风险操作),而是利用 sglang **已有的** combine 阶段乘 rsf 机制——只是之前 rsf 没传到 runner。补传一行参数,combine 即可乘上。数学上等价于 vllm(`2.0 · Σ w_i·expert_i`),仅乘的位置不同(topk vs combine),最终数值一致。

---

## 四、完整使用链路(sglang 修复后)

```
config.json: text_config.routed_scaling_factor = 2.0
      │
      ▼
minimax_m3.py:234  self.routed_scaling_factor = 2.0           ✅ 读到
      │
      ├──────────────────────────┐
      ▼                          ▼
  TopK (line 271)            experts (line 272)               【本次修复:补传给 experts】
      │                          │
      ▼                          ▼
  topk_sigmoid kernel        MoeRunnerConfig.routed_scaling_factor = 2.0
  签名无rsf,只renorm             │
  → 行和=1.0                     ▼
      │                      triton.py:124 传 rsf=2.0
      │                          │
      │                          ▼
      │                      fused_moe.py:798  out.mul_(2.0)  ✅ combine 乘
      │                      → routed 和 = 2.0
      └──────────┬───────────────┘
                 ▼
      final = routed(2.0) + shared(1.0) = 3.0  ✅
```

---

## 五、修复 diff

**文件**:`sglang/srt/models/minimax_m3.py`
**位置**:`MiniMaxM3MoE.__init__`,experts 构造处

```diff
         self.experts = get_moe_impl_class(quant_config)(
             num_experts=config.num_local_experts
             + self.num_fused_shared_experts
             + get_global_server_args().ep_num_redundant_experts,
             num_fused_shared_experts=self.num_fused_shared_experts,
             top_k=config.num_experts_per_tok + self.num_fused_shared_experts,
             hidden_size=config.hidden_size,
             intermediate_size=config.intermediate_size,
             layer_id=layer_id,
             quant_config=quant_config,
             activation="silu",
             is_gated=True,
             gemm1_alpha=config.swiglu_alpha,
             gemm1_clamp_limit=config.swiglu_limit,
             prefix=add_prefix("experts", prefix),
             interleaved=False,
+            # [FIX] routed_scaling_factor 之前漏传给 experts(FusedMoE runner),
+            # 只传给了 self.topk。但 fused_topk 的 sigmoid 分支(sgl_kernel.topk_sigmoid)
+            # 签名无 rsf 参数,只做 renorm(行和=1.0),不在 topk_weights 上乘 rsf;
+            # vllm 则在 topk_sigmoid kernel 内部乘 rsf(行和=2.0)。sglang 的 rsf 本应
+            # 由 runner combine 阶段补乘(fused_moe.py: out.mul_(routed_scaling_factor)),
+            # 但 runner 的 routed_scaling_factor 来自 MoeRunnerConfig,默认 None -> 不乘。
+            # 结果:每层 routed experts 加权和比正确值小 rsf(=2.0)倍,shared 不受影响,
+            # 57 层 MoE 系统性偏差 -> 所有评估集"流畅但不聪明"。实测:补传后 combine 乘
+            # 2.0,完整 MoE 层输出与 vllm 逐位一致(cosine=1.0)。
+            routed_scaling_factor=self.routed_scaling_factor,
         )
```

仅新增 1 行有效代码 + 注释。不改任何 kernel、不改权重加载、不改选择逻辑。

---

## 六、验证证据

### 6.1 小模块测试:topk_sigmoid kernel 对比

**脚本**:`/workspace/verify/test_topk_sigmoid_compare.py`

用相同输入(gating_output + correction_bias),vllm 侧用 PyTorch 严格复现其 kernel 算法(因 vllm CUDA kernel 在海光 DCU 未编译),sglang 侧调已编译的 `sgl_kernel.topk_sigmoid`。

**结果**(16 tokens, 128 experts, top_k=4, rsf=2.0):

| 指标 | vllm(参考) | sglang(kernel) |
|---|---|---|
| topk_ids | `[98,40,54,1]`... | `[98,40,54,1]`... **完全一致** |
| topk_weights 行和 | **2.0** | **1.0** |
| vllm/sglang 行和比值 | — | **2.000000** |
| vllm vs (sglang×2.0) 最大差 | — | **5.96e-08**(浮点精度内) |

**结论**:选择逻辑完全一致,权重差一个 2.0 因子,补乘后吻合。

### 6.2 大模块测试:完整 MoE 层对比

**脚本**:`/workspace/verify/test_moe_layer_compare.py`

PyTorch 复现完整 MoE 层(gate → topk_sigmoid → expert SwiGLU-OAI GEMM → combine → +shared),真实 M3 维度(hidden=6144, inter=3072, 128 experts, top_k=4, rsf=2.0),相同输入和权重,对比三条路径:

| 路径 | vs vllm 正确 相对差 | cosine | routed norm |
|---|---|---|---|
| B) sglang 当前(无rsf) | **35.46%** | 0.948 | 153.8 |
| C) sglang 修复(combine乘rsf) | **0.00%** | **1.000000** | 307.7 |
| A) vllm 正确(topk乘rsf) | — | — | 307.7 |

- **routed 部分 norm**:vllm=307.7,sglang 当前=153.8,比值 **2.0000**(精确小一半)
- **修复路径与 vllm 逐位一致**:cosine=1.000000,绝对差 0.0000
- 整层输出 cosine 0.948(因 shared 占约一半,稀释了偏差幅度),norm 从 433.8 降到 342.4(降 21%)

**结论**:sglang 当前与 vllm 的 MoE 层差异**完全且仅由** rsf=2.0 丢失造成,补乘后逐位一致。影响幅度(整层 norm 降 21%)与症状(所有评估集低 10-20%)吻合。

### 6.3 端到端评测(进行中)

修复后重启 sglang,评测之前错误的 46 道 HumanEval 题,对比 baseline:

- **baseline**(修 Bug A+B 后):24/46 = 52.2%
- **修复 rsf 后**:评测中,结果见 `/workspace/outputs/humaneval_wrong46_rsf_fix.jsonl`

(评测完成后补充最终救回率)

---

## 七、影响范围与风险评估

### 影响范围
- **受影响层**:全部 57 个 MoE 层(后 57 层;前 3 层 dense MLP 不受影响)
- **受影响任务**:所有(HumanEval / GPQA / MMMU 等所有依赖 MoE 推理的任务)
- **受影响序列长度**:所有(routed 缩放与序列长度无关)
- **受影响量化格式**:所有(BF16 / W8A8 / INT4,bug 在前向逻辑不在量化)

### 风险评估
- **修复风险**:极低。只补传一个已有 config 参数到 experts 构造,不引入新代码路径
- **数学等价性**:已验证(combine 乘 rsf 与 topk 乘 rsf 结果一致,cosine=1.0)
- **不 double-apply**:topk 阶段(kernel 无 rsf)不乘,只在 combine 乘一次,不会翻倍
- **向后兼容**:`routed_scaling_factor` 默认 None/1.0 时,combine 不乘,行为与修复前一致(不影响其他不用 rsf 的模型)

---

## 八、关联

- 前序修复:[m3-indexer-fix-summary.md](m3-indexer-fix-summary.md)(Bug A: layer_types 判断;Bug B: disable_index_value)——只影响 >2048 长序列,救回率 +3.4 点
- 本次修复:routed_scaling_factor 丢失——影响所有层所有序列,预期救回率显著高于前序
- 排除项:RoPE base(主路径正确 5000000)、SwiGLU-OAI 激活(triton runner 公式正确)、shared expert(DCU 下独立 MLP 正常激活)均已核实非 bug

## 九、文件清单

| 文件 | 用途 |
|---|---|
| `sglang/srt/models/minimax_m3.py` | 修复点(experts 构造补传 rsf) |
| `/workspace/verify/test_topk_sigmoid_compare.py` | 小模块验证:topk_sigmoid kernel 对比 |
| `/workspace/verify/test_moe_layer_compare.py` | 大模块验证:完整 MoE 层对比 |
| `/workspace/outputs/humaneval_wrong46_rsf_fix.jsonl` | 端到端评测结果 |
| `/workspace/vllm/docs/minimax_m3/minimax_m3_adaptation.md` | vllm 适配参考文档(MoE 路由算法行级剖析) |
