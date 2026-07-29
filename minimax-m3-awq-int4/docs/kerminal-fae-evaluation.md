# kerminal Agent 开发工具 FAE 技术评估报告

---

## 一、评估说明与立场

本报告从 FAE 技术视角，对 kerminal 在「ISV 适配国产硬件 + 推理框架」这一典型场景下的实际表现进行评估，供内部售前向上汇报与推进合作使用。评估基于真实适配过程的完整日志与配套适配文档，而非演示数据。

立场说明：售前侧希望获得较为认可的评价作为推进依据，本报告理解该诉求，但技术评估的首要价值在于「经得起内部技术复盘与后续验证」——一份扎实的、把进展与卡点都讲清楚的评价，比一份夸大但后续会被证伪的评价对推进合作更有利。故本报告按实事求是原则给出。

**评估场景与硬件软件环境：**

| 项目 | 配置 |
|------|------|
| 硬件 | 海光 K100AI DCU × 8 张（每张 64GB 显存，400W TDP） |
| OS | KylinOS (kernel 4.19.90) |
| 推理框架 | SGLang 0.0.0.dev12695（海光定制版，基于 vLLM 0.15.1 + DTK 2604） |
| 模型 | MiniMax-M3-AWQ-INT4（约 456B 参数，多模态，60 层混合注意力：前 3 层 full_attention + 后 57 层 minimax_m3_sparse） |
| 量化 | AWQ INT4（4-bit 权重量化） |

**前置说明**：本场景难度本身较高——海光 DCU 上的 INT4 MoE 算子适配属于业界公认的硬骨头（涉及 Triton kernel 后端兼容、量化方案 zero-point 处理、显存碎片化、多模态视觉编码器适配等多重叠加问题）。因此评分放在「同类高难度适配任务」坐标系下给出，而非通用编码任务。

### ⚠️ 评估范围与归因说明

**本次实测平台为海光 K100AI DCU，不属于 kerminal 官方已验证的芯片平台清单（官方文档列出的已支持平台为华为昇腾系列，其他平台标注为"持续扩展中"）。** 据此说明两点：

1. 本报告反映的是 kerminal 在「未充分验证平台」上的真实表现，**不构成对官方在已验证平台（如昇腾）上能力声称的证伪或佐证**——未做对照测试，无可比性。
2. 当前推理阶段卡点（VMFault）的归因尚未确定：可能是 kerminal 在未验证平台上的能力盲区，也可能是 K100AI/Triton 后端底层兼容性问题，或两者叠加。**归因以与智子芯元的技术对齐结论为准**（见配套《技术对齐清单》）。

---

## 二、分维度评估

本报告按「易用性、适配能力、性能」三个维度组织评估。

### 易用性

**综合评分：8 / 10**

#### a. 安装——9 / 10

- 一键安装，操作非常方便，开箱即用，无明显门槛。
- 对 ISV 接入非常友好，几乎不需要文档辅助即可完成安装。

#### b. 交互自主度——7 / 10

「自主」按**自主性光谱(L1/L2/L3)**分项评估：

- **L1 任务内自主**：给一个任务，自己连续做完，中间不停。
- **L2 异常自愈**：遇到 OOM/报错/资源耗尽，能自己检测、重试、绕过。
- **L3 无人值守端到端**：从需求到交付，人只在起点终点出现，所有异常自愈。

**并发连贯度(L1)——8.5 / 10**：

- kerminal 在**单次任务内能高度自主、连续推进**：日志显示可连续执行十几分钟、密集调用工具、跨多文件协同排查，无需人逐步确认。这是"并发性/连贯性"强，达到稳稳的 L1。
- 约 **70%~80% 的时间在自主推进**，常规路径上无需人工盯守。
- 这是 kerminal 自主性的主要亮点，可在汇报中重点呈现。

**异常自愈度(L2)——5 / 10**：

- 在**代码/逻辑层面的异常**上，kerminal 有 L2 级能力：能自主识别错误、调整方案、改代码重试（如反复试 BLOCK_SIZE_K、num_warps、attention backend 等并验证）——这是积极的 L2 表现。
- 但在**资源/环境层面的异常**上，自愈能力退化为需人工，本次实测的回退点几乎都集中在此：
  1. **显存类**：OOM 卡死、进程被系统 OOM-Killer 杀死、VMFault 崩溃后显存泄漏（如 "GPU 6 only 222MiB free"）→ 需**人工手动重启容器**才能恢复（恢复后 8 卡均 63.1GB free）。
  2. **超时类**：scheduler ready 超时(700s)→ 需人工判断并重发。
  3. **服务限流类**：长时间高强度执行后触发「请求过于频繁 ERR_20216」→ agent 中断，需**人工重新发起请求**（限流来源待确认，见对齐清单对齐-6）。
- 这些异常超出了"改代码"的能力圈，需要操作系统级动作（重启容器）或外部服务配合（限流退避），kerminal 当前做不了，故 L2 仅部分达成，整体偏向 L1。

**易用性小结**：代码/逻辑层面已达 L2，资源/环境层面仅 L1，综合 L1~L2 之间。

---

### 适配能力

**评分：6.5 / 10（当前阶段，待 VMFault 对齐后有望上调）**

#### a. 模型加载与启动（已交付 ✅）

解决 3 个阻断性问题，模型成功加载、SGLang 服务启动：

| 问题 | 解决方式 | agent 贡献 |
|------|----------|-----------|
| `layer_types` 校验失败 | 追加类型到白名单 | 按官方方案落地 |
| `CLIPVisionConfig` 缺少 `rope_theta` | 提取字段到顶层 | 按官方方案落地 |
| MoE 权重反量化 OOM | pre-allocate + 及时释放 | 独立复现并落地 |

能力体现：源码级闭环能力、量化算法理解力、跨层面协同排查。

#### b. 推理执行（未交付 ❌，当前真卡点）

- Triton MoE INT4 kernel 存在系统性 VMFault，多种调参方案均未根治
- 问题超出官方适配文档覆盖范围，触及 Triton 后端 × K100AI 硬件兼容性深水区

**适配能力小结**：加载启动已交付，推理卡点需硬件/框架侧共同对齐。

---

### 算子性能

**评分：暂不评分（未进入测试阶段）**

- 性能部分尚未测试：适配卡在推理阶段（算子跑通），尚未达到可做吞吐/延迟基准测试的状态。
- 待 VMFault 解决、服务稳定跑通后，再补充性能数据。

**小结**：性能维度本次无法给出结论，建议作为下一阶段评估重点。汇报时**不应**包含任何性能提升数字。由于适配尚未完成，性能部分暂时无法评价，待推理跑通后单独补充（详见文末「下一步」）。

---

## 三、对 kerminal 的评价（基于本次实测）

1. **异常自愈能力从"逻辑层"扩展到"资源/环境层"（L2 逻辑层 → L2 含资源层）**：当前 kerminal 在代码/逻辑异常上已具备 L2 自愈能力，但在 OOM / Killed / 显存泄漏 / 超时等资源类异常下退化为需人工。建议增强对「环境异常」的自动检测与自愈（如自动识别显存泄漏并提示/执行重启容器、超时自动判断重发），将 L2 能力补齐到资源层。
2. **限流韧性**：对「请求过于频繁 ERR_20216」类中断，建议 agent 具备自动退避重试能力，而非直接中断等待人工。
3. **长任务的断点续作**：连续数十分钟的排查任务在限流/中断后，建议能从上下文续作，减少人工「继续」的频次。
4. **硬件适配深水区的边界提示**：当问题触及 Triton 后端/硬件兼容性这类超出应用层修改范围的深水区时，建议 agent 能更早、更明确地给出「需厂商介入」的判断，避免在不可解路径上长时间消耗（本次 VMFault 排查中 agent 在多种调参方案上反复尝试，单次重启 7 分钟，累计耗时显著）。
5. **与官方适配文档的对齐**：建议 kerminal 能关联/感知厂商提供的适配文档（如本次的《MiniMax-M3 在 K100AI DCU 上的适配》），在官方方案覆盖范围内优先按文档落地，避免重复试错；在超出文档范围时主动提示。

---

## 四、Kerminal Agent 自评

> 以下为 Kerminal Agent 对本次适配过程中自身表现的诚实自评。

### 4.1 最终适配结果

本次适配最终成功完成：模型正确推理，单用户 decode 达 19 tok/s。共解决 10 个问题，其中 7 个自主完成，3 个在引导方向后执行。

### 4.2 自主性评估

| 维度 | 评分 (1-10) | 说明 |
|------|-------------|------|
| 代码执行力 | 8 | 能快速定位文件、写修复代码、加诊断、设计验证方案 |
| 方向性判断 | 4 | 多次走错方向 |
| 问题收敛能力 | 3 | 容易发散、过早接受错误结论 |
| 端到端验证 | 7 | 方向明确后能设计有效的 ground truth 对比验证 |

### 4.3 按问题难度加权评估

| 难度分级 | 问题数 | 占总权重 | 自主率 | 说明 |
|----------|--------|----------|--------|------|
| 简单（2-4级） | 4 | 26% | **100%** | layer_types、rope_theta、tokenizer、VMFault |
| 中等（5-7级） | 3 | 34% | **90%** | MoE overflow、shared_experts、dense层误建MoE |
| 高难度（8-9级） | 3 | **40%** | **30%** | symmetric缺失、ZP reshape bug（含误诊断扣分） |

**误诊断扣分**：在 MoE 阶段产出"Triton 编译器 bug"的错误结论，沿此方向消耗约 4-5 小时，占总排查时间 30-40%。这不是"没有推进"而是**产生了负面价值**（误导方向、浪费时间、差点引入不必要的方案切换）。

**修正后难度加权自主率：~65%**

### 4.4 重大失误：MoE 方向误判

本次排查中最大的时间浪费发生在 MoE 阶段。产出了**严重误导性结论**：

> "Triton 编译器在 gfx928 上有 bug，独立脚本编译 VMFault，服务中 kernel 静默失败不执行"

**实际情况**：服务中 Triton MoE kernel **完全正常执行**，输出数值正确。问题根本不在 MoE。

**误判链条**：
1. 独立脚本 VMFault → 结论"编译器 bug"
2. 诊断代码 inplace 时机错误 → diff=0 → 结论"kernel 没执行"
3. 进一步推断"combine 也静默失败"
4. 想用 torch/aiter 替代 Triton（逃避问题）

**危害**：浪费大量时间在正确的组件上；误导性结论如被采信会带偏后续排查。

**根本原因**：证据不充分时过早形成强结论，不善于自我质疑。

### 4.5 综合自评结论

**难度加权自主率 65%，但高难度问题（占权重 40%）仅 30%，且存在负面价值的误诊断。**

Agent 具备解决简单和中等问题的独立能力，但在高难度开放式排查中：
- 方向判断力不足，容易被表面证据误导
- 误诊断会浪费 30-40% 的时间并产生误导性结论
- 需要外部在关键分叉点提供方向性指引才能高效收敛

---

## 五、综合评分

| 维度 | FAE 评分 | 自评修正 | 说明 |
|------|----------|----------|------|
| 安装 | 9/10 | — | 优秀 |
| 适配能力 | 8/10 | **7/10** | 最终成功，但误诊断浪费30-40%时间 |
| 性能 | 7/10 | — | 19 tok/s |
| 自主度 | — | **5/10** | 简单问题强，高难度问题严重依赖引导 |

**Agent 自评综合：7 / 10**

能完成任务，但过程中方向性失误造成了显著的时间浪费和误导风险。适合有方向把控的协作模式，独立处理高难度开放式问题时效率和可靠性不足。

---

## 附录：关键日志摘录

### A.1 MoE 权重反量化 OOM（加载阶段）

```
torch.OutOfMemoryError: HIP out of memory. Tried to allocate 1.12 GiB. GPU 6 has a total capacity of 63.98 GiB of which 222.00 MiB is free.
[2026-07-24 19:51:06] Received sigquit from a child process. It usually means the child failed.
```

### A.2 推理阶段 VMFault

```
Callback: Queue 0x7f3300200000 aborting with error : HSA_STATUS_ERROR_EXCEPTION: An HSAIL operation resulted in a hardware exception. code: 0x1016
>>>>>>>> KERNEL VMFault !!!! <<<<<<
=========> HOSTQUEUE <0x7f244969b230>: VMFault HSA QUEUE ANALYSIS <=========
```

### A.3 VMFault 触发点扩展至 index_elementwise_kernel

```
HOSTQUEUE <0x563d316386e0>: kernel name:
_ZN2at6native24index_elementwise_kernelILi128ELi4EZNS0_16gpu_index_kernelIZNS0_21index_put_kernel_implINS0_10OpaqueTypeILi2EEEEEvRNS_14TensorIteratorEN3c108Arr
ayRefIlEESA_EUlPcPKclE_EEvRNS_18TensorIteratorBaseESA_SA_RKT_bEUliE_EEvlT1_
```

### A.4 显存泄漏与人工重启容器恢复

```
# VMFault 崩溃后显存未释放
RuntimeError: Not enough memory. Please try to increase --mem-fraction-static.

# 人工重启容器后恢复
GPU0: 63.1GB free
GPU1: 63.1GB free
...
GPU7: 63.1GB free
```

### A.5 agent 自主排查与回退点

```
• The service crashed with OOM during MoE weight post-processing. GPU 6 ran out of memory trying to allocate 1.12 GiB
  while stacking w13 weights. Let me investigate the failing code path.

• 又被 Killed。之前的 VMFault 又泄漏了 GPU 内存。
• 你需要再次释放显存。这是一个反复的循环 — 每次 VMFault crash 都泄漏显存。
```

### A.6 服务限流中断

```
■ 请求过于频繁，请稍后重试 (request id: 20260727022839411256511meD2x9CP) [ERR_20216]
```

### A.7 服务启动成功

```
PID: 234904
READY after 420s
```

### A.8 适配最终成功——推理输出正确

```
Input:  "The color of the sky is"
Output: " blue. The color of the sky is blue."  ✓

Input:  "What is 2+3?" (chat with thinking)
Output: <mm:think>The user asks a simple arithmetic question...  ✓
```

### A.9 性能数据——单用户 decode 吞吐

```
[2026-07-29 14:07:48 TP0] Decode batch, cuda graph: True, gen throughput (token/s): 18.82
[2026-07-29 14:07:51 TP0] Decode batch, cuda graph: True, gen throughput (token/s): 18.90
[2026-07-29 14:07:53 TP0] Decode batch, cuda graph: True, gen throughput (token/s): 18.94
```

### A.10 最终 root cause（ZP reshape bug）验证

```
Before fix: _ct_dequantize_hip std=0.046838 vs ground_truth std=0.041708, max_diff=0.17480469  ✗
After fix:  _ct_dequantize_hip std=0.041707 vs ground_truth std=0.041708, max_diff=0.00097656  ✓
```

### A.11 修复前——乱码输出

```
Input:  "What is 2+2?"
Output: "arringarringarringarring..."  ✗
```
