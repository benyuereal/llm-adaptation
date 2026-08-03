# MiniMax-M3 sglang 适配 — 埋点/探针策略文档

> 用途:记录我们在 MiniMax-M3-AWQ-INT4 + sglang + 海光 DCU (K100_AI, ROCm/HIP) 适配过程中用过的所有埋点/探针/诊断策略,供未来遇到类似"模型不聪明"(精度偏低 10-20%)问题时快速复用。
>
> 根因已于 2026-08-03 定位并修复(MoE `routed_scaling_factor=2.0` 丢失 + lightning indexer 权重未加载 + index_v/o_proj 假支路)。所有运行时埋点已从已安装的 site-packages 副本中清理,准备最佳性能全量评测。

---

## 1. 概述

### 1.1 为什么需要埋点

M3-AWQ-INT4 在 sglang 上表现为"流畅但不聪明":所有评估集精度比官方低 10-20%,但输出通顺、不崩。这种症状意味着没有显式报错、没有 NaN/Inf,bug 藏在数值量的系统性偏差里,必须靠埋点把中间张量的统计量打出来才能定位。

我们排查过的几类问题:

- **非确定性**:temp=0 下同一请求多次推理输出不一致(怀疑量化/MoE 路由/cuda graph)。
- **量化零点(zp)加载**:AWQ-INT4 的 `weight_zero_point` 是否正确加载、是否被全零 expert 污染。
- **MoE 数值**:fused MoE 的 gate/topk/combine 各阶段中间张量是否合理、`routed_scaling_factor` 是否被乘。
- **Indexer 执行**:lightning sparse indexer 的 index_v/o_proj 是否走了不存在的假支路、indexer 权重是否真的加载。

### 1.2 埋点策略总体原则

1. **可开关**:所有运行时埋点都受环境变量(`M3_TRACE`)或"仅前 N 次"守卫控制,默认关闭,不影响正常评测性能。
2. **最小性能影响**:
   - 优先用 `_xxx_diag_done` 类一次性标志,只在第一个 forward / 第一层 / 加载时触发一次,之后自动失效。
   - 只存统计量(mean/std/min/max/l2/nan/inf),不存完整张量(除非显式开 `M3_TRACE_SAVE_TENSORS`)。
   - 加载期诊断(权重加载后、`process_weights_after_loading`)只在加载时跑一次,零运行时开销。
3. **用完即清**:定位完成后,埋点从已安装代码中删除,只把代码片段留档在 `sglang_patches/diagnostics/*.py` 供未来复用。
4. **不改控制流**:埋点只读不写张量,不改变模型行为(运行时 trace 工具的 docstring 明确声明 "never changes tensors or model control flow")。

---

## 2. 埋点分类清单

### 2.1 M3_TRACE 逐层/逐算子运行时 trace(非确定性定位)

| 项 | 内容 |
|---|---|
| **用途** | 定位 temp=0 下输出不一致的根因:逐层逐算子记录中间张量统计量,对比两次相同请求的 trace,找首个数值差异节点 |
| **实现** | `sglang_patches/diagnostics/runtime_trace.py`(`M3RuntimeTrace` 类,环境变量 `M3_TRACE=1` 启用,每 rank 独立 JSONL,只存统计不存张量)。`minimax_m3.py` 里曾有约 20 处 `trace_tensor(op, stage, tensor, layer=...)` 调用(model/layer/attention/mlp/moe 各阶段)。`apply_patch.sh` 可把它装为 `sglang_m3_runtime_trace.py` |
| **配套脚本** | `start_awq_trace.sh`(诊断启动:M3_TRACE=1 + max-running-requests=1 + cuda-graph-max-bs=1)、`analyze_m3_trace.py`(离线对比两次 trace)、`run_trace_repro.py`(跑两次相同请求分别存 trace_run1/trace_run2) |
| **性能影响** | 仅前 `M3_TRACE_MAX_FORWARDS` 次 forward,每 forward 最多 `M3_TRACE_MAX_EVENTS` 个事件,默认关 |
| **当时发现** | (1) prefill 阶段两次 7568 个事件 0 处差异 → AWQ INT4 + MoE fused gate + Triton 在相同输入的 prefill 上不产生数值非确定性;(2) 非确定性在低并发(max-req=1, cuda-graph-bs=1, top_p=1.0, disable-radix-cache)下消失 → 高度怀疑高并发 batch 调度 / cuda graph 多 batch 路径 / top_p 采样边界,而非量化本身;(3) **cuda graph 会绕过 Python 埋点**:decode 走 graph 重放不经 `MiniMaxM3Model.forward`,decode 中间张量不被记录,要埋 decode 必须 `--disable-cuda-graph`(eager ~3s/token) |
| **当前状态** | 已清理(已安装的 `minimax_m3.py` 无 `trace_tensor` 调用)。`runtime_trace.py` 留档在 `sglang_patches/diagnostics/`,`apply_patch.sh` 中安装行已注释 |

### 2.2 MoE combine/激活数值诊断(`fused_moe.py`)

| 项 | 内容 |
|---|---|
| **用途** | 定位 MoE 各 kernel 阶段数值是否合理:W13 gate-up kernel 输出、SiLU 后、W2 down kernel 输出、HIP combine 前后 |
| **实现** | `sglang_patches/diagnostics/fused_moe_diag.py`(从 `fused_moe.py` 提取的三个诊断块)。写 `/workspace/kernel_seq_diag.txt` 和 `/workspace/combine_diag.txt`。第一、二块带 `torch.cuda.synchronize()`(强制等 kernel 完成再读统计),combine 块不带 synchronize |
| **三个块** | (1) `after_w13_kernel`:`_fused_moe_kernel_sequence` 里第一个 `invoke_fused_moe_kernel` 后,记 cache1/hidden_states/w1/config/sorted_token_ids;(2) `after_silu/after_w2_kernel`:第二个 kernel 后,记 cache2/cache3;(3) `BEFORE/AFTER combine`:HIP 分支(`_is_hip` 且非 aiter)里 combine 前后,记 cache3/out_hidden_states/data_ptr/inplace |
| **守卫机制** | 每块用 `_kernel_diag_done` / `_kernel_diag2_done` / `_combine_diag_done` / `_combine_diag2_done` 类属性做一次性标志,只在第一次执行时写文件 |
| **性能影响** | 仅第一次 forward(一次性标志),但有 `torch.cuda.synchronize()` 会强制同步一次 |
| **当时发现** | `combine_diag.txt` 里 cache3 shape=[1,4,6144] 证实 topk=4;shared expert 在 DCU/HIP 下 fusion 被禁、走独立 MLP 正常激活(非 bug);激活 SwiGLU-OAI 走 `swiglu_no_interleaved_with_alpha_and_limit` 与 vllm 一致(非 bug) |
| **当前状态** | **已安装副本已清理**(site-packages `fused_moe.py` 无任何 diag)。代码留档在 `sglang_patches/diagnostics/fused_moe_diag.py`。注意:repo 的 `sglang_patches/modified/fused_moe.py` 仍残留第二、三块(L730 `_kernel_diag2_done`、L788 `_combine_diag_done`、L799 `_combine_diag2_done`),第一块已删;`fused_moe.py.patch` 仍含全部三块。重新跑 `apply_patch.sh` 会把残留块装回去,需在清理清单中注意 |

### 2.3 量化零点加载诊断(`compressed_tensors_wNa16_moe.py`)

| 项 | 内容 |
|---|---|
| **用途** | 诊断 AWQ-INT4 的 `weight_zero_point` 是否正确加载、哪些 expert 的 zp 是全零(全零 expert 会让 dequant 出零权重,污染 MoE 输出) |
| **实现** | `sglang_patches/diagnostics/wNa16_moe_diag.py`(从 `compressed_tensors_wNa16_moe.py` 提取的两个块),写在 `CompressedTensorsWNA16TritonMoE.process_weights_after_loading` 里 |
| **两块** | (1) **pre-conversion**(`zp_preconv_diag.txt`):权重转换前,记每层 `w13_weight_zero_point` 的 all_zero / nz_experts / shape / sample,前 5 层或全零层必记;(2) **post-conversion**(`zp_all_layers.pt`):转换+填充后,在最后一层(L57)用 `torch.save` dump 全 57 层的 nonzero/zero 层统计、zero_expert_info、最后一层 sample |
| **守卫机制** | `_pre_conv_count` / `_zp_dump_count` 类属性计数,pre 块"前 5 层或全零层"才写,post 块在 count==57 时 dump 一次 |
| **性能影响** | **仅加载时**(零运行时开销),但 post 块遍历所有 expert 检查全零有少量 CPU 开销 |
| **当时发现** | 部分层 zp 全零是 checkpoint 设计(某些 expert 在该 group 无量化),需用默认 zp=8(`0x88` 字节)填充,否则 dequant 出零 |
| **当前状态** | **已安装副本已清理**(site-packages `wNa16_moe.py` 无 `_pre_conv_count`/`_zp_dump_count`)。代码留档在 `sglang_patches/diagnostics/wNa16_moe_diag.py`。**重要**:`wNa16_moe.py` 里的 zp 填充逻辑(`# Fill zero experts with default zp=8` + `fill_(0x88)`)不是埋点,是必要修复,必须保留(已安装副本 L492-520 仍在)。注意:repo 的 `sglang_patches/modified/compressed_tensors_wNa16_moe.py` 仍残留两个诊断块(L468 pre-conv、L550 post-conv),重新跑 `apply_patch.sh` 会装回去 |

### 2.4 权重加载后 zp 状态诊断(`minimax_m3_vl.py`)

| 项 | 内容 |
|---|---|
| **用途** | 在 `load_weights` 路径里,确认 layer 3 的 `w13_weight_zero_point` 在 expert 0 加载后是否有非零值(验证 zp 是否真的进了 param) |
| **实现** | `minimax_m3_vl.py` 的 `load_weights` expert-weight 分支里,条件 `if "zero_point" in name and "layers.3." in name and "w13" in new_name and expert_id == 0 and shard_id == "w1"`,写 `/workspace/zp_load_diag.txt`(记 param_nz/sample/data_ptr/param_id) |
| **守卫机制** | 多条件过滤(只 layer 3 + expert 0 + w1 shard),无一次性标志,但条件极窄,实际只触发几次 |
| **性能影响** | **仅加载时**,极低(条件窄) |
| **当时发现** | 确认 zp 在 `weight_loader` 调用后确实写进了 param(配合 2.3 的 pre/post 诊断形成完整证据链) |
| **当前状态** | **已安装副本已清理**(site-packages `minimax_m3_vl.py` 无 `zp_load_diag`)。代码留档在 `minimax_m3_vl.py.patch`(diff 里仍可见)。注意:repo 的 `sglang_patches/modified/minimax_m3_vl.py` 仍残留该块(L357-364),重新跑 `apply_patch.sh` 会装回去 |

### 2.5 逐层 hidden_states / attention 诊断(`minimax_m3.py`)

| 项 | 内容 |
|---|---|
| **用途** | 初始 bring-up 阶段:确认 embedding 输出、每层 hidden_states/residual、前 6 层 attention 输出是否合理(无 NaN/Inf,量级正常) |
| **实现** | `sglang_patches/diagnostics/minimax_m3_diag.py`(从 `minimax_m3.py` 提取的三个块):embed 后、for 循环里每层、decoder layer attention 后,写 `/workspace/layer_diag.txt` |
| **守卫机制** | `_layer_diag_done` 类属性一次性标志(全层只 dump 一次首 forward);attention 块用 `_attn_diag_{layer_id}` 每层一个标志,只 layer_id<=5 |
| **性能影响** | 仅第一次 forward + 仅前 6 层 attention |
| **当时发现** | bring-up 期确认各层量级正常、无 NaN/Inf,排除"中间层爆炸/塌缩"类问题 |
| **当前状态** | **已清理**(已安装 `minimax_m3.py` 无 `layer_diag`/`_attn_diag`)。代码留档在 `sglang_patches/diagnostics/minimax_m3_diag.py` |

### 2.6 IDX 探针(sparse indexer 执行诊断)

| 项 | 内容 |
|---|---|
| **用途** | 诊断 lightning sparse indexer 是否真的执行、`index_v_proj`/`index_o_proj` 是否走了不存在的假支路(Bug B)、`disable_index_value` 是否恒 False |
| **实现** | `minimax_m3.py` 里临时加的 IDX/IDX-REAL 打印探针,验证 `idx_v` 是否为 None / 0、`disable_index_value` 取值、`set_index_kv_buffer` 是否崩 |
| **守卫机制** | 直接 print,无守卫(短期排查用) |
| **性能影响** | 每次执行打 print(短期排查,定位后立即删) |
| **当时发现** | `iv.weight_shape absmax=0`、`idx_v=0.0000` → 当前运行是"情况 A"(idx_v=0,数值碰巧无害但依赖 CUDA 零页,是未定义行为);`disable_index_value` 恒 False → 确认 Bug B(index_v/o_proj 假支路) |
| **当前状态** | **已清理**(已安装 `minimax_m3.py` 无 IDX/IDX-REAL)。无独立留档文件(直接 print 类探针,定位后即删)。修复逻辑(用 `layer_types` 判定 sparse 层 + `get_minimax_sparse_disable_value_layer_ids` 修复)保留在 `minimax_m3.py` 和 `model_config.py` |

---

## 3. M3_TRACE 逐层 trace 机制

### 3.1 环境变量(定义在 `start_awq_dcu.sh`)

```bash
# Optional MiniMax runtime tensor tracing. Disabled for normal evaluations.
# Enable for a controlled single-request run with M3_TRACE=1.
export M3_TRACE="${M3_TRACE:-0}"                    # 总开关:1/true/yes/on 启用
export M3_TRACE_DIR="${M3_TRACE_DIR:-/workspace/trace}"  # 输出目录(每 rank 一个 JSONL);注意:该变量被设置时即使 M3_TRACE 未设也会启用(sglang scheduler 子进程可能 strip 掉 M3_TRACE,用 trace dir 兜底)
export M3_TRACE_MAX_FORWARDS="${M3_TRACE_MAX_FORWARDS:-1}"   # 最多 trace 多少次 forward
export M3_TRACE_MAX_LAYERS="${M3_TRACE_MAX_LAYERS:--1}"     # 最多 trace 到第几层(-1 全部)
export M3_TRACE_MAX_EVENTS="${M3_TRACE_MAX_EVENTS:-10000}"  # 每 forward 最多事件数
# 额外(在 runtime_trace.py 里读,启动脚本未列):
# M3_TRACE_SAVE_TENSORS=1   存完整张量到 .pt(默认关,只存统计)
# M3_TRACE_TENSOR_OPS=op1,op2  配合 SAVE_TENSORS,只存指定 op 的张量
```

### 3.2 输出格式

- 每 rank 一个文件:`trace.rank{RANK}.pid{PID}.jsonl`,行缓冲(`buffering=1`),JSON per line。
- 首行 `meta` 事件:host/pid/rank/tp_rank/pp_rank/enabled/max_forwards。
- 每个 `event` 记录:`{kind:"tensor", time, rank, tp_rank, pp_rank, forward_id, layer, op, stage, shape, dtype, numel, mean, std, min, max, l2, nan, inf, finite}`。
- 张量统计在 GPU 上先 `.float()` 再算;`save_tensors` 时额外 `torch.save(x.cpu(), ...)`。

### 3.3 如何开启 / 关闭

- **正常评测**:`M3_TRACE=0`(默认),`runtime_trace.py` 不被安装(`apply_patch.sh` 安装行已注释),零开销。
- **诊断非确定性**:
  1. 取消 `apply_patch.sh` 中 `runtime_trace.py` 安装行的注释,重新 apply;
  2. `minimax_m3.py` 里恢复约 20 处 `trace_tensor` 调用(从留档的旧版本/记忆还原);
  3. 用诊断启动脚本:`M3_TRACE=1 --disable-cuda-graph --disable-radix-cache --max-running-requests=1`(关 cuda graph 才能埋 decode;关 radix cache 才能让 run2 重跑 prefill 对比);
  4. `M3_TRACE_MAX_FORWARDS` 要够大容纳 prefill+N*decode(一个请求约 20 forward)。

### 3.4 关键陷阱

- **cuda graph 绕过 Python 埋点**:decode 走 graph 重放不经 `MiniMaxM3Model.forward`,decode 中间张量不被记录。要埋 decode 必须 `--disable-cuda-graph`(eager 慢很多)。
- **MLP layer_id 标签可能为 None**:dense 层(0-2)用 `MiniMaxM3MLP`,MoE 层(3-59)用 `MiniMaxM3MoE`;若 `MLP.__init__` 没存 layer_id,MLP 事件的 layer 标签为 None(影响小,只有 3 层)。
- **`M3_TRACE` 在子进程可能被 strip**:sglang scheduler worker 继承环境变量时 `M3_TRACE` 可能丢失,所以 `runtime_trace.py` 把"trace dir 被设置"也当作启用信号兜底。

---

## 4. 清理清单

### 4.1 本次已清理的埋点(已安装 site-packages 副本)

| 文件 | 埋点 | 行号范围(已安装清理后) | 留档位置 |
|---|---|---|---|
| `sglang/srt/models/minimax_m3.py` | 逐层 hidden_states/attn 诊断、IDX 探针、`trace_tensor` 调用 | 全部删除 | `sglang_patches/diagnostics/minimax_m3_diag.py` |
| `sglang/srt/layers/moe/.../fused_moe.py` | combine_diag / kernel_seq_diag 三个块 | 全部删除 | `sglang_patches/diagnostics/fused_moe_diag.py` |
| `sglang/srt/layers/quantization/.../compressed_tensors_wNa16_moe.py` | zp_preconv_diag / zp_all_layers 两个块 | 全部删除 | `sglang_patches/diagnostics/wNa16_moe_diag.py` |
| `sglang/srt/models/minimax_m3_vl.py` | zp_load_diag 块 | 全部删除 | `minimax_m3_vl.py.patch`(diff 中可见) |
| `apply_patch.sh` | `runtime_trace.py` 安装行 | 注释掉 | — |

### 4.2 保留的逻辑(非埋点,必须保留)

| 文件 | 逻辑 | 为什么保留 |
|---|---|---|
| `compressed_tensors_wNa16_moe.py`(L492-520) | zp 填充:`# Fill zero experts with default zp=8` + `w13_zp[ei].view(torch.uint8).fill_(0x88)` | 修复:全零 expert 的 zp 用默认 8 填充,否则 dequant 出零权重。这是必要修复不是诊断 |
| `minimax_m3.py`(L779+) | sparse 层判定用 `layer_types` 而非全零的 `sparse_attention_freq` | 修复(Bug A):否则 57 个 sparse 层全被判成 dense,indexer 权重不创建 |
| `model_config.py` | `get_minimax_sparse_attention_config` 注入 layer_types;`get_minimax_sparse_disable_value_layer_ids` 在无字段时返回 sparse_layer_ids | 修复(Bug A+B):model 层和 backend 层必须用同一份带 layer_types 的 sparse_cfg |
| `minimax_m3.py`(L259+) | `routed_scaling_factor=self.routed_scaling_factor` 传给 experts | 修复(MoE rsf):否则 combine 不乘 2.0,routed 贡献少一半 |

### 4.3 待清理(repo `sglang_patches/modified/` 源文件仍残留)

> **重要风险**:repo 里的 patch 源文件(`sglang_patches/modified/*.py` 和 `*.patch`)仍残留部分诊断块。当前已安装的 site-packages 副本是干净的,但如果有人重新跑 `apply_patch.sh`,会把残留诊断块装回去。下次正式发版前应把这些源文件也清理干净,或确保 `apply_patch.sh` 用的是已清理版本。

| 文件 | 残留诊断块 | repo 行号 |
|---|---|---|
| `sglang_patches/modified/fused_moe.py` | `_kernel_diag2_done` / `_combine_diag_done` / `_combine_diag2_done` | L730-745、L788-792、L799-803(第一块 `_kernel_diag_done`/`after_w13_kernel` 已删) |
| `sglang_patches/modified/fused_moe.py.patch` | 全部三块(含 `after_w13_kernel`) | diff 中 L175-198、L223-235、L263-280 |
| `sglang_patches/modified/compressed_tensors_wNa16_moe.py` | pre-conv / post-conv 两个块 | L468-484、L550-585 |
| `sglang_patches/modified/minimax_m3_vl.py` | zp_load_diag 块 | L357-364 |

### 4.4 残留诊断输出文件(`/workspace/`,可删)

清理代码后这些历史输出文件仍在磁盘上,可安全删除:

- `/workspace/combine_diag.txt`(69KB)
- `/workspace/kernel_seq_diag.txt`(47KB)
- `/workspace/zp_all_layers.pt`(1.8KB)
- `/workspace/zp_load_diag.txt`(59KB)
- `/workspace/zp_preconv_diag.txt`(244KB)

---

## 5. 未来复用建议

### 5.1 遇到"模型不聪明"时的排查优先级

1. **先查量化零点**(成本最低,仅加载时):
   - 恢复 `wNa16_moe_diag.py` 的 pre/post 两块 + `minimax_m3_vl.py` 的 zp_load_diag。
   - 只在加载时触发,零运行时开销,重启一次即得结果。
   - 看是否有层的 zp 全零且没被填充、或 zp 根本没进 param。
2. **再查 MoE 数值**(combine/kernel_diag,一次性):
   - 恢复 `fused_moe_diag.py` 三个块。
   - 一次性标志,只第一次 forward 触发,但有 `synchronize`。
   - 重点看 combine 前后量级、`routed_scaling_factor` 是否被乘(topk_weights 行和应≈2.0 若 rsf 生效,≈1.0 若丢失)。
3. **最后查非确定性**(M3_TRACE,成本最高):
   - 仅当 1、2 都正常但仍有精度问题时才用。
   - 必须 `--disable-cuda-graph --disable-radix-cache --max-running-requests=1`。
   - 跑两次相同请求对比 JSONL,找首个数值差异节点。
   - 注意 prefill 通常确定,差异多在 decode / 高并发路径。

### 5.2 最小化性能影响的做法

- **优先用一次性标志**(`_xxx_diag_done`):只在第一次 forward/加载触发,之后自动失效,稳态零开销。
- **加载期诊断优于运行期诊断**:权重加载、`process_weights_after_loading` 阶段的诊断完全不影响推理性能,优先用。
- **只存统计不存张量**:`M3_TRACE_SAVE_TENSORS=0`(默认),`M3_TRACE_MAX_EVENTS` 设小一点(如 1000)够定位即可。
- **分层/分 op 限流**:`M3_TRACE_MAX_LAYERS` 限制只看前几层,`M3_TRACE_TENSOR_OPS` 只存指定 op 的张量。
- **诊断完立即清理**:定位后把埋点从已安装代码删掉,别带进评测运行;`apply_patch.sh` 的 `runtime_trace.py` 安装行保持注释。

### 5.3 埋点设计模式(供未来新写埋点参考)

- **守卫标志命名**:`_xxx_diag_done` / `_xxx_diag2_done` 类属性挂在 impl 类上,跨实例共享,确保全进程只触发一次。
- **输出文件**:`/workspace/xxx_diag.txt`(文本追加)或 `/workspace/xxx.pt`(`torch.save`),文件名带 diag 后缀便于清理时一把抓。
- **统计量集合**:shape/dtype/numel/mean/std/min/max/l2/nan/inf/finite(与 `runtime_trace.py` 的 `event()` 一致)。
- **synchronize 谨慎用**:读 GPU 张量统计前若不等同步会读到旧值,但 `synchronize` 有性能代价;加载期/一次性诊断可以加,运行期热点路径不要加。
- **不改控制流**:埋点只读张量、只写文件,绝不修改 tensor 数据或分支条件。

---

## 附:相关记忆与文档索引

- 记忆:`m3-trace-instrumentation`、`m3-trace-findings`、`m3-precision-nondeterminism`、`m3-moe-scaling-shared-rootcause`、`m3-indexer-voproj-bug`、`m3-lightning-indexer-fix`
- 修复文档:`/workspace/docs/m3-moe-routed-scaling-fix.md`、`/workspace/docs/m3-indexer-fix-summary.md`
- 诊断代码留档:`/workspace/llm-adaptation/minimax-m3-awq-int4/sglang_patches/diagnostics/`(runtime_trace.py、fused_moe_diag.py、wNa16_moe_diag.py、minimax_m3_diag.py)
- 启动脚本:`/workspace/start_awq_dcu.sh`(M3_TRACE 环境变量定义)
