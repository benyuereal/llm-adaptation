# MiniMax-M3 AWQ INT4 在 K100AI DCU 上的适配

## 适配成果

| 指标 | 数据 |
|------|------|
| 模型 | MiniMax-M3-AWQ-INT4 (456B, 128 experts MoE) |
| 硬件 | 8× K100AI DCU (gfx928), TP=8 |
| 单用户 decode 吞吐 | **~19 tokens/s** |
| CUDA Graph | 已启用 (bs=1-8) |
| 量化精度验证 | MoE kernel max_diff=0.024, Attention dequant max_diff<0.003 |
| 输出质量 | 通过事实问答、chat 多轮对话、思维链推理验证 |

---

## 一、项目背景

### 1.1 硬件与软件环境

| 项目 | 配置 |
|------|------|
| CPU | Hygon C86 7490 64-core |
| DCU | 海光 K100AI (gfx928) × 8，64GB/卡 |
| OS | KylinOS (kernel 4.19.90) |
| DTK | 2604 (ROCm 兼容栈) |
| Python | 3.10 |
| SGLang | 0.0.0.dev12695 (海光定制版) |
| Triton | 3.5.1 |
| PyTorch | 2.9.0 |

### 1.2 模型信息

| 项目 | 说明 |
|------|------|
| 模型 | MiniMax-M3-AWQ-INT4 |
| 参数量 | ~456B |
| 架构 | 60 层（3 dense + 57 MoE），128 experts/层，top_k=4 |
| 量化 | compressed-tensors pack-quantized, 4-bit, symmetric=false, group_size=32 |
| 视觉 | CLIP-based ViT（3D RoPE） |

### 1.3 最终启动命令

```bash
sglang serve \
    --model-path /models/MiniMax-M3-AWQ-INT4 \
    --mem-fraction-static 0.55 \
    --tp 8 \
    --dtype bfloat16 \
    --quantization compressed-tensors \
    --attention-backend triton \
    --mm-attention-backend triton_attn \
    --trust-remote-code \
    --host 0.0.0.0 --port 8080 \
    --cuda-graph-max-bs 8 \
    --chunked-prefill-size 4096 \
    --max-running-requests 64 \
    --schedule-policy fcfs
```

---

## 二、Patch 使用方法

### 2.1 获取 Patch

```bash
cd /workspace
git clone git@github.com:benyuereal/llm-adaptation.git
cd llm-adaptation/minimax-m3-awq-int4
```

仓库结构：
```
minimax-m3-awq-int4/
├── README.md                 # 概览
├── apply_patch.sh            # 一键应用脚本
├── start.sh                  # 启动脚本
├── docs/                     # 文档
├── sglang_patches/
│   ├── minimax-m3-awq-int4.patch  # unified diff（备用）
│   ├── modified/             # 修改后的源文件 + 独立 .patch
│   │   ├── compressed_tensors.py
│   │   ├── compressed_tensors.py.patch
│   │   ├── compressed_tensors_wNa16.py
│   │   ├── compressed_tensors_wNa16.py.patch
│   │   ├── compressed_tensors_wNa16_moe.py
│   │   ├── compressed_tensors_wNa16_moe.py.patch
│   │   ├── configuration_utils.py
│   │   ├── configuration_utils.py.patch
│   │   ├── fused_moe.py
│   │   ├── fused_moe.py.patch
│   │   ├── minimax_m3.py
│   │   ├── minimax_m3.py.patch
│   │   ├── minimax_m3_vl.py
│   │   └── minimax_m3_vl.py.patch
│   └── diagnostics/          # 诊断埋点备份（调试用）
```

### 2.2 一键应用（推荐）

```bash
bash apply_patch.sh
```

脚本会将 `sglang_patches/modified/` 下的文件复制到对应位置，并自动备份原文件（`.bak`）。

### 2.3 启动服务

```bash
bash start.sh
```

### 2.4 手动应用（备用）

如需手动操作：

```bash
SGLANG_ROOT=/usr/local/lib/python3.10/dist-packages/sglang/srt
TRANSFORMERS_ROOT=/usr/local/lib/python3.10/dist-packages/transformers
PATCH_DIR=sglang_patches/modified

cp $PATCH_DIR/compressed_tensors.py       $SGLANG_ROOT/layers/quantization/compressed_tensors/compressed_tensors.py
cp $PATCH_DIR/compressed_tensors_wNa16.py $SGLANG_ROOT/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py
cp $PATCH_DIR/compressed_tensors_wNa16_moe.py $SGLANG_ROOT/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py
cp $PATCH_DIR/minimax_m3.py               $SGLANG_ROOT/models/minimax_m3.py
cp $PATCH_DIR/minimax_m3_vl.py            $SGLANG_ROOT/models/minimax_m3_vl.py
cp $PATCH_DIR/fused_moe.py                $SGLANG_ROOT/layers/moe/moe_runner/triton_utils/fused_moe.py
cp $PATCH_DIR/configuration_utils.py      $TRANSFORMERS_ROOT/configuration_utils.py
```

### 2.5 修改文件列表

| 文件 | 包 | 修改内容 |
|------|-----|----------|
| `compressed_tensors_wNa16.py` | sglang | ZP reshape permute 修复 + HIP dequant |
| `compressed_tensors.py` | sglang | `symmetric` 参数传递 |
| `compressed_tensors_wNa16_moe.py` | sglang | MoE zp overflow 修复 |
| `minimax_m3.py` | sglang | dense/MoE 层判断 + quant_config 控制 |
| `minimax_m3_vl.py` | sglang | 权重名映射 + fallback |
| `fused_moe.py` | sglang | HIP combine 改为 torch.sum |
| `configuration_utils.py` | transformers | ALLOWED_LAYER_TYPES 加入 minimax_m3_sparse |

---

## 三、问题列表与解决方案

### 3.1 layer_types 校验失败

- **现象**：启动报 `minimax_m3_sparse` 不在支持的 layer_types 中
- **根因**：SGLang layer_type 校验白名单未包含此类型
- **修复**：白名单中添加 `minimax_m3_sparse`

### 3.2 CLIPVisionConfig.rope_theta 缺失

- **现象**：视觉编码器初始化时 `AttributeError: rope_theta`
- **根因**：`rope_theta` 嵌套在子配置中，未提取到顶层
- **修复**：config 解析时将 `rope_theta` 提取到 `CLIPVisionConfig` 顶层

### 3.3 Tokenizer 回退

- **现象**：`AutoTokenizer` 加载失败
- **根因**：模型使用自定义 tokenizer 类，环境中缺少依赖
- **修复**：回退为 `PreTrainedTokenizerFast` + 嵌入的 `chat_template`

### 3.4 VMFault 崩溃

- **现象**：推理时 GPU 报 `Invalid address access`，SIGABRT
- **根因**：Triton cache 中残留的不兼容 kernel binary
- **修复**：清理 `/root/.triton/cache/tmp.*`，服务重新编译 kernel

### 3.5 MoE zero_point overflow

- **现象**：`RuntimeError: value cannot be converted to type int without overflow`
- **根因**：对全零 zp 的 fallback `fill_(0x88888888)` 超出 int32 范围
- **修复**：
```python
for ei in range(w13_zp.shape[0]):
    if (w13_zp[ei] == 0).all():
        w13_zp[ei].view(torch.uint8).fill_(0x88)
```
- **文件**：`compressed_tensors_wNa16_moe.py`

### 3.6 Dense layers 0-2 误建为 MoE

- **现象**：layers 0-2 有 128 个空 expert 参数，checkpoint 权重无法加载
- **根因**：`moe_layer_freq=None` 时默认全 MoE，`mlp_layer_types` 字段未被使用；同时 dense 层传入 `quant_config` 但 checkpoint 为全精度
- **修复**：
```python
mlp_layer_types = getattr(config, "mlp_layer_types", None)
if mlp_layer_types is not None:
    self.is_layer_sparse = mlp_layer_types[layer_id] != "dense"
layer_quant_config = quant_config if self.is_layer_sparse else None
```
- **文件**：`minimax_m3.py`

### 3.7 Shared experts 权重未加载

- **现象**：`shared_experts.gate_proj.weight not found in params_dict`
- **根因**：checkpoint 中 shared_experts 为全精度，模型传入 `quant_config` 导致参数名为 `weight_packed`，与 checkpoint `weight` 不匹配
- **修复**：`self.shared_experts = MiniMaxM3MLP(..., quant_config=None)`
- **文件**：`minimax_m3.py`

### 3.8 Attention symmetric 参数缺失

- **现象**：所有层 attention `weight_zero_point` 无法加载
- **根因**：`_get_scheme_from_parts()` 创建 `CompressedTensorsWNA16` 时未传 `symmetric` 参数，默认 `True`，导致 `weight_zero_point` 不注册
- **修复**：
```python
return CompressedTensorsWNA16(
    ...,
    symmetric=weight_quant.symmetric,  # 传递实际值 False
)
```
- **文件**：`compressed_tensors.py:552`

### 3.9 Attention dequant ZP reshape 错误（最终精度 root cause）

- **现象**：模型能理解部分上下文但输出错误（"sky is the color of water"）、重复、chat 复述 system prompt
- **根因**：`_ct_dequantize_hip` 中 ZP 解包 reshape 顺序错误。`[N//8, K_groups, 8]` 做 C-order reshape 时 nibble 维度和 K_groups 混合，应先 permute 让 nibble 与 N 合并
- **修复**：
```python
# 错误
zp_unpacked = zp_unpacked.reshape(N // pack_factor * pack_factor, num_groups)

# 正确
zp_unpacked = zp_unpacked.permute(0, 2, 1).contiguous().reshape(-1, num_groups)
zp_unpacked = zp_unpacked[:N, :]
```
- **影响**：57 层 × 4 projection = 228 个 attention 矩阵的 dequant 全部有系统偏差
- **验证**：修复后 7 个不同层/projection 测试，max_diff < 0.003
- **文件**：`compressed_tensors_wNa16.py:81`

### 3.10 MoE combine 兼容性

- **现象**：HIP 上 `moe_sum_reduce_triton` 结果不可靠
- **根因**：Triton sum reduce kernel 在 gfx928 上兼容性问题
- **修复**：`torch.sum(intermediate_cache3, dim=1, out=out_hidden_states)`
- **文件**：`fused_moe.py`

---

## 四、验证方法

### 4.1 MoE Kernel 端到端验证

使用真实 checkpoint 数据，TP=8 实际 per-GPU 形状（E=128, N=768, K=6144），CPU ground truth 对比 GPU kernel 输出。结果：max_diff = 0.024（BF16 精度）。

### 4.2 Attention Dequant 验证

对 `_ct_dequantize_hip` 用 layer 3/10/30/59 的 q_proj/k_proj/o_proj 真实权重测试。结果：max_diff 全部 < 0.003。

### 4.3 逐层 Hook 验证

在 Model.forward 中 dump 每层 hidden_states/residual 统计，确认无 NaN/Inf，数值在合理范围。

### 4.4 最终输出测试

```
"The color of the sky is" → " blue."  ✓
"What is 2+3?" (chat)     → <mm:think>... 正常推理  ✓
```

---

## 五、排查方法论与教训

### 5.1 方法论

1. **逐层 hook** → 快速定位异常层
2. **分阶段诊断** → MoE 各环节独立检查（gate→experts→combine→allreduce）
3. **端到端数值验证** → checkpoint ground truth 对比函数输出（逐元素，非仅统计量）
4. **不接受模糊结论** → "精度损失"前必须做逐元素验证
5. **reshape/permute 必验证** → 多维 tensor 操作后检查元素映射正确性

### 5.2 关键教训

| 教训 | 说明 |
|------|------|
| 默认参数是 bug 源 | `symmetric=True` 默认值导致 zp 丢失 |
| reshape 顺序高频出错 | C-order reshape 容易混淆维度语义 |
| 日志警告不能忽略 | "not found" 直接指向权重加载失败 |
| 部分正确 ≠ 无 bug | "sky is water" 说明 dequant 有系统偏差 |
| inplace 诊断需 clone | 必须在 inplace 操作前保存快照 |

### 5.3 自主性评估

| 阶段 | 自主/引导 | 说明 |
|------|-----------|------|
| Overflow 修复 | 自主 | 独立发现并修复 |
| Dense/MoE 判断 | 自主 70% | 找到根因但过程绕路 |
| Shared experts | 自主 | 独立发现 |
| MoE kernel 验证 | 引导 60% | 用户要求先验证再改方案 |
| Attention symmetric | 引导 70% | 用户指出检查方向 |
| ZP reshape bug | 引导 50% | 用户拒绝"精度损失"结论，要求继续排查 |

核心弱点：容易基于不充分证据做强结论（"Triton bug"、"正常精度损失"），需要引导拉回正确方向。

---

## 六、运维注意事项

| 事项 | 说明 |
|------|------|
| Triton cache | 不要清理 `/root/.triton/cache/`，已编译 kernel 有效 |
| 显存配置 | `mem_fraction_static=0.55`，每卡权重约 30GB |
| 残留进程 | 启动前确认：`ps aux \| grep sglang` |
| CUDA Graph | 捕获 bs=1-8，首次请求触发 graph capture |
| 诊断埋点 | 备份在 `sglang_patches/diagnostics/`，需要时可恢复 |

### MoE Kernel Config（可选，性能调优）

仓库中 `sglang_patches/added/configs/` 包含 K100_AI 上 autotuning 得到的 MoE kernel 配置：
- `E=128,N=192,device_name=K100_AI,dtype=int4_w4a16.json`
- `E=128,N=192,device_name=K100_AI,dtype=int4_w4a16_down.json`

**默认不启用**（不影响精度，仅影响性能）。如需启用：

```bash
# 方法一：取消 apply_patch.sh 中 config 相关行的注释后重新执行
# 方法二：手动复制
CONFIG_DIR=/usr/local/lib/python3.10/dist-packages/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_5_1
cp sglang_patches/added/configs/*.json $CONFIG_DIR/
```

不启用时日志会提示 "Using default MoE kernel config. Performance might be sub-optimal!"，可忽略。
