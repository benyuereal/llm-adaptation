# MTP 投机解码优化：[DEQUANT-ATTN] attention int8 线性层加载时 dequant 成 bf16

模型：GLM-5.2-Channel-INT4-w4a8（slimquant_w4a8 量化），AMD ROCm gfx928，TP=8，
MTP 投机解码（`num_speculative_tokens=4`）。

## 根因（2026-08-24 profile 结论）

- w8a8 int8 GEMM（lmslim `matmul_int8`，Triton kernel）占 MTP decode GPU 时间
  **~60%**（见 `logs/profile_baseline_int8.txt`，`matmul_kernel` 60.49%）。
- 在 decode 小 M（verify M=5 / draft M=1）下，该 int8 GEMM 卡在 **~90µs 地板**，
  任何 Triton config 都打不破。
- 而库 bf16 GEMM（rocBLAS）没有这个地板，在这些 attention shape 上快 **2.5–2.8x**。

## 改动

唯一改动文件：`vllm/model_executor/layers/quantization/slimquant_w4a8.py`（共 2 处 hunk）：

1. 新增 `_dequant_attn_enabled()`：读环境变量 `VLLM_DEQUANT_ATTN`。
2. `get_quant_method()` 的 `LinearBase` 分支：`VLLM_DEQUANT_ATTN=1` 时强制
   `dequant=True` → 所有 attention 的 int8（w8a8）线性层在**加载时** dequant 成
   bf16，int8 GEMM 和 activation quant 被完全移除。
   **MoE（int4 w4a8）走 FusedMoE 分支，不受影响**（它本来就快）。

patch 本身不改变默认行为（`VLLM_DEQUANT_ATTN` 未设置时行为与原版完全一致），
需启动时显式开启。

## 环境变量

```bash
export VLLM_DEQUANT_ATTN=1   # 开启 attention int8 -> bf16 dequant
```

## 实测结果（单请求 decode，bench_mtp.py）

| 指标 | baseline (int8) | DEQUANT-ATTN | 变化 |
|---|---|---|---|
| 吞吐 | 19.68 tok/s | **39.25 tok/s** | **~2x** |
| 接受长度 (acc/drafts+1) | 2.887 | 2.960 | 无损（略升） |
| 显存 | — | — | **+3.59 GB/rank**（attention 权重 int8→bf16） |

## 目录结构

```
mtp/
├── apply_patch.sh              # 一键应用 patch（整文件覆盖，幂等，.bak 备份）
├── revert_patch.sh             # 从 .bak 回滚
├── run_mtp_dequant_attn.sh     # 启动脚本（已内置 export VLLM_DEQUANT_ATTN=1）
├── bench_mtp.py                # 吞吐 + 接受长度 benchmark
├── vllm_patches/
│   ├── modified/slimquant_w4a8.py   # 改后整文件（apply 用）
│   └── slimquant_w4a8.patch         # unified diff（参考/审阅用）
└── logs/
    ├── profile_baseline_int8.txt    # baseline profile（int8 GEMM 60.49%）
    ├── profile_dequant_attn.txt     # dequant 后 profile
    └── mtp_dequant_attn_serve.log   # 服务启动日志
```

## 使用方法

### 1. 应用 patch

```bash
bash /data1/csy/llm-adaptation/glm-5.2-channel-int4-w4a8/mtp/apply_patch.sh
```

- 幂等：目标文件已是最新版本则 SKIP。
- 覆盖前自动备份为 `slimquant_w4a8.py.bak`。
- 应用后自动做 `ast.parse` 语法检查。

### 2. 回滚

```bash
bash /data1/csy/llm-adaptation/glm-5.2-channel-int4-w4a8/mtp/revert_patch.sh
```

从 `.bak` 恢复原文件并清理 `__pycache__`。

### 3. 启动服务

```bash
bash /data1/csy/llm-adaptation/glm-5.2-channel-int4-w4a8/mtp/run_mtp_dequant_attn.sh
```

与 champion baseline 配置一致（TP=8、MTP num_spec=4、max-model-len 32768 等），
额外 `export VLLM_DEQUANT_ATTN=1`。

### 4. 验证（服务起来后）

```bash
python3 /data1/csy/llm-adaptation/glm-5.2-channel-int4-w4a8/mtp/bench_mtp.py [warmup] [measure] [max_tokens]
# 默认: 2 次 warmup, 3 次测量, 每次 512 token
```

输出：聚合 tok/s + spec decode 指标（drafts / accepted / 接受长度 /
per-position 命中率）。预期 ~39 tok/s、接受长度 ~2.9。

## 注意事项

- 显存 +3.59 GB/rank：attention 权重从 int8 变 bf16。当前
  `--gpu-memory-utilization 0.92` 下无压力，若显存紧张可下调该参数。
- 该优化只针对 decode 小 M 场景（MTP 投机解码）；大 batch prefill 下
  int8 GEMM 的利用率更高，收益会缩小（本 patch 默认关闭，按需开启）。
