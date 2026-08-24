# DSpark 投机解码 backport（GLM-5.2-Channel-INT4-w4a8）

## 背景

DSpark 是 vLLM 上游的**半自回归并行投机解码**方法：一次并行 backbone forward
起草一整块 `num_speculative_tokens` 个 token（复用 DFlash 的 context-KV 预计算 +
query-block forward 机制），再用一个轻量级**顺序 Markov head** 注入块内依赖。
相比 MTP 的逐 token 自回归 draft，DSpark 的 draft 阶段是并行的，理论上接受率
和吞吐更好。

预装 vllm（`/usr/local/lib/python3.10/dist-packages/vllm`，
`0.15.1+das.opt1.alpha.dtk2604` 定制版）**不含** DSpark 支持。本目录把
2026-08-22 已 backport 到预装版的 DSpark 改动打包成独立 patch，模式仿照
`../mtp/`（整文件覆盖式 apply、幂等、`.bak` 备份、`ast.parse` 检查）。

参考实现（更新版 vllm，含完整 dspark 支持）：`/data1/csy/vllm`。
原始基线（用于生成 diff）：`/data1/lxf/vllm-0.15.1+das.opt1.alpha.dtk2604-cp310-cp310-linux_x86_64.whl`
（与预装版同版本 wheel；wheel 内文件为 CRLF 行尾，diff 已做 LF 归一化）。

## ⚠️ 当前状态：部分完成、未启用、未测试

**DSpark 当前不使用。本 patch 仅供以后有场景时应用，默认不要在生产环境启用。**

| 任务 | 状态 | 说明 |
|---|---|---|
| #3 backport 框架 | in_progress | 配置/注册/runner 接线已完成（见文件清单），但整体未联调 |
| #13 适配预装版 9 参接口的 DSparkSpeculator | in_progress | `spec_decode/dspark/speculator.py` 已按预装版 `EagleSpeculator` 接口（3 参 `set_attn` / 9 参 `propose` / `run_model` / `capture_model`）独立重写，**未跑通** |
| #14 DSparkCudaGraphManager + non-causal 支持 | pending | 当前 speculator 走预装版 `capture_graphs` 通用路径；上游的 `DFlashCudaGraphManager`（FULL 图覆盖并行 backbone + 顺序 Markov 采样）尚未 backport；`AttentionConfig.use_non_causal` 字段已加但 non-causal 路径未验证 |

已知缺口（启用前必须补齐/确认）：

1. **`v1/worker/gpu/spec_decode/dflash/speculator.py` 缺失**。
   `spec_decode/__init__.py` 的 `use_dflash()` 分支会 import 它，但 DFlash
   speculator 尚未 backport（`dflash/` 下只有 `__init__.py` + `utils.py`）。
   只跑 `method="dspark"` 不受影响（该分支是惰性 import），但
   `method="dflash"` 会直接 ImportError。
2. **`qwen3_dflash.py` 是 DSpark 的硬依赖**（`qwen3_dspark.py` 从它 import
   `DFlashQwen3ForCausalLM` / `DFlashQwen3Model`，`dspark/utils.py` 从它 import
   `dflash_has_any_non_causal`），已一并打包。
3. 预装版 `SpeculativeConfig` 没有上游的 `attention_backend` / `kv_cache_dtype`
   / `dspark_draft_topk` / `enable_adaptive_verification` 字段，
   `dspark/utils.py` 已用 `getattr` 防御性处理；上游测试里的这些配置项
   **不要**写进 speculative_config。
4. 整条链路（draft 加载 → aux hidden states → Markov 采样 → CUDA graph）
   **从未端到端跑过**。

## 文件清单

### 纯新增（9 个，`vllm_patches/added/`，保持相对路径）

| 文件 | 作用 |
|---|---|
| `model_executor/models/qwen3_dspark.py` | DSpark draft 模型（`Qwen3DSparkForCausalLM`，继承 DFlash Qwen3 draft，加 Markov head） |
| `model_executor/models/qwen3_dflash.py` | DFlash Qwen3 draft 模型（DSpark 的基类/依赖） |
| `v1/worker/gpu/spec_decode/dspark/__init__.py` | 包标记 |
| `v1/worker/gpu/spec_decode/dspark/speculator.py` | **核心**：`DSparkSpeculator`（按预装版 9 参接口独立重写，#13） |
| `v1/worker/gpu/spec_decode/dspark/utils.py` | `load_dspark_model()`：draft 加载 + embed/lm_head 共享 |
| `v1/worker/gpu/spec_decode/dflash/__init__.py` | 包标记 |
| `v1/worker/gpu/spec_decode/dflash/utils.py` | `load_dflash_model()`（DFlash 用，DSpark 不直接依赖） |
| `v1/worker/gpu/spec_decode/eagle3_utils.py` | aux hidden state 层解析（eagle3/dflash/dspark 通用） |
| `v1/worker/gpu/spec_decode/eagle_utils.py` | `_should_share` / `get_target_lm_head`（draft 加载辅助） |

### 被修改的原有文件（15 个，`vllm_patches/modified/`，整文件覆盖）

| 文件 | dspark 相关改动 |
|---|---|
| `transformers_utils/configs/speculators/algos.py` | 新增 `update_dspark` / `update_dflash` speculator 配置转换 |
| `config/attention.py` | 新增 `use_non_causal` 字段（`load_dspark_model` 需要）；另含 `tq_max_kv_splits_for_cuda_graph`（TurboQuant 相关，随整文件带入） |
| `config/speculative.py` | `DSparkModelTypes`/`DFlashModelTypes`、`use_dspark()`/`use_dflash()`、`glm_moe_dsa` model_type、eagle 系 method 校验分支 |
| `config/vllm.py` | `use_v2_model_runner` property（method=dspark 时强制 V2 runner）；**另含非 dspark 改动**（ray async scheduling、lightly_cp 校验、cudagraph sizes 环境变量） |
| `config/utils.py` | 新增 pydantic 兼容的 `replace()`（draft 加载用） |
| `model_executor/models/registry.py` | 注册 `Qwen3DSparkModel` / `DFlashDraftModel`；**另含** `GlmMoeDsaForCausalLM`、`Qwen3ASRForConditionalGeneration`（GLM-5.2 / qwen3-asr backport 带入） |
| `model_executor/models/deepseek_v2.py` | `SupportsEagle3` 接口 + `aux_hidden_state_layers` 收集（target 侧 aux 输出）；**注意：该文件同时承载 GLM-5.2 基础 backport（DSA/MLA 等），diff 相对 wheel 很大（~876 行），其中大部分不是 dspark 改动** |
| `model_executor/models/glm4_moe_lite.py` | 同 deepseek_v2：`SupportsEagle3` + aux hidden state 收集 |
| `attention/layer.py` | draft 层（layer idx ≥ target 层数）豁免 MLA/sliding-window 互斥断言；**另含大量非 dspark 改动**（TurboQuant buffers、fp8 overflow、`FusedQkvSplitRmsNormRopeAttention` 等，~568 行 diff） |
| `v1/worker/gpu/model_runner.py` | V2 runner 接线：`use_aux_hidden_state_outputs`、aux 层配置、`(hidden_states, aux_hidden_states)` 解包、`take_draft_token_ids()`、`attn_groups` 改名 |
| `v1/worker/gpu/cudagraph_utils.py` | CUDA graph 模式下 aux hidden states 的持久 buffer（否则 drafter 回退到 last_hidden_states）；`attn_groups` 改名 |
| `v1/worker/gpu/attn_utils.py` | `init_attn_backend` 按 (backend, kv_cache_spec) 拆子组返回 `attn_groups`（target MLA + DSA indexer + draft SWA 混排需要） |
| `v1/worker/gpu/spec_decode/__init__.py` | `init_speculator` 增加 dspark / dflash 分支 |
| `v1/worker/gpu_worker.py` | `use_v2_model_runner` 尊重 `VllmConfig` property（dspark 强制 V2） |
| `v1/core/sched/scheduler.py` | 同 gpu_worker 的 V2 判定（否则 scheduler 漏发 `prefill_token_ids`）；**另含非 dspark 改动**（PP balance、DP connector 调度） |

> 标注"另含非 dspark 改动"的文件是**整文件覆盖**：apply 会把当时（2026-08-22
> backport 时点）的完整文件状态带过去，包括同文件里其他 backport 的改动。
> 这些改动当时已在线上运行，回滚（revert）会一并撤掉，请知悉。

## 目录结构

```
dspark/
├── apply_patch.sh              # 一键应用（整文件覆盖/新增，幂等，.bak 备份，ast.parse 检查）
├── revert_patch.sh             # 从 .bak 回滚 + 删除新增文件
├── README.md
└── vllm_patches/
    ├── modified/               # 15 个被修改文件的当前（已 backport）整文件副本（保持相对路径）
    │   ├── attention/layer.py
    │   ├── config/{attention,speculative,utils,vllm}.py
    │   ├── model_executor/models/{deepseek_v2,glm4_moe_lite,registry}.py
    │   ├── transformers_utils/configs/speculators/algos.py
    │   ├── v1/core/sched/scheduler.py
    │   ├── v1/worker/gpu/{attn_utils,cudagraph_utils,model_runner}.py
    │   ├── v1/worker/gpu/spec_decode/__init__.py
    │   └── v1/worker/gpu_worker.py
    ├── added/                  # 9 个纯新增文件副本（保持相对路径）
    │   ├── model_executor/models/{qwen3_dspark,qwen3_dflash}.py
    │   ├── v1/worker/gpu/spec_decode/dspark/{__init__,speculator,utils}.py
    │   ├── v1/worker/gpu/spec_decode/dflash/{__init__,utils}.py
    │   └── v1/worker/gpu/spec_decode/{eagle3_utils,eagle_utils}.py
    └── *.patch                 # 15 个 unified diff（wheel 原始版 → 当前版，LF 归一化，供审阅）
```

## 使用方法

### 1. 应用 patch

```bash
bash /data1/csy/llm-adaptation/glm-5.2-channel-int4-w4a8/dspark/apply_patch.sh
```

- 幂等：目标文件已是最新版本则 SKIP。
- 覆盖前自动备份为 `<file>.bak`（只在首次覆盖时创建，保留最初原始版）。
- 应用后自动做 `ast.parse` 语法检查。
- 支持 `DRY_RUN=1`（只打印不改动）和 `VLLM_ROOT=/path`（指定 vllm 根目录，测试用）。

### 2. 回滚

```bash
bash /data1/csy/llm-adaptation/glm-5.2-channel-int4-w4a8/dspark/revert_patch.sh
```

从 `.bak` 恢复 15 个被修改文件，删除 9 个新增文件及空的 `dspark/`、`dflash/`
子目录。`.bak` 文件保留在原地，彻底清理需手动删除。

## 启用方式（backport 完成并测试通过之后）

DSpark 只由 **V2 GPU model runner** 实现；`config/vllm.py` 的
`use_v2_model_runner` property 会在 `method="dspark"` 时自动强制 V2
（显式设置 `VLLM_USE_V2_MODEL_RUNNER` 环境变量可覆盖）。

参考 `/data1/csy/vllm` 的测试（`tests/v1/e2e/spec_decode/acceptance_rates/dspark/test_dspark.py`），
speculative_config 用法：

```bash
vllm serve <target-model> \
  --trust-remote-code \
  --speculative-config '{
    "method": "dspark",
    "model": "<dspark-draft-checkpoint>",
    "num_speculative_tokens": 7,
    "draft_sample_method": "probabilistic"
  }'
```

注意：预装版 `SpeculativeConfig` 没有上游的 `attention_backend` /
`kv_cache_dtype` / `dspark_draft_topk` / `enable_adaptive_verification`
字段，**不要**写进配置（写了会报 unexpected field）。draft checkpoint 的
config.json 需含 `aux_hidden_state_layer_ids`（或 `dspark_target_layer_ids` /
`target_layer_ids`）、`mask_token_id`（或 `dspark_noise_token_id`）等字段，
由 `update_dspark` / `eagle3_utils` 解析。

## 风险提示

1. **未验证**：整条 DSpark 链路从未端到端跑通（#13 in_progress、#14 pending）。
   启用前必须先在小流量/测试环境跑通正确性 + 接受率测试。
2. **整文件覆盖的副作用**：`config/vllm.py`、`attention/layer.py`、
   `v1/core/sched/scheduler.py`、`model_executor/models/registry.py`、
   `model_executor/models/deepseek_v2.py` 等文件同时携带其他 backport 的改动
   （TurboQuant、ray、PP balance、GLM-5.2 基础 backport 等）。apply 会把
   2026-08-22 时点的完整文件状态带过去；revert 会一并撤掉这些改动。
   在"其他 backport 已应用"的环境上 revert 本 patch 可能破坏其他功能。
3. **dflash 分支是坏的**：`spec_decode/__init__.py` 的 dflash 分支指向不存在的
   `dflash/speculator.py`。只跑 dspark 无影响；跑 dflash 会 ImportError。
4. **CUDA graph 路径未验证**（#14 pending）：当前依赖通用 `capture_graphs`
   路径 + aux buffer 持久化，上游的 `DFlashCudaGraphManager` 未 backport；
   建议先 `--enforce-eager` 验证正确性，再开 CUDA graph。
5. 本 patch 生成时线上 vllm 已处于 backport 状态，apply 到"当前线上"会是
   全 SKIP（幂等）；它的主要用途是**在新环境/新镜像上复现这套 backport**。
