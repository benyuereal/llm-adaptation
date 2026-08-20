# GLM-5.2-Channel-INT4-w4a8 — K100AI DCU 适配

本仓库是 GLM-5.2-Channel-INT4-w4a8 (GlmMoeDsaForCausalLM, slimquant INT4-w4a8) 模型
在海光 K100AI DCU (gfx928) 上的适配方案,修复**长输入(>2048 tokens)输出乱码**问题。

## 适配成果

| 指标 | 数据 |
|------|------|
| 模型 | GLM-5.2-Channel-INT4-w4a8 (78层 MoE, 256 experts, DSA 稀疏注意力) |
| 草稿模型 | MTP 投机解码 (num_speculative_tokens=3) |
| 硬件 | 8× K100AI DCU (gfx928), TP=8 |
| 修复前 | 长输入乱码率 ~70-90% (输出数字/符号垃圾 `0.0.0.0...` 或 `|}{}{}...`) |
| 修复后 | 长输入乱码率 0/10 (输出连贯中文) |
| 短输入 | 修复前后均正常 (短输入不走 DSA sparse indexer 的 topk 路径) |

## 问题现象

GLM-5.2 在 K100AI 上短输入正常,但**长输入(>2048 tokens)输出乱码**:

- 重复文本类输入:输出 `0.0.0.0.0...` 或 `|}{}{}{}...` 等无意义符号
- 乱码率约 70-90%,且高度可复现
- GLM-5.1 同代码同环境无此问题 → 5.1→5.2 回归

## 环境要求

| 项目 | 配置 |
|------|------|
| DCU | 海光 K100AI (gfx928) × 8, 64GB/卡 |
| DTK | 2604 (ROCm 兼容栈) |
| Python | 3.10 |
| vLLM | 0.15.1+das.opt1 (海光定制版, 无 vllm._C 原生 topk 算子) |
| tilelang | 定制版 (gfx928 DSA indexer 算子) |
| lightop | 定制版 (top_k_per_row / mqa_logits 等 HIP ASM) |
| PyTorch | 2.9.0 |

> 本分支 vLLM 未编译 `vllm._C` 原生 `top_k_per_row_prefill/decode` 算子,DSA indexer
> 只能走 lightop `top_k_per_row` + tilelang `mqa_logits`/`page_mqa_logits`。本修复针对
> 这条 gfx928 定制路径。

---

## 全流程(从零到跑通)

### 第 1 步:环境与镜像

拉取海光 DCU 基础镜像并启动容器,环境内已含 DTK / vLLM / tilelang / lightop 栈。

### 第 2 步:下载模型

```bash
# GLM-5.2-Channel-INT4-w4a8 (slimquant w4a8)
# 放到 /models/GLM-5.2-Channel-INT4-w4a8 (与 start.sh 默认路径一致)
```

### 第 3 步:拉代码并应用 patch

```bash
cd /workspace
git clone https://github.com/benyuereal/llm-adaptation.git
cd llm-adaptation

# 应用 DSA indexer 修复 (skip_topk + mqa_logits 返回值)
bash glm-5.2-channel-int4-w4a8/apply_patch.sh
```

`apply_patch.sh` 会:
- 把 `vllm_patches/modified/` 下 4 个修改文件覆盖到 vllm 安装目录(自动备份 `.bak`)
- 把 `vllm_patches/added/` 下 3 个新增文件复制到 vllm 安装目录
- 语法检查全部文件

### 第 4 步:启动服务

```bash
bash glm-5.2-channel-int4-w4a8/start.sh
```

### 第 5 步:验证

```bash
# 短输入
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/models/GLM-5.2-Channel-INT4-w4a8","prompt":"你好，请用一句话介绍深度学习。","max_tokens":40,"temperature":0}' \
  | python3 -m json.tool

# 长输入 (>2048 tokens, 修复前会乱码)
python3 -c '
import json, urllib.request
LONG = ("深度学习是机器学习的一个分支，它使用多层神经网络来学习数据的表示。" * 200
        + "\n\n请用一句话总结上面的内容。")
p = json.dumps({"model":"/models/GLM-5.2-Channel-INT4-w4a8","prompt":LONG,"max_tokens":40,"temperature":0}).encode()
print(json.loads(urllib.request.urlopen(urllib.request.Request("http://localhost:8000/v1/completions",data=p,headers={"Content-Type":"application/json"})).read())["choices"][0]["text"])
'
```

---

## 修复原理(两个独立根因)

### 根因 1:shared DSA indexer 层跑零权重(5.1→5.2 回归)

**背景**:GLM-5.2 的 DSA(Dynamic Sparse Attention)每个 attention 层有一个 indexer,
为每个 query 选 top-2048 个最相关的 key token 做稀疏注意力。GLM-5.1 所有 78 层都是
`full`(每层有 indexer 权重,每层自己算)。GLM-5.2 引入 `indexer_types`,改为
**full/shared 分组共享**:每 4 层一组,组内只有 1 层 `full`(有权重,跑 indexer),
其余 3 层 `shared`(**无权重,应复用同组 full 层算出的 topk_indices**)。

config 关键字段:
- `index_topk_freq=4`:每 4 层做一次 indexer
- `index_skip_topk_offset=3`:shared 层跳过 topk 的偏移
- `indexer_types`:78 层 full/shared 模式

full 层位置 = `[0,1,2,6,10,14,18,...,74]`(21 个),shared 层 = 其余 57 个。
官方用公式判定:`skip_topk = max(layer_id - 3 + 1, 0) % 4 != 0`,与 indexer_types 完全一致。

**Bug**:定制分支把 GLM-5.2 当 GLM-5.1,所有层都建 Indexer、都跑 indexer。shared 层
没有 checkpoint 权重(零初始化),跑零权重 indexer 算出全零 query → 无意义 topk;
更严重的是 `sparse_attn_indexer` 开头 `topk_indices_buffer[:n] = -1` 会**清空共享
buffer**,再用零 query 算出垃圾 indices 覆盖进去 → **把 full 层刚写好的有效结果冲掉**。

**修复**:移植官方 `skip_topk` 机制(照官方 vllm `deepseek_v2.py` 实现):
- `DeepseekV2MLAAttention.__init__`:按公式算 `_skip_topk`,shared 层**不建 Indexer**,
  MTP/nextn 层(layer_id >= num_hidden_layers)强制建 full indexer
- `MultiHeadLatentAttentionWrapper`:加 `skip_topk` 参数,forward 里
  `if self.indexer is not None and self.is_sparse and not self.skip_topk:` 才跑 indexer;
  shared 层直接复用 full 层写入的共享 `topk_indices_buffer`,不清零、不覆盖
- `flashmla_sparse.py`:shared 层 `indexer=None`,需透传 `topk_indices_buffer` 给 backend

涉及文件:`deepseek_v2.py`、`mla.py`、`flashmla_sparse.py`

### 根因 2:gfx928 prefill tilelang mqa_logits 返回值被丢弃(定制分支 bug)

**背景**:DSA indexer 的 prefill 打分用 `mqa_logits` 算 q·k 相关性。gfx938 走 lightop
`op.mqa_logits(..., logits_slice_view)`(**原地写入**输出 buffer);gfx928 走 tilelang
`mqa_logits(...)`(**返回新 tensor**,非原地写入)。

**Bug**:gfx928 分支照抄 gfx938 的原地写入模式,**丢弃了 `mqa_logits` 的返回值**:
```python
mqa_logits(q_slice, k_fp8, weights, ks, ke)   # 返回值被丢弃!
logits_slice = logits_slice_view[:q_len, :num_k]  # 读从未被写入的 view → 全 0
```
→ `logits_slice` 全 0 → topk 在全 0 上退化 → 输出垃圾 indices(-1/未初始化值)
→ 稀疏注意力选错 token → 长输入乱码。

探针证据:`pf_logits mean=0 std=0 zero%=100.0`(全 0),`pf_topk_out min=-1 max=2147483647`(垃圾)。

**修复**:接收 `mqa_logits` 返回值,直接作为 `logits_slice` 传给 topk(与官方
`fp8_fp4_mqa_logits` 用法一致),不 copy 到 padded view(避免 padding -inf 与非连续
stride 问题):
```python
logits_slice = mqa_logits(q_slice, k_fp8, weights, ks, ke)  # 直接用返回值
```

涉及文件:`sparse_attn_indexer.py`

### 新增的 tilelang 算子文件

gfx928 定制路径需要的三个 tilelang 算子(本仓库 `vllm_patches/added/`):
- `mqa_logits.py` — prefill indexer 打分(q·k 相关性,返回 logits)
- `paged_mqa_logits.py` — decode indexer 打分(分页 KV cache)
- `sparse_mla_fwd.py` — 主稀疏 MLA attention forward(按 topk_indices 选 token 做注意力)

调用处:
- `mqa_logits` → `sparse_attn_indexer.py` gfx928 prefill 分支
- `page_mqa_logits` → `sparse_attn_indexer.py` gfx928 decode 分支
- `flash_mla_sparse_fwd` → `flashmla_sparse.py` 主 attention forward

---

## 已知遗留(非乱码,量化精度边界)

**greedy 解码下部分问答类输入首 token = EOS(空输出)**:
- knowledge 类:sampling(temperature=0.7)输出**完全正常**,仅 greedy 空 → 量化精度
  使 EOS 与首 token 概率极接近,greedy 选错
- qa 类(期望单字回答):greedy 与 sampling 都可能 EOS,更极端
- **这不是乱码 bug**,是 w4a8 量化精度边界 + greedy 特性,用 sampling 或调整 EOS 逻辑缓解
- 详见 `docs/glm52-greedy-eos-boundary.md`

## MTP 投机解码

本修复在 MTP 开启下验证(乱码 0/10)。MTP 的 `index_share_for_mtp_iteration`
(官方 draft step 0 算 indices、step 1+ 复用的 set_skip_topk 机制)**未移植**,
仅影响 draft 效率,不影响主模型精度(MTP draft token 经主模型 verify,错误会被拒绝)。

---

## 目录结构

```
glm-5.2-channel-int4-w4a8/
├── README.md                  # 本文档
├── apply_patch.sh             # 一键应用修复
├── start.sh                   # 启动脚本
├── config.json                # 模型配置(含 indexer_types 等)
├── vllm_patches/
│   ├── modified/              # 修改的文件(覆盖到 vllm)
│   │   ├── deepseek_v2.py     # skip_topk 机制
│   │   ├── mla.py             # skip_topk 参数 + buffer 透传
│   │   ├── sparse_attn_indexer.py  # mqa_logits 返回值修复
│   │   └── flashmla_sparse.py # buffer 透传 + sparse_mla_fwd 调用
│   └── added/                 # 新增的 tilelang 算子
│       ├── mqa_logits.py
│       ├── paged_mqa_logits.py
│       └── sparse_mla_fwd.py
├── docs/                      # 详细文档
│   └── glm52-long-input-garbage-rootcause.md   # 完整根因分析与排查现场
└── memory/                    # 排查记忆
    └── ...
```

## 排查过程

详见 `docs/glm52-long-input-garbage-rootcause.md`(含探针定位现场、数据证据、
逐步收敛的排查思路)。
