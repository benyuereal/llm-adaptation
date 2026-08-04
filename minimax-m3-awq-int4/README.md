# MiniMax-M3 AWQ INT4 — K100AI DCU 适配

本仓库是 MiniMax-M3-AWQ-INT4 模型在海光 K100AI DCU (gfx928) 上的完整适配方案，包含量化精度修复、lightning indexer 修复，以及 EAGLE3 投机解码（专用 verify kernel + graph buffer 根治 VMFault）。

## 适配成果

| 指标 | 数据 |
|------|------|
| 模型 | MiniMax-M3-AWQ-INT4 (456B, 128 experts MoE) |
| 草稿模型 | MiniMax-M3-EAGLE3 (EAGLE3 投机解码) |
| 硬件 | 8× K100AI DCU (gfx928), TP=8 |
| 单用户 decode | ~19 tokens/s (CUDA Graph) |
| EAGLE3 并发压测 | 8 并发 × 1000 tokens 无 VMFault, accept rate ~0.59 |
| 精度验证 | MoE max_diff=0.024, Attention max_diff<0.003 |

## 环境要求

| 项目 | 配置 |
|------|------|
| DCU | 海光 K100AI (gfx928) × 8，64GB/卡 |
| DTK | 2604 (ROCm 兼容栈) |
| Python | 3.10 |
| SGLang | 0.0.0.dev12695 (海光定制版) |
| Triton | 3.5.1 |
| PyTorch | 2.9.0 |

> 以下环境与镜像请参考官方文档：<https://developer.sourcefind.cn/codes/modelzoo/minimax-m3>

---

## 全流程（从零到跑通）

### 第 1 步：下载镜像

按照官方文档 <https://developer.sourcefind.cn/codes/modelzoo/minimax-m3> 拉取适配 MiniMax-M3 的 DCU 基础镜像并启动容器，环境内已包含上述 DTK / SGLang / Triton / PyTorch 栈。

### 第 2 步：下载模型

需要两个模型：**AWQ 主模型** 和 **EAGLE3 草稿模型**。

| 模型 | 说明 | 下载地址 |
|------|------|----------|
| MiniMax-M3-AWQ-INT4 | 主模型，AWQ INT4 量化 | <https://www.modelscope.cn/models/cyankiwi/MiniMax-M3-AWQ-INT4> |
| MiniMax-M3-EAGLE3 | 草稿模型，EAGLE3 投机解码 | <https://www.modelscope.cn/models/Inferact/MiniMax-M3-EAGLE3> |

推荐用 modelscope 下载，并放到 `/models` 目录下（与 `start.sh` 中的默认路径一致）：

```bash
pip install modelscope

# 主模型
modelscope download --model cyankiwi/MiniMax-M3-AWQ-INT4 \
    --local_dir /models/MiniMax-M3-AWQ-INT4

# 草稿模型
modelscope download --model Inferact/MiniMax-M3-EAGLE3 \
    --local_dir /models/MiniMax-M3-EAGLE3
```

> 若放其它路径，需同步修改 `start.sh` 中的 `--model-path` 与 `--speculative-draft-model-path`。

### 第 3 步：拉代码并安装 patch

```bash
cd /workspace
git clone https://github.com/benyuereal/llm-adaptation.git
cd llm-adaptation

# (1) 应用 AWQ INT4 + lightning indexer 量化精度修复
bash minimax-m3-awq-int4/apply_patch.sh

# (2) 安装 EAGLE3 投机解码 patch (专用 verify kernel + graph buffer)
bash minimax-m3-awq-int4/sglang_patches/eagle3-v2/install.sh
```

`apply_patch.sh` 会把 `sglang_patches/modified/` 下的文件复制到
`/usr/local/lib/python3.10/dist-packages/sglang/srt/`，并自动备份原文件（`.bak`）。

`install.sh` 把 EAGLE3 适配代码应用到容器内的 sglang，使 MiniMax-M3 (VL 类)
支持 EAGLE3 投机解码。常用参数：

```bash
bash minimax-m3-awq-int4/sglang_patches/eagle3-v2/install.sh            # 安装
bash minimax-m3-awq-int4/sglang_patches/eagle3-v2/install.sh --rollback # 回滚
bash minimax-m3-awq-int4/sglang_patches/eagle3-v2/install.sh --check    # 仅检查是否已安装
```

### 第 4 步：等待模型启动并使用

```bash
# 启动服务（默认端口 8082，开启 EAGLE3 投机解码）
bash minimax-m3-awq-int4/start.sh
```

`start.sh` 已内置 DTK/HIP/lightop 环境变量与 EAGLE3 speculative 参数，
要求先执行第 3 步的 `install.sh`。日志默认写入 `/workspace/logs/sglang_eagle3.log`。

首次启动会先加载主模型与草稿模型，并触发 CUDA Graph 捕获，需等待若干分钟。
看到日志中出现 `The server is fired up and ready to roll!` 即可使用：

```bash
# OpenAI 兼容接口
curl http://localhost:8082/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "default",
        "messages": [{"role": "user", "content": "你好，请介绍一下你自己"}]
    }'
```

---

## 目录结构

```
minimax-m3-awq-int4/
├── README.md                 # 本文件
├── apply_patch.sh            # 一键应用量化精度修复 patch
├── start.sh                  # EAGLE3 投机解码启动脚本
├── patch_config.py           # patch 配置
├── docs/                     # 适配技术文档
│   └── minimax-m3-k100ai-adaptation.md  # 适配技术文档
├── sglang_patches/
│   ├── README.md             # patch 说明
│   ├── minimax-m3-awq-int4.patch  # unified diff（备用）
│   ├── modified/             # 修改后的源文件 + .patch
│   ├── added/                # 新增文件（MoE kernel config 等）
│   ├── diagnostics/          # 诊断埋点备份
│   └── eagle3-v2/            # EAGLE3 投机解码 patch
│       ├── install.sh        # 一键安装 / 回滚 / 检查
│       ├── README.md         # EAGLE3 VMFault 根因与修复
│       ├── modified/         # 适配源文件（含 verify/ 子模块）
│       └── tests/            # VMFault 复现 / graph buffer 测试
└── memory/                   # 排查记录与根因文档
```

## 修改文件列表

| 文件 | 修改内容 |
|------|----------|
| compressed_tensors_wNa16.py | ZP reshape permute 修复 + HIP dequant |
| compressed_tensors.py | symmetric 参数传递 |
| compressed_tensors_wNa16_moe.py | MoE zp overflow 修复 |
| minimax_m3.py | dense/MoE 层判断 + quant_config 控制 + lightning indexer 修复 |
| minimax_m3_vl.py | 权重名映射 + EAGLE3 target 侧接口 |
| model_config.py | lightning indexer fix Bug A/B |
| minimax_sparse_backend.py | TARGET_VERIFY 路由 + graph buffer 预分配 |
| fused_moe.py | HIP combine 改为 torch.sum |
| configuration_utils.py | ALLOWED_LAYER_TYPES 加入 minimax_m3_sparse |

## 相关文档

- 适配技术全流程与问题清单：[docs/minimax-m3-k100ai-adaptation.md](docs/minimax-m3-k100ai-adaptation.md)
- EAGLE3 VMFault 两层根因与修复：[sglang_patches/eagle3-v2/README.md](sglang_patches/eagle3-v2/README.md)
- patch 详细说明：[sglang_patches/README.md](sglang_patches/README.md)
