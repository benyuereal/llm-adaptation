# EAGLE3 投机解码 patch — MiniMax-M3 (Hygon DCU)

给 MiniMax-M3 AWQ-INT4 (W4A16 moe-only) 接 EAGLE3 投机解码的 sglang 补丁。
适配海光 DCU (gfx936/gfx928),搭配 `Inferact/MiniMax-M3-EAGLE3` draft head。

## 成果

- 纯 W4A16 eager 5 tok/s → **EAGLE3 + cuda graph 16-22 tok/s (峰 21.7)**,accept ~0.78
- 输出正确(纯文本,不乱码不复读)
- = **4x over eager,1.7x over 纯 W4A16 cuda graph (13 tok/s)**

## 文件清单

| 文件 | 作用 |
|---|---|
| `install.sh` | **一键安装** patch 到 site-packages(含备份/验证/回滚) |
| `start_eagle3.sh` | 启动 EAGLE3 sglang 服务(端口 8082) |
| `modified/minimax_m3_vl.py{,.patch}` | VL 类补 EAGLE3 target 侧接口:`set_eagle3_layers_to_capture` / `get_embed_and_head` / aux-aware forward |
| `modified/minimax_sparse_backend.py{,.patch}` | sparse backend 兜底 EAGLE3 TARGET_VERIFY 字段补全 + cuda graph 7 处 graph-unsafe 修复 |
| `modified/utils.py{,.patch}` | `get_cu_seqblocks` graph-safe(host sync → 静态上界) |
| `tests/test_m3_eagle3_verify_sparse.py` | 单元测试(不起服务,验证 verify 字段补全逻辑) |
| `docs/MiniMax-M3-EAGLE3-工作记录.md` | 完整工作记录(背景/踩坑/修复/选型/待办) |

## 一键使用

```bash
# 1. 安装 patch(备份原文件 → 覆盖 → 清缓存 → 验证)
bash sglang_patches/eagle3/install.sh

# 2. 启动 EAGLE3 服务
bash sglang_patches/eagle3/start_eagle3.sh

# 3. (可选)跑单元测试
cd sglang_patches/eagle3/tests && python3 test_m3_eagle3_verify_sparse.py

# 回滚 patch
bash sglang_patches/eagle3/install.sh --rollback
```

## 前置依赖

- sglang (dev,DCU dtk2604 build)
- MiniMax-M3 AWQ-INT4 模型权重(moe-only W4A16 量化产物)
- `Inferact/MiniMax-M3-EAGLE3` draft head(BF16,6.5GB)
- 量化 patch(见上级目录 `sglang_patches/modified/`,EAGLE3 依赖模型能正确加载量化权重)
- `sitecustomize.py` 注册 `minimax_m3_sparse` layer type(见上级目录,使 transformers 能加载 M3 config)

## 解决的上游缺口

EAGLE3 + MiniMax-M3 + sparse attention 三者组合在 sglang 上游有 4 处缺口(工作记录第四章详述):

1. **VL 类缺 EAGLE3 接口** — `MiniMaxM3SparseForConditionalGeneration`(量化产物加载的类)没有 `set_eagle3_layers_to_capture` / `get_embed_and_head` / aux-aware forward,只有 text-only 类有 → 启动即 AttributeError
2. **aux 捕获链路断裂** — `MiniMaxM3Model.forward` 读 `_is_layer_to_capture` 但无代码 setattr 它 → 补在 `set_eagle3_layers_to_capture` 里
3. **TARGET_VERIFY 字段全 None** — `ForwardBatch.init_new` 把 target_verify 归 decode 分支不填 extend 字段,sparse backend 的 forward_extend 直接读 → `max(None)` / `.device` 崩
4. **verify 时 seq_lens 语义** — verify 路径 `batch.seq_lens` 是 prefix(非 prefix+draft,scheduler 验后加 accept_lens),sparse backend 误当 prefix+draft → attention 位置错 → 输出乱码/复读。正确:`extend_prefix_lens=seq_lens`,`seq_lens=prefix+draft` 重建

另 7 处 cuda graph 不安全点(host sync 在 capture 下非法),详见 `minimax_sparse_backend.py.patch`。

## 与量化 patch 的关系

本目录只含 **EAGLE3 投机解码** 专属改动。量化适配(W4A16 MoE kernel、compressed_tensors、sparse attention 共享内存等)在上级 `sglang_patches/modified/`,需先应用量化 patch 让模型能加载,再应用本 EAGLE3 patch。

详见 `docs/MiniMax-M3-EAGLE3-工作记录.md`。
