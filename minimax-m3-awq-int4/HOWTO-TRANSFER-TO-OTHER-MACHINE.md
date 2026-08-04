# 转移到其他机器/容器 — 操作指南

本文件说明如何把当前工作(MiniMax-M3-AWQ-INT4 + EAGLE3 适配 + graph-safety 修复)
转移到其他机器/容器继续验证。有三种方式,推荐 **方式 A(git push/pull)**。

---

## 方式 A:git push/pull(推荐,最可靠)

当前仓库已是 git 仓库,remote = `git@github.com:benyuereal/llm-adaptation.git`(ssh 已通)。

### 在当前机器(本机)推送

```bash
cd /workspace/llm-adaptation/minimax-m3-awq-int4

# 1. 添加所有改动 (含新增 memory/ verify/ tests/)
git add -A

# 2. 提交
git commit -m "EAGLE3 verify: Strategy B + graph-safety 修复

- 新增 verify/ 3 个 graph-safe kernel (固定上界 score buffer + OOB 双保险)
- minimax_sparse_backend: target_verify 路由新 verify kernel;
  forward_extend verify 分支 graph-safety 修复 (绕过临时 extend_seq_lens,
  改用 seq_lens buffer + arange, 修复 bs>=并发阈值 VMFault)
- start_eagle3.sh: speculative-attention-mode 改回 prefill
- 新增 tests/ (graph-safety/精度/越界/性能/graph复现/HumanEval)
- 新增 memory/ (Claude 工作记忆快照, 可移植)
- 废弃 Strategy A (decode kernel 无 causal, 对话 garbage)

Co-Authored-By: Claude <noreply@anthropic.com>"

# 3. 推送
git push origin main   # 或 master, 看 git branch 确定
```

### 在新机器/容器拉取

```bash
# 1. clone (或已有仓库则 pull)
git clone git@github.com:benyuereal/llm-adaptation.git
cd llm-adaptation/minimax-m3-awq-int4
# 或: git pull origin main

# 2. 恢复 Claude 记忆 (让新会话继承上下文)
mkdir -p /root/.claude/projects/-workspace/memory/
cp memory/*.md /root/.claude/projects/-workspace/memory/

# 3. 安装 EAGLE3 patch
bash sglang_patches/eagle3/install.sh

# 4. 跑离线测试 (前 5 个全过再起服务)
cd sglang_patches/eagle3/tests
bash README.md  # 看 tests/README.md 的运行顺序
python test_verify_graph_buffer.py        # 最先: graph-safety 修复验证
python test_verify_prefill_precision.py   # 精度
python test_verify_oob_collision.py       # 越界碰撞
python test_verify_perf.py                # 性能
python test_verify_graph_capture_replay.py # kernel 级 graph 复现

# 5. 起服务 + 端到端 HumanEval
bash sglang_patches/eagle3/start_eagle3.sh  # 等 "server is fired up"
python verify_humaneval_eagle3.py
```

---

## 方式 B:rsync/scp 整目录复制(无 git 时)

```bash
# 在本机: 打包 (排除 __pycache__)
cd /workspace/llm-adaptation
tar --exclude='__pycache__' --exclude='.git' -czf minimax-m3-awq-int4.tar.gz minimax-m3-awq-int4/

# 传到新机器
scp minimax-m3-awq-int4.tar.gz <新机器>:/workspace/

# 在新机器: 解压 + 恢复记忆 + 装 patch + 跑测试
cd /workspace && tar xzf minimax-m3-awq-int4.tar.gz
mkdir -p /root/.claude/projects/-workspace/memory/
cp /workspace/llm-adaptation/minimax-m3-awq-int4/memory/*.md /root/.claude/projects/-workspace/memory/
bash /workspace/llm-adaptation/minimax-m3-awq-int4/sglang_patches/eagle3/install.sh
```

---

## 方式 C:复制 Claude Code 工作空间(保留会话上下文)

Claude Code 的会话历史在 `/root/.claude/projects/-workspace/`,含:
- `memory/` — 持久记忆(已在仓库 memory/ 里)
- `<session-id>.jsonl` — 历史会话 transcript(可选,体积大)

```bash
# 在本机打包 Claude 工作空间
tar -czf claude-workspace.tar.gz -C /root/.claude/projects -workspace

# 传到新机器
scp claude-workspace.tar.gz <新机器>:/tmp/

# 在新机器恢复 (会保留记忆 + 历史 transcript)
mkdir -p /root/.claude/projects
tar xzf /tmp/claude-workspace.tar.gz -C /root/.claude/projects
```

恢复后新开 Claude Code 会话能召回记忆 + 历史(但历史 transcript 体积可能几百 MB,
仅当需要完整延续会话时才传;一般只需 `memory/` 即可,见方式 A/B)。

---

## 前置条件(新机器必须满足)

1. **sglang 环境**:已装 sglang(dev,DCU dtk2604 build),路径
   `/usr/local/lib/python3.10/dist-packages/sglang/srt`
2. **模型权重**:
   - 主模型 `/models/MiniMax-M3-AWQ-INT4`(W4A16 moe-only)
   - 草稿 `/models/Inferact/MiniMax-M3-EAGLE3`(BF16)
   (路径可在 `start_eagle3.sh` 里改)
3. **量化 patch**:先应用上级 `sglang_patches/modified/` 的量化 patch
   (`bash apply_patch.sh`),让模型能正确加载,再装 EAGLE3 patch
4. **sitecustomize.py**:注册 `minimax_m3_sparse` layer type(见上级目录)
5. **GPU**:DCU gfx936/gfx928,8 卡 TP(或按需调整 `--tp`)
6. **Python 3.10 + torch + triton**(DCU 版)

## 验证转移成功

```bash
# 装完 patch 后, 验证 import
python3 -c "
from sglang.srt.layers.attention.minimax_sparse_ops.verify import minimax_sparse_verify_prefill
from sglang.srt.layers.attention.minimax_sparse_backend import MiniMaxSparseAttnBackend
print('patch 安装成功, import OK')
"

# 跑 graph-safety 测试 (最快, 验证修复在)
cd /workspace/llm-adaptation/minimax-m3-awq-int4/sglang_patches/eagle3/tests
python test_verify_graph_buffer.py
# 预期: PASS — 旧实现 graph-unsafe, 新实现 graph-safe
```

## 当前状态(2026-08-04)

- **离线测试**:graph-safety / 精度 / 越界 / 性能 / graph复现 **全 PASS**
- **端到端**:单请求对话 OK;并发 16 HumanEval 触发 VMFault(根因 = forward_extend 临时
  extend_seq_lens,已修复但**未在新代码下重跑端到端**——需在新机器验证)
- **待办**:在新机器 `bash install.sh` → 跑离线测试 → 起服务 → 并发 16 HumanEval 确认不崩
