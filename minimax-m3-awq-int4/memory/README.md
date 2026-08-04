# Claude 工作记忆 (可移植)

本目录是 Claude Code 的持久化记忆快照,记录了 MiniMax-M3-AWQ-INT4 + EAGLE3 适配过程中的
所有关键发现、根因、修复方案。跨容器/跨机器工作时,把这些记忆恢复到 Claude 的工作目录,
新会话即可继承上下文继续工作。

## 文件说明

- `MEMORY.md` — 索引(每条记忆一行 hook),Claude 启动时自动加载
- `m3-*.md` / `humaneval-*.md` / `llm-adaptation-*.md` — 各条记忆正文

## 在新容器恢复记忆

Claude Code 的记忆目录是 `/root/.claude/projects/<工作路径转义>/memory/`。
本仓库工作路径是 `/workspace`,转义后是 `-workspace`。恢复步骤:

```bash
# 1. 确认新容器的工作目录仍是 /workspace/llm-adaptation/...
# 2. 创建 Claude 记忆目录(若不存在)
mkdir -p /root/.claude/projects/-workspace/memory/

# 3. 复制本目录所有 .md 进去
cp /workspace/llm-adaptation/minimax-m3-awq-int4/memory/*.md \
   /root/.claude/projects/-workspace/memory/

# 4. 验证索引
head /root/.claude/projects/-workspace/memory/MEMORY.md
```

恢复后,新开的 Claude Code 会话会在 system-reminder 里看到 MEMORY.md 索引,
按需召回具体记忆文件。

## 当前最关键的几条(按重要性)

1. **`m3-verify-graph-temp-tensor-rootcause.md`** — EAGLE3 verify bs>=并发阈值 VMFault 的
   真正根因(forward_extend 用临时 extend_seq_lens 算 cu_seqlens,capture 后 GC,replay 读失效地址)
   及 graph-safety 修复(改用 seq_lens buffer + arange)。**这是本次端到端崩溃的根因,必读。**

2. **`m3-sglang-verify-vmfault-rootcause.md`** — 早期的 score buffer 动态尺寸 VMFault 根因
   (Strategy B 固定上界 max_seqblock_k_upper 解决)。

3. **`llm-adaptation-push-workflow.md`** — 修复代码进仓库后 commit+push 到 GitHub 的流程。

4. **`humaneval-det-eval-result.md`** — 确定配置 71.34% vs v3 79.27% 的基线对照。

## 注意

记忆反映的是**写入时**的事实。若记忆里提到某个文件/函数/标志,在新环境使用前应先
verify 它仍然存在(代码可能已变)。记忆是背景上下文,不是当前指令。
