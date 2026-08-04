---
name: llm-adaptation-push-workflow
description: llm-adaptation仓库修复代码必须commit+push到GitHub(ssh已通)
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 2b4cf476-1e6e-4434-abc0-b3a2c3a36a95
  modified: 2026-08-02T16:54:58.207Z
---

整理修复代码进 llm-adaptation 仓库后,不仅要打 patch,还必须 **commit 并 push 到 GitHub**。

**Why:** 用户明确要求修复要提交推送,方便团队同步和版本管理。只本地改/只生成 patch 文件不够。

**How to apply:**
- 仓库路径:`/workspace/llm-adaptation`(git 仓库,分支 main)
- remote: `git@github.com:benyuereal/llm-adaptation.git`(ssh 已认证通,账号 benyuereal)
- ssh 验证:`ssh -T git@github.com` 返回 "Hi benyuereal! ... authenticated" 即可用
- 流程:改代码 → `git add` → `git commit`(用 `git -c user.name=... -c user.email=...` 若无全局配置)→ `git push origin main`
- 之前的修复如 [[m3-lightning-indexer-fix]] 都要这样提交

相关:[[m3-lightning-indexer-fix]]
