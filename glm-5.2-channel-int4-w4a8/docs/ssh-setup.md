# SSH 免密配置说明

> ⚠️ **安全警告**:你之前在对话中粘贴了私钥内容。私钥一旦出现在对话/日志中即视为已泄露,
> **请勿继续使用该密钥**。正确做法是重新生成一对新密钥。

## 正确配置步骤(在 master 机器上操作)

### 1. 重新生成密钥对(覆盖旧的可能已泄露的密钥)

```bash
# 在 master 上,生成新的 ed25519 密钥(会覆盖旧的 id_ed25519)
ssh-keygen -t ed25519 -C "your_email@example.com" -f ~/.ssh/id_ed25519 -N ""
```

### 2. 把公钥分发到目标机器

```bash
# 方法 A: 用 ssh-copy-id(需要输入一次目标机器密码)
ssh-copy-id -i ~/.ssh/id_ed25519.pub user@target_host

# 方法 B: 手动追加
cat ~/.ssh/id_ed25519.pub | ssh user@target_host "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
```

### 3. 验证免密登录

```bash
ssh user@target_host "hostname"   # 应无需密码直接返回
```

### 4. (可选)配置 ~/.ssh/config 简化命令

```bash
cat >> ~/.ssh/config <<'EOF'
Host target
    HostName target_host
    User user
    IdentityFile ~/.ssh/id_ed25519
    StrictHostKeyChecking no
EOF
chmod 600 ~/.ssh/config
# 之后可直接 ssh target
```

## 重要安全原则

- **私钥永远不要粘贴到对话、聊天、日志、文档中**——只能存放在 `~/.ssh/` 本地
- 只分享**公钥**(`.pub` 结尾),公钥可以随便给
- 怀疑私钥泄露后立即重新生成并更新所有目标机器的 authorized_keys
- AI 助手(包括我)不应接触你的私钥内容,配置步骤自己执行即可
