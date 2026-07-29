# MiniMax-M3 AWQ INT4 — K100AI DCU 适配

## 适配成果

| 指标 | 数据 |
|------|------|
| 模型 | MiniMax-M3-AWQ-INT4 (456B, 128 experts MoE) |
| 硬件 | 8× K100AI DCU (gfx928), TP=8 |
| 单用户 decode | ~19 tokens/s (CUDA Graph) |
| 精度验证 | MoE max_diff=0.024, Attention max_diff<0.003 |

## 快速使用

```bash
# 1. 应用 patch
bash apply_patch.sh

# 2. 启动服务
bash start.sh
```

## 目录结构

```
├── README.md                 # 本文件
├── apply_patch.sh            # 一键应用脚本
├── start.sh                  # 启动脚本
├── docs/
│   ├── minimax-m3-k100ai-adaptation.md  # 适配技术文档
│   └── kerminal-fae-evaluation.md       # FAE 评估报告
└── sglang_patches/
    ├── README.md             # patch 说明
    ├── minimax-m3-awq-dcu.patch  # unified diff（备用）
    ├── modified/             # 修改后的源文件 + .patch
    └── diagnostics/          # 诊断埋点备份
```

## 修改文件列表

| 文件 | 修改内容 |
|------|----------|
| compressed_tensors_wNa16.py | ZP reshape permute 修复 |
| compressed_tensors.py | symmetric 参数传递 |
| compressed_tensors_wNa16_moe.py | MoE zp overflow 修复 |
| minimax_m3.py | dense/MoE 层判断 + quant_config 控制 |
| minimax_m3_vl.py | 权重名映射 |
| fused_moe.py | HIP combine 兼容性 |
