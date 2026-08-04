#!/usr/bin/env python3
"""patch config.json: 把新机器的 M3 config 改到和本机一致.

本机 (/models/MiniMax-M3-AWQ-INT4/config.json) 和部分新机器
(如 cyankiwi 版 /models/cyankiwi/MiniMax-M3-AWQ-INT4/config.json) 的 config 有两处差异,
会导致 sglang 启动失败或 sparse backend 配置缺失:

1. vision_config 缺顶层 rope_theta (只有嵌套 rope_parameters)
   → CLIPVisionConfig.__init__() missing 'rope_theta'
   修复: 从 rope_parameters.rope_theta 补到顶层

2. 缺 sparse_attention_config 聚合字段 (顶层 + text_config 都缺)
   sglang get_minimax_sparse_attention_config 只认 sparse_attention_config 字段,
   缺它则 sparse backend 拿不到 sparse_num_index_heads/sparse_block_size 等.
   新机器把这些散在 text_config 里 (index_n_heads/index_topk_blocks 等).
   修复: 从散落字段重建 sparse_attention_config, 补到顶层 + text_config (值相同,
   与本机一致).

patch 后与本机 config 在这两处一致. 幂等 (已有且值对则不动).

用法:
  python patch_config.py /path/to/config.json
  python patch_config.py /path/to/config.json --dry-run   # 只看不改
"""
import json
import sys


# 本机 sparse_attention_config (作为目标, 顶层 == text_config 里的)
# sparse_attention_freq: 60 个 0 (60 层全 sparse, 前 3 层 dense 由 layer_types 控制)
# 其余字段从 text_config 的 index_* 字段映射 (本机两者一致)
def build_sparse_attention_config(text_cfg: dict) -> dict:
    """从 text_config 的散落 index_* 字段重建 sparse_attention_config."""
    num_layers = text_cfg.get("num_hidden_layers", 60)
    return {
        "sparse_attention_freq": [0] * num_layers,
        "sparse_num_index_heads": text_cfg.get("index_n_heads", 4),
        "sparse_index_dim": text_cfg.get("index_head_dim", 128),
        "sparse_block_size": text_cfg.get("index_block_size", 128),
        "sparse_local_block": text_cfg.get("index_local_blocks", 1),
        "sparse_topk_blocks": text_cfg.get("index_topk_blocks", 16),
        "sparse_init_block": 0,
    }


def patch_vision_rope_theta(cfg: dict) -> list:
    """补 vision_config 顶层 rope_theta. 返回改动描述列表."""
    changes = []
    vc = cfg.get("vision_config")
    if vc is None:
        return changes
    if "rope_theta" in vc:
        return changes  # 已有
    rp = vc.get("rope_parameters") or {}
    if isinstance(rp, dict) and "rope_theta" in rp:
        vc["rope_theta"] = rp["rope_theta"]
        changes.append(f"vision_config.rope_theta = {rp['rope_theta']} (取自 rope_parameters)")
    return changes


def patch_sparse_attention_config(cfg: dict) -> list:
    """补 sparse_attention_config 到顶层 + text_config. 返回改动描述列表."""
    changes = []
    text_cfg = cfg.get("text_config")
    if text_cfg is None:
        return changes

    target = build_sparse_attention_config(text_cfg)

    # 顶层
    if cfg.get("sparse_attention_config") != target:
        cfg["sparse_attention_config"] = target
        changes.append("顶层 sparse_attention_config (重建自 text_config 的 index_* 字段)")
    # text_config 里
    if text_cfg.get("sparse_attention_config") != target:
        text_cfg["sparse_attention_config"] = target
        changes.append("text_config.sparse_attention_config (同顶层值)")
    return changes


def patch(path: str, dry_run: bool = False) -> bool:
    with open(path) as f:
        cfg = json.load(f)

    changes = []
    changes += patch_vision_rope_theta(cfg)
    changes += patch_sparse_attention_config(cfg)

    if not changes:
        print(f"[{path}] config 已和本机一致 (rope_theta + sparse_attention_config 都在), 无需改")
        return False

    print(f"[{path}] 需要补丁:")
    for c in changes:
        print(f"  + {c}")

    if dry_run:
        print("  (--dry-run, 未写入)")
        return True

    with open(path, "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"  已写入 {path}")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python patch_config.py /path/to/config.json [--dry-run]")
        sys.exit(1)
    path = sys.argv[1]
    dry = "--dry-run" in sys.argv
    patch(path, dry_run=dry)
