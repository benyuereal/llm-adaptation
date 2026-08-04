#!/usr/bin/env python3
"""patch config.json: 给 vision_config 顶层补 rope_theta (从 rope_parameters 取).

问题: 部分 M3 模型 config 的 vision_config 只有嵌套 rope_parameters,
缺顶层 rope_theta. sglang 的 CLIPVisionConfig.from_dict 需要顶层 rope_theta,
否则报: TypeError: CLIPVisionConfig.__init__() missing 'rope_theta'.

修复: 若 vision_config 缺顶层 rope_theta 但有 rope_parameters.rope_theta,
则补到顶层. 幂等 (已有则不动).

用法:
  python patch_config_rope_theta.py /path/to/config.json
  python patch_config_rope_theta.py /path/to/config.json --dry-run   # 只看不改
"""
import json
import sys


def patch(path: str, dry_run: bool = False) -> bool:
    with open(path) as f:
        cfg = json.load(f)

    vc = cfg.get("vision_config")
    if vc is None:
        print(f"[{path}] 无 vision_config, 跳过")
        return False

    if "rope_theta" in vc:
        print(f"[{path}] vision_config 已有顶层 rope_theta={vc['rope_theta']}, 无需改")
        return False

    rp = vc.get("rope_parameters") or {}
    if not isinstance(rp, dict) or "rope_theta" not in rp:
        print(f"[{path}] vision_config 缺顶层 rope_theta, 且 rope_parameters 里也没有, 无法补!")
        print(f"        rope_parameters = {rp}")
        return False

    rope_theta = rp["rope_theta"]
    print(f"[{path}] 补丁: vision_config.rope_theta = {rope_theta} (取自 rope_parameters)")
    if dry_run:
        print("  (--dry-run, 未写入)")
        return True

    vc["rope_theta"] = rope_theta
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"  已写入 {path}")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python patch_config_rope_theta.py /path/to/config.json [--dry-run]")
        sys.exit(1)
    path = sys.argv[1]
    dry = "--dry-run" in sys.argv
    patch(path, dry_run=dry)
