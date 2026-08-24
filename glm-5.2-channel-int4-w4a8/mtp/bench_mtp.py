#!/usr/bin/env python3
"""MTP latency + acceptance benchmark for GLM-5.2-Channel-INT4-w4a8.

Sends single-request decode workloads, measures wall-clock tok/s, then pulls
the vLLM /metrics Prometheus endpoint to extract speculative-decoding
acceptance stats (per-position hit rate, acceptance length, draft/verify
forward counts).

Usage: python3 bench_mtp.py [num_warmup] [num_measure] [max_tokens]
"""
import sys, time, json, urllib.request, re

HOST = "http://127.0.0.1:8000"
PROMPT = "请详细介绍Transformer架构中多头自注意力机制的工作原理，包括QKV矩阵的计算、缩放点积注意力、多头拼接以及位置编码的作用。"

def gen(max_tokens, warmup=False):
    payload = json.dumps({
        "model": "/models/GLM-5.2-Channel-INT4-w4a8",
        "prompt": PROMPT,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(f"{HOST}/v1/completions", data=payload,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        data = json.loads(r.read())
    dt = time.time() - t0
    ntok = data["usage"]["completion_tokens"]
    if not warmup:
        print(f"  {ntok} tok in {dt:.2f}s -> {ntok/dt:.2f} tok/s")
    return ntok, dt

def metrics():
    try:
        with urllib.request.urlopen(f"{HOST}/metrics", timeout=30) as r:
            txt = r.read().decode()
    except Exception as e:
        print(f"  [metrics fetch failed: {e}]")
        return {}
    out = {}
    def grab(pat):
        m = re.search(pat, txt, re.M)
        return float(m.group(1)) if m else 0.0
    out["drafts"] = grab(r'^vllm:spec_decode_num_drafts_total\b.*?\s+([0-9eE+.]+)')
    out["draft_tokens"] = grab(r'^vllm:spec_decode_num_draft_tokens_total\b.*?\s+([0-9eE+.]+)')
    out["accepted"] = grab(r'^vllm:spec_decode_num_accepted_tokens_total\b.*?\s+([0-9eE+.]+)')
    # per-position acceptance (position="N")
    for m in re.finditer(r'vllm:spec_decode_num_accepted_tokens_per_pos_total\b.*?position="(\d+)"\}\s+([0-9eE+.]+)', txt):
        out[f"pos{m.group(1)}"] = float(m.group(2))
    return out

def main():
    nwarm = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    nmeas = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    maxtok = int(sys.argv[3]) if len(sys.argv) > 3 else 512

    print(f"=== warmup x{nwarm} ({maxtok} tok) ===")
    for _ in range(nwarm):
        gen(maxtok, warmup=True)

    m_before = metrics()
    print(f"=== measure x{nmeas} ({maxtok} tok) ===")
    tot_tok, tot_dt = 0.0, 0.0
    for i in range(nmeas):
        print(f"[{i+1}/{nmeas}]")
        n, d = gen(maxtok)
        tot_tok += n; tot_dt += d

    m_after = metrics()
    print(f"\n=== aggregate ===")
    print(f"total {tot_tok} tok / {tot_dt:.2f}s = {tot_tok/tot_dt:.2f} tok/s")

    # delta metrics
    def delta(k):
        return m_after.get(k, 0) - m_before.get(k, 0)
    drafts = delta("drafts")
    draft_tok = delta("draft_tokens")
    acc = delta("accepted")
    print(f"\n=== spec decode metrics (delta over measure) ===")
    print(f"drafts        = {drafts:.0f}")
    print(f"draft_tokens  = {draft_tok:.0f}  (per draft = {draft_tok/drafts:.2f})" if drafts else "")
    print(f"accepted      = {acc:.0f}")
    if drafts > 0:
        # acceptance length = accepted/drafts + 1 (bonus token always emitted)
        print(f"acceptance length (acc/drafts + 1) = {acc/drafts + 1:.3f}")
        print(f"draft token accept rate (acc/draft_tok) = {acc/draft_tok:.3f}")
    for p in ["pos0", "pos1", "pos2", "pos3"]:
        if p in m_after:
            db = m_before.get(p, 0)
            d = m_after[p] - db
            print(f"  {p}: accepted {d:.0f} / drafts {drafts:.0f} = {d/drafts:.3f}" if drafts else f"  {p}: {d:.0f}")

if __name__ == "__main__":
    main()
