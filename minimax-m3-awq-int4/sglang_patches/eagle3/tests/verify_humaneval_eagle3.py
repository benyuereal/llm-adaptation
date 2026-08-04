#!/usr/bin/env python3
"""全量 HumanEval (164 题) 评估 — EAGLE3 投机解码。
配置: v3 PROMPT_TEMPLATE、thinking_mode=adaptive、max_tokens=32768、
      temperature=0.0、top_p=0.95、repetition_penalty=1.05、并发 16。
直接打 sglang EAGLE3 服务 (端口 8082), 本地 exec 测试函数判分。
输出: /workspace/outputs/humaneval_eagle3.jsonl
日志: /workspace/logs/humaneval_eagle3.log (本脚本 stdout 也写到这里)
"""
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from modelscope import MsDataset
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8082/v1", api_key="EMPTY", timeout=900)
MODEL = "/models/MiniMax-M3-AWQ-INT4"

# 全量: 不过滤, 跑 ds 里所有题 (164 道)
WRONG_IDS = None  # None 表示全量

CONCURRENCY = 16

# v3 prompt 模板
PROMPT_TEMPLATE = (
    "Read the following function signature and docstring, and fully implement the function described.\n\n"
    "Follow this process strictly:\n"
    "1. RESTATE: In one or two sentences, restate what the function must do, what it takes as input, and what it returns. "
    "Pay close attention to subtle wording (e.g. 'overlapping', 'monotonic', 'every third element', 'longest suffix that is a palindrome').\n"
    "2. EDGE CASES: Explicitly list the edge cases your code must handle: empty input, single element, zero, "
    "negative numbers, and the largest/smallest possible values. Confirm your plan covers each.\n"
    "3. IMPLEMENT: Write the function. Make sure the logic matches your restatement and handles every edge case you listed.\n"
    "4. SELF-TEST: Mentally trace your code against EACH example in the docstring. If any example would fail, fix the code before finalizing.\n\n"
    "{question}\n\n"
    "Output ONLY the final implemented function inside a single ```python code block. "
    "Do not include your restatement, edge case analysis, or self-test in the final output—only the code block."
)


def extract_code(text):
    """剔除 <mm:think>...</mm:think> 和提取 python 代码块。"""
    text = re.sub(r"<mm:think>.*?</mm:think>", "", text, flags=re.DOTALL)
    blocks = re.findall(r"```python\s*(.*?)```", text, flags=re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    blocks = re.findall(r"```\s*(.*?)```", text, flags=re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    return text.strip()


def run_test(prompt, test_code, entry_point, code):
    """exec 运行: 定义候选函数 + check 函数, 跑 assert。"""
    full = code + "\n\n" + test_code + f"\ncheck({entry_point})\n"
    try:
        exec(full, {})
        return True, None
    except Exception as e:
        return False, str(e)[:200]


def call_model(prompt_text):
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt_text}],
            max_tokens=32768,
            temperature=0.0,
            top_p=0.95,
            extra_body={
                "repetition_penalty": 1.05,
                "chat_template_kwargs": {"thinking_mode": "adaptive"},
            },
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        return f"__API_ERROR__: {e}"


def main():
    print("加载 humaneval 数据集...", flush=True)
    ds = MsDataset.load("opencompass/humaneval", subset_name="openai_humaneval", split="test")
    by_id = {}
    for i in range(len(ds)):
        s = ds[i]
        tid = int(s["task_id"].split("/")[1])
        by_id[tid] = s

    if WRONG_IDS is None:
        tasks = [(tid, by_id[tid]) for tid in sorted(by_id)]
    else:
        tasks = [(tid, by_id[tid]) for tid in WRONG_IDS if tid in by_id]
    total = len(tasks)
    print(f"共 {total} 道题, 并发 {CONCURRENCY}, 开始测试...\n", flush=True)

    results = {}
    done = 0
    correct = 0
    t0 = time.time()
    # 记录每题耗时与 outlen 用于统计
    durations = []
    outlens = []

    def task(item):
        tid, s = item
        ts = time.time()
        prompt_text = PROMPT_TEMPLATE.format(question=s["prompt"])
        out = call_model(prompt_text)
        dt = time.time() - ts
        code = extract_code(out)
        ok, err = run_test(s["prompt"], s["test"], s["entry_point"], code)
        return tid, ok, err, len(out), code, dt

    out_path = "/workspace/outputs/humaneval_eagle3.jsonl"
    # 增量写入: 每完成一题即落盘, 防止中途丢失
    with open(out_path, "w") as fout:
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
            futures = {ex.submit(task, item): item for item in tasks}
            for fut in as_completed(futures):
                tid, ok, err, outlen, code, dt = fut.result()
                done += 1
                if ok:
                    correct += 1
                results[tid] = (ok, err, outlen, dt)
                durations.append(dt)
                outlens.append(outlen)
                mark = "OK" if ok else "FAIL"
                print(f"[{done}/{total}] /{tid} {mark} | 累计 {correct}/{done}={correct/done*100:.1f}% | outlen={outlen} | {dt:.1f}s"
                      + (f" | err={err}" if not ok and err else ""), flush=True)
                fout.write(json.dumps({
                    "task_id": tid,
                    "correct": ok,
                    "error": err,
                    "out_len": outlen,
                    "duration": round(dt, 2),
                    "code": code,
                }) + "\n")
                fout.flush()

    elapsed = time.time() - t0
    print(f"\n===== 全量 HumanEval (EAGLE3 投机解码) 结果 =====", flush=True)
    print(f"通过: {correct}/{total} = {correct/total*100:.2f}%", flush=True)
    print(f"耗时: {elapsed/60:.1f} 分钟", flush=True)
    if durations:
        avg = sum(durations) / len(durations)
        print(f"平均每题耗时: {avg:.1f}s", flush=True)
    if outlens:
        outlens_sorted = sorted(outlens)
        n = len(outlens_sorted)
        med = outlens_sorted[n // 2]
        print(f"outlen: min={outlens_sorted[0]} med={med} max={outlens_sorted[-1]} mean={sum(outlens)/n:.0f}", flush=True)
    print(f"\n通过的题: {sorted([t for t,r in results.items() if r[0]])}", flush=True)
    print(f"仍错的题: {sorted([t for t,r in results.items() if not r[0]])}", flush=True)
    print(f"\n详细结果: {out_path}", flush=True)


if __name__ == "__main__":
    main()
