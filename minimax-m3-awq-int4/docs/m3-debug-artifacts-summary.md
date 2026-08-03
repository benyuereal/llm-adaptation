# M3 调试产物归档总结

本文档归档 M3 (MiniMax-M3-AWQ-INT4) 适配调试期间产生的临时日志、诊断输出、散落脚本的**有价值内容**,
是清理前的提炼留底。生成时间: 2026-08-03。

---

## 一、诊断输出文件清单

这些是 M3 INT4 算子/inject 路径逐层埋点产生的数值 dump,体量大、高度重复,但记录了关键诊断结论。
清理后只保留本节提炼的结论,原始 txt 已删除。

### 1. `/workspace/combine_diag.txt` (1800 行, 68KB) — 已清理
- **用途**: 在 MoE `combine` 步骤前后打印 `cache3`(w2 输出)、`out_hs`、`hidden_states` 的 mean 与指针,
  验证 inplace combine 的内存别名是否正确。
- **关键发现**: 每条 `BEFORE` 显示 `out_hs.ptr == hidden_states.ptr` 且 `inplace=True, _use_intermediate=True`;
  `AFTER` 显示 `out_hs.mean == hidden_states.mean` 且指针不变。**结论: inplace combine 内存别名行为正确,
  hidden_states 被原地更新,未发生意外的 buffer 复用或写穿。** `out_hs` 在 combine 前固定 mean≈0.001014
  (silu+量化前的占位值),combine 后变为 cache3 的加权和。
- **代表性片段**:
  ```
  BEFORE combine: cache3 mean=-0.000738  out_hs mean=0.001014 ptr=140033967279616  hidden_states ptr=同
  AFTER  combine: out_hs mean=-0.002953 ptr=140033967279616  hidden_states mean=-0.002953 ptr=同
  ```

### 2. `/workspace/kernel_seq_diag.txt` (675 行, 46KB) — 已清理
- **用途**: 在 MoE MLP 内部逐 token 打印 `after_silu`(w1 后 silu)和 `after_w2_kernel`(w2 kernel 输出)的
  mean/std/nonzero,验证 INT4 量化 GEMM 的中间激活是否合理、是否有全零/NaN。
- **关键发现**: `after_silu` mean≈0.04–0.08, std≈0.26–0.41, nonzero≈160–197(正常激活范围);
  `after_w2_kernel` shape=[1,4,6144] mean≈±0.001, std≈0.11–0.19, nonzero≈1057–1567。
  **结论: INT4 MoE MLP kernel 的中间激活数值健康,无全零/NaN,非确定性不来自 kernel 内部数值异常。**
  (与 m3-trace-findings 记忆"prefill 确定,非确定性来自并发/cuda-graph"一致。)

### 3. `/workspace/zp_load_diag.txt` (296 行, 58KB) — 已清理
- **用途**: 加载后打印 `weight_zero_point` 参数的 `param_nz`(非零数)/`sample`/`data_ptr`/`param_id`,
  验证 zero_point 是否真正加载到显存(早期怀疑"indexer 权重未加载"/zp 全零)。
- **关键发现**: 全部行 `param_nz=9216/18432`(50% 非零,符合 INT4 对称/非对称量化预期),
  `sample` 为有符号 int32(实际是打包 int4 的 int32 容器),data_ptr 各不相同(每张量独立显存)。
  **结论: zero_point 权重已正确加载且非全零,"流畅但不聪明"的根因不是 zp 未加载。**

### 4. `/workspace/zp_preconv_diag.txt` (1320 行, 244KB) — 已清理
- **用途**: pre-conv 前对每层所有专家的 zero_point 做汇总检查:`layer_ord`/`all_zero`/`nz_experts`/`shape`/`sample_e0`。
- **关键发现**: 全部行 `all_zero=False nz_experts=128/128 shape=[128,192,96]`。
  **结论: 128 个专家的 zero_point 全部非空,排除了"部分专家 zp 缺失导致稀疏路由退化"的假设。**
  这是推翻 m3-lightning-indexer-fix 记忆里"indexer 权重未加载"早期怀疑的直接证据——
  真正根因是 layer_types 配置导致 indexer 整体没被构造,而非 zp 单点缺失。

### 5. `/workspace/zp_all_layers.pt` (1.8KB) — 已清理
- **用途**: 上述 zp 诊断的 torch.save 中间产物(各层 zp 汇总张量)。无独立结论价值,已被 txt dump 覆盖。

---

## 二、验证脚本片段归档

按用途分组。所有脚本打 sglang OpenAI 兼容接口 (`http://127.0.0.1:8080/v1/chat/completions`),
模型名 `MiniMax-M3-AWQ-INT4` (或路径 `/models/MiniMax-M3-AWQ-INT4`)。清理后保留两个核心验证脚本
(test_topk_sigmoid_compare.py、test_moe_layer_compare.py) 和正在用的 verify_humaneval_full_rsf.py,
其余脚本已删除,关键逻辑见下。

### A. 权重/算子诊断脚本(高复用价值)

#### `/workspace/precision_diag/moe_precision_diag.py`
MoE INT4 量化精度诊断。只加载单层单专家权重,显存极小。
- **诊断思路**:
  1. `find_layer_shard(layer)` 扫所有 safetensors 找含目标层的分片;
  2. `load_expert_weights` 取 w1/w2/w3 的 `weight_packed/scale/zero_point/shape`;
  3. `unpack_int4` 把 int32 打包(每 int32 存 8 个 int4,小端 nibble)解包,按 group_size=32 对齐 zp/scale,
     反量化 `(unpacked - zp) * scale`;
  4. 对比 f32 vs bf16 反量化相对误差,定位量化信息损失。
- **关键代码(int4 解包)**:
  ```python
  packed_flat = packed.view(torch.int32).flatten()
  nibbles = []
  for i in range(8):
      nibbles.append((packed_flat >> (i*4)) & 0xF)
  unpacked = torch.stack(nibbles, dim=-1).view(out_features, in_packed * 8)
  # zp 同样解包; group_size=32 对齐:
  gs = unpacked.shape[1] // zero_point.shape[1]
  zp = zero_point.repeat_interleave(gs, dim=1)
  dequant = (unpacked.float() - zp_unpacked.float()) * sc.float()
  ```

#### `/workspace/verify/check_index_weights.py`
快速检查任意 M3 checkpoint 的 lightning indexer 权重是否齐全,**只读 `*.safetensors.index.json`,几秒出结果**。
- 检查项: `index_q_proj / index_k_proj / index_v_proj / index_o_proj / index_q_norm / index_k_norm` 是否存在;
- 读 `config.json` 的 `sparse_attention_config.sparse_disable_index_value`,若为 None 且 v/o proj MISSING → Bug B 确认。
- **用法**: `python3 check_index_weights.py /path/to/MiniMax-M3-xxx`

#### `/workspace/verify/check_idx_v_weight.py`
直接检查加载后 `index_v_proj / index_o_proj` 权重张量是否全零/垃圾,判断 Bug B 是否污染主输出。
- 扫 checkpoint 确认无 `index_v_proj/index_o_proj`(含量化变体);
- 对比 `index_q_proj` 的量化结构看正常权重长啥样;
- 用 `inspect.getsource(CompressedTensorsW4A16Scheme.create_weights)` 看 `weight_packed` 初始值
  (空 `torch.empty`/未加载 → 保持 create_weights 时的随机初始值)。

#### `/workspace/verify/probe_long_req.py`
长请求探针:`base*120`(~7200 tokens)构造长 prompt,验证 sglang 长上下文端到端可用,
打印 `usage`/首字/gen_chars。纯 urllib,无外部依赖。

### B. HumanEval 评测脚本族(核心模板)

**通用结构**(所有 verify_*.py 共享):
- `extract_code`: 先 `re.sub(r"<mm:think>.*?</mm:think>","",text)` 剔除思考块,再取**最后一个** ```python 代码块;
- `run_test`: `exec(code + "\n\n" + test_code + f"\ncheck({entry_point})\n", {})`,捕获异常判分;
- `call_model`: `extra_body={"repetition_penalty":1.05,"chat_template_kwargs":{"thinking_mode":"..."}}`;
- 增量落盘: 每完成一题 `fout.write(...); fout.flush()` 防中途丢失;
- 并发: `ThreadPoolExecutor(max_workers=8)`, `as_completed` 实时打印进度。

#### `/workspace/verify/verify_humaneval_full.py` — **全量评测主模板**(已清理,逻辑已被 rsf 版继承)
164 题全量, `max_tokens=16384`, `thinking_mode=adaptive`, `temp=0`, 并发 8, 增量落盘到 jsonl。
**这是最完整的可复用评测模板**,关键字段: `task_id/correct/error/out_len/duration/code`。

#### `/workspace/verify/verify_humaneval_full_t02.py` — 已清理
verify_humaneval_full.py 的 temp=0.2 复跑版;改进: client timeout 1200s + 单次重试
(`call_model_retry`: 首次 `__API_ERROR__` 则 sleep 5s 重试一次),避免单题 API 超时耗 45 分钟。

#### `/workspace/verify/verify_humaneval_wrong46.py` / `verify_humaneval_wrong46_rsf.py` — 已清理(逻辑已继承到受保护的 rsf 版)
跑 v3 之前错误的 46 道题 (WRONG_IDS 列表见原文件), `max_tokens=32768`, `thinking_mode=adaptive`。
rsf 版输出到 `/workspace/outputs/humaneval_wrong46_rsf_fix.jsonl`(**受保护保留**)。
**关键 prompt 模板(v3 4 步结构)**:
```python
PROMPT_TEMPLATE = (
    "Read the following function signature and docstring, and fully implement the function described.\n\n"
    "Follow this process strictly:\n"
    "1. RESTATE: 复述题意(注意 'overlapping'/'monotonic'/'longest suffix palindrome' 等措辞);\n"
    "2. EDGE CASES: 列边界(空输入/单元素/0/负数/极值);\n"
    "3. IMPLEMENT: 写函数, 逻辑匹配复述并覆盖所有边界;\n"
    "4. SELF-TEST: 逐例手算 trace, 失败则修。\n\n"
    "{question}\n\nOutput ONLY the final implemented function inside a single ```python code block."
)
```

#### `/workspace/verify/eval_remaining11.py` — 已清理
续跑剩余 11 道错题 `[36,83,116,134,138,130,145,147,129,163,132]`, append 到 fix_b.jsonl。
复用上述模板 + `extra_body={"thinking_mode":"adaptive"}`, `max_tokens=32768`, `top_p=0.95`。
(eval_remaining.log 记录: 11 题仅 /134 通过, 其余失败, 单题耗时 100s–2186s, 长生成是主要失败原因。)

#### 单题/小批验证脚本(已清理,均用同一 run_one/exec 框架,差异仅在 TODO 列表与 prompt 变体)
| 脚本 | 验证题 | thinking_mode | 用途 |
|------|--------|---------------|------|
| verify_3problems.py | /2 /17 /19 | enabled | 翻盘验证(早期 prompt) |
| verify_3problems_v2.py | /5 /24 /26 | enabled | 加强忠实 trace |
| verify_2_19_25_26.py | /2 /19 /25 /26 | enabled | 4 题批量 |
| verify_26.py | /26 | enabled | remove_duplicates 单题 |
| verify_57_59_61_62.py | /57 /59 /61 /62 | enabled | 优化版 prompt(回退 h 过度+加 i 状态重置) |
| verify_57_59_v2.py | /57 /59 | enabled | 加 (j)(k) 后能否救回 |
| verify_failed_all.py | 40 题 | enabled | 定稿 prompt 批量验证,**带断点续跑** |
| verify_v3_fails.py | 19 题(各失败类型) | adaptive | v3 优化版 prompt 验证,BE CONCISE + 陷阱清单 |

**verify_failed_all.py 的断点续跑模式**(可复用):
```python
done = {}
if OUTFILE.exists():
    for r in json.load(open(OUTFILE)):
        done[r['task']] = r
todo = [t for t in TODO if t not in done]
# ... as_completed 循环里每完成一题:
all_results.sort(key=lambda x: int(x['task'].split('/')[1]))
OUTFILE.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))  # 整体覆写, 支持续跑
```

**verify_failed_all.py 的强化 SELF-TEST prompt**(陷阱清单 a–k,可复用于任何代码生成任务):
> 4. SELF-TEST — FAITHFUL EXECUTION: 对每个 docstring 例子,按"哑解释器"逐行执行你**实际写的**代码
> (不是你 intended 的逻辑),写出每条语句后关键变量的真实累加值。常见陷阱: (a) 循环内重复 append;
> (b) 语义混淆(deduplicate 保留首次 vs 按出现次数过滤); (c) 返回错极值(min vs max);
> (d) 返回类型不匹配(join ints); (e) tokenization 不匹配(split vs 遍历字符);
> (f) 迭代中修改列表; (g) name 未定义/import 在代码块外; (h) `and` 该用嵌套循环(因式分解: `while n%i==0: n//=i`);
> (i) 跨迭代状态被重置(direction flag 应只设一次); (j) 边界包含性 `<=`/`>=`;
> (k) 迭代归约每步必须消费元素(`n //= i`)。

### C. evalscope 评测启动脚本(已清理,配置可复用)

均用 `from evalscope.config import TaskConfig; from evalscope.run import run_task`,关键共用配置:
```python
generation_config={'temperature':0.0,'top_p':0.95,'repetition_penalty':1.05,
    'extra_body':{'chat_template_kwargs':{'thinking_mode':'adaptive'}}, 'max_tokens':...},
dataset_args={'humaneval':{'few_shot_num':0,'review_timeout':10,
    'filters':{'remove_until':{'delimiter':'</mm:think>'}},  # 剔除 think 块
    'prompt_template':PROMPT_TEMPLATE}},
```

| 脚本 | 数据集 | max_tokens | batch | work_dir | 说明 |
|------|--------|-----------|-------|----------|------|
| run_humaneval_opt_v3.py | humaneval | 16384 | 16 | humaneval_think_opt_v3 | v3 增强prompt(复述+边界+自测) |
| run_humaneval_opt_fast.py | humaneval | 16384 | 16 | humaneval_opt_fast | 生产配置+adaptive+简洁prompt |
| humaneval_strong_think.py | humaneval | 65536 | 16 | humaneval_adaptive_v2 | 强思考校准靶心,**最长 prompt**(BE CONCISE + 陷阱 a–m) |
| run_gpqa.py | gpqa_diamond | 16384 | 16 | gpqa_int4 | 198题4选1,验证量化精度损失(短输出,对照 HumanEval 长生成) |

### D. 监控脚本(已清理,框架可复用)

#### `_humaneval_monitor.py` / `monitor_humaneval.py` / `monitor_eval.py`
evalscope 评测实时监控,通用模式:
- `find_active_dir`: 找最新带 `predictions/`/`reviews/` 的时间戳目录;
- `count_predictions/reviews`: 流式数 jsonl 行数,review 解析 `sample_score.score.metadata.execution_result.passed` (humaneval) 或 `sample_score.score.value.acc` (gsm8k);
- `get_last_progress`: 从 `eval_log.log` grep `Evaluating[xxx]` 进度条最后一行;
- `eval_running`: `pgrep -f 'run_humaneval_opt'` 检测进程;
- 截断启发式: `pred_text.count("```") % 2 != 0` → 截断; `</mm:think>` 缺失 → think 未闭合。
`_humaneval_monitor.py` 的截断检测片段(可复用):
```python
fences = pred_text.count("```")
is_trunc = (fences % 2 != 0)
if "</mm:think>" not in pred_text:
    think_unclosed += 1
```

### E. 其他根目录脚本(已清理)

#### `/workspace/fix_tokenizer_config.py`
修复 M3 tokenizer_config.json 消除 `TokenizersBackend` 警告:
1. `tokenizer_class: "TokenizersBackend" → "PreTrainedTokenizerFast"`;
2. 把 `chat_template.jinja` 内容嵌入 `chat_template` 字段(若不存在)。

#### `/workspace/test_humaneval10.py` — 已清理(校准靶心单题脚本)
HumanEval/10 (make_palindrome) 单题强思考测试,`thinking_mode=enabled, max_tokens=32768, temp=0`。
分离 think/code 后用 5 个手写 case `['','x','xyz','xyx','jerry']` 验证,完整留存到 `humaneval10_output.json`。
对应记忆 `humaneval10-strong-thinking-calibration` 的"换容器先复现此题"靶心。

#### `/workspace/run_mmmu.py` — 已清理(可复用多模态评测模板)
MMMU 全量(30 学科)多模态评测,关键可复用片段:
- `img_to_b64`: PIL→PNG base64(RGBA 先转 RGB);
- `build_content`: 把 `<image N>` 占位符换为 `image_url` content block,支持多图;
- `extract_pred`: 优先 `ANSWER: X` 正则,退而取最后一个独立字母(MC)或最后一行(open);
- `thinking_mode=disabled`(选择题不需长思考,避免答案提取失败);
- argparse: `--subsets / --limit / --concurrency`,按学科汇总正确率。

#### `/workspace/verify/verify_mmmu_pipeline.py` — 已清理
MMMU 管线验证(Accounting 前 5 题),快速确认图像能进、能出答案、能判分,通则全量跑。

#### `/workspace/chat.py` — 已清理
GLM-5.1 通用流式聊天脚本(`MODEL_NAME=/models/GLM-5.1-Channel-INT4-w4a8`),与 M3 适配无关,
价值低。流式 SSE 解析模式可参考。

---

## 三、可清理的临时文件清单(已清理)

### 诊断输出 txt (5 个)
- combine_diag.txt, kernel_seq_diag.txt, zp_load_diag.txt, zp_preconv_diag.txt, zp_all_layers.pt

### 旧 sglang 日志 (logs/ 下,共 24 个,保留 sglang_perf16.log / sglang_perf16_clean.log / humaneval_full_rsf.log)
- sglang_eval*.log (4), sglang_fix*.log (5), sglang_idxcheck*.log (2), sglang_mem*.log (2),
  sglang_nograph*.log (2), sglang_probe.log, sglang_prod.log, sglang_rsf_fix.log, sglang_verify*.log (6)
- humaneval_*.log (11,除 humaneval_full_rsf.log)
- eval_remaining.log, gpqa_int4.log, mmmu_*.log (3)

### 根目录旧日志 (2 个)
- sglang_prod.log, humaneval_strong_think_v3.log

### 旧评测输出 (outputs/ 下,保留 humaneval_full_rsf.jsonl / humaneval_wrong46_rsf_fix.jsonl)
- humaneval_wrong46_fix_b.jsonl, humaneval_wrong46_fix32k.jsonl, humaneval_wrong46_fix32k_baseline.jsonl,
  humaneval_full_mem90.jsonl, humaneval_full_t02.jsonl, eval_fix_b_run.log
- 空时间戳目录: 20260729_143858, 20260729_144010

### 散落脚本 (22 个,保留两个核心验证脚本 + verify_humaneval_full_rsf.py)
- /workspace/ 根目录: chat.py, fix_tokenizer_config.py, _humaneval_monitor.py, monitor_eval.py,
  monitor_humaneval.py, run_gpqa.py, run_humaneval_opt_fast.py, run_humaneval_opt_v3.py, run_mmmu.py,
  test_humaneval10.py, humaneval_strong_think.py
- /workspace/verify/ 下(除核心): check_idx_v_weight.py, check_index_weights.py, eval_remaining11.py,
  probe_long_req.py, verify_2_19_25_26.py, verify_26.py, verify_3problems.py, verify_3problems_v2.py,
  verify_57_59_61_62.py, verify_57_59_v2.py, verify_failed_all.py, verify_humaneval_full.py,
  verify_humaneval_full_t02.py, verify_humaneval_wrong46.py, verify_mmmu_pipeline.py, verify_v3_fails.py
  (及对应 *_output.json / *.log)
- /workspace/precision_diag/moe_precision_diag.py

---

## 四、保留的关键文件(清理后仍存在)

- `/workspace/logs/sglang_perf16.log`, `sglang_perf16_clean.log` (sglang 运行中)
- `/workspace/logs/humaneval_full_rsf.log` (评测正在产生)
- `/workspace/verify/verify_humaneval_full_rsf.py` (用户正在跑全量评测)
- `/workspace/verify/test_topk_sigmoid_compare.py`, `test_moe_layer_compare.py` (核心验证脚本)
- `/workspace/outputs/humaneval_full_rsf.jsonl` (正在产生的评测结果)
- `/workspace/outputs/humaneval_wrong46_rsf_fix.jsonl` (rsf 修复证据)
- `/workspace/llm-adaptation/` (git 仓库), `/workspace/patch/`, `/workspace/docs/` (已归档 patch/文档)
- `/workspace/vllm/` (vllm 参考代码), `/models/` (模型权重)
