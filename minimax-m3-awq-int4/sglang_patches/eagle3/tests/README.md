# EAGLE3 verify kernel 测试套件 (Strategy B + graph-safety)

本目录是 EAGLE3 verify sparse kernel 的测试。**不起 sglang 服务**,直接测 kernel 逻辑,
验证 graph-safety 修复正确。运行顺序 = 从单元到端到端,前一个过了再跑下一个。

## 前置

- 已运行 `../install.sh` 安装 patch(verify/ kernel 已到 site-packages)
- GPU 可用(DCU/CUDA),`torch.cuda.is_available()` 为 True
- triton 缓存已清(install.sh 会清)

## 测试清单与运行顺序

### 1. `test_verify_graph_buffer.py` — graph-safety 修复验证(**最先跑**)

验证本次端到端 VMFault 的根因修复:forward_extend verify 分支构建 cu_seqlens/seq_lens
不再依赖临时 `extend_seq_lens` tensor(旧实现 capture 后 GC,replay 读失效地址 → 崩)。

- 复现旧实现 bug:`extend_seq_lens` 被复用成 999 → `seq_lens=500+999=1499`(错)
- 验证新实现:`seq_lens_buffer + D` + `torch.arange` → replay `seq_lens=504`(正确)
- 含**真实 `torch.cuda.CUDAGraph` capture/replay** 验证

```bash
python test_verify_graph_buffer.py
# 预期: PASS — 旧实现 graph-unsafe (复现根因), 新实现 graph-safe (修复有效)
```

### 2. `test_verify_prefill_precision.py` — 精度测试

验证 Strategy B 固定上界改造不改变计算:verify kernel vs 原 prefill kernel 的
`topk_idx` 完全一致,`o` 的 max_abs_diff < 1e-2。覆盖 bs=1/4/16、prefix=16~2000、
disable_value True/False 共 9 个用例。

```bash
python test_verify_prefill_precision.py
# 预期: ALL PASS (9/9) — verify kernel 与原 prefill kernel topk_idx 完全一致
```

### 3. `test_verify_oob_collision.py` — 越界碰撞测试

验证 graph-safety 的越界保护:
- T1: 原 prefill kernel 在 capture/replay 尺寸错配下越界写(根因)
- T2: verify kernel 固定上界 `max_seqblock_k_upper` 不越界
- T3: Step3 OOB 双保险 `pos < pos_upper` 挡住越界 topk_idx(构造严重越界值 11599)

```bash
python test_verify_oob_collision.py
# 预期: ALL PASS (3/3) — 固定上界 + OOB 双保险有效防止越界
```

### 4. `test_verify_perf.py` — 性能测试

对比 verify kernel(固定上界 score buffer)vs 原 prefill kernel(动态)的纯计算耗时。
verify kernel score 第3维固定 = cdiv(context_len, block_size_k)=1600,prefill 动态 ~16,
但 kernel 内部循环只按真实 seq_len,计算量相同。期望开销 < 10%。

```bash
python test_verify_perf.py
# 预期: ALL PASS (5/5) — 开销 < 10% (实测 -1.7% ~ +1.2%, 噪声范围)
```

### 5. `test_verify_graph_capture_replay.py` — kernel 级 graph 复现

用 `torch.cuda.CUDAGraph` 直接捕获 verify kernel 调用:capture 用 dummy seq_lens=1,
replay 改 buffer 为真实 seq_lens=2000。验证 kernel 本身在 graph 下不崩且输出与 eager 一致。

```bash
python test_verify_graph_capture_replay.py
# 预期: PASS — graph capture/replay 不崩, eager vs graph diff=0
```

### 6. `verify_humaneval_eagle3.py` — 端到端 HumanEval(**需先起服务**)

全量 164 题 HumanEval,并发 16,打 sglang EAGLE3 服务(端口 8082)。验证端到端不崩 +
正确率 + 吞吐。**这是最终验证,前 5 个测试全过后,起服务再跑。**

```bash
# 先起服务: bash ../start_eagle3.sh (等 "server is fired up")
python verify_humaneval_eagle3.py
# 预期: 不崩 + 通过率 ~70%+ + 吞吐 16-22 tok/s
```

## 已废弃

- `test_verify_decode_causal_strategyA_DEPRECATED.py` — Strategy A(verify→decode kernel)
  的 causal 单测。Strategy A 已废弃(decode kernel 无 causal mask,对话输出 garbage),
  保留备查。**不要再用 Strategy A。**

## 老测试

- `test_m3_eagle3_verify_sparse.py` — verify 字段补全逻辑单测(不起服务),仍有效。

## 一键跑前 5 个离线测试

```bash
cd /workspace/llm-adaptation/minimax-m3-awq-int4/sglang_patches/eagle3/tests
for t in test_verify_graph_buffer test_verify_prefill_precision test_verify_oob_collision test_verify_perf test_verify_graph_capture_replay; do
  echo "===== $t ====="
  python $t.py 2>&1 | grep -vE "UserWarning|setattr|_float_to_str|smallest_subnormal|NUMA|aiter" | tail -8
done
```

全部 PASS 后,再起服务跑 `verify_humaneval_eagle3.py` 端到端验证。
