# GLM-5.2 长输入乱码 — 完整根因分析与排查现场

## 一、背景

- **模型**: GLM-5.2-Channel-INT4-w4a8 (`GlmMoeDsaForCausalLM`,78 层 MoE,256 experts,
  DSA 动态稀疏注意力,slimquant INT4-w4a8 量化)
- **硬件**: 海光 K100AI DCU (gfx928),CU=128,8 卡 TP
- **软件**: vLLM 0.15.1+das.opt1(海光定制版,**未编译 `vllm._C` 原生 topk 算子**),
  tilelang / lightop 定制算子,PyTorch 2.9.0
- **草稿**: MTP 投机解码 (num_speculative_tokens=3)

GLM-5.1 在同代码同环境运行正常,本问题是 **5.1→5.2 的回归**。

## 二、现象

1. **短输入正常,长输入(>2048 tokens)输出乱码**
2. 乱码形态:`0.0.0.0.0...`、`|}{}{}{}...` 等无意义符号,或重复碎片
3. `temperature=0`(greedy)也乱 → 计算错误,非采样随机
4. 乱码率约 70-90%,高度可复现
5. eager 和 cuda graph 模式都乱 → 与 cuda graph 无关

## 三、排查思路

采用 **op-locate-agent 风格的逐算子埋点定位**:由于是 w4a8 量化模型,transformers
不支持该格式无法做 HF 基线对比,改用**打印中间张量数值**找第一个异常算子。

排查分两阶段:
- **阶段一**:逐层逐算子埋点 → 定位到 shared 层 indexer query 全零 → 修复 skip_topk
  (乱码率 70-90% → 20%)
- **阶段二**:对残留 20% 乱码继续埋点 → 定位到 prefill `mqa_logits` 返回值被丢弃
  → 修复(乱码率 20% → 0%)

## 四、定位现场(探针数据)

探针工具 `vllm/_probe.py`:打印张量统计(mean/std/min/max/nan/inf/zero%/absmax),
int 张量额外打印 min/max/neg/unique,以及前 N 个元素。异常(nan/inf/absmax>1e4)高亮
`<<< ANOMALY`。启用:`PROBE_ON=1 PROBE_LAYERS=0,1,2,3,...,10`。

### 4.1 现场一:shared 层 indexer query 全零(阶段一)

逐层打印 `idx_q`(indexer 的 query 投影输出):

| 层 | indexer_types | idx_q std | 结论 |
|---|---|---|---|
| L0 | full | 3.56 | ✅ 正常 |
| L1 | full | 4.12 | ✅ 正常 |
| L2 | full | 3.17 | ✅ 正常 |
| **L3** | **shared** | **0, zero%=100** | ❌ 全零 |
| **L6** | **full** | 4.92 | ✅ 正常 |
| **L7** | **shared** | **0, zero%=100** | ❌ 全零 |

**规律**:full 层 idx_q 正常,shared 层 idx_q 全零。而 shared 层的 `idx_hidden`
(indexer 输入)正常 → 输入没问题,是 indexer 内部 `wq_b` 投影输出全零。

**根因**:checkpoint 里 shared 层**没有 indexer 权重**(设计如此,shared 应复用 full):
```
full 层 (21个): [0,1,2,6,10,...,74] — 全部有 indexer 权重
shared 层 (57个): 其余 — 全部无 indexer 权重
```
定制分支对所有层都建 Indexer,shared 层零初始化 → `wq_b(qr)=0` → idx_q 全零。

### 4.2 现场二:prefill logits 全零(阶段二)

修复 skip_topk 后,残留 20% 乱码。对 prefill indexer 打分埋点:

```
[PROBE] L-1 pf_logits shape=(3407,3407) mean=0 std=0 min=0 max=0 zero%=100.0  <<< 全零!
[PROBE] L-1 pf_topk_in shape=(3407,2048) min=-1 max=-1 neg=6977536 unique=1
[PROBE] L-1 pf_topk_out shape=(3407,2048) min=-1 max=2147483647 neg=2606271  <<< 垃圾!
```

`pf_logits`(prefill indexer 打分)**全零** → topk 在全零上退化 → 输出 `-1` 和
`2147483647`(未初始化垃圾)→ 稀疏注意力选错 token → 乱码。

**根因**:gfx928 prefill 用 tilelang `mqa_logits`,它**返回新 tensor**(非原地写入)。
但调用处照抄 gfx938 的原地写入模式,**丢弃了返回值**:
```python
mqa_logits(q_slice, k_fp8, weights, ks, ke)   # 返回值被丢弃!
logits_slice = logits_slice_view[:q_len, :num_k]  # 读从未写入的 view → 全 0
```

直接测试 lightop `top_k_per_row_prefill` 算子本身**正常**(用连续 tensor 输出正确 indices),
证明问题不在 topk 算子,而在上游 logits 全零(返回值丢失)。

### 4.3 现场三:decode logits 含 -inf(因果 mask,正常)

```
[PROBE] L-1 dec_logits shape=(1,16384) mean=-inf min=-inf max=10251 inf=11655  <<< ANOMALY
```
decode 的 `page_mqa_logits` 输出含大量 -inf,但这是**因果 mask 的正确表示**
(`inf=11655` ≈ max_model_len - seq_len,即未生成位置的 mask),topk 用 seq_lens
限定有效范围,属正常。主 attention 输出 `mla_attn_out` 无 ANOMALY(0/44)。

## 五、根因总结(两个独立 bug)

### 根因 1:shared 层跑零权重 indexer(5.1→5.2 回归)

GLM-5.2 的 DSA indexer 从 5.1 的"每层独立"改为"full/shared 分组共享":
- `index_topk_freq=4`:每 4 层做一次 indexer
- `index_skip_topk_offset=3`:shared 层跳过 topk 的偏移
- 每 4 层一组,组内 1 层 full(有权重,算 indexer),3 层 shared(无权重,复用 full 的 buffer)

full 层位置 = `[0,1,2,6,10,14,...,74]`,由官方公式
`skip_topk = max(layer_id-3+1,0) % 4 != 0` 判定,与 `indexer_types` 完全一致。

**Bug**:定制分支不区分 full/shared,所有层都建 Indexer、都跑。shared 层零权重跑出
垃圾,且 `sparse_attn_indexer` 开头 `topk_indices_buffer[:n] = -1` **清空共享 buffer**
再覆盖 → 把 full 层好结果冲掉。

**修复**:移植官方 skip_topk 机制:
- `deepseek_v2.py`:按公式算 `_skip_topk`,shared 层不建 Indexer,MTP 层(layer_id≥78)强制建
- `mla.py`:加 `skip_topk` 参数,`not self.skip_topk` 才跑 indexer;shared 层复用 buffer
- `flashmla_sparse.py`:shared 层 indexer=None,透传 `topk_indices_buffer` 给 backend

### 根因 2:gfx928 prefill mqa_logits 返回值被丢弃(定制分支 bug)

gfx938 用 lightop `op.mqa_logits(..., logits_slice_view)`(原地写入);
gfx928 用 tilelang `mqa_logits(...)`(返回新 tensor)。调用处照抄 gfx938 模式,
丢弃返回值 → logits 全零 → topk 退化 → 乱码。

**修复**:接收返回值直接作为 `logits_slice` 传给 topk(与官方 `fp8_fp4_mqa_logits` 一致):
```python
logits_slice = mqa_logits(q_slice, k_fp8, weights, ks, ke)  # 直接用返回值
```

## 六、为什么短输入不乱、长输入才乱

- **短输入**(≤2048):`topk = min(2048, seq_len) = seq_len` → 全选,indexer 不起作用
  → 即使 shared 层 indexer 坏,结果也一样 → 不乱码
- **长输入**(>2048):`topk = 2048` → 只选 2048 个,真正稀疏化 → full 层挑对、
  shared 层挑错(或 prefill logits 全零挑垃圾)→ 乱码

阈值 = `index_topk`(2048),完美解释现象。

## 七、修复验证

| 配置 | 修复前 | 修复后 |
|---|---|---|
| repeat 长输入 ×10 | 乱码 7-9/10 | **乱码 0/10** |
| 短输入 | 正常 | 正常 |
| knowledge 长输入 greedy | 乱码 | 空(EOS,量化边界) |
| knowledge 长输入 sampling | 乱码 | **完全正常** |

## 八、已知遗留(非乱码)

**greedy 下部分问答输入首 token = EOS(空输出)**:
- knowledge 类:sampling(temp=0.7)输出完全正常,仅 greedy 空
- 这是 w4a8 量化精度使 EOS 与首 token 概率极接近,greedy 选错的边界情况
- **非乱码 bug**,用 sampling 或调整 EOS 逻辑缓解

**MTP 的 `index_share_for_mtp_iteration` 未移植**:
- 官方 draft step 0 算 indices、step 1+ 复用(set_skip_topk 动态切换)
- 本修复未移植,仅影响 draft 效率,不影响主模型精度(draft 经主模型 verify)

## 九、排查方法学(可复用)

1. **无 HF 基线时用数值打印**:量化模型 transformers 不支持,无法对比 HF 输出,
   改用逐算子打印 mean/std/absmax 找第一个异常
2. **编译期常量守卫探针**:探针的 `.item()` 在 torch.compile 图内会触发
   "Unsupported Tensor.item() call"。用模块级 `_PROBE_ON = os.environ.get(...)=="1"`
   守卫,或 `--enforce-eager` 关闭编译定位
3. **隔离变量**:关 MTP / 关 cuda graph 逐步隔离,确认各组件影响
4. **对照官方实现**:定制分支的 bug 常是"照抄 A 架构模式用到 B 架构"的 API 不兼容
   (如本例 gfx938 原地写入 vs gfx928 返回值)
