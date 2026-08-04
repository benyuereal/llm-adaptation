# M3 sglang verify VMFault 根因 + 根治(Strategy A)

## 现象
sglang EAGLE3 + MiniMax-M3-AWQ-INT4(MiniMaxSparseAttnBackend),target verify 全 cuda graph 下:
`Invalid address access ... Error code: 3`(GPU 显存越界),bs=3 崩,bs=1 不崩。并发 16 必崩。

## 根因(精确到张量/kernel)
sglang 把 sparse prefill(`forward_extend` → `minimax_sparse_prefill`)整条放进 cuda graph capture。
`flash_prefill_with_topk_index`(prefill/flash_with_topk_idx.py:441-585)在每次调用里 **新建 3 个形状由运行时值驱动的张量**:

1. `score = torch.full((num_heads, total_q, max_seqblock_k), -inf)` (line 502-507)
   - `max_seqblock_k = triton.cdiv(max_seqlen_k, block_size_k)` (line 495)
   - `max_seqlen_k` = backend `self._max_seqlen_k`(Python int,见下)
2. `o = torch.empty(total_q, num_heads, v_head_dim)` (line 499-501,indexer 的 idx_o)
3. `topk_idx = torch.full((num_heads, all_seqblock_q, topk), -1)` (line 557-562)
   - `all_seqblock_q` 已在 common/utils.py:141-147 改 graph-safe(capture 用 batch*max_seqblock 上界),**这个不是 bug**

关键:`self._max_seqlen_k` 是 Python int,由 `init_forward_metadata_capture_cuda_graph` /
`init_forward_metadata_replay_cuda_graph` 设置(minimax_sparse_backend.py:150-192):
- **capture 时**:`seq_len_fill_value = get_cuda_graph_seq_len_fill_value() = 1`(sparse backend 返回 1,
  hybrid backend 透传 sparse 的值)。verify graph 的 `buffers.seq_lens` 全填 1,所以
  `_max_seqlen_k = max(seq_lens[:bs]).item() + draft_token_num = 1 + draft_token_num` ≈ 很小。
  → capture 时 `score` 形状 `[H, total_q, cdiv(1+draft, 128)]` 极小。
- **replay 时**:`init_forward_metadata_replay_cuda_graph` 用 `seq_lens_cpu`(真实 prefix)重算,
  `_max_seqlen_k = max(prefix) + draft` = 真实长上下文(几千~几万 token)。
  → replay 期望 `score` 形状 `[H, total_q, cdiv(prefix+draft, 128)]` 远大于 capture。

但 `score` 是在 capture 时按小形状分配并 baked 进 graph 的;replay 调用同一个 captured kernel,
kernel 的 `s_ptrs = make_block_ptr(... shape=(q_len, block_num=seq_len//block_size) ...)`(line 118-125)
按 replay 时的 `seq_len`(从 `seq_lens` tensor 读,真实大值)迭代 K 块并 `tl.store(s_ptrs, score, boundary_check=(0,1))`。
`boundary_check` 只保护 block_ptr 的形状边界,**不保护底层分配大小**;底层 `score` buffer 只有小 capture 形状的内存,
store 到 `block_num` 列(大)→ 写出 buffer 末尾 → VMFault/Error code 3。

bs=1 不崩的原因:bs=1 时单请求,grid/迭代次数少,且 capture 时 `total_q = bs*draft` 也小,
block_ptr 的列边界碰巧没越过小 buffer;bs=3 时 total_q 和 block_num 都涨,越界。

> 注:`make_block_ptr + boundary_check` 确实保护"不越过 shape 声明的列",但 `shape=(q_len, block_num)`
> 里的 `block_num` 是 replay 时从 `seq_lens` 算的真实值,**不是** capture 时分配的 `max_seqblock_k`。
> 即 block_ptr 的 shape 在 replay 已被"放大",boundary_check 只挡这个放大后的 shape,不挡原始分配。
> 这是 score 张量越界的本质(用户已排除的"score 越界"是 make_block_ptr 层面,真正的越界是分配 vs replay shape 不一致)。

## vllm 怎么避免(对照)
vllm 的 M3 sparse attention 在 cuda graph 下 **不把 sparse attention 本身放进 graph**:
- `MiniMaxM3SparseAttention._run_sparse_attn` 用 `@eager_break_during_capture` 装饰
  (nvidia/model.py:646, 693-700),capture 时在这里结束当前 graph 段、eager 跑 sparse attn、再开新段。
- 只有 **indexer**(lightning indexer,score+topk 选择)是 graph-safe 的:
  commit 8f5070c44(`Make MiniMax M3 MSA indexer cudagraph-capturable`)把 indexer 的
  score plan workspace、max_score、topk_indices 全改成 builder 持有的持久 buffer(固定上界形状),
  `num_kv_splits` 用 `estimate_num_kv_splits`(只依赖 batch size,不依赖真实 KV len)。
  当前 vllm 主线更进一步:indexer 用统一 `unified_scores_buffer [T,H,MAX_K_TILES=8192]` 持久 buffer
  + `sparse_topk_select` 写进 model 级 `topk_indices_buffer`,tile 维恒为 8192(1M 上下文上界),
  -inf 填充 + num_valid_pages 因果裁剪,所以 capture/replay 形状恒定。
- 即 **vllm 的分工**:indexer(选块)graph 内 + 持久上界 buffer;sparse attn(用块算)graph 外 eager。
  两者都不依赖"capture dummy 小 / replay 真实大"的形状。

## sglang 当前缺失
1. sglang 的 sparse prefill kernel **在 graph 内** 分配 score/o/topk_idx(动态形状),无 eager break。
2. sglang 的 `_max_seqlen_k` 是 Python int,capture 小 / replay 大 → score buffer 形状不一致。
3. sglang 有 `eager_on_graph` / `break_graph`(breakable_cuda_graph.py:198-260),是 vllm
   `eager_break_during_capture` 的等价物,但默认 `enable_breakable_cuda_graph=False`,
   且现用的是 `CudaGraphRunner`(全图),不是 `BreakableCudaGraphRunner`/`PiecewiseCudaGraphRunner`。
4. `get_cu_seqblocks` 的 graph-safe 改动只救了 `all_seqblock_q`(topk_idx 形状),没救 score。

## 解法(参照 vllm,不用 Fix A)
**方案 1(最小改动,贴近 vllm 的 eager-break 思路)**:把 sparse prefill 的 attention 部分
(`flash_prefill_with_gqa_share_sparse` 或整条 `minimax_sparse_prefill`)从 verify graph 里 break 出来 eager 跑。
- 启用 `--enable-breakable-cuda-graph`(或环境 `SGLANG_USE_BREAKABLE_CUDA_GRAPH=1`)让 verify 走
  `BreakableCudaGraphRunner`,在 `MiniMaxSparseAttnBackend.forward_extend` 上加 `@eager_on_graph(True)`
  装饰(或在 `minimax_sparse_prefill` 入口插 `break_graph()`)。
- 要求:输出 `o`/`idx_o` 必须写进 caller 预分配的稳定 buffer(sglang `eager_on_graph` 用 `_copy_output`
  in-place 回填,和 vllm 一样要求输出地址稳定)。当前 `forward_extend` 里 `o`/`idx_o` 是 kernel 内
  `torch.empty` 新建 → 需改成写进 graph 外预分配的 `max_num_batched_tokens` 上界 buffer。
- 代价:sparse attn eager(每层一次),比全 graph 慢,但比 Fix A(整个 verify eager)快得多,
  因为 dense 层 + indexer + MLP 仍在 graph 内。

**方案 2(根治,贴近 vllm 当前主线)**:让 sparse prefill kernel 自己 graph-safe,不 break。
- 在 `MiniMaxSparseAttnBackend` 的 `init_cuda_graph_state` 里预分配持久 buffer:
  - `score_buffer = [H, max_num_batched_tokens, MAX_K_TILES]`(MAX_K_TILES=cdiv(max_model_len,128) 上界),
    每 forward fill -inf。
  - `o_buffer = [max_num_batched_tokens, H, v_head_dim]`、`idx_o_buffer` 同。
  - `topk_idx_buffer = [H, max_num_batched_tokens, topk]`(对应 vllm 的 topk_indices_buffer)。
- 改 `flash_prefill_with_topk_index` / `flash_prefill_with_gqa_share_sparse` 接受 `score_out`/`o_out`/`topk_out`
  入参(像 vllm `index_topk` 的 `out=`),kernel 写进这些固定形状 buffer;`max_seqlen_k` 用 max_model_len 上界
  而非运行时值;`max_seqblock_k` 固定 = MAX_K_TILES,因果裁剪靠 `seq_lens`/`prefix_lens` 在 kernel 内做。
- `_max_seqlen_q`/`_max_seqlen_k` 不再作为张量形状来源,只作为 grid 计算的 hint(且用上界)。
- 这样 capture/replay 形状恒定,verify 全 graph 不崩。
- 代价:改 kernel 签名 + backend 预分配 + 管理 buffer 生命周期;score buffer 显存
  `H * max_tokens * 8192 * 4B` 可能较大(需评估,必要时按 max_model_len 算更紧上界)。

**推荐**:先用方案 1 快速止血(启用 breakable + 装饰 forward_extend + 预分配 o buffer),
验证不崩后再做方案 2 拿回性能。

## 涉及文件(sglang,只读分析,未改)
- /usr/local/lib/python3.10/dist-packages/sglang/srt/layers/attention/minimax_sparse_backend.py
  (forward_extend:223-388;_max_seqlen_q/k:50-51,128-145,166-192;get_cuda_graph_seq_len_fill_value:194-195)
- .../minimax_sparse_ops/prefill/flash_with_topk_idx.py(score:502-507, o:499-501, topk_idx:557-562)
- .../minimax_sparse_ops/prefill/topk_sparse.py(o:286, grid 用 max_seqlen_q:293-297)
- .../minimax_sparse_ops/common/utils.py(get_cu_seqblocks graph-safe:141-147)
- /usr/local/lib/python3.10/dist-packages/sglang/srt/speculative/eagle_info_v2.py
  (prepare_for_v2_verify:259-331, seq_lens 是 prefix 不加 draft:274;can_run_cuda_graph:319-331)
- /usr/local/lib/python3.10/dist-packages/sglang/srt/model_executor/cuda_graph_runner.py
  (verify capture seq_len_fill_value=1:666-667; populate:275-389; replay:1290-1296)
- .../breakable_cuda_graph/breakable_cuda_graph.py(eager_on_graph:198-237, break_graph:336-339)
- model_runner.py:3082-3084(BreakableCudaGraphRunner 默认不开)

## vllm 对照文件
- vllm/v1/spec_decode/eagle.py(仅 stub,真逻辑在 llm_base_proposer.py)
- vllm/models/minimax_m3/nvidia/model.py(_run_attention 不 break;_run_sparse_attn @eager_break_during_capture:646,693-700)
- vllm/models/minimax_m3/nvidia/indexer_msa.py(UNIFORM_BATCH:116; unified_scores_buffer:132-138; MAX_K_TILES=8192:62)
- vllm/models/minimax_m3/common/sparse_attention.py(UNIFORM_BATCH:205)
- vllm/models/minimax_m3/common/ops/index_topk.py(out= 参数)
- vllm/compilation/breakable_cudagraph.py(eager_break_during_capture:59-117)
- commit 8f5070c44(MSA indexer cudagraph-capturable)

## 实际采用的根治:Path 1 Strategy A(2026-08-03,已验证不崩)

**没有用上面的方案 1/2**,改用更优的 Strategy A:**让 verify 走 graph-safe 的 decode kernel**,而非 crashing 的 prefill kernel。

### 关键洞察
- EAGLE3 是**首个**让 sparse **prefill** 进入 cuda graph 的场景。无 EAGLE3 时:prefill=eager(不进 graph),decode=graph-safe decode kernel。所以 prefill kernel 从来没在 graph 里跑过,它的动态 score 分配 bug 一直没暴露。
- m3 sparse **decode kernel** 本就 graph-safe:`score` 按 `req_to_token.shape[1]`(静态)分配,grid=`batch_size*NUM_KV_CHUNKS`(与 seq_len 无关),causal 靠 `seq_len` per-request 控制。见 `minimax_sparse_ops/decode/flash_with_topk_idx.py:839`、`topk_sparse.py:88,98,161,192`。
- 真正崩的 kernel 是 Step3 `_gqa_share_sparse_fwd_kernel`(HIP_LAUNCH_BLOCKING 同步日志证实),不是 Step1(异步 HSA dump 误导)。

### 实现(已 commit 到 llm-adaptation,push 到 GitHub)
- `--speculative-attention-mode decode`(sglang 既有参数,server_args.py:588)。
- `forward_extend` 入口:target_verify + mode=="decode" → 路由到新方法 `_forward_verify_via_decode`,else 走原 prefill 路径(fallback)。
- `_forward_verify_via_decode`(minimax_sparse_backend.py):verify 的 q/idx_q/out_cache_loc 已是 flat `[bs*D]` 交错布局(`assign_extend_cache_locs`,eagle_info_v2.py:527),**无需 reshape**;只需:
  1. `seq_lens_kv = prefix + D`(verify 的 `forward_batch.seq_lens` 是 prefix,scheduler 验后才加 accept_lens)
  2. `seqlens_expand_triton(seq_lens_kv, D)` → `[prefix+1..prefix+D]` causal 递增(移植自 fork NSA `seqlens_expand_kernel`,在 `minimax_sparse_ops/common/utils.py`)
  3. `req_pool_indices_expanded = unsqueeze(1).expand(bs,D).reshape(bs*D)`(view 操作,无 kernel,graph-safe;**不要用 `torch.repeat_interleave`**,它在 HIP 上 launch 的 kernel 不保证 capture 干净)
  4. 调 `minimax_sparse_decode`(与 `forward_decode` 完全相同的 kernel 路径),**decode kernel 零改动**
  5. 输出 reshape 回 `[bs*D, -1]` 匹配 forward_extend 返回形状
- mode 读取:`self.runner.server_args.speculative_attention_mode`(与 `hybrid_attn_backend.py:47` 同路径),`EAGLE3_VERIFY_DECODE=1/0` env var 作 fallback override。
- **extend_seq_lens 类型陷阱**:eager 路径它是 Python list(eagle_info_v2.py:236 `[D,D,...]`),graph 内为 None。代码必须处理 list/tensor/None 三种(不能对 list 调 `.to()`)。

### 代价
verify 注意力算力 ×4(每 draft token 独立 topk),但注意力仅占 M3 verify ~1-2%(MoE 主导),净增 ~3-4% verify 时间,EAGLE3 整体仍 2-3× 净增益。可接受。

### 验证(2026-08-03)
- 单测 `tests/test_verify_decode_causal.py` PASS:decode 展开路径 vs full-causal 参考,max_abs_diff=0.002(atol/rtol=1e-2)。
- capture 12 个 batch(bs=1..16)全程未崩;探针确认 `[V CAPTURE] verify->decode` 路径生效,`seq_lens_expanded=[2,3,4,5]` causal 递增正确。
- **并发 16 HumanEval 全量 164 题完成**:sglang 全程稳定不崩(之前并发即崩),**正确率 159/164 = 96.95%**(vs baseline 97.56%,差距 0.6% 在误差内)。错误题 [38,50,57,116,32]:38/50 是 HumanEval harness 的 `encode_cyclic`/`encode_shift` 未定义问题(与解码无关),57/116/32 是模型本身逻辑错(超长思考后错答,非 Strategy A 引入)。accept rate ~0.80,accept len ~3.3/4,gen throughput 29-35 tok/s,cuda graph 全程启用。耗时 40.6 分钟。
- 代码已 commit + push 到 `benyuereal/llm-adaptation` main 分支(commit "fix: EAGLE3 verify VMFault 根治 — Path 1 Strategy A")。增量 patch:`minimax_sparse_backend_strategyA.patch`、`utils_strategyA.patch`。

### 上游对比
sglang 官方(`/workspace/sglang-official/.../minimax_sparse_backend.py:162-170`)对此场景**仅 `raise NotImplementedError`**,未修复。本 patch 自行实现 Strategy A。相关:[[m3-precision-nondeterminism]]、[[humaneval-det-eval-result]]、[[llm-adaptation-push-workflow]]。
