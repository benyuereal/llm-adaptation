#!/usr/bin/env python3
"""
Benchmark: tilelang paged_mqa_logits vs lightop paged_mqa_logits.

API (matches lightop gemmopt.paged_mqa_logits):
    logits = paged_mqa_logits(q, fused_kv_cache, weights, context_lens,
                              block_table, schedule_meta, max_context_len,
                              clean_logits=True)

Where:
    q:               [batch_size, next_n, heads, head_dim]  BF16
    fused_kv_cache:  [num_blocks, block_size, 1, head_dim]  BF16 (block_size=64)
    weights:         [batch_size * next_n, heads]            float32
    context_lens:    [batch_size]                            int32
    block_table:     [batch_size, max_block_len]             int32
    schedule_meta:   [num_sms+1, 2] or None                 int32
    max_context_len: int

Returns:
    logits:          [batch_size * next_n, max_context_len]  float32

Architecture:
  - Grid: (max_block_len, batch_size) — one block per logical KV block (64 rows)
  - Q loaded as [heads, D], KV loaded as [block_N, D] with Pipelining
  - T.gemm with Square policy, k_pack=1, relu×weight, reduce_sum over heads
"""

import argparse
import random
import sys
from typing import Optional, Tuple

import tilelang
from tilelang import language as T
import torch


# ============================================================================
# Helpers
# ============================================================================

def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


# ============================================================================
# tilelang kernel: paged_mqa_logits (BF16)
# ============================================================================

LDS_LIMIT = 80 * 1024

BLOCK_KV = 64


def _pick_block_config(heads: int, index_dim: int):
    """Pick block_N, num_stages for shared memory budget.

    No D-splitting needed: accumulator [block_N, heads] is small enough for registers.
    """
    D = index_dim
    H = heads

    q_smem = H * D * 2  # [heads, D] in BF16

    best = None
    for bn in (32, 64):
        for ns in (2, 1, 0):
            k_smem = (ns + 1) * bn * D * 2
            total = q_smem + k_smem
            if total <= LDS_LIMIT:
                if best is None:
                    best = (bn, ns, total)
                else:
                    bN0, ns0, tot0 = best
                    if ns > ns0:
                        best = (bn, ns, total)
                    elif ns == ns0 and bn < bN0:
                        best = (bn, ns, total)

    if best is None:
        best = (32, 0, 0)

    return best[0], best[1], best[2]


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def paged_mqa_logits_tl(
    heads: int,
    index_dim: int,
    block_N: int = 64,
    num_stages: int = 1,
    threads: int = 256,
):
    """TileLang BF16 paged MQA logits kernel.

    Grid: (max_block_len, batch_size) — one block per logical KV block (64 rows).
    Each block handles all heads for 1 query position.

    Architecture:
      1. Load Q [heads, D] and weights [1, heads] into smem
      2. Look up physical block from block_table
      3. For each block_N tile of the 64 KV rows:
         - T.clear accumulator, load KV tile into smem (pipelined)
         - T.gemm: [block_N, D] × [heads, D]^T → [block_N, heads]
         - relu×weight, reduce_sum → [block_N]
         - Write to global logits
    """
    D = index_dim

    dtype = T.bfloat16
    accum_dtype = T.float32
    index_dtype = T.int32

    # k_pack=1 for RDNA (gfx936); k_pack=2 would be for CDNA
    K_PACK = 1

    batch_size = T.dynamic("batch_size")
    num_blocks = T.dynamic("num_blocks")
    max_block_len = T.dynamic("max_block_len")
    max_context_len = T.dynamic("max_context_len")

    @T.prim_func
    def kernel(
        Q: T.Tensor([batch_size, heads, D], dtype),
        KV: T.Tensor([num_blocks, BLOCK_KV, 1, D], dtype),
        Logits: T.Tensor([batch_size, max_context_len], accum_dtype),
        Weights: T.Tensor([batch_size, heads], accum_dtype),
        BlockTable: T.Tensor([batch_size, max_block_len], index_dtype),
    ):
        with T.Kernel(max_block_len, batch_size, threads=threads) as (logical_block, b_idx):
            phys_block = BlockTable[b_idx, logical_block]
            kv_offset_global = logical_block * BLOCK_KV

            q_smem = T.alloc_shared([heads, D], dtype)
            k_smem = T.alloc_shared([block_N, D], dtype)
            s = T.alloc_fragment([block_N, heads], accum_dtype)
            logits_tile = T.alloc_fragment([block_N], accum_dtype)
            w_frag = T.alloc_fragment([heads], accum_dtype)

            T.copy(Q[b_idx, 0:heads, 0:D], q_smem)
            T.copy(Weights[b_idx, 0:heads], w_frag)

            for nbn_i in T.Pipelined(T.ceildiv(BLOCK_KV, block_N), num_stages=num_stages):
                kv_row = nbn_i * block_N
                T.copy(KV[phys_block, kv_row:kv_row + block_N, 0, 0:D], k_smem)

                T.clear(s)
                T.gemm(
                    k_smem, q_smem, s,
                    k_pack=K_PACK, transpose_B=True,
                    policy=T.GemmWarpPolicy.Square,
                )

                for bn_i, h_i in T.Parallel(block_N, heads):
                    s[bn_i, h_i] = T.max(s[bn_i, h_i], T.cast(0, accum_dtype)) * w_frag[h_i]

                T.reduce_sum(s, logits_tile, dim=1, clear=True)

                for bn_i in T.Parallel(block_N):
                    gkv = kv_offset_global + kv_row + bn_i
                    if gkv < max_context_len:
                        Logits[b_idx, gkv] = logits_tile[bn_i]

    return kernel


@tilelang.jit
def clean_paged_logits_tl(threads: int = 256, block_K: int = 4096):
    """Mask positions >= context_len with -inf for each batch row."""
    batch_size = T.dynamic("batch_size")
    max_context_len = T.dynamic("max_context_len")
    dtype = T.float
    indices_dtype = T.int32

    @T.prim_func
    def kernel(
        Logits: T.Tensor([batch_size, max_context_len], dtype),
        ContextLens: T.Tensor([batch_size], indices_dtype),
    ):
        with T.Kernel(batch_size, threads=threads) as bx:
            tx = T.thread_binding(0, threads, thread="threadIdx.x")
            ctx_len = ContextLens[bx]
            for n_i in T.Pipelined(T.ceildiv(max_context_len, block_K)):
                for k_i in T.serial(block_K // threads):
                    idx = n_i * block_K + k_i * threads + tx
                    if idx >= ctx_len:
                        Logits[bx, idx] = -T.infinity(dtype)

    return kernel


# ============================================================================
# PyTorch reference implementation (matches lightop's ref_fp8_paged_mqa_logits)
# ============================================================================

def ref_paged_mqa_logits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    max_model_len: int,
):
    """PyTorch reference: same algorithm as lightop's ref_fp8_paged_mqa_logits."""
    batch_size, next_n, heads, dim = q.shape
    block_size = kv_cache.shape[1]

    logits = torch.full(
        [batch_size * next_n, max_model_len],
        float('-inf'), device=q.device, dtype=torch.float32,
    )
    ctx_list = context_lens.tolist()

    for i in range(batch_size):
        ctx_len = ctx_list[i]
        q_offsets = torch.arange(ctx_len - next_n, ctx_len, device=q.device)
        weight_slice = weights[i * next_n:(i + 1) * next_n, :].transpose(0, 1).contiguous()

        for block_rk in range(ceil_div(ctx_len, block_size)):
            block_idx = block_table[i][block_rk].item()
            qx = q[i].to(torch.float32)
            kx = kv_cache[block_idx].to(torch.float32).squeeze(1)

            k_offsets = torch.arange(block_rk * block_size, (block_rk + 1) * block_size, device=q.device)
            mask = (k_offsets[None, :] < ctx_len) & (k_offsets[None, :] <= q_offsets[:, None])

            s = torch.where(
                mask[None, :, :],
                (qx.transpose(0, 1) @ kx.T).to(logits.dtype),
                float('-inf'),
            )
            s = torch.relu(s) * weight_slice[..., None]
            s = s.sum(dim=0)

            logits[i * next_n:(i + 1) * next_n, block_rk * block_size:(block_rk + 1) * block_size] = \
                torch.where(k_offsets[None, :] <= q_offsets[:, None], s, float('-inf'))

    return logits


# ============================================================================
# High-level interface (tilelang)
# ============================================================================

_kernel_cache = {}


def page_mqa_logits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    max_context_len: int,
    clean_logits: bool = True,
):
    """Run the tilelang paged_mqa_logits kernel (BF16). next_n=1 only."""
    batch_size, next_n, heads, index_dim = q.shape
    assert next_n == 1

    block_N, num_stages, lds_bytes = _pick_block_config(heads, index_dim)
    cache_key = (heads, index_dim, block_N, num_stages)
    if cache_key not in _kernel_cache:
        print(f"  [tilelang config] H={heads}, D={index_dim}, "
              f"block_N={block_N}, num_stages={num_stages}, "
              f"LDS={lds_bytes / 1024:.1f}KB / {LDS_LIMIT / 1024:.0f}KB")
        _kernel_cache[cache_key] = paged_mqa_logits_tl(
            heads=heads, index_dim=index_dim,
            block_N=block_N, num_stages=num_stages,
        )

    logits_kernel = _kernel_cache[cache_key]
    logits = torch.full(
        [batch_size, max_context_len],
        float("-inf"), device=q.device, dtype=torch.float32,
    )
    logits_kernel(
        q.view(batch_size, heads, index_dim),  # [batch_size, next_n, heads, D] → [batch_size, heads, D]
        kv_cache, logits, weights.view(batch_size, heads), block_table,
    )

    if clean_logits:
        clean_kernel = clean_paged_logits_tl()
        clean_kernel(logits, context_lens)

    # Match lightop output shape: [batch_size * next_n, max_context_len]
    return logits.view(batch_size * next_n, max_context_len)


# ============================================================================
# lightop interface
# ============================================================================

try:
    from lightop import op, gemmopt
    HAS_LIGHTOP = True
except ImportError:
    HAS_LIGHTOP = False


def run_lightop(
    q, kv_cache, weights, context_lens, block_table,
    max_context_len, clean_logits=True,
):
    """Run the lightop paged_mqa_logits kernel."""
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    schedule_meta = gemmopt.get_paged_mqa_logits_metadata(
        context_lens, block_kv=64, num_sms=num_sms,
    )
    return gemmopt.paged_mqa_logits(
        q, kv_cache, weights, context_lens, block_table,
        schedule_meta, max_context_len, clean_logits=clean_logits,
    )


# ============================================================================
# Correctness helpers
# ============================================================================

def compute_correlation(a: torch.Tensor, b: torch.Tensor, label: str = "tensor") -> float:
    a, b = a.data.double(), b.data.double()
    norm_sum = (a * a + b * b).sum()
    if norm_sum == 0:
        return 1.0
    return (2 * (a * b).sum() / norm_sum).item()


def calc_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return 1.0 - compute_correlation(a, b)


def bench_event(fn, warmup=10, rep=50):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(rep):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return sum(times) / len(times)


# ============================================================================
# Test data generation
# ============================================================================

def create_test_data(
    batch_size: int,
    next_n: int,
    heads: int,
    head_dim: int,
    avg_ctx_len: int,
    max_model_len: int,
    device: str = "cuda",
):
    torch.manual_seed(42)

    block_size = BLOCK_KV
    total_blocks_needed = batch_size * ceil_div(int(avg_ctx_len * 1.1), block_size)
    num_blocks = total_blocks_needed * 3

    q = torch.randn(batch_size, next_n, heads, head_dim, device=device, dtype=torch.bfloat16)
    kv_cache = torch.randn(num_blocks, block_size, 1, head_dim, device=device, dtype=torch.bfloat16)
    weights = torch.randn(batch_size * next_n, heads, device=device, dtype=torch.float32)

    context_lens = torch.randint(
        int(0.9 * avg_ctx_len), int(1.1 * avg_ctx_len), (batch_size,),
        device=device,
    ).to(torch.int32)

    max_block_len = (context_lens.max().item() + block_size - 1) // block_size
    block_table = torch.zeros(batch_size, max_block_len, device=device, dtype=torch.int32)

    block_pool = list(range(num_blocks))
    random.shuffle(block_pool)
    counter = 0
    for i in range(batch_size):
        ctx_len = context_lens[i].item()
        for j in range(ceil_div(ctx_len, block_size)):
            block_table[i, j] = block_pool[counter]
            counter += 1

    return q, kv_cache, weights, context_lens, block_table


# ============================================================================
# Validation
# ============================================================================

CORRECTNESS_CASES = [
    ("bs1_H32_D128",          1,     1,      32,    128,  256),
    ("bs1_H64_D128",          1,     1,      64,    128,  256),
    ("bs2_H32_D128",          2,     1,      32,    128,  512),
    ("bs4_H32_D128",          4,     1,      32,    128,  1024),
    ("bs1_H32_D128_long",     1,     1,      32,    128,  4096),
    ("bs1_H32_D128_tiny",     1,     1,      32,    128,  32),
    ("bs1_H32_D128_small",    1,     1,      32,    128,  64),
    ("unalign_99",            1,     1,      32,    128,  99),
    ("unalign_257",           1,     1,      32,    128,  257),
    ("unalign_500_H64",       1,     1,      64,    128,  500),
    ("bs8_H32_D128",          8,     1,      32,    128,  2048),
    ("bs32_H32_D128",         32,    1,      32,    128,  1024),
]


def run_correctness_check(device="cuda"):
    all_pass = True
    failed = []

    print("=" * 90)
    print("Correctness Validation: tilelang vs lightop")
    print("=" * 90)

    header = f"{'Case':<22} {'B':>4} {'H':>4} {'D':>4} {'avg_ctx':>8}  {'tl_vs_lo':>12}  {'mask':>8}  {'status':>6}"
    print(header)
    print("-" * 90)

    for name, batch_size, next_n, heads, head_dim, avg_ctx in CORRECTNESS_CASES:
        max_model_len = avg_ctx * 2
        q, kv_cache, weights, context_lens, block_table = create_test_data(
            batch_size, next_n, heads, head_dim, avg_ctx, max_model_len, device,
        )

        logits_tl = page_mqa_logits(
            q, kv_cache, weights, context_lens, block_table, max_model_len,
        )

        diff_tl_lo = float('nan')
        mask_match = False
        if HAS_LIGHTOP:
            logits_lo = run_lightop(
                q, kv_cache, weights, context_lens, block_table, max_model_len, clean_logits=True,
            )
            tl_neginf = (logits_tl == float('-inf'))
            lo_neginf = (logits_lo == float('-inf'))
            mask_match = torch.equal(tl_neginf, lo_neginf)

            both_neginf = tl_neginf & lo_neginf
            diff_tl_lo = calc_diff(
                logits_tl.masked_fill(both_neginf, 0),
                logits_lo.masked_fill(both_neginf, 0),
            )
        else:
            print("  lightop not available, skipping")
            continue

        status = "PASS" if (diff_tl_lo < 1e-3 and mask_match) else "FAIL"
        max_abs_err = (logits_tl.float() - logits_lo.float()).abs().max().item()
        print(f"{name:<22} {batch_size:>4} {heads:>4} {head_dim:>4} {avg_ctx:>8}  "
              f"{diff_tl_lo:>12.2e}  {str(mask_match):>8}  [{status}]  max_err={max_abs_err:.4e}")

        if status == "FAIL":
            # Debug: print first few values
            print(f"    DEBUG tl[0,:8]={logits_tl[0,:8].tolist()}")
            print(f"    DEBUG lo[0,:8]={logits_lo[0,:8].tolist()}")
            tl_fin = torch.isfinite(logits_tl)
            lo_fin = torch.isfinite(logits_lo)
            print(f"    DEBUG tl finite: {tl_fin.sum().item()}, lo finite: {lo_fin.sum().item()}")
            if tl_fin.any():
                print(f"    DEBUG tl finite range: [{logits_tl[tl_fin].min().item():.4f}, {logits_tl[tl_fin].max().item():.4f}]")
            if lo_fin.any():
                print(f"    DEBUG lo finite range: [{logits_lo[lo_fin].min().item():.4f}, {logits_lo[lo_fin].max().item():.4f}]")
            # Also check reference
            logits_ref = ref_paged_mqa_logits(
                q, kv_cache, weights, context_lens, block_table, max_model_len,
            )
            print(f"    DEBUG ref[0,:8]={logits_ref[0,:8].tolist()}")
            ref_fin = torch.isfinite(logits_ref)
            print(f"    DEBUG ref finite: {ref_fin.sum().item()}")
            if ref_fin.any():
                print(f"    DEBUG ref finite range: [{logits_ref[ref_fin].min().item():.4f}, {logits_ref[ref_fin].max().item():.4f}]")
            # Only debug first failure
            break

        if not (diff_tl_lo < 1e-3 and mask_match):
            all_pass = False
            failed.append(name)

    print("-" * 90)
    if all_pass:
        print("ALL PASSED")
    else:
        print(f"FAILURES: {', '.join(failed)}")
    return all_pass


# ============================================================================
# Benchmark
# ============================================================================

BENCHMARK_CASES = [
    ("bs1_H32_D128_4k",       1,     1,      32,    128,  4096),
    ("bs64_H32_D128_4k",      64,    1,      32,    128,  4096),
    ("bs128_H32_D128_4k",     128,   1,      32,    128,  4096),
    ("bs1_H64_D128_4k",       1,     1,      64,    128,  4096),
    ("bs64_H64_D128_4k",      64,    1,      64,    128,  4096),
    ("bs1_H32_D128_72k",      1,     1,      32,    128,  72000),
    ("bs1_H64_D128_72k",      1,     1,      64,    128,  72000),
]


def run_benchmark(name, batch_size, next_n, heads, head_dim, avg_ctx, warmup, rep, device):
    print(f"\n{'=' * 70}")
    print(f"Config: {name}")
    print(f"  B={batch_size}, next_n={next_n}, H={heads}, D={head_dim}, avg_ctx~{avg_ctx}")

    max_model_len = avg_ctx * 2
    q, kv_cache, weights, context_lens, block_table = create_test_data(
        batch_size, next_n, heads, head_dim, avg_ctx, max_model_len, device,
    )

    sum_lens = context_lens.to(torch.int64).sum().item()
    tflops = 2 * sum_lens * next_n * heads * head_dim / 1e12

    # TileLang
    tl_fn = lambda: page_mqa_logits(
        q, kv_cache, weights, context_lens, block_table, max_model_len, clean_logits=False,
    )
    tl_ms = bench_event(tl_fn, warmup=warmup, rep=rep)
    print(f"  tilelang:  {tl_ms:8.3f} ms  ({tflops / (tl_ms * 1e-3):.2f} TFLOPS)")

    # Lightop
    lo_ms = None
    if HAS_LIGHTOP:
        lo_fn = lambda: run_lightop(
            q, kv_cache, weights, context_lens, block_table, max_model_len, clean_logits=False,
        )
        lo_ms = bench_event(lo_fn, warmup=warmup, rep=rep)
        speedup = tl_ms / lo_ms
        print(f"  lightop:   {lo_ms:8.3f} ms  ({tflops / (lo_ms * 1e-3):.2f} TFLOPS)")
        print(f"  speedup:   {speedup:8.2f}x  (lightop vs tilelang)")

    return {
        "name": name, "B": batch_size, "H": heads, "D": head_dim, "avg_ctx": avg_ctx,
        "tl_ms": tl_ms, "lo_ms": lo_ms,
        "tl_tflops": tflops / (tl_ms * 1e-3),
        "lo_tflops": tflops / (lo_ms * 1e-3) if lo_ms else None,
        "speedup": tl_ms / lo_ms if lo_ms else None,
        "sum_lens": sum_lens,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark tilelang vs lightop HIP ASM paged_mqa_logits (BF16)"
    )
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--csv", type=str, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA device not available.")
        sys.exit(1)

    device = "cuda"
    prop = torch.cuda.get_device_properties(0)
    gcn = getattr(prop, 'gcnArchName', prop.name)
    print(f"Device: {prop.name} ({gcn})")
    print(f"tilelang:   available")
    print(f"lightop:    {'available' if HAS_LIGHTOP else 'NOT available'}")

    if args.validate:
        run_correctness_check(device)
        if not args.benchmark:
            return

    results = []
    for name, batch_size, next_n, heads, head_dim, avg_ctx in BENCHMARK_CASES:
        r = run_benchmark(
            name, batch_size, next_n, heads, head_dim, avg_ctx,
            args.warmup, args.rep, device,
        )
        results.append(r)

    print("\n" + "=" * 95)
    print("Summary")
    print("=" * 95)
    header = f"{'Case':<28} {'B':>5} {'H':>4} {'D':>4} {'avg_ctx':>8}  {'tilelang(ms)':>12}  {'lightop(ms)':>12}  {'speedup':>8}"
    print(header)
    print("-" * 95)
    for r in results:
        lo_str = f"{r['lo_ms']:.3f}" if r['lo_ms'] else "N/A"
        sp_str = f"{r['speedup']:.2f}x" if r['speedup'] else "N/A"
        print(f"{r['name']:<28} {r['B']:>5} {r['H']:>4} {r['D']:>4} {r['avg_ctx']:>8}  "
              f"{r['tl_ms']:>12.3f}  {lo_str:>12}  {sp_str:>8}")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Case", "B", "H", "D", "avg_ctx",
                             "tilelang_ms", "lightop_ms", "speedup",
                             "tl_tflops", "lo_tflops", "sum_lens"])
            for r in results:
                writer.writerow([
                    r['name'], r['B'], r['H'], r['D'], r['avg_ctx'],
                    f"{r['tl_ms']:.6f}",
                    f"{r['lo_ms']:.6f}" if r['lo_ms'] else "",
                    f"{r['speedup']:.4f}" if r['speedup'] else "",
                    f"{r['tl_tflops']:.4f}",
                    f"{r['lo_tflops']:.4f}" if r['lo_tflops'] else "",
                    r['sum_lens'],
                ])
        print(f"\nResults saved to: {args.csv}")


if __name__ == "__main__":
    main()

