# ruff: noqa
import torch
import tilelang
from tilelang import language as T
# from utils import assert_tensors_similar
import functools
import logging
import math
from typing import Optional, Tuple

try:
    import flash_mla
    HAS_FLASH_MLA = True
except ImportError:
    HAS_FLASH_MLA = False

logger = logging.getLogger(__name__)

cu_count = torch.cuda.get_device_properties("cuda").multi_processor_count
# def get_configs():
#     import itertools
#     block_I = [16, 32]
#     threads = [128, 256]
#     num_split = [1, 2, 4, 8, 16]
#     num_stages = [1]

#     _configs = list(itertools.product(block_I, threads, num_split, num_stages))

#     return [{
#         "block_I": c[0],
#         "threads": c[1],
#         "num_split": c[2],
#         "num_stages": c[3],
#     } for c in _configs]

# @tilelang.autotune(configs=get_configs())
config_map_cu72 = {
    1: {
        "block_I": 32,
        "threads": 256,
        "num_split": 32,
        "num_stages": 0,
        "batch_head": 1,
        "num_split_tail": 0,
    },
    2: {
        "block_I": 32,
        "threads": 256,
        "num_split": 32,
        "num_stages": 0,
        "batch_head": 2,
        "num_split_tail": 0,
    },
    3: {
        "block_I": 32,
        "threads": 256,
        "num_split": 32,
        "num_stages": 0,
        "batch_head": 3,
        "num_split_tail": 0,
    },
    4: {
        "block_I": 32,
        "threads": 256,
        "num_split": 32,
        "num_stages": 0,
        "batch_head": 4,
        "num_split_tail": 0,
    },
    8: {
        "block_I": 32,
        "threads": 256,
        "num_split": 16,
        "num_stages": 0,
        "batch_head": 8,
        "num_split_tail": 0,
    },
    16: {
        "block_I": 32,
        "threads": 256,
        "num_split": 8,
        "num_stages": 0,
        "batch_head": 16,
        "num_split_tail": 0,
    },
    32: {
        "block_I": 32,
        "threads": 256,
        "num_split": 4,
        "num_stages": 0,
        "batch_head": 32,
        "num_split_tail": 0,
    },
    64: {
        "block_I": 32,
        "threads": 256,
        "num_split": 2,
        "num_stages": 0,
        "batch_head": 64,
        "num_split_tail": 0,
    },
    128: {
        "block_I": 32,
        "threads": 256,
        "num_split": 1,
        "num_stages": 0,
        "batch_head": 128,
        "num_split_tail": 0,
    }
}


@tilelang.jit(
    # if we set output idx, it will cuase error when create output tensor in cython wrapper when cuda_graph is used
    # out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
        tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def sparse_mla_fwd(
    num_heads,
    dim,
    tail_dim,
    topk,
    num_split=1,
    num_split_tail=0,
    *,
    kv_group=1,
    sm_scale=None,
    is_causal=True,
    block_I=32,
    num_stages=1,
    threads=256,
    kv_stride=1,
    dtype="bfloat16",
):
    assert dim == tilelang.math.next_power_of_2(
        dim
    ), f"haven't check padding correctness yet, dim={dim}"
    if tail_dim > 0:
        assert tail_dim == tilelang.math.next_power_of_2(
            tail_dim
        ), f"haven't check padding correctness yet, dim={tail_dim}"
    assert is_causal == True, "non-casual is not supported"
    assert (
        topk % block_I == 0
    ), "otherwise will load some index=0 thus causing wrong kv to be loaded"
    if num_split > 1:
        assert (
            topk % (num_split * block_I) == 0
        ), f"topk={topk} must be divisible by num_split * block_I={num_split} * {block_I}"
    if num_split_tail > 0:
        assert (
            topk % (num_split_tail * block_I) == 0
        ), f"topk={topk} must be divisible by num_split_tail * block_I={num_split_tail} * {block_I}"
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5 * 1.44269504  # log2(e)
    else:
        sm_scale = sm_scale * 1.44269504  # log2(e)

    dim_plus_tail = dim + tail_dim
    head_kv = num_heads // kv_group

    # Symbolic variables
    batch = T.symbolic("batch")
    batch_head = T.symbolic("batch_head")
    batch_tail = T.symbolic("batch_tail")
    seq_len = T.symbolic("seq_len")
    seq_len_kv = T.symbolic("seq_len_kv")

    q_shape = [batch, seq_len, num_heads, dim_plus_tail]
    kv_shape = [batch, seq_len_kv, kv_group, dim_plus_tail]
    o_shape = [batch, seq_len, num_heads, dim]
    indices_shape = [batch, seq_len, kv_group, topk]

    glse_shape = [batch_head, seq_len, num_split, num_heads]
    output_partial_shape = [batch_head, seq_len, num_split, num_heads, dim]
    if num_split_tail > 0:
        glse_shape_tail = [batch_tail, seq_len, num_split_tail, num_heads]
        output_partial_shape_tail = [batch_tail, seq_len, num_split_tail, num_heads, dim]

    indices_dtype = "int32"
    accum_dtype = "float"
    intermediate_dtype = "float16"
    # intermediate_dtype = dtype

    H = head_kv
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    if padded_H != H:
        assert kv_group == 1
    BI = block_I

    # e.g. num_split = 2, we take like BI_split0 BI_split1 BI_split0 BI_split1 ... on topk
    topk_per_split = topk // num_split if num_split > 1 else topk
    NI = tilelang.cdiv(topk_per_split, BI)
    split_stride = BI * num_split

    topk_per_split_tail = topk // num_split_tail if num_split_tail > 1 else topk
    NI_tail = tilelang.cdiv(topk_per_split_tail, BI)
    split_stride_tail = BI * num_split_tail

    D = dim
    D_spilt = dim // 4
    D_tail = tail_dim

    # Optimized: simplify max_block_m calculation
    max_block_m = 32 if head_kv == 128 else 16

    if head_kv > max_block_m:
        assert head_kv % max_block_m == 0, f"head_kv should be a multiple of {max_block_m}"
        REPLICATE_H = head_kv // max_block_m
    else:
        REPLICATE_H = 1

    H_per_block = padded_H if REPLICATE_H == 1 else max_block_m

    hd_div_threads = (H_per_block * D_spilt) // threads
    bid_div_threads = (BI * D_spilt) // threads
    kv_vectorized = max(min(min(hd_div_threads, bid_div_threads), 8), 1)
    # if BI is 16, can not kpack for gemm2
    if BI < 32 and kv_vectorized == 8:
        kv_vectorized = 4

    threads_per_line = D_spilt // kv_vectorized
    warps_line_stride = threads // threads_per_line
    kv_serial_count = BI // warps_line_stride
    kpack = min((kv_vectorized + 3) // 4, 2)

    mmac_k = 16 * kpack
    warps = threads // 64
    max_warp_k = D_spilt // mmac_k
    max_warp_n = BI // 16
    max_warp_m = H_per_block // 16
    out_shared_reuse = H_per_block > 16

    gemm1_policy = T.GemmWarpPolicy.FullColK
    gemm2_policy = T.GemmWarpPolicy.FullCol

    # print(f"kv_serial_count={kv_serial_count}, warps_line_stride={warps_line_stride}, "
    #     f"threads_per_line={threads_per_line}, kv_vectorized={kv_vectorized}, kpack={kpack}, "
    #     f"block_M={H_per_block}, block_I={BI}, gemm1_policy={gemm1_policy}, gemm2_policy={gemm2_policy}")

    @T.macro
    def sparse_mla(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        KV: T.Tensor(kv_shape, dtype),  # type: ignore
        Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
        Output: T.Tensor(o_shape, dtype),  # type: ignore
    ):
        with T.Kernel(seq_len * REPLICATE_H, batch, kv_group, threads=threads) as (
                bx,
                by,
                bz,
        ):
            Q_spilt0_shared = T.alloc_fragment([H_per_block, D_spilt], dtype)
            Q_spilt1_shared = T.alloc_fragment([H_per_block, D_spilt], dtype)
            Q_spilt2_shared = T.alloc_fragment([H_per_block, D_spilt], dtype)
            Q_spilt3_shared = T.alloc_fragment([H_per_block, D_spilt], dtype)
            Q_tail_shared = T.alloc_fragment([H_per_block, D_tail], dtype)

            KV_spilt0_shared = T.alloc_shared([BI, D_spilt], dtype)
            KV_spilt1_shared = T.alloc_shared([BI, D_spilt], dtype)
            KV_spilt2_shared = T.alloc_shared([BI, D_spilt], dtype)
            # KV_spilt3_shared = T.alloc_shared([BI, D_spilt], dtype)
            KV_spilt0_local = T.alloc_fragment([BI, D_spilt], dtype)
            KV_spilt1_local = T.alloc_fragment([BI, D_spilt], dtype)
            KV_spilt2_local = T.alloc_fragment([BI, D_spilt], dtype)
            KV_spilt3_local = T.alloc_fragment([BI, D_spilt], dtype)

            gemm2_kv_split0_local = T.alloc_fragment([BI, D_spilt], dtype)
            K_tail_local = T.alloc_fragment([BI, D_tail], dtype)

            # float32 copies of KV for GEMM2 (S @ V)
            gemm2_kv_split0_f32 = T.alloc_fragment([BI, D_spilt], accum_dtype)
            gemm2_kv_split1_f32 = T.alloc_fragment([BI, D_spilt], accum_dtype)
            gemm2_kv_split2_f32 = T.alloc_fragment([BI, D_spilt], accum_dtype)
            gemm2_kv_split3_f32 = T.alloc_fragment([BI, D_spilt], accum_dtype)

            mask = T.alloc_fragment([BI], "bool")

            acc_o0 = T.alloc_fragment([H_per_block, D_spilt], accum_dtype)
            acc_o1 = T.alloc_fragment([H_per_block, D_spilt], accum_dtype)
            acc_o2 = T.alloc_fragment([H_per_block, D_spilt], accum_dtype)
            acc_o3 = T.alloc_fragment([H_per_block, D_spilt], accum_dtype)

            acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
            S_shared = T.alloc_shared([H_per_block, BI], accum_dtype)
            sumexp = T.alloc_fragment([H_per_block], accum_dtype)
            sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
            alpha = T.alloc_fragment([H_per_block], accum_dtype)
            m_i = T.alloc_fragment([H_per_block], accum_dtype)
            m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)
            indices_local = T.alloc_fragment([1], indices_dtype)
            indices_mask = T.alloc_fragment([BI], indices_dtype)
            indices_local_1 = T.alloc_fragment([1], indices_dtype)
            indices_tail = T.alloc_fragment([1], indices_dtype)
            valid_NI = T.alloc_fragment([1], "int")

            b_i, g_i = by, bz
            s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)
            # q_i = q_start_index_s[0] + s_i
            # max_kv_i = (q_i + 1 - kv_stride) // kv_stride
            # kv_i = (q_i + 1 - kv_stride) // kv_stride
            # max_kv_i = kv_i if (kv_i <= seq_len_kv - 1) else seq_len_kv - 1

            H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * max_block_m)
            H1 = H0 + H_per_block

            tx = T.get_thread_binding()
            T.copy(Q[b_i, s_i, H0:H1, :D_spilt], Q_spilt0_shared, coalesced_width=kv_vectorized)
            T.copy(Q[b_i, s_i, H0:H1, D_spilt:2 * D_spilt], Q_spilt1_shared, coalesced_width=kv_vectorized)
            T.copy(Q[b_i, s_i, H0:H1, 2 * D_spilt:3 * D_spilt], Q_spilt2_shared, coalesced_width=kv_vectorized)
            T.copy(Q[b_i, s_i, H0:H1, 3 * D_spilt:4 * D_spilt], Q_spilt3_shared, coalesced_width=kv_vectorized)
            T.copy(Q[b_i, s_i, H0:H1, D:], Q_tail_shared)

            T.fill(acc_o0, 0)
            T.fill(acc_o1, 0)
            T.fill(acc_o2, 0)
            T.fill(acc_o3, 0)
            T.fill(sumexp, 1)
            T.fill(m_i, -(2 ** 30))  # avoid -inf - inf to cause nan
            T.fill(valid_NI, 0)

            for i_i in T.serial(NI):
                first_indices = Indices[b_i, s_i, g_i, i_i * BI]
                # if first_indices <= max_kv_i and first_indices >= 0:
                if first_indices >= 0:
                    valid_NI[0] += 1

            for i_i in T.Pipelined(valid_NI[0], num_stages=num_stages):
                for bi_i in T.Parallel(BI):
                    indices_mask[bi_i] = Indices[b_i, s_i, g_i, i_i * BI + bi_i]
                    mask[bi_i] = indices_mask[bi_i] >= 0

                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.if_then_else(
                        mask[bi_i], 0, -T.infinity(acc_s.dtype)
                    )

                # for u in T.serial(kv_serial_count):
                #     line_stride = u * warps_line_stride
                #     indices_local[0] = Indices[b_i, s_i, g_i, i_i * BI + line_stride + tx // threads_per_line]
                #     # indices_local[0] = T.if_then_else(indices_local[0] <= max_kv_i and indices_local[0] >= 0, indices_local[0], 0)
                #     indices_local[0] = T.if_then_else(indices_local[0] >= 0, indices_local[0], 0)
                #     for v in T.vectorized(kv_vectorized):
                #         KV_spilt0_local[line_stride + tx // threads_per_line,
                #                         (tx % threads_per_line) * kv_vectorized + v] = KV[b_i, indices_local[0], g_i,
                #                         (tx % threads_per_line) * kv_vectorized + v]
                #         KV_spilt1_local[line_stride + tx // threads_per_line,
                #                         (tx % threads_per_line) * kv_vectorized + v] = KV[b_i, indices_local[0], g_i,
                #                         D_spilt + (tx % threads_per_line) * kv_vectorized + v]
                #         KV_spilt2_local[line_stride + tx // threads_per_line,
                #                         (tx % threads_per_line) * kv_vectorized + v] = KV[b_i, indices_local[0], g_i,
                #                         2*D_spilt + (tx % threads_per_line) * kv_vectorized + v]

                for bi_i, d_i in T.Parallel(BI, D_spilt):
                    KV_spilt0_local[bi_i, d_i] = T.if_then_else(Indices[b_i, s_i, g_i, i_i * BI + bi_i] >= 0,
                                                                KV[b_i, Indices[
                                                                    b_i, s_i, g_i, i_i * BI + bi_i], g_i, d_i], 0)
                    KV_spilt1_local[bi_i, d_i] = T.if_then_else(Indices[b_i, s_i, g_i, i_i * BI + bi_i] >= 0,
                                                                KV[b_i, Indices[
                                                                    b_i, s_i, g_i, i_i * BI + bi_i], g_i, D_spilt + d_i],
                                                                0)
                    KV_spilt2_local[bi_i, d_i] = T.if_then_else(Indices[b_i, s_i, g_i, i_i * BI + bi_i] >= 0,
                                                                KV[b_i, Indices[
                                                                    b_i, s_i, g_i, i_i * BI + bi_i], g_i, 2 * D_spilt + d_i],
                                                                0)
                for bi_i, d_i in T.Parallel(BI, D_spilt):
                    KV_spilt3_local[bi_i, d_i] = T.if_then_else(Indices[b_i, s_i, g_i, i_i * BI + bi_i] >= 0,
                                                                KV[b_i, Indices[
                                                                    b_i, s_i, g_i, i_i * BI + bi_i], g_i, 3 * D_spilt + d_i],
                                                                0)

                for bi_i, d_i in T.Parallel(BI, D_tail):
                    K_tail_local[bi_i, d_i] = T.if_then_else(Indices[b_i, s_i, g_i, i_i * BI + bi_i] >= 0,
                                                             KV[b_i, Indices[
                                                                 b_i, s_i, g_i, i_i * BI + bi_i], g_i, D + d_i], 0)

                T.copy(KV_spilt0_local, KV_spilt0_shared)
                T.copy(KV_spilt1_local, KV_spilt1_shared)
                T.copy(KV_spilt2_local, KV_spilt2_shared)

                T.gemm(Q_spilt0_shared, KV_spilt0_shared, acc_s, transpose_B=True, k_pack=kpack, policy=gemm1_policy)
                T.gemm(Q_spilt1_shared, KV_spilt1_shared, acc_s, transpose_B=True, k_pack=kpack, policy=gemm1_policy)
                T.gemm(Q_spilt2_shared, KV_spilt2_shared, acc_s, transpose_B=True, k_pack=kpack, policy=gemm1_policy)
                T.gemm(Q_spilt3_shared, KV_spilt3_local, acc_s, transpose_B=True, k_pack=kpack, policy=gemm1_policy)
                T.gemm(Q_tail_shared, K_tail_local, acc_s, transpose_B=True, policy=gemm1_policy)

                T.copy(m_i, m_i_prev)
                if gemm1_policy == T.GemmWarpPolicy.FullColK:
                    T.reduce_sum_warp(acc_s, acc_s, clear=False)

                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                for h_i in T.Parallel(H_per_block):
                    alpha[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.exp2(
                        acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale
                    )
                T.reduce_sum(acc_s, sumexp_i, dim=1)
                for h_i in T.Parallel(H_per_block):
                    sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                for h_i, d_i in T.Parallel(H_per_block, D_spilt):
                    acc_o0[h_i, d_i] = acc_o0[h_i, d_i] * alpha[h_i]
                    acc_o1[h_i, d_i] = acc_o1[h_i, d_i] * alpha[h_i]
                    acc_o2[h_i, d_i] = acc_o2[h_i, d_i] * alpha[h_i]
                    acc_o3[h_i, d_i] = acc_o3[h_i, d_i] * alpha[h_i]

                T.copy(KV_spilt0_shared, gemm2_kv_split0_local)
                T.copy(acc_s, S_shared)
                T.copy(KV_spilt3_local, KV_spilt0_shared)
                # Cast KV to float32 for GEMM2 precision
                T.copy(gemm2_kv_split0_local, gemm2_kv_split0_f32)
                T.copy(KV_spilt1_shared, gemm2_kv_split1_f32)
                T.copy(KV_spilt2_shared, gemm2_kv_split2_f32)
                T.copy(KV_spilt0_shared, gemm2_kv_split3_f32)
                T.gemm(S_shared, gemm2_kv_split0_f32, acc_o0, policy=gemm2_policy)
                T.gemm(S_shared, gemm2_kv_split1_f32, acc_o1, policy=gemm2_policy)
                T.gemm(S_shared, gemm2_kv_split2_f32, acc_o2, policy=gemm2_policy)
                T.gemm(S_shared, gemm2_kv_split3_f32, acc_o3, policy=gemm2_policy)
            # Rescale
            for h_i, d_i in T.Parallel(H_per_block, D_spilt):
                acc_o0[h_i, d_i] /= sumexp[h_i]
                acc_o1[h_i, d_i] /= sumexp[h_i]
                acc_o2[h_i, d_i] /= sumexp[h_i]
                acc_o3[h_i, d_i] /= sumexp[h_i]

            if out_shared_reuse:
                acc_oshared0 = T.alloc_shared([H_per_block, D_spilt], dtype)
                acc_oshared1 = T.alloc_shared([H_per_block, D_spilt], dtype)
                T.annotate_layout({
                    acc_oshared0: tilelang.layout.make_hcu_swizzled_layout(acc_oshared0, major_pack=2),
                    acc_oshared1: tilelang.layout.make_hcu_swizzled_layout(acc_oshared1, major_pack=2),
                })
                T.copy(acc_o0, acc_oshared0)
                T.copy(acc_o1, acc_oshared1)
                T.copy(acc_oshared0, Output[b_i, s_i, H0:H1, :D_spilt])
                T.copy(acc_oshared1, Output[b_i, s_i, H0:H1, D_spilt:2 * D_spilt])
                T.copy(acc_o2, acc_oshared0)
                T.copy(acc_o3, acc_oshared1)
                T.copy(acc_oshared0, Output[b_i, s_i, H0:H1, 2 * D_spilt:3 * D_spilt])
                T.copy(acc_oshared1, Output[b_i, s_i, H0:H1, 3 * D_spilt:4 * D_spilt])
            else:
                acc_oshared0 = T.alloc_shared([H_per_block, D_spilt], dtype)
                acc_oshared1 = T.alloc_shared([H_per_block, D_spilt], dtype)
                acc_oshared2 = T.alloc_shared([H_per_block, D_spilt], dtype)
                acc_oshared3 = T.alloc_shared([H_per_block, D_spilt], dtype)
                T.annotate_layout({
                    acc_oshared0: tilelang.layout.make_hcu_swizzled_layout(acc_oshared0, major_pack=2),
                    acc_oshared1: tilelang.layout.make_hcu_swizzled_layout(acc_oshared1, major_pack=2),
                    acc_oshared2: tilelang.layout.make_hcu_swizzled_layout(acc_oshared2, major_pack=2),
                    acc_oshared3: tilelang.layout.make_hcu_swizzled_layout(acc_oshared3, major_pack=2),
                })
                T.copy(acc_o0, acc_oshared0)
                T.copy(acc_o1, acc_oshared1)
                T.copy(acc_o2, acc_oshared2)
                T.copy(acc_o3, acc_oshared3)
                T.copy(acc_oshared0, Output[b_i, s_i, H0:H1, :D_spilt])
                T.copy(acc_oshared1, Output[b_i, s_i, H0:H1, D_spilt:2 * D_spilt])
                T.copy(acc_oshared2, Output[b_i, s_i, H0:H1, 2 * D_spilt:3 * D_spilt])
                T.copy(acc_oshared3, Output[b_i, s_i, H0:H1, 3 * D_spilt:4 * D_spilt])

    @T.macro
    def sparse_mla_split(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        KV: T.Tensor(kv_shape, dtype),  # type: ignore
        Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
        glse: T.Tensor(glse_shape, intermediate_dtype),  # type: ignore
        Output_partial: T.Tensor(output_partial_shape, intermediate_dtype),  # type: ignore
    ):
        with T.Kernel(seq_len * REPLICATE_H, batch_head * kv_group, num_split, threads=threads) as (
                bx,
                by,
                bz,
        ):
            split_idx = bz
            b_i = by if kv_group == 1 else (by // kv_group)
            g_i = 0 if kv_group == 1 else (by % kv_group)
            s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)
            # q_i = q_start_index_s[0] + s_i
            # max_kv_i = (q_i + 1 - kv_stride) // kv_stride
            # kv_i = (q_i + 1 - kv_stride) // kv_stride
            # max_kv_i = kv_i if (kv_i <= seq_len_kv - 1) else seq_len_kv - 1

            H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * max_block_m)
            H1 = H0 + H_per_block

            valid_NI = T.alloc_fragment([1], "int")
            T.fill(valid_NI, 0)
            for i_i in T.serial(NI):
                first_indices = Indices[b_i, s_i, g_i, i_i * split_stride + split_idx * BI]
                # if first_indices <= max_kv_i and first_indices >= 0:
                if first_indices >= 0:
                    valid_NI[0] += 1

            if valid_NI[0] == 0:
                acc_o = T.alloc_fragment([H_per_block, D], accum_dtype)
                sumexp = T.alloc_fragment([H_per_block], accum_dtype)
                T.fill(sumexp, -T.infinity(accum_dtype))
                T.fill(acc_o, 0)
                T.copy(sumexp, glse[b_i, s_i, split_idx, H0:H1])
                T.copy(acc_o, Output_partial[b_i, s_i, split_idx, H0:H1, :D])
            else:
                Q_spilt0_shared = T.alloc_fragment([H_per_block, D_spilt], dtype)
                Q_spilt1_shared = T.alloc_fragment([H_per_block, D_spilt], dtype)
                Q_spilt2_shared = T.alloc_fragment([H_per_block, D_spilt], dtype)
                Q_spilt3_shared = T.alloc_fragment([H_per_block, D_spilt], dtype)
                Q_tail_shared = T.alloc_fragment([H_per_block, D_tail], dtype)

                KV_spilt0_shared = T.alloc_shared([BI, D_spilt], dtype)
                KV_spilt1_shared = T.alloc_shared([BI, D_spilt], dtype)
                KV_spilt2_shared = T.alloc_shared([BI, D_spilt], dtype)
                # KV_spilt3_shared = T.alloc_shared([BI, D_spilt], dtype)
                KV_spilt0_local = T.alloc_fragment([BI, D_spilt], dtype)
                KV_spilt1_local = T.alloc_fragment([BI, D_spilt], dtype)
                KV_spilt2_local = T.alloc_fragment([BI, D_spilt], dtype)
                KV_spilt3_local = T.alloc_fragment([BI, D_spilt], dtype)

                gemm2_kv_split0_local = T.alloc_fragment([BI, D_spilt], dtype)
                K_tail_local = T.alloc_fragment([BI, D_tail], dtype)

                # float32 copies of KV for GEMM2 (S @ V)
                gemm2_kv_split0_f32 = T.alloc_fragment([BI, D_spilt], accum_dtype)
                gemm2_kv_split1_f32 = T.alloc_fragment([BI, D_spilt], accum_dtype)
                gemm2_kv_split2_f32 = T.alloc_fragment([BI, D_spilt], accum_dtype)
                gemm2_kv_split3_f32 = T.alloc_fragment([BI, D_spilt], accum_dtype)

                mask = T.alloc_fragment([BI], "bool")
                acc_o0 = T.alloc_fragment([H_per_block, D_spilt], accum_dtype)
                acc_o1 = T.alloc_fragment([H_per_block, D_spilt], accum_dtype)
                acc_o2 = T.alloc_fragment([H_per_block, D_spilt], accum_dtype)
                acc_o3 = T.alloc_fragment([H_per_block, D_spilt], accum_dtype)

                acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
                S_shared = T.alloc_shared([H_per_block, BI], accum_dtype)
                sumexp = T.alloc_fragment([H_per_block], accum_dtype)
                sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
                alpha = T.alloc_fragment([H_per_block], accum_dtype)
                m_i = T.alloc_fragment([H_per_block], accum_dtype)
                m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)
                indices_local = T.alloc_fragment([1], indices_dtype)
                indices_mask = T.alloc_fragment([BI], indices_dtype)
                indices_local_1 = T.alloc_fragment([1], indices_dtype)
                indices_tail = T.alloc_fragment([1], indices_dtype)

                tx = T.get_thread_binding()
                T.copy(Q[b_i, s_i, H0:H1, :D_spilt], Q_spilt0_shared, coalesced_width=kv_vectorized)
                T.copy(Q[b_i, s_i, H0:H1, D_spilt:2 * D_spilt], Q_spilt1_shared, coalesced_width=kv_vectorized)
                T.copy(Q[b_i, s_i, H0:H1, 2 * D_spilt:3 * D_spilt], Q_spilt2_shared, coalesced_width=kv_vectorized)
                T.copy(Q[b_i, s_i, H0:H1, 3 * D_spilt:4 * D_spilt], Q_spilt3_shared, coalesced_width=kv_vectorized)
                T.copy(Q[b_i, s_i, H0:H1, D:], Q_tail_shared)

                T.fill(acc_o0, 0)
                T.fill(acc_o1, 0)
                T.fill(acc_o2, 0)
                T.fill(acc_o3, 0)
                T.fill(sumexp, 1)
                T.fill(m_i, -(2 ** 30))  # avoid -inf - inf to cause nan

                for i_i in T.Pipelined(valid_NI[0], num_stages=num_stages):
                    idx_in_split = i_i * split_stride + split_idx * BI
                    for bi_i in T.Parallel(BI):
                        indices_mask[bi_i] = Indices[b_i, s_i, g_i, idx_in_split + bi_i]
                        # mask[bi_i] = indices_mask[bi_i] <= max_kv_i and indices_mask[bi_i] >= 0
                        mask[bi_i] = indices_mask[bi_i] >= 0

                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s[h_i, bi_i] = T.if_then_else(
                            mask[bi_i], 0, -T.infinity(acc_s.dtype)
                        )

                    for bi_i, d_i in T.Parallel(BI, D_spilt):
                        KV_spilt0_local[bi_i, d_i] = T.if_then_else(Indices[b_i, s_i, g_i, idx_in_split + bi_i] >= 0,
                                                                    KV[b_i, Indices[
                                                                        b_i, s_i, g_i, idx_in_split + bi_i], g_i, d_i],
                                                                    0)
                        KV_spilt1_local[bi_i, d_i] = T.if_then_else(Indices[b_i, s_i, g_i, idx_in_split + bi_i] >= 0,
                                                                    KV[b_i, Indices[
                                                                        b_i, s_i, g_i, idx_in_split + bi_i], g_i, D_spilt + d_i],
                                                                    0)
                        KV_spilt2_local[bi_i, d_i] = T.if_then_else(Indices[b_i, s_i, g_i, idx_in_split + bi_i] >= 0,
                                                                    KV[b_i, Indices[
                                                                        b_i, s_i, g_i, idx_in_split + bi_i], g_i, 2 * D_spilt + d_i],
                                                                    0)
                    for bi_i, d_i in T.Parallel(BI, D_spilt):
                        KV_spilt3_local[bi_i, d_i] = T.if_then_else(Indices[b_i, s_i, g_i, idx_in_split + bi_i] >= 0,
                                                                    KV[b_i, Indices[
                                                                        b_i, s_i, g_i, idx_in_split + bi_i], g_i, 3 * D_spilt + d_i],
                                                                    0)

                    for bi_i, d_i in T.Parallel(BI, D_tail):
                        K_tail_local[bi_i, d_i] = T.if_then_else(Indices[b_i, s_i, g_i, idx_in_split + bi_i] >= 0,
                                                                 KV[b_i, Indices[
                                                                     b_i, s_i, g_i, idx_in_split + bi_i], g_i, D + d_i],
                                                                 0)

                    T.copy(KV_spilt0_local, KV_spilt0_shared)
                    T.copy(KV_spilt1_local, KV_spilt1_shared)
                    T.copy(KV_spilt2_local, KV_spilt2_shared)

                    T.gemm(Q_spilt0_shared, KV_spilt0_shared, acc_s, transpose_B=True, k_pack=kpack,
                           policy=gemm1_policy)
                    T.gemm(Q_spilt1_shared, KV_spilt1_shared, acc_s, transpose_B=True, k_pack=kpack,
                           policy=gemm1_policy)
                    T.gemm(Q_spilt2_shared, KV_spilt2_shared, acc_s, transpose_B=True, k_pack=kpack,
                           policy=gemm1_policy)
                    T.gemm(Q_spilt3_shared, KV_spilt3_local, acc_s, transpose_B=True, k_pack=kpack, policy=gemm1_policy)
                    T.gemm(Q_tail_shared, K_tail_local, acc_s, transpose_B=True, policy=gemm1_policy)

                    T.copy(m_i, m_i_prev)
                    if gemm1_policy == T.GemmWarpPolicy.FullColK:
                        T.reduce_sum_warp(acc_s, acc_s, clear=False)
                    T.reduce_max(acc_s, m_i, dim=1, clear=False)

                    for h_i in T.Parallel(H_per_block):
                        alpha[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s[h_i, bi_i] = T.exp2(
                            acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale
                        )
                    T.reduce_sum(acc_s, sumexp_i, dim=1)
                    for h_i in T.Parallel(H_per_block):
                        sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                    for h_i, d_i in T.Parallel(H_per_block, D_spilt):
                        acc_o0[h_i, d_i] = acc_o0[h_i, d_i] * alpha[h_i]
                        acc_o1[h_i, d_i] = acc_o1[h_i, d_i] * alpha[h_i]
                        acc_o2[h_i, d_i] = acc_o2[h_i, d_i] * alpha[h_i]
                        acc_o3[h_i, d_i] = acc_o3[h_i, d_i] * alpha[h_i]

                    T.copy(KV_spilt0_shared, gemm2_kv_split0_local)
                    T.copy(acc_s, S_shared)
                    T.copy(KV_spilt3_local, KV_spilt0_shared)
                    # Cast KV to float32 for GEMM2 precision
                    T.copy(gemm2_kv_split0_local, gemm2_kv_split0_f32)
                    T.copy(KV_spilt1_shared, gemm2_kv_split1_f32)
                    T.copy(KV_spilt2_shared, gemm2_kv_split2_f32)
                    T.copy(KV_spilt0_shared, gemm2_kv_split3_f32)
                    T.gemm(S_shared, gemm2_kv_split0_f32, acc_o0, policy=gemm2_policy)
                    T.gemm(S_shared, gemm2_kv_split1_f32, acc_o1, policy=gemm2_policy)
                    T.gemm(S_shared, gemm2_kv_split2_f32, acc_o2, policy=gemm2_policy)
                    T.gemm(S_shared, gemm2_kv_split3_f32, acc_o3, policy=gemm2_policy)

                # Rescale
                for h_i, d_i in T.Parallel(H_per_block, D_spilt):
                    acc_o0[h_i, d_i] /= sumexp[h_i]
                    acc_o1[h_i, d_i] /= sumexp[h_i]
                    acc_o2[h_i, d_i] /= sumexp[h_i]
                    acc_o3[h_i, d_i] /= sumexp[h_i]
                for h_i in T.Parallel(H_per_block):
                    sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale

                if out_shared_reuse:
                    acc_oshared0 = T.alloc_shared([H_per_block, D_spilt], intermediate_dtype)
                    acc_oshared1 = T.alloc_shared([H_per_block, D_spilt], intermediate_dtype)
                    T.annotate_layout({
                        acc_oshared0: tilelang.layout.make_hcu_swizzled_layout(acc_oshared0, major_pack=2),
                        acc_oshared1: tilelang.layout.make_hcu_swizzled_layout(acc_oshared1, major_pack=2),
                    })
                    T.copy(acc_o0, acc_oshared0)
                    T.copy(acc_o1, acc_oshared1)
                    T.copy(acc_oshared0, Output_partial[b_i, s_i, split_idx, H0:H1, :D_spilt])
                    T.copy(acc_oshared1, Output_partial[b_i, s_i, split_idx, H0:H1, D_spilt:2 * D_spilt])
                    T.copy(acc_o2, acc_oshared0)
                    T.copy(acc_o3, acc_oshared1)
                    T.copy(acc_oshared0, Output_partial[b_i, s_i, split_idx, H0:H1, 2 * D_spilt:3 * D_spilt])
                    T.copy(acc_oshared1, Output_partial[b_i, s_i, split_idx, H0:H1, 3 * D_spilt:4 * D_spilt])
                else:
                    acc_oshared0 = T.alloc_shared([H_per_block, D_spilt], intermediate_dtype)
                    acc_oshared1 = T.alloc_shared([H_per_block, D_spilt], intermediate_dtype)
                    acc_oshared2 = T.alloc_shared([H_per_block, D_spilt], intermediate_dtype)
                    acc_oshared3 = T.alloc_shared([H_per_block, D_spilt], intermediate_dtype)
                    T.annotate_layout({
                        acc_oshared0: tilelang.layout.make_hcu_swizzled_layout(acc_oshared0, major_pack=2),
                        acc_oshared1: tilelang.layout.make_hcu_swizzled_layout(acc_oshared1, major_pack=2),
                        acc_oshared2: tilelang.layout.make_hcu_swizzled_layout(acc_oshared2, major_pack=2),
                        acc_oshared3: tilelang.layout.make_hcu_swizzled_layout(acc_oshared3, major_pack=2),
                    })
                    T.copy(acc_o0, acc_oshared0)
                    T.copy(acc_o1, acc_oshared1)
                    T.copy(acc_o2, acc_oshared2)
                    T.copy(acc_o3, acc_oshared3)
                    T.copy(acc_oshared0, Output_partial[b_i, s_i, split_idx, H0:H1, :D_spilt])
                    T.copy(acc_oshared1, Output_partial[b_i, s_i, split_idx, H0:H1, D_spilt:2 * D_spilt])
                    T.copy(acc_oshared2, Output_partial[b_i, s_i, split_idx, H0:H1, 2 * D_spilt:3 * D_spilt])
                    T.copy(acc_oshared3, Output_partial[b_i, s_i, split_idx, H0:H1, 3 * D_spilt:4 * D_spilt])

                T.copy(sumexp, glse[b_i, s_i, split_idx, H0:H1])

    @T.macro
    def combine(
        glse: T.Tensor(glse_shape, intermediate_dtype),  # type: ignore
        Output_partial: T.Tensor(output_partial_shape, intermediate_dtype),  # type: ignore
        Output: T.Tensor(o_shape, dtype),  # type: ignore
    ):
        with T.Kernel(seq_len, num_heads, batch_head, threads=128) as (bx, by, bz):
            po_local = T.alloc_fragment([dim], intermediate_dtype)
            o_accum_local = T.alloc_fragment([dim], accum_dtype)
            # lse_local_split = T.alloc_local([1], accum_dtype)
            lse_local_split = T.alloc_local([num_split], accum_dtype)
            lse_logsum_local = T.alloc_local([1], accum_dtype)
            lse_max_local = T.alloc_local([1], accum_dtype)
            scale_local = T.alloc_local([1], accum_dtype)

            # T.annotate_layout({
            #     lse_logsum_local: T.Fragment(lse_logsum_local.shape, forward_thread_fn=lambda i: i),
            # })
            T.clear(lse_logsum_local)
            T.clear(o_accum_local)
            lse_max_local[0] = -T.infinity(accum_dtype)
            for k in T.serial(num_split):
                # lse_max_local[0] = T.max(lse_max_local[0], glse[bz, bx, k, by])
                lse_local_split[k] = glse[bz, bx, k, by]
                lse_max_local[0] = T.max(lse_max_local[0], lse_local_split[k])

            for k in T.Pipelined(num_split, num_stages=0):
                # lse_local_split[k] = glse[bz, bx, k, by]
                lse_logsum_local[0] += T.exp2(lse_local_split[k] - lse_max_local[0])
            lse_logsum_local[0] = T.log2(lse_logsum_local[0]) + lse_max_local[0]
            for k in T.serial(num_split):
                for i in T.Parallel(dim):
                    po_local[i] = Output_partial[bz, bx, k, by, i]
                # lse_local_split[0] = glse[bz, bx, k, by]
                scale_local[0] = T.exp2(lse_local_split[k] - lse_logsum_local[0])
                for i in T.Parallel(dim):
                    o_accum_local[i] += po_local[i] * scale_local[0]
            for i in T.Parallel(dim):
                Output[bz, bx, by, i] = o_accum_local[i]

    @T.prim_func
    def main_split(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        KV: T.Tensor(kv_shape, dtype),  # type: ignore
        Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
        glse: T.Tensor(glse_shape, intermediate_dtype),  # type: ignore
        Output_partial: T.Tensor(output_partial_shape, intermediate_dtype),  # type: ignore
        Output: T.Tensor(o_shape, dtype),  # type: ignore
    ):
        sparse_mla_split(Q, KV, Indices, glse, Output_partial)
        combine(glse, Output_partial, Output)

    @T.prim_func
    def main_no_split(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        KV: T.Tensor(kv_shape, dtype),  # type: ignore
        Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
        glse: T.Tensor(glse_shape, intermediate_dtype),  # type: ignore
        Output_partial: T.Tensor(output_partial_shape, intermediate_dtype),  # type: ignore
        Output: T.Tensor(o_shape, dtype),  # type: ignore
    ):
        sparse_mla(Q, KV, Indices, Output)

    if num_split_tail > 0:
        @T.macro
        def sparse_mla_tail_split(
            Q: T.Tensor(q_shape, dtype),  # type: ignore
            KV: T.Tensor(kv_shape, dtype),  # type: ignore
            Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
            glse_tail: T.Tensor(glse_shape_tail, intermediate_dtype),  # type: ignore
            Output_partial_tail: T.Tensor(output_partial_shape_tail, intermediate_dtype),  # type: ignore
        ):
            with T.Kernel(seq_len * REPLICATE_H, batch_tail * kv_group, num_split_tail, threads=threads) as (
                    bx,
                    by,
                    bz,
                # split_idx,
            ):
                split_idx = bz
                b_head = batch - batch_tail
                b_i = (by + b_head) if kv_group == 1 else (by // kv_group + b_head)
                b_i_tail = by if kv_group == 1 else (by // kv_group)
                g_i = 0 if kv_group == 1 else (by % kv_group)
                s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)
                # q_i = q_start_index_s[0] + s_i
                # max_kv_i = (q_i + 1 - kv_stride) // kv_stride
                # kv_i = (q_i + 1 - kv_stride) // kv_stride
                # max_kv_i = kv_i if (kv_i <= seq_len_kv - 1) else seq_len_kv - 1

                H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * max_block_m)
                H1 = H0 + H_per_block

                valid_NI = T.alloc_fragment([1], "int")
                T.fill(valid_NI, 0)
                for i_i in T.serial(NI_tail):
                    first_indices = Indices[b_i, s_i, g_i, i_i * split_stride_tail + split_idx * BI]
                    # if first_indices <= max_kv_i and first_indices >= 0:
                    if first_indices >= 0:
                        valid_NI[0] += 1

                if valid_NI[0] == 0:
                    acc_o = T.alloc_fragment([H_per_block, D], accum_dtype)
                    sumexp = T.alloc_fragment([H_per_block], accum_dtype)
                    T.fill(sumexp, -T.infinity(accum_dtype))
                    T.fill(acc_o, 0)
                    T.copy(sumexp, glse_tail[b_i_tail, s_i, split_idx, H0:H1])
                    T.copy(acc_o, Output_partial_tail[b_i_tail, s_i, split_idx, H0:H1, :D])
                else:
                    Q_spilt0_shared = T.alloc_fragment([H_per_block, D_spilt], dtype)
                    Q_spilt1_shared = T.alloc_fragment([H_per_block, D_spilt], dtype)
                    Q_spilt2_shared = T.alloc_fragment([H_per_block, D_spilt], dtype)
                    Q_spilt3_shared = T.alloc_fragment([H_per_block, D_spilt], dtype)
                    Q_tail_shared = T.alloc_fragment([H_per_block, D_tail], dtype)

                    KV_spilt0_shared = T.alloc_shared([BI, D_spilt], dtype)
                    KV_spilt1_shared = T.alloc_shared([BI, D_spilt], dtype)
                    KV_spilt2_shared = T.alloc_shared([BI, D_spilt], dtype)
                    # KV_spilt3_shared = T.alloc_shared([BI, D_spilt], dtype)
                    KV_spilt0_local = T.alloc_fragment([BI, D_spilt], dtype)
                    KV_spilt1_local = T.alloc_fragment([BI, D_spilt], dtype)
                    KV_spilt2_local = T.alloc_fragment([BI, D_spilt], dtype)
                    KV_spilt3_local = T.alloc_fragment([BI, D_spilt], dtype)
                    K_tail_local = T.alloc_fragment([BI, D_tail], dtype)
                    mask = T.alloc_fragment([BI], "bool")

                    # float32 copies of KV for GEMM2 (S @ V)
                    gemm2_kv_split0_f32 = T.alloc_fragment([BI, D_spilt], accum_dtype)
                    gemm2_kv_split1_f32 = T.alloc_fragment([BI, D_spilt], accum_dtype)
                    gemm2_kv_split2_f32 = T.alloc_fragment([BI, D_spilt], accum_dtype)
                    gemm2_kv_split3_f32 = T.alloc_fragment([BI, D_spilt], accum_dtype)

                    acc_o0 = T.alloc_fragment([H_per_block, D_spilt], accum_dtype)
                    acc_o1 = T.alloc_fragment([H_per_block, D_spilt], accum_dtype)
                    acc_o2 = T.alloc_fragment([H_per_block, D_spilt], accum_dtype)
                    acc_o3 = T.alloc_fragment([H_per_block, D_spilt], accum_dtype)

                    acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
                    S_shared = T.alloc_shared([H_per_block, BI], accum_dtype)
                    sumexp = T.alloc_fragment([H_per_block], accum_dtype)
                    sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
                    alpha = T.alloc_fragment([H_per_block], accum_dtype)
                    m_i = T.alloc_fragment([H_per_block], accum_dtype)
                    m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)
                    indices_local = T.alloc_local([1], indices_dtype)
                    indices_mask = T.alloc_fragment([BI], indices_dtype)
                    indices_local_1 = T.alloc_fragment([1], indices_dtype)

                    tx = T.get_thread_binding()
                    T.copy(Q[b_i, s_i, H0:H1, :D_spilt], Q_spilt0_shared, coalesced_width=kv_vectorized)
                    T.copy(Q[b_i, s_i, H0:H1, D_spilt:2 * D_spilt], Q_spilt1_shared, coalesced_width=kv_vectorized)
                    T.copy(Q[b_i, s_i, H0:H1, 2 * D_spilt:3 * D_spilt], Q_spilt2_shared, coalesced_width=kv_vectorized)
                    T.copy(Q[b_i, s_i, H0:H1, 3 * D_spilt:4 * D_spilt], Q_spilt3_shared, coalesced_width=kv_vectorized)
                    T.copy(Q[b_i, s_i, H0:H1, D:], Q_tail_shared)

                    T.fill(acc_o0, 0)
                    T.fill(acc_o1, 0)
                    T.fill(acc_o2, 0)
                    T.fill(acc_o3, 0)
                    T.fill(sumexp, 1)
                    T.fill(m_i, -(2 ** 30))  # avoid -inf - inf to cause nan

                    for i_i in T.Pipelined(valid_NI[0], num_stages=num_stages):
                        idx_in_split = i_i * split_stride_tail + split_idx * BI
                        for bi_i in T.Parallel(BI):
                            indices_mask[bi_i] = Indices[b_i, s_i, g_i, idx_in_split + bi_i]
                            # mask[bi_i] = indices_mask[bi_i] <= max_kv_i and indices_mask[bi_i] >= 0
                            mask[bi_i] = indices_mask[bi_i] >= 0

                        for h_i, bi_i in T.Parallel(H_per_block, BI):
                            acc_s[h_i, bi_i] = T.if_then_else(
                                mask[bi_i], 0, -T.infinity(acc_s.dtype)
                            )

                        for bi_i, d_i in T.Parallel(BI, D_spilt):
                            KV_spilt0_local[bi_i, d_i] = T.if_then_else(
                                Indices[b_i, s_i, g_i, idx_in_split + bi_i] >= 0,
                                KV[b_i, Indices[b_i, s_i, g_i, idx_in_split + bi_i], g_i, d_i], 0)
                            KV_spilt1_local[bi_i, d_i] = T.if_then_else(
                                Indices[b_i, s_i, g_i, idx_in_split + bi_i] >= 0,
                                KV[b_i, Indices[b_i, s_i, g_i, idx_in_split + bi_i], g_i, D_spilt + d_i], 0)
                            KV_spilt2_local[bi_i, d_i] = T.if_then_else(
                                Indices[b_i, s_i, g_i, idx_in_split + bi_i] >= 0,
                                KV[b_i, Indices[b_i, s_i, g_i, idx_in_split + bi_i], g_i, 2 * D_spilt + d_i], 0)
                        for bi_i, d_i in T.Parallel(BI, D_spilt):
                            KV_spilt3_local[bi_i, d_i] = T.if_then_else(
                                Indices[b_i, s_i, g_i, idx_in_split + bi_i] >= 0,
                                KV[b_i, Indices[b_i, s_i, g_i, idx_in_split + bi_i], g_i, 3 * D_spilt + d_i], 0)

                        for bi_i, d_i in T.Parallel(BI, D_tail):
                            K_tail_local[bi_i, d_i] = T.if_then_else(Indices[b_i, s_i, g_i, idx_in_split + bi_i] >= 0,
                                                                     KV[b_i, Indices[
                                                                         b_i, s_i, g_i, idx_in_split + bi_i], g_i, D + d_i],
                                                                     0)
                        T.copy(KV_spilt0_local, KV_spilt0_shared)
                        T.copy(KV_spilt1_local, KV_spilt1_shared)
                        T.copy(KV_spilt2_local, KV_spilt2_shared)

                        T.gemm(Q_spilt0_shared, KV_spilt0_shared, acc_s, transpose_B=True, k_pack=kpack,
                               policy=gemm1_policy)
                        T.gemm(Q_spilt1_shared, KV_spilt1_shared, acc_s, transpose_B=True, k_pack=kpack,
                               policy=gemm1_policy)
                        T.gemm(Q_spilt2_shared, KV_spilt2_shared, acc_s, transpose_B=True, k_pack=kpack,
                               policy=gemm1_policy)
                        T.gemm(Q_spilt3_shared, KV_spilt3_local, acc_s, transpose_B=True, k_pack=kpack,
                               policy=gemm1_policy)
                        T.gemm(Q_tail_shared, K_tail_local, acc_s, transpose_B=True, policy=gemm1_policy)

                        T.copy(m_i, m_i_prev)
                        if gemm1_policy == T.GemmWarpPolicy.FullColK:
                            T.reduce_sum_warp(acc_s, acc_s, clear=False)
                        T.reduce_max(acc_s, m_i, dim=1, clear=False)
                        for h_i in T.Parallel(H_per_block):
                            alpha[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                        for h_i, bi_i in T.Parallel(H_per_block, BI):
                            acc_s[h_i, bi_i] = T.exp2(
                                acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale
                            )
                        T.reduce_sum(acc_s, sumexp_i, dim=1)
                        for h_i in T.Parallel(H_per_block):
                            sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                        for h_i, d_i in T.Parallel(H_per_block, D_spilt):
                            acc_o0[h_i, d_i] = acc_o0[h_i, d_i] * alpha[h_i]
                            acc_o1[h_i, d_i] = acc_o1[h_i, d_i] * alpha[h_i]
                            acc_o2[h_i, d_i] = acc_o2[h_i, d_i] * alpha[h_i]
                            acc_o3[h_i, d_i] = acc_o3[h_i, d_i] * alpha[h_i]

                        T.copy(acc_s, S_shared)
                        # Cast KV to float32 for GEMM2 precision
                        T.copy(KV_spilt0_shared, gemm2_kv_split0_f32)
                        T.copy(KV_spilt1_shared, gemm2_kv_split1_f32)
                        T.copy(KV_spilt2_shared, gemm2_kv_split2_f32)
                        T.copy(KV_spilt3_local, KV_spilt0_shared)
                        T.copy(KV_spilt0_shared, gemm2_kv_split3_f32)
                        T.gemm(S_shared, gemm2_kv_split0_f32, acc_o0, policy=gemm2_policy)
                        T.gemm(S_shared, gemm2_kv_split1_f32, acc_o1, policy=gemm2_policy)
                        T.gemm(S_shared, gemm2_kv_split2_f32, acc_o2, policy=gemm2_policy)
                        T.gemm(S_shared, gemm2_kv_split3_f32, acc_o3, policy=gemm2_policy)

                    # Rescale
                    for h_i, d_i in T.Parallel(H_per_block, D_spilt):
                        acc_o0[h_i, d_i] /= sumexp[h_i]
                        acc_o1[h_i, d_i] /= sumexp[h_i]
                        acc_o2[h_i, d_i] /= sumexp[h_i]
                        acc_o3[h_i, d_i] /= sumexp[h_i]

                    if out_shared_reuse:
                        acc_oshared0 = T.alloc_shared([H_per_block, D_spilt], intermediate_dtype)
                        acc_oshared1 = T.alloc_shared([H_per_block, D_spilt], intermediate_dtype)
                        T.annotate_layout({
                            acc_oshared0: tilelang.layout.make_hcu_swizzled_layout(acc_oshared0, major_pack=2),
                            acc_oshared1: tilelang.layout.make_hcu_swizzled_layout(acc_oshared1, major_pack=2),
                        })
                        T.copy(acc_o0, acc_oshared0)
                        T.copy(acc_o1, acc_oshared1)
                        T.copy(acc_oshared0, Output_partial_tail[b_i_tail, s_i, split_idx, H0:H1, :D_spilt])
                        T.copy(acc_oshared1, Output_partial_tail[b_i_tail, s_i, split_idx, H0:H1, D_spilt:2 * D_spilt])
                        T.copy(acc_o2, acc_oshared0)
                        T.copy(acc_o3, acc_oshared1)
                        T.copy(acc_oshared0,
                               Output_partial_tail[b_i_tail, s_i, split_idx, H0:H1, 2 * D_spilt:3 * D_spilt])
                        T.copy(acc_oshared1,
                               Output_partial_tail[b_i_tail, s_i, split_idx, H0:H1, 3 * D_spilt:4 * D_spilt])
                    else:
                        acc_oshared0 = T.alloc_shared([H_per_block, D_spilt], intermediate_dtype)
                        acc_oshared1 = T.alloc_shared([H_per_block, D_spilt], intermediate_dtype)
                        acc_oshared2 = T.alloc_shared([H_per_block, D_spilt], intermediate_dtype)
                        acc_oshared3 = T.alloc_shared([H_per_block, D_spilt], intermediate_dtype)
                        T.annotate_layout({
                            acc_oshared0: tilelang.layout.make_hcu_swizzled_layout(acc_oshared0, major_pack=2),
                            acc_oshared1: tilelang.layout.make_hcu_swizzled_layout(acc_oshared1, major_pack=2),
                            acc_oshared2: tilelang.layout.make_hcu_swizzled_layout(acc_oshared2, major_pack=2),
                            acc_oshared3: tilelang.layout.make_hcu_swizzled_layout(acc_oshared3, major_pack=2),
                        })
                        T.copy(acc_o0, acc_oshared0)
                        T.copy(acc_o1, acc_oshared1)
                        T.copy(acc_o2, acc_oshared2)
                        T.copy(acc_o3, acc_oshared3)
                        T.copy(acc_oshared0, Output_partial_tail[b_i_tail, s_i, split_idx, H0:H1, :D_spilt])
                        T.copy(acc_oshared1, Output_partial_tail[b_i_tail, s_i, split_idx, H0:H1, D_spilt:2 * D_spilt])
                        T.copy(acc_oshared2,
                               Output_partial_tail[b_i_tail, s_i, split_idx, H0:H1, 2 * D_spilt:3 * D_spilt])
                        T.copy(acc_oshared3,
                               Output_partial_tail[b_i_tail, s_i, split_idx, H0:H1, 3 * D_spilt:4 * D_spilt])

                    for h_i in T.Parallel(H_per_block):
                        sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale
                    T.copy(sumexp, glse_tail[b_i_tail, s_i, split_idx, H0:H1])

        @T.macro
        def sparse_mla_head_no_split(
            Q: T.Tensor(q_shape, dtype),  # type: ignore
            KV: T.Tensor(kv_shape, dtype),  # type: ignore
            Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
            Output: T.Tensor(o_shape, dtype),  # type: ignore
        ):
            with T.Kernel(seq_len * REPLICATE_H, batch - batch_tail, kv_group, threads=threads) as (
                    bx,
                    by,
                    bz,
            ):
                Q_spilt0_shared = T.alloc_fragment([H_per_block, D_spilt], dtype)
                Q_spilt1_shared = T.alloc_fragment([H_per_block, D_spilt], dtype)
                Q_spilt2_shared = T.alloc_fragment([H_per_block, D_spilt], dtype)
                Q_spilt3_shared = T.alloc_fragment([H_per_block, D_spilt], dtype)
                Q_tail_shared = T.alloc_fragment([H_per_block, D_tail], dtype)

                KV_spilt0_shared = T.alloc_shared([BI, D_spilt], dtype)
                KV_spilt1_shared = T.alloc_shared([BI, D_spilt], dtype)
                KV_spilt2_shared = T.alloc_shared([BI, D_spilt], dtype)
                KV_split0_local = T.alloc_fragment([BI, D_spilt], dtype)
                KV_split1_local = T.alloc_fragment([BI, D_spilt], dtype)
                KV_split2_local = T.alloc_fragment([BI, D_spilt], dtype)
                KV_spilt3_local = T.alloc_fragment([BI, D_spilt], dtype)
                K_tail_local = T.alloc_fragment([BI, D_tail], dtype)
                mask = T.alloc_fragment([BI], "bool")

                # float32 copies of KV for GEMM2 (S @ V)
                gemm2_kv_split0_f32 = T.alloc_fragment([BI, D_spilt], accum_dtype)
                gemm2_kv_split1_f32 = T.alloc_fragment([BI, D_spilt], accum_dtype)
                gemm2_kv_split2_f32 = T.alloc_fragment([BI, D_spilt], accum_dtype)
                gemm2_kv_split3_f32 = T.alloc_fragment([BI, D_spilt], accum_dtype)

                acc_o0 = T.alloc_fragment([H_per_block, D_spilt], accum_dtype)
                acc_o1 = T.alloc_fragment([H_per_block, D_spilt], accum_dtype)
                acc_o2 = T.alloc_fragment([H_per_block, D_spilt], accum_dtype)
                acc_o3 = T.alloc_fragment([H_per_block, D_spilt], accum_dtype)

                acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
                S_shared = T.alloc_shared([H_per_block, BI], accum_dtype)
                sumexp = T.alloc_fragment([H_per_block], accum_dtype)
                sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
                alpha = T.alloc_fragment([H_per_block], accum_dtype)
                m_i = T.alloc_fragment([H_per_block], accum_dtype)
                m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)
                indices_local = T.alloc_local([1], indices_dtype)
                indices_mask = T.alloc_fragment([BI], indices_dtype)
                indices_local_1 = T.alloc_fragment([1], indices_dtype)
                valid_NI = T.alloc_fragment([1], "int")

                b_i, g_i = by, bz
                s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)
                # q_i = q_start_index_s[0] + s_i
                # max_kv_i = (q_i + 1 - kv_stride) // kv_stride
                # kv_i = (q_i + 1 - kv_stride) // kv_stride
                # max_kv_i = kv_i if (kv_i <= seq_len_kv - 1) else seq_len_kv - 1

                H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * max_block_m)
                H1 = H0 + H_per_block

                tx = T.get_thread_binding()
                T.copy(Q[b_i, s_i, H0:H1, :D_spilt], Q_spilt0_shared, coalesced_width=kv_vectorized)
                T.copy(Q[b_i, s_i, H0:H1, D_spilt:2 * D_spilt], Q_spilt1_shared, coalesced_width=kv_vectorized)
                T.copy(Q[b_i, s_i, H0:H1, 2 * D_spilt:3 * D_spilt], Q_spilt2_shared, coalesced_width=kv_vectorized)
                T.copy(Q[b_i, s_i, H0:H1, 3 * D_spilt:4 * D_spilt], Q_spilt3_shared, coalesced_width=kv_vectorized)
                T.copy(Q[b_i, s_i, H0:H1, D:], Q_tail_shared)

                T.fill(acc_o0, 0)
                T.fill(acc_o1, 0)
                T.fill(acc_o2, 0)
                T.fill(acc_o3, 0)
                T.fill(sumexp, 1)
                T.fill(m_i, -(2 ** 30))  # avoid -inf - inf to cause nan
                T.fill(valid_NI, 0)

                for i_i in T.serial(NI):
                    first_indices = Indices[b_i, s_i, g_i, i_i * BI]
                    # if first_indices <= max_kv_i and first_indices >= 0:
                    if first_indices >= 0:
                        valid_NI[0] += 1

                for i_i in T.Pipelined(valid_NI[0], num_stages=num_stages):
                    for bi_i in T.Parallel(BI):
                        indices_mask[bi_i] = Indices[b_i, s_i, g_i, i_i * BI + bi_i]
                        mask[bi_i] = indices_mask[bi_i] >= 0

                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s[h_i, bi_i] = T.if_then_else(
                            mask[bi_i], 0, -T.infinity(acc_s.dtype)
                        )

                    for bi_i, d_i in T.Parallel(BI, D_spilt):
                        KV_spilt0_local[bi_i, d_i] = T.if_then_else(Indices[b_i, s_i, g_i, i_i * BI + bi_i] >= 0,
                                                                    KV[b_i, Indices[
                                                                        b_i, s_i, g_i, i_i * BI + bi_i], g_i, d_i], 0)
                        KV_spilt1_local[bi_i, d_i] = T.if_then_else(Indices[b_i, s_i, g_i, i_i * BI + bi_i] >= 0,
                                                                    KV[b_i, Indices[
                                                                        b_i, s_i, g_i, i_i * BI + bi_i], g_i, D_spilt + d_i],
                                                                    0)
                        KV_spilt2_local[bi_i, d_i] = T.if_then_else(Indices[b_i, s_i, g_i, i_i * BI + bi_i] >= 0,
                                                                    KV[b_i, Indices[
                                                                        b_i, s_i, g_i, i_i * BI + bi_i], g_i, 2 * D_spilt + d_i],
                                                                    0)
                    for bi_i, d_i in T.Parallel(BI, D_spilt):
                        KV_spilt3_local[bi_i, d_i] = T.if_then_else(Indices[b_i, s_i, g_i, i_i * BI + bi_i] >= 0,
                                                                    KV[b_i, Indices[
                                                                        b_i, s_i, g_i, i_i * BI + bi_i], g_i, 3 * D_spilt + d_i],
                                                                    0)

                    for bi_i, d_i in T.Parallel(BI, D_tail):
                        K_tail_local[bi_i, d_i] = T.if_then_else(Indices[b_i, s_i, g_i, i_i * BI + bi_i] >= 0,
                                                                 KV[b_i, Indices[
                                                                     b_i, s_i, g_i, i_i * BI + bi_i], g_i, D + d_i], 0)

                    T.copy(KV_spilt0_local, KV_spilt0_shared)
                    T.copy(KV_spilt1_local, KV_spilt1_shared)
                    T.copy(KV_spilt2_local, KV_spilt2_shared)
                    T.gemm(Q_spilt0_shared, KV_spilt0_shared, acc_s, transpose_B=True, k_pack=kpack,
                           policy=gemm1_policy)
                    T.gemm(Q_spilt1_shared, KV_spilt1_shared, acc_s, transpose_B=True, k_pack=kpack,
                           policy=gemm1_policy)
                    T.gemm(Q_spilt2_shared, KV_spilt2_shared, acc_s, transpose_B=True, k_pack=kpack,
                           policy=gemm1_policy)
                    T.gemm(Q_spilt3_shared, KV_spilt3_local, acc_s, transpose_B=True, k_pack=kpack, policy=gemm1_policy)
                    T.gemm(Q_tail_shared, K_tail_local, acc_s, transpose_B=True, policy=gemm1_policy)

                    T.copy(m_i, m_i_prev)
                    if gemm1_policy == T.GemmWarpPolicy.FullColK:
                        T.reduce_sum_warp(acc_s, acc_s, clear=False)
                    T.reduce_max(acc_s, m_i, dim=1, clear=False)
                    for h_i in T.Parallel(H_per_block):
                        alpha[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                    for h_i, bi_i in T.Parallel(H_per_block, BI):
                        acc_s[h_i, bi_i] = T.exp2(
                            acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale
                        )
                    T.reduce_sum(acc_s, sumexp_i, dim=1)
                    for h_i in T.Parallel(H_per_block):
                        sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                    for h_i, d_i in T.Parallel(H_per_block, D_spilt):
                        acc_o0[h_i, d_i] = acc_o0[h_i, d_i] * alpha[h_i]
                        acc_o1[h_i, d_i] = acc_o1[h_i, d_i] * alpha[h_i]
                        acc_o2[h_i, d_i] = acc_o2[h_i, d_i] * alpha[h_i]
                        acc_o3[h_i, d_i] = acc_o3[h_i, d_i] * alpha[h_i]

                    T.copy(acc_s, S_shared)
                    # Cast KV to float32 for GEMM2 precision
                    T.copy(KV_spilt0_shared, gemm2_kv_split0_f32)
                    T.copy(KV_spilt1_shared, gemm2_kv_split1_f32)
                    T.copy(KV_spilt2_shared, gemm2_kv_split2_f32)
                    T.copy(KV_spilt3_local, KV_spilt0_shared)
                    T.copy(KV_spilt0_shared, gemm2_kv_split3_f32)
                    T.gemm(S_shared, gemm2_kv_split0_f32, acc_o0, policy=gemm2_policy)
                    T.gemm(S_shared, gemm2_kv_split1_f32, acc_o1, policy=gemm2_policy)
                    T.gemm(S_shared, gemm2_kv_split2_f32, acc_o2, policy=gemm2_policy)
                    T.gemm(S_shared, gemm2_kv_split3_f32, acc_o3, policy=gemm2_policy)
                # Rescale
                for h_i, d_i in T.Parallel(H_per_block, D_spilt):
                    acc_o0[h_i, d_i] /= sumexp[h_i]
                    acc_o1[h_i, d_i] /= sumexp[h_i]
                    acc_o2[h_i, d_i] /= sumexp[h_i]
                    acc_o3[h_i, d_i] /= sumexp[h_i]

                if out_shared_reuse:
                    acc_oshared0 = T.alloc_shared([H_per_block, D_spilt], dtype)
                    acc_oshared1 = T.alloc_shared([H_per_block, D_spilt], dtype)
                    T.annotate_layout({
                        acc_oshared0: tilelang.layout.make_hcu_swizzled_layout(acc_oshared0, major_pack=2),
                        acc_oshared1: tilelang.layout.make_hcu_swizzled_layout(acc_oshared1, major_pack=2),
                    })
                    T.copy(acc_o0, acc_oshared0)
                    T.copy(acc_o1, acc_oshared1)
                    T.copy(acc_oshared0, Output[b_i, s_i, H0:H1, 2 * D_spilt:3 * D_spilt])
                    T.copy(acc_oshared1, Output[b_i, s_i, H0:H1, 3 * D_spilt:4 * D_spilt])
                    T.copy(acc_o2, acc_oshared0)
                    T.copy(acc_o3, acc_oshared1)
                    T.copy(acc_oshared0, Output[b_i, s_i, H0:H1, :D_spilt])
                    T.copy(acc_oshared1, Output[b_i, s_i, H0:H1, D_spilt:2 * D_spilt])
                else:
                    acc_oshared0 = T.alloc_shared([H_per_block, D_spilt], dtype)
                    acc_oshared1 = T.alloc_shared([H_per_block, D_spilt], dtype)
                    acc_oshared2 = T.alloc_shared([H_per_block, D_spilt], dtype)
                    acc_oshared3 = T.alloc_shared([H_per_block, D_spilt], dtype)
                    T.annotate_layout({
                        acc_oshared0: tilelang.layout.make_hcu_swizzled_layout(acc_oshared0, major_pack=2),
                        acc_oshared1: tilelang.layout.make_hcu_swizzled_layout(acc_oshared1, major_pack=2),
                        acc_oshared2: tilelang.layout.make_hcu_swizzled_layout(acc_oshared2, major_pack=2),
                        acc_oshared3: tilelang.layout.make_hcu_swizzled_layout(acc_oshared3, major_pack=2),
                    })
                    T.copy(acc_o0, acc_oshared0)
                    T.copy(acc_o1, acc_oshared1)
                    T.copy(acc_o2, acc_oshared2)
                    T.copy(acc_o3, acc_oshared3)
                    T.copy(acc_oshared0, Output[b_i, s_i, H0:H1, :D_spilt])
                    T.copy(acc_oshared1, Output[b_i, s_i, H0:H1, D_spilt:2 * D_spilt])
                    T.copy(acc_oshared2, Output[b_i, s_i, H0:H1, 2 * D_spilt:3 * D_spilt])
                    T.copy(acc_oshared3, Output[b_i, s_i, H0:H1, 3 * D_spilt:4 * D_spilt])

        @T.macro
        def combine_tail(
            glse_tail: T.Tensor(glse_shape_tail, intermediate_dtype),
            Output_partial_tail: T.Tensor(output_partial_shape_tail, intermediate_dtype),
            Output: T.Tensor(o_shape, dtype),
        ):
            with T.Kernel(seq_len, num_heads, batch_tail, threads=128) as (bx, by, bz):
                po_local = T.alloc_fragment([dim], intermediate_dtype)
                o_accum_local = T.alloc_fragment([dim], accum_dtype)
                lse_local_split = T.alloc_local([num_split_tail], accum_dtype)
                lse_logsum_local = T.alloc_local([1], accum_dtype)
                lse_max_local = T.alloc_local([1], accum_dtype)
                scale_local = T.alloc_local([1], accum_dtype)

                T.clear(lse_logsum_local)
                T.clear(o_accum_local)

                b_head = batch - batch_tail
                out_bz = bz + b_head
                lse_max_local[0] = -T.infinity(accum_dtype)
                for k in T.serial(num_split_tail):
                    lse_local_split[k] = glse_tail[bz, bx, k, by]
                    lse_max_local[0] = T.max(lse_max_local[0], lse_local_split[k])
                for k in T.Pipelined(num_split_tail, num_stages=0):
                    lse_logsum_local[0] += T.exp2(lse_local_split[k] - lse_max_local[0])
                lse_logsum_local[0] = T.log2(lse_logsum_local[0]) + lse_max_local[0]
                for k in T.serial(num_split_tail):
                    for i in T.Parallel(dim):
                        po_local[i] = Output_partial_tail[bz, bx, k, by, i]
                    scale_local[0] = T.exp2(lse_local_split[k] - lse_logsum_local[0])
                    for i in T.Parallel(dim):
                        o_accum_local[i] += po_local[i] * scale_local[0]
                for i in T.Parallel(dim):
                    Output[out_bz, bx, by, i] = o_accum_local[i]

        @T.macro
        def combine_all(
            glse: T.Tensor(glse_shape, intermediate_dtype),
            Output_partial: T.Tensor(output_partial_shape, intermediate_dtype),
            glse_tail: T.Tensor(glse_shape_tail, intermediate_dtype),
            Output_partial_tail: T.Tensor(output_partial_shape_tail, intermediate_dtype),
            Output: T.Tensor(o_shape, dtype),
        ):
            with T.Kernel(seq_len, num_heads, batch, threads=128) as (bx, by, bz):
                po_local = T.alloc_fragment([dim], intermediate_dtype)
                o_accum_local = T.alloc_fragment([dim], accum_dtype)
                lse_logsum_local = T.alloc_local([1], accum_dtype)
                lse_max_local = T.alloc_local([1], accum_dtype)
                scale_local = T.alloc_local([1], accum_dtype)

                T.clear(lse_logsum_local)
                T.clear(o_accum_local)

                if bz < batch_head:
                    lse_local_split = T.alloc_local([num_split], accum_dtype)
                    lse_max_local[0] = -T.infinity(accum_dtype)
                    for k in T.serial(num_split):
                        lse_local_split[k] = glse[bz, bx, k, by]
                        lse_max_local[0] = T.max(lse_max_local[0], lse_local_split[k])
                    for k in T.Pipelined(num_split, num_stages=0):
                        lse_logsum_local[0] += T.exp2(lse_local_split[k] - lse_max_local[0])
                    lse_logsum_local[0] = T.log2(lse_logsum_local[0]) + lse_max_local[0]
                    for k in T.serial(num_split):
                        for i in T.Parallel(dim):
                            po_local[i] = Output_partial[bz, bx, k, by, i]
                        scale_local[0] = T.exp2(lse_local_split[k] - lse_logsum_local[0])
                        for i in T.Parallel(dim):
                            o_accum_local[i] += po_local[i] * scale_local[0]
                    for i in T.Parallel(dim):
                        Output[bz, bx, by, i] = o_accum_local[i]
                else:
                    bz_tail = bz - batch_head
                    lse_max_local[0] = -T.infinity(accum_dtype)
                    lse_local_split = T.alloc_local([num_split_tail], accum_dtype)
                    for k in T.serial(num_split_tail):
                        lse_local_split[k] = glse_tail[bz_tail, bx, k, by]
                        lse_max_local[0] = T.max(lse_max_local[0], lse_local_split[k])
                    for k in T.Pipelined(num_split_tail, num_stages=0):
                        lse_logsum_local[0] += T.exp2(lse_local_split[k] - lse_max_local[0])
                    lse_logsum_local[0] = T.log2(lse_logsum_local[0]) + lse_max_local[0]
                    for k in T.serial(num_split_tail):
                        for i in T.Parallel(dim):
                            po_local[i] = Output_partial_tail[bz_tail, bx, k, by, i]
                        scale_local[0] = T.exp2(lse_local_split[k] - lse_logsum_local[0])
                        for i in T.Parallel(dim):
                            o_accum_local[i] += po_local[i] * scale_local[0]
                    for i in T.Parallel(dim):
                        Output[bz, bx, by, i] = o_accum_local[i]

        @T.prim_func
        def main_all_split(
            Q: T.Tensor(q_shape, dtype),  # type: ignore
            KV: T.Tensor(kv_shape, dtype),  # type: ignore
            Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
            glse: T.Tensor(glse_shape, intermediate_dtype),  # type: ignore
            Output_partial: T.Tensor(output_partial_shape, intermediate_dtype),  # type: ignore
            glse_tail: T.Tensor(glse_shape_tail, intermediate_dtype),  # type: ignore
            Output_partial_tail: T.Tensor(output_partial_shape_tail, intermediate_dtype),  # type: ignore
            Output: T.Tensor(o_shape, dtype),  # type: ignore
        ):
            sparse_mla_split(Q, KV, Indices, glse, Output_partial)
            sparse_mla_tail_split(Q, KV, Indices, glse_tail, Output_partial_tail)
            combine_all(glse, Output_partial, glse_tail, Output_partial_tail, Output)

        @T.prim_func
        def main_tail_split(
            Q: T.Tensor(q_shape, dtype),  # type: ignore
            KV: T.Tensor(kv_shape, dtype),  # type: ignore
            Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
            glse: T.Tensor(glse_shape, intermediate_dtype),  # type: ignore
            Output_partial: T.Tensor(output_partial_shape, intermediate_dtype),  # type: ignore
            glse_tail: T.Tensor(glse_shape_tail, intermediate_dtype),  # type: ignore
            Output_partial_tail: T.Tensor(output_partial_shape_tail, intermediate_dtype),  # type: ignore
            Output: T.Tensor(o_shape, dtype),  # type: ignore
        ):
            sparse_mla_head_no_split(Q, KV, Indices, Output)
            sparse_mla_tail_split(Q, KV, Indices, glse_tail, Output_partial_tail)
            combine_tail(glse_tail, Output_partial_tail, Output)

    if num_split_tail > 0:
        if num_split > 1 and num_split_tail > 1:
            return main_all_split
        elif num_split_tail > 1:
            return main_tail_split
        else:
            assert False, "for num_split_tail > 0, at least num_split_tail need > 1"
    if num_split > 1:
        return main_split
    else:
        return main_no_split


def get_benifit(ceil, occupancy):
    if occupancy <= 1:
        return 1.0
    benifit = 1.0
    if ceil >= occupancy:
        benifit = 1.8
    return benifit


@functools.lru_cache(maxsize=8192)
def get_score(key, q_heads, cu_count, max_split, split, occupancy, combine_base):
    combine_score = (key * q_heads + cu_count - 1) // cu_count * combine_base * (split >> 1)
    score_base = (max_split // split)
    ceil = (key * split + cu_count - 1) // cu_count
    benifit = get_benifit(ceil, occupancy)

    remain = ceil % occupancy
    floor = ceil - remain
    mla_score = (floor * score_base / benifit + remain * score_base) * (1.05 ** (ceil >> 1))
    score = mla_score + combine_score
    return score


@functools.lru_cache(maxsize=4096)
def get_best_split(key, cu_count, q_heads, max_split, combine_base, occupancy, split_base=1):
    min_score = get_score(key, q_heads, cu_count, max_split, split_base, occupancy, combine_base)
    num_split = split_base

    # Select splits list based on key value
    if key <= 4:
        splits = [16, 32]
    elif key <= cu_count // 2:
        splits = [1, 2, 4, 8, 16]
    elif key <= cu_count:
        splits = [1, 2, 4, 8]
    elif key <= cu_count * 2:
        splits = [1, 2, 4]
    elif key <= cu_count * 4:
        splits = [1, 2]
    else:
        splits = [1]

    # Optimized: filter splits by split_base before loop to reduce iterations
    for split in splits:
        if split % split_base != 0:
            continue
        score = get_score(key, q_heads, cu_count, max_split, split, occupancy, combine_base)
        if score < min_score:
            min_score = score
            num_split = split
    return num_split, min_score


@functools.lru_cache(maxsize=4096)
def get_streamk_config(key, count, cu_count, q_heads, max_split, combine_base, occupancy):
    tail = key % count
    head = key // count * count
    head_score = 0
    tail_score = 0
    num_split = 0
    num_split_tail = 0
    if head > 0:
        num_split = cu_count // count
        gcd = math.gcd(head // count, num_split)
        num_split = num_split // gcd
        if occupancy > 1:
            num_split, head_score = get_best_split(head, cu_count, q_heads, max_split, combine_base, occupancy,
                                                   split_base=num_split)

    if tail > 0:
        num_split_tail, tail_score = get_best_split(tail, cu_count, q_heads, max_split, combine_base, occupancy)

    merged = False
    if num_split == num_split_tail or num_split == 0:
        # split of head and tail are the same, use splitk not streamk
        head = key
        tail = 0
        num_split = num_split_tail
        num_split_tail = 0
        merged = True
    elif num_split_tail == 1:
        # not support tail no split
        head = key
        tail = 0
        num_split, head_score = get_best_split(key, cu_count, q_heads, max_split, combine_base, occupancy)
        num_split_tail = 0
        tail_score = 0

    if merged:
        total_score = get_score(key, q_heads, cu_count, max_split, num_split, occupancy, combine_base)
    else:
        total_score = head_score + tail_score

    return total_score, head, num_split, num_split_tail


@functools.lru_cache(maxsize=2048)
def get_best_streamk_config(batch, seq_len, q_heads):
    # in tp mode, q_heads is 16, so we set block_M == 16, if dp mod need a new kernel for better performance
    block_M = 16 if q_heads <= 16 else 32

    replicat_H = (q_heads + block_M - 1) // block_M
    seq_len_replicat = seq_len * replicat_H
    key = batch * seq_len_replicat
    assert key > 0, "batch * seq_len_replicat must be greater than 0"
    assert cu_count > 0, "cu_count must be greater than 0"

    config_map = {}
    key_min = cu_count * 2 // 16
    counts = []

    if cu_count == 72:
        counts = [72, 36, 18, 9]
        key_min = 9
        config_map = config_map_cu72
    elif cu_count == 64:
        counts = [64, 32, 16, 8]
        key_min = 8

    if key in config_map.keys() and seq_len_replicat == 1 and block_M == 16:
        config = config_map[key]
        block_I, threads, num_stages, num_split, num_split_tail, batch_head = \
            config["block_I"], config["threads"], config["num_stages"], config["num_split"], config["num_split_tail"], \
            config["batch_head"]
        # logger.info(f"Using best config for batch={batch}, seq_len={seq_len}, q_heads={q_heads}, cu_count={cu_count}: "
        #             f"block_I={block_I}, threads={threads}, num_stages={num_stages}, num_split={num_split}, num_split_tail={num_split_tail}, batch_head={batch_head}")
        return block_I, threads, num_stages, num_split, num_split_tail, batch_head

    batch_head = batch
    num_split_tail = 0
    threads = 256
    num_stages = 0
    block_I = 32
    combine_base = 0.04
    if len(counts) > 0 and cu_count % seq_len_replicat == 0 and key >= key_min and key <= 128:
        # streamk
        max_split = 128
        # when block_I = 32, occupancy is limited by lds as 2
        occupancy = 2
        min_score = get_score(key, q_heads, cu_count, max_split, 1, occupancy, combine_base)
        num_split = 1
        for count in counts:
            if count % seq_len_replicat != 0:
                continue
            score, head_, num_split_, num_split_tail_ = get_streamk_config(key, count, cu_count, q_heads, max_split,
                                                                           combine_base, occupancy)
            # print(f"count={count}, score:{score:.3f} vs {min_score:.3f}, batch_head={head_ // seq_len_replicat}, num_split_={num_split_}, num_split_tail_={num_split_tail_}")
            if score < min_score:
                min_score = score
                num_split = num_split_
                num_split_tail = num_split_tail_
                batch_head = head_ // seq_len_replicat
    else:
        # splitK
        # when block_I = 32, occupancy is limited by lds as 2
        occupancy = 2
        max_split = 128
        num_split, score = get_best_split(key, cu_count, q_heads, max_split, combine_base, occupancy)

    # logger.info(f"Using best config for batch={batch}, seq_len={seq_len}, q_heads={q_heads}, cu_count={cu_count}: "
    #             f"block_I={block_I}, threads={threads}, num_stages={num_stages}, num_split={num_split}, num_split_tail={num_split_tail}, batch_head={batch_head}")
    return block_I, threads, num_stages, num_split, num_split_tail, batch_head


def get_config_fast(batch, seq_len, q_heads):
    block_M = 16 if q_heads <= 16 else 32
    key = batch * seq_len * ((q_heads + block_M - 1) // block_M)

    # Original logic: find smallest power of 2 where key * power > cu_count
    # Optimized: combine comparisons to reduce branching
    if key > cu_count:
        num_split = 1
    elif (key << 1) > cu_count:  # key * 2 > cu_count
        num_split = 2
    elif (key << 2) > cu_count:  # key * 4 > cu_count
        num_split = 4
    elif (key << 3) > cu_count:  # key * 8 > cu_count
        num_split = 8
    elif (key << 4) > cu_count:  # key * 16 > cu_count
        num_split = 16
    else:
        num_split = 32
    return (32, 256, 0, num_split, 0, batch)


def get_best_config(batch, seq_len, q_heads):
    # for now batch will always be 1
    if (seq_len <= 8 and batch <= 256) or (batch == 1 and seq_len <= 256):
        return get_best_streamk_config(batch, seq_len, q_heads)
    else:
        return get_config_fast(batch, seq_len, q_heads)


@functools.lru_cache(maxsize=64)
def _get_sparse_mla_fwd_kernel(heads, dim, tail_dim, topk, kv_group, sm_scale, block_I,
                               threads, num_stages, num_split, num_split_tail, kv_stride, dtype):
    """Cached kernel creation to avoid re-executing sparse_mla_fwd function body."""
    return sparse_mla_fwd(
        heads,
        dim,
        tail_dim,
        topk,
        dtype=dtype,
        num_split=num_split,
        num_split_tail=num_split_tail,
        kv_group=kv_group,
        sm_scale=sm_scale,
        is_causal=True,
        block_I=block_I,
        num_stages=num_stages,
        threads=threads,
        kv_stride=kv_stride)


def sparse_mla_fwd_interface(q,
                             kv,
                             indices,
                             kv_stride=1,
                             sm_scale=None,
                             d_v=512,
                             dtype="float16"):
    is_causal = True
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()
    batch, seq_len, heads, dim_plus_tail_dim = q.shape
    _, seq_len_kv, kv_group, _ = kv.shape

    dim = d_v
    tail_dim = dim_plus_tail_dim - dim
    assert kv.shape[0] == batch
    _, _, _, topk = indices.shape
    assert indices.shape == (batch, seq_len, kv_group, topk)

    # Auto-configure parameters using get_best_config
    block_I, threads, num_stages, num_split, num_split_tail, batch_head = get_best_config(batch, seq_len, heads)

    # Use cached kernel creation to avoid re-executing sparse_mla_fwd function body
    kernel = _get_sparse_mla_fwd_kernel(
        heads, dim, tail_dim, topk, kv_group, sm_scale, block_I,
        threads, num_stages, num_split, num_split_tail, kv_stride, dtype)

    return kernel, num_split, num_split_tail, batch_head


def tilelang_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
) -> torch.Tensor:
    """
    TileLang sparse MLA forward pass interface.

    Args:
        q: Query tensor of shape (S, H, DQK)
        kv: Key-Value tensor of shape (SKV, HKV, DQK)
        indices: Indices tensor of shape (S, HKV, topk)
        sm_scale: Softmax scale factor
        d_v: Value dimension (default: 512)

    Returns:
        Output tensor of shape (B=1, S, H, d_v)
    """
    assert q.dim() == 3 and kv.dim() == 3 and indices.dim() == 3
    # Infer dtype from input tensor
    if q.dtype == torch.bfloat16:
        dtype_str = "bfloat16"
    elif q.dtype == torch.float16:
        dtype_str = "float16"
    else:
        raise ValueError(f"Unsupported dtype: {q.dtype}, only bfloat16 and float16 are supported")

    # Get output shape
    B = 1
    S, H, _ = q.shape
    # Call sparse_mla_fwd_interface to get kernel
    tilelang_kernel, num_split, num_split_tail, batch_head = sparse_mla_fwd_interface(
        q.unsqueeze(0), kv.unsqueeze(0), indices.unsqueeze(0),
        kv_stride=1,
        sm_scale=sm_scale,
        d_v=d_v,
        dtype=dtype_str
    )

    intermediate_dtype = torch.float16
    # intermediate_dtype = q.dtype
    tl_out = torch.empty((B, S, H, d_v), dtype=q.dtype, device=q.device)
    # Allocate intermediate tensors and execute kernel
    if num_split_tail > 0:
        assert B > batch_head, "B must be greater than batch_head"
        glse = torch.empty((batch_head, S, num_split, H), dtype=intermediate_dtype, device=q.device)
        output_partial = torch.empty((batch_head, S, num_split, H, d_v), dtype=intermediate_dtype, device=q.device)
        glse_tail = torch.empty((B - batch_head, S, num_split_tail, H), dtype=intermediate_dtype, device=q.device)
        output_partial_tail = torch.empty((B - batch_head, S, num_split_tail, H, d_v), dtype=intermediate_dtype,
                                          device=q.device)
        tilelang_kernel(
            q.unsqueeze(0), kv.unsqueeze(0), indices.unsqueeze(0), glse, output_partial, glse_tail, output_partial_tail,
            tl_out)
    else:
        glse = torch.empty((B, S, num_split, H), dtype=intermediate_dtype, device=q.device)
        output_partial = torch.empty((B, S, num_split, H, d_v), dtype=intermediate_dtype, device=q.device)
        tilelang_kernel(
            q.unsqueeze(0), kv.unsqueeze(0), indices.unsqueeze(0), glse, output_partial, tl_out)

    return tl_out


def flash_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    TileLang sparse MLA forward pass with inference framework compatible interface.

    This function provides the same interface as flash_mla_sparse_fwd_torch but uses
    the optimized TileLang kernel for better performance.

    Args:
        q: Query tensor of shape (S, H, DQK) where S is sequence length, H is num heads,
           DQK is query/key dimension (default: 576 for DeepSeek, 256 for GLM-5.1)
        kv: Key-Value tensor of shape (SKV, HKV, DQK) where HKV is KV groups (usually 1)
        indices: Sparse indices tensor of shape (S, HKV, topk) with values in [0, SKV-1]
                 or -1 for invalid positions
        sm_scale: Softmax scale factor, typically 1/sqrt(DQK)
        d_v: Value/output dimension (default: 512)
        attn_sink: Optional attention sink tensor of shape (H,) for sink attention
        topk_length: Optional tensor of shape (S,) specifying valid topk count per position

    Returns:
        output: Output tensor of shape (S, H, d_v)
        max_logits: Max logits tensor of shape (S, H) for numerical stability
        lse: Log-sum-exp tensor of shape (S, H) for backward pass
    """
    s_q, h_q, d_qk = q.shape
    s_kv, h_kv, _ = kv.shape
    topk = indices.shape[2]
    device = q.device
    dtype = q.dtype

    # Infer dtype string for kernel
    if dtype == torch.bfloat16:
        dtype_str = "bfloat16"
    elif dtype == torch.float16:
        dtype_str = "float16"
    else:
        raise ValueError(f"Unsupported dtype: {dtype}, only bfloat16 and float16 are supported")

    q_batch = q  # (B, S, H, DQK) or (S, H, DQK)
    kv_batch = kv  # (B, S, HKV, DQK) or (S, HKV, DQK)
    indices_batch = indices  # (B, S, HKV, topk) or (S, HKV, topk)

    # Fallback to 3D tensors: add batch dim
    if q.dim() == 3:
        q_batch = q.unsqueeze(0)
        kv_batch = kv.unsqueeze(0)
        indices_batch = indices.unsqueeze(0)

    B = q_batch.shape[0]

    # Get kernel and configuration
    tilelang_kernel, num_split, num_split_tail, batch_head = sparse_mla_fwd_interface(
        q_batch, kv_batch, indices_batch,
        kv_stride=1,
        sm_scale=sm_scale,
        d_v=d_v,
        dtype=dtype_str
    )

    # Allocate output and intermediate tensors
    intermediate_dtype = torch.float16
    tl_out = torch.empty((B, s_q, h_q, d_v), dtype=dtype, device=device)

    if num_split_tail > 0:
        assert B > batch_head, "B must be greater than batch_head"
        glse = torch.empty((batch_head, s_q, num_split, h_q), dtype=intermediate_dtype, device=device)
        output_partial = torch.empty((batch_head, s_q, num_split, h_q, d_v), dtype=intermediate_dtype, device=device)
        glse_tail = torch.empty((B - batch_head, s_q, num_split_tail, h_q), dtype=intermediate_dtype, device=device)
        output_partial_tail = torch.empty((B - batch_head, s_q, num_split_tail, h_q, d_v), dtype=intermediate_dtype,
                                          device=device)
        tilelang_kernel(
            q_batch, kv_batch, indices_batch, glse, output_partial, glse_tail, output_partial_tail, tl_out)
    else:
        glse = torch.empty((B, s_q, num_split, h_q), dtype=intermediate_dtype, device=device)
        output_partial = torch.empty((B, s_q, num_split, h_q, d_v), dtype=intermediate_dtype, device=device)
        tilelang_kernel(
            q_batch, kv_batch, indices_batch, glse, output_partial, tl_out)

    # Compute lse and max_logits from glse
    # glse shape: (B, S, num_split, H)
    lse_tensor = glse  # (B, S, num_split, H)

    if num_split == 1:
        lse_tensor = lse_tensor.squeeze(2)  # (B, S, H)
        max_logits = lse_tensor.clone()
    else:
        lse_tensor, _ = lse_tensor.max(dim=2)  # (B, S, H)
        max_logits = lse_tensor.clone()

    lse_tensor = lse_tensor.to(dtype)

    output = tl_out  # (B, S, H, d_v)
    lse = lse_tensor  # (B, S, H)

    # Squeeze batch dim if input was 3D
    if q.dim() == 3:
        output = output.squeeze(0)
        lse = lse.squeeze(0)
        max_logits = max_logits.squeeze(0)

    # Apply attention sink if provided
    if attn_sink is not None:
        sink_factor = 1.0 / (1.0 + torch.exp(attn_sink.unsqueeze(0) - lse.float()) + 1e-9)
        output = output * sink_factor.unsqueeze(-1).to(dtype)

    return output, max_logits, lse


def flash_mla_sparse_fwd_torch(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    PyTorch reference implementation of sparse MLA forward pass.

    This implementation uses PyTorch operations for correctness verification
    and as a fallback when TileLang kernel is not available.

    Args:
        q: Query tensor of shape (S, H, DQK)
        kv: Key-Value tensor of shape (SKV, HKV, DQK)
        indices: Sparse indices tensor of shape (S, HKV, topk)
        sm_scale: Softmax scale factor
        d_v: Value dimension (default: 512)
        attn_sink: Optional attention sink tensor of shape (H,)
        topk_length: Optional tensor of shape (S,) for valid topk per position

    Returns:
        output: Output tensor of shape (S, H, d_v)
        max_logits: Max logits tensor of shape (S, H)
        lse: Log-sum-exp tensor of shape (S, H)
    """
    s_q, h_q, d_qk = q.shape
    s_kv, h_kv, _ = kv.shape
    topk = indices.shape[2]
    device = q.device
    dtype = q.dtype

    heads_per_kv = h_q // h_kv if h_kv > 0 else h_q
    qk_rope_head_dim = 64
    qk_nope_head_dim = d_qk - qk_rope_head_dim

    valid_mask = indices >= 0
    safe_indices = torch.where(valid_mask, indices, torch.zeros_like(indices))
    batch_idx = torch.arange(h_kv, device=device).view(1, h_kv, 1)

    gathered_kv = kv[safe_indices, batch_idx, :]
    expanded_kv = gathered_kv.unsqueeze(2).expand(s_q, h_kv, heads_per_kv, topk, d_qk).reshape(s_q, h_q, topk, d_qk)
    expanded_mask = valid_mask.unsqueeze(2).expand(s_q, h_kv, heads_per_kv, topk).reshape(s_q, h_q, topk)

    if topk_length is not None:
        range_tensor = torch.arange(topk, device=device).view(1, 1, topk)
        expanded_mask = expanded_mask & (range_tensor < topk_length.view(s_q, 1, 1))

    kv_nope = expanded_kv[..., :qk_nope_head_dim]
    kv_rope = expanded_kv[..., qk_nope_head_dim:]

    scores = (torch.matmul(q[..., :qk_nope_head_dim].unsqueeze(2), kv_nope.transpose(-1, -2)).squeeze(2) + \
              torch.matmul(q[..., qk_nope_head_dim:].unsqueeze(2), kv_rope.transpose(-1, -2)).squeeze(2)) * sm_scale

    scores = torch.where(expanded_mask, scores, torch.full_like(scores, float('-inf'), dtype=dtype)).to(torch.float32)

    max_logits, _ = torch.max(scores, dim=-1)
    shifted_scores = scores - max_logits.unsqueeze(-1)
    exp_scores = torch.exp(shifted_scores)
    exp_scores = torch.where(expanded_mask, exp_scores, torch.zeros_like(exp_scores))
    sum_exp = torch.sum(exp_scores, dim=-1)

    lse = max_logits + torch.log(sum_exp + 1e-9)
    attn_probs = (exp_scores / (sum_exp.unsqueeze(-1) + 1e-9)).to(dtype)

    context_layer = torch.matmul(attn_probs.unsqueeze(2), kv_nope).squeeze(2)

    if context_layer.shape[-1] < d_v:
        padding = torch.zeros((*context_layer.shape[:-1], d_v - context_layer.shape[-1]), dtype=dtype, device=device)
        output = torch.cat([context_layer, padding], dim=-1)
    else:
        output = context_layer[..., :d_v]

    if attn_sink is not None:
        sink_factor = 1.0 / (1.0 + torch.exp(attn_sink.unsqueeze(0) - lse) + 1e-9)
        output = output * sink_factor.unsqueeze(-1).to(dtype)

    return output, max_logits.to(dtype), lse.to(dtype)


def ref_sparse_mla_fwd_interface(q, kv, indices, output_dtype, q_start_s_index=0, kv_stride=1, sm_scale=None,
                                 is_casual=True):
    q = q.unsqueeze(0)
    kv = kv.unsqueeze(0)
    indices = indices.unsqueeze(0)
    q = q.float()
    kv = kv.float()
    indices = indices.transpose(1, 2)
    b, sq, h, dim_q = q.shape
    b, sk, g, _ = kv.shape

    assert kv.shape[-1] == 576, "you should assign dim otherwise"
    dim = 512
    k = kv
    v = kv[..., :dim]

    b, _, _, dim_v = v.shape
    g_index = g
    h_index = h // g
    compressed_casual_mask = torch.arange(
        q_start_s_index, sq + q_start_s_index, dtype=torch.int32,
        device="cuda").view(-1, 1) >= torch.arange(
        kv_stride - 1, sk * kv_stride, kv_stride, dtype=torch.int32, device="cuda").view(1, -1)

    indices = torch.where(indices >= 0, indices, sk)
    mask = q.new_zeros(b, g_index, sq, sk + 1, dtype=torch.bool).scatter(3, indices.long(), 1)
    mask = mask[..., :-1]
    mask = mask & compressed_casual_mask.view(1, 1, sq, sk)
    mask[:, :, :kv_stride - 1, 0] = True
    mask = mask.view(b, g_index, 1, sq, sk)

    q = q.view(b, sq, g, -1, dim_q)
    score = torch.einsum("bmghd,bngd->bghmn", q, k)
    sm_scale = dim_q ** -0.5 if sm_scale is None else sm_scale
    score = score.masked_fill(~mask, float("-inf")).mul(sm_scale)
    p = score.softmax(dim=-1)
    p = p.view(b, g_index, h_index, -1, sq, sk)
    p = p.view(b, g, -1, sq, sk)
    o = torch.einsum("bghmn,bngd->bmghd", p.type(v.dtype), v)
    o = o.reshape(b, sq, h, dim_v)
    return o.to(output_dtype)


def assert_tensors_similar(a, b, eps=1e-2, name="tensor"):
    """Assert that two tensors are similar within tolerance."""
    assert a.shape == b.shape, f"{name} shape mismatch: {a.shape} vs {b.shape}"
    if a.dtype != b.dtype:
        a = a.float()
        b = b.float()
    diff = (a - b).abs().max().item()
    rel_diff = (a - b).abs().max() / (b.abs().max() + 1e-8)
    assert diff < eps, f"{name} max diff: {diff}, rel_diff: {rel_diff}"
    print(f"  {name}: max_diff={diff:.6f}, rel_diff={rel_diff:.6f} [PASS]")


def test_sparse_mla_fwd(
    B=1,
    S=512,
    H=128,
    HKV=1,
    DQK=576,
    DV=512,
    topk=1024,
    dtype=torch.bfloat16,
    check_correctness=True,
    warmup=25,
    rep=100,
):
    """
    Test sparse MLA forward pass with correctness verification and performance benchmark.

    Args:
        B: Batch size (must be 1 for now)
        S: Sequence length
        H: Number of query heads
        HKV: Number of KV groups
        DQK: Query/Key dimension (including tail)
        DV: Value dimension
        topk: Number of top-k indices
        dtype: Data type
        check_correctness: Whether to verify correctness against reference
        warmup: Number of warmup iterations
        rep: Number of benchmark repetitions
    """
    torch.random.manual_seed(42)

    # Create input tensors
    q = torch.randn((B, S, H, DQK), dtype=dtype, device="cuda")
    kv = torch.randn((B, S, HKV, DQK), dtype=dtype, device="cuda")

    # Create indices (simulate causal sparse attention)
    indices = torch.full((B, S, HKV, topk), -1, dtype=torch.int32, device="cuda")
    for b in range(B):
        for t in range(S):
            for g in range(HKV):
                # Sample valid indices from causal positions
                valid_positions = min(t + 1, S)
                if valid_positions >= topk:
                    sampled_idx = torch.randperm(valid_positions, device="cuda")[:topk]
                else:
                    sampled_idx = torch.arange(valid_positions, device="cuda")
                    padding = torch.zeros(topk - valid_positions, dtype=torch.long, device="cuda") - 1
                    sampled_idx = torch.cat([sampled_idx, padding])
                indices[b, t, g, :topk] = sampled_idx[:topk]

    sm_scale = (1.0 / DQK) ** 0.5

    print(f"\n{'=' * 60}")
    print(f"Sparse MLA Forward Test")
    print(f"{'=' * 60}")
    print(f"Config: B={B}, S={S}, H={H}, HKV={HKV}, DQK={DQK}, DV={DV}, topk={topk}")
    print(f"dtype={dtype}")

    # Get kernel and configuration
    kernel, num_split, num_split_tail, batch_head = sparse_mla_fwd_interface(
        q, kv, indices, sm_scale=sm_scale, d_v=DV,
        dtype="bfloat16" if dtype == torch.bfloat16 else "float16"
    )

    print(f"Kernel config: num_split={num_split}, num_split_tail={num_split_tail}, batch_head={batch_head}")

    # Allocate output tensor
    tl_out = torch.empty((B, S, H, DV), dtype=dtype, device="cuda")

    # Execute kernel
    intermediate_dtype = torch.float16
    if num_split_tail > 0:
        assert B > batch_head
        glse = torch.empty((batch_head, S, num_split, H), dtype=intermediate_dtype, device="cuda")
        output_partial = torch.empty((batch_head, S, num_split, H, DV), dtype=intermediate_dtype, device="cuda")
        glse_tail = torch.empty((B - batch_head, S, num_split_tail, H), dtype=intermediate_dtype, device="cuda")
        output_partial_tail = torch.empty((B - batch_head, S, num_split_tail, H, DV), dtype=intermediate_dtype,
                                          device="cuda")
        kernel(q, kv, indices, glse, output_partial, glse_tail, output_partial_tail, tl_out)
    else:
        glse = torch.empty((B, S, num_split, H), dtype=intermediate_dtype, device="cuda")
        output_partial = torch.empty((B, S, num_split, H, DV), dtype=intermediate_dtype, device="cuda")
        kernel(q, kv, indices, glse, output_partial, tl_out)

    # Correctness verification
    if check_correctness:
        print("\n--- Correctness Verification ---")
        ref_out = ref_sparse_mla_fwd_interface(
            q.squeeze(0), kv.squeeze(0), indices.squeeze(0),
            output_dtype=dtype, sm_scale=sm_scale
        )
        assert_tensors_similar(tl_out.squeeze(0), ref_out.squeeze(0), eps=1e-2, name="TileLang vs PyTorch einsum")

        # FlashMLA correctness comparison
        if HAS_FLASH_MLA:
            fm_out, _, _ = flash_mla.flash_mla_sparse_fwd(
                q.squeeze(0), kv.squeeze(0), indices.squeeze(0), sm_scale=sm_scale, d_v=DV)
            assert_tensors_similar(tl_out.squeeze(0), fm_out, eps=1e-2, name="TileLang vs FlashMLA")
        else:
            print("  FlashMLA: not available, skipping correctness check")

    # Performance benchmark
    from tilelang.profiler import do_bench

    def tilelang_fn():
        if num_split_tail > 0:
            kernel(q, kv, indices, glse, output_partial, glse_tail, output_partial_tail, tl_out)
        else:
            kernel(q, kv, indices, glse, output_partial, tl_out)

    def pytorch_einsum_fn():
        return ref_sparse_mla_fwd_interface(q.squeeze(0), kv.squeeze(0), indices.squeeze(0),
                                            output_dtype=dtype, sm_scale=sm_scale)

    def pytorch_matmul_fn():
        return flash_mla_sparse_fwd_torch(q.squeeze(0), kv.squeeze(0), indices.squeeze(0),
                                          sm_scale=sm_scale, d_v=DV)

    print("\n--- Performance Benchmark ---")

    # TileLang benchmark
    tl_time = do_bench(tilelang_fn, warmup=warmup, rep=rep)
    print(f"TileLang:     {tl_time:.3f} ms")

    # FlashMLA benchmark
    if HAS_FLASH_MLA:
        def flash_mla_fn():
            return flash_mla.flash_mla_sparse_fwd(
                q.squeeze(0), kv.squeeze(0), indices.squeeze(0), sm_scale=sm_scale, d_v=DV)
        try:
            fm_time = do_bench(flash_mla_fn, warmup=warmup, rep=rep)
            print(f"FlashMLA:     {fm_time:.3f} ms")
            speedup_fm = fm_time / tl_time
            print(f"Speedup vs FlashMLA: {speedup_fm:.2f}x")
        except torch.OutOfMemoryError:
            print("FlashMLA:     OOM (skipped)")
            fm_time = None
    else:
        print("FlashMLA:     not available")
        fm_time = None

    # PyTorch einsum benchmark
    try:
        pt_einsum_time = do_bench(pytorch_einsum_fn, warmup=warmup // 2, rep=rep // 2)
        print(f"PyTorch einsum: {pt_einsum_time:.3f} ms")
    except torch.OutOfMemoryError:
        print("PyTorch einsum: OOM (skipped)")
        pt_einsum_time = None

    # PyTorch matmul benchmark
    try:
        pt_matmul_time = do_bench(pytorch_matmul_fn, warmup=warmup // 2, rep=rep // 2)
        print(f"PyTorch matmul: {pt_matmul_time:.3f} ms")
    except torch.OutOfMemoryError:
        print("PyTorch matmul: OOM (skipped)")
        pt_matmul_time = None

    # Calculate speedup
    if pt_einsum_time is not None:
        speedup_einsum = pt_einsum_time / tl_time
        print(f"Speedup vs PyTorch einsum: {speedup_einsum:.2f}x")
    if pt_matmul_time is not None:
        speedup_matmul = pt_matmul_time / tl_time
        print(f"Speedup vs PyTorch matmul: {speedup_matmul:.2f}x")

    # Compute throughput
    flops = B * S * (DQK + DV) * topk * 2 * H
    io_bytes = B * S * DQK * topk * 2 * 2  # Q and KV reads

    tflops = flops / (tl_time * 1e-3) / 1e12
    bandwidth = io_bytes / (tl_time * 1e-3) / 1e12

    print(f"\nThroughput: {tflops:.2f} TFLOPS")
    print(f"Bandwidth: {bandwidth:.2f} TB/s")

    return tl_out


def benchmark_comparison(
    configs=None,
    dtype=torch.bfloat16,
    warmup=25,
    rep=100,
):
    """
    Run performance comparison across multiple configurations.

    Args:
        configs: List of (name, params) tuples. If None, uses default configs.
        dtype: Data type
        warmup: Warmup iterations
        rep: Benchmark repetitions
    """
    if configs is None:
        configs = [
            ("S=512, H=128, topk=1024", {
                'B': 1, 'S': 512, 'H': 128, 'HKV': 1, 'DQK': 576, 'DV': 512, 'topk': 1024
            }),
            ("S=1024, H=128, topk=1024", {
                'B': 1, 'S': 1024, 'H': 128, 'HKV': 1, 'DQK': 576, 'DV': 512, 'topk': 1024
            }),
            ("S=2048, H=128, topk=2048", {
                'B': 1, 'S': 2048, 'H': 128, 'HKV': 1, 'DQK': 576, 'DV': 512, 'topk': 2048
            }),
            ("S=4096, H=128, topk=2048", {
                'B': 1, 'S': 4096, 'H': 128, 'HKV': 1, 'DQK': 576, 'DV': 512, 'topk': 2048
            }),
        ]

    print("\n" + "=" * 80)
    print("Sparse MLA Performance Comparison")
    print("=" * 80)

    results = []

    for name, params in configs:
        print(f"\n--- Config: {name} ---")

        B = params['B']
        S = params['S']
        H = params['H']
        HKV = params['HKV']
        DQK = params['DQK']
        DV = params['DV']
        topk = params['topk']

        torch.random.manual_seed(42)

        # Create inputs
        q = torch.randn((B, S, H, DQK), dtype=dtype, device="cuda")
        kv = torch.randn((B, S, HKV, DQK), dtype=dtype, device="cuda")

        # Create indices
        indices = torch.full((B, S, HKV, topk), -1, dtype=torch.int32, device="cuda")
        for b in range(B):
            for t in range(S):
                for g in range(HKV):
                    valid_positions = min(t + 1, S)
                    if valid_positions >= topk:
                        sampled_idx = torch.randperm(valid_positions, device="cuda")[:topk]
                    else:
                        sampled_idx = torch.arange(valid_positions, device="cuda")
                        padding = torch.zeros(topk - valid_positions, dtype=torch.long, device="cuda") - 1
                        sampled_idx = torch.cat([sampled_idx, padding])
                    indices[b, t, g, :topk] = sampled_idx[:topk]

        sm_scale = (1.0 / DQK) ** 0.5

        # Get kernel
        kernel, num_split, num_split_tail, batch_head = sparse_mla_fwd_interface(
            q, kv, indices, sm_scale=sm_scale, d_v=DV,
            dtype="bfloat16" if dtype == torch.bfloat16 else "float16"
        )

        # Allocate output
        tl_out = torch.empty((B, S, H, DV), dtype=dtype, device="cuda")
        intermediate_dtype = torch.float16

        if num_split_tail > 0:
            glse = torch.empty((batch_head, S, num_split, H), dtype=intermediate_dtype, device="cuda")
            output_partial = torch.empty((batch_head, S, num_split, H, DV), dtype=intermediate_dtype, device="cuda")
            glse_tail = torch.empty((B - batch_head, S, num_split_tail, H), dtype=intermediate_dtype, device="cuda")
            output_partial_tail = torch.empty((B - batch_head, S, num_split_tail, H, DV), dtype=intermediate_dtype,
                                              device="cuda")

            def tilelang_fn():
                kernel(q, kv, indices, glse, output_partial, glse_tail, output_partial_tail, tl_out)
        else:
            glse = torch.empty((B, S, num_split, H), dtype=intermediate_dtype, device="cuda")
            output_partial = torch.empty((B, S, num_split, H, DV), dtype=intermediate_dtype, device="cuda")

            def tilelang_fn():
                kernel(q, kv, indices, glse, output_partial, tl_out)

        from tilelang.profiler import do_bench

        # Benchmark TileLang
        tl_time = do_bench(tilelang_fn, warmup=warmup, rep=rep)

        # Benchmark FlashMLA
        if HAS_FLASH_MLA:
            def flash_mla_fn():
                return flash_mla.flash_mla_sparse_fwd(
                    q.squeeze(0), kv.squeeze(0), indices.squeeze(0), sm_scale=sm_scale, d_v=DV)
            try:
                fm_time = do_bench(flash_mla_fn, warmup=warmup, rep=rep)
            except torch.OutOfMemoryError:
                fm_time = None
        else:
            fm_time = None

        # Benchmark PyTorch matmul
        def pytorch_fn():
            return flash_mla_sparse_fwd_torch(q.squeeze(0), kv.squeeze(0), indices.squeeze(0),
                                              sm_scale=sm_scale, d_v=DV)

        try:
            pt_time = do_bench(pytorch_fn, warmup=warmup // 2, rep=rep // 2)
        except torch.OutOfMemoryError:
            pt_time = None

        speedup_pt = pt_time / tl_time if pt_time is not None else float('inf')
        speedup_fm = fm_time / tl_time if fm_time is not None else float('inf')

        # Compute metrics
        flops = B * S * (DQK + DV) * topk * 2 * H
        tflops = flops / (tl_time * 1e-3) / 1e12

        results.append({
            'name': name,
            'tl_time': tl_time,
            'fm_time': fm_time,
            'pt_time': pt_time,
            'speedup_pt': speedup_pt,
            'speedup_fm': speedup_fm,
            'tflops': tflops,
        })

        fm_str = f"{fm_time:.3f} ms" if fm_time is not None else ("N/A" if not HAS_FLASH_MLA else "OOM")
        pt_str = f"{pt_time:.3f} ms" if pt_time is not None else "OOM"
        print(f"  TileLang: {tl_time:.3f} ms | FlashMLA: {fm_str} | PyTorch: {pt_str} | "
              f"vsFM: {speedup_fm:.2f}x | vsPT: {speedup_pt:.2f}x | TFLOPS: {tflops:.2f}")

    # Summary table
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"{'Config':<30} {'TileLang':<12} {'FlashMLA':<12} {'PyTorch':<12} {'vsFM':<8} {'vsPT':<8} {'TFLOPS':<10}")
    print("-" * 80)
    for r in results:
        fm_str = f"{r['fm_time']:.3f}" if r['fm_time'] is not None else ("N/A" if not HAS_FLASH_MLA else "OOM")
        print(f"{r['name']:<30} {r['tl_time']:<12.3f} {fm_str:<12} {r['pt_time']:<12.3f} "
              f"{r['speedup_fm']:<8.2f} {r['speedup_pt']:<8.2f} {r['tflops']:<10.2f}")

    return results


def benchmark_flash_mla_sparse_fwd(
    configs=None,
    dtype=torch.bfloat16,
    warmup=25,
    rep=100,
):
    """
    Benchmark flash_mla_sparse_fwd (TileLang) vs flash_mla_sparse_fwd_torch (PyTorch).

    Uses the inference framework compatible interface with (S, H, DQK) 3D inputs.
    """

    if configs is None:
        configs = [
            ("S=512, H=128, topk=1024", {
                'S': 512, 'H': 128, 'HKV': 1, 'DQK': 576, 'DV': 512, 'topk': 1024
            }),
            ("S=1024, H=128, topk=1024", {
                'S': 1024, 'H': 128, 'HKV': 1, 'DQK': 576, 'DV': 512, 'topk': 1024
            }),
            ("S=2048, H=128, topk=2048", {
                'S': 2048, 'H': 128, 'HKV': 1, 'DQK': 576, 'DV': 512, 'topk': 2048
            }),
            ("S=4096, H=128, topk=2048", {
                'S': 4096, 'H': 128, 'HKV': 1, 'DQK': 576, 'DV': 512, 'topk': 2048
            }),
            ("S=512, H=64, topk=512", {
                'S': 512, 'H': 64, 'HKV': 1, 'DQK': 576, 'DV': 512, 'topk': 512
            }),
            ("S=1024, H=64, topk=512", {
                'S': 1024, 'H': 64, 'HKV': 1, 'DQK': 576, 'DV': 512, 'topk': 512
            }),
        ]

    print("\n" + "=" * 80)
    print("flash_mla_sparse_fwd Performance Comparison")
    print("=" * 80)

    results = []

    for name, params in configs:
        print(f"\n--- Config: {name} ---")

        S = params['S']
        H = params['H']
        HKV = params['HKV']
        DQK = params['DQK']
        DV = params['DV']
        topk = params['topk']

        torch.random.manual_seed(42)

        q = torch.randn((S, H, DQK), dtype=dtype, device="cuda")
        kv = torch.randn((S, HKV, DQK), dtype=dtype, device="cuda")

        indices = torch.full((S, HKV, topk), -1, dtype=torch.int32, device="cuda")
        for t in range(S):
            for g in range(HKV):
                valid_positions = min(t + 1, S)
                if valid_positions >= topk:
                    sampled_idx = torch.randperm(valid_positions, device="cuda")[:topk]
                else:
                    sampled_idx = torch.arange(valid_positions, device="cuda")
                    padding = torch.zeros(topk - valid_positions, dtype=torch.long, device="cuda") - 1
                    sampled_idx = torch.cat([sampled_idx, padding])
                indices[t, g, :topk] = sampled_idx[:topk]

        sm_scale = (1.0 / DQK) ** 0.5

        from tilelang.profiler import do_bench

        def tl_fn():
            return flash_mla_sparse_fwd(q, kv, indices, sm_scale, d_v=DV)

        def pt_fn():
            return flash_mla_sparse_fwd_torch(q, kv, indices, sm_scale, d_v=DV)

        tl_time = do_bench(tl_fn, warmup=warmup, rep=rep)

        # FlashMLA benchmark
        if HAS_FLASH_MLA:
            def fm_fn():
                return flash_mla.flash_mla_sparse_fwd(q, kv, indices, sm_scale, d_v=DV)
            try:
                fm_time = do_bench(fm_fn, warmup=warmup, rep=rep)
            except torch.OutOfMemoryError:
                fm_time = None
        else:
            fm_time = None

        try:
            pt_time = do_bench(pt_fn, warmup=warmup // 2, rep=rep // 2)
        except torch.OutOfMemoryError:
            pt_time = None

        speedup_fm = fm_time / tl_time if fm_time is not None else float('inf')
        speedup_pt = pt_time / tl_time if pt_time is not None else float('inf')
        flops = S * (DQK + DV) * topk * 2 * H
        tflops = flops / (tl_time * 1e-3) / 1e12

        results.append({
            'name': name,
            'tl_time': tl_time,
            'fm_time': fm_time,
            'pt_time': pt_time,
            'speedup_fm': speedup_fm,
            'speedup_pt': speedup_pt,
            'tflops': tflops,
        })

        fm_str = f"{fm_time:.3f} ms" if fm_time is not None else ("N/A" if not HAS_FLASH_MLA else "OOM")
        pt_str = f"{pt_time:.3f} ms" if pt_time is not None else "OOM"
        print(f"  TileLang: {tl_time:.3f} ms | FlashMLA: {fm_str} | PyTorch: {pt_str} | "
              f"vsFM: {speedup_fm:.2f}x | vsPT: {speedup_pt:.2f}x | TFLOPS: {tflops:.2f}")

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"{'Config':<30} {'TileLang':<12} {'FlashMLA':<12} {'PyTorch':<12} {'vsFM':<8} {'vsPT':<8} {'TFLOPS':<10}")
    print("-" * 80)
    for r in results:
        fm_str = f"{r['fm_time']:.3f}" if r['fm_time'] is not None else ("N/A" if not HAS_FLASH_MLA else "OOM")
        pt_str = f"{r['pt_time']:.3f}" if r['pt_time'] is not None else "OOM"
        print(f"{r['name']:<30} {r['tl_time']:<12.3f} {fm_str:<12} {pt_str:<12} "
              f"{r['speedup_fm']:<8.2f} {r['speedup_pt']:<8.2f} {r['tflops']:<10.2f}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sparse MLA forward pass test and benchmark")
    parser.add_argument("--check", action="store_true", default=False,
                        help="Enable correctness verification (default: False)")
    parser.add_argument("--benchmark", action="store_true", default=True,
                        help="Run benchmark comparison (default: True)")
    parser.add_argument("--no-benchmark", action="store_false", dest="benchmark",
                        help="Skip benchmark comparison")
    parser.add_argument("--S", type=int, default=512, help="Sequence length (default: 512)")
    parser.add_argument("--H", type=int, default=128, help="Number of heads (default: 128)")
    parser.add_argument("--topk", type=int, default=1024, help="Top-k (default: 1024)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16"], help="Data type (default: bfloat16)")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    # Run single test
    test_sparse_mla_fwd(
        B=1,
        S=args.S,
        H=args.H,
        HKV=1,
        DQK=576,
        DV=512,
        topk=args.topk,
        dtype=dtype,
        check_correctness=args.check,
    )

    # Run benchmark comparison
    if args.benchmark:
        print("\n\n")
        benchmark_comparison(dtype=dtype)
        print("\n\n")
        benchmark_flash_mla_sparse_fwd(dtype=dtype)

