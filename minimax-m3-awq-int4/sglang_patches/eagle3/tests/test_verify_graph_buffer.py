#!/usr/bin/env python3
"""验证修复: forward_extend verify 分支的 cu_seqlens/seq_lens/prefix_lens 在
graph capture/replay 下不再依赖临时 tensor (extend_seq_lens).

根因: 旧实现用 forward_batch.extend_seq_lens (forward_extend 里 torch.full 临时新建)
算 cu_seqlens/seq_lens. capture 后 forward_batch (局部变量) 被 GC, extend_seq_lens
内存释放/复用, replay 时 graph 读老地址 → 垃圾 → VMFault (bs=1 碰巧不炸, bs>=阈值必崩).

修复: verify 分支改用
  - forward_batch.seq_lens (graph buffer, replay_prepare 填充, 地址稳定)
  - self.num_draft_tokens D (Python int 常量)
  - torch.arange(0, (bs+1)*D, D) (Python int 输入, graph-safe, 同 triton qo_indptr)

本测试模拟 graph 流程, 验证修复后:
  1. capture 时构建的 cu_seqlens/seq_lens/prefix_lens 地址不依赖临时 tensor
  2. replay 时改 forward_batch.seq_lens buffer 的值, 输出正确跟随更新
  3. 对比 eager (真实值) vs graph replay, 结果一致

直接测 backend 的 forward_extend 不现实 (需要完整 ForwardBatch/KVPool), 这里测
修复的 *核心逻辑* (cu_seqlens/seq_lens/prefix_lens 构建) 在 graph 下的行为.
"""
import os
import sys
import torch

sys.path.insert(0, "/usr/local/lib/python3.10/dist-packages")

torch.manual_seed(0)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
D = 4  # num_draft_tokens


def build_verify_tensors_old(seq_lens_buffer, extend_seq_lens_temp):
    """旧实现 (graph-unsafe): 依赖 extend_seq_lens 临时 tensor."""
    raw_seq_lens = seq_lens_buffer.to(torch.int32)
    prefix_lens = raw_seq_lens
    seq_lens = raw_seq_lens + extend_seq_lens_temp.to(torch.int32)
    cu_seqlens = torch.cat([
        torch.zeros(1, dtype=torch.int32, device=extend_seq_lens_temp.device),
        extend_seq_lens_temp.to(torch.int32).cumsum(0).to(torch.int32),
    ])
    return cu_seqlens, seq_lens, prefix_lens


def build_verify_tensors_new(seq_lens_buffer, D_const):
    """新实现 (graph-safe): 只用 graph buffer + Python int + arange."""
    raw_seq_lens = seq_lens_buffer.to(torch.int32)
    bs = raw_seq_lens.shape[0]
    device = raw_seq_lens.device
    cu_seqlens = torch.arange(0, (bs + 1) * D_const, step=D_const, dtype=torch.int32, device=device)
    prefix_lens = raw_seq_lens
    seq_lens = raw_seq_lens + D_const
    return cu_seqlens, seq_lens, prefix_lens


def test_graph_replay():
    """模拟 graph: capture (dummy seq_lens=1) + replay (真实 seq_lens=500)."""
    print("=" * 70)
    print("验证: 修复后 cu_seqlens/seq_lens 在 graph capture/replay 下正确")
    print("=" * 70)

    bs = 16
    real_prefix = 500

    # graph buffer: seq_lens (capture dummy=1, replay 真实=500). 模拟 sglang buffers.seq_lens
    seq_lens_buffer = torch.full((bs,), 1, dtype=torch.int32, device=DEVICE)  # capture: dummy=1

    # ---- 旧实现: 依赖临时 extend_seq_lens ----
    print("\n[旧实现] 依赖 extend_seq_lens 临时 tensor:")
    extend_seq_lens_temp = torch.full((bs,), D, dtype=torch.int32, device=DEVICE)
    # capture
    cu_old_cap, sl_old_cap, pl_old_cap = build_verify_tensors_old(seq_lens_buffer, extend_seq_lens_temp)
    print(f"  capture: cu_seqlens={cu_old_cap.tolist()[:5]}... seq_lens[:3]={sl_old_cap[:3].tolist()} "
          f"(prefix_lens=seq_lens_buffer=1, seq_lens=1+D=5)")
    # 模拟 capture 后 extend_seq_lens 临时 tensor 被 GC + 内存复用 (填垃圾)
    extend_seq_lens_temp.fill_(999)  # 模拟内存复用, 值变垃圾
    seq_lens_buffer.fill_(real_prefix)  # replay: graph buffer 被填真实值
    # replay: 旧实现重放时读 extend_seq_lens 老地址 → 垃圾 999
    cu_old_rep, sl_old_rep, pl_old_rep = build_verify_tensors_old(seq_lens_buffer, extend_seq_lens_temp)
    print(f"  replay:  cu_seqlens={cu_old_rep.tolist()[:5]}... seq_lens[:3]={sl_old_rep[:3].tolist()}")
    print(f"  ↑ extend_seq_lens 被复用成 999 → seq_lens={real_prefix}+999={real_prefix+999} (错! 应={real_prefix+D})")
    old_correct = (sl_old_rep[0].item() == real_prefix + D)
    print(f"  旧实现 replay 正确? {old_correct} (应为 {real_prefix+D}, 实得 {sl_old_rep[0].item()})")

    # ---- 新实现: 只用 graph buffer + Python int ----
    print("\n[新实现] 只用 seq_lens buffer + Python int D + arange:")
    seq_lens_buffer.fill_(1)  # 重置 capture dummy
    # capture
    cu_new_cap, sl_new_cap, pl_new_cap = build_verify_tensors_new(seq_lens_buffer, D)
    print(f"  capture: cu_seqlens={cu_new_cap.tolist()[:5]}... seq_lens[:3]={sl_new_cap[:3].tolist()}")
    # replay: 只改 seq_lens_buffer (graph buffer), 无临时 tensor 依赖
    seq_lens_buffer.fill_(real_prefix)
    cu_new_rep, sl_new_rep, pl_new_rep = build_verify_tensors_new(seq_lens_buffer, D)
    print(f"  replay:  cu_seqlens={cu_new_rep.tolist()[:5]}... seq_lens[:3]={sl_new_rep[:3].tolist()}")
    new_correct = (sl_new_rep[0].item() == real_prefix + D)
    cu_correct = (cu_new_rep.tolist() == list(range(0, (bs+1)*D, D)))
    print(f"  新实现 replay seq_lens 正确? {new_correct} (应={real_prefix+D}, 实得 {sl_new_rep[0].item()})")
    print(f"  新实现 cu_seqlens 正确? {cu_correct} (=[0,4,8,...,{bs*D}])")

    # ---- 关键: 在真实 graph capture/replay 下验证 ----
    print("\n[真实 graph capture/replay 验证]")
    seq_lens_buffer.fill_(1)  # capture dummy
    g = torch.cuda.CUDAGraph()
    # capture: 新实现
    with torch.cuda.graph(g):
        cu_g, sl_g, pl_g = build_verify_tensors_new(seq_lens_buffer, D)
    print(f"  capture OK: cu_g.shape={tuple(cu_g.shape)} sl_g.shape={tuple(sl_g.shape)}")
    # replay: 改 buffer 值
    seq_lens_buffer.fill_(real_prefix)
    g.replay()
    torch.cuda.synchronize()
    print(f"  replay: sl_g[:3]={sl_g[:3].tolist()} (应=[{real_prefix+D}]*3)")
    print(f"  replay: cu_g[:5]={cu_g[:5].tolist()} (应=[0,4,8,12,16])")
    graph_correct = (sl_g[0].item() == real_prefix + D) and (cu_g[0].item() == 0) and (cu_g[1].item() == D)
    print(f"  graph replay 正确? {graph_correct}")

    print("\n" + "=" * 70)
    ok = (not old_correct) and new_correct and cu_correct and graph_correct
    if ok:
        print("PASS: 旧实现 graph-unsafe (复现根因), 新实现 graph-safe (修复有效)")
    else:
        print(f"FAIL: old_correct={old_correct}(应False) new_correct={new_correct} cu_correct={cu_correct} graph_correct={graph_correct}")
    print("=" * 70)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("SKIP: 需要 CUDA/HIP")
        sys.exit(0)
    test_graph_replay()
