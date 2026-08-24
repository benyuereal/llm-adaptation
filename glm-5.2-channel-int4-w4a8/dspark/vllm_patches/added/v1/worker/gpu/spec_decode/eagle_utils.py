# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backported helpers for loading draft models (DFlash/DSpark).

Provides _should_share and get_target_lm_head, which the official vLLM keeps in
v1/worker/gpu/spec_decode/eagle/utils.py. The preinstalled vLLM has a flat
spec_decode/ layout (no eagle/ subdir), so they live here.
"""
import torch
import torch.nn as nn


def _should_share(eagle: nn.Module, flag: str, draft, target) -> bool:
    """Share when the draft has no own copy, or its copy matches the target."""
    if not getattr(eagle, flag, False) or draft is None:
        return True
    if target is None:
        return False
    # torch.equal on GPU allocates a bool mask the size of the input.
    # Use the faster GPU path when there is plenty of headroom;
    # otherwise compare on CPU.
    # Backported: the preinstalled torch (2.9) lacks
    # ``torch.accelerator.get_memory_info``; use ``torch.cuda.mem_get_info``
    # (returns (free, total)) and fall back to a CPU comparison if the API
    # is unavailable or fails.
    w = draft.weight
    compare_on_cpu = False
    if w.is_cuda:
        try:
            free, _ = torch.cuda.mem_get_info(w.device)
            compare_on_cpu = free < w.numel() * w.element_size() * 2
        except Exception:
            compare_on_cpu = True
    if compare_on_cpu:
        return torch.equal(w.cpu(), target.weight.cpu())
    return torch.equal(w, target.weight)


def get_target_lm_head(target_model: nn.Module, target_language_model: nn.Module):
    """The target's lm_head — from get_language_model() for
    *ForConditionalGeneration targets, else the top-level module."""
    return getattr(target_language_model, "lm_head", None) or getattr(
        target_model, "lm_head", None
    )
