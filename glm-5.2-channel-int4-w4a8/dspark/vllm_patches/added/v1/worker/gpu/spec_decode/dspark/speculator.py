# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSpark speculator backported to the preinstalled vLLM V2 model runner.

DSpark (semi-autoregressive parallel drafting) drafts a block of
``num_speculative_tokens`` tokens in one parallel backbone forward, then injects
intra-block dependency with a lightweight sequential Markov head.

This is a STANDALONE reimplementation. The upstream DSpark/DFlash speculators
subclass ``DraftModelSpeculator`` and depend on V2-speculator infrastructure
(``ModelState``, ``dispatch_cg_and_sync_dp``, ``DFlashCudaGraphManager``,
``seq_lens_cpu_upper_bound``, context-parallel ``cp_local_slot``, ...) that the
preinstalled vLLM 0.15.1 does not ship. Instead, this class mirrors the
preinstalled ``EagleSpeculator`` interface that the preinstalled V2
``GPUModelRunner`` drives:

  * ``set_attn(kv_cache_config, attn_groups, block_tables)``  (3-arg)
  * ``run_model(num_tokens, attn_metadata, slot_mappings, num_tokens_across_dp)``
  * ``propose(input_batch, last_hidden_states, aux_hidden_states,
              num_sampled, num_rejected, last_sampled, next_prefill_tokens,
              temperature, seeds)``                          (9-arg)
  * ``capture_model()``

The DSpark-specific algorithm (parallel query-block forward + sequential Markov
sampling) is ported from the upstream ``DSparkSpeculator`` / ``_prepare_dflash_inputs_kernel``;
the input-preparation triton kernel is inlined here and simplified for the
single-device, no-context-parallel case (cp_size == 1), so ``cp_local_slot``
collapses to a plain ``block_id * block_size + pos % block_size``.
"""

import os
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    build_slot_mappings_by_layer,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.dp_utils import make_num_tokens_across_dp
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.spec_decode.dspark.utils import load_dspark_model

logger = init_logger(__name__)


def _get_parallel_drafting_token_id(hf_config) -> int:
    """Resolve the mask token id used for parallel drafting slots.

    Backported from upstream vllm.v1.worker.gpu.spec_decode.utils (which the
    preinstalled vLLM does not ship). Checks dflash_config.mask_token_id,
    top-level mask_token_id, dspark_noise_token_id, pard_token, ptd_token_id.
    """
    dflash_config = getattr(hf_config, "dflash_config", None) or {}
    if "mask_token_id" in dflash_config:
        return int(dflash_config["mask_token_id"])
    if getattr(hf_config, "mask_token_id", None) is not None:
        return int(hf_config.mask_token_id)
    if hasattr(hf_config, "dspark_noise_token_id"):
        return int(hf_config.dspark_noise_token_id)
    if hasattr(hf_config, "pard_token"):
        return int(hf_config.pard_token)
    if hasattr(hf_config, "ptd_token_id"):
        return int(hf_config.ptd_token_id)
    raise ValueError(
        "Model config must specify `dflash_config.mask_token_id`,"
        " `mask_token_id`, `dspark_noise_token_id`, `pard_token`, or"
        " `ptd_token_id` for parallel drafting."
    )


class DSparkSpeculator:
    """Standalone DSpark speculator for the preinstalled V2 model runner."""

    _speculator_name = "DSpark"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.vllm_config = vllm_config
        self.device = device

        self.speculative_config = vllm_config.speculative_config
        assert self.speculative_config is not None
        self.method = self.speculative_config.method
        self.num_speculative_steps = self.speculative_config.num_speculative_tokens
        self.draft_model_config = self.speculative_config.draft_model_config

        self.scheduler_config = vllm_config.scheduler_config
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.max_model_len = vllm_config.model_config.max_model_len
        self.draft_max_seq_len = self.max_model_len
        self.hidden_size = self.draft_model_config.get_hidden_size()
        # Target vocab (for the final sampled logits / d2t scatter).
        self.vocab_size = vllm_config.model_config.get_vocab_size()
        self.dtype = vllm_config.model_config.dtype

        # DSpark samples from the anchor: each request emits exactly N =
        # num_speculative_steps query tokens (anchor + N-1 mask), and every
        # query position is a prediction (sample_pos = query_pos + 1).
        self.sample_from_anchor = getattr(
            self.draft_model_config.hf_config, "sample_from_anchor", True
        )
        if self.sample_from_anchor:
            self.num_query_per_req = self.num_speculative_steps
        else:
            self.num_query_per_req = 1 + self.num_speculative_steps

        self.parallel_drafting_token_id = _get_parallel_drafting_token_id(
            self.draft_model_config.hf_config
        )

        # Reduced draft vocab (probabilistic drafting scatters draft logits into
        # target vocab rows). Set up in load_model.
        self._draft_topk: int | None = getattr(
            self.draft_model_config.hf_config, "dspark_draft_topk", None
        )
        self._d2t_scatter_index: torch.Tensor | None = None
        self._draft_scatter_buf: torch.Tensor | None = None

        # Adaptive verification (confidence head) is opt-in; the preinstalled
        # SpeculativeConfig has no such field, so default to fixed-count
        # verification. The confidence head is still exercised when present so
        # its buffers stay sized, but the values are unused without the flag.
        self.enable_adaptive_verification = bool(
            getattr(self.speculative_config, "enable_adaptive_verification", False)
        )

        self.input_buffers = InputBuffers(
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            device=device,
        )
        # Persistent draft seq_lens buffer. The draft attends over
        # (target_seq_len + num_query_per_req) tokens, so the draft attention
        # metadata needs seq_lens = target_seq_lens + step. _build_draft_attn
        # metadata writes into this persistent buffer (instead of cloning) so
        # the metadata references a stable tensor that CUDA-graph replay can
        # read after we update it in-place before each replay.
        self.draft_seq_lens = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=device
        )
        # Draft backbone consumes mean-pooled target aux hidden states combined
        # to hidden_size via main_proj (DSpark does not reuse the MTP buffer).
        self.hidden_states = torch.zeros(
            self.max_num_tokens, self.hidden_size, dtype=self.dtype, device=device
        )

        self.idx_mapping = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=device
        )
        self.temperature = torch.zeros(
            self.max_num_reqs, dtype=torch.float32, device=device
        )
        self.seeds = torch.zeros(
            self.max_num_reqs, dtype=torch.int64, device=device
        )
        self.draft_tokens = torch.zeros(
            self.max_num_reqs,
            self.num_speculative_steps,
            dtype=torch.int64,
            device=device,
        )
        self.draft_token_confidence_probs = torch.empty_like(
            self.draft_tokens, dtype=torch.float32
        )

        # Per-(req, position) sampling buffers, flattened (req, step).
        max_num_sampled_tokens = self.max_num_reqs * self.num_speculative_steps
        self.sample_indices = torch.zeros(
            max_num_sampled_tokens, dtype=torch.int64, device=device
        )
        self.sample_pos = torch.zeros(
            max_num_sampled_tokens, dtype=torch.int64, device=device
        )
        # -1 marks an inert sampling row so padded rows do not scatter into a
        # live request during CUDA-graph replay.
        self.sample_idx_mapping = torch.full(
            (max_num_sampled_tokens,), -1, dtype=torch.int32, device=device
        )

        self._step_cols = torch.arange(
            self.num_speculative_steps, dtype=torch.int32, device=device
        )
        # Anchor query offset per request = req_idx * num_query_per_req.
        self._anchor_idx = (
            torch.arange(self.max_num_reqs, dtype=torch.int64, device=device)
            * self.num_query_per_req
        )
        self._arange_reqs = torch.arange(
            self.max_num_reqs + 1, dtype=torch.int32, device="cpu"
        )

        # Context positions / slots for the K/V precompute (filled by the
        # prepare-inputs kernel, consumed by precompute_and_store_context_kv).
        self.context_positions = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device=device
        )
        self.context_slot_mappings = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device=device
        )

        # Lazily initialized in set_attn once we know the draft's KV group id.
        self.draft_kv_cache_group_id: int = -1
        self.kv_cache_config: KVCacheConfig | None = None
        # attn_groups[kv_cache_group_id] = list[AttentionGroup]; each sub-group
        # has its own metadata builder so target MLA-attn / DSA-indexer / draft
        # layers (which share one uniform KV-cache group but use different
        # attention backends) each get backend-correct metadata.
        self.attn_groups: list | None = None
        self.block_tables: BlockTables | None = None
        self.draft_attn_layer_names: set[str] = set()

        self.cudagraph_manager = DSparkCudaGraphManager(vllm_config, device)
        # Captured draft attention metadata / slot mappings, keyed by padded
        # num_reqs (cudagraph size). Built once in capture_model and reused on
        # every replay; their underlying tensors are the speculator's
        # persistent buffers, updated in-place before each replay.
        self._captured_draft_attn_metadata: dict[int, Any] = {}
        self._captured_draft_slot_mappings: dict[int, Any] = {}

        # --- training-data capture (off unless VLLM_DSPARK_CAPTURE_DIR set) ---
        # When enabled, each propose() dumps the target's per-step
        # aux-hidden-states (concatenated to [num_tokens, 30720]) plus the
        # corresponding token ids and absolute positions as cap-<n>.pt, the
        # format consumed by the DSpark finetune script (dspark_finetune.py).
        self._capture_dir = os.environ.get("VLLM_DSPARK_CAPTURE_DIR", "")
        self._capture_n = 0
        self._capture_rank = int(os.environ.get("RANK", "0"))
        if self._capture_dir:
            os.makedirs(self._capture_dir, exist_ok=True)
            logger.warning(
                "DSpark capture ENABLED -> %s (rank %d)",
                self._capture_dir, self._capture_rank)

    def _maybe_capture(
        self,
        input_batch: InputBatch,
        aux_hidden_states: list[torch.Tensor] | None,
    ) -> None:
        """Dump one cap-<n>.pt per propose() when capture is enabled.

        Each file holds {aux: [T,30720] bf16, input_ids: [T], positions: [T]}
        for the T target tokens in this step, matching the finetune loader.
        Only rank 0 writes (single-rank capture is enough for training data).
        """
        if not self._capture_dir or self._capture_rank != 0:
            return
        if not aux_hidden_states:
            return
        num_tokens = input_batch.num_tokens
        # Concatenate the per-layer aux hidden states along the feature dim
        # (layers [2,20,39,58,75] x 6144 = 30720), as the draft's fc() expects.
        aux = torch.cat(
            [h[:num_tokens].to(torch.bfloat16) for h in aux_hidden_states],
            dim=-1,
        ).cpu()
        ids = input_batch.input_ids[:num_tokens].cpu()
        pos = input_batch.positions[:num_tokens].cpu()
        path = os.path.join(
            self._capture_dir, f"cap-{self._capture_n:08d}.pt")
        self._capture_n += 1
        torch.save({"aux": aux, "input_ids": ids, "positions": pos}, path)

    def load_model(self, target_model: nn.Module) -> None:
        self.model = load_dspark_model(target_model, self.vllm_config)

        # Reduced draft vocab: precompute draft->target column map and a scratch
        # buffer to scatter draft logits into target vocab before sampling.
        if self.model.draft_id_to_target_id is not None:
            d2t = self.model.draft_id_to_target_id
            self._d2t_scatter_index = (
                torch.arange(d2t.shape[0], device=d2t.device) + d2t
            )
            self._draft_scatter_buf = torch.full(
                (self.max_num_reqs, self.vocab_size),
                float("-inf"),
                dtype=self.dtype,
                device=self.device,
            )

        if self.enable_adaptive_verification and self.model.model.confidence_head is None:
            raise ValueError(
                "Adaptive verification needs a DSpark checkpoint with a confidence "
                "head, and this one has none. Pass enable_adaptive_verification=false "
                "in the speculative config to verify a fixed number of drafts instead."
            )

    # ------------------------------------------------------------------ #
    # Attention wiring
    # ------------------------------------------------------------------ #
    def set_attn(
        self,
        kv_cache_config: KVCacheConfig,
        attn_groups: list,
        block_tables: BlockTables,
    ) -> None:
        self.kv_cache_config = kv_cache_config
        self.attn_groups = attn_groups
        self.block_tables = block_tables

        # Discover the draft's attention layer names from the loaded draft model.
        if hasattr(self.model, "get_draft_kv_cache_layer_names"):
            self.draft_attn_layer_names = set(
                self.model.get_draft_kv_cache_layer_names()
            )
        # Map draft layer names -> the kv-cache group index that owns them.
        self.draft_kv_cache_group_id = -1
        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            if any(ln in self.draft_attn_layer_names for ln in group.layer_names):
                self.draft_kv_cache_group_id = gid
                break
        # Fallback: if discovery failed, assume the last group is the draft
        # (the draft layers were appended to static_forward_context after the
        # target layers, so they land in a trailing group).
        if self.draft_kv_cache_group_id < 0:
            self.draft_kv_cache_group_id = len(kv_cache_config.kv_cache_groups) - 1
            logger.warning(
                "DSpark: could not locate draft attention layers by name; "
                "assuming kv-cache group %d is the draft. Draft layer names: %s",
                self.draft_kv_cache_group_id,
                sorted(self.draft_attn_layer_names),
            )

    # ------------------------------------------------------------------ #
    # Parallel backbone forward
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def run_model(
        self,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
    ) -> torch.Tensor:
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            num_tokens_across_dp=num_tokens_across_dp,
            slot_mapping=slot_mappings,
        ):
            last_hidden_states = self.model(
                input_ids=self.input_buffers.input_ids[:num_tokens],
                positions=self.input_buffers.positions[:num_tokens],
                inputs_embeds=None,
            )
        return last_hidden_states

    # ------------------------------------------------------------------ #
    # Sequential Markov sampling (ported from upstream DSparkSpeculator)
    # ------------------------------------------------------------------ #
    def _sample_logits(
        self,
        logits: torch.Tensor,
        idx_map: torch.Tensor,
        sample_pos: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        # The preinstalled SpeculativeConfig has no draft_sample_method field
        # and the DSpark checkpoint ships a greedy proposal config, so we draft
        # greedily: argmax in draft-vocab space, then remap to target vocab via
        # map_draft_to_target (identity for full-vocab drafts, +d2t otherwise).
        # This matches upstream's `draft_logits is None` greedy branch and keeps
        # the sampled distribution exact for greedy rejection sampling.
        return self.model.map_draft_to_target(logits.argmax(dim=-1))

    def _sample_sequential(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        if self._draft_topk is not None:
            self._sample_sequential_topk(num_reqs, head_hidden)
            return

        n_spec = self.num_speculative_steps
        num_sample = num_reqs * n_spec
        # Per-(req, position) head hidden, ordered (req, step).
        sample_hidden = head_hidden[self.sample_indices[:num_sample]]
        # Draft-vocab logits; sampled ids are remapped to target vocab below.
        base_logits = self.model.compute_draft_logits(sample_hidden)
        vocab_size = base_logits.shape[-1]
        base_logits = base_logits.view(num_reqs, n_spec, vocab_size)

        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)
        confidence_markov_embeds = []

        # Anchor (bonus) token per request = the input id at query offset 0.
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]

        for i in range(n_spec):
            # Sequential stage: Markov bias from the previously sampled token.
            markov_embed = self.model.markov_embed(prev)
            if self.enable_adaptive_verification:
                confidence_markov_embeds.append(markov_embed)
            bias = self.model.markov_bias(markov_embed)
            logits_i = base_logits[:, i] + bias
            draft_sampled_i = self._sample_logits(
                logits_i, idx_map[:, i], sample_pos[:, i], i
            )
            self.draft_tokens[:num_reqs, i] = draft_sampled_i
            prev = draft_sampled_i

        if self.enable_adaptive_verification:
            confidence = self.model.compute_confidence(
                sample_hidden,
                torch.stack(confidence_markov_embeds, dim=1).flatten(0, 1),
            )
            self.draft_token_confidence_probs[:num_reqs] = confidence.view(
                num_reqs, n_spec
            )

    def _sample_sequential_topk(
        self, num_reqs: int, head_hidden: torch.Tensor
    ) -> None:
        assert self._draft_topk is not None
        n_spec = self.num_speculative_steps
        num_sample = num_reqs * n_spec
        sample_hidden = head_hidden[self.sample_indices[:num_sample]]
        base_logits = self.model.compute_draft_logits(sample_hidden)
        base_logits = base_logits.view(num_reqs, n_spec, -1)
        base_values, draft_indices = base_logits.topk(self._draft_topk, dim=-1)
        base_logits.fill_(float("-inf"))
        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)
        confidence_markov_embeds = []
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]

        for i in range(n_spec):
            markov_embed = self.model.markov_embed(prev)
            if self.enable_adaptive_verification:
                confidence_markov_embeds.append(markov_embed)
            logits_i = self.model.apply_markov_bias_gathered(
                markov_embed,
                base_logits[:, i],
                base_values[:, i],
                draft_indices[:, i],
            )
            draft_sampled_i = self._sample_logits(
                logits_i, idx_map[:, i], sample_pos[:, i], i
            )
            self.draft_tokens[:num_reqs, i] = draft_sampled_i
            prev = draft_sampled_i

        if self.enable_adaptive_verification:
            confidence = self.model.compute_confidence(
                sample_hidden,
                torch.stack(confidence_markov_embeds, dim=1).flatten(0, 1),
            )
            self.draft_token_confidence_probs[:num_reqs] = confidence.view(
                num_reqs, n_spec
            )

    # ------------------------------------------------------------------ #
    # Full draft step (captured under CUDA graph): forward + Markov sampling
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
    ) -> None:
        head_hidden = self.run_model(
            num_tokens_padded, attn_metadata, slot_mappings, num_tokens_across_dp
        )
        self._sample_sequential(num_reqs, head_hidden)

    # ------------------------------------------------------------------ #
    # CUDA graph capture
    # ------------------------------------------------------------------ #
    def capture_model(self) -> None:
        # Capture the full draft step (_generate_draft: parallel backbone
        # forward + sequential Markov sampling) keyed by padded num_reqs.
        # Unlike Eagle, DSpark's draft attends over (target_seq_len +
        # num_query_per_req) tokens, so we CANNOT reuse prepare_inputs_to_capture
        # (which builds metadata with the target's seq_lens). Instead we build
        # the draft's own metadata via _build_draft_attn_metadata, which writes
        # draft seq_lens into the persistent self.draft_seq_lens buffer. The
        # captured metadata references the speculator's persistent buffers
        # (query_start_loc, draft_seq_lens, block_tables, slot_mappings); before
        # each replay we update those buffers in-place so the graph sees the new
        # request set. Padded rows are inert (sample_idx_mapping=-1).
        if self.num_speculative_steps == 1:
            return
        if not self.draft_attn_layer_names:
            # No draft attention layers to graph over.
            return
        mgr = self.cudagraph_manager
        if not mgr.cudagraph_sizes:
            return
        logger.info("Capturing model for DSpark speculator...")

        # Set up dummy persistent-buffer contents so the metadata built at
        # capture references valid (if dummy) data. prepare_dspark_inputs will
        # overwrite these before each real replay.
        num_query_per_req = self.num_query_per_req

        def _capture_one(padded_num_reqs: int) -> None:
            num_query_tokens = padded_num_reqs * num_query_per_req
            # query_start_loc: uniform layout [0, N, 2N, ..., padded*N].
            qs = (
                torch.arange(
                    padded_num_reqs + 1, dtype=torch.int32, device=self.device
                )
                * num_query_per_req
            )
            self.input_buffers.query_start_loc[: padded_num_reqs + 1] = qs
            # Dummy seq_lens (will be overwritten before replay).
            self.input_buffers.seq_lens[:padded_num_reqs] = num_query_per_req
            self.input_buffers.seq_lens[padded_num_reqs:] = 0
            # Build draft metadata into the persistent buffers. Use
            # num_reqs=padded_num_reqs so the whole padded batch is inert-active
            # (padded rows have seq_len 0 + step).
            attn_metadata = self._build_draft_attn_metadata(
                num_reqs=padded_num_reqs,
                num_reqs_padded=padded_num_reqs,
                num_tokens_padded=num_query_tokens,
                step=num_query_per_req,
            )
            slot_mappings_tensor = self.block_tables.slot_mappings[
                :, :num_query_tokens
            ]
            slot_mappings_by_layer = build_slot_mappings_by_layer(
                slot_mappings_tensor, self.kv_cache_config
            )
            num_tokens_across_dp = make_num_tokens_across_dp(
                self.vllm_config.parallel_config.data_parallel_size,
                num_query_tokens,
            )

            # Warm up.
            self._generate_draft(
                padded_num_reqs,
                num_query_tokens,
                attn_metadata,
                slot_mappings_by_layer,
                num_tokens_across_dp,
            )
            # Capture.
            assert padded_num_reqs not in mgr.graphs
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, mgr.pool):
                self._generate_draft(
                    padded_num_reqs,
                    num_query_tokens,
                    attn_metadata,
                    slot_mappings_by_layer,
                    num_tokens_across_dp,
                )
            mgr.graphs[padded_num_reqs] = graph
            # Stash the captured metadata / slot mappings for this size; their
            # underlying tensors are the persistent buffers updated per replay.
            self._captured_draft_attn_metadata[padded_num_reqs] = attn_metadata
            self._captured_draft_slot_mappings[padded_num_reqs] = (
                slot_mappings_by_layer
            )

        from vllm.v1.worker.gpu.cudagraph_utils import capture_graphs
        capture_graphs(
            mgr.cudagraph_sizes,
            self.device,
            lambda size, **_: _capture_one(size),
        )

    # ------------------------------------------------------------------ #
    # Build the draft's own attention metadata (query block forward).
    # ------------------------------------------------------------------ #
    def _build_draft_attn_metadata(
        self,
        num_reqs: int,
        num_reqs_padded: int,
        num_tokens_padded: int,
        step: int,
    ) -> dict[str, Any] | None:
        if not self.draft_attn_layer_names:
            return None
        assert self.attn_groups is not None
        assert self.block_tables is not None
        assert self.kv_cache_config is not None

        # Uniform query layout: query_start_loc[i] = min(i, num_reqs) * N.
        query_start_loc_cpu = (
            torch.clamp(self._arange_reqs[: num_reqs_padded + 1], max=num_reqs)
            * self.num_query_per_req
        )
        # Draft seq_lens = target seq_len + step (the query tokens appended).
        # Write into the persistent self.draft_seq_lens buffer (NOT a clone) so
        # the returned metadata references a stable tensor. This lets a
        # captured CUDA graph read updated seq_lens after we rewrite this buffer
        # in-place before each replay; the eager path is unaffected since the
        # metadata is rebuilt every call anyway.
        self.draft_seq_lens[:num_reqs] = torch.clamp(
            self.input_buffers.seq_lens[:num_reqs] + step,
            max=self.max_model_len,
        )
        # Pad the tail (rows >= num_reqs) with 0 so inert padded rows don't
        # read out-of-range KV slots during graph replay.
        if num_reqs_padded > num_reqs:
            self.draft_seq_lens[num_reqs:num_reqs_padded] = 0
        draft_seq_lens = self.draft_seq_lens[:num_reqs_padded]
        block_tables = [
            x[:num_reqs_padded] for x in self.block_tables.input_block_tables
        ]
        slot_mappings = self.block_tables.slot_mappings[:, :num_tokens_padded]
        attn_metadata = build_attn_metadata(
            attn_groups=self.attn_groups,
            num_reqs=num_reqs_padded,
            num_tokens=num_tokens_padded,
            query_start_loc_gpu=self.input_buffers.query_start_loc[
                : num_reqs_padded + 1
            ],
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=draft_seq_lens,
            max_seq_len=self.draft_max_seq_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=self.kv_cache_config,
        )
        return attn_metadata

    # ------------------------------------------------------------------ #
    # Entry point driven by the preinstalled V2 model runner (9-arg).
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        # [num_tokens, hidden_size]
        last_hidden_states: torch.Tensor,
        # num_layers x [num_tokens, hidden_size]
        aux_hidden_states: list[torch.Tensor] | None,
        # [num_reqs]
        num_sampled: torch.Tensor,
        # [num_reqs]
        num_rejected: torch.Tensor,
        # [max_num_reqs]
        last_sampled: torch.Tensor,
        # [max_num_reqs]
        next_prefill_tokens: torch.Tensor,
        # [max_num_reqs]
        temperature: torch.Tensor,
        # [max_num_reqs]
        seeds: torch.Tensor,
    ) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        num_target_tokens = input_batch.num_tokens
        num_query_tokens = num_reqs * self.num_query_per_req

        # Dump training data (no-op unless VLLM_DSPARK_CAPTURE_DIR is set).
        self._maybe_capture(input_batch, aux_hidden_states)

        # DSpark consumes mean-pooled target aux hidden states combined to
        # hidden_size via main_proj.
        if aux_hidden_states:
            hidden_states = self.model.combine_hidden_states(
                torch.cat(aux_hidden_states, dim=-1)
            )
        else:
            hidden_states = last_hidden_states
        self.hidden_states[:num_target_tokens].copy_(
            hidden_states[:num_target_tokens]
        )

        # Copy sampling state for the draft.
        idx_mapping = self.idx_mapping[:num_reqs]
        idx_mapping.copy_(input_batch.idx_mapping)
        self.temperature[:num_reqs].copy_(temperature[:num_reqs])
        self.seeds[:num_reqs].copy_(seeds[:num_reqs])

        assert self.block_tables is not None
        assert self.kv_cache_config is not None
        gid = self.draft_kv_cache_group_id
        assert gid >= 0

        # Prepare the draft's query/context inputs in one kernel launch.
        # The query slot mapping is written into the shared BlockTables buffer
        # at the draft group's row so the captured CUDA graph reads it at replay.
        query_slot_mapping = self.block_tables.slot_mappings[gid]
        prepare_dspark_inputs(
            self.input_buffers,
            query_slot_mapping,
            self.context_positions,
            self.context_slot_mappings,
            self.sample_indices,
            self.sample_pos,
            self.sample_idx_mapping,
            input_batch,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            self.block_tables.input_block_tables[gid],
            self.block_tables.block_sizes[gid],
            self.parallel_drafting_token_id,
            self.num_query_per_req,
            self.num_speculative_steps,
            self.max_num_reqs,
            self.max_num_tokens,
            self.max_model_len,
            self.sample_from_anchor,
        )

        # Pre-insert context K/V into the draft cache. Runs eagerly outside the
        # captured graph because the context shape varies per step.
        self.model.precompute_and_store_context_kv(
            self.hidden_states[:num_target_tokens],
            self.context_positions[:num_target_tokens],
            self.context_slot_mappings[:num_target_tokens],
        )

        # Draft slot mappings by layer (for the forward context).
        draft_slot_mappings_tensor = self.block_tables.slot_mappings[
            :, :num_query_tokens
        ]
        draft_slot_mappings_by_layer = build_slot_mappings_by_layer(
            draft_slot_mappings_tensor, self.kv_cache_config
        )

        # Build the draft's own attention metadata for the query-block forward.
        # In graph mode we key on padded num_reqs: rebuild metadata over the
        # padded batch (writing draft seq_lens into the persistent buffer) so the
        # captured graph -- which bound the SAME persistent buffers at capture
        # time -- reads the live request set at replay. Padded rows get
        # draft_seq_len 0 (inert) and sample_idx_mapping -1 (no scatter).
        mgr = self.cudagraph_manager
        cudagraph_size = mgr.get_cudagraph_size(num_reqs)
        if cudagraph_size is not None and cudagraph_size in mgr.graphs:
            padded = cudagraph_size
            num_query_tokens_padded = padded * self.num_query_per_req
            # Rebuild draft attention metadata over the padded batch into the
            # persistent buffers (draft_seq_lens, query_start_loc views).
            self._build_draft_attn_metadata(
                num_reqs=num_reqs,
                num_reqs_padded=padded,
                num_tokens_padded=num_query_tokens_padded,
                step=self.num_query_per_req,
            )
            mgr.run(padded)
        else:
            # Eager fallback (no captured graph for this batch size).
            draft_attn_metadata = self._build_draft_attn_metadata(
                num_reqs=num_reqs,
                num_reqs_padded=num_reqs,
                num_tokens_padded=num_query_tokens,
                step=self.num_query_per_req,
            )

            num_tokens_across_dp = make_num_tokens_across_dp(
                self.vllm_config.parallel_config.data_parallel_size,
                num_query_tokens,
            )

            # Run the draft step (eager here; CUDA graph replay handled below).
            self._generate_draft(
                num_reqs,
                num_query_tokens,
                draft_attn_metadata,
                draft_slot_mappings_by_layer,
                num_tokens_across_dp,
            )

        return self.draft_tokens[:num_reqs]


# ===========================================================================
# Input-preparation triton kernel (cp_size == 1 simplification of upstream
# _prepare_dflash_inputs_kernel).
# ===========================================================================
@triton.jit
def _prepare_dspark_inputs_kernel(
    # Outputs
    out_input_ids_ptr,
    out_query_positions_ptr,
    out_query_start_loc_ptr,
    out_seq_lens_ptr,
    out_query_slot_mapping_ptr,
    out_context_positions_ptr,
    out_context_slot_mapping_ptr,
    out_sample_indices_ptr,
    out_sample_pos_ptr,
    out_sample_idx_mapping_ptr,
    # Inputs from target batch
    target_positions_ptr,
    target_query_start_loc_ptr,
    idx_mapping_ptr,
    last_sampled_ptr,
    next_prefill_tokens_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    # Block table for slot mapping lookup.
    block_table_ptr,
    block_table_stride,
    # Scalars
    parallel_drafting_token_id,
    block_size,
    num_query_per_req,
    num_speculative_steps,
    max_num_reqs,
    max_num_tokens,
    max_model_len,
    SAMPLE_FROM_ANCHOR: tl.constexpr,
    PAD_SLOT_ID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    num_reqs = tl.num_programs(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)

    ctx_start = tl.load(target_query_start_loc_ptr + req_idx)
    ctx_end = tl.load(target_query_start_loc_ptr + req_idx + 1)
    num_ctx = ctx_end - ctx_start

    num_rejected = tl.load(num_rejected_ptr + req_idx)
    valid_ctx_end = ctx_end - num_rejected
    num_valid_ctx = valid_ctx_end - ctx_start

    num_sampled = tl.load(num_sampled_ptr + req_idx)
    if num_sampled > 0:
        bonus_token = tl.load(last_sampled_ptr + req_state_idx).to(tl.int32)
    else:
        # Chunked prefilling: splice in the next prefill token.
        bonus_token = tl.load(next_prefill_tokens_ptr + req_state_idx).to(tl.int32)

    last_valid_pos = tl.load(target_positions_ptr + valid_ctx_end - 1)
    query_base = req_idx * num_query_per_req

    j = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    is_ctx = j < num_ctx
    is_valid_ctx = j < num_valid_ctx
    is_query = (j >= num_valid_ctx) & (j < num_valid_ctx + num_query_per_req)
    query_off = j - num_valid_ctx

    # --- Context positions / slots (cp_size == 1) ---
    ctx_pos_idx = ctx_start + tl.where(is_ctx, j, 0)
    ctx_pos = tl.load(target_positions_ptr + ctx_pos_idx, mask=is_valid_ctx, other=0)
    ctx_block_num = ctx_pos // block_size
    ctx_block_num = tl.minimum(ctx_block_num, block_table_stride - 1)
    ctx_block_id = tl.load(
        block_table_ptr + req_idx * block_table_stride + ctx_block_num,
        mask=is_valid_ctx,
        other=0,
    ).to(tl.int64)
    # Block 0 is the null block; evicted sliding-window context can map to it.
    ctx_resident = is_valid_ctx & (ctx_block_id != 0)
    ctx_slot = tl.where(
        ctx_resident,
        ctx_block_id * block_size + (ctx_pos % block_size),
        PAD_SLOT_ID,
    )
    tl.store(out_context_positions_ptr + ctx_start + j, ctx_pos, mask=is_ctx)
    tl.store(out_context_slot_mapping_ptr + ctx_start + j, ctx_slot, mask=is_ctx)

    # --- Query positions / input_ids / slots ---
    query_pos = last_valid_pos + 1 + query_off
    query_idx = query_base + query_off
    is_bonus = is_query & (query_off == 0)
    input_id = tl.where(is_bonus, bonus_token, parallel_drafting_token_id)

    q_block_num = query_pos // block_size
    q_block_num = tl.minimum(q_block_num, block_table_stride - 1)
    q_block_id = tl.load(
        block_table_ptr + req_idx * block_table_stride + q_block_num,
        mask=is_query,
        other=0,
    ).to(tl.int64)
    q_resident = is_query & (q_block_id != 0)
    q_slot = tl.where(
        q_resident,
        q_block_id * block_size + (query_pos % block_size),
        PAD_SLOT_ID,
    )

    tl.store(out_input_ids_ptr + query_idx, input_id, mask=is_query)
    clamped_query_pos = tl.minimum(query_pos, max_model_len - 1)
    tl.store(out_query_positions_ptr + query_idx, clamped_query_pos, mask=is_query)
    tl.store(out_query_slot_mapping_ptr + query_idx, q_slot, mask=is_query)

    # --- Sample indices / positions / idx_mapping ---
    # When SAMPLE_FROM_ANCHOR (DSpark), sample at EVERY query position; each
    # position k predicts the NEXT token (sampled position = query_pos + 1).
    sample_off = 0 if SAMPLE_FROM_ANCHOR else 1
    is_sample = is_query & (query_off >= sample_off)
    sample_idx = req_idx * num_speculative_steps + (query_off - sample_off)
    sample_pos = query_pos + 1 if SAMPLE_FROM_ANCHOR else query_pos
    tl.store(out_sample_indices_ptr + sample_idx, query_idx, mask=is_sample)
    tl.store(out_sample_pos_ptr + sample_idx, sample_pos, mask=is_sample)
    tl.store(out_sample_idx_mapping_ptr + sample_idx, req_state_idx, mask=is_sample)

    if block_idx == 0:
        tl.store(out_query_start_loc_ptr + req_idx, query_base)
        tl.store(
            out_seq_lens_ptr + req_idx,
            tl.minimum(last_valid_pos + 1 + num_query_per_req, max_model_len),
        )
        if req_idx == num_reqs - 1:
            # Pad per-request buffers to max_num_reqs for CUDA graph safety.
            last_query_end = num_reqs * num_query_per_req
            for i in range(num_reqs, max_num_reqs + 1, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < max_num_reqs + 1
                tl.store(out_query_start_loc_ptr + block, last_query_end, mask=mask)
            for i in range(num_reqs, max_num_reqs, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < max_num_reqs
                tl.store(out_seq_lens_ptr + block, 0, mask=mask)
            pad_start = num_reqs * num_speculative_steps
            pad_end = max_num_reqs * num_speculative_steps
            for i in range(pad_start, pad_end, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < pad_end
                tl.store(out_sample_indices_ptr + block, 0, mask=mask)
                tl.store(out_sample_pos_ptr + block, 0, mask=mask)
                tl.store(out_sample_idx_mapping_ptr + block, -1, mask=mask)
            q_pad_start = num_reqs * num_query_per_req
            for i in range(q_pad_start, max_num_tokens, BLOCK_SIZE):
                block = i + tl.arange(0, BLOCK_SIZE)
                mask = block < max_num_tokens
                tl.store(out_query_slot_mapping_ptr + block, PAD_SLOT_ID, mask=mask)


def prepare_dspark_inputs(
    input_buffers: InputBuffers,
    query_slot_mapping: torch.Tensor,
    context_positions: torch.Tensor,
    context_slot_mapping: torch.Tensor,
    sample_indices: torch.Tensor,
    sample_pos: torch.Tensor,
    sample_idx_mapping: torch.Tensor,
    input_batch: InputBatch,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    last_sampled: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    parallel_drafting_token_id: int,
    num_query_per_req: int,
    num_speculative_steps: int,
    max_num_reqs: int,
    max_num_tokens: int,
    max_model_len: int,
    sample_from_anchor: bool = True,
) -> None:
    num_reqs = input_batch.num_reqs
    assert num_reqs > 0
    # Cover the longest per-request span (ctx + query).
    max_target_query_len = int(input_batch.num_scheduled_tokens.max())
    max_tokens_per_req = max_target_query_len + num_query_per_req
    BLOCK_SIZE = min(256, triton.next_power_of_2(max(1, max_tokens_per_req)))
    num_blocks = triton.cdiv(max_tokens_per_req, BLOCK_SIZE)
    _prepare_dspark_inputs_kernel[(num_reqs, num_blocks)](
        input_buffers.input_ids,
        input_buffers.positions,
        input_buffers.query_start_loc,
        input_buffers.seq_lens,
        query_slot_mapping,
        context_positions,
        context_slot_mapping,
        sample_indices,
        sample_pos,
        sample_idx_mapping,
        input_batch.positions,
        input_batch.query_start_loc,
        input_batch.idx_mapping,
        last_sampled,
        next_prefill_tokens,
        num_sampled,
        num_rejected,
        block_table,
        block_table.stride(0),
        parallel_drafting_token_id,
        block_size,
        num_query_per_req,
        num_speculative_steps,
        max_num_reqs,
        max_num_tokens,
        max_model_len,
        SAMPLE_FROM_ANCHOR=sample_from_anchor,
        PAD_SLOT_ID=PAD_SLOT_ID,
        BLOCK_SIZE=BLOCK_SIZE,
    )


# ===========================================================================
# CUDA graph manager (modeled on the preinstalled EagleCudaGraphManager).
# ===========================================================================
from vllm.v1.worker.gpu.cudagraph_utils import (  # noqa: E402
    capture_graphs,
    get_cudagraph_sizes,
)


class DSparkCudaGraphManager:
    """CUDA graph manager for DSpark's full draft step.

    Captures ``_generate_draft`` (parallel backbone forward + sequential Markov
    sampling) keyed by ``num_reqs``. The captured function signature matches
    Eagle's 4-arg ``generate_fn(num_tokens, attn_metadata, slot_mappings,
    num_tokens_across_dp)``; here ``num_tokens`` is ``num_reqs`` and the
    ``num_query_tokens`` is derived inside the speculator.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device = device

        self.max_model_len = vllm_config.model_config.max_model_len
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.dp_size = vllm_config.parallel_config.data_parallel_size
        self.compilation_config = vllm_config.compilation_config
        assert self.compilation_config is not None

        cudagraph_mode: CUDAGraphMode
        if self.compilation_config.cudagraph_mode is None:
            cudagraph_mode = CUDAGraphMode.NONE
        else:
            cudagraph_mode = self.compilation_config.cudagraph_mode

        self.cudagraph_mode = cudagraph_mode

        # DSpark's draft forward processes a FIXED num_query_per_req tokens per
        # request, so the captured graph shape is determined entirely by
        # num_reqs. We therefore key cudagraphs by num_reqs (NOT by the target's
        # token-count cudagraph_capture_sizes, which spec-decode rounds to
        # multiples of num_spec+1 and are not valid request counts). Build our
        # own request-count capture sizes, capped at max_num_reqs, and the
        # lookup dict {num_reqs -> padded_num_reqs}.
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.pool = torch.cuda.graph_pool_handle()
        # Speculator reference set at capture time so the replay wrapper can
        # rebuild per-replay draft attention metadata.
        self.speculator: DSparkSpeculator | None = None
        self.num_query_per_req: int = 1

        if cudagraph_mode == CUDAGraphMode.NONE:
            self.cudagraph_sizes: dict[int, int] = {}
            return

        # Request-count capture sizes: powers of two up to max_num_reqs, plus
        # max_num_reqs itself if not already present.
        sizes: list[int] = []
        s = 1
        while s <= self.max_num_reqs:
            sizes.append(s)
            s *= 2
        if not sizes or sizes[-1] != self.max_num_reqs:
            sizes.append(self.max_num_reqs)
        sizes = sorted(set(sizes))
        # {num_reqs -> padded_num_reqs}: every num_reqs rounds up to the next
        # captured size.
        self.cudagraph_sizes = {}
        for i in range(1, sizes[-1] + 1):
            for x in sizes:
                if i <= x:
                    self.cudagraph_sizes[i] = x
                    break

    def get_cudagraph_size(self, num_reqs: int) -> int | None:
        return self.cudagraph_sizes.get(num_reqs)

    def capture_graph(
        self,
        num_tokens: int,
        generate_fn: Any,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        attn_groups: list,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        from vllm.v1.worker.gpu.cudagraph_utils import prepare_inputs_to_capture

        num_reqs = min(num_tokens, self.max_num_reqs)
        num_query_tokens = num_reqs * self.num_query_per_req
        # DSpark's draft forward processes num_query_tokens, not num_reqs.
        # prepare_inputs_to_capture builds dummy metadata sized for num_reqs;
        # we override the token count the drafter sees via the speculator.
        attn_metadata, slot_mappings = prepare_inputs_to_capture(
            num_reqs,
            num_query_tokens,
            input_buffers,
            block_tables,
            attn_groups,
            self.max_model_len,
            kv_cache_config,
        )
        num_tokens_across_dp = make_num_tokens_across_dp(self.dp_size, num_query_tokens)

        # generate_fn = speculator._generate_draft, signature
        # (num_reqs, num_tokens_padded, attn_metadata, slot_mappings,
        #  num_tokens_across_dp). The first arg (num_reqs) is what we key on.
        def _wrapped(_num_reqs_arg, attn_md, sm, ntdp):
            generate_fn(
                num_reqs,
                num_query_tokens,
                attn_md,
                sm,
                ntdp,
            )

        # Warm up.
        _wrapped(num_reqs, attn_metadata, slot_mappings, num_tokens_across_dp)

        # Capture.
        assert num_tokens not in self.graphs
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, self.pool):
            _wrapped(num_reqs, attn_metadata, slot_mappings, num_tokens_across_dp)
        self.graphs[num_tokens] = graph

    @torch.inference_mode()
    def capture(
        self,
        generate_fn: Any,
        input_buffers: InputBuffers,
        block_tables: BlockTables | None,
        attn_groups: list | None,
        kv_cache_config: KVCacheConfig | None,
        num_query_per_req: int,
    ) -> None:
        assert block_tables is not None
        assert attn_groups is not None
        assert kv_cache_config is not None
        self.num_query_per_req = num_query_per_req
        # Padded sample rows must not scatter into a live request during capture.
        capture_graphs(
            self.cudagraph_sizes,
            self.device,
            self.capture_graph,
            generate_fn=generate_fn,
            input_buffers=input_buffers,
            block_tables=block_tables,
            attn_groups=attn_groups,
            kv_cache_config=kv_cache_config,
        )

    def run(self, num_tokens: int) -> None:
        assert num_tokens in self.graphs
        self.graphs[num_tokens].replay()
