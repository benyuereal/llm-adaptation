"""Low-overhead, opt-in runtime tracing for MiniMax-M3.

The tracer records statistics only; it never changes tensors or model control flow.
Enable with M3_TRACE=1. Each TP/PP process writes its own JSONL file.
"""
from __future__ import annotations

import json
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any, Optional


class M3RuntimeTrace:
    def __init__(self) -> None:
        # M3_TRACE_DIR is reliably inherited by sglang scheduler workers,
        # while M3_TRACE itself may be stripped. Treat a set trace dir as
        # the enable signal when M3_TRACE is absent.
        trace_dir_set = bool(os.getenv("M3_TRACE_DIR"))
        self.enabled = os.getenv("M3_TRACE", "").lower() in {"1", "true", "yes", "on"} or trace_dir_set
        self.max_forwards = int(os.getenv("M3_TRACE_MAX_FORWARDS", "20"))
        self.max_layers = int(os.getenv("M3_TRACE_MAX_LAYERS", "-1"))
        self.max_events = int(os.getenv("M3_TRACE_MAX_EVENTS", "10000"))
        self.save_tensors = os.getenv("M3_TRACE_SAVE_TENSORS", "0").lower() in {"1", "true", "yes"}
        self.tensor_ops = {x.strip() for x in os.getenv("M3_TRACE_TENSOR_OPS", "").split(",") if x.strip()}
        self.out_dir = Path(os.getenv("M3_TRACE_DIR", "/workspace/trace"))
        self.rank = os.getenv("RANK", os.getenv("LOCAL_RANK", "unknown"))
        self.tp_rank = os.getenv("TP_RANK", os.getenv("TENSOR_MODEL_PARALLEL_RANK", "unknown"))
        self.pp_rank = os.getenv("PP_RANK", os.getenv("PIPELINE_PARALLEL_RANK", "unknown"))
        self._rank_resolved = False
        self._lock = threading.Lock()
        self._forward_id = 0
        self._events = 0
        self._active_forward: Optional[int] = None
        self._fp = None
        if self.enabled:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            path = self.out_dir / f"trace.rank{self.rank}.pid{os.getpid()}.jsonl"
            self._fp = path.open("a", buffering=1, encoding="utf-8")
            self._write({"kind": "meta", "time": time.time(), "host": socket.gethostname(), "pid": os.getpid(),
                         "rank": self.rank, "tp_rank": self.tp_rank, "pp_rank": self.pp_rank,
                         "enabled": self.enabled, "max_forwards": self.max_forwards,
                         "m3_trace_env": os.getenv("M3_TRACE", "<unset>"),
                         "m3_trace_dir_env": os.getenv("M3_TRACE_DIR", "<unset>")})

    def _resolve_rank(self) -> None:
        if self._rank_resolved:
            return
        self._rank_resolved = True
        try:
            from sglang.srt.distributed import get_tensor_model_parallel_rank, get_pp_rank
            try:
                self.tp_rank = str(get_tensor_model_parallel_rank())
            except Exception:
                pass
            try:
                self.pp_rank = str(get_pp_rank())
            except Exception:
                pass
        except Exception:
            pass
        try:
            import torch
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                self.rank = str(torch.distributed.get_rank())
        except Exception:
            pass

    def begin_forward(self) -> Optional[int]:
        if not self.enabled or self._forward_id >= self.max_forwards:
            return None
        self._resolve_rank()
        self._forward_id += 1
        self._active_forward = self._forward_id
        self._events = 0
        self.event("forward", "begin", None, extra={"forward_id": self._active_forward})
        return self._active_forward

    def end_forward(self) -> None:
        if self._active_forward is not None:
            self.event("forward", "end", None)
        self._active_forward = None

    def _write(self, obj: dict[str, Any]) -> None:
        if self._fp is None:
            return
        with self._lock:
            self._fp.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")

    def event(self, op: str, stage: str, tensor: Any, *, layer: Optional[int] = None,
              extra: Optional[dict[str, Any]] = None) -> None:
        if not self.enabled or self._active_forward is None or self._events >= self.max_events:
            return
        if layer is not None and self.max_layers >= 0 and layer >= self.max_layers:
            return
        if tensor is None:
            stats: dict[str, Any] = {"shape": None, "dtype": None}
        else:
            try:
                import torch
                if not isinstance(tensor, torch.Tensor):
                    return
                stats = {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "numel": tensor.numel()}
                if tensor.numel():
                    x = tensor.detach()
                    if x.is_cuda:
                        x = x.float()
                    else:
                        x = x.float()
                    finite = torch.isfinite(x)
                    stats.update({"mean": float(x.mean().item()), "std": float(x.std(unbiased=False).item()),
                                  "min": float(x.min().item()), "max": float(x.max().item()),
                                  "l2": float(torch.linalg.vector_norm(x).item()),
                                  "nan": int(torch.isnan(x).sum().item()), "inf": int(torch.isinf(x).sum().item()),
                                  "finite": int(finite.sum().item())})
                    if self.save_tensors and (not self.tensor_ops or op in self.tensor_ops):
                        fname = f"f{self._active_forward:03d}.l{layer if layer is not None else 'x'}.{op}.{stage}.pt"
                        torch.save(x.cpu(), self.out_dir / fname)
            except Exception as exc:
                stats = {"trace_error": type(exc).__name__, "trace_error_message": str(exc)}
        obj = {"kind": "tensor", "time": time.time(), "rank": self.rank, "tp_rank": self.tp_rank,
               "pp_rank": self.pp_rank, "forward_id": self._active_forward, "layer": layer,
               "op": op, "stage": stage, **stats}
        if extra:
            obj.update(extra)
        self._write(obj)
        self._events += 1


TRACE = M3RuntimeTrace()


def trace_tensor(op: str, stage: str, tensor: Any, *, layer: Optional[int] = None,
                 extra: Optional[dict[str, Any]] = None) -> None:
    TRACE.event(op, stage, tensor, layer=layer, extra=extra)
