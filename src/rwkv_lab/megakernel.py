"""Compiled low-latency execution backend for recurrent RWKV decoding.

The execution model is inspired by HazyResearch's Megakernels throughput
branch and TileRT's tile-level runtime:

* https://github.com/HazyResearch/Megakernels/tree/throughput
* https://github.com/tile-ai/TileRT

Those projects currently target their own model layouts, so RWKV-Lab does not
load their generated code or claim binary compatibility.  Instead this module
implements the equivalent boundary for native ``RWKV7Small`` checkpoints: a
single fused Triton program performs each RWKV-7 DPLR state transition, while
``torch.compile`` and a CUDA Graph cache remove Python dispatch and fuse the
surrounding fixed-shape execution plan.  Adoption remains fail-closed through
token parity, measured latency, and profiler-derived launch receipts.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Sequence

import torch

try:  # Triton is supplied by the CUDA PyTorch environment, not pyproject.toml.
    import triton
    import triton.language as tl
    _HAVE_TRITON = True
    _TRITON_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - depends on the installed CUDA stack
    triton = tl = None
    _HAVE_TRITON = False
    _TRITON_ERROR = exc


if _HAVE_TRITON:
    @triton.jit
    def _rwkv7_state_step_kernel(
        state_ptr, r_ptr, gk_ptr, k_ptr, v_ptr, a_ptr, b_ptr, out_ptr,
        K: tl.constexpr, V: tl.constexpr, BLOCK_K: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        """One program owns a disjoint [K, BLOCK_V] state tile.

        This is the RWKV-7 DPLR transition used by ``_rwkv7_python_ref``:
        pre-decay removal read, decay, rank-one removal/write, then state read.
        State accumulation remains fp32, matching the recurrent reference.
        """
        bh = tl.program_id(0)
        v_block = tl.program_id(1)
        offs_k = tl.arange(0, BLOCK_K)
        offs_v = v_block * BLOCK_V + tl.arange(0, BLOCK_V)
        mask_k = offs_k < K
        mask_v = offs_v < V
        state_offsets = bh * K * V + offs_k[:, None] * V + offs_v[None, :]
        matrix = tl.load(
            state_ptr + state_offsets,
            mask=mask_k[:, None] & mask_v[None, :],
            other=0.0,
        ).to(tl.float32)
        vector_offsets = bh * K + offs_k
        r = tl.load(r_ptr + vector_offsets, mask=mask_k, other=0.0).to(tl.float32)
        gk = tl.load(gk_ptr + vector_offsets, mask=mask_k, other=0.0).to(tl.float32)
        key = tl.load(k_ptr + vector_offsets, mask=mask_k, other=0.0).to(tl.float32)
        remove_a = tl.load(a_ptr + vector_offsets, mask=mask_k, other=0.0).to(tl.float32)
        remove_b = tl.load(b_ptr + vector_offsets, mask=mask_k, other=0.0).to(tl.float32)
        value = tl.load(v_ptr + bh * V + offs_v, mask=mask_v, other=0.0).to(tl.float32)

        removal_read = tl.sum(matrix * remove_a[:, None], axis=0)
        updated = matrix * tl.exp(gk)[:, None]
        updated += remove_b[:, None] * removal_read[None, :]
        updated += key[:, None] * value[None, :]
        output = tl.sum(updated * r[:, None], axis=0)

        tl.store(
            state_ptr + state_offsets, updated,
            mask=mask_k[:, None] & mask_v[None, :],
        )
        tl.store(out_ptr + bh * V + offs_v, output, mask=mask_v)


def triton_status() -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "CUDA is unavailable"
    if not _HAVE_TRITON:
        return False, f"Triton is unavailable: {_TRITON_ERROR!r}"
    return True, f"Triton {triton.__version__}"


def rwkv7_recurrent_step(
    r: torch.Tensor, gk: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    remove_a: torch.Tensor, remove_b: torch.Tensor, state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the fused inference-only RWKV-7 transition for a single token.

    Inputs use FLA's ``[B, 1, H, K]`` convention and state is
    ``[B, H, K, V]``.  The returned state is a clone so the public model API
    remains functional; CUDA Graph plans subsequently copy it into their
    persistent static state inside the captured replay.
    """
    available, reason = triton_status()
    if not available:
        raise RuntimeError(reason)
    if r.ndim != 4 or r.shape[1] != 1:
        raise ValueError("the fused RWKV transition requires [batch,1,heads,width]")
    if state.ndim != 4 or state.shape[:2] != (r.shape[0], r.shape[2]):
        raise ValueError("RWKV recurrent state does not match batch/head geometry")
    if state.shape[-2:] != (r.shape[-1], value.shape[-1]):
        raise ValueError("RWKV recurrent state does not match key/value geometry")
    if any(t.device != r.device for t in (gk, key, value, remove_a, remove_b, state)):
        raise ValueError("all fused RWKV transition tensors must share one CUDA device")
    if not all(t.is_contiguous() for t in (r, gk, key, value, remove_a, remove_b, state)):
        r, gk, key, value, remove_a, remove_b, state = (
            t.contiguous() for t in (r, gk, key, value, remove_a, remove_b, state)
        )
    B, _, H, K = r.shape
    V = value.shape[-1]
    next_state = state.to(torch.float32).clone()
    output = torch.empty((B, H, V), device=r.device, dtype=value.dtype)
    block_k = triton.next_power_of_2(K)
    # Do not use ``triton.autotune`` directly on this mutating kernel: every
    # benchmark invocation would advance state. Plan tuning benchmarks from a
    # restored snapshot instead. Width 64 is the native RWKV head geometry;
    # smaller heads retain one program and mask unused lanes.
    block_v = min(64, triton.next_power_of_2(V))
    grid = (B * H, triton.cdiv(V, block_v))
    _rwkv7_state_step_kernel[grid](
        next_state, r, gk, key, value, remove_a, remove_b, output,
        K=K, V=V, BLOCK_K=block_k, BLOCK_V=block_v,
        num_warps=(8 if block_v == 64 else 4),
    )
    return output[:, None], next_state


def _state_spec(value: Any, leaves: list[torch.Tensor]):
    if isinstance(value, torch.Tensor):
        index = len(leaves)
        leaves.append(value)
        return ("tensor", index)
    if value is None:
        return ("constant", None)
    if isinstance(value, dict):
        return ("dict", tuple((key, _state_spec(child, leaves))
                              for key, child in value.items()))
    if isinstance(value, list):
        return ("list", tuple(_state_spec(child, leaves) for child in value))
    if isinstance(value, tuple):
        return ("tuple", tuple(_state_spec(child, leaves) for child in value))
    raise TypeError(
        "megakernel state must contain only tensors, None, dicts, lists, and tuples; "
        f"got {type(value).__name__}"
    )


def _state_rebuild(spec, leaves: Sequence[torch.Tensor]):
    kind, payload = spec
    if kind == "tensor":
        return leaves[payload]
    if kind == "constant":
        return payload
    if kind == "dict":
        return {key: _state_rebuild(child, leaves) for key, child in payload}
    values = tuple(_state_rebuild(child, leaves) for child in payload)
    return list(values) if kind == "list" else values


@dataclass(frozen=True)
class StateCodec:
    """Stable tensor-only ABI for nested recurrent state."""

    spec: Any
    leaf_count: int

    @classmethod
    def from_state(cls, state: Any) -> tuple["StateCodec", tuple[torch.Tensor, ...]]:
        leaves: list[torch.Tensor] = []
        spec = _state_spec(state, leaves)
        return cls(spec, len(leaves)), tuple(leaves)

    def flatten(self, state: Any) -> tuple[torch.Tensor, ...]:
        leaves: list[torch.Tensor] = []
        if _state_spec(state, leaves) != self.spec:
            raise ValueError("recurrent state structure changed after plan compilation")
        return tuple(leaves)

    def rebuild(self, leaves: Sequence[torch.Tensor]):
        if len(leaves) != self.leaf_count:
            raise ValueError("incorrect recurrent-state tensor count")
        return _state_rebuild(self.spec, leaves)


def _architecture_key(model: torch.nn.Module, ids: torch.Tensor,
                      state: Sequence[torch.Tensor]) -> str:
    geometry = {
        "model": type(model).__qualname__,
        "device": str(ids.device),
        "capability": list(torch.cuda.get_device_capability(ids.device)),
        "torch": torch.__version__,
        "triton": getattr(triton, "__version__", "unavailable"),
        "ids": [list(ids.shape), str(ids.dtype)],
        "state": [[list(t.shape), str(t.dtype)] for t in state],
        "parameters": [[name, list(parameter.shape), str(parameter.dtype)]
                       for name, parameter in model.named_parameters()],
    }
    return hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest()


class CUDAGraphDecodePlan:
    """Persistent, fixed-shape one-token execution plan with mutable state."""

    def __init__(self, model: torch.nn.Module, sample_ids: torch.Tensor, state: Any,
                 *, compile_mode: str = "max-autotune-no-cudagraphs"):
        if sample_ids.device.type != "cuda" or sample_ids.shape[1:] != (1,):
            raise ValueError("CUDA Graph decode plans require CUDA ids shaped [batch,1]")
        self.model = model
        self.codec, leaves = StateCodec.from_state(state)
        self.static_ids = sample_ids.clone()
        self.static_state = tuple(t.detach().clone() for t in leaves)
        self.key = _architecture_key(model, sample_ids, self.static_state)
        self.compile_mode = compile_mode
        self.compile_seconds = 0.0
        self.graph = torch.cuda.CUDAGraph()
        self._build()

    def _build(self) -> None:
        device = self.static_ids.device

        def execute(ids, *flat_state):
            logits, next_state = self.model.forward_recurrent(
                ids, self.codec.rebuild(flat_state))
            return (logits, *self.codec.flatten(next_state))

        started = time.perf_counter()
        compiled: Callable = torch.compile(
            execute, mode=self.compile_mode, fullgraph=False, dynamic=False)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream), torch.no_grad():
            for _ in range(3):
                compiled(self.static_ids, *self.static_state)
        torch.cuda.current_stream(device).wait_stream(stream)
        original_ids = self.static_ids.clone()
        original_state = tuple(t.clone() for t in self.static_state)
        with torch.cuda.graph(self.graph), torch.no_grad():
            result = compiled(self.static_ids, *self.static_state)
            self.static_logits = result[0]
            next_state = result[1:]
            for destination, source in zip(self.static_state, next_state):
                destination.copy_(source)
        # Capture executes once; restore the caller's logical state before use.
        self.static_ids.copy_(original_ids)
        for destination, source in zip(self.static_state, original_state):
            destination.copy_(source)
        torch.cuda.synchronize(device)
        self.compile_seconds = time.perf_counter() - started

    def load_state(self, state: Any) -> None:
        leaves = self.codec.flatten(state)
        if len(leaves) != len(self.static_state):
            raise ValueError("recurrent state is incompatible with cached execution plan")
        for destination, source in zip(self.static_state, leaves):
            if destination.shape != source.shape or destination.dtype != source.dtype:
                raise ValueError("recurrent state geometry is incompatible with cached plan")
            destination.copy_(source)

    def replay(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.shape != self.static_ids.shape or ids.dtype != self.static_ids.dtype:
            raise ValueError("decode token geometry differs from the compiled plan")
        self.static_ids.copy_(ids)
        self.graph.replay()
        return self.static_logits


def megakernel_incompatibility(model: torch.nn.Module, device: str | torch.device) -> str | None:
    selected = torch.device(device)
    available, reason = triton_status()
    if selected.type != "cuda":
        return "megakernel execution requires CUDA"
    if not available:
        return reason
    if not hasattr(model, "forward_recurrent"):
        return "model does not expose forward_recurrent"
    recurrent_reason = (model.recurrent_incompatibility()
                        if hasattr(model, "recurrent_incompatibility") else None)
    if recurrent_reason:
        return recurrent_reason
    if getattr(model, "state_offset_adapter", None) is not None:
        return "state-offset step counters are not tensor-only graph state"
    if getattr(model, "online_memory", None) is not None:
        return "online-memory dataclass state is not yet graph-plan compatible"
    return None


def _set_fused_state_step(model: torch.nn.Module, enabled: bool) -> None:
    for module in model.modules():
        if type(module).__name__ == "RWKV8TimeMixDeltaNet":
            module._megakernel_recurrent = enabled


class MegakernelBackend:
    """Own prefill plus cached one-token CUDA Graph plans for one model."""

    def __init__(self, model: torch.nn.Module, *, device: str | torch.device,
                 compile_mode: str = "max-autotune-no-cudagraphs"):
        reason = megakernel_incompatibility(model, device)
        if reason:
            raise RuntimeError(f"megakernel backend unavailable: {reason}")
        self.model = model
        self.device = torch.device(device)
        self.compile_mode = compile_mode
        self.plans: dict[tuple, CUDAGraphDecodePlan] = {}
        self.plan: CUDAGraphDecodePlan | None = None

    def prefill(self, ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits, state = self.model.forward_recurrent(ids)
        sample = ids[:, -1:].clone()
        codec, leaves = StateCodec.from_state(state)
        key = (sample.shape[0], str(sample.dtype), tuple(
            (tuple(t.shape), str(t.dtype)) for t in leaves))
        plan = self.plans.get(key)
        if plan is None:
            _set_fused_state_step(self.model, True)
            try:
                plan = CUDAGraphDecodePlan(
                    self.model, sample, state, compile_mode=self.compile_mode)
            finally:
                # The captured graph retains the compiled Triton launch. Keep
                # ordinary recurrent calls on their independently qualified path.
                _set_fused_state_step(self.model, False)
            self.plans[key] = plan
        else:
            plan.load_state(state)
        self.plan = plan
        return logits

    def step(self, ids: torch.Tensor) -> torch.Tensor:
        if self.plan is None:
            raise RuntimeError("megakernel prefill must run before decode")
        return self.plan.replay(ids)

    def receipt(self) -> dict:
        plan = self.plan
        return {
            "schema": "rwkv-lab.megakernel-plan.v1",
            "available": True,
            "backend": "triton+inductor+cudagraph",
            "plan_sha256": plan.key if plan else "",
            "compile_seconds": plan.compile_seconds if plan else 0.0,
            "cached_plans": len(self.plans),
            "triton": getattr(triton, "__version__", "unavailable"),
            "torch": torch.__version__,
            "device": str(self.device),
        }


def get_megakernel_backend(model: torch.nn.Module, *, device: str | torch.device,
                           compile_mode: str = "max-autotune-no-cudagraphs") -> MegakernelBackend:
    backend = getattr(model, "_megakernel_backend", None)
    if (backend is None or backend.device != torch.device(device)
            or backend.compile_mode != compile_mode):
        backend = MegakernelBackend(model, device=device, compile_mode=compile_mode)
        model._megakernel_backend = backend
    return backend


def _cuda_launch_count(work: Callable[[], Any], device: torch.device) -> int:
    from torch.profiler import ProfilerActivity, profile
    work()
    torch.cuda.synchronize(device)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
        work()
        torch.cuda.synchronize(device)
    return sum(event.device_type == torch.autograd.DeviceType.CUDA
               for event in trace.events())


def qualify_megakernel_generation(
    model: torch.nn.Module, prompt_ids: Sequence[int], *, device: str = "cuda",
    max_new: int = 32, repeats: int = 5, minimum_speedup: float = 1.02,
    compile_mode: str = "max-autotune-no-cudagraphs",
    checkpoint_sha256: str = "",
) -> dict:
    """Qualify token parity, warm latency, and measured CUDA launch reduction."""
    schema = "rwkv-lab.megakernel-qualification.v1"
    reason = megakernel_incompatibility(model, device)
    if reason:
        return {"schema": schema, "available": False, "adopted": False,
                "error": reason}
    from rwkv_lab.generate import sample_with_stats

    backend = get_megakernel_backend(model, device=device, compile_mode=compile_mode)
    reference_rates, candidate_rates = [], []
    reference_tokens = candidate_tokens = None
    exact = True
    # The first candidate invocation compiles and captures. Record cold-start
    # cost in the plan receipt but compare steady-state serving after warmup.
    for repeat in range(max(1, repeats) + 1):
        reference_tokens, reference_stats = sample_with_stats(
            model, list(prompt_ids), max_new=max_new, temperature=0,
            stop_at_sep=False, device=device, seed=17, engine="recurrent")
        candidate_tokens, candidate_stats = sample_with_stats(
            model, list(prompt_ids), max_new=max_new, temperature=0,
            stop_at_sep=False, device=device, seed=17, engine="megakernel")
        exact &= candidate_tokens == reference_tokens
        if repeat:
            reference_rates.append(reference_stats["tokens_per_second"])
            candidate_rates.append(candidate_stats["tokens_per_second"])

    selected = torch.device(device)
    prompt = torch.tensor([list(prompt_ids)], dtype=torch.long, device=selected)
    token = prompt[:, -1:].clone()
    with torch.no_grad():
        _, state = model.forward_recurrent(prompt)
    launches_before = _cuda_launch_count(
        lambda: model.forward_recurrent(token, state), selected)
    backend.prefill(prompt)
    launches_after = _cuda_launch_count(lambda: backend.step(token), selected)
    reference_rate = sorted(reference_rates)[len(reference_rates) // 2]
    candidate_rate = sorted(candidate_rates)[len(candidate_rates) // 2]
    speedup = candidate_rate / max(reference_rate, 1e-12)
    launch_reduction = launches_before - launches_after
    adopted = bool(exact and speedup >= minimum_speedup and launch_reduction > 0)
    model._megakernel_adopted = adopted
    environment = {
        "device_name": torch.cuda.get_device_name(selected),
        "compute_capability": list(torch.cuda.get_device_capability(selected)),
        "torch": torch.__version__,
        "triton": getattr(triton, "__version__", "unavailable"),
    }
    return {
        "schema": schema, "available": True, "exact_tokens": exact,
        "tokens": len(candidate_tokens or ()),
        "reference_tokens_per_second": reference_rate,
        "megakernel_tokens_per_second": candidate_rate,
        "speedup": speedup, "cuda_kernels_before": launches_before,
        "cuda_kernels_after": launches_after,
        "launch_reduction": launch_reduction,
        "minimum_speedup": minimum_speedup, "adopted": adopted,
        "checkpoint_sha256": checkpoint_sha256,
        "environment": environment,
        "plan": backend.receipt(),
    }


def file_sha256(path: str | Path, *, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adopt_megakernel_receipt(model: torch.nn.Module, receipt_path: str | Path,
                             checkpoint_path: str | Path) -> dict:
    """Validate a persisted receipt before allowing ``--engine auto`` adoption."""
    payload = json.loads(Path(receipt_path).read_text())
    if payload.get("schema") == "rwkv-lab.production-kernel-qualification.v1":
        report = payload.get("reports", {}).get("megakernel_decode", {})
    elif payload.get("schema") == "rwkv-lab.megakernel-qualification.v1":
        report = payload
    else:
        raise ValueError("qualification receipt has an unsupported schema")
    if not report.get("adopted"):
        raise ValueError("qualification receipt did not adopt the megakernel backend")
    expected = str(report.get("checkpoint_sha256", ""))
    if len(expected) != 64:
        raise ValueError("qualification receipt is not bound to a checkpoint hash")
    actual = file_sha256(checkpoint_path)
    if actual != expected:
        raise ValueError("qualification receipt checkpoint hash does not match")
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        raise ValueError("cannot adopt a megakernel receipt for a parameterless model") from None
    if model_device.type != "cuda":
        raise ValueError("megakernel qualification receipts require a CUDA model")
    environment = report.get("environment", {})
    required_environment = {
        "compute_capability": list(torch.cuda.get_device_capability(model_device)),
        "torch": torch.__version__,
        "triton": getattr(triton, "__version__", "unavailable"),
    }
    for key, current in required_environment.items():
        if environment.get(key) != current:
            raise ValueError(f"qualification receipt {key} does not match this runtime")
    model._megakernel_adopted = True
    model._megakernel_qualification = report
    return report
