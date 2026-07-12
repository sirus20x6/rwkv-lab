"""Compiled low-latency execution backend for recurrent RWKV decoding.

The execution model is inspired by HazyResearch's Megakernels throughput
branch and TileRT's tile-level runtime:

* https://github.com/HazyResearch/Megakernels/tree/throughput
* https://github.com/tile-ai/TileRT

Those projects currently target their own model layouts, so RWKV-Lab does not
load their generated code or claim binary compatibility.  Instead this module
implements the equivalent boundary for native ``RWKV7Small`` checkpoints:
compiler-visible Triton state, epilogue, layer-boundary, projection, and FFN
candidates sit inside strict fixed-shape execution plans. ``torch.compile`` and
single-replay CUDA Graph generation remove Python dispatch across complete
greedy segments. Adoption remains fail-closed through token parity, incremental
latency, allocation traffic, and profiler-derived kernel receipts.
"""
from __future__ import annotations

import argparse
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
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_V": 8}, num_warps=2),
            triton.Config({"BLOCK_V": 16}, num_warps=4),
            triton.Config({"BLOCK_V": 32}, num_warps=4),
            triton.Config({"BLOCK_V": 64}, num_warps=8),
            triton.Config({"BLOCK_V": 128}, num_warps=8),
        ],
        key=["B", "H", "K", "V"],
        # The state transition is intentionally in-place. Triton's tuner must
        # restore it around every candidate or benchmark trials would observe
        # different recurrent histories. cache_results also persists the
        # selected geometry in Triton's compiler cache.
        restore_value=["state_ptr"],
        cache_results=True,
    )
    @triton.jit
    def _rwkv7_state_step_kernel(
        state_ptr, r_ptr, gk_ptr, k_ptr, v_ptr, a_ptr, b_ptr, out_ptr,
        B: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
        BLOCK_K: tl.constexpr,
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

    @torch.library.triton_op(
        "rwkv_lab::rwkv7_state_step_", mutates_args={"state"})
    def _rwkv7_state_step_op(
        state: torch.Tensor, r: torch.Tensor, gk: torch.Tensor,
        key: torch.Tensor, value: torch.Tensor, remove_a: torch.Tensor,
        remove_b: torch.Tensor,
    ) -> torch.Tensor:
        """Compiler-visible mutable op; see PyTorch ``triton_op`` recipe.

        https://docs.pytorch.org/tutorials/recipes/
        torch_compile_user_defined_triton_kernel_tutorial.html
        """
        B, _, H, K = r.shape
        V = value.shape[-1]
        output = torch.empty((B, 1, H, V), device=r.device, dtype=value.dtype)
        block_k = triton.next_power_of_2(K)
        def grid(meta):
            return B * H, triton.cdiv(V, meta["BLOCK_V"])
        torch.library.wrap_triton(_rwkv7_state_step_kernel)[grid](
            state, r, gk, key, value, remove_a, remove_b, output,
            B=B, H=H, K=K, V=V, BLOCK_K=block_k,
        )
        return output

    @triton.jit
    def _rwkv7_epilogue_batched_kernel(
        wkv_ptr, r_ptr, key_ptr, value_ptr, r_k_ptr, gate_ptr,
        norm_weight_ptr, norm_bias_ptr, output_ptr,
        H: tl.constexpr, N: tl.constexpr, BLOCK_N: tl.constexpr,
        EPS: tl.constexpr,
    ):
        bh = tl.program_id(0)
        head = bh % H
        offs = tl.arange(0, BLOCK_N)
        mask = offs < N
        base = bh * N + offs
        channel = head * N + offs
        wkv = tl.load(wkv_ptr + base, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(wkv, axis=0) / N
        centered = tl.where(mask, wkv - mean, 0.0)
        variance = tl.sum(centered * centered, axis=0) / N
        normalized = centered * tl.rsqrt(variance + EPS)
        weight = tl.load(norm_weight_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(norm_bias_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(r_ptr + base, mask=mask, other=0.0).to(tl.float32)
        key = tl.load(key_ptr + base, mask=mask, other=0.0).to(tl.float32)
        rk = tl.load(r_k_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        value = tl.load(value_ptr + base, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(gate_ptr + base, mask=mask, other=0.0).to(tl.float32)
        bonus_scale = tl.sum(r * key * rk, axis=0)
        tl.store(output_ptr + base,
                 (normalized * weight + bias + bonus_scale * value) * gate,
                 mask=mask)

    @torch.library.triton_op("rwkv_lab::rwkv7_epilogue", mutates_args={})
    def _rwkv7_epilogue_op(
        wkv: torch.Tensor, r: torch.Tensor, key: torch.Tensor,
        value: torch.Tensor, r_k: torch.Tensor, gate: torch.Tensor,
        norm_weight: torch.Tensor, norm_bias: torch.Tensor, eps: float,
    ) -> torch.Tensor:
        B, _, H, N = wkv.shape
        output = torch.empty_like(wkv)
        torch.library.wrap_triton(_rwkv7_epilogue_batched_kernel)[(B * H,)](
            wkv, r, key, value, r_k, gate, norm_weight, norm_bias, output,
            H=H, N=N, BLOCK_N=triton.next_power_of_2(N), EPS=eps,
            num_warps=4,
        )
        return output

    @triton.autotune(
        configs=[triton.Config({}, num_warps=4),
                 triton.Config({}, num_warps=8)],
        key=["B", "H", "K", "V"], restore_value=["state_ptr"],
        cache_results=True,
    )
    @triton.jit
    def _rwkv7_state_epilogue_kernel(
        state_ptr, r_ptr, gk_ptr, key_ptr, value_ptr, a_ptr, b_ptr,
        rk_ptr, gate_ptr, norm_weight_ptr, norm_bias_ptr, output_ptr,
        B: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
        BLOCK_K: tl.constexpr, BLOCK_V: tl.constexpr, EPS: tl.constexpr,
    ):
        """Update fp32 state and emit the normalized TimeMix result directly.

        This removes the intermediate WKV tensor between the recurrent update
        and GroupNorm/bonus/gate boundary. The combined direction follows the
        fp32-state + lnx variants in Albatross faster3b_2606:
        https://github.com/BlinkDL/Albatross/tree/main/faster3b_2606
        """
        bh = tl.program_id(0)
        head = bh % H
        offs_k = tl.arange(0, BLOCK_K)
        offs_v = tl.arange(0, BLOCK_V)
        mask_k = offs_k < K
        mask_v = offs_v < V
        state_offsets = bh * K * V + offs_k[:, None] * V + offs_v[None, :]
        matrix = tl.load(
            state_ptr + state_offsets,
            mask=mask_k[:, None] & mask_v[None, :], other=0.0,
        ).to(tl.float32)
        vector_offsets = bh * K + offs_k
        r = tl.load(r_ptr + vector_offsets, mask=mask_k, other=0.0).to(tl.float32)
        gk = tl.load(gk_ptr + vector_offsets, mask=mask_k, other=0.0).to(tl.float32)
        key = tl.load(key_ptr + vector_offsets, mask=mask_k, other=0.0).to(tl.float32)
        remove_a = tl.load(a_ptr + vector_offsets, mask=mask_k, other=0.0).to(tl.float32)
        remove_b = tl.load(b_ptr + vector_offsets, mask=mask_k, other=0.0).to(tl.float32)
        value_base = bh * V + offs_v
        value = tl.load(value_ptr + value_base, mask=mask_v, other=0.0).to(tl.float32)
        removal_read = tl.sum(matrix * remove_a[:, None], axis=0)
        updated = matrix * tl.exp(gk)[:, None]
        updated += remove_b[:, None] * removal_read[None, :]
        updated += key[:, None] * value[None, :]
        wkv = tl.sum(updated * r[:, None], axis=0)
        tl.store(state_ptr + state_offsets, updated,
                 mask=mask_k[:, None] & mask_v[None, :])

        mean = tl.sum(wkv, axis=0) / V
        centered = tl.where(mask_v, wkv - mean, 0.0)
        variance = tl.sum(centered * centered, axis=0) / V
        channel = head * V + offs_v
        weight = tl.load(norm_weight_ptr + channel, mask=mask_v, other=0.0).to(tl.float32)
        bias = tl.load(norm_bias_ptr + channel, mask=mask_v, other=0.0).to(tl.float32)
        rk = tl.load(rk_ptr + channel, mask=mask_v, other=0.0).to(tl.float32)
        gate = tl.load(gate_ptr + value_base, mask=mask_v, other=0.0).to(tl.float32)
        # Native RWKV heads use K == V. The public wrapper guards this so the
        # bonus reduction indexes the same head channels as the state query.
        bonus = tl.sum(r * key * rk, axis=0)
        normalized = centered * tl.rsqrt(variance + EPS)
        tl.store(output_ptr + value_base,
                 (normalized * weight + bias + bonus * value) * gate,
                 mask=mask_v)

    @torch.library.triton_op(
        "rwkv_lab::rwkv7_state_epilogue_", mutates_args={"state"})
    def _rwkv7_state_epilogue_op(
        state: torch.Tensor, r: torch.Tensor, gk: torch.Tensor,
        key: torch.Tensor, value: torch.Tensor, remove_a: torch.Tensor,
        remove_b: torch.Tensor, r_k: torch.Tensor, gate: torch.Tensor,
        norm_weight: torch.Tensor, norm_bias: torch.Tensor, eps: float,
    ) -> torch.Tensor:
        B, _, H, K = r.shape
        V = value.shape[-1]
        output = torch.empty_like(value)
        torch.library.wrap_triton(_rwkv7_state_epilogue_kernel)[(B * H,)](
            state, r, gk, key, value, remove_a, remove_b, r_k, gate,
            norm_weight, norm_bias, output,
            B=B, H=H, K=K, V=V,
            BLOCK_K=triton.next_power_of_2(K),
            BLOCK_V=triton.next_power_of_2(V), EPS=float(eps),
        )
        return output


def triton_status() -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "CUDA is unavailable"
    if not _HAVE_TRITON:
        return False, f"Triton is unavailable: {_TRITON_ERROR!r}"
    return True, f"Triton {triton.__version__}"


def rwkv7_recurrent_step(
    r: torch.Tensor, gk: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    remove_a: torch.Tensor, remove_b: torch.Tensor, state: torch.Tensor, *,
    inplace: bool = False, persistent_sm120: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the fused inference-only RWKV-7 transition for a single token.

    Inputs use FLA's ``[B, 1, H, K]`` convention and state is
    ``[B, H, K, V]``. Ordinary callers receive a functional clone. The
    megakernel CUDA Graph opts into in-place state because its static buffers
    are private to one execution plan and each Triton program owns a disjoint
    state tile.
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
    r, gk, key, value, remove_a, remove_b = (
        t if t.is_contiguous() else t.contiguous()
        for t in (r, gk, key, value, remove_a, remove_b)
    )
    if inplace:
        if torch.is_grad_enabled():
            raise ValueError("in-place RWKV state is inference-only")
        if state.dtype != torch.float32 or not state.is_contiguous():
            raise ValueError("in-place RWKV state must be contiguous fp32")
        next_state = state
    else:
        next_state = state.to(torch.float32).contiguous().clone()
    if persistent_sm120:
        from rwkv_lab.sm120_kernels import persistent_rwkv7_state_step
        output = persistent_rwkv7_state_step(
            next_state, r, gk, key, value, remove_a, remove_b)
    else:
        output = _rwkv7_state_step_op(
            next_state, r, gk, key, value, remove_a, remove_b)
    return output, next_state


def rwkv7_time_mix_epilogue(
    wkv: torch.Tensor, r: torch.Tensor, key: torch.Tensor,
    value: torch.Tensor, r_k: torch.Tensor, gate: torch.Tensor,
    norm_weight: torch.Tensor, norm_bias: torch.Tensor, *, eps: float,
) -> torch.Tensor:
    """Fused one-token GroupNorm + RWKV bonus + gate, returning [B,1,H,N]."""
    if wkv.ndim != 4 or wkv.shape[1] != 1:
        raise ValueError("fused RWKV epilogue requires [batch,1,heads,width]")
    expected = wkv.shape
    if any(t.shape != expected for t in (r, key, value, gate)):
        raise ValueError("fused RWKV epilogue activation shapes must match")
    channels = expected[2] * expected[3]
    if (r_k.numel() != channels or norm_weight.numel() != channels
            or norm_bias.numel() != channels):
        raise ValueError("fused RWKV epilogue parameter geometry mismatch")
    tensors = (wkv, r, key, value, r_k, gate, norm_weight, norm_bias)
    if any(t.device != wkv.device for t in tensors):
        raise ValueError("fused RWKV epilogue tensors must share one CUDA device")
    return _rwkv7_epilogue_op(
        *(t if t.is_contiguous() else t.contiguous() for t in tensors), float(eps))


def rwkv7_recurrent_step_epilogue(
    r: torch.Tensor, gk: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    remove_a: torch.Tensor, remove_b: torch.Tensor, state: torch.Tensor,
    r_k: torch.Tensor, gate: torch.Tensor, norm_weight: torch.Tensor,
    norm_bias: torch.Tensor, *, eps: float, inplace: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse the one-token fp32 state transition through the TimeMix gate."""
    available, reason = triton_status()
    if not available:
        raise RuntimeError(reason)
    if r.ndim != 4 or r.shape[1] != 1 or value.shape != r.shape:
        raise ValueError("combined RWKV state/epilogue requires matching [B,1,H,N]")
    if r.shape[-1] != value.shape[-1]:
        raise ValueError("combined RWKV state/epilogue requires equal key/value widths")
    if state.shape != (r.shape[0], r.shape[2], r.shape[3], value.shape[3]):
        raise ValueError("combined RWKV state/epilogue state geometry mismatch")
    channels = r.shape[2] * r.shape[3]
    if (r_k.numel() != channels or norm_weight.numel() != channels
            or norm_bias.numel() != channels or gate.shape != value.shape):
        raise ValueError("combined RWKV state/epilogue parameter geometry mismatch")
    tensors = (r, gk, key, value, remove_a, remove_b, state, r_k, gate,
               norm_weight, norm_bias)
    if any(t.device != r.device for t in tensors):
        raise ValueError("combined RWKV state/epilogue tensors must share one device")
    if inplace:
        if torch.is_grad_enabled():
            raise ValueError("in-place RWKV state is inference-only")
        if state.dtype != torch.float32 or not state.is_contiguous():
            raise ValueError("in-place RWKV state must be contiguous fp32")
        next_state = state
    else:
        next_state = state.to(torch.float32).contiguous().clone()
    contiguous = tuple(t if t.is_contiguous() else t.contiguous()
                       for t in tensors[:6] + tensors[7:])
    output = _rwkv7_state_epilogue_op(
        next_state, *contiguous[:6], *contiguous[6:], float(eps))
    return output, next_state


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
                      state: Sequence[torch.Tensor], *, kind: str,
                      options: dict[str, Any] | None = None) -> str:
    geometry = {
        "kind": kind,
        "options": options or {},
        "model": type(model).__qualname__,
        "device": str(ids.device),
        "capability": list(torch.cuda.get_device_capability(ids.device)),
        "torch": torch.__version__,
        "triton": getattr(triton, "__version__", "unavailable"),
        "ids": [list(ids.shape), str(ids.dtype)],
        "state": [[list(t.shape), str(t.dtype)] for t in state],
        "parameters": [[name, list(parameter.shape), str(parameter.dtype)]
                       for name, parameter in model.named_parameters()],
        "buffers": [[name, list(buffer.shape), str(buffer.dtype)]
                    for name, buffer in model.named_buffers()],
    }
    return hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest()


class CUDAGraphDecodePlan:
    """Persistent, fixed-shape one-token execution plan with mutable state."""

    def __init__(self, model: torch.nn.Module, sample_ids: torch.Tensor, state: Any,
                 *, compile_mode: str = "max-autotune-no-cudagraphs",
                 greedy_feedback: bool = False, aot_artifact: str = ""):
        if sample_ids.device.type != "cuda" or sample_ids.shape[1:] != (1,):
            raise ValueError("CUDA Graph decode plans require CUDA ids shaped [batch,1]")
        self.model = model
        self.codec, leaves = StateCodec.from_state(state)
        self.static_ids = sample_ids.clone()
        self.static_state = tuple(t.detach().clone() for t in leaves)
        self.key = _architecture_key(
            model, sample_ids, self.static_state, kind="decode",
            options={"compile_mode": compile_mode,
                     "greedy_feedback": bool(greedy_feedback)})
        self.compile_mode = compile_mode
        self.greedy_feedback = bool(greedy_feedback)
        self.aot_artifact = str(aot_artifact)
        self.aot_loaded = False
        self.compile_seconds = 0.0
        self.graph = torch.cuda.CUDAGraph()
        self._build()

    def _build(self) -> None:
        device = self.static_ids.device
        model, codec, greedy_feedback = self.model, self.codec, self.greedy_feedback

        def execute(ids, *flat_state):
            logits, next_state = model.forward_recurrent(ids, codec.rebuild(flat_state))
            if greedy_feedback:
                next_ids = logits[:, -1].float().argmax(dim=-1, keepdim=True)
                return (logits, next_ids, *codec.flatten(next_state))
            return (logits, *codec.flatten(next_state))

        started = time.perf_counter()
        original_ids = self.static_ids.clone()
        original_state = tuple(t.clone() for t in self.static_state)
        if self.aot_artifact:
            # The serialized fullgraph already contains the fused operations;
            # its external model guard reflects the post-capture public flags.
            artifact_path = Path(self.aot_artifact)
            manifest = json.loads(
                artifact_path.with_suffix(artifact_path.suffix + ".json").read_text())
            if manifest.get("plan_sha256") != self.key:
                raise ValueError("AOT artifact plan geometry does not match this decode plan")
            _set_fused_state_step(model, False)
            with artifact_path.open("rb") as artifact:
                self.compiled = torch.compiler.load_compiled_function(
                    artifact, f_globals=globals(), external_data={"model": model})
            self.aot_loaded = True
            self.fullgraph = True
        else:
            self.compiled = torch.compile(
                execute, mode=self.compile_mode, fullgraph=True, dynamic=False)
        stream = torch.cuda.Stream(device=device)
        if (l2_controller := getattr(model, "_megakernel_l2_controller", None)) is not None:
            l2_controller.apply_to_stream(stream)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream), torch.no_grad():
            for _ in range(3):
                self.compiled(self.static_ids, *self.static_state)
        torch.cuda.current_stream(device).wait_stream(stream)
        self.static_ids.copy_(original_ids)
        for destination, source in zip(self.static_state, original_state):
            destination.copy_(source)
        with torch.cuda.graph(self.graph), torch.no_grad():
            result = self.compiled(self.static_ids, *self.static_state)
            self.static_logits = result[0]
            state_offset = 1
            if self.greedy_feedback:
                self.static_next_ids = result[1]
                self.static_ids.copy_(self.static_next_ids)
                state_offset = 2
            next_state = result[state_offset:]
            for destination, source in zip(self.static_state, next_state):
                if destination.data_ptr() != source.data_ptr():
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

    def generate_greedy(self, first_ids: torch.Tensor, max_new: int) -> torch.Tensor:
        """Generate into one device buffer with no per-token host synchronization."""
        if not self.greedy_feedback:
            raise RuntimeError("execution plan was not compiled for greedy feedback")
        if max_new < 1:
            return torch.empty((first_ids.shape[0], 0), device=first_ids.device,
                               dtype=first_ids.dtype)
        if first_ids.shape != self.static_ids.shape:
            raise ValueError("first greedy token geometry differs from the compiled plan")
        tokens = torch.empty((first_ids.shape[0], max_new), device=first_ids.device,
                             dtype=first_ids.dtype)
        self.static_ids.copy_(first_ids)
        tokens[:, :1].copy_(first_ids)
        for position in range(1, max_new):
            self.graph.replay()
            tokens[:, position:position + 1].copy_(self.static_ids)
        return tokens

    def export_aot(self, artifact_path: str | Path, *,
                   checkpoint_sha256: str = "") -> dict:
        """Serialize the fullgraph compile artifact; weights remain external.

        PyTorch 2.12's experimental AOT package stores guards and generated
        Inductor/Triton code while resolving the model object at load time.
        https://docs.pytorch.org/docs/main/user_guide/torch_compiler/
        torch.compiler_aot_compile.html
        """
        artifact_path = Path(artifact_path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        original = tuple(t.clone() for t in self.static_state)
        try:
            with torch.no_grad():
                aot = self.compiled.aot_compile(
                    ((self.static_ids, *self.static_state), {}))
            aot.save_compiled_function(
                str(artifact_path), external_data={"model": self.model})
        finally:
            for destination, source in zip(self.static_state, original):
                destination.copy_(source)
        manifest = {
            "schema": "rwkv-lab.megakernel-aot.v1",
            "artifact": artifact_path.name,
            "artifact_sha256": file_sha256(artifact_path),
            "checkpoint_sha256": checkpoint_sha256,
            "plan_sha256": self.key,
            "greedy_feedback": self.greedy_feedback,
            "external_data": ["model"],
            "torch": torch.__version__,
            "triton": getattr(triton, "__version__", "unavailable"),
            "compute_capability": list(torch.cuda.get_device_capability(self.static_ids.device)),
            "state": [[list(t.shape), str(t.dtype)] for t in self.static_state],
        }
        manifest_path = artifact_path.with_suffix(artifact_path.suffix + ".json")
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return manifest


class CUDAGraphGreedyPlan:
    """One-replay, fixed-budget greedy generation built from a decode plan.

    The token loop is unrolled while the CUDA Graph is captured.  Replaying it
    therefore performs every recurrent step and writes every generated token
    without returning to Python between positions.  The plan owns independent
    recurrent buffers so its capture and replay cannot perturb the reusable
    one-token/AOT decode plan.
    """

    def __init__(self, decode_plan: CUDAGraphDecodePlan, state: Any, *,
                 max_new: int, stop_token_id: int | None = None):
        if not decode_plan.greedy_feedback:
            raise ValueError("multi-token greedy plans require greedy feedback")
        if max_new < 1:
            raise ValueError("multi-token greedy plans require max_new >= 1")
        self.decode_plan = decode_plan
        self.codec = decode_plan.codec
        self.max_new = int(max_new)
        self.stop_token_id = stop_token_id
        leaves = self.codec.flatten(state)
        self.static_ids = decode_plan.static_ids.clone()
        self.static_state = tuple(t.detach().clone() for t in leaves)
        self.static_tokens = torch.empty(
            (self.static_ids.shape[0], self.max_new),
            device=self.static_ids.device, dtype=self.static_ids.dtype)
        self.static_finished = (
            torch.empty_like(self.static_ids, dtype=torch.bool)
            if stop_token_id is not None else None)
        self.graph = torch.cuda.CUDAGraph()
        self.capture_seconds = 0.0
        self.key = hashlib.sha256(json.dumps({
            "decode_plan": decode_plan.key,
            "max_new": self.max_new,
            "stop_token_id": stop_token_id,
        }, sort_keys=True).encode()).hexdigest()
        self._build()

    def _execute_segment(self) -> None:
        ids = self.static_ids
        state = self.static_state
        self.static_tokens[:, :1].copy_(ids)
        if self.static_finished is not None:
            self.static_finished.copy_(ids == self.stop_token_id)
        for position in range(1, self.max_new):
            result = self.decode_plan.compiled(ids, *state)
            raw_next_ids = result[1]
            next_state = result[2:]
            if self.static_finished is not None:
                stop_ids = torch.full_like(raw_next_ids, self.stop_token_id)
                next_ids = torch.where(self.static_finished, stop_ids, raw_next_ids)
                self.static_finished.logical_or_(next_ids == self.stop_token_id)
            else:
                next_ids = raw_next_ids
            ids.copy_(next_ids)
            self.static_tokens[:, position:position + 1].copy_(ids)
            for destination, source in zip(state, next_state):
                if destination.data_ptr() != source.data_ptr():
                    destination.copy_(source)

    def _build(self) -> None:
        device = self.static_ids.device
        original_ids = self.static_ids.clone()
        original_state = tuple(t.clone() for t in self.static_state)
        started = time.perf_counter()
        _set_fused_state_step(
            self.decode_plan.model, not self.decode_plan.aot_loaded, inplace=True)
        try:
            stream = torch.cuda.Stream(device=device)
            if (l2_controller := getattr(
                    self.decode_plan.model, "_megakernel_l2_controller", None)) is not None:
                l2_controller.apply_to_stream(stream)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream), torch.no_grad():
                self._execute_segment()
            torch.cuda.current_stream(device).wait_stream(stream)
            self.static_ids.copy_(original_ids)
            for destination, source in zip(self.static_state, original_state):
                destination.copy_(source)
            with torch.cuda.graph(self.graph), torch.no_grad():
                self._execute_segment()
            # Capture executes the segment once. Restore the prefill boundary.
            self.static_ids.copy_(original_ids)
            for destination, source in zip(self.static_state, original_state):
                destination.copy_(source)
            torch.cuda.synchronize(device)
        finally:
            _set_fused_state_step(self.decode_plan.model, False)
        self.capture_seconds = time.perf_counter() - started

    def load_state(self, state: Any) -> None:
        leaves = self.codec.flatten(state)
        if len(leaves) != len(self.static_state):
            raise ValueError("recurrent state is incompatible with greedy plan")
        for destination, source in zip(self.static_state, leaves):
            if destination.shape != source.shape or destination.dtype != source.dtype:
                raise ValueError("recurrent state geometry is incompatible with greedy plan")
            destination.copy_(source)

    def replay(self, first_ids: torch.Tensor) -> torch.Tensor:
        if (first_ids.shape != self.static_ids.shape
                or first_ids.dtype != self.static_ids.dtype):
            raise ValueError("first greedy token geometry differs from the compiled plan")
        self.static_ids.copy_(first_ids)
        self.graph.replay()
        return self.static_tokens


class CUDAGraphPrefillPlan:
    """Compiled/captured exact-shape prompt plan; recurrent padding is unsafe."""

    def __init__(self, model: torch.nn.Module, sample_ids: torch.Tensor,
                 sample_state: Any, *, compile_mode: str = "default",
                 aot_artifact: str = "", checkpoint_sha256: str = ""):
        if sample_ids.device.type != "cuda" or sample_ids.ndim != 2:
            raise ValueError("prefill plans require CUDA ids shaped [batch,time]")
        self.model = model
        self.codec, _ = StateCodec.from_state(sample_state)
        self.static_ids = sample_ids.clone()
        self.compile_mode = compile_mode
        self.aot_artifact = str(aot_artifact)
        self.checkpoint_sha256 = checkpoint_sha256
        self.aot_loaded = False
        self.graph = torch.cuda.CUDAGraph()
        self.key = _architecture_key(
            model, sample_ids, self.codec.flatten(sample_state), kind="prefill",
            options={"compile_mode": compile_mode})
        self.compile_seconds = 0.0
        self._build()

    def _build(self) -> None:
        device = self.static_ids.device
        model, codec = self.model, self.codec

        def execute(ids):
            logits, state = model.forward_recurrent(ids)
            return (logits, *codec.flatten(state))

        started = time.perf_counter()
        if self.aot_artifact:
            artifact_path = Path(self.aot_artifact)
            manifest = json.loads(
                artifact_path.with_suffix(artifact_path.suffix + ".json").read_text())
            if (manifest.get("schema") != "rwkv-lab.megakernel-prefill-aot.v1"
                    or manifest.get("plan_sha256") != self.key
                    or manifest.get("checkpoint_sha256") != self.checkpoint_sha256
                    or manifest.get("artifact_sha256") != file_sha256(artifact_path)):
                raise ValueError("prefill AOT artifact identity does not match this plan")
            required = {
                "torch": torch.__version__,
                "triton": getattr(triton, "__version__", "unavailable"),
                "compute_capability": list(torch.cuda.get_device_capability(device)),
            }
            if any(manifest.get(key) != value for key, value in required.items()):
                raise ValueError("prefill AOT artifact runtime does not match")
            with artifact_path.open("rb") as artifact:
                self.compiled = torch.compiler.load_compiled_function(
                    artifact, f_globals=globals(), external_data={"model": model})
            self.aot_loaded = True
            self.fullgraph = True
        else:
            # FLA exposes its chunk kernel as a compiler-visible custom op in
            # supported builds, allowing the exact-shape prefill to share the
            # same AOT/fullgraph contract as decode.
            self.compiled = torch.compile(
                execute, mode=self.compile_mode, fullgraph=True, dynamic=False)
            self.fullgraph = True
        stream = torch.cuda.Stream(device=device)
        if (l2_controller := getattr(model, "_megakernel_l2_controller", None)) is not None:
            l2_controller.apply_to_stream(stream)
        stream.wait_stream(torch.cuda.current_stream(device))
        try:
            with torch.cuda.stream(stream), torch.no_grad():
                for _ in range(3):
                    self.compiled(self.static_ids)
        except torch._dynamo.exc.Unsupported:
            if self.aot_loaded:
                raise
            # Some FLA releases explicitly disable Dynamo around their chunk
            # scan. Keep compiled/captured exact-shape prefill, but report that
            # this runtime cannot export a fullgraph prefill artifact.
            self.fullgraph = False
            self.compiled = torch.compile(
                execute, mode=self.compile_mode, fullgraph=False, dynamic=False)
            with torch.cuda.stream(stream), torch.no_grad():
                for _ in range(3):
                    self.compiled(self.static_ids)
        torch.cuda.current_stream(device).wait_stream(stream)
        with torch.cuda.graph(self.graph), torch.no_grad():
            result = self.compiled(self.static_ids)
            self.static_logits = result[0]
            self.static_state = result[1:]
        torch.cuda.synchronize(device)
        self.compile_seconds = time.perf_counter() - started

    def replay(self, ids: torch.Tensor) -> tuple[torch.Tensor, Any]:
        if ids.shape != self.static_ids.shape or ids.dtype != self.static_ids.dtype:
            raise ValueError("prompt geometry differs from the compiled prefill plan")
        self.static_ids.copy_(ids)
        self.graph.replay()
        return self.static_logits, self.codec.rebuild(self.static_state)

    def export_aot(self, artifact_path: str | Path, *,
                   checkpoint_sha256: str) -> dict:
        if not self.fullgraph:
            raise RuntimeError(
                "prefill AOT requires an FLA build that is visible to fullgraph compilation")
        artifact_path = Path(artifact_path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        _set_fused_state_step(self.model, True, inplace=False)
        try:
            with torch.no_grad():
                aot = self.compiled.aot_compile(((self.static_ids,), {}))
        finally:
            _set_fused_state_step(self.model, False)
        aot.save_compiled_function(
            str(artifact_path), external_data={"model": self.model})
        manifest = {
            "schema": "rwkv-lab.megakernel-prefill-aot.v1",
            "artifact": artifact_path.name,
            "artifact_sha256": file_sha256(artifact_path),
            "checkpoint_sha256": checkpoint_sha256,
            "plan_sha256": self.key,
            "prompt_shape": list(self.static_ids.shape),
            "external_data": ["model"],
            "torch": torch.__version__,
            "triton": getattr(triton, "__version__", "unavailable"),
            "compute_capability": list(
                torch.cuda.get_device_capability(self.static_ids.device)),
        }
        artifact_path.with_suffix(artifact_path.suffix + ".json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return manifest


def prefill_artifact_path(decode_artifact: str | Path, batch: int,
                          time_tokens: int) -> Path:
    path = Path(decode_artifact)
    return path.with_name(
        f"{path.stem}.prefill-b{batch}-t{time_tokens}{path.suffix}")


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


def _prepare_folded_embedding(model: torch.nn.Module) -> None:
    """Fold block-0 ln0 into the immutable inference embedding table.

    Albatross performs the same B1T1 preparation before decode. Source:
    https://github.com/BlinkDL/Albatross (faster3b_2606 implementation).
    """
    if hasattr(model, "_megakernel_folded_embedding"):
        return
    blocks = getattr(model, "blocks", ())
    embedding = getattr(model, "emb", None)
    if embedding is None or not blocks or not hasattr(blocks[0], "ln0"):
        return
    with torch.no_grad():
        folded = blocks[0].ln0(embedding.weight).detach().contiguous()
    model.register_buffer("_megakernel_folded_embedding", folded, persistent=False)


def finalize_megakernel_serving_embedding(
    model: torch.nn.Module, *, require_adopted: bool = True,
) -> dict:
    """Destructively replace the inference embedding and release its duplicate.

    Call this only before constructing a backend/compiled plan. It preserves
    ordinary inference by permanently skipping block-0 ln0, but intentionally
    makes the in-memory model serving-only. DeepEmbed's embedding-residual mode
    still requires the raw table and therefore fails closed.
    """
    if model.training:
        raise ValueError("serving embedding finalization requires eval mode")
    if require_adopted and not getattr(model, "_megakernel_adopted", False):
        raise ValueError("serving embedding finalization requires an adopted backend")
    if getattr(model, "deepembed", False) and getattr(model, "de_emb_res", False):
        raise ValueError("DeepEmbed embedding-residual mode requires the raw embedding")
    if getattr(model, "_megakernel_embedding_finalized", False):
        return dict(model._megakernel_embedding_finalization)
    _prepare_folded_embedding(model)
    folded = getattr(model, "_megakernel_folded_embedding", None)
    embedding = getattr(model, "emb", None)
    blocks = getattr(model, "blocks", ())
    if folded is None or embedding is None or not blocks:
        raise ValueError("model does not expose a foldable block-0 embedding")
    reclaimed = folded.numel() * folded.element_size()
    with torch.no_grad():
        embedding.weight.copy_(folded)
    delattr(model, "_megakernel_folded_embedding")
    blocks[0]._megakernel_skip_ln0_permanent = True
    model._megakernel_use_folded_embedding = False
    model._megakernel_embedding_finalized = True
    receipt = {
        "schema": "rwkv-lab.megakernel-serving-preparation.v1",
        "folded_embedding": True, "serving_only": True,
        "reclaimed_duplicate_bytes": reclaimed,
        "source": "https://github.com/BlinkDL/Albatross/tree/main/faster3b_2606",
    }
    model._megakernel_embedding_finalization = receipt
    return dict(receipt)


@torch.no_grad()
def qualify_model_b1t1_kernels(model: torch.nn.Module, *, device: torch.device,
                               repeats: int = 10) -> dict:
    """Qualify optional projection/FFN candidates on representative weights."""
    from rwkv_lab.megakernel_linear import qualify_b1t1_kernels
    blocks = getattr(model, "blocks", ())
    if not blocks:
        return {"schema": "rwkv-lab.b1t1-linear-qualification.v1",
                "available": False, "adopted": False,
                "reason": "model has no native RWKV blocks"}
    block = blocks[0]
    attention = getattr(block, "att", None)
    ffn = getattr(block, "ffn", None)
    required = ("receptance", "key", "value", "output")
    if (attention is None or ffn is None
            or not all(hasattr(attention, name) for name in required)
            or not hasattr(ffn, "key") or not hasattr(ffn, "value")):
        return {"schema": "rwkv-lab.b1t1-linear-qualification.v1",
                "available": False, "adopted": False,
                "reason": "model block geometry is not a native dense RWKV block"}
    dtype = attention.receptance.weight.dtype
    channels = attention.receptance.weight.shape[1]
    sample = torch.randn(1, 1, channels, device=device, dtype=dtype)
    rkv_sample = torch.randn(1, 1, 3, channels, device=device, dtype=dtype)
    rkv_weights = torch.stack((attention.receptance.weight,
                               attention.key.weight,
                               attention.value.weight)).contiguous()
    report = qualify_b1t1_kernels(
        sample, attention.output.weight, rkv_weights,
        ffn.key.weight, ffn.value.weight, rkv_x=rkv_sample,
        repeats=max(3, repeats), min_speedup=1.0,
        atol=(3e-2 if dtype in (torch.float16, torch.bfloat16) else 2e-5),
        rtol=(3e-2 if dtype in (torch.float16, torch.bfloat16) else 2e-5),
    )
    candidates = report.get("candidates", {})
    row_adopted = bool(candidates.get("row_one", {}).get("adopted"))
    rkv_adopted = bool(candidates.get("packed_rkv", {}).get("adopted"))
    ffn_adopted = bool(candidates.get("ffn_sqrelu_value", {}).get("adopted"))
    for candidate_block in blocks:
        candidate_attention = getattr(candidate_block, "att", None)
        candidate_ffn = getattr(candidate_block, "ffn", None)
        if candidate_attention is not None:
            candidate_attention._megakernel_row_one = row_adopted
            candidate_attention._megakernel_packed_rkv = rkv_adopted
            if rkv_adopted and not hasattr(candidate_attention, "_megakernel_rkv_weight"):
                packed = torch.stack((candidate_attention.receptance.weight,
                                      candidate_attention.key.weight,
                                      candidate_attention.value.weight)).detach().contiguous()
                candidate_attention.register_buffer(
                    "_megakernel_rkv_weight", packed, persistent=False)
        if candidate_ffn is not None:
            candidate_ffn._megakernel_ffn_candidate = ffn_adopted
    model._megakernel_row_one = row_adopted
    model._megakernel_b1t1_qualification = report
    return report


@torch.no_grad()
def qualify_sm120_features(model: torch.nn.Module, prompt_ids: Sequence[int], *,
                           device: torch.device, repeats: int = 10) -> dict:
    """Gate the SM120 scheduler and L2 policy; report native CUTLASS honestly."""
    from rwkv_lab.sm120_kernels import (
        L2PersistenceController, cutlass_sm120_status,
        qualify_persistent_state, sm120_profile,
    )
    profile = sm120_profile(device)
    report = {
        "schema": "rwkv-lab.sm120-megakernel-qualification.v1",
        "available": profile.available, "profile": profile.__dict__,
        "cutlass_projection": cutlass_sm120_status(),
        "persistent_scheduler": {"available": False, "adopted": False},
        "l2_persistence": {"available": False, "adopted": False},
    }
    if not profile.available:
        report["reason"] = profile.reason
        return report
    blocks = getattr(model, "blocks", ())
    attention = getattr(blocks[0], "att", None) if blocks else None
    if attention is not None:
        heads = int(getattr(attention, "num_heads", 0))
        width = int(getattr(attention, "head_size", 0))
        dtype = attention.receptance.weight.dtype
        if heads and width:
            shape = (1, 1, heads, width)
            generator = torch.Generator(device=device).manual_seed(120)
            inputs = [torch.randn(shape, device=device, dtype=dtype,
                                  generator=generator) for _ in range(6)]
            inputs[1] = -inputs[1].float().abs().clamp_max(4).to(dtype)
            state = torch.randn((1, heads, width, width), device=device,
                                dtype=torch.float32, generator=generator) * 0.02
            report["persistent_scheduler"] = qualify_persistent_state(
                *inputs, state, repeats=max(3, repeats), minimum_speedup=1.0)
    persistent_adopted = bool(report["persistent_scheduler"].get("adopted"))
    model._megakernel_persistent_sm120_adopted = persistent_adopted

    hot_tensor, hot_name = None, ""
    for index, block in enumerate(blocks):
        candidate = getattr(getattr(block, "att", None),
                            "_megakernel_rkv_weight", None)
        if candidate is not None:
            hot_tensor, hot_name = candidate, f"blocks.{index}.att.packed_rkv"
            break
    if hot_tensor is None:
        hot_tensor = getattr(model, "_megakernel_folded_embedding", None)
        hot_name = "folded_embedding"
    if hot_tensor is not None:
        controller = L2PersistenceController(device)
        try:
            prompt = torch.tensor([list(prompt_ids)], device=device, dtype=torch.long)
            token = prompt[:, -1:]
            _, state = model.forward_recurrent(prompt)

            def work():
                return model.forward_recurrent(token, state)[0]

            expected = work()
            baseline_us = _cuda_median_us(work, device, repeats=max(3, repeats))
            window = controller.apply(hot_tensor, name=hot_name)
            actual = work()
            candidate_us = _cuda_median_us(work, device, repeats=max(3, repeats))
            speedup = baseline_us / max(candidate_us, 1e-12)
            parity = bool(torch.allclose(actual, expected, atol=8e-2, rtol=3e-2))
            adopted = bool(window.get("adopted") and parity and speedup >= 1.0)
            report["l2_persistence"] = {
                **window, "parity": parity, "baseline_median_us": baseline_us,
                "candidate_median_us": candidate_us, "speedup": speedup,
                "adopted": adopted,
                "source": "https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#l2-access-properties",
            }
            if adopted:
                model._megakernel_l2_controller = controller
            else:
                controller.clear()
        except Exception as exc:
            controller.clear()
            report["l2_persistence"] = {
                "available": False, "adopted": False, "error": repr(exc)}
    report["adopted"] = bool(
        persistent_adopted or report["l2_persistence"].get("adopted")
        or report["cutlass_projection"].get("available"))
    model._megakernel_sm120_qualification = report
    return report


def _set_fused_state_step(model: torch.nn.Module, enabled: bool, *,
                          inplace: bool | None = None) -> None:
    model._megakernel_use_folded_embedding = enabled
    model._megakernel_final_norm = (
        enabled and getattr(model, "_megakernel_boundaries_adopted", False))
    blocks = getattr(model, "blocks", ())
    if blocks and hasattr(blocks[0], "ln0"):
        blocks[0]._megakernel_skip_ln0 = (
            enabled or getattr(blocks[0], "_megakernel_skip_ln0_permanent", False))
    for block in blocks:
        block._megakernel_boundaries = (
            enabled and getattr(model, "_megakernel_boundaries_adopted", False))
    for module in model.modules():
        if type(module).__name__ == "RWKV8TimeMixDeltaNet":
            module._megakernel_recurrent = enabled
            module._megakernel_combined_epilogue = (
                enabled and getattr(model, "_megakernel_combined_adopted", False))
            module._megakernel_persistent_sm120 = (
                enabled and getattr(model, "_megakernel_persistent_sm120_adopted", False))
            module._megakernel_inplace_state = enabled if inplace is None else inplace


@torch.no_grad()
def qualify_fusion_features(model: torch.nn.Module, prompt_ids: Sequence[int], *,
                            device: torch.device, repeats: int = 10,
                            minimum_speedup: float = 1.0) -> dict:
    """Gate combined-state and portable boundary fusions independently."""
    prompt = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    token = prompt[:, -1:].clone()
    _set_fused_state_step(model, False)
    _, state = model.forward_recurrent(prompt)

    def configure(*, combined: bool, boundaries: bool) -> None:
        model._megakernel_combined_adopted = combined
        model._megakernel_boundaries_adopted = boundaries
        _set_fused_state_step(model, True, inplace=False)

    def measure() -> tuple[torch.Tensor, Any, float]:
        logits, next_state = model.forward_recurrent(token, state)
        elapsed = _cuda_median_us(
            lambda: model.forward_recurrent(token, state), device,
            repeats=max(3, repeats))
        return logits, next_state, elapsed

    try:
        configure(combined=False, boundaries=False)
        expected, _, baseline_us = measure()
        configure(combined=True, boundaries=False)
        combined_logits, _, combined_us = measure()
        combined_parity = bool(torch.allclose(
            combined_logits, expected, atol=8e-2, rtol=3e-2))
        combined_tokens = bool(torch.equal(
            combined_logits.argmax(-1), expected.argmax(-1)))
        combined_speedup = baseline_us / max(combined_us, 1e-12)
        combined_adopted = bool(
            combined_parity and combined_tokens
            and combined_speedup >= minimum_speedup)

        configure(combined=combined_adopted, boundaries=True)
        boundary_logits, _, boundary_us = measure()
        boundary_parity = bool(torch.allclose(
            boundary_logits, expected, atol=8e-2, rtol=3e-2))
        boundary_tokens = bool(torch.equal(
            boundary_logits.argmax(-1), expected.argmax(-1)))
        boundary_baseline = combined_us if combined_adopted else baseline_us
        boundary_speedup = boundary_baseline / max(boundary_us, 1e-12)
        boundaries_adopted = bool(
            boundary_parity and boundary_tokens
            and boundary_speedup >= minimum_speedup)
    finally:
        _set_fused_state_step(model, False)
    model._megakernel_combined_adopted = combined_adopted
    model._megakernel_boundaries_adopted = boundaries_adopted
    report = {
        "schema": "rwkv-lab.megakernel-fusion-qualification.v1",
        "baseline_median_us": baseline_us,
        "minimum_speedup": minimum_speedup,
        "combined_state_epilogue": {
            "parity": combined_parity, "exact_argmax": combined_tokens,
            "median_us": combined_us, "speedup": combined_speedup,
            "adopted": combined_adopted,
        },
        "layer_boundaries": {
            "parity": boundary_parity, "exact_argmax": boundary_tokens,
            "median_us": boundary_us, "speedup": boundary_speedup,
            "adopted": boundaries_adopted,
        },
        "adopted": bool(combined_adopted or boundaries_adopted),
        "source": "https://github.com/BlinkDL/Albatross/tree/main/faster3b_2606",
    }
    model._megakernel_fusion_qualification = report
    return report


class MegakernelBackend:
    """Own prefill plus cached one-token CUDA Graph plans for one model."""

    def __init__(self, model: torch.nn.Module, *, device: str | torch.device,
                 compile_mode: str = "max-autotune-no-cudagraphs",
                 aot_artifact: str = ""):
        reason = megakernel_incompatibility(model, device)
        if reason:
            raise RuntimeError(f"megakernel backend unavailable: {reason}")
        self.model = model
        _prepare_folded_embedding(model)
        self.device = torch.device(device)
        self.compile_mode = compile_mode
        self.aot_artifact = str(aot_artifact or getattr(model, "_megakernel_artifact", ""))
        self.checkpoint_sha256 = ""
        if self.aot_artifact:
            decode_path = Path(self.aot_artifact)
            manifest_path = decode_path.with_suffix(decode_path.suffix + ".json")
            if manifest_path.exists():
                self.checkpoint_sha256 = str(
                    json.loads(manifest_path.read_text()).get("checkpoint_sha256", ""))
        self.plans: dict[tuple, CUDAGraphDecodePlan] = {}
        self.greedy_plans: dict[tuple, CUDAGraphGreedyPlan] = {}
        self.prefill_plans: dict[tuple, CUDAGraphPrefillPlan] = {}
        self.plan: CUDAGraphDecodePlan | None = None

    def prefill(self, ids: torch.Tensor, *, greedy_feedback: bool = False) -> torch.Tensor:
        prefill_key = (ids.shape[0], ids.shape[1], str(ids.dtype))
        prefill_plan = self.prefill_plans.get(prefill_key)
        if prefill_plan is None:
            with torch.no_grad():
                _, sample_state = self.model.forward_recurrent(ids)
            _set_fused_state_step(self.model, True, inplace=False)
            try:
                candidate = (prefill_artifact_path(
                    self.aot_artifact, ids.shape[0], ids.shape[1])
                    if self.aot_artifact else None)
                prefill_plan = CUDAGraphPrefillPlan(
                    self.model, ids, sample_state, compile_mode="default",
                    aot_artifact=(str(candidate) if candidate and candidate.exists() else ""),
                    checkpoint_sha256=self.checkpoint_sha256)
            finally:
                _set_fused_state_step(self.model, False)
            self.prefill_plans[prefill_key] = prefill_plan
        logits, state = prefill_plan.replay(ids)
        sample = ids[:, -1:].clone()
        codec, leaves = StateCodec.from_state(state)
        key = (sample.shape[0], str(sample.dtype), bool(greedy_feedback), tuple(
            (tuple(t.shape), str(t.dtype)) for t in leaves))
        plan = self.plans.get(key)
        if plan is None:
            _set_fused_state_step(self.model, True)
            try:
                plan = CUDAGraphDecodePlan(
                    self.model, sample, state, compile_mode=self.compile_mode,
                    greedy_feedback=greedy_feedback,
                    aot_artifact=(self.aot_artifact if greedy_feedback else ""))
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

    def generate_greedy(self, ids: torch.Tensor, *, max_new: int,
                        stop_token_id: int | None = None) -> torch.Tensor:
        logits = self.prefill(ids, greedy_feedback=True)
        if max_new < 1:
            return ids.new_empty((ids.shape[0], 0))
        first = logits[:, -1].float().argmax(dim=-1, keepdim=True)
        plan = self.plan
        assert plan is not None
        state = plan.codec.rebuild(plan.static_state)
        key = (plan.key, int(max_new), stop_token_id)
        greedy_plan = self.greedy_plans.get(key)
        if greedy_plan is None:
            greedy_plan = CUDAGraphGreedyPlan(
                plan, state, max_new=max_new, stop_token_id=stop_token_id)
            self.greedy_plans[key] = greedy_plan
        else:
            greedy_plan.load_state(state)
        return greedy_plan.replay(first)

    def receipt(self) -> dict:
        plan = self.plan
        return {
            "schema": "rwkv-lab.megakernel-plan.v1",
            "available": True,
            "backend": "triton+inductor+cudagraph",
            "plan_sha256": plan.key if plan else "",
            "compile_seconds": plan.compile_seconds if plan else 0.0,
            "cached_plans": len(self.plans),
            "cached_greedy_plans": len(self.greedy_plans),
            "greedy_capture_seconds": sum(
                item.capture_seconds for item in self.greedy_plans.values()),
            "cached_prefill_plans": len(self.prefill_plans),
            "prefill_compile_seconds": sum(
                item.compile_seconds for item in self.prefill_plans.values()),
            "prefill_aot_loaded": sum(
                bool(item.aot_loaded) for item in self.prefill_plans.values()),
            "prefill_fullgraph_plans": sum(
                bool(item.fullgraph) for item in self.prefill_plans.values()),
            "state_tuning": {
                "key": ["compute_capability", "batch", "heads", "key_width",
                        "value_width"],
                "block_v": [8, 16, 32, 64, 128],
                "warps": [2, 4, 4, 8, 8],
                "cached_results": len(getattr(
                    globals().get("_rwkv7_state_step_kernel"), "cache", {})),
            },
            "b1t1_kernels": getattr(
                self.model, "_megakernel_b1t1_qualification", {
                    "available": False, "adopted": False,
                    "reason": "not run outside production qualification"}),
            "sm120_features": getattr(
                self.model, "_megakernel_sm120_qualification", {
                    "available": False, "adopted": False,
                    "reason": "not run outside production qualification"}),
            "fusion_qualification": getattr(
                self.model, "_megakernel_fusion_qualification", {
                    "available": False, "adopted": False,
                    "reason": "not run outside production qualification"}),
            "fused_boundaries": (
                (["state+groupnorm+bonus+gate"]
                 if getattr(self.model, "_megakernel_combined_adopted", False) else [])
                + (["ln1+six_mix", "attention_add+ln2+channel_mix",
                    "final_add+ln_out"]
                   if getattr(self.model, "_megakernel_boundaries_adopted", False) else [])),
            "aot_exportable": bool(plan is not None),
            "aot_loaded": bool(plan and plan.aot_loaded),
            "triton": getattr(triton, "__version__", "unavailable"),
            "torch": torch.__version__,
            "device": str(self.device),
        }


def get_megakernel_backend(model: torch.nn.Module, *, device: str | torch.device,
                           compile_mode: str = "max-autotune-no-cudagraphs",
                           aot_artifact: str = "") -> MegakernelBackend:
    aot_artifact = str(aot_artifact or getattr(model, "_megakernel_artifact", ""))
    backend = getattr(model, "_megakernel_backend", None)
    if (backend is None or backend.device != torch.device(device)
            or backend.compile_mode != compile_mode
            or backend.aot_artifact != aot_artifact):
        backend = MegakernelBackend(
            model, device=device, compile_mode=compile_mode,
            aot_artifact=aot_artifact)
        model._megakernel_backend = backend
    return backend


def _cuda_kernel_evidence(work: Callable[[], Any], device: torch.device) -> dict:
    from torch.profiler import ProfilerActivity, profile
    work()
    torch.cuda.synchronize(device)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 profile_memory=True) as trace:
        work()
        torch.cuda.synchronize(device)
    events = trace.events()
    cuda_events = [event for event in events
                   if event.device_type == torch.autograd.DeviceType.CUDA]
    kernels: dict[str, dict[str, Any]] = {}
    for event in cuda_events:
        row = kernels.setdefault(event.name, {"name": event.name, "calls": 0,
                                              "total_us": 0.0, "max_us": 0.0})
        elapsed = float(getattr(event, "device_time_total", 0.0))
        row["calls"] += 1
        row["total_us"] += elapsed
        row["max_us"] = max(row["max_us"], elapsed)
    return {
        "cuda_kernels": len(cuda_events),
        "cuda_time_us": sum(float(getattr(event, "device_time_total", 0.0))
                            for event in cuda_events),
        "positive_device_allocation_bytes": sum(
            max(0, int(getattr(event, "device_memory_usage", 0))) for event in events),
        "top_cuda_kernels": sorted(
            kernels.values(), key=lambda row: row["total_us"], reverse=True)[:12],
    }


def _cuda_launch_count(work: Callable[[], Any], device: torch.device) -> int:
    return int(_cuda_kernel_evidence(work, device)["cuda_kernels"])


def _cuda_median_us(work: Callable[[], Any], device: torch.device, *,
                    repeats: int) -> float:
    for _ in range(3):
        work()
    torch.cuda.synchronize(device)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(max(1, repeats))]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(max(1, repeats))]
    for start, end in zip(starts, ends):
        start.record()
        work()
        end.record()
    torch.cuda.synchronize(device)
    values = sorted(start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends))
    return values[len(values) // 2]


@torch.no_grad()
def benchmark_execution_paths(model: torch.nn.Module, prompt_ids: Sequence[int], *,
                              backend: MegakernelBackend, device: str = "cuda",
                              repeats: int = 20) -> dict:
    """Attribute latency and launches across each optimization layer."""
    selected = torch.device(device)
    prompt = torch.tensor([list(prompt_ids)], dtype=torch.long, device=selected)
    token = prompt[:, -1:].clone()
    _set_fused_state_step(model, False)
    with torch.no_grad():
        _, state = model.forward_recurrent(prompt)
    codec, leaves = StateCodec.from_state(state)

    def reference_step():
        logits, _ = model.forward_recurrent(token, state)
        return logits[:, -1].float().argmax(dim=-1, keepdim=True)

    reference_us = _cuda_median_us(reference_step, selected, repeats=repeats)
    reference_evidence = _cuda_kernel_evidence(reference_step, selected)

    _set_fused_state_step(model, True, inplace=False)
    try:
        def fused_step():
            logits, _ = model.forward_recurrent(token, state)
            return logits[:, -1].float().argmax(dim=-1, keepdim=True)

        fused_us = _cuda_median_us(fused_step, selected, repeats=repeats)
        fused_evidence = _cuda_kernel_evidence(fused_step, selected)

        def compiled_execute(ids, *flat_state):
            logits, next_state = model.forward_recurrent(ids, codec.rebuild(flat_state))
            next_ids = logits[:, -1].float().argmax(dim=-1, keepdim=True)
            return (next_ids, *codec.flatten(next_state))

        compiled = torch.compile(
            compiled_execute, mode=backend.compile_mode, fullgraph=True, dynamic=False)

        def compiled_step():
            return compiled(token, *leaves)

        compiled_us = _cuda_median_us(compiled_step, selected, repeats=repeats)
        compiled_evidence = _cuda_kernel_evidence(compiled_step, selected)
    finally:
        _set_fused_state_step(model, False)

    backend.prefill(prompt, greedy_feedback=True)
    def graph_work():
        return backend.plan.replay(backend.plan.static_ids)
    graph_us = _cuda_median_us(graph_work, selected, repeats=repeats)
    graph_evidence = _cuda_kernel_evidence(graph_work, selected)
    rows = {
        "eager_reference": {"median_us": reference_us, **reference_evidence},
        "fused_state_epilogue": {"median_us": fused_us, **fused_evidence},
        "compiled_fullgraph": {"median_us": compiled_us, **compiled_evidence},
        "cuda_graph": {"median_us": graph_us, **graph_evidence},
    }
    for row in rows.values():
        row["speedup_vs_eager"] = reference_us / max(row["median_us"], 1e-12)
    return {"schema": "rwkv-lab.megakernel-ablation.v1", "repeats": repeats,
            "paths": rows}


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

    b1t1_kernels = qualify_model_b1t1_kernels(
        model, device=torch.device(device), repeats=max(3, min(repeats * 2, 20)))
    _prepare_folded_embedding(model)
    sm120_features = qualify_sm120_features(
        model, prompt_ids, device=torch.device(device),
        repeats=max(3, min(repeats * 2, 20)))
    fusion_features = qualify_fusion_features(
        model, prompt_ids, device=torch.device(device),
        repeats=max(3, min(repeats * 2, 20)))
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
    ablation = benchmark_execution_paths(
        model, prompt_ids, backend=backend, device=device,
        repeats=max(3, min(20, repeats * 3)))
    launches_before = ablation["paths"]["eager_reference"]["cuda_kernels"]
    launches_after = ablation["paths"]["cuda_graph"]["cuda_kernels"]
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
        "b1t1_kernels": b1t1_kernels,
        "sm120_features": sm120_features,
        "fusion_features": fusion_features,
        "ablation": ablation,
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


def adopt_megakernel_artifact(model: torch.nn.Module, artifact_path: str | Path,
                              checkpoint_path: str | Path) -> dict:
    artifact_path = Path(artifact_path)
    manifest_path = artifact_path.with_suffix(artifact_path.suffix + ".json")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "rwkv-lab.megakernel-aot.v1":
        raise ValueError("megakernel artifact manifest has an unsupported schema")
    if file_sha256(artifact_path) != manifest.get("artifact_sha256"):
        raise ValueError("megakernel artifact hash does not match its manifest")
    if file_sha256(checkpoint_path) != manifest.get("checkpoint_sha256"):
        raise ValueError("megakernel artifact checkpoint hash does not match")
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        raise ValueError("cannot load a megakernel artifact for a parameterless model") from None
    required = {
        "compute_capability": list(torch.cuda.get_device_capability(model_device)),
        "torch": torch.__version__,
        "triton": getattr(triton, "__version__", "unavailable"),
    }
    for key, current in required.items():
        if manifest.get(key) != current:
            raise ValueError(f"megakernel artifact {key} does not match this runtime")
    model._megakernel_artifact = str(artifact_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile and export a checkpoint-bound RWKV megakernel plan")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt-ids", default="1,2,3,4")
    parser.add_argument("--compile-mode",
                        choices=("default", "max-autotune-no-cudagraphs"),
                        default="max-autotune-no-cudagraphs")
    args = parser.parse_args()
    from rwkv_lab.generate import build_from_ckpt
    model, _ = build_from_ckpt(args.checkpoint, args.device)
    prompt_ids = [int(value) for value in args.prompt_ids.split(",") if value.strip()]
    if not prompt_ids:
        raise SystemExit("--prompt-ids must contain at least one integer token")
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=args.device)
    backend = MegakernelBackend(
        model, device=args.device, compile_mode=args.compile_mode)
    backend.prefill(prompt, greedy_feedback=True)
    manifest = backend.plan.export_aot(
        args.artifact, checkpoint_sha256=file_sha256(args.checkpoint))
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
