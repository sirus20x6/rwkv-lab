"""Test-time learned associative memory for recurrent language models.

Primary references:

* Behrouz et al., "Titans: Learning to Memorize at Test Time",
  arXiv:2501.00663, https://arxiv.org/abs/2501.00663.
* Behrouz et al., "It's All Connected" (the MIRAS design framework),
  arXiv:2504.13173, https://arxiv.org/abs/2504.13173.
* Behrouz et al., "ATLAS: Learning to Optimally Memorize the Context at Test
  Time", arXiv:2505.23735, https://arxiv.org/abs/2505.23735.
* Behrouz et al., "Nested Learning: The Illusion of Deep Learning
  Architectures", arXiv:2512.24695, https://arxiv.org/abs/2512.24695.
* Lee et al., "Do Language Models Need Sleep? Offline Recurrence for Improved
  Online Inference", arXiv:2605.26099, https://arxiv.org/abs/2605.26099.

The module implements their shared model-side mechanism: an associative-memory
matrix is optimized *inside the forward pass*.  MIRAS choices are explicit:
the memory model, attentional-bias loss, retention rule, and learning rule.
``atlas`` mode adds a short window of past key/value pairs to the update rather
than optimizing only against the current token. ``nested`` mode adds a learned
controller that adjusts update rate and retention from the current surprise.

This CPU-readable path uses an exact sequential scan. It is intentionally a
reference implementation for A/B validation before a chunk-parallel kernel.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import copy
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class OnlineMemoryState:
    memory: torch.Tensor
    momentum: torch.Tensor
    steps: int = 0
    history: tuple[tuple[torch.Tensor, torch.Tensor], ...] = ()

    def detach(self) -> "OnlineMemoryState":
        return OnlineMemoryState(
            self.memory.detach(), self.momentum.detach(), self.steps,
            tuple((key.detach(), value.detach()) for key, value in self.history),
        )


class OnlineAssociativeMemory(nn.Module):
    """Differentiable in-forward memory update with Titans/MIRAS/ATLAS modes."""

    MODES = ("titans", "miras", "atlas", "nested")

    def __init__(self, d_model: int, *, d_memory: int | None = None, mode: str = "titans",
                 learning_rate: float = 0.05, retention: float = 0.99,
                 momentum: float = 0.9, atlas_window: int = 4, detach_updates: bool = False):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.d_model = int(d_model)
        self.d_memory = int(d_memory or min(d_model, 128))
        self.mode = mode
        self.learning_rate = float(learning_rate)
        self.retention = float(retention)
        self.momentum_beta = float(momentum)
        self.atlas_window = max(1, int(atlas_window))
        self.detach_updates = bool(detach_updates)

        self.norm = nn.RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, self.d_memory, bias=False)
        self.k_proj = nn.Linear(d_model, self.d_memory, bias=False)
        self.v_proj = nn.Linear(d_model, self.d_memory, bias=False)
        self.out_proj = nn.Linear(self.d_memory, d_model, bias=False)
        self.gate = nn.Linear(d_model, 1)
        # Identity-at-init makes this a clean off-by-default lever.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.constant_(self.gate.bias, -4.0)
        self.controller = nn.Sequential(nn.Linear(1, 8), nn.SiLU(), nn.Linear(8, 2)) \
            if mode == "nested" else None
        if self.controller is not None:
            nn.init.zeros_(self.controller[-1].weight)
            nn.init.zeros_(self.controller[-1].bias)
        self.last_stats: dict[str, float] = {}

    def initial_state(self, batch: int, *, device=None, dtype=torch.float32) -> OnlineMemoryState:
        z = torch.zeros(batch, self.d_memory, self.d_memory, device=device, dtype=dtype)
        return OnlineMemoryState(z, torch.zeros_like(z), 0)

    def _bias_error(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # MIRAS treats the internal objective as an architectural choice.
        if self.mode == "miras":
            pn, tn = F.normalize(pred, dim=-1), F.normalize(target, dim=-1)
            return tn - pn
        return target - pred

    def forward(self, hidden: torch.Tensor, state: OnlineMemoryState | None = None,
                *, return_state: bool = False, record_stats: bool = True):
        if hidden.ndim != 3:
            raise ValueError("hidden must have shape [batch,time,channels]")
        B, T, _ = hidden.shape
        x = self.norm(hidden)
        q = F.normalize(self.q_proj(x).float(), dim=-1)
        k = F.normalize(self.k_proj(x).float(), dim=-1)
        v = self.v_proj(x).float()
        st = state or self.initial_state(B, device=hidden.device)
        memory, mom = st.memory.float(), st.momentum.float()
        if memory.shape != (B, self.d_memory, self.d_memory):
            raise ValueError("online-memory state geometry does not match this module")

        outputs, surprises = [], []
        history: deque[tuple[torch.Tensor, torch.Tensor]] = deque(
            st.history, maxlen=self.atlas_window)
        for t in range(T):
            qt, kt, vt = q[:, t], k[:, t], v[:, t]
            pred = torch.einsum("bi,bij->bj", qt, memory)
            err = self._bias_error(pred, vt)
            surprise = err.square().mean(-1, keepdim=True).sqrt()
            outputs.append(pred)
            surprises.append(surprise)

            history.append((kt, vt))
            pairs = list(history) if self.mode == "atlas" else [(kt, vt)]
            grad = torch.zeros_like(memory)
            for kh, vh in pairs:
                ph = torch.einsum("bi,bij->bj", kh, memory)
                eh = self._bias_error(ph, vh)
                grad = grad + torch.einsum("bi,bj->bij", kh, eh)
            grad = grad / len(pairs)

            eta = hidden.new_full((B, 1, 1), self.learning_rate, dtype=torch.float32)
            retain = hidden.new_full((B, 1, 1), self.retention, dtype=torch.float32)
            if self.controller is not None:
                ctrl = self.controller(surprise.to(hidden.dtype)).float()
                eta = eta * (2.0 * torch.sigmoid(ctrl[:, :1, None]))
                retain = 1.0 - (1.0 - retain) * (2.0 * torch.sigmoid(ctrl[:, 1:, None]))
            mom = self.momentum_beta * mom + (1.0 - self.momentum_beta) * grad
            memory = retain * memory + eta * mom
            if self.detach_updates:
                memory, mom = memory.detach(), mom.detach()

        recalled = torch.stack(outputs, dim=1).to(hidden.dtype)
        gate = torch.sigmoid(self.gate(x))
        out = hidden + gate * self.out_proj(recalled)
        ss = torch.cat(surprises, dim=1)
        if record_stats:
            self.last_stats = {
                "surprise_mean": float(ss.detach().mean()),
                "memory_norm": float(memory.detach().norm(dim=(-2, -1)).mean()),
                "gate_mean": float(gate.detach().mean()),
            }
        next_state = OnlineMemoryState(memory, mom, st.steps + T, tuple(history))
        return (out, next_state) if return_state else out

    def sleep_consolidate(self, context: torch.Tensor, state: OnlineMemoryState | None = None,
                          *, passes: int = 1, clear_history: bool = True) -> OnlineMemoryState:
        """Run bounded offline recurrent consolidation without changing wake outputs.

        Inspired by Lee et al. (2026), https://arxiv.org/abs/2605.26099. Their
        sleep phase repeatedly revisits recent context to update persistent fast
        weights, shifting extra compute away from latency-sensitive wake inference.
        This reference implementation reuses the same cited local update rule as
        the live memory and returns an explicit state; callers own scheduling and
        budgets. ``passes=0`` is an exact no-op.
        """
        if passes < 0:
            raise ValueError("sleep consolidation passes must be non-negative")
        current = state or self.initial_state(context.shape[0], device=context.device)
        for _ in range(int(passes)):
            _, current = self(context, current, return_state=True, record_stats=False)
        if clear_history and current.history:
            current = OnlineMemoryState(current.memory, current.momentum, current.steps, ())
        return current


class SleepConsolidator:
    """Bounded context buffer and scheduler for offline memory consolidation."""

    def __init__(self, memory: OnlineAssociativeMemory, *, interval: int = 1024,
                 passes: int = 2, max_context: int = 4096):
        if interval < 1 or passes < 1 or max_context < 1:
            raise ValueError("sleep interval, passes, and context bound must be positive")
        self.memory, self.interval, self.passes = memory, int(interval), int(passes)
        self.max_context = int(max_context)
        self._chunks: list[torch.Tensor] = []
        self._tokens = 0            # retained tokens (buffer occupancy)
        self._observed = 0          # tokens seen since last consolidate — drives due()

    def observe(self, hidden: torch.Tensor) -> None:
        chunk = hidden.detach()
        self._observed += chunk.shape[1]
        if chunk.shape[1] > self.max_context:              # oversize: keep the newest tokens
            chunk = chunk[:, -self.max_context:]
        self._chunks.append(chunk)
        while len(self._chunks) > 1 and sum(c.shape[1] for c in self._chunks) > self.max_context:
            self._chunks.pop(0)
        self._tokens = sum(c.shape[1] for c in self._chunks)   # count only retained tokens

    def due(self) -> bool:
        # Cumulative observed count, not retained-buffer occupancy: occupancy
        # is capped at max_context, so interval > max_context would otherwise
        # make due() permanently False and consolidation silently never run.
        return self._observed >= self.interval and bool(self._chunks)

    def consolidate(self, state: OnlineMemoryState | None = None) -> OnlineMemoryState:
        if not self._chunks:
            raise ValueError("sleep consolidator has no observed context")
        context = torch.cat(self._chunks, dim=1)[:, -self.max_context:]
        result = self.memory.sleep_consolidate(context, state, passes=self.passes)
        self._chunks.clear(); self._tokens = 0; self._observed = 0
        return result


def install_online_memory(model: nn.Module, **kwargs) -> OnlineAssociativeMemory:
    """Register a final-hidden online memory on an RWKV-Lab model."""

    d_model = int(getattr(getattr(model, "head", None), "in_features", 0))
    if not d_model:
        raise ValueError("model must expose head.in_features")
    mem = OnlineAssociativeMemory(d_model, **kwargs)
    p = next(model.parameters())
    mem.to(device=p.device, dtype=p.dtype)
    model.online_memory = mem
    return mem


class _OnlineMemoryKernel(nn.Module):
    """Tensor-only wrapper suitable for ``torch.compile(fullgraph=True)``."""

    def __init__(self, memory: OnlineAssociativeMemory):
        super().__init__()
        self.memory = memory

    def forward(self, hidden: torch.Tensor, matrix: torch.Tensor, momentum: torch.Tensor,
                history_k: torch.Tensor | None = None, history_v: torch.Tensor | None = None):
        """``history_k``/``history_v`` ([n, batch, d_memory], n <= atlas_window) carry the
        ATLAS sliding window across chunks; omit them ONLY for a fresh state (otherwise
        mode="atlas" silently restarts the window and diverges from eager). When passed
        (empty tensors are fine for chunk 0), the updated window is returned as two extra
        outputs to thread into the next call."""
        history: tuple[tuple[torch.Tensor, torch.Tensor], ...] = ()
        if history_k is not None:
            history = tuple((history_k[i], history_v[i]) for i in range(history_k.shape[0]))
        output, state = self.memory(
            hidden, OnlineMemoryState(matrix, momentum, 0, history), return_state=True,
            record_stats=False)
        if history_k is None:
            return output, state.memory, state.momentum
        if state.history:
            new_k = torch.stack([key for key, _ in state.history])
            new_v = torch.stack([value for _, value in state.history])
        else:
            new_k, new_v = history_k[:0], history_v[:0]
        return output, state.memory, state.momentum, new_k, new_v


def compile_online_memory(memory: OnlineAssociativeMemory, *, backend: str = "inductor"):
    """Compile the live module while retaining the eager stateful path as oracle."""
    return torch.compile(_OnlineMemoryKernel(memory), backend=backend, fullgraph=True)


def install_compiled_online_memory(model: nn.Module, *, backend: str = "inductor"):
    """Attach a compiled callable without registering duplicate module parameters."""
    memory = getattr(model, "online_memory", None)
    if not isinstance(memory, OnlineAssociativeMemory):
        raise ValueError("model has no OnlineAssociativeMemory to compile")
    kernel = compile_online_memory(memory, backend=backend)
    object.__setattr__(model, "_online_memory_kernel", kernel)
    return kernel


def qualify_compiled_online_memory(memory: OnlineAssociativeMemory, sample: torch.Tensor, *,
                                   compile_backend: str = "inductor",
                                   tolerance: float = 2e-5,
                                   minimum_speedup: float = 1.02,
                                   repeats: int = 10) -> dict:
    """Gate a compiled update scan on output/state/gradient parity before speed.

    ``torch.compile`` is the supported PyTorch graph compiler rather than a
    handwritten kernel ABI: https://docs.pytorch.org/docs/stable/generated/torch.compile.html
    Fixed serving chunk sizes avoid recompilation and let Inductor fuse the
    projection/update scan while the eager module remains the correctness oracle.
    """

    if sample.ndim != 3:
        raise ValueError("online-memory qualification sample must be [batch,time,channels]")
    try:
        eager = _OnlineMemoryKernel(copy.deepcopy(memory)).to(sample.device)
        candidate_base = _OnlineMemoryKernel(copy.deepcopy(memory)).to(sample.device)
        candidate = compile_online_memory(candidate_base.memory, backend=compile_backend)
    except Exception as exc:
        return {"schema": "rwkv-lab.online-memory-kernel-qualification.v1",
                "backend": compile_backend, "available": False, "adopted": False,
                "error": repr(exc)}
    state = memory.initial_state(sample.shape[0], device=sample.device)
    x0 = sample.detach().clone().requires_grad_(True)
    x1 = sample.detach().clone().requires_grad_(True)
    inputs0 = (x0, state.memory.clone(), state.momentum.clone())
    inputs1 = (x1, state.memory.clone(), state.momentum.clone())
    try:
        out0, out1 = eager(*inputs0), candidate(*inputs1)
        output_error = max(float((left.detach().float() - right.detach().float()).abs().max())
                           for left, right in zip(out0, out1))
        params0 = tuple(eager.parameters())
        params1 = tuple(candidate_base.parameters())
        grad0 = torch.autograd.grad(sum(value.float().sum() for value in out0), (x0,) + params0)
        grad1 = torch.autograd.grad(sum(value.float().sum() for value in out1), (x1,) + params1)
        gradient_error = max(float((left.float() - right.float()).abs().max())
                             for left, right in zip(grad0, grad1))

        # Multi-chunk carried-state parity: the stateful eager module is the oracle;
        # the kernel must thread the ATLAS window (history) across chunks, not drop it.
        half = sample.shape[1] // 2
        if half:
            oracle_state = eager.memory.initial_state(sample.shape[0], device=sample.device)
            mat, mom = state.memory.clone(), state.momentum.clone()
            hk = torch.zeros(0, sample.shape[0], memory.d_memory,
                             device=sample.device, dtype=torch.float32)
            hv = hk.clone()
            for chunk in (sample.detach()[:, :half], sample.detach()[:, half:]):
                ref_out, oracle_state = eager.memory(chunk, oracle_state,
                                                     return_state=True, record_stats=False)
                cand_out, mat, mom, hk, hv = candidate(chunk, mat, mom, hk, hv)
                chunk_error = max(
                    float((ref_out.detach().float() - cand_out.detach().float()).abs().max()),
                    float((oracle_state.memory.detach().float() - mat.detach().float()).abs().max()),
                    float((oracle_state.momentum.detach().float() - mom.detach().float()).abs().max()))
                output_error = max(output_error, chunk_error)

        def median_ms(fn, inputs) -> float:
            for _ in range(2):
                fn(*inputs)
            if sample.is_cuda:
                torch.cuda.synchronize(sample.device)
            timings = []
            for _ in range(repeats):
                started = time.perf_counter()
                fn(*inputs)
                if sample.is_cuda:
                    torch.cuda.synchronize(sample.device)
                timings.append((time.perf_counter() - started) * 1000)
            return sorted(timings)[len(timings) // 2]

        timing_inputs = (sample.detach(), state.memory, state.momentum)
        eager_ms = median_ms(eager, timing_inputs)
        candidate_ms = median_ms(candidate, timing_inputs)
    except Exception as exc:
        return {"schema": "rwkv-lab.online-memory-kernel-qualification.v1",
                "backend": compile_backend, "available": False, "adopted": False,
                "error": repr(exc)}
    speedup = eager_ms / max(candidate_ms, 1e-12)
    parity = output_error <= tolerance and gradient_error <= tolerance
    performance = speedup >= minimum_speedup
    return {"schema": "rwkv-lab.online-memory-kernel-qualification.v1",
            "backend": compile_backend, "available": True,
            "output_max_abs": output_error, "gradient_max_abs": gradient_error,
            "tolerance": tolerance, "eager_ms": eager_ms,
            "candidate_ms": candidate_ms, "speedup": speedup,
            "minimum_speedup": minimum_speedup, "parity_passed": parity,
            "performance_passed": performance,
            "adopted": bool(parity and performance)}
