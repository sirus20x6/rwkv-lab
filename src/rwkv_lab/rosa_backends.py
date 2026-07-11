"""Explicit backend registry and qualification for ROSA suffix matching.

The golden semantics are ROSA-Tuning (https://arxiv.org/abs/2602.02499).
CPU/CUDA/JAX/FPGA integration is motivated by the open ROSA-FPGA reference,
https://github.com/KakaruHayate/ROSA-FPGA, surfaced in the RWKV community at
https://discord.com/channels/992359628979568762/1426889957221466153/1523691227957170358
Hardware numbers are metadata, never trusted without a local parity receipt.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable
import numpy as np
from rwkv_lab.rosa_reference import rosa_reference


@dataclass(frozen=True)
class ROSABackend:
    name: str
    retrieve: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
    device: str = "cpu"
    energy_watts: float | None = None


_BACKENDS: dict[str, ROSABackend] = {"cpu-reference": ROSABackend("cpu-reference", rosa_reference)}


def register_rosa_backend(backend: ROSABackend, *, replace: bool = False) -> None:
    if backend.name in _BACKENDS and not replace:
        raise ValueError(f"ROSA backend already registered: {backend.name}")
    _BACKENDS[backend.name] = backend


def rosa_backends() -> tuple[str, ...]:
    return tuple(sorted(_BACKENDS))


def qualify_rosa_backend(name: str, aq: np.ndarray, ak: np.ndarray, *, repeats: int = 3) -> dict:
    if name not in _BACKENDS:
        return {"schema": "rwkv-lab.rosa-backend.v1", "backend": name,
                "available": False, "adopted": False, "error": "backend not registered"}
    backend = _BACKENDS[name]
    expected = rosa_reference(aq, ak)
    timings, actual = [], None
    for _ in range(max(1, repeats)):
        started = time.perf_counter(); actual = backend.retrieve(aq, ak)
        timings.append(time.perf_counter() - started)
    exact = (len(actual) == 2 and all(np.array_equal(a, b) for a, b in zip(expected, actual)))
    seconds = sorted(timings)[len(timings) // 2]
    return {"schema": "rwkv-lab.rosa-backend.v1", "backend": name,
            "device": backend.device, "available": True, "exact": exact,
            "seconds": seconds, "tokens_per_second": aq.shape[0] * aq.shape[1] / max(seconds, 1e-12),
            "energy_watts_reported": backend.energy_watts, "adopted": bool(exact)}
