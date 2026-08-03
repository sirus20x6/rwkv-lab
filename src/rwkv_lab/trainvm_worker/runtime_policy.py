"""Worker-owned CPU runtime policy, separate from host kernel authority.

The controller seals the complete resource document into every invocation.
This module owns only settings a non-root Python worker can apply and verify:
CPU affinity, tensor-library thread pools, and the preprocessing-worker hint.
CPU and I/O cgroup weights remain hostd-owned and are deliberately never
treated as effective merely because they appear in the invocation.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any

from ._canonical import canonical_dumps, sha256_digest

_POLICY_FIELDS = frozenset(
    {
        "cpuset",
        "cpus",
        "cpu_weight",
        "io_weight",
        "omp_threads",
        "preprocessing_workers",
        "nice",
    }
)
_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


class WorkerRuntimePolicyError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class WorkerRuntimePolicy:
    cpus: tuple[int, ...] | None
    cpu_weight: int | None
    io_weight: int | None
    omp_threads: int | None
    preprocessing_workers: int | None
    nice: int | None
    request_digest: str


@dataclass(frozen=True, slots=True)
class EffectiveWorkerRuntimePolicy:
    request_digest: str
    affinity: tuple[int, ...] | None
    omp_threads: int | None
    preprocessing_workers: int | None
    nice: int | None
    host_controls_pending: tuple[str, ...]


def _bounded_integer(
    value: Any, name: str, minimum: int, maximum: int
) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise WorkerRuntimePolicyError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise WorkerRuntimePolicyError(f"{name} is outside its declared bound")
    return value


def _cpuset(value: str) -> tuple[int, ...]:
    if not isinstance(value, str) or not value or len(value) > 4096:
        raise WorkerRuntimePolicyError("cpuset is not a bounded CPU list")
    result: list[int] = []
    previous = -2
    for item in value.split(","):
        if not item or item.count("-") > 1:
            raise WorkerRuntimePolicyError("cpuset is not canonical")
        endpoints = item.split("-")
        if any(not endpoint.isascii() or not endpoint.isdecimal() for endpoint in endpoints):
            raise WorkerRuntimePolicyError("cpuset contains a non-integer CPU")
        first = int(endpoints[0])
        last = int(endpoints[-1])
        if first > last or first > 1_048_575 or last > 1_048_575:
            raise WorkerRuntimePolicyError("cpuset range is invalid")
        if first <= previous + 1:
            raise WorkerRuntimePolicyError("cpuset ranges overlap or are adjacent")
        if len(result) + last - first + 1 > 1_048_576:
            raise WorkerRuntimePolicyError("cpuset expands beyond its bound")
        result.extend(range(first, last + 1))
        previous = last
    return tuple(result)


def _structured_cpus(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= 1024:
        raise WorkerRuntimePolicyError("cpus must contain 1 to 1024 entries")
    cpus = tuple(
        _bounded_integer(cpu, "cpus entry", 0, 1_048_575) for cpu in value
    )
    if any(cpu is None for cpu in cpus):  # pragma: no cover - defensive typing
        raise WorkerRuntimePolicyError("cpus contains a null entry")
    concrete = tuple(int(cpu) for cpu in cpus)
    if concrete != tuple(sorted(set(concrete))):
        raise WorkerRuntimePolicyError("cpus must be sorted and unique")
    return concrete


def load_worker_runtime_policy(
    resources: Mapping[str, Any],
) -> WorkerRuntimePolicy | None:
    raw = resources.get("cpu_io_policy")
    if raw is None:
        return None
    if not isinstance(raw, Mapping) or not raw or set(raw) - _POLICY_FIELDS:
        raise WorkerRuntimePolicyError("CPU/I/O policy fields are inexact")
    if "cpuset" in raw and "cpus" in raw:
        raise WorkerRuntimePolicyError("cpuset and cpus are mutually exclusive")
    cpus = None
    if "cpuset" in raw:
        cpus = _cpuset(raw["cpuset"])
    elif "cpus" in raw:
        cpus = _structured_cpus(raw["cpus"])
    request = dict(raw)
    return WorkerRuntimePolicy(
        cpus=cpus,
        cpu_weight=_bounded_integer(raw.get("cpu_weight"), "cpu_weight", 1, 10_000),
        io_weight=_bounded_integer(raw.get("io_weight"), "io_weight", 1, 10_000),
        omp_threads=_bounded_integer(
            raw.get("omp_threads"), "omp_threads", 1, 65_536
        ),
        preprocessing_workers=_bounded_integer(
            raw.get("preprocessing_workers"),
            "preprocessing_workers",
            0,
            65_536,
        ),
        nice=_bounded_integer(raw.get("nice"), "nice", -20, 19),
        request_digest=sha256_digest(canonical_dumps(request)),
    )


def _torch_thread_setter(threads: int) -> None:
    import torch

    torch.set_num_threads(threads)


def apply_worker_runtime_policy(
    resources: Mapping[str, Any],
    *,
    environment: MutableMapping[str, str] | None = None,
    get_affinity: Callable[[int], set[int]] = os.sched_getaffinity,
    set_affinity: Callable[[int, set[int]], None] = os.sched_setaffinity,
    get_priority: Callable[[int, int], int] = os.getpriority,
    set_priority: Callable[[int, int, int], None] = os.setpriority,
    set_torch_threads: Callable[[int], None] = _torch_thread_setter,
) -> EffectiveWorkerRuntimePolicy | None:
    """Apply and re-read the non-root portion of one sealed resource policy."""

    policy = load_worker_runtime_policy(resources)
    if policy is None:
        return None
    environment = os.environ if environment is None else environment
    affinity = None
    if policy.cpus is not None:
        requested = set(policy.cpus)
        available = get_affinity(0)
        if not requested.issubset(available):
            raise WorkerRuntimePolicyError(
                "requested CPU affinity is outside the hostd-assigned set"
            )
        set_affinity(0, requested)
        affinity = tuple(sorted(get_affinity(0)))
        if affinity != policy.cpus:
            raise WorkerRuntimePolicyError("effective CPU affinity is inexact")

    if policy.omp_threads is not None:
        value = str(policy.omp_threads)
        for name in _THREAD_ENVIRONMENT:
            environment[name] = value
        set_torch_threads(policy.omp_threads)
    if policy.preprocessing_workers is not None:
        environment["TRAINVM_PREPROCESSING_WORKERS"] = str(
            policy.preprocessing_workers
        )

    effective_nice = None
    if policy.nice is not None:
        observed = get_priority(os.PRIO_PROCESS, 0)
        if observed != policy.nice:
            set_priority(os.PRIO_PROCESS, 0, policy.nice)
            observed = get_priority(os.PRIO_PROCESS, 0)
        if observed != policy.nice:
            raise WorkerRuntimePolicyError("effective nice level is inexact")
        effective_nice = observed

    host_controls = tuple(
        name
        for name, value in (
            ("cpu_weight", policy.cpu_weight),
            ("io_weight", policy.io_weight),
        )
        if value is not None
    )
    return EffectiveWorkerRuntimePolicy(
        request_digest=policy.request_digest,
        affinity=affinity,
        omp_threads=policy.omp_threads,
        preprocessing_workers=policy.preprocessing_workers,
        nice=effective_nice,
        host_controls_pending=host_controls,
    )


__all__ = [
    "EffectiveWorkerRuntimePolicy",
    "WorkerRuntimePolicy",
    "WorkerRuntimePolicyError",
    "apply_worker_runtime_policy",
    "load_worker_runtime_policy",
]
