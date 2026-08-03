from __future__ import annotations

import os

import pytest

from rwkv_lab.trainvm_worker import (
    WorkerRuntimePolicyError,
    apply_worker_runtime_policy,
    load_worker_runtime_policy,
)


def test_policy_parser_keeps_host_and_worker_controls_separate() -> None:
    policy = load_worker_runtime_policy(
        {
            "cpu_io_policy": {
                "cpuset": "2-4,7",
                "cpu_weight": 300,
                "io_weight": 80,
                "omp_threads": 3,
                "preprocessing_workers": 1,
                "nice": 4,
            }
        }
    )
    assert policy is not None
    assert policy.cpus == (2, 3, 4, 7)
    assert policy.cpu_weight == 300
    assert policy.io_weight == 80
    assert policy.request_digest.startswith("sha256:")


@pytest.mark.parametrize(
    "raw",
    [
        {},
        {"cpuset": "0-2,3"},
        {"cpuset": "2-1"},
        {"cpuset": "0", "cpus": [0]},
        {"cpus": [1, 0]},
        {"cpus": [0, 0]},
        {"omp_threads": True},
        {"preprocessing_workers": -1},
        {"cpu_weight": 0},
        {"surprise": 1},
    ],
)
def test_policy_parser_rejects_noncanonical_or_unbounded_values(raw) -> None:
    with pytest.raises(WorkerRuntimePolicyError):
        load_worker_runtime_policy({"cpu_io_policy": raw})


def test_worker_policy_applies_and_reattests_only_worker_owned_controls() -> None:
    environment: dict[str, str] = {}
    affinity = {0, 1, 2, 3}
    priority = 0
    torch_threads: list[int] = []

    def get_affinity(_pid: int) -> set[int]:
        return set(affinity)

    def set_affinity(_pid: int, cpus: set[int]) -> None:
        affinity.clear()
        affinity.update(cpus)

    def get_priority(which: int, who: int) -> int:
        assert (which, who) == (os.PRIO_PROCESS, 0)
        return priority

    def set_priority(which: int, who: int, value: int) -> None:
        nonlocal priority
        assert (which, who) == (os.PRIO_PROCESS, 0)
        priority = value

    effective = apply_worker_runtime_policy(
        {
            "cpu_io_policy": {
                "cpus": [1, 2],
                "cpu_weight": 200,
                "io_weight": 90,
                "omp_threads": 2,
                "preprocessing_workers": 1,
                "nice": 3,
            }
        },
        environment=environment,
        get_affinity=get_affinity,
        set_affinity=set_affinity,
        get_priority=get_priority,
        set_priority=set_priority,
        set_torch_threads=torch_threads.append,
    )
    assert effective is not None
    assert effective.affinity == (1, 2)
    assert effective.omp_threads == 2
    assert effective.preprocessing_workers == 1
    assert effective.nice == 3
    assert effective.host_controls_pending == ("cpu_weight", "io_weight")
    assert torch_threads == [2]
    assert environment == {
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "NUMEXPR_NUM_THREADS": "2",
        "TRAINVM_PREPROCESSING_WORKERS": "1",
    }


def test_worker_policy_fails_if_host_affinity_does_not_contain_request() -> None:
    with pytest.raises(WorkerRuntimePolicyError, match="hostd-assigned"):
        apply_worker_runtime_policy(
            {"cpu_io_policy": {"cpus": [4]}},
            get_affinity=lambda _pid: {0, 1},
            set_affinity=lambda _pid, _cpus: pytest.fail("must not set"),
        )
