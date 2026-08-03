"""The input pipeline may be scheduled differently; it may not train on
different data.

Prefetching and worker-parallel loading are exactly the changes that can hand
back the right batches in the wrong order, or resume on the wrong cursor, while
reporting a perfectly good steps-per-second. Before this, `ordering_parity` and
`content_parity` were hardcoded true in the emitted evidence, so a loader that
reordered batches would have been qualified on a speed number alone.

These tests pin two things: that the gate reports parity it actually observed,
and — the part that makes it a gate rather than a nicer-looking constant — that
it FAILS on a reordering.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import subprocess
import sys

import pytest

REPOSITORY = pathlib.Path(__file__).resolve().parents[1]
RUNNER = REPOSITORY / "scripts/run_benchmark_fixture.py"
WORKLOAD_DIRECTORY = REPOSITORY / "scripts/benchmark_workloads"
AO3_CORPUS_WORKLOAD = WORKLOAD_DIRECTORY / "ao3_corpus_lm_step.py"

MODULE_SPEC = importlib.util.spec_from_file_location(
    "run_benchmark_fixture", RUNNER)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
benchmark_runner = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(benchmark_runner)

if str(WORKLOAD_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(WORKLOAD_DIRECTORY))
import batch_identity  # noqa: E402


def corpus_is_available() -> bool:
    """The AO3 corpus is a host resource; hosted CI does not have it."""
    spec = importlib.util.spec_from_file_location(
        "ao3_corpus_lm_step_probe", AO3_CORPUS_WORKLOAD)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    index, root, vocabulary = module.corpus_configuration()
    return index.is_file() and root.is_dir() and vocabulary.is_file()


requires_corpus = pytest.mark.skipif(
    not corpus_is_available(),
    reason="AO3 corpus or RWKV vocabulary is not present on this host",
)


def arm(step_digests: list[str]) -> dict:
    return {
        "step_batch_digests": step_digests,
        "batch_sequence_digest": batch_identity.chain_digests(step_digests),
    }


# --------------------------------------------------------------------------
# The gate must be able to fail. Everything else is decoration.
# --------------------------------------------------------------------------


def test_reordered_batches_fail_ordering_parity():
    """The whole point: same batches, wrong order, still rejected.

    A prefetching loader that completes out of order and forgets to restore
    submission order produces exactly this. Content parity still holds — the
    run saw every batch — which is why ordering needs its own answer.
    """
    baseline = arm(["sha256:a", "sha256:b", "sha256:c"])
    reordered = arm(["sha256:b", "sha256:a", "sha256:c"])

    verdict = benchmark_runner.compare_batch_identity(baseline, reordered)

    assert verdict["ordering_parity"] is False
    assert verdict["content_parity"] is True, (
        "a permutation trains on the same batches; only their order drifted"
    )
    assert verdict["batch_identity_observed"] is True


def test_changed_batch_content_fails_both_parities():
    baseline = arm(["sha256:a", "sha256:b"])
    altered = arm(["sha256:a", "sha256:different"])

    verdict = benchmark_runner.compare_batch_identity(baseline, altered)

    assert verdict["content_parity"] is False
    assert verdict["ordering_parity"] is False


def test_dropped_batch_fails_parity():
    """A loader that silently skips a batch is faster and wrong."""
    verdict = benchmark_runner.compare_batch_identity(
        arm(["sha256:a", "sha256:b", "sha256:c"]),
        arm(["sha256:a", "sha256:b"]),
    )

    assert verdict["content_parity"] is False
    assert verdict["ordering_parity"] is False


def test_absent_digests_are_not_parity():
    """Missing evidence must not read as passing evidence.

    This is the failure mode the hardcoded true had: nothing was measured, and
    the receipt said parity anyway.
    """
    verdict = benchmark_runner.compare_batch_identity({}, {})

    assert verdict["batch_identity_observed"] is False
    assert verdict["ordering_parity"] is False
    assert verdict["content_parity"] is False


def test_identical_arms_pass():
    digests = arm(["sha256:a", "sha256:b", "sha256:c"])

    verdict = benchmark_runner.compare_batch_identity(digests, dict(digests))

    assert verdict["ordering_parity"] is True
    assert verdict["content_parity"] is True


# --------------------------------------------------------------------------
# The digest itself must notice position, shape, and content.
# --------------------------------------------------------------------------


def test_digest_binds_the_step_index():
    """Content that repeats must not make two different steps look equal.

    This fixture's document set is small enough that batch content cycles with
    a short period, so a digest over content alone could not tell step 0 from
    step 2 and a reordering across that period would pass.
    """
    torch = pytest.importorskip("torch")
    tokens = torch.zeros((2, 4), dtype=torch.long)

    assert (batch_identity.batch_digest(0, tokens)
            != batch_identity.batch_digest(2, tokens))


def test_digest_notices_content_and_shape():
    torch = pytest.importorskip("torch")
    tokens = torch.zeros((2, 4), dtype=torch.long)
    changed = tokens.clone()
    changed[0, 0] = 1
    reshaped = torch.zeros((4, 2), dtype=torch.long)

    assert (batch_identity.batch_digest(0, tokens)
            != batch_identity.batch_digest(0, changed))
    assert (batch_identity.batch_digest(0, tokens)
            != batch_identity.batch_digest(0, reshaped)), (
        "a padding or mask change must not preserve the digest"
    )


def test_chain_notices_reordering():
    forward = batch_identity.chain_digests(["a", "b", "c"])
    swapped = batch_identity.chain_digests(["b", "a", "c"])

    assert forward != swapped


# --------------------------------------------------------------------------
# Real loader behaviour, against the real corpus.
# --------------------------------------------------------------------------


def run_workload(**options) -> dict:
    command = [
        sys.executable, str(AO3_CORPUS_WORKLOAD),
        "--phase", "timed", "--bucket", "seq256xbatch4",
        "--seed", "0", "--steps", "6",
    ]
    for name, value in options.items():
        command.extend([f"--{name.replace('_', '-')}", str(value)])
    completed = subprocess.run(
        command, capture_output=True, text=True, cwd=REPOSITORY, timeout=900,
        env={**os.environ, "PYTHONPATH": str(REPOSITORY / "src"),
             "CUDA_VISIBLE_DEVICES": ""},
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    return json.loads(completed.stdout)


@pytest.mark.slow
@requires_corpus
def test_parallel_prefetch_loads_identical_batches():
    """Scheduling the same work differently must not change what is trained on.

    Total tokenized work is asserted equal too: a loader that got faster by
    tokenizing less would be a different fixture, not an optimization.
    """
    serial = run_workload(input_workers=0, input_prefetch_depth=0)
    parallel = run_workload(input_workers=8, input_prefetch_depth=2)

    assert serial["input_pipeline_mode"] == "serial"
    assert parallel["input_pipeline_mode"] == "parallel_prefetch"
    assert (parallel["batch_sequence_digest"]
            == serial["batch_sequence_digest"])
    assert parallel["step_batch_digests"] == serial["step_batch_digests"]
    assert parallel["tokens_encoded"] == serial["tokens_encoded"], (
        "the parallel loader must do the same work, not less of it"
    )
    assert parallel["documents_read"] == serial["documents_read"]
    assert (parallel["result_fingerprint"]["final_loss"]
            == serial["result_fingerprint"]["final_loss"])


@pytest.mark.slow
@requires_corpus
def test_resumed_cursor_matches_an_uninterrupted_run():
    """The card's resume-cursor gate: step K must be step K either way."""
    full = run_workload()
    resumed = run_workload(start_step=3)

    assert resumed["executed_steps"] == 3
    assert resumed["step_batch_digests"] == full["step_batch_digests"][3:], (
        "a run resumed at step 3 must see the batches steps 3..5 saw"
    )


@pytest.mark.slow
@requires_corpus
def test_resume_holds_under_parallel_prefetch():
    """Resume and parallelism must compose; each alone proves little."""
    serial_resume = run_workload(start_step=3)
    parallel_resume = run_workload(
        start_step=3, input_workers=8, input_prefetch_depth=2)

    assert (parallel_resume["step_batch_digests"]
            == serial_resume["step_batch_digests"])


@pytest.mark.slow
@requires_corpus
def test_prefetch_reduces_measured_input_stall():
    """The card's actual claim, measured rather than asserted.

    Deliberately not a fixed speedup threshold: this is a shared workstation
    and a tight bound would be a flake. What must hold is the direction and
    that the win is not bought by doing less work.
    """
    serial = run_workload(input_workers=0, input_prefetch_depth=0)
    parallel = run_workload(input_workers=8, input_prefetch_depth=2)

    assert parallel["input_wait_seconds"] < serial["input_wait_seconds"], (
        "prefetch overlaps input with compute; the training thread should "
        "block for less time"
    )
    assert parallel["tokens_encoded"] == serial["tokens_encoded"]


@pytest.mark.slow
@requires_corpus
def test_serial_loader_remains_the_untouched_default():
    """Zero workers and zero depth must reproduce the pre-change loader."""
    default = run_workload()

    assert default["input_workers"] == 0
    assert default["input_prefetch_depth"] == 0
    assert default["input_pipeline_mode"] == "serial"
    assert default["start_step"] == 0


def test_start_step_outside_the_run_is_refused():
    """--steps is the total run length, not the number remaining."""
    completed = subprocess.run(
        [sys.executable, str(AO3_CORPUS_WORKLOAD),
         "--phase", "timed", "--bucket", "seq8xbatch1",
         "--seed", "0", "--steps", "4", "--start-step", "4"],
        capture_output=True, text=True, cwd=REPOSITORY, timeout=300,
        env={**os.environ, "PYTHONPATH": str(REPOSITORY / "src")},
        check=False,
    )

    assert completed.returncode != 0
    assert "start step must be inside the run" in completed.stderr


def test_negative_input_options_are_refused():
    for option in ("--input-workers", "--input-prefetch-depth"):
        completed = subprocess.run(
            [sys.executable, str(AO3_CORPUS_WORKLOAD),
             "--phase", "timed", "--bucket", "seq8xbatch1",
             "--seed", "0", "--steps", "2", option, "-1"],
            capture_output=True, text=True, cwd=REPOSITORY, timeout=300,
            env={**os.environ, "PYTHONPATH": str(REPOSITORY / "src")},
            check=False,
        )
        assert completed.returncode != 0, option


def test_every_workload_publishes_batch_identity():
    """No workload may leave the runner guessing at ordering parity.

    A workload that emitted no digest would send `batch_identity_observed`
    false and fail the gate, which is the safe direction — but the synthetic
    workloads can publish real digests cheaply, so they do.
    """
    for workload in sorted(WORKLOAD_DIRECTORY.glob("*_lm_step.py")):
        source = workload.read_text()
        assert '"batch_sequence_digest"' in source, workload.name
        assert '"step_batch_digests"' in source, workload.name
