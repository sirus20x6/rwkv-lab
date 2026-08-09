"""Ordering, not presence, and a baseline that is not assumed to be zero.

Two properties are easy to conflate:

* the boundary call *exists* — cheap, and satisfied by a call in the wrong
  place or by a helper that some other update path never goes through; and
* every optimizer mutation is *preceded* by it — the property the
  attempt-baseline eval gate is actually about.

`perform_rwkv_optimizer_step` writes the two calls in the right order, which
makes the ordering a convention of that function. The sentinel makes it
enforceable for the whole process, and the loop tests below read the observed
sequence of crossings and mutations rather than trusting the source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch.optim.optimizer import register_optimizer_step_pre_hook

from rwkv_lab.rwkv_pretrain import perform_rwkv_optimizer_step
from rwkv_lab.trainvm_worker import MutationSentinelError, OptimizerMutationSentinel


def _parameter() -> torch.nn.Parameter:
    parameter = torch.nn.Parameter(torch.zeros(4))
    parameter.grad = torch.ones(4)
    return parameter


# --------------------------------------------------------------------------
# Sentinel semantics
# --------------------------------------------------------------------------


def test_mutation_without_a_preceding_boundary_fails_closed() -> None:
    sentinel = OptimizerMutationSentinel()
    optimizer = torch.optim.AdamW([_parameter()], lr=0.1)
    with sentinel.installed(), pytest.raises(MutationSentinelError, match="without first"):
        optimizer.step()
    assert sentinel.observed_mutations == 0


def test_a_boundary_crossed_after_the_mutation_is_rejected() -> None:
    """The exact regression the invariant exists to prevent.

    Presence is identical in both orders; only the sentinel distinguishes
    them, and it must reject the mutation that ran first.
    """

    crossings: list[int] = []
    sentinel = OptimizerMutationSentinel()
    optimizer = torch.optim.AdamW([_parameter()], lr=0.1)
    with sentinel.installed():
        with pytest.raises(MutationSentinelError):
            optimizer.step()                       # the wrong order: mutate ...
        sentinel.cross(1, crossings.append)        # ... then announce
    assert crossings == [1], "the boundary itself ran; only its position was wrong"


def test_one_boundary_authorizes_exactly_one_mutation() -> None:
    sentinel = OptimizerMutationSentinel()
    optimizer = torch.optim.AdamW([_parameter()], lr=0.1)
    with sentinel.installed():
        sentinel.cross(1)
        optimizer.step()
        with pytest.raises(MutationSentinelError, match="without first"):
            optimizer.step()
    assert sentinel.journal == (("boundary", 1), ("mutation", 1))


def test_an_alternate_or_later_built_optimizer_cannot_bypass_the_boundary() -> None:
    """A fused or second update path is a bypass, not an exemption.

    This is what wrapping one optimizer object cannot catch: the alternate is
    constructed after installation and never passes through
    `perform_rwkv_optimizer_step`.
    """

    sentinel = OptimizerMutationSentinel()
    guarded = torch.optim.AdamW([_parameter()], lr=0.1)
    with sentinel.installed():
        class FusedAdamW(torch.optim.AdamW):
            pass

        alternate = FusedAdamW([_parameter()], lr=0.1)
        sentinel.cross(1)
        guarded.step()
        with pytest.raises(MutationSentinelError, match="without first"):
            alternate.step()


def test_an_update_that_skips_the_optimizer_entirely_is_caught_next_step() -> None:
    """A hand-rolled update consumes no token, so the next crossing trips."""

    sentinel = OptimizerMutationSentinel()
    with sentinel.installed():
        sentinel.cross(1)
        with pytest.raises(MutationSentinelError, match="twice without a mutation"):
            sentinel.cross(2)


def test_the_boundary_names_the_step_about_to_run() -> None:
    sentinel = OptimizerMutationSentinel()
    with sentinel.installed():
        with pytest.raises(MutationSentinelError, match="positive step"):
            sentinel.cross(0)


def test_a_refusing_boundary_leaves_the_sentinel_disarmed() -> None:
    """A rejected gate must not leave the next mutation authorized."""

    def refuse(_next_step: int) -> None:
        raise RuntimeError("gate rejected")

    sentinel = OptimizerMutationSentinel()
    optimizer = torch.optim.AdamW([_parameter()], lr=0.1)
    with sentinel.installed():
        with pytest.raises(RuntimeError, match="gate rejected"):
            sentinel.cross(1, refuse)
        with pytest.raises(MutationSentinelError, match="without first"):
            optimizer.step()


def test_the_step_helper_routes_its_crossing_through_the_sentinel() -> None:
    events: list[tuple[str, int]] = []

    class Controls:
        def pre_optimizer_step(self, next_step: int, _applier: Any) -> None:
            events.append(("boundary", next_step))

    sentinel = OptimizerMutationSentinel()
    optimizer = torch.optim.AdamW([_parameter()], lr=0.1)
    handle = register_optimizer_step_pre_hook(
        lambda _optimizer, _args, _kwargs: events.append(("mutation", -1))
    )
    try:
        with sentinel.installed():
            perform_rwkv_optimizer_step(
                optimizer,
                Controls(),
                next_step=1,
                control_applier=lambda *_: None,
                sentinel=sentinel,
            )
    finally:
        handle.remove()
    assert events == [("boundary", 1), ("mutation", -1)]
    assert sentinel.journal == (("boundary", 1), ("mutation", 1))


# --------------------------------------------------------------------------
# The real scratch-RWKV loop
# --------------------------------------------------------------------------


@dataclass
class _LoopControls:
    """Only the surface `rwkv_pretrain.main` actually touches."""

    step_zero_eval_gate_required: bool = False
    step_zero_eval_gate_satisfied: bool = False
    attempt_baseline_optimizer_step: int = 0
    events: list[tuple[str, int]] = field(default_factory=list)

    def pre_optimizer_step(self, next_step: int, _applier: Any) -> tuple[()]:
        if (
            self.step_zero_eval_gate_required
            and not self.step_zero_eval_gate_satisfied
        ):
            raise RuntimeError(
                "optimizer mutation is blocked until durable attempt-baseline evidence"
            )
        self.events.append(("boundary", next_step))
        return ()

    def microbatch(self, _step: int, _applier: Any) -> tuple[()]:
        return ()

    def evaluation(self, _step: int, _applier: Any) -> tuple[()]:
        return ()

    def checkpoint(self, _step: int, _applier: Any) -> tuple[()]:
        return ()

    def checkpoint_state(self) -> dict[str, Any]:
        return {"effective_control_revision": 0, "effective_controls": {}}

    def verify_checkpoint_state(self, _state: Any) -> None:
        return None


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    path = tmp_path / "tokens.bin"
    rng = np.random.default_rng(11)
    rng.integers(1, 500, size=20_000, dtype=np.uint16).tofile(path)
    return path


def _arguments(corpus: Path, out: Path, *, steps: int, resume: str | None = None) -> list[str]:
    argv = [
        "--data", str(corpus),
        "--out", str(out),
        "--save", str(out / "state.pt"),
        "--steps", str(steps),
        "--d-model", "64",
        "--n-layers", "1",
        "--head-size", "16",
        "--seq-len", "16",
        "--batch", "2",
        "--val-windows", "2",
        "--eval-every", "4",
        "--log-every", "4",
        "--warmup", "1",
        "--seed", "0",
        "--gpu-data", "off",
        "--no-cpu-prefetch",
    ]
    if resume is not None:
        argv.extend(("--resume", resume))
    return argv


@pytest.fixture
def cpu_trainer(monkeypatch: pytest.MonkeyPatch):
    """Run the real loop on CPU: no accelerator, no Triton kernel."""

    monkeypatch.setenv("RWKV8_FORCE_PYREF", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    from rwkv_lab.rwkv_pretrain import main

    return main


@pytest.fixture
def observed_mutations():
    """Every optimizer mutation the process performs, in order."""

    events: list[tuple[str, int]] = []
    handle = register_optimizer_step_pre_hook(
        lambda optimizer, args, kwargs: events.append(("mutation", -1))
    )
    try:
        yield events
    finally:
        handle.remove()


def test_every_mutation_in_the_real_loop_is_preceded_by_its_boundary(
    corpus: Path, tmp_path: Path, cpu_trainer, observed_mutations
) -> None:
    """The load-bearing ordering assertion.

    `controls.events` and `observed_mutations` append to the SAME list, so this
    is about sequence, not counts. It fails if the boundary is missing, moved
    after `optimizer.step()`, or bypassed by another update path.
    """

    controls = _LoopControls(events=observed_mutations)
    result = cpu_trainer(
        _arguments(corpus, tmp_path / "run", steps=3),
        worker_controls=controls,
    )

    assert result["step"] == 3
    assert observed_mutations == [
        ("boundary", 1),
        ("mutation", -1),
        ("boundary", 2),
        ("mutation", -1),
        ("boundary", 3),
        ("mutation", -1),
    ]


def test_a_run_without_the_declared_gate_still_trains(
    corpus: Path, tmp_path: Path, cpu_trainer
) -> None:
    """An invocation that declares no eval-examples output owes no evidence.

    The trainer is also a standalone research entrypoint; publishing baseline
    evidence unconditionally made every such run fail before its first step.
    """

    result = cpu_trainer(_arguments(corpus, tmp_path / "run", steps=2))
    assert result["step"] == 2


def test_a_refused_boundary_stops_the_run_before_any_mutation(
    corpus: Path, tmp_path: Path, cpu_trainer, observed_mutations
) -> None:
    controls = _LoopControls(
        events=observed_mutations,
        step_zero_eval_gate_required=True,
        step_zero_eval_gate_satisfied=False,
        attempt_baseline_optimizer_step=0,
    )
    # The gate is declared but this stub publishes nothing, so the boundary
    # refuses. Nothing may reach the parameters.
    with pytest.raises((RuntimeError, ValueError)):
        cpu_trainer(
            _arguments(corpus, tmp_path / "run", steps=3),
            worker_controls=controls,
        )
    assert observed_mutations == [], "the parameters were mutated past a refused gate"


def test_a_resumed_attempt_recovers_the_gate_and_still_guards_every_step(
    corpus: Path, tmp_path: Path, cpu_trainer, observed_mutations
) -> None:
    """Resume must deadlock on neither a literal zero nor an unguarded update.

    A replacement attempt resumes at its checkpoint's step. Keying baseline
    evidence to `step == 0` leaves that attempt owing evidence at a step it
    will never reach again, so it never satisfies the gate and never takes a
    step. Here the controller has replayed durable evidence at the real
    baseline, and training proceeds — still guarded at every step.
    """

    first = tmp_path / "first"
    cpu_trainer(_arguments(corpus, first, steps=3), worker_controls=_LoopControls())
    observed_mutations.clear()

    resumed = _LoopControls(
        events=observed_mutations,
        step_zero_eval_gate_required=True,
        step_zero_eval_gate_satisfied=True,
        attempt_baseline_optimizer_step=3,
    )
    result = cpu_trainer(
        _arguments(corpus, tmp_path / "second", steps=5, resume=str(first / "state.pt")),
        worker_controls=resumed,
    )
    assert result["step"] == 5
    assert observed_mutations == [
        ("boundary", 4),
        ("mutation", -1),
        ("boundary", 5),
        ("mutation", -1),
    ]


def test_a_resumed_attempt_whose_baseline_disagrees_refuses_to_train(
    corpus: Path, tmp_path: Path, cpu_trainer, observed_mutations
) -> None:
    """Silence here would become a deadlock at the first boundary instead."""

    first = tmp_path / "first"
    cpu_trainer(_arguments(corpus, first, steps=3), worker_controls=_LoopControls())
    observed_mutations.clear()

    mismatched = _LoopControls(
        events=observed_mutations,
        step_zero_eval_gate_required=True,
        step_zero_eval_gate_satisfied=False,
        attempt_baseline_optimizer_step=0,
    )
    with pytest.raises(ValueError, match="does not gate"):
        cpu_trainer(
            _arguments(
                corpus, tmp_path / "second", steps=5, resume=str(first / "state.pt")
            ),
            worker_controls=mismatched,
        )
    assert observed_mutations == []


@dataclass
class _PublishedArtifact:
    artifact_id: str = "checkpoint-baseline"
    manifest_sha256: str = "sha256:" + "c" * 64


@dataclass
class _PublishingControls(_LoopControls):
    published: list[Any] = field(default_factory=list)

    def publish_policy_checkpoint(self, request: Any) -> _PublishedArtifact:
        self.events.append(("checkpoint", request.optimizer_step))
        self.published.append(request)
        return _PublishedArtifact()

    def publish_evaluation_examples(self, request: Any) -> Any:
        self.events.append(("eval_examples", request.optimizer_step))
        self.published.append(request)
        self.step_zero_eval_gate_satisfied = True
        return _PublishedArtifact(artifact_id="eval-examples-baseline")


def _eval_policy():
    from rwkv_lab.trainvm_adapters.rwkv_scratch import RWKVTextEvalPolicy

    return RWKVTextEvalPolicy(
        heldout_tokens={"heldout-1": (3, 4, 5)},
        identity_field="id",
        identities_digest="sha256:" + "1" * 64,
        selector_digest="sha256:" + "2" * 64,
        evaluator_component_digest="sha256:" + "3" * 64,
        metric_names=("perplexity", "validation_loss"),
        generation_policy_digest="sha256:" + "4" * 64,
    )


def test_a_replacement_attempt_owes_its_evidence_at_the_resumed_baseline(
    corpus: Path, tmp_path: Path, cpu_trainer, observed_mutations
) -> None:
    """The deadlock a literal step-zero key produces, asserted directly.

    A fresh replacement attempt resumes at step 3 and has no durable evidence
    of its own — the controller's gate evidence is attempt-local. If the
    publication is keyed to `step == 0` it never fires, the gate is never
    satisfied, and the boundary refuses every step forever. The evidence must
    be published at the baseline the controller actually named.
    """

    first = tmp_path / "first"
    cpu_trainer(_arguments(corpus, first, steps=3), worker_controls=_LoopControls())
    observed_mutations.clear()

    replacement = _PublishingControls(
        events=observed_mutations,
        step_zero_eval_gate_required=True,
        step_zero_eval_gate_satisfied=False,
        attempt_baseline_optimizer_step=3,
    )
    second = tmp_path / "second"
    result = cpu_trainer(
        _arguments(corpus, second, steps=5, resume=str(first / "state.pt")),
        worker_controls=replacement,
        worker_eval_examples=_eval_policy(),
    )

    assert result["step"] == 5
    assert observed_mutations == [
        ("checkpoint", 3),
        ("eval_examples", 3),
        ("boundary", 4),
        ("mutation", -1),
        ("boundary", 5),
        ("mutation", -1),
    ]
    evidence = replacement.published[-1]
    assert evidence.optimizer_step == 3
    assert evidence.examples, "the baseline artifact must not be empty"
    assert evidence.checkpoint_artifact_id in evidence.parent_artifact_ids
    assert (second / "checkpoint-baseline-3" / "state.pt").is_file()
