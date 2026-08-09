"""Ordering, not presence, for the RLVR trainer's *two* optimizer boundaries.

``tests/test_step_zero_interception_enumeration.py`` is a static reader: it can
only see that ``rlvr_train.py`` names ``pre_optimizer_step`` somewhere in the
file. For every other route in that countdown that is a weak-but-tolerable
claim. For this one it is actively misleading, because RLVR mutates on **two**
independent paths -- the DeepSeek-R1-style supervised warm start
(``supervised_warm_start``) and the grouped policy update
(``optimize_rollouts``) -- and a single boundary anywhere in the file satisfies
the enumeration while half the route still mutates unguarded.

So the property these tests exist to hold is sharper than "a boundary exists":
**each mutation path must cross the boundary on its own, and the two paths must
be distinguishable.** Removing the crossing from the warm start and removing it
from the rollout update must redden *different* sets of tests here. A suite
where both mutations redden the same set cannot tell the two sites apart, which
is exactly the failure this route is exposed to; it would pass while shipping
the bypass.

The instrument is the one ``tests/test_rwkv_mutation_sentinel.py`` established
and ``tests/test_hf_multimodal_sft_mutation_boundary.py`` reused: a
process-global ``register_optimizer_step_pre_hook`` appends to the *same* list
the fake controller appends its crossings to, so what is asserted is an
interleaving rather than two independent counts.

What is deliberately NOT asserted here: that attempt-baseline eval evidence
precedes the first mutation. RLVR cannot publish that evidence at all today --
its authority profile declares no ``eval_examples`` output, so
``EvalExamplesPublisher`` refuses the publication -- and a test written against
evidence the route cannot emit would assert the harness rather than the
trainer. What ``run()`` does instead is refuse to mutate when the controller
requires evidence it has none of, and that refusal is asserted below.
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.optim.optimizer import register_optimizer_step_pre_hook

from rwkv_lab import rlvr_train
from rwkv_lab.rlvr_train import RLVRTask, Rollout
from rwkv_lab.trainvm_worker.mutation_sentinel import MutationSentinelError


class ToyLM(nn.Module):
    """The smallest model the real scoring and update paths accept."""

    def __init__(self, vocab: int = 16, width: int = 8) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, width)
        self.head = nn.Linear(width, vocab, bias=False)

    def forward(self, ids):
        return self.head(self.emb(ids))


class ToyVocab:
    def encode(self, text: str) -> list[int]:
        return [2 + (ord(character) % 8) for character in text[:6]] or [2]

    def decode(self, ids) -> str:
        return "".join(chr(97 + (int(i) % 8)) for i in ids)


class FakeControls:
    """A controller that records crossings into the shared event list.

    ``pre_optimizer_step`` is the only boundary; ``optimizer_step`` is the same
    call under its pre-rename name, and both refuse a step that does not follow
    the immutable attempt baseline, exactly as ``WorkerControlRuntime`` does.
    """

    def __init__(
        self,
        events: list[tuple[str, int]],
        *,
        attempt_baseline: int = 0,
        gate_required: bool = False,
        gate_satisfied: bool = False,
    ) -> None:
        self.events = events
        self.attempt_baseline_optimizer_step = attempt_baseline
        self.step_zero_eval_gate_required = gate_required
        self.step_zero_eval_gate_satisfied = gate_satisfied
        self.effective_values: dict[str, object] = {}

    def pre_optimizer_step(self, next_step: int, applier) -> tuple:
        if next_step <= self.attempt_baseline_optimizer_step:
            raise ValueError(
                "optimizer mutation step must follow the immutable attempt baseline"
            )
        if self.step_zero_eval_gate_required and not self.step_zero_eval_gate_satisfied:
            raise ValueError(
                "optimizer mutation is blocked until durable attempt-baseline evidence"
            )
        applier(next_step, ())
        self.events.append(("boundary", next_step))
        return ()


@pytest.fixture
def events():
    """One list holding both crossings and mutations, in the order they happened.

    Two independent counters would agree in a run where every crossing landed
    after its mutation. A single list makes the interleaving itself the
    assertion, which is the property these tests are about.
    """

    recorded: list[tuple[str, int]] = []
    handle = register_optimizer_step_pre_hook(
        lambda optimizer, args, kwargs: recorded.append(("mutation", 0))
    )
    try:
        yield recorded
    finally:
        handle.remove()


def _task(index: int) -> RLVRTask:
    return RLVRTask(
        f"t{index}",
        f"{index}+1",
        {"kind": "numeric", "expected": index + 1},
        split="train" if index % 3 else "eval",
        metadata={"sft_answer": str(index + 1)},
    )


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """Drive the real ``rlvr_train.run`` over doubles for model, vocab and rollouts.

    Everything between the two mutation sites is real: the warm-start forward
    and backward, ``optimize_rollouts``'s clipped policy loss, the schedules,
    the checkpointing. Only generation and verification are replaced, because
    both are slow, neither mutates parameters, and a deterministic reward keeps
    the update path informative on every step.
    """

    from rwkv_lab import generate as generate_module

    model = ToyLM()
    source = tmp_path / "source.pt"
    source.write_bytes(b"toy-rlvr-checkpoint")
    monkeypatch.setattr(
        generate_module, "build_from_ckpt", lambda *a, **k: (model, {"arch": {}})
    )
    monkeypatch.setattr(generate_module, "WorldVocab", lambda *a, **k: ToyVocab())

    generated: list[int] = []

    def fake_generate(_model, _vocab, tasks, *, group_size, return_stats=False, **kwargs):
        generated.append(len(tasks))
        rollouts = [
            Rollout(f"{task.id}:{member}", task, [2, 3], [4 + member, 1], str(member))
            for task in tasks
            for member in range(group_size)
        ]
        stats = {"tokens": 2 * len(rollouts), "truncated": 0}
        return (rollouts, stats) if return_stats else rollouts

    def fake_verify(rollouts, **kwargs):
        rewards = torch.tensor(
            [float(index % 2) for index in range(len(rollouts))]
        )
        return rewards, [{"source": "fake"} for _ in rollouts]

    monkeypatch.setattr(rlvr_train, "generate_rollouts", fake_generate)
    monkeypatch.setattr(rlvr_train, "verify_rollouts", fake_verify)

    def invoke(*, steps: int = 2, sft_steps: int = 2, epochs: int = 1, **overrides):
        arguments = {
            "ckpt": str(source), "resume": "", "out": str(tmp_path / "run"),
            "tasks": "", "heldout_tasks": "", "algorithm": "gspo", "steps": steps,
            "prompts_per_step": 2, "group_size": 2, "epochs": epochs, "max_new": 4,
            "rollout_engine": "auto", "rollout_devices": "", "temperature": 1.0,
            "eval_temperature": 0.0, "top_p": 1.0, "top_k": 0, "stop_token": 1,
            "lr": 1e-3, "weight_decay": 0.0, "optimizer": "adamw", "warmup": 1,
            "grad_clip": 1.0, "clip_low": 0.2, "clip_high": 0.2, "kl_coef": 0.0,
            "reference": "none", "reference_ckpt": "", "train_tasks": 6,
            "eval_tasks": 3, "difficulty": 1, "curriculum_stages": "",
            "sft_steps": sft_steps, "sft_batch_size": 2, "sft_lr": 1e-3,
            "preflight_prompts": 0, "min_preflight_reward": 0.0,
            "max_preflight_reward": 1.0, "min_preflight_active_groups": 0,
            "eval_every": 0, "eval_prompts": 2, "eval_group_size": 1,
            "min_heldout_delta": 0.0, "confidence": 0.95, "bootstrap_samples": 0,
            "require_confidence": False, "max_family_regression": 1.0,
            "max_rollout_tokens": 0, "max_train_seconds": 0, "save_every": 0,
            "verifier_command": (), "verifier_timeout": 5.0, "log_samples": 0,
            "seed": 5, "device": "cpu", "use_ema": False,
            "vocab": str(source),
        }
        controls = overrides.pop("worker_controls", None)
        arguments.update(overrides)
        return rlvr_train.run(Namespace(**arguments), worker_controls=controls)

    invoke.generated = generated
    return invoke


# --- The two sites, separately -------------------------------------------


def test_the_warm_start_crosses_the_boundary_before_every_update(harness, events):
    """Site A: every supervised warm-start update is preceded by a crossing.

    Scoped to the warm start by running zero RLVR steps, so a crossing the
    rollout path performs cannot pay for one the warm start owes.
    """

    harness(steps=0, sft_steps=3, worker_controls=FakeControls(events))
    assert events == [("boundary", 1), ("mutation", 0)] * 3


def test_the_rollout_update_crosses_the_boundary_before_every_update(harness, events):
    """Site B: every grouped policy update is preceded by its own crossing.

    Scoped to the rollout path by running no warm start at all, so this cannot
    pass on the strength of site A's crossings.
    """

    harness(steps=2, sft_steps=0, worker_controls=FakeControls(events))
    assert events == [
        ("boundary", 1), ("mutation", 0), ("boundary", 2), ("mutation", 0)
    ]


def test_every_rollout_epoch_gets_its_own_crossing(harness, events):
    """One crossing arms one mutation, so N epochs owe N crossings.

    ``optimize_rollouts`` mutates once per epoch. A crossing hoisted out of its
    epoch loop -- the obvious "one boundary per training step" simplification --
    would arm one token for several mutations and fails here.
    """

    harness(steps=1, sft_steps=0, epochs=3, worker_controls=FakeControls(events))
    assert events == [("boundary", 1), ("mutation", 0)] * 3


def test_both_paths_are_guarded_in_one_run(harness, events):
    """The whole-route claim: warm start and rollouts, one sentinel, no gaps.

    This is the test the card's "both sites or neither" decision rests on. Its
    interleaving contains mutations from both paths, so it reddens whichever
    site loses its crossing -- which is precisely why it cannot be the only
    test here, and why the two scoped tests above exist.
    """

    harness(steps=2, sft_steps=2, worker_controls=FakeControls(events))
    assert [kind for kind, _ in events] == ["boundary", "mutation"] * 4
    assert [step for kind, step in events if kind == "boundary"] == [1, 1, 1, 2]


# --- Fail-closed properties ----------------------------------------------


def test_an_unguarded_mutation_inside_the_loop_fails_closed(harness, events):
    """A second optimizer that never routes through either site cannot mutate.

    This is the bypass the process-global pre-hook exists for: not a missing
    call at a known site, but an update path nobody wrote a site for. The
    stand-in is a foreign optimizer stepped from inside the run.
    """

    intruder = torch.optim.SGD(ToyLM().parameters(), lr=0.1)
    original = rlvr_train.optimize_rollouts

    def bypass(*args, **kwargs):
        intruder.step()
        return original(*args, **kwargs)

    rlvr_train.optimize_rollouts = bypass
    try:
        with pytest.raises(MutationSentinelError, match="without first crossing"):
            harness(steps=1, sft_steps=0, worker_controls=FakeControls(events))
    finally:
        rlvr_train.optimize_rollouts = original


def test_a_refused_crossing_leaves_the_mutation_unperformed(harness, events):
    """A boundary that raises must not be followed by the mutation it guarded.

    The controller refuses because the attempt baseline is ahead of the step
    the crossing names. Nothing may mutate afterwards -- the sentinel is left
    disarmed by construction, so even a mutation that ignored the exception
    would fail.
    """

    controls = FakeControls(events, attempt_baseline=99)
    with pytest.raises(ValueError, match="does not gate"):
        harness(steps=1, sft_steps=1, worker_controls=controls)
    assert events == []


def test_a_required_gate_with_no_durable_evidence_refuses_before_it_generates(
    harness, events
):
    """Fail closed when the controller requires evidence this attempt lacks.

    RLVR cannot publish attempt-baseline eval examples today, so the only
    correct behaviour under a required gate is refusal.

    What is asserted is *where* the refusal happens, not merely that one does.
    ``pre_optimizer_step`` would refuse this attempt anyway, at the first
    crossing -- which is after the baseline evaluation and a full step of
    rollouts, i.e. minutes of generation spent on a step that can never land.
    So the assertion is that nothing was generated at all: the trainer read the
    gate itself and stopped. A test that only asserted the exception would pass
    with the trainer's check deleted, because the fake controller raises too.
    """

    controls = FakeControls(events, gate_required=True, gate_satisfied=False)
    with pytest.raises(ValueError, match="durable"):
        harness(steps=1, sft_steps=1, worker_controls=controls)
    assert events == []
    assert harness.generated == []


def test_recovered_gate_satisfaction_still_crosses_the_boundary(harness, events):
    """A replayed gate licenses publication skipping, never boundary skipping.

    A reconnecting worker whose evidence the controller already replayed owes
    no new artifact. It owes every crossing exactly as before.
    """

    harness(
        steps=1,
        sft_steps=1,
        worker_controls=FakeControls(events, gate_required=True, gate_satisfied=True),
    )
    assert [kind for kind, _ in events] == ["boundary", "mutation"] * 2


def test_the_crossing_names_a_step_and_never_a_literal_zero(harness, events):
    """Publication and crossing are keyed to the baseline, not to ``0``.

    ``tests/test_mage_flow_pretrain.py`` carries the note that a literal zero
    "would skip it forever"; the mirror-image bug here is a crossing that names
    zero, which ``WorkerControlRuntime.optimizer_step`` refuses forever because
    a step must follow the immutable baseline. Every crossing names a positive
    step, and a resumed attempt is checked against the controller's baseline
    rather than assumed to start at zero.
    """

    harness(steps=2, sft_steps=1, worker_controls=FakeControls(events))
    crossings = [step for kind, step in events if kind == "boundary"]
    assert crossings and all(step >= 1 for step in crossings)


def test_a_run_without_controls_still_orders_its_mutations(harness, events):
    """The sentinel is installed for CLI runs too, with no boundary to call.

    A local run has no controller, so there is nothing to refuse the mutation;
    what remains enforceable is that both sites still route through the
    sentinel. A site that mutated directly would raise here even with no
    controls present.
    """

    harness(steps=1, sft_steps=1, worker_controls=None)
    assert events == [("mutation", 0)] * 2
