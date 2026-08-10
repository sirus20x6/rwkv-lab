"""Attempt-baseline step-zero evidence, shared by the four vision trainers.

The four vision routes -- frozen adapter, native head, raw-pixel student and
teacher compressor -- arm the universal pre-mutation eval gate together, and
everything about that arming except *what the evidence is* is identical for all
four. This module holds the identical part.

Written once rather than four times on purpose. The Transformer MLA family got
away with one copy because its eight routes share one trainer; these four are
four separate trainers, so the alternative here is four copies of the same
ordering, the same tri-state, the same refusal and the same two publish calls.
Copies of a publication sequence drift silently -- a fix applied to three of
four leaves the fourth deadlocking with no diagnostic, which is precisely the
failure mode arming a route is supposed to remove.

What is deliberately *not* here is any evidence construction. Three of the four
routes emit an image / reference-caption / predicted-caption triple and the
fourth emits held-out teacher-feature reconstruction with no text anywhere in
it; each trainer builds its own examples and hands them in. A helper that tried
to render both would have to invent the text half of the compressor's triple,
which is the synthetic evidence the gate exists to refuse.

None of this module may import ``rwkv_lab.trainvm_worker`` at module scope. All
four trainers are runnable as standalone CLIs on a host without the
``trainvm-worker`` extra installed, and that package raises at import when the
generated protobuf module is absent.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from typing import Any


def selection_digest(value: object) -> str:
    """Digest a JSON-serialisable description of a frozen selection.

    ``ensure_ascii`` is left at its default rather than matching
    ``trainvm_worker._canonical.canonical_dumps``, and that difference is load
    bearing here in a way it was not for the text routes: a vision selector
    names *image manifest paths*, which can hold non-ASCII bytes. Escaping them
    keeps this fold total over any string the selector can carry, and the digest
    only ever has to agree with itself across attempts of one experiment -- it
    is never compared against a canonical-dumps digest computed elsewhere.
    """

    return "sha256:" + hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def select_heldout_indices(population: int, sample_count: int) -> tuple[int, ...]:
    """Choose the frozen held-out row offsets, deterministically.

    Evenly-spaced distinct rows over the held-out population, keyed to nothing
    but the population size and the count. That makes the selection a pure
    function of the composition and the evaluation manifest, so a replacement
    attempt against the same inputs draws the same rows and its evidence is
    comparable with the attempt it replaced -- which is the whole point of
    freezing a selection rather than sampling one.

    Deliberately not seeded from the run's ``--seed``. The training stream is
    already seeded from it, and every one of these four routes resumes, so a
    held-out selection that moved with the resume point would make two attempts
    of one run incomparable.

    A population too small for the requested count is an error rather than a
    quietly overlapping or truncated selection: duplicate held-out identities
    are refused by the publisher anyway, and the diagnostic is much better here.
    """

    if (
        isinstance(population, bool)
        or isinstance(sample_count, bool)
        or not isinstance(population, int)
        or not isinstance(sample_count, int)
        or sample_count < 1
    ):
        raise ValueError("held-out selection geometry is invalid")
    if population < sample_count:
        raise ValueError(
            f"held-out population holds {population} rows, fewer than the "
            f"{sample_count} distinct rows the composition asks to evaluate"
        )
    stride = population // sample_count
    return tuple(index * stride for index in range(sample_count))


def refuse_ungated_attempt_baseline(
    worker_controls: Any | None,
    start_step: int,
    *,
    family: str,
    can_publish_baseline_evidence: bool = False,
) -> None:
    """Refuse an attempt the controller's baseline does not admit.

    The controller gates an attempt at an immutable baseline step: zero for a
    fresh attempt, the resume checkpoint's step for a replacement one. The
    pre-mutation crossing names ``step + 1`` and ``WorkerControlRuntime``
    refuses any step at or below the baseline, so an attempt whose loop begins
    somewhere else can never reach a legal first mutation -- it would stall at
    the first crossing with nothing saying why, after a full teacher-cache
    validation and a frozen-stack load.

    All four vision routes declare ``resumable_training_lifecycle()``, so the
    replacement-attempt case is reached here for real rather than in principle,
    and the comparison is against ``attempt_baseline_optimizer_step`` and never
    against a literal zero. A literal zero is right for the fresh attempt and
    silently wrong for every replacement one -- the bug ``mage_flow_pretrain.py``
    shipped and 8f0da3e fixed -- and a branch keyed to zero that never fires
    looks exactly like a branch with nothing to do.

    ``can_publish_baseline_evidence`` separates the two ways an unsatisfied gate
    can be reached, and getting it wrong is expensive in both directions. While
    these routes could not publish at all, an unsatisfied gate was terminal and
    refusing it early was strictly right. Now that they can, an unconditional
    refusal would reject every *armed fresh attempt* before it had the chance to
    publish the very evidence the refusal complains is missing -- a deadlock the
    trainer creates for itself. So the refusal is scoped to the case where
    nothing downstream will ever publish: no eval-examples policy was handed to
    this run.

    Neither branch is a way to skip the crossing. It happens on every step
    regardless, and a controller that replayed durable evidence from its journal
    on reconnect reports the gate satisfied and this returns.
    """

    if worker_controls is None:
        return
    baseline = worker_controls.attempt_baseline_optimizer_step
    if start_step != baseline:
        raise ValueError(
            f"{family} resumed at a step the controller does not gate: "
            f"{start_step} != {baseline}"
        )
    if (
        worker_controls.step_zero_eval_gate_required
        and not worker_controls.step_zero_eval_gate_satisfied
        and not can_publish_baseline_evidence
    ):
        raise ValueError(
            f"{family} cannot mutate: the controller requires durable "
            f"attempt-baseline evidence at step {baseline}, none is recorded "
            "for this attempt, and this run was given no eval-examples policy "
            "to publish it with"
        )


def attempt_baseline_gate(worker_controls: Any | None) -> int | None:
    """The step this attempt owes evidence at, or ``None`` when it owes none.

    Three outcomes, not two. ``None`` when there is no controller, when the
    controller does not require the gate, or when a reconnecting or replacement
    worker's evidence was already replayed from the controller's journal -- that
    last case must not republish, and it never licenses skipping the
    pre-mutation crossing itself. Otherwise the controller's own immutable
    baseline step, read from the controller rather than inferred from the
    trainer's resume point.
    """

    if worker_controls is None or not worker_controls.step_zero_eval_gate_required:
        return None
    if worker_controls.step_zero_eval_gate_satisfied:
        return None
    return worker_controls.attempt_baseline_optimizer_step


def publish_attempt_baseline_evidence(
    *,
    worker_controls: Any,
    worker_observability: Any | None,
    policy: Any,
    baseline: int,
    series_id: str,
    identities: Sequence[str],
    selector: Mapping[str, object],
    examples: tuple[Any, ...],
    stage_checkpoint: Callable[[], str],
    resume_grade: str,
    state_components: tuple[str, ...],
) -> None:
    """Publish the attempt-baseline evidence the controller is waiting on.

    Ordering, not decoration. The caller has already published this attempt's
    declared ``eval.*`` scalars at this step, and the controller's
    ``validate_eval_examples_gate_provenance`` requires both a prior durable
    declared scalar *and* a prior durable checkpoint artifact at the manifest's
    own step before it will accept the examples. So: scalars, then checkpoint,
    then examples -- and all three before the loop can reach a mutation.
    """

    from rwkv_lab.trainvm_worker.checkpoint import CheckpointPublicationRequest
    from rwkv_lab.trainvm_worker.eval_examples import EvalExamplesPublicationRequest

    if worker_controls is None or policy is None:
        raise ValueError(
            "vision attempt-baseline publication requires controls and an "
            "eval-examples policy"
        )
    if not examples:
        raise ValueError("vision attempt-baseline evidence is empty")
    with (
        worker_observability.keepalive(baseline, "checkpointing")
        if worker_observability is not None
        else nullcontext()
    ):
        staged = stage_checkpoint()
    published = worker_controls.publish_policy_checkpoint(
        CheckpointPublicationRequest(
            source_directory=staged,
            optimizer_step=baseline,
            resume_grade=resume_grade,
            state_components=state_components,
        )
    )
    worker_controls.publish_evaluation_examples(
        EvalExamplesPublicationRequest(
            output_name="eval_examples",
            optimizer_step=baseline,
            series_id=series_id,
            identity_field=policy.identity_field,
            identities_digest=selection_digest(list(identities)),
            selector_digest=selection_digest(dict(selector)),
            evaluator_component_digest=policy.evaluator_component_digest,
            metric_names=policy.metric_names,
            checkpoint_artifact_id=published.artifact_id,
            checkpoint_manifest_digest=published.manifest_sha256,
            # None of the four vision composition contracts declares a
            # `generation_policy` slot, so there is no authored decode policy to
            # carry a digest of. The frozen evaluation manifest's policy digest
            # is composed instead, from the three resolved components that
            # decide what was evaluated and how it is rendered. Cadence is
            # absent on purpose: an evaluation-schedule change must leave this
            # digest where it was, or two revisions of one experiment stop being
            # comparable for a reason unrelated to what was evaluated.
            policy_digest=selection_digest(
                {
                    "artifact_renderer": policy.artifact_renderer_digest,
                    "evaluator": policy.evaluator_component_digest,
                    "qualitative_sample": policy.qualitative_sample_digest,
                }
            ),
            examples=examples,
            parent_artifact_ids=(published.artifact_id,),
        )
    )
    print(
        f"attempt-baseline eval gate: published {len(examples)} held-out "
        f"{series_id} examples at step {baseline}",
        flush=True,
    )


def bounded_text(value: object, *, limit: int = 512) -> str:
    """Fit arbitrary caption text inside the manifest's per-part budget.

    The publisher bounds a text part at 16 KiB and the whole manifest at 60 KiB,
    and a manifest that exceeds either is refused *after* a model has run. With
    up to 512 held-out examples carrying three parts each, the manifest bound is
    the one that binds, so the truncation here is much tighter than the part
    bound and is applied to every rendered string rather than only to the ones
    that look long.

    Newlines and the other separators the publisher rejects are folded to
    spaces. A caption containing one is otherwise a refusal at publication time
    for a reason that has nothing to do with the model.
    """

    text = " ".join(str(value).split())
    if not text:
        # An empty part is refused by the publisher, and an empty reference
        # caption is a real thing in a scraped manifest. Say so rather than
        # failing the whole attempt over one row.
        return "(empty)"
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text


def caption_eval_examples(
    items: Sequence[tuple[str, str, str, str]], *, optimizer_step: int
) -> tuple[Any, ...]:
    """Render held-out caption rows as typed step-zero evidence.

    Each item is ``(identity, prompt, reference, predicted)``. Modality
    appropriate for the three vision routes that emit text: the input is the
    held-out image's identity and the prompt it was captioned under, the target
    is the caption the manifest actually carries, and the prediction is what the
    model emits at those positions under teacher forcing. That is the
    image/caption/prediction triple ``artifact_renderer/caption_triplet``
    describes, which is why the three caption routes are allowed that renderer
    and the compressor route is not.

    ``example_id`` folds the step in and ``heldout_item_id`` deliberately does
    not: the publisher requires both unique within one manifest, and the item id
    has to stay stable across attempts or two attempts of one run stop being
    comparable. The digest is over the held-out item -- identity, prompt and
    reference -- and never over the model's output, so it identifies the
    question rather than the answer.
    """

    from rwkv_lab.trainvm_worker.eval_examples import EvalEvidencePart, EvalExample

    if (
        isinstance(optimizer_step, bool)
        or not isinstance(optimizer_step, int)
        or optimizer_step < 0
    ):
        raise ValueError("vision held-out optimizer step is invalid")
    examples: list[Any] = []
    for identity, prompt, reference, predicted in items:
        item = bounded_text(identity, limit=1024)
        examples.append(
            EvalExample(
                example_id=f"caption-{item}:step:{optimizer_step}",
                heldout_item_id=item,
                heldout_item_digest=selection_digest(
                    [item, str(prompt), str(reference)]
                ),
                input=(
                    EvalEvidencePart(
                        kind="text",
                        text=bounded_text(f"image: {item} | prompt: {prompt}"),
                    ),
                ),
                target=(EvalEvidencePart(kind="text", text=bounded_text(reference)),),
                prediction=(
                    EvalEvidencePart(kind="text", text=bounded_text(predicted)),
                ),
            )
        )
    return tuple(examples)


def reconstruction_eval_examples(
    items: Sequence[tuple[str, Mapping[str, float], Mapping[str, float]]],
    *,
    optimizer_step: int,
    schema: str,
) -> tuple[Any, ...]:
    """Render held-out feature-reconstruction rows as typed step-zero evidence.

    Each item is ``(identity, target_statistics, predicted_statistics)``. For
    ``vision_teacher_compressor`` there is no caption to render -- it distils
    frozen teacher features and its objective contains no text at all -- so the
    evidence is structured rather than textual: the input names the held-out
    teacher-cache row, the target summarises the teacher features that row
    carries, and the prediction summarises what the compressor reconstructed
    from them together with the error against the target.

    This is exactly why ``caption_triplet`` is withheld from that route in
    ``rwkv_lab_worker_contract.cpp``: a renderer that had to invent the text
    half of a triple would be manufacturing the evidence the gate exists to
    check.
    """

    from rwkv_lab.trainvm_worker.eval_examples import EvalEvidencePart, EvalExample

    if (
        isinstance(optimizer_step, bool)
        or not isinstance(optimizer_step, int)
        or optimizer_step < 0
    ):
        raise ValueError("vision held-out optimizer step is invalid")

    def numbers(values: Mapping[str, float]) -> dict[str, float]:
        rendered = {
            str(name): round(float(value), 6) for name, value in values.items()
        }
        if not rendered:
            raise ValueError("vision reconstruction evidence carries no statistics")
        return rendered

    examples: list[Any] = []
    for identity, target, predicted in items:
        item = bounded_text(identity, limit=1024)
        target_values = numbers(target)
        examples.append(
            EvalExample(
                example_id=f"reconstruction-{item}:step:{optimizer_step}",
                heldout_item_id=item,
                heldout_item_digest=selection_digest([item, target_values]),
                input=(
                    EvalEvidencePart(
                        kind="text", text=bounded_text(f"teacher_cache_row: {item}")
                    ),
                ),
                target=(
                    EvalEvidencePart(
                        kind="structured", schema=schema, value=target_values
                    ),
                ),
                prediction=(
                    EvalEvidencePart(
                        kind="structured", schema=schema, value=numbers(predicted)
                    ),
                ),
            )
        )
    return tuple(examples)


__all__ = [
    "attempt_baseline_gate",
    "bounded_text",
    "caption_eval_examples",
    "reconstruction_eval_examples",
    "publish_attempt_baseline_evidence",
    "refuse_ungated_attempt_baseline",
    "select_heldout_indices",
    "selection_digest",
]
