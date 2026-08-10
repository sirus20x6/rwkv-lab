"""The composition-derived half of the vision family's step-zero evidence.

Split the way :class:`~rwkv_lab.trainvm_adapters.rlvr.RLVRHeldoutEvalPolicy`
and :class:`~rwkv_lab.trainvm_adapters.transformer_mla.TransformerMLAEvalPolicy`
are split, and for the same reason: everything here is decided by the
*resolved* training composition and is therefore knowable in the handler,
before a teacher cache is opened or a frozen caption stack is loaded. The
data-derived half -- which held-out rows were drawn, and what the model
predicted for them -- is not here, because it is not decidable until the
trainer has built its evaluation dataset.

One class for four routes rather than four near-identical ones. The four vision
contracts do differ in what their evidence *is*: three predict a caption from
teacher features or raw pixels, and ``vision_teacher_compressor`` distils
frozen teacher features with no text anywhere in its objective. But that
difference lives entirely in the trainer's manifest payload. What the
composition contributes -- the evaluator whose digest and metric names
``validate_eval_examples_gate_provenance`` cross-checks against the published
manifest, and the renderer and held-out selector that compose the frozen
``policy_digest`` -- is the same four fields in every one of them, because
``evaluation_slots()`` in ``rwkv_lab_worker_contract.cpp`` declares one slot
set for all four. A per-route copy would be the same six fields written four
times, and the copies would drift.

``RWKVTextEvalPolicy`` is the other candidate template and is the wrong one
twice over, exactly as it was for RLVR and Transformer MLA. It bounds every
held-out token to ``0 <= token < 65_536``, which is the scratch-RWKV
vocabulary; the caption routes here tokenize with ``WorldVocab`` and the
compressor route has no tokens at all. And it carries a
``generation_policy_digest``, while none of the four vision composition
contracts declares a ``generation_policy`` slot -- there would be no digest to
put there. So the frozen evaluation manifest's ``policy_digest`` is *composed*
by the trainer from the renderer, the held-out selector and the evaluator.

The evaluation cadence is deliberately not an input to any of this. A schedule
change must leave the digest where it was, or two revisions of the same
experiment stop being comparable for a reason that has nothing to do with what
was evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass


def _digest_string(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ValueError(f"{label} must be a sha256: digest")
    return value


def _integer(value: object, label: str, minimum: int, maximum: int) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{label} must be an integer in [{minimum}, {maximum}]")
    return value


@dataclass(frozen=True, slots=True)
class VisionEvalPolicy:
    """What the resolved composition decides about vision step-zero evidence."""

    identity_field: str
    evaluator_component_digest: str
    metric_names: tuple[str, ...]
    artifact_renderer_digest: str
    qualitative_sample_digest: str
    sample_count: int

    def __post_init__(self) -> None:
        for label in (
            "evaluator_component_digest",
            "artifact_renderer_digest",
            "qualitative_sample_digest",
        ):
            _digest_string(getattr(self, label), label)
        if (
            not isinstance(self.identity_field, str)
            or not self.identity_field
            or len(self.identity_field.encode("utf-8")) > 256
        ):
            raise ValueError("vision eval identity field is required")
        if (
            not isinstance(self.metric_names, tuple)
            or not self.metric_names
            or self.metric_names != tuple(sorted(set(self.metric_names)))
            or any(
                not isinstance(metric, str) or not metric
                for metric in self.metric_names
            )
        ):
            raise ValueError(
                "vision eval metric names must be nonempty, sorted and unique"
            )
        # The publisher's own ceiling is 512 examples per manifest, and the
        # `fixed_held_out` descriptor bounds `sample_count` to the same number.
        # Restating it here means a policy that could never be published is
        # refused in the handler rather than after a model load.
        _integer(self.sample_count, "sample_count", 1, 512)


__all__ = ["VisionEvalPolicy"]
