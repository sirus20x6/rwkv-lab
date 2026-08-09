"""Measure whether a multimodal model's captions depend on pixel content.

A caption-distillation run reports one number: loss against the reference
caption.  That number stays plausible when the vision tower is broken, because
a language model good enough to distil is also good enough to guess a caption
from the prompt, the caption-length prior, and the token statistics of the
target itself.  The Qwen3.6 caption audit is the worked example: the checkpoint
nests its ViT at ``model.language_model.visual.*`` while
``Qwen3_5MoeForConditionalGeneration`` looks for ``model.visual.*``, so 333
pretrained vision tensors loaded as *missing* and were freshly initialised,
and training proceeded against a random image encoder with the loss curve
looking entirely ordinary.

The obvious test -- feed different images, check the outputs differ -- does not
detect that.  A *random* frozen ViT is still a function of its input, so it
also produces different outputs for different images.  Sensitivity to pixels
proves only that pixels reach the graph; it does not prove the model reads
them.  This module therefore reports two independent quantities and keeps them
separate:

``responsiveness``
    Does image content reach the computation at all?  Perturbing the image
    while holding the prompt fixed must move the caption's log-likelihood.  A
    zero here means the vision path is disconnected, the pixels are being
    dropped, or the image tokens never make it into the sequence.  It is a
    wiring check and nothing more.

``discrimination``
    Does the model read the image *for its content*?  Scored across a set of
    image/caption pairs, the model must prefer each image's own caption to
    another image's caption.  The margin is measured in nats per target token
    and is positive only when the encoder carries semantics.  A random ViT
    scores about zero here however responsive it is, which is exactly what
    makes this the load-bearing measurement.

Only the second one can distinguish a working vision tower from a broken one,
so a verdict requires both and names which one failed.

The image variants are fixed and deliberately chosen so that the pair of
metrics is interpretable:

``original``
    The image as the dataset stores it.  The reference condition.
``repeated``
    The same pixels a second time.  A control, not a perturbation: it must
    reproduce ``original`` to within numerical tolerance.  If it does not, the
    forward pass is nondeterministic and no other comparison here means
    anything, so the verdict short-circuits on it.
``shuffled``
    The image's own patch grid permuted by a seeded permutation.  Identical
    colour histogram, identical patch content, destroyed spatial structure --
    so a model that has merely learned "photographs are beige" is not credited
    for it.
``blank``
    A uniform mid-grey field of the same size.  The furthest departure from the
    original that is still a valid image, and the variant a disconnected vision
    path cannot distinguish from any other.

Nothing here imports torch or transformers.  A ``ScoreFunction`` maps
(image, caption) to a mean negative log-likelihood per target token, and the
module never learns how that is computed.  That keeps the decision logic
runnable, and tested, on a CPU with a stub -- including against a deliberately
pixel-blind stub, which the suite requires to FAIL -- while the real 35B
evaluation stays an opt-in GPU job.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

SCHEMA = "rwkv-lab.image-conditioning-probe.v1"

#: The variant an evaluation is measured against.
REFERENCE_VARIANT = "original"

#: The identical-pixels control. Required, not optional -- see
#: :func:`evaluate_image_conditioning`.
DETERMINISM_VARIANT = "repeated"

#: Repeating identical pixels must reproduce the reference score.  This is a
#: determinism control, so the bound is numerical rather than behavioural:
#: bf16 accumulation over a caption-length sequence lands well inside it, while
#: a genuinely nondeterministic kernel or a stateful cache does not.
REPEAT_DETERMINISM_TOLERANCE_NATS = 1.0e-3

#: A perturbation must move the caption NLL by at least this much (nats per
#: target token) to count as reaching the computation.  Chosen an order of
#: magnitude above the determinism tolerance so that "responsive" can never be
#: satisfied by the noise floor that the repeat control measures.
DEFAULT_RESPONSIVENESS_THRESHOLD_NATS = 1.0e-2

#: A model reads image content when its own caption is cheaper than another
#: image's caption by at least this margin, in nats per target token. A random
#: frozen ViT scores ~0 here; the threshold sits above plausible drift from
#: caption-length and vocabulary effects without demanding a large margin.
DEFAULT_DISCRIMINATION_THRESHOLD_NATS = 5.0e-2

#: Fraction of pairs whose own caption must win outright. Chance is 0.5, so
#: this asks for a clear majority rather than unanimity: a caption corpus
#: contains near-duplicate scenes, and one pair losing is not evidence of a
#: blind encoder.
DEFAULT_DISCRIMINATION_ACCURACY = 0.75


class ScoreFunction(Protocol):
    """Return mean negative log-likelihood per target token, in nats.

    ``image`` is whatever the caller's variant builder produced; this module
    passes it through untouched.  A lower score means the caption is a better
    fit for the image.
    """

    def __call__(self, image: Any, caption: str) -> float: ...


@dataclass(frozen=True, slots=True)
class ProbeExample:
    """One image with the caption that belongs to it."""

    identifier: str
    image: Any
    caption: str

    def __post_init__(self) -> None:
        if not self.identifier:
            raise ValueError("probe example needs a non-empty identifier")
        if not self.caption.strip():
            raise ValueError(f"probe example {self.identifier} has an empty caption")


@dataclass(frozen=True, slots=True)
class VariantScore:
    variant: str
    score_nats: float
    #: Absolute distance from the reference variant's score. Zero for the
    #: reference itself.
    delta_from_reference_nats: float


@dataclass(frozen=True, slots=True)
class ExampleReport:
    identifier: str
    variants: tuple[VariantScore, ...]
    #: NLL of this example's own caption against its own image.
    matched_nats: float
    #: Mean NLL of *other* examples' captions against this image.
    mismatched_mean_nats: float

    @property
    def discrimination_margin_nats(self) -> float:
        return self.mismatched_mean_nats - self.matched_nats

    @property
    def prefers_own_caption(self) -> bool:
        return self.discrimination_margin_nats > 0.0


@dataclass(frozen=True, slots=True)
class ConditioningVerdict:
    """The measurement, its thresholds, and the pass/fail each one produced."""

    schema: str
    subject: str
    example_count: int
    variants: tuple[str, ...]
    #: Worst-case |score(repeat) - score(original)| over the examples.
    repeat_determinism_nats: float
    #: Smallest perturbation response over (example, perturbed variant). The
    #: minimum rather than the mean: one variant the model cannot distinguish
    #: is the interesting case, and a mean hides it.
    minimum_responsiveness_nats: float
    #: Mean over examples of (mismatched mean NLL - matched NLL).
    discrimination_margin_nats: float
    #: Fraction of examples that prefer their own caption.
    discrimination_accuracy: float
    responsiveness_threshold_nats: float
    discrimination_threshold_nats: float
    discrimination_accuracy_threshold: float
    deterministic: bool
    responsive: bool
    discriminating: bool
    failures: tuple[str, ...]
    examples: tuple[ExampleReport, ...] = field(default=())

    @property
    def consumes_image_content(self) -> bool:
        return not self.failures

    def canonical_dict(self) -> dict[str, Any]:
        document = asdict(self)
        document["consumes_image_content"] = self.consumes_image_content
        return document

    def evidence_json(self) -> str:
        document = self.canonical_dict()
        encoded = json.dumps(
            document, ensure_ascii=False, allow_nan=False,
            separators=(",", ":"), sort_keys=True,
        )
        document["verdict_digest"] = (
            "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        )
        return json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True)


def _finite(value: float, *, what: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{what} is not a finite score: {value!r}")
    return numeric


def evaluate_image_conditioning(
    examples: Sequence[ProbeExample],
    score: ScoreFunction,
    *,
    variant_builder: Callable[[Any, str], Any],
    subject: str,
    variants: Sequence[str] = ("original", "repeated", "shuffled", "blank"),
    responsiveness_threshold_nats: float = DEFAULT_RESPONSIVENESS_THRESHOLD_NATS,
    discrimination_threshold_nats: float = DEFAULT_DISCRIMINATION_THRESHOLD_NATS,
    discrimination_accuracy_threshold: float = DEFAULT_DISCRIMINATION_ACCURACY,
) -> ConditioningVerdict:
    """Score every (example, variant) pair and decide whether pixels are read.

    ``variant_builder(image, variant)`` returns the perturbed image; see
    :func:`build_image_variant` for the implementation this module ships.
    ``score`` is called once per (variant, caption) combination, so the cost is
    ``len(examples) * (len(variants) + len(examples) - 1)`` forward passes.
    """
    if len(examples) < 2:
        raise ValueError(
            "image conditioning needs at least two examples: the discrimination "
            "measurement compares each image against another image's caption"
        )
    if REFERENCE_VARIANT not in variants:
        raise ValueError(f"variants must include the {REFERENCE_VARIANT!r} reference")
    if DETERMINISM_VARIANT not in variants:
        # Without the control, `deterministic` would report True because
        # nothing contradicted it -- an unmeasured claim indistinguishable in
        # the evidence from a measured one. The whole verdict rests on it, so
        # it is required rather than defaulted.
        raise ValueError(
            f"variants must include the {DETERMINISM_VARIANT!r} determinism control"
        )
    if len(set(variants)) != len(variants):
        raise ValueError("variants must be unique")
    identifiers = [example.identifier for example in examples]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("probe examples must have unique identifiers")

    reports: list[ExampleReport] = []
    worst_repeat = 0.0
    minimum_response = math.inf

    for example in examples:
        scores: dict[str, float] = {}
        for variant in variants:
            image = variant_builder(example.image, variant)
            scores[variant] = _finite(
                score(image, example.caption),
                what=f"{example.identifier}/{variant}",
            )
        reference = scores[REFERENCE_VARIANT]
        variant_scores = tuple(
            VariantScore(
                variant=variant,
                score_nats=scores[variant],
                delta_from_reference_nats=abs(scores[variant] - reference),
            )
            for variant in variants
        )
        for entry in variant_scores:
            if entry.variant == REFERENCE_VARIANT:
                continue
            if entry.variant == DETERMINISM_VARIANT:
                worst_repeat = max(worst_repeat, entry.delta_from_reference_nats)
                continue
            minimum_response = min(minimum_response, entry.delta_from_reference_nats)

        others = [other for other in examples if other.identifier != example.identifier]
        mismatched = [
            _finite(
                score(variant_builder(example.image, REFERENCE_VARIANT), other.caption),
                what=f"{example.identifier}/mismatched/{other.identifier}",
            )
            for other in others
        ]
        reports.append(
            ExampleReport(
                identifier=example.identifier,
                variants=variant_scores,
                matched_nats=reference,
                mismatched_mean_nats=sum(mismatched) / len(mismatched),
            )
        )

    if minimum_response is math.inf:
        # Only the reference and its control were requested, so nothing
        # perturbs the image and responsiveness is unmeasured, not perfect.
        raise ValueError("variants must include at least one perturbed variant")

    margin = sum(report.discrimination_margin_nats for report in reports) / len(reports)
    accuracy = sum(report.prefers_own_caption for report in reports) / len(reports)

    deterministic = worst_repeat <= REPEAT_DETERMINISM_TOLERANCE_NATS
    responsive = minimum_response >= responsiveness_threshold_nats
    discriminating = (
        margin >= discrimination_threshold_nats
        and accuracy >= discrimination_accuracy_threshold
    )

    failures: list[str] = []
    if not deterministic:
        # Reported alone: with a nondeterministic forward pass the other two
        # numbers are measuring noise, and saying so is more useful than
        # listing three failures with one cause.
        failures.append(
            f"repeating identical pixels moved the score by {worst_repeat:.3e} nats "
            f"(tolerance {REPEAT_DETERMINISM_TOLERANCE_NATS:.3e}); the forward pass "
            "is not deterministic, so the remaining measurements are unreliable"
        )
    else:
        if not responsive:
            failures.append(
                f"the least-distinguished perturbation moved the score by only "
                f"{minimum_response:.3e} nats (threshold "
                f"{responsiveness_threshold_nats:.3e}); image content is not "
                "reaching the computation"
            )
        if not discriminating:
            failures.append(
                f"own-caption preference is {margin:+.3e} nats at "
                f"{accuracy:.0%} accuracy (thresholds "
                f"{discrimination_threshold_nats:.3e} nats, "
                f"{discrimination_accuracy_threshold:.0%}); the model responds to "
                "pixels but does not read them for content, which is what a "
                "randomly initialised vision tower looks like"
            )

    return ConditioningVerdict(
        schema=SCHEMA,
        subject=subject,
        example_count=len(examples),
        variants=tuple(variants),
        repeat_determinism_nats=worst_repeat,
        minimum_responsiveness_nats=minimum_response,
        discrimination_margin_nats=margin,
        discrimination_accuracy=accuracy,
        responsiveness_threshold_nats=responsiveness_threshold_nats,
        discrimination_threshold_nats=discrimination_threshold_nats,
        discrimination_accuracy_threshold=discrimination_accuracy_threshold,
        deterministic=deterministic,
        responsive=responsive,
        discriminating=discriminating,
        failures=tuple(failures),
        examples=tuple(reports),
    )


def hf_caption_score_function(
    model: Any,
    processor: Any,
    *,
    system_prompt: str | None,
    prompt: str,
    device: Any = None,
) -> ScoreFunction:
    """Bind :func:`evaluate_image_conditioning` to a Hugging Face image-text model.

    The conversation is rendered exactly as
    ``rwkv_lab.trainvm_adapters.hf_multimodal_sft`` renders it for training --
    one ``{"type": "image"}`` part followed by the prompt text, then the caption
    as the assistant turn -- so a score here describes the same input path the
    caption run used rather than an approximation of it.

    Only the assistant target is scored.  Including the prompt would average the
    caption's likelihood together with a long, image-independent preamble, which
    shrinks every difference this probe is trying to measure.
    """
    import torch

    from rwkv_lab.trainvm_adapters.hf_multimodal_sft import _conversation, _template

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("multimodal processor exposes no tokenizer")

    def score(image: Any, caption: str) -> float:
        rendered_prompt = _template(
            processor, _conversation(system_prompt, prompt), generation=True
        )
        rendered_full = _template(
            processor, _conversation(system_prompt, prompt, caption), generation=False
        )
        if not rendered_full.startswith(rendered_prompt):
            raise ValueError("chat template has no stable assistant-only boundary")
        encoded = processor(
            text=[rendered_full], images=[image], padding=False, return_tensors="pt"
        )
        prompt_encoded = processor(
            text=[rendered_prompt], images=[image], padding=False, return_tensors="pt"
        )
        prompt_width = int(prompt_encoded["input_ids"].shape[1])
        input_ids = encoded["input_ids"]
        if int(input_ids.shape[1]) <= prompt_width:
            raise ValueError("rendered caption contributed no target tokens")
        labels = input_ids.clone()
        labels[:, :prompt_width] = -100
        tensors = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in encoded.items()
        }
        tensors["labels"] = labels.to(device) if device is not None else labels
        with torch.inference_mode():
            outputs = model(**tensors)
        # The model's own shifted mean cross-entropy over unmasked positions is
        # already nats per target token, which is the unit this module compares.
        return float(outputs.loss.detach().float().cpu())

    return score


def patch_permutation(patch_count: int, *, seed: int) -> tuple[int, ...]:
    """A seeded derangement-ish permutation of patch indices.

    Seeded rather than random so a recorded verdict is reproducible, and
    rejected when it is the identity so "shuffled" is never silently a second
    copy of "original".
    """
    if patch_count < 2:
        raise ValueError("patch shuffling needs at least two patches")
    import random

    order = list(range(patch_count))
    generator = random.Random(seed)
    for _ in range(8):
        generator.shuffle(order)
        if any(index != position for position, index in enumerate(order)):
            return tuple(order)
    raise ValueError("patch permutation kept landing on the identity")


def build_image_variant(image: Any, variant: str, *, patch: int = 64, seed: int = 20260809) -> Any:
    """Build one variant of a PIL image.

    ``patch`` is the shuffle block size in pixels. It is independent of the
    model's own patch size on purpose: the point is to destroy spatial layout
    at a scale the encoder cannot undo, not to align with its tokenisation.
    """
    if variant in (REFERENCE_VARIANT, DETERMINISM_VARIANT):
        return image.copy()
    if variant == "blank":
        from PIL import Image

        # Per-band mid-grey. A scalar fill would land on (128, 0, 0) for an RGB
        # image, which is a saturated red field rather than the neutral one the
        # variant is supposed to be.
        bands = len(Image.new(image.mode, (1, 1)).getbands())
        return Image.new(image.mode, image.size, color=(128,) * bands)
    if variant == "shuffled":
        from PIL import Image

        width, height = image.size
        columns, rows = max(1, width // patch), max(1, height // patch)
        if columns * rows < 2:
            raise ValueError(f"image {image.size} is too small to shuffle at patch {patch}")
        boxes = [
            (
                column * patch,
                row * patch,
                width if column == columns - 1 else (column + 1) * patch,
                height if row == rows - 1 else (row + 1) * patch,
            )
            for row in range(rows)
            for column in range(columns)
        ]
        order = patch_permutation(len(boxes), seed=seed)
        shuffled = Image.new(image.mode, image.size)
        for destination, source in enumerate(order):
            tile = image.crop(boxes[source])
            target = boxes[destination]
            shuffled.paste(
                tile.resize((target[2] - target[0], target[3] - target[1])),
                (target[0], target[1]),
            )
        return shuffled
    raise ValueError(f"unknown image variant {variant!r}")
