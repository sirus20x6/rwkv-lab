"""Tests for the image-conditioning probe.

The probe exists because a caption run's own loss curve cannot tell a working
vision tower from a randomly initialised one -- the Qwen3.6 audit found 333
pretrained ViT tensors silently discarded while training looked healthy.  So
the thing these tests have to establish is not that the probe runs, but that it
*fails* the case it was built to catch.

The load-bearing test is therefore
``test_pixel_blind_model_fails_the_conditioning_verdict``: a model that ignores
pixels entirely must be rejected.  Its near-miss neighbour matters just as
much.  A randomly initialised frozen ViT is still a function of its input, so a
probe that only asked "do different images give different outputs" would pass
one.  ``test_pixel_responsive_but_semantically_blind_model_fails`` pins that
distinction, and is the reason the verdict carries two independent metrics
instead of one.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys

import pytest

from rwkv_lab.image_conditioning_probe import (
    DEFAULT_DISCRIMINATION_THRESHOLD_NATS,
    REPEAT_DETERMINISM_TOLERANCE_NATS,
    SCHEMA,
    ProbeExample,
    build_image_variant,
    evaluate_image_conditioning,
    patch_permutation,
)

_PROMPT_VARIANTS = ("original", "repeated", "shuffled", "blank")


def _stub_variant_builder(image: object, variant: str) -> object:
    """Represent a variant as a plain tuple so the stubs stay pixel-free."""
    return (image, "original" if variant == "repeated" else variant)


def _examples(count: int = 4) -> tuple[ProbeExample, ...]:
    return tuple(
        ProbeExample(
            identifier=f"image-{index}",
            image=index,
            caption=f"a photograph of subject {index}",
        )
        for index in range(count)
    )


def _evaluate(score, examples=None, **overrides):
    overrides.setdefault("variants", _PROMPT_VARIANTS)
    return evaluate_image_conditioning(
        examples if examples is not None else _examples(),
        score,
        variant_builder=_stub_variant_builder,
        subject="stub",
        **overrides,
    )


def _caption_index(caption: str) -> int:
    return int(caption.rsplit(" ", 1)[1])


def _seeing_score(image_and_variant, caption: str) -> float:
    """A model that reads the image: its own caption is cheapest."""
    image, variant = image_and_variant
    if variant != "original":
        # A perturbed image no longer matches any caption particularly well.
        return 3.0 + 0.1 * variant.count("a")
    return 1.0 if _caption_index(caption) == image else 2.0


def _pixel_blind_score(image_and_variant, caption: str) -> float:
    """A model that never looks at pixels: only the caption's length matters."""
    return 1.0 + 0.001 * len(caption)


def _random_feature_score(image_and_variant, caption: str) -> float:
    """A frozen *randomly initialised* encoder.

    Strongly image-dependent -- a random projection of the pixels still varies
    with them -- but carrying no semantics, so the image's own caption is no
    cheaper than anyone else's.  This is what the broken Qwen3.6 load produced.
    """
    image, variant = image_and_variant
    digest = hashlib.sha256(f"{image}/{variant}".encode()).digest()
    return 2.0 + digest[0] / 255.0


def test_probe_accepts_a_model_that_reads_the_image() -> None:
    verdict = _evaluate(_seeing_score)

    assert verdict.consumes_image_content
    assert verdict.deterministic and verdict.responsive and verdict.discriminating
    assert verdict.failures == ()
    assert verdict.schema == SCHEMA
    assert verdict.discrimination_accuracy == 1.0
    assert verdict.discrimination_margin_nats == pytest.approx(1.0)


def test_pixel_blind_model_fails_the_conditioning_verdict() -> None:
    verdict = _evaluate(_pixel_blind_score)

    assert not verdict.consumes_image_content
    assert not verdict.responsive
    assert not verdict.discriminating
    # Determinism must still hold: the failure is blindness, not noise, and a
    # verdict that blamed nondeterminism here would send a reader hunting the
    # wrong bug.
    assert verdict.deterministic
    assert any("not reaching the computation" in failure for failure in verdict.failures)


def test_pixel_responsive_but_semantically_blind_model_fails() -> None:
    """The random-ViT case: sensitive to pixels, blind to their content."""
    verdict = _evaluate(_random_feature_score)

    assert verdict.responsive, "a random encoder is still a function of its input"
    assert not verdict.discriminating
    assert not verdict.consumes_image_content
    assert abs(verdict.discrimination_margin_nats) < DEFAULT_DISCRIMINATION_THRESHOLD_NATS
    assert any("does not read them for content" in failure for failure in verdict.failures)


def test_nondeterministic_forward_pass_is_reported_alone() -> None:
    calls: dict[str, int] = {}

    def drifting(image_and_variant, caption: str) -> float:
        key = f"{image_and_variant}/{caption}"
        calls[key] = calls.get(key, 0) + 1
        return _seeing_score(image_and_variant, caption) + 0.5 * calls[key]

    verdict = _evaluate(drifting)

    assert not verdict.deterministic
    assert not verdict.consumes_image_content
    assert verdict.repeat_determinism_nats > REPEAT_DETERMINISM_TOLERANCE_NATS
    # One failure, not three: the other measurements have a single known cause.
    assert len(verdict.failures) == 1
    assert "not deterministic" in verdict.failures[0]


def test_repeated_variant_is_scored_as_a_control_not_a_perturbation() -> None:
    verdict = _evaluate(_seeing_score)

    assert verdict.repeat_determinism_nats == pytest.approx(0.0)
    # "repeated" must never be counted toward responsiveness; if it were, an
    # identical-pixels control would prove sensitivity, which is nonsense.
    assert verdict.minimum_responsiveness_nats > 0.0


def test_verdict_evidence_json_is_canonical_and_digested() -> None:
    document = json.loads(_evaluate(_seeing_score).evidence_json())

    assert document["schema"] == SCHEMA
    assert document["consumes_image_content"] is True
    assert document["verdict_digest"].startswith("sha256:")
    assert [entry["identifier"] for entry in document["examples"]] == [
        f"image-{index}" for index in range(4)
    ]
    assert [entry["variant"] for entry in document["examples"][0]["variants"]] == list(
        _PROMPT_VARIANTS
    )


def test_probe_refuses_inputs_that_cannot_support_a_verdict() -> None:
    with pytest.raises(ValueError, match="at least two examples"):
        _evaluate(_seeing_score, examples=_examples(1))
    with pytest.raises(ValueError, match="unique identifiers"):
        _evaluate(
            _seeing_score,
            examples=(_examples(1)[0], ProbeExample("image-0", 1, "another caption")),
        )
    with pytest.raises(ValueError, match="at least one perturbed variant"):
        _evaluate(_seeing_score, variants=("original", "repeated"))
    with pytest.raises(ValueError, match="must include the 'original' reference"):
        _evaluate(_seeing_score, variants=("blank", "shuffled"))


def test_probe_rejects_a_nonfinite_score() -> None:
    with pytest.raises(ValueError, match="not a finite score"):
        _evaluate(lambda image, caption: float("nan"))


def test_probe_example_rejects_empty_identity_or_caption() -> None:
    with pytest.raises(ValueError, match="non-empty identifier"):
        ProbeExample("", 0, "caption")
    with pytest.raises(ValueError, match="empty caption"):
        ProbeExample("image-0", 0, "   ")


def test_patch_permutation_is_seeded_and_never_the_identity() -> None:
    order = patch_permutation(16, seed=20260809)

    assert sorted(order) == list(range(16))
    assert order != tuple(range(16))
    assert order == patch_permutation(16, seed=20260809)
    assert order != patch_permutation(16, seed=1)
    with pytest.raises(ValueError, match="at least two patches"):
        patch_permutation(1, seed=0)


class TestImageVariants:
    """Variant construction, exercised against real PIL images."""

    @staticmethod
    def _image():
        Image = pytest.importorskip("PIL.Image")
        generator = random.Random(7)
        image = Image.new("RGB", (256, 192))
        image.putdata(
            [
                (
                    generator.randrange(256),
                    generator.randrange(256),
                    generator.randrange(256),
                )
                for _ in range(256 * 192)
            ]
        )
        return image

    def test_original_and_repeated_preserve_the_pixels(self) -> None:
        image = self._image()

        for variant in ("original", "repeated"):
            built = build_image_variant(image, variant)
            assert built is not image, "a variant must not alias the caller's image"
            assert built.tobytes() == image.tobytes()

    def test_blank_is_uniform_and_same_sized(self) -> None:
        image = self._image()

        blank = build_image_variant(image, "blank")

        assert blank.size == image.size
        assert set(blank.tobytes()) == {128}

    def test_shuffled_keeps_content_but_destroys_layout(self) -> None:
        image = self._image()

        shuffled = build_image_variant(image, "shuffled")

        assert shuffled.size == image.size
        assert shuffled.tobytes() != image.tobytes()
        # Same tiles, rearranged: the colour histogram is preserved, so a model
        # cannot score "shuffled" differently merely by average brightness.
        assert sorted(shuffled.tobytes()) == sorted(image.tobytes())

    def test_shuffled_is_reproducible_for_a_fixed_seed(self) -> None:
        image = self._image()

        assert (
            build_image_variant(image, "shuffled", seed=3).tobytes()
            == build_image_variant(image, "shuffled", seed=3).tobytes()
        )
        assert (
            build_image_variant(image, "shuffled", seed=3).tobytes()
            != build_image_variant(image, "shuffled", seed=4).tobytes()
        )

    def test_unknown_variant_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown image variant"):
            build_image_variant(self._image(), "greyscale")

    def test_image_too_small_to_shuffle_is_refused(self) -> None:
        Image = pytest.importorskip("PIL.Image")

        with pytest.raises(ValueError, match="too small to shuffle"):
            build_image_variant(Image.new("RGB", (32, 32)), "shuffled", patch=64)


# The real-model half of the probe. It is marked gpu, so CI never runs it: the
# gpu job is gated on the unset TRAINVM_SELF_HOSTED repository variable. It
# exists so the measurement is reproducible on a host that has the checkpoint,
# and it skips with a reason rather than being absent, because "no test" and
# "test that could not run here" are different claims about coverage.
#
# The recorded on-hardware result this reproduces lives in
# evidence/image-conditioning-qwen36-caption.json.
_MERGED_CAPTION_MODEL = (
    "/thearray/git/ob/text-generation-webui/models/"
    "Qwen3.6-35B-A3B-heretic-caption-step000745-merged-bf16"
)
_CAPTION_VALIDATION_SPLIT = (
    "/thearray/git/datasets/captiontest/qwen36-caption-finetune-v1/"
    "validation-fixed-100.jsonl"
)
# The merged caption model is ~70 GiB in bf16; the probe refuses to start below
# this so it cannot evict a resident training job.
_REQUIRED_FREE_VRAM_GIB = 78.0


@pytest.mark.gpu
@pytest.mark.slow
def test_real_checkpoint_consumes_image_content(tmp_path) -> None:
    """The pretrained checkpoint must pass, and a random ViT must fail.

    Asserting both halves is the point. A probe that only checked the model it
    was built for would pass just as happily if its thresholds were vacuous.
    """
    import pathlib
    import subprocess

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("image-conditioning probe requires a CUDA device")
    for asset in (_MERGED_CAPTION_MODEL, _CAPTION_VALIDATION_SPLIT):
        if not pathlib.Path(asset).exists():
            pytest.skip(f"probe asset is not present on this host: {asset}")
    free_gib = torch.cuda.mem_get_info(0)[0] / (1024**3)
    if free_gib < _REQUIRED_FREE_VRAM_GIB:
        pytest.skip(
            f"device 0 has {free_gib:.1f} GiB free, below the "
            f"{_REQUIRED_FREE_VRAM_GIB:.0f} GiB the probe needs; refusing to "
            "disturb resident work"
        )

    output = tmp_path / "conditioning.json"
    script = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "probe_image_conditioning.py"
    completed = subprocess.run(
        [
            sys.executable, str(script),
            "--model-dir", _MERGED_CAPTION_MODEL,
            "--dataset", _CAPTION_VALIDATION_SPLIT,
            "--output", str(output),
            "--examples", "5",
            "--max-pixels", "200704",
            "--minimum-free-vram-gib", str(_REQUIRED_FREE_VRAM_GIB),
        ],
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    evidence = json.loads(output.read_text())
    assert evidence["missing_keys"] == 0 and evidence["unexpected_keys"] == 0
    conditions = {entry["condition"]: entry for entry in evidence["conditions"]}
    assert conditions["pretrained-vision"]["consumes_image_content"] is True
    assert conditions["randomized-vision"]["consumes_image_content"] is False
    assert (
        conditions["pretrained-vision"]["discrimination_margin_nats"]
        > conditions["randomized-vision"]["discrimination_margin_nats"]
    )
