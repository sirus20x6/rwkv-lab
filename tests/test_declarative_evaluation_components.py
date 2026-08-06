from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rwkv_lab.training_components import (
    ArtifactRendererImplementation,
    AtomicCheckpointPolicy,
    EvaluationSchedule,
    FixedHeldOutSamples,
    ScalarLossEvaluator,
    artifact_renderer_from_resolved_component,
    checkpoint_policy_from_resolved_component,
    evaluation_schedule_from_resolved_component,
    evaluator_from_resolved_component,
    qualitative_sample_from_resolved_component,
)
from rwkv_lab.training_runtime.qualitative_samples import (
    DerivedFixedHeldOutSamples,
    FixedManifestSamples,
)

ROOT = Path(__file__).resolve().parents[1]
CATALOG = json.loads(
    (ROOT / "docs/experiment-vm/examples/training-components.v1.json").read_text()
)


def _descriptor(
    category: str, name: str, version: str = "1.0.0"
) -> dict[str, object]:
    return next(
        component
        for component in CATALOG["components"]
        if component["key"]["category"] == category and component["key"]["name"] == name
        and component["key"]["version"] == version
    )


def _envelope(
    category: str,
    name: str,
    configuration: dict[str, object],
    version: str = "1.0.0",
):
    descriptor = _descriptor(category, name, version)
    encoded = json.dumps(descriptor, separators=(",", ":"), sort_keys=True).encode()
    return {
        "descriptor": descriptor,
        "descriptor_digest": "sha256:" + hashlib.sha256(encoded).hexdigest(),
        "configuration": configuration,
    }


def _identity_digest(identities: tuple[str, ...]) -> str:
    encoded = json.dumps(identities, ensure_ascii=False, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def test_launch_gate_and_full_scalar_schedules_are_independent() -> None:
    schedule = evaluation_schedule_from_resolved_component(
        _envelope(
            "evaluation_schedule",
            "launch_gate_periodic",
            {
                "launch_gate_examples": 10,
                "full_step_zero": True,
                "qualitative_every_steps": 25,
                "full_every_steps": 100,
                "defer_full_scalar": True,
                "final": True,
            },
        )
    )
    assert isinstance(schedule, EvaluationSchedule)
    assert schedule.for_step(0).launch_gate
    assert schedule.for_step(0).qualitative
    assert schedule.for_step(0).full_scalar
    assert schedule.for_step(25).qualitative
    assert not schedule.for_step(25).full_scalar
    assert schedule.for_step(100).full_scalar
    assert schedule.for_step(7, final=True).qualitative
    assert schedule.for_step(7, final=True).full_scalar


def test_true_step_zero_baseline_cannot_be_disabled() -> None:
    with pytest.raises(ValueError, match="step-zero baseline"):
        evaluation_schedule_from_resolved_component(
            _envelope(
                "evaluation_schedule",
                "launch_gate_periodic",
                {
                    "launch_gate_examples": 10,
                    "full_step_zero": False,
                    "qualitative_every_steps": 0,
                    "full_every_steps": 0,
                    "defer_full_scalar": True,
                    "final": True,
                },
            )
        )


def test_fixed_held_out_identities_are_content_locked() -> None:
    identities = ("image-001", "image-017", "image-042")
    samples = qualitative_sample_from_resolved_component(
        _envelope(
            "qualitative_sample",
            "fixed_held_out",
            {
                "identity_field": "sample_id",
                "identities_digest": _identity_digest(identities),
                "selector_digest": "sha256:" + "a" * 64,
                "sample_count": 3,
            },
        )
    )
    assert isinstance(samples, FixedHeldOutSamples)
    assert samples.bind(identities, selector_digest="sha256:" + "a" * 64) == identities
    with pytest.raises(ValueError, match="split membership"):
        samples.bind(identities, selector_digest="sha256:" + "b" * 64)
    with pytest.raises(ValueError, match="identity digest"):
        samples.bind(tuple(reversed(identities)), selector_digest="sha256:" + "a" * 64)
    with pytest.raises(ValueError, match="unique"):
        samples.bind(
            ("image-001", "image-001", "image-042"),
            selector_digest="sha256:" + "a" * 64,
        )


def test_derived_fixed_held_out_receipts_authority_selected_identities() -> None:
    identities = ("image-001", "image-017", "image-042")
    samples = qualitative_sample_from_resolved_component(
        _envelope(
            "qualitative_sample",
            "fixed_held_out",
            {"identity_field": "sample_id", "sample_count": 3},
            "2.0.0",
        )
    )
    assert isinstance(samples, DerivedFixedHeldOutSamples)
    binding = samples.bind(identities, selector_digest="sha256:" + "a" * 64)
    assert binding.identities == identities
    assert binding.identities_digest == _identity_digest(identities)
    assert binding.selector_digest == "sha256:" + "a" * 64


def test_fixed_manifest_selects_ordered_membership_inside_dataset_root(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "balanced-validation.jsonl"
    encoded = b'{"id":"image-042"}\n{"id":"image-001"}\n{"id":"unused"}\n'
    manifest.write_bytes(encoded)
    samples = qualitative_sample_from_resolved_component(
        _envelope(
            "qualitative_sample",
            "fixed_manifest",
            {
                "identity_field": "id",
                "manifest_name": manifest.name,
                "manifest_sha256": "sha256:" + hashlib.sha256(encoded).hexdigest(),
                "sample_count": 2,
            },
        )
    )
    assert isinstance(samples, FixedManifestSamples)
    binding = samples.select(
        ("image-001", "image-017", "image-042"),
        selector_digest="sha256:" + "a" * 64,
        dataset_root=tmp_path,
    )
    assert binding.identities == ("image-042", "image-001")
    assert binding.identities_digest == _identity_digest(binding.identities)

    manifest.write_bytes(encoded + b'{"id":"changed"}\n')
    with pytest.raises(ValueError, match="content digest changed"):
        samples.select(
            ("image-001", "image-017", "image-042"),
            selector_digest="sha256:" + "a" * 64,
            dataset_root=tmp_path,
        )


def test_fixed_manifest_reads_from_pinned_nofollow_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import rwkv_lab.training_runtime.qualitative_samples as module

    manifest = tmp_path / "held-out.jsonl"
    original = b'{"id":"stable"}\n'
    manifest.write_bytes(original)
    samples = qualitative_sample_from_resolved_component(
        _envelope(
            "qualitative_sample",
            "fixed_manifest",
            {
                "identity_field": "id",
                "manifest_name": manifest.name,
                "manifest_sha256": "sha256:" + hashlib.sha256(original).hexdigest(),
                "sample_count": 1,
            },
        )
    )
    replacement = tmp_path / "replacement.jsonl"
    replacement.write_bytes(b'{"id":"attacker"}\n')
    real_fdopen = module.os.fdopen

    def replace_after_open(descriptor, *args, **kwargs):
        replacement.replace(manifest)
        return real_fdopen(descriptor, *args, **kwargs)

    monkeypatch.setattr(module.os, "fdopen", replace_after_open)
    binding = samples.select(
        ("stable", "attacker"),
        selector_digest="sha256:" + "a" * 64,
        dataset_root=tmp_path,
    )
    assert binding.identities == ("stable",)

    manifest.unlink()
    manifest.symlink_to(tmp_path / "replacement-target.jsonl")
    (tmp_path / "replacement-target.jsonl").write_bytes(original)
    monkeypatch.setattr(module.os, "fdopen", real_fdopen)
    with pytest.raises(ValueError, match="unavailable"):
        samples.select(
            ("stable",),
            selector_digest="sha256:" + "a" * 64,
            dataset_root=tmp_path,
        )


def test_caption_renderer_requires_aligned_target_baseline_current_triplet() -> None:
    renderer = artifact_renderer_from_resolved_component(
        _envelope(
            "artifact_renderer",
            "caption_triplet",
            {
                "modality": "multimodal",
                "schema": "trainvm.caption-triplet.v1",
            },
        )
    )
    assert renderer.implementation is ArtifactRendererImplementation.CAPTION_TRIPLET_V1
    evidence = renderer.render(
        sample_identity="image-017",
        step=25,
        evidence={
            "teacher_target": "A red fox in snow.",
            "baseline": "An animal outdoors.",
            "current": "A red fox standing in snow.",
        },
    )
    assert evidence["sample_identity"] == "image-017"
    assert evidence["step"] == 25
    assert evidence["evidence"]["teacher_target"] == "A red fox in snow."
    with pytest.raises(ValueError, match="target/baseline/current"):
        renderer.render(
            sample_identity="image-017",
            step=25,
            evidence={"baseline": "wrong sample", "current": "still wrong"},
        )
    with pytest.raises(ValueError, match="nonnegative step"):
        renderer.render(
            sample_identity="image-017",
            step=True,
            evidence={
                "teacher_target": "target",
                "baseline": "baseline",
                "current": "current",
            },
        )


def test_scalar_evaluator_and_atomic_checkpoint_policy_are_typed() -> None:
    evaluator = evaluator_from_resolved_component(
        _envelope(
            "evaluator",
            "scalar_loss",
            {
                "split_slot": "evaluation_split",
                "metrics": ("loss", "perplexity"),
                "reduction": "weighted_mean",
                "maximum_examples": 0,
            },
        )
    )
    assert isinstance(evaluator, ScalarLossEvaluator)
    assert evaluator.reduce((2.0, 4.0), (3.0, 1.0)) == pytest.approx(2.5)

    policy = checkpoint_policy_from_resolved_component(
        _envelope(
            "checkpoint_policy",
            "atomic_retained",
            {
                "every_steps": 100,
                "keep_last": 2,
                "keep_best": 1,
                "publish_final": True,
                "resume_grade": "exact",
            },
        )
    )
    assert isinstance(policy, AtomicCheckpointPolicy)
    assert policy.due(100)
    assert not policy.due(101)
    assert policy.due(101, final=True)
    assert policy.retained_steps((100, 200, 300, 400), (100, 300)) == (
        100,
        300,
        400,
    )
    with pytest.raises(ValueError, match="atomic publication"):
        policy.component_state(
            last_published_step=400,
            publication_manifest="sha256:" + "b" * 64,
            retention_manifest="sha256:" + "c" * 64,
            atomic_publication_complete=False,
        )
    assert (
        policy.component_state(
            last_published_step=400,
            publication_manifest="sha256:" + "b" * 64,
            retention_manifest="sha256:" + "c" * 64,
            atomic_publication_complete=True,
        )["last_published_step"]
        == 400
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("every_steps", 1_000_000_001),
        ("keep_last", 1_000_001),
        ("keep_best", 1_000_001),
    ),
)
def test_checkpoint_policy_runtime_enforces_catalog_upper_bounds(
    field: str, value: int
) -> None:
    configuration = {
        "every_steps": 100,
        "keep_last": 2,
        "keep_best": 1,
        "publish_final": True,
        "resume_grade": "exact",
    }
    configuration[field] = value
    with pytest.raises(ValueError, match="bounded integer"):
        checkpoint_policy_from_resolved_component(
            _envelope("checkpoint_policy", "atomic_retained", configuration)
        )
