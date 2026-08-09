"""Coverage for the HF multimodal SFT handler's controller-selected resume.

No test imported `_hf_multimodal_sft`, so the branch that turns a
controller-selected resume authority into `run_hf_multimodal_sft`'s lineage
arguments ran only in production. The other `_resume_payload` call sites that
*are* driven by `tests/test_trainvm_adapter_entrypoint.py` every one pass
`resume=None`, which short-circuits at `handlers.py:166` before a payload is
ever resolved, so nothing in the suite reached the populated branch from either
side. (`_qwen_ao3` is likewise imported by no test; that is filed separately.)

Everything here that production builds is built by production: the workspace's
content roots come from `measure_input_content_root`, the composition from
`load_resolved_training_composition` over the shipped example registry, the
checkpoint from `CheckpointPublisher`, and the invocation from
`load_worker_invocation` over a canonical document. Only the engine
(`run_hf_multimodal_sft`) is replaced, by a recorder that captures the call —
the engine has its own suite in `tests/test_hf_multimodal_sft_engine.py`, and
loading a real Hugging Face multimodal model is not available to a CPU-only
pytest run.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from rwkv_lab.trainvm_adapters import WorkerTrainingComponents
from rwkv_lab.trainvm_adapters.content_authority import measure_input_content_root
from rwkv_lab.trainvm_adapters.handlers import (
    AdapterDispatchError,
    _hf_multimodal_sft,
)
from rwkv_lab.trainvm_worker import (
    CheckpointPublisher,
    load_resolved_training_composition,
    load_worker_invocation,
)

from test_trainvm_checkpoint_publisher import FakeCheckpointSession, resume_invocation
from test_trainvm_worker_documents import invocation_document

# The eleven components `_hf_multimodal_sft` demands of a resume checkpoint.
# Restated here rather than imported so that widening the handler's requirement
# without widening what a publisher can actually produce fails in this file.
REQUIRED_STATE = (
    "component_composition",
    "control_revision",
    "data_cursor",
    "lr_schedule",
    "model",
    "optimizer",
    "rng_accelerator",
    "rng_numpy",
    "rng_python",
    "rng_torch",
    "weight_decay_schedule",
)

REGISTRY_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs/experiment-vm/examples/training-components.v1.json"
)
RECIPE_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs/experiment-vm/examples/hf-multimodal-sft.recipe-profiles.v1.json"
)


def canonical(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def file_digest(path: Path) -> str:
    return digest(path.read_bytes())


def _frozen_dataset(root: Path) -> None:
    """Write the frozen image-split dataset `manifested_jsonl_image_splits` needs.

    The receipt shape is the one `RegisteredDataSource.verify_content` enforces,
    so this is a real dataset as far as the production pre-load validation is
    concerned, not a stub that validation is taught to accept.
    """

    images = root / "images"
    images.mkdir(parents=True)
    counts = {"train": 2, "validation": 1, "test": 1}
    files: dict[str, dict[str, object]] = {}
    for split, count in counts.items():
        rows = []
        for index in range(count):
            image = images / f"{split}-{index}.png"
            image.write_bytes(b"\x89PNG\r\n\x1a\n" + split.encode() + bytes([index]))
            rows.append(
                {
                    "caption": f"a {split} caption {index}",
                    "id": f"{split}-{index}",
                    "image": str(image),
                    "split": split,
                }
            )
        path = root / f"{split}.jsonl"
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        files[path.name] = {
            "rows": count,
            "sha256": file_digest(path).removeprefix("sha256:"),
        }
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "rwkv-lab.frozen-image-splits.v1",
                "dataset_digest": digest(b"dataset"),
                "counts": counts,
                "files": files,
                "unique_content_hashes": sum(counts.values()),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _model_snapshot(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "config.json").write_text('{"model_type":"qwen"}', encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"weights")


def _composition(*, model_path: Path, model_fingerprint: str,
                 dataset_root: Path, dataset_fingerprint: str):
    """Resolve the shipped HF multimodal SFT recipe against the shipped registry.

    The slot set, the component keys and every configuration value come from
    `hf-multimodal-sft.recipe-profiles.v1.json` — the document the controller
    authors real runs from. Only the two paths and their two content
    fingerprints are rewritten to point at this test's tree, plus the
    trainability slot: the recipe's `lora_target_manifest` preflights selectors
    against a real Hugging Face snapshot, which a CPU-only run has no way to
    provide. `full` is a registry-valid trainability for this family and leaves
    the resume branch identical.
    """

    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    recipe = json.loads(RECIPE_PATH.read_text(encoding="utf-8"))
    requested = dict(
        recipe["recipes"][0]["template_document"]["spec"]["workflow"]["nodes"][
            "train"
        ]["invoke"]["training"]["components"]
    )
    requested["model_loader"] = {
        "key": requested["model_loader"]["key"],
        "configuration": {
            **requested["model_loader"]["configuration"],
            "model_path": str(model_path),
            "checkpoint_fingerprint": model_fingerprint,
        },
    }
    requested["data"] = {
        "key": requested["data"]["key"],
        "configuration": {
            **requested["data"]["configuration"],
            "dataset_root": str(dataset_root),
            "content_fingerprint": dataset_fingerprint,
        },
    }
    requested["trainability"] = {
        "key": {"category": "trainability", "name": "full", "version": "1.0.0"},
        "configuration": {},
    }
    components = {}
    for slot, value in requested.items():
        key = value["key"]
        descriptor = next(
            item
            for item in registry["components"]
            if item["key"] == {
                "category": key["category"],
                "name": key["name"],
                "version": key["version"],
            }
        )
        components[slot] = {
            "configuration": value["configuration"],
            "descriptor": descriptor,
            "descriptor_digest": digest(canonical(descriptor)),
        }
    body = {
        "api_version": "trainvm.resolved-training-composition/v1",
        "components": components,
        "model_family": "transformer",
        "registry_digest": digest(canonical(registry)),
    }
    return load_resolved_training_composition(
        {**body, "composition_digest": digest(canonical(body))}
    )


def _harness(tmp_path: Path):
    read_root = tmp_path / "read"
    run_directory = tmp_path / "write" / "run"
    run_directory.mkdir(parents=True)
    model_path = read_root / "model"
    dataset_root = read_root / "dataset"
    _model_snapshot(model_path)
    _frozen_dataset(dataset_root)
    model_identity = measure_input_content_root(model_path)
    dataset_identity = measure_input_content_root(dataset_root)
    workspace = {
        "run_directory": str(run_directory),
        "allowed_read_roots": [str(read_root)],
        # verify_input_content_roots requires the roots strictly path-sorted,
        # exactly as the controller emits them.
        "input_content_roots": [
            asdict(identity)
            for identity in sorted(
                (model_identity, dataset_identity), key=lambda item: item.path
            )
        ],
        "allowed_write_roots": [str(run_directory)],
    }
    composition = _composition(
        model_path=model_path,
        model_fingerprint=model_identity.tree_sha256,
        dataset_root=dataset_root,
        dataset_fingerprint=dataset_identity.tree_sha256,
    )
    return SimpleNamespace(
        read_root=read_root,
        run_directory=run_directory,
        workspace=workspace,
        components=WorkerTrainingComponents(composition, "transformer"),
    )


def _published_resume(harness, *, state_components=REQUIRED_STATE):
    session = FakeCheckpointSession(harness.run_directory, workspace=harness.workspace)
    checkpoint = harness.run_directory / "checkpoint-00000012"
    checkpoint.mkdir()
    (checkpoint / "trainer-state.pt").write_bytes(b"optimizer+rng")
    result = CheckpointPublisher(session).publish(
        checkpoint,
        optimizer_step=12,
        resume_grade="compatible",
        state_components=tuple(sorted(state_components)),
        parent_artifact_ids=("base-model-1",),
    )
    return session, result


class _Engine:
    """Stand in for `run_hf_multimodal_sft` and record what the handler passed."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, **arguments: object) -> int:
        self.calls.append(arguments)
        return 12


@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch) -> _Engine:
    from rwkv_lab.trainvm_adapters import hf_multimodal_sft

    recorder = _Engine()
    monkeypatch.setattr(hf_multimodal_sft, "run_hf_multimodal_sft", recorder)
    return recorder


def _runtime():
    return SimpleNamespace(), SimpleNamespace()


def test_hf_sft_handler_hands_the_resolved_lineage_to_the_engine(
    tmp_path: Path, engine: _Engine
) -> None:
    harness = _harness(tmp_path)
    session, published = _published_resume(harness)
    invocation = resume_invocation(session, published, inputs={})

    observability, controls = _runtime()
    result = _hf_multimodal_sft(
        invocation,
        harness.components,
        observability=observability,
        controls=controls,
    )

    (call,) = engine.calls
    assert call["resume_parent_artifact_ids"] == (published.artifact_id,)
    assert call["resume_checkpoint_manifest_digest"] == published.manifest_sha256
    assert call["resume_directory"] == published.manifest_path.parent / "payload"
    # A plausible-looking path is not enough: the directory handed to the
    # engine has to be the promoted payload, with the published bytes in it.
    assert (
        call["resume_directory"] / "trainer-state.pt"
    ).read_bytes() == b"optimizer+rng"
    assert call["run_directory"] == harness.run_directory
    assert result.event_type == "worker.completed"
    assert result.optimizer_step == 12


def test_hf_sft_handler_refuses_a_resume_missing_a_required_state_component(
    tmp_path: Path, engine: _Engine
) -> None:
    harness = _harness(tmp_path)
    session, published = _published_resume(
        harness,
        state_components=tuple(
            component for component in REQUIRED_STATE if component != "lr_schedule"
        ),
    )
    invocation = resume_invocation(session, published, inputs={})

    observability, controls = _runtime()
    with pytest.raises(AdapterDispatchError, match="omits required state"):
        _hf_multimodal_sft(
            invocation,
            harness.components,
            observability=observability,
            controls=controls,
        )
    assert engine.calls == []


def test_hf_sft_handler_passes_empty_lineage_without_a_resume(
    tmp_path: Path, engine: _Engine
) -> None:
    harness = _harness(tmp_path)
    invocation = load_worker_invocation(
        invocation_document(workspace=harness.workspace, inputs={})
    )

    observability, controls = _runtime()
    _hf_multimodal_sft(
        invocation,
        harness.components,
        observability=observability,
        controls=controls,
    )

    (call,) = engine.calls
    assert call["resume_directory"] is None
    assert call["resume_parent_artifact_ids"] == ()
    assert call["resume_checkpoint_manifest_digest"] is None
