from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import jsonschema
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _module(name: str, relative: str):
    specification = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


MATERIALIZER = _module(
    "trainvm_production_materializer",
    "scripts/materialize_trainvm_production_qualification.py",
)
RUNNER = _module(
    "trainvm_production_runner_for_materializer",
    "scripts/run_trainvm_production_qualification.py",
)


def _inputs(tmp_path: Path) -> dict:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    mage_model = tmp_path / "mage-model"
    mage_model.mkdir()
    images = tmp_path / "images"
    images.mkdir()
    train = tmp_path / "train.jsonl"
    train.write_text("{}\n", encoding="utf-8")
    evaluate = tmp_path / "eval.jsonl"
    evaluate.write_text("{}\n", encoding="utf-8")
    rwkv_model = tmp_path / "rwkv.pth"
    rwkv_model.write_bytes(b"rwkv")
    rwkv_tokens = tmp_path / "rwkv-tokens.bin"
    rwkv_tokens.write_bytes(b"\0" * 4096)
    transformer_model = tmp_path / "transformer-model"
    transformer_model.mkdir()
    transformer_patch = tmp_path / "transformer-patch"
    transformer_patch.mkdir()
    (transformer_patch / "manifest.json").write_text(
        json.dumps({"source_checkpoint": str(transformer_model)}),
        encoding="utf-8",
    )
    transformer_tokens = tmp_path / "transformer-tokens.bin"
    transformer_tokens.write_bytes(b"\0" * 16384)
    return {
        "api_version": MATERIALIZER.API_VERSION,
        "workspace_root": str(workspace),
        "run_root": str(tmp_path / "runs"),
        "concurrency_key": "production-qualification",
        "resources": {
            "minimum_accelerator_memory_gib": 24,
            "minimum_host_memory_gib": 32,
            "cpu_threads": 4,
            "omp_threads": 2,
            "preprocessing_workers": 1,
        },
        "mageflow": {
            "model_path": str(mage_model),
            "train_manifest": str(train),
            "eval_manifest": str(evaluate),
            "image_roots": [str(images)],
        },
        "rwkv": {
            "model_path": str(rwkv_model),
            "data_path": str(rwkv_tokens),
        },
        "transformer": {
            "model_dir": str(transformer_model),
            "patch_dir": str(transformer_patch),
            "tokens_bin": str(transformer_tokens),
            "total_tokens_in_bin": 4096,
        },
    }


def test_materialized_documents_are_schema_valid_and_runner_qualified(tmp_path) -> None:
    destination = tmp_path / "qualification"
    manifest_path = MATERIALIZER.materialize(_inputs(tmp_path), destination)
    schema = json.loads(
        (ROOT / "docs/experiment-vm/experiment-v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["api_version"] == MATERIALIZER.MATERIALIZATION_VERSION
    assert manifest["pause_steps"] == {
        "mageflow": 2,
        "rwkv": 2,
        "transformer": 2,
    }
    for family in MATERIALIZER.FAMILIES:
        document_path = destination / manifest["files"][family]["document"]
        document = json.loads(document_path.read_text(encoding="utf-8"))
        jsonschema.validate(document, schema)
        loaded, source = RUNNER.load_qualification_document(family, document_path)
        assert loaded == document
        assert json.loads(source) == document
        assert document["spec"]["execution"]["compile"] == {"enabled": False}
        assert document["spec"]["recovery"]["reconcile"] == "restart_from_checkpoint"
        assert document["spec"]["recovery"]["release_accelerators_when_paused"] is True


def test_materializer_emits_exact_content_root_sets_and_no_launch_authority(tmp_path) -> None:
    inputs = _inputs(tmp_path)
    destination = tmp_path / "qualification"
    MATERIALIZER.materialize(inputs, destination)

    expected = {
        "mageflow": {
            inputs["mageflow"]["model_path"],
            inputs["mageflow"]["train_manifest"],
            inputs["mageflow"]["eval_manifest"],
            inputs["mageflow"]["image_roots"][0],
        },
        "rwkv": {
            inputs["rwkv"]["model_path"],
            inputs["rwkv"]["data_path"],
        },
        "transformer": {
            inputs["transformer"]["model_dir"],
            inputs["transformer"]["patch_dir"],
            inputs["transformer"]["tokens_bin"],
        },
    }
    forbidden = {"argv", "command", "environment", "executable", "module"}
    for family in MATERIALIZER.FAMILIES:
        roots = json.loads(
            (destination / f"{family}.input-roots.json").read_text(encoding="utf-8")
        )
        assert roots == {
            "api_version": "trainvm.input-content-root-set/v1",
            "paths": sorted(expected[family]),
        }
        document = json.loads(
            (destination / f"{family}.json").read_text(encoding="utf-8")
        )
        stack = [document]
        observed_keys = set()
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                observed_keys.update(value)
                stack.extend(value.values())
            elif isinstance(value, list):
                stack.extend(value)
        assert forbidden.isdisjoint(observed_keys)


def test_generated_inline_configs_match_closed_adapter_contracts(tmp_path) -> None:
    from rwkv_lab.mage_flow_pretrain import MageFlowTrainConfig
    from rwkv_lab.trainvm_adapters.transformer_mla import TransformerMLATrainConfig

    documents = MATERIALIZER.build_documents(_inputs(tmp_path))
    mage = documents["mageflow"][0]
    transformer = documents["transformer"][0]
    mage_config = MageFlowTrainConfig(
        **mage["spec"]["workflow"]["nodes"]["train"]["invoke"]["inputs"]["config"]["literal"]
    )
    transformer_config = TransformerMLATrainConfig(
        **transformer["spec"]["workflow"]["nodes"]["train"]["invoke"]["inputs"]["config"]["literal"]
    )

    mage_config.validate()
    assert transformer_config.adapter == "rwkv-lab.transformer-mla"
    assert mage_config.max_steps == transformer_config.max_steps == 8
    assert mage_config.checkpoint_every == transformer_config.save_every_steps == 1


def test_input_schema_accepts_the_materializer_contract(tmp_path) -> None:
    schema = json.loads(
        (
            ROOT
            / "docs/experiment-vm/production-qualification-inputs-v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.validators.validator_for(schema).check_schema(schema)
    jsonschema.validate(_inputs(tmp_path), schema)


def test_materialization_is_deterministic_and_refuses_overwrite(tmp_path) -> None:
    inputs = _inputs(tmp_path)
    left = tmp_path / "left"
    right = tmp_path / "right"
    MATERIALIZER.materialize(inputs, left)
    MATERIALIZER.materialize(inputs, right)
    assert {
        path.name: path.read_bytes() for path in left.iterdir()
    } == {
        path.name: path.read_bytes() for path in right.iterdir()
    }
    with pytest.raises(MATERIALIZER.MaterializationError, match="already exists"):
        MATERIALIZER.materialize(inputs, left)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(extra=True), "unknown fields"),
        (
            lambda value: value["transformer"].update(model_dir=value["transformer"]["tokens_bin"]),
            "must be a directory",
        ),
        (lambda value: value["mageflow"].update(image_roots=[]), "nonempty bounded"),
        (lambda value: value["rwkv"].update(max_steps=2), "integer in"),
    ],
)
def test_materializer_fails_closed_on_invalid_deployment_inputs(
    tmp_path, mutation, message
) -> None:
    inputs = _inputs(tmp_path)
    mutation(inputs)
    with pytest.raises(MATERIALIZER.MaterializationError, match=message):
        MATERIALIZER.build_documents(inputs)


def test_materializer_rejects_transformer_patch_from_a_different_base(
    tmp_path,
) -> None:
    inputs = _inputs(tmp_path)
    different_base = tmp_path / "different-transformer-base"
    different_base.mkdir()
    patch_manifest = Path(inputs["transformer"]["patch_dir"]) / "manifest.json"
    patch_manifest.write_text(
        json.dumps({"source_checkpoint": str(different_base)}),
        encoding="utf-8",
    )

    with pytest.raises(
        MATERIALIZER.MaterializationError,
        match="source checkpoint disagrees",
    ):
        MATERIALIZER.build_documents(inputs)


@pytest.mark.parametrize(
    ("token_bytes", "total_tokens", "message"),
    [
        (4097, 1024, "packed uint32"),
        (4096, 1025, "packed uint32"),
        (16384, 300, "partitions must each exceed"),
    ],
)
def test_materializer_rejects_unusable_transformer_token_partitions(
    tmp_path, token_bytes, total_tokens, message
) -> None:
    inputs = _inputs(tmp_path)
    Path(inputs["transformer"]["tokens_bin"]).write_bytes(b"\0" * token_bytes)
    inputs["transformer"]["total_tokens_in_bin"] = total_tokens

    with pytest.raises(MATERIALIZER.MaterializationError, match=message):
        MATERIALIZER.build_documents(inputs)


def test_loader_rejects_duplicate_and_nonfinite_json(tmp_path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"api_version":"x","api_version":"y"}', encoding="utf-8")
    with pytest.raises(MATERIALIZER.MaterializationError, match="duplicate JSON field"):
        MATERIALIZER.load_inputs(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value":NaN}', encoding="utf-8")
    with pytest.raises(MATERIALIZER.MaterializationError, match="non-finite"):
        MATERIALIZER.load_inputs(nonfinite)
