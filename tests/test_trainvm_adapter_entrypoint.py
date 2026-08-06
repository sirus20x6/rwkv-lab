from __future__ import annotations

import json
from dataclasses import asdict
from types import MappingProxyType, SimpleNamespace
from typing import Self

import pytest

from rwkv_lab.trainvm_adapters.content_authority import measure_input_content_root
from rwkv_lab.trainvm_adapters.entrypoint import (
    WORKER_BOOTSTRAP_DESCRIPTOR,
    WorkerEntrypointError,
    main,
    run_worker,
)
from rwkv_lab.trainvm_adapters.handlers import (
    AdapterDispatchError,
    HandlerResult,
    _appearance_expert,
    _mageflow_full_backbone,
    _rlvr,
    _rwkv_posttraining,
    _rwkv_scratch,
    _scalar_metric_decision,
    _terminal_expert,
    _transformer_mla,
    _vision_frozen_adapter,
    _vision_native_head,
    _vision_rwkv_student,
    _vision_teacher_compressor,
    execute_invocation,
    supported_adapter_keys,
)
from rwkv_lab.trainvm_adapters.io import (
    AdapterInputError,
    WorkspacePathAuthority,
    read_inline_config,
    require_run_directory,
)
from rwkv_lab.trainvm_worker import (
    ExecutionPhase,
    ExecutionPhaseRequest,
    WorkerCancellationRequested,
    WorkerExecutionPhases,
    WorkerResourcesReleasedPause,
)


class FakeSession:
    def __init__(self, bootstrap: object, *, completed: bool = False) -> None:
        self.bootstrap = bootstrap
        self.completed_before_connect = completed
        self.invocation = SimpleNamespace(
            run_id="run-1",
            controls={},
            effective_control_revision=0,
            observability={
                "heartbeat_seconds": 5,
                "metrics": [],
                "retain_raw_metrics_days": 1,
            },
        )
        self.finished: list[tuple[str, object, int | None]] = []
        self.heartbeats: list[tuple[int, str]] = []
        self.execution_phase_requests = ()
        self.closed = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.closed = True

    def finish(
        self, event_type: str, payload: object, *, optimizer_step: int | None = None
    ) -> None:
        self.finished.append((event_type, payload, optimizer_step))

    def heartbeat(self, optimizer_step: int, phase: str, *, wait: bool = False) -> int:
        assert wait is False
        self.heartbeats.append((optimizer_step, phase))
        return len(self.heartbeats)


def _sealed_workspace(read_root, run_directory):
    return {
        "run_directory": str(run_directory),
        "allowed_read_roots": [str(read_root)],
        "input_content_roots": [asdict(measure_input_content_root(read_root))],
        "allowed_write_roots": [str(run_directory.parent)],
    }


def test_inline_config_is_frozen_content_not_a_mutable_path(tmp_path) -> None:
    frozen = MappingProxyType(
        {"config": MappingProxyType({"steps": 12, "buckets": (512, 768)})}
    )
    assert read_inline_config(frozen) == {"steps": 12, "buckets": [512, 768]}
    with pytest.raises(AdapterInputError, match="inline object"):
        read_inline_config({"config": str(tmp_path / "mutable.json")})
    with pytest.raises(AdapterInputError, match="exactly one"):
        read_inline_config({"config": {}, "argv": ["--import", "anything"]})


def test_run_directory_must_equal_the_authority_workspace(tmp_path) -> None:
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    workspace = {
        "run_directory": str(run_directory),
        "allowed_read_roots": [str(tmp_path)],
        "allowed_write_roots": [str(tmp_path)],
    }
    assert require_run_directory(str(run_directory), workspace) == run_directory
    with pytest.raises(AdapterInputError, match="workspace authority"):
        require_run_directory(str(tmp_path / "other"), workspace)
    with pytest.raises(AdapterInputError, match="absolute normalized"):
        require_run_directory("relative/run", workspace)


def test_workspace_path_authority_confines_reads_writes_and_symlinks(tmp_path) -> None:
    read_root = tmp_path / "read"
    write_root = tmp_path / "write"
    outside = tmp_path / "outside"
    for directory in (read_root, write_root, outside):
        directory.mkdir()
    source = read_root / "manifest.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    (read_root / "escape").symlink_to(outside, target_is_directory=True)
    authority = WorkspacePathAuthority.from_workspace(
        {
            "run_directory": str(write_root / "run"),
            "allowed_read_roots": [str(read_root)],
            "allowed_write_roots": [str(write_root)],
        }
    )

    assert authority.read_path(str(source), label="manifest", kind="file") == source
    assert (
        authority.write_directory(str(write_root / "cache" / "new"), label="cache")
        == write_root / "cache" / "new"
    )
    assert authority.node_run_directory("moonvit_arm") == (
        write_root / "run" / "nodes" / "moonvit_arm"
    )
    with pytest.raises(AdapterInputError, match="node ID"):
        authority.node_run_directory("../escape")
    with pytest.raises(AdapterInputError, match="node ID"):
        authority.node_run_directory("мoonvit_arm")
    with pytest.raises(AdapterInputError, match="outside declared read roots"):
        authority.read_path(
            str(read_root / "escape"), label="dataset", kind="directory"
        )
    with pytest.raises(AdapterInputError, match="outside declared write roots"):
        authority.write_directory(str(outside / "cache"), label="cache")


def test_workspace_reads_require_declared_content_coverage(tmp_path) -> None:
    read_root = tmp_path / "read"
    write_root = tmp_path / "write"
    read_root.mkdir()
    write_root.mkdir()
    sealed = read_root / "sealed.bin"
    uncovered = read_root / "uncovered.bin"
    sealed.write_bytes(b"sealed")
    uncovered.write_bytes(b"uncovered")
    authority = WorkspacePathAuthority.from_workspace(
        {
            "run_directory": str(write_root / "run"),
            "allowed_read_roots": [str(read_root)],
            "input_content_roots": (
                MappingProxyType(asdict(measure_input_content_root(sealed))),
            ),
            "allowed_write_roots": [str(write_root)],
        },
        require_content=True,
    )
    assert authority.read_path(str(sealed), label="sealed", kind="file") == sealed
    with pytest.raises(AdapterInputError, match="not covered"):
        authority.read_path(str(uncovered), label="uncovered", kind="file")


def test_packed_manifest_cannot_escape_its_verified_directory(tmp_path) -> None:
    read_root = tmp_path / "read"
    pack = read_root / "pack"
    write_root = tmp_path / "write"
    pack.mkdir(parents=True)
    write_root.mkdir()
    (read_root / "outside.bin").write_bytes(b"packed")
    (pack / "manifest.json").write_text(
        json.dumps({"packed_file": "../outside.bin"}), encoding="utf-8"
    )
    authority = WorkspacePathAuthority.from_workspace(
        {
            "run_directory": str(write_root / "run"),
            "allowed_read_roots": [str(read_root)],
            "input_content_roots": [asdict(measure_input_content_root(read_root))],
            "allowed_write_roots": [str(write_root)],
        },
        require_content=True,
    )
    with pytest.raises(AdapterInputError, match="escapes"):
        authority.verify_json_relative_file_reference(
            pack,
            manifest_name="manifest.json",
            field="packed_file",
            label="train pack",
        )


def test_jsonl_reference_scan_rejects_an_oversized_line(tmp_path) -> None:
    read_root = tmp_path / "read"
    write_root = tmp_path / "write"
    read_root.mkdir()
    write_root.mkdir()
    manifest = read_root / "oversized.jsonl"
    manifest.write_bytes(b"x" * (4 * 1024 * 1024 + 1))
    authority = WorkspacePathAuthority.from_workspace(
        {
            "run_directory": str(write_root / "run"),
            "allowed_read_roots": [str(read_root)],
            "input_content_roots": [asdict(measure_input_content_root(read_root))],
            "allowed_write_roots": [str(write_root)],
        },
        require_content=True,
    )

    with pytest.raises(AdapterInputError, match="line 1 exceeds its size bound"):
        authority.verify_jsonl_file_references(
            manifest,
            fields=("image", "image_path"),
            label="training manifest",
        )


def test_appearance_handler_passes_only_canonical_authorized_paths(
    tmp_path, monkeypatch
) -> None:
    from rwkv_lab import mage_flow_expert_train

    read_root = tmp_path / "read"
    run_directory = tmp_path / "write" / "run"
    read_root.mkdir()
    run_directory.mkdir(parents=True)
    manifest = read_root / "train.jsonl"
    image = read_root / "image.png"
    image.write_bytes(b"image")
    manifest.write_text(json.dumps({"image": str(image)}) + "\n", encoding="utf-8")
    observed = []
    monkeypatch.setattr(
        mage_flow_expert_train,
        "train",
        lambda config, *, worker_components, worker_step_profiler, worker_observability, worker_controls, worker_execution_phases: (
            observed.append(
                (
                    config,
                    worker_components,
                    worker_step_profiler,
                    worker_observability,
                    worker_controls,
                    worker_execution_phases,
                )
            )
        ),
    )
    invocation = SimpleNamespace(
        inputs={
            "config": {
                "train_manifest": str(manifest),
                "output_dir": str(run_directory),
            }
        },
        workspace=_sealed_workspace(read_root, run_directory),
    )
    components = SimpleNamespace()
    observability = SimpleNamespace(
        keepalive=lambda *_args: __import__("contextlib").nullcontext()
    )
    controls = SimpleNamespace(
        effective_values={
            "learning_rate": 2.0e-5,
            "eval_every": 25,
            "caption_dropout": 0.2,
            "mixed_precision": "bf16",
        }
    )
    execution_phases = SimpleNamespace(
        request=lambda phase: (
            SimpleNamespace(enabled=True)
            if phase is ExecutionPhase.COMPILE
            else None
        )
    )

    assert _appearance_expert(
        invocation,
        components,
        observability=observability,
        controls=controls,
        execution_phases=execution_phases,
    ) == HandlerResult("worker.completed", {"reason": "training_complete"})
    assert observed[0][0].train_manifest == str(manifest.resolve())
    assert observed[0][0].output_dir == str(run_directory.resolve())
    assert observed[0][1] is components
    assert observed[0][3] is observability
    assert observed[0][4] is controls
    assert observed[0][5] is execution_phases
    assert observed[0][0].compile_transformer_blocks is True
    assert observed[0][0].compile_vae_encoder is True
    assert observed[0][0].learning_rate == pytest.approx(2.0e-5)
    assert observed[0][0].eval_every == 25
    assert observed[0][0].caption_dropout == pytest.approx(0.2)


def test_family_handler_rejects_same_path_mutation_before_trainer_call(
    tmp_path, monkeypatch
) -> None:
    from rwkv_lab import mage_flow_expert_train

    read_root = tmp_path / "read"
    run_directory = tmp_path / "write" / "run"
    read_root.mkdir()
    run_directory.mkdir(parents=True)
    manifest = read_root / "train.jsonl"
    image = read_root / "image.png"
    image.write_bytes(b"image")
    manifest.write_text(json.dumps({"image": str(image)}) + "\n", encoding="utf-8")
    workspace = _sealed_workspace(read_root, run_directory)
    manifest.write_text('{"changed":true}\n', encoding="utf-8")
    called = False

    def train(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(mage_flow_expert_train, "train", train)
    invocation = SimpleNamespace(
        inputs={
            "config": {
                "train_manifest": str(manifest),
                "output_dir": str(run_directory),
            }
        },
        workspace=workspace,
    )
    with pytest.raises(AdapterInputError, match="content identity verification"):
        _appearance_expert(invocation, SimpleNamespace())
    assert called is False


def test_terminal_handler_lowers_compile_phase_into_trainer_config(
    tmp_path, monkeypatch
) -> None:
    from rwkv_lab import mage_flow_terminal_train

    read_root = tmp_path / "read"
    run_directory = tmp_path / "write" / "run"
    read_root.mkdir()
    run_directory.mkdir(parents=True)
    image = read_root / "image.png"
    image.write_bytes(b"image")
    manifest = read_root / "train.jsonl"
    manifest.write_text(json.dumps({"image": str(image)}) + "\n", encoding="utf-8")
    expert = read_root / "expert.safetensors"
    expert.write_bytes(b"expert")
    observed = []

    def train(config, **kwargs):
        observed.append((config, kwargs))

    monkeypatch.setattr(mage_flow_terminal_train, "train", train)
    invocation = SimpleNamespace(
        inputs={
            "config": {
                "domain": "animation",
                "train_manifest": str(manifest),
                "expert_checkpoint": str(expert),
                "output_dir": str(run_directory),
                "model_path": None,
            }
        },
        workspace=_sealed_workspace(read_root, run_directory),
    )
    execution_phases = SimpleNamespace(
        request=lambda phase: (
            SimpleNamespace(enabled=True)
            if phase is ExecutionPhase.COMPILE
            else None
        )
    )

    result = _terminal_expert(
        invocation,
        SimpleNamespace(),
        observability=SimpleNamespace(),
        execution_phases=execution_phases,
    )
    assert result == HandlerResult(
        "worker.completed", {"reason": "training_complete"}
    )
    config, keyword_arguments = observed[0]
    assert config.compile_transformer_blocks is True
    assert config.compile_vae_encoder is True
    assert keyword_arguments["worker_execution_phases"] is execution_phases


def test_full_backbone_handler_seals_model_tree_and_lowers_phases(
    tmp_path, monkeypatch
) -> None:
    from rwkv_lab import mage_flow_pretrain

    read_root = tmp_path / "read"
    run_directory = tmp_path / "write" / "run"
    model_path = read_root / "model"
    model_path.mkdir(parents=True)
    (model_path / "model_index.json").write_text("{}\n", encoding="utf-8")
    run_directory.mkdir(parents=True)
    image = read_root / "image.png"
    image.write_bytes(b"image")
    manifest = read_root / "train.jsonl"
    manifest.write_text(json.dumps({"image": str(image)}) + "\n", encoding="utf-8")
    observed = []
    monkeypatch.setattr(
        mage_flow_pretrain,
        "train",
        lambda config, **kwargs: observed.append((config, kwargs)),
    )
    invocation = SimpleNamespace(
        inputs={
            "config": {
                "train_manifest": str(manifest),
                "model_path": str(model_path),
                "output_dir": str(run_directory),
            }
        },
        workspace=_sealed_workspace(read_root, run_directory),
    )
    phases = SimpleNamespace(
        request=lambda phase: (
            SimpleNamespace(enabled=True)
            if phase is ExecutionPhase.COMPILE
            else None
        )
    )
    components = SimpleNamespace()
    result = _mageflow_full_backbone(
        invocation,
        components,
        observability=SimpleNamespace(),
        controls=SimpleNamespace(effective_values={}),
        execution_phases=phases,
    )
    assert result == HandlerResult(
        "worker.completed", {"reason": "training_complete"}
    )
    config, kwargs = observed[0]
    assert config.train_manifest == str(manifest.resolve())
    assert config.model_path == str(model_path.resolve())
    assert config.output_dir == str(run_directory.resolve())
    assert config.compile_transformer_blocks is True
    assert config.compile_vae_encoder is True
    assert kwargs["worker_components"] is components
    assert kwargs["worker_execution_phases"] is phases


def test_mageflow_manifest_cannot_reference_unsealed_payload(
    tmp_path, monkeypatch
) -> None:
    from rwkv_lab import mage_flow_expert_train

    read_root = tmp_path / "read"
    run_directory = tmp_path / "write" / "run"
    read_root.mkdir()
    run_directory.mkdir(parents=True)
    image = read_root / "outside-root.png"
    image.write_bytes(b"image")
    manifest = read_root / "train.jsonl"
    manifest.write_text(json.dumps({"image": str(image)}) + "\n", encoding="utf-8")
    called = False

    def train(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(mage_flow_expert_train, "train", train)
    invocation = SimpleNamespace(
        inputs={
            "config": {
                "train_manifest": str(manifest),
                "output_dir": str(run_directory),
            }
        },
        workspace={
            "run_directory": str(run_directory),
            "allowed_read_roots": [str(read_root)],
            "input_content_roots": [asdict(measure_input_content_root(manifest))],
            "allowed_write_roots": [str(run_directory.parent)],
        },
    )
    with pytest.raises(AdapterInputError, match="not covered"):
        _appearance_expert(invocation, SimpleNamespace())
    assert called is False


def test_appearance_handler_returns_declared_immutable_checkpoint_request(
    tmp_path, monkeypatch
) -> None:
    from rwkv_lab import mage_flow_expert_train

    read_root = tmp_path / "read"
    run_directory = tmp_path / "write" / "run"
    read_root.mkdir()
    run_directory.mkdir(parents=True)
    manifest = read_root / "train.jsonl"
    image = read_root / "image.png"
    image.write_bytes(b"image")
    manifest.write_text(json.dumps({"image": str(image)}) + "\n", encoding="utf-8")

    def train(config, **_kwargs):
        checkpoint = run_directory / "checkpoint-00000023"
        checkpoint.mkdir()
        (checkpoint / "state.pt").write_bytes(b"state")
        (run_directory / "complete.json").write_text(
            json.dumps(
                {
                    "global_step": 23,
                    "checkpoint": str(checkpoint),
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(mage_flow_expert_train, "train", train)
    invocation = SimpleNamespace(
        inputs={
            "config": {
                "train_manifest": str(manifest),
                "output_dir": str(run_directory),
            }
        },
        workspace=_sealed_workspace(read_root, run_directory),
        publishes={"checkpoint": {}},
    )

    result = _appearance_expert(
        invocation, SimpleNamespace(), observability=SimpleNamespace()
    )
    assert result.optimizer_step == 23
    assert result.payload == {"reason": "training_complete"}
    assert len(result.checkpoint_requests) == 1
    request = result.checkpoint_requests[0]
    assert request.source_directory == run_directory / "checkpoint-00000023"
    assert request.optimizer_step == 23
    assert request.resume_grade == "compatible"
    assert request.state_components[0] == "component_composition"


def test_rwkv_scratch_handler_lowers_only_typed_arguments_and_terminal_checkpoint(
    tmp_path, monkeypatch
) -> None:
    import hashlib
    from pathlib import Path

    from rwkv_lab import rwkv_pretrain
    from rwkv_lab.training_components import (
        CausalTokensMapperConfiguration,
        CollatorImplementation,
        DataSourceImplementation,
        DeclarativeDataPipeline,
        DerivedFixedHeldOutConfiguration,
        DerivedFixedHeldOutSamples,
        DeterministicSamplerConfiguration,
        FixedBatchingConfiguration,
        FixedGradientAccumulation,
        FixedGradientAccumulationConfiguration,
        FrozenNamedSplitConfiguration,
        FullTrainabilityConfiguration,
        JsonlFrozenTokenSplitsConfiguration,
        ModelLoaderImplementation,
        PaddedCollatorConfiguration,
        PowerCoolConfiguration,
        RegisteredBatching,
        RegisteredCollator,
        RegisteredDataSource,
        RegisteredSampleMapper,
        RegisteredSampleProcessor,
        RegisteredSampler,
        RegisteredSplitSelector,
        RegisteredTrainability,
        RWKVModelFactory,
        RWKVModelFactoryConfiguration,
        SampleMapperImplementation,
        SampleProcessorImplementation,
        SamplerImplementation,
        ScheduleImplementation,
        SplitSelectorImplementation,
        TokenIdsProcessorConfiguration,
        TrainabilityImplementation,
    )

    read_root = tmp_path / "read"
    run_directory = tmp_path / "write" / "run"
    read_root.mkdir()
    run_directory.parent.mkdir()
    checkpoint_root = tmp_path / "checkpoint-input"
    checkpoint_root.mkdir()
    initial_checkpoint = checkpoint_root / "scratch-state.pt"
    initial_checkpoint.write_bytes(b"sealed checkpoint fixture")
    rows = {
        "train": [
            {"id": "train-a", "split": "train", "tokens": list(range(1, 17))},
            {"id": "train-b", "split": "train", "tokens": list(range(17, 33))},
        ],
        "validation": [
            {"id": "val-a", "split": "validation", "tokens": list(range(33, 49))},
            {"id": "val-b", "split": "validation", "tokens": list(range(49, 65))},
        ],
        "test": [{"id": "test-a", "split": "test", "tokens": [65, 66]}],
    }
    files = {}
    counts = {}
    for split, values in rows.items():
        path = read_root / f"{split}.jsonl"
        path.write_text(
            "".join(json.dumps(value) + "\n" for value in values), encoding="utf-8"
        )
        encoded = path.read_bytes()
        counts[split] = len(values)
        files[path.name] = {
            "rows": len(values),
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }
    (read_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "fixture.manifested-token-splits.v1",
                "dataset_digest": "fixture",
                "counts": counts,
                "files": files,
                "unique_content_hashes": sum(counts.values()),
            }
        ),
        encoding="utf-8",
    )
    data_fingerprint = measure_input_content_root(read_root).tree_sha256
    source = RegisteredDataSource(
        DataSourceImplementation.JSONL_FROZEN_TOKEN_SPLITS_V1,
        JsonlFrozenTokenSplitsConfiguration(
            dataset_root=str(read_root),
            content_fingerprint=data_fingerprint,
            declared_columns=("id", "split", "tokens"),
            token_column="tokens",
            id_column="id",
        ),
    )

    def pipeline(selection):
        return DeclarativeDataPipeline(
            source=source,
            processor=RegisteredSampleProcessor(
                SampleProcessorImplementation.TOKEN_IDS_V1,
                TokenIdsProcessorConfiguration(
                    token_column="tokens",
                    vocabulary_size=65_536,
                    minimum_tokens=2,
                    maximum_tokens=128,
                ),
            ),
            mapper=RegisteredSampleMapper(
                SampleMapperImplementation.CAUSAL_TOKENS_V1,
                CausalTokensMapperConfiguration(
                    token_column="tokens", maximum_tokens=128
                ),
            ),
            collator=RegisteredCollator(
                CollatorImplementation.PADDED_V1,
                PaddedCollatorConfiguration(
                    pad_token_id=0,
                    label_pad_token_id=-100,
                    pad_to_multiple=1,
                    maximum_sequence_length=4,
                ),
            ),
            sampler=RegisteredSampler(
                SamplerImplementation.DETERMINISTIC_V1,
                DeterministicSamplerConfiguration(seed=7, shuffle=True),
            ),
            batching=RegisteredBatching(
                implementation="rwkv_lab.batching.fixed.v1",
                configuration=FixedBatchingConfiguration(
                    batch_size=2, drop_last=False, prefetch_workers=0
                ),
            ),
            split_selector=RegisteredSplitSelector(
                SplitSelectorImplementation.FROZEN_NAMED_V1,
                FrozenNamedSplitConfiguration(selection=selection),
            ),
        )

    class Components:
        composition = SimpleNamespace(
            components={
                "evaluator": SimpleNamespace(descriptor_digest="sha256:" + "e" * 64)
            }
        )

        def model_loader(self):
            return RWKVModelFactory(
                implementation="rwkv_lab.model_loader.rwkv_scratch.v1",
                configuration=RWKVModelFactoryConfiguration(65_536, 64, 1, 16, 7),
            )

        def trainability(self):
            return RegisteredTrainability(
                TrainabilityImplementation.FULL_V1,
                FullTrainabilityConfiguration(),
            )

        def learning_rate_configuration(self):
            return ScheduleImplementation.POWERCOOL_V1, PowerCoolConfiguration(
                warmup_steps=2,
                max_steps=120,
                minimum_ratio=0.1,
                cooldown_fraction=0.2,
                power=2.0,
            )

        def configuration(self, slot, *, category):
            del category
            return {
                "optimizer": {"learning_rate": 3.0e-4},
                "weight_decay": {"weight_decay": 0.1},
                "gradient_clipping": {"max_norm": 1.0},
                "sampler": {"seed": 7},
                "batching": {"batch_size": 2},
                "collation": {"maximum_sequence_length": 4},
                "processor": {"vocabulary_size": 65_536},
            }[slot]

        def gradient_accumulation(self):
            return FixedGradientAccumulation(
                FixedGradientAccumulationConfiguration(1)
            )

        def curriculum(self):
            return SimpleNamespace(
                configuration=SimpleNamespace(
                    maximum_sequence_length=4, base_batch_size=2
                )
            )

        def evaluator(self):
            return SimpleNamespace(
                configuration=SimpleNamespace(
                    maximum_examples=2,
                    metrics=("perplexity", "validation_loss"),
                )
            )

        def evaluation_schedule(self):
            return SimpleNamespace(
                configuration=SimpleNamespace(
                    full_every_steps=20,
                    qualitative_every_steps=20,
                )
            )

        def artifact_renderer(self):
            return SimpleNamespace(configuration=SimpleNamespace(modality="text"))

        def generation_policy(self):
            return SimpleNamespace(digest="sha256:" + "a" * 64)

        def checkpoint_policy(self):
            return SimpleNamespace(
                configuration=SimpleNamespace(publish_final=True),
                due=lambda step, final=False: final and step >= 0,
            )

        def data_pipeline(self, *, split_slot):
            return pipeline("train" if split_slot == "split" else "validation")

        def qualitative_samples(self):
            return DerivedFixedHeldOutSamples(
                DerivedFixedHeldOutConfiguration("id", 2)
            )

    components = Components()
    observed = []

    def train(arguments, **kwargs):
        observed.append((arguments, kwargs))
        checkpoint = Path(arguments[arguments.index("--save") + 1])
        checkpoint.write_bytes(b"checkpoint")
        return {"checkpoint": str(checkpoint), "step": 120}

    monkeypatch.setattr(rwkv_pretrain, "main", train)
    invocation = SimpleNamespace(
        inputs={"config": {}},
        workspace=_sealed_workspace(read_root, run_directory),
        publishes={"checkpoint": {}},
    )
    profiler = SimpleNamespace()
    observability = SimpleNamespace()
    controls = SimpleNamespace()
    execution_phases = SimpleNamespace()

    result = _rwkv_scratch(
        invocation,
        components,
        step_profiler=profiler,
        observability=observability,
        controls=controls,
        execution_phases=execution_phases,
    )
    arguments, keyword_arguments = observed[0]
    assert arguments[arguments.index("--data") + 1] == str(
        run_directory / "registered-corpus.uint16"
    )
    assert arguments[arguments.index("--out") + 1] == str(run_directory.resolve())
    assert arguments[arguments.index("--optimizer") + 1] == "adamw"
    assert arguments[arguments.index("--lr-schedule") + 1] == "powercool"
    assert "--compile" not in arguments
    eval_policy = keyword_arguments.pop("worker_eval_examples")
    assert tuple(eval_policy.heldout_tokens) == ("val-a", "val-b")
    assert eval_policy.identity_field == "id"
    assert keyword_arguments == {
        "worker_components": components,
        "worker_step_profiler": profiler,
        "worker_observability": observability,
        "worker_controls": controls,
        "worker_execution_phases": execution_phases,
    }
    assert result.optimizer_step == 120
    assert len(result.checkpoint_requests) == 1
    request = result.checkpoint_requests[0]
    assert request.source_directory == run_directory / "checkpoint-final"
    assert request.resume_grade == "terminal_checkpoint"

    class ContinuationComponents(Components):
        def model_loader(self):
            return RWKVModelFactory(
                implementation=ModelLoaderImplementation.RWKV_CHECKPOINT_V1,
                configuration=RWKVModelFactoryConfiguration(
                    65_536,
                    64,
                    1,
                    16,
                    7,
                    str(initial_checkpoint),
                    measure_input_content_root(initial_checkpoint).tree_sha256,
                ),
            )

    continuation_run_directory = tmp_path / "write" / "continued"
    continuation_workspace = _sealed_workspace(
        read_root, continuation_run_directory
    )
    continuation_workspace["allowed_read_roots"].append(str(checkpoint_root))
    continuation_workspace["allowed_read_roots"].sort()
    continuation_workspace["input_content_roots"].append(
        asdict(measure_input_content_root(initial_checkpoint))
    )
    continuation_workspace["input_content_roots"].sort(key=lambda item: item["path"])
    continuation = _rwkv_scratch(
        SimpleNamespace(
            inputs={"config": {}},
            workspace=continuation_workspace,
            publishes={"checkpoint": {}},
        ),
        ContinuationComponents(),
        step_profiler=profiler,
        observability=observability,
        controls=controls,
        execution_phases=execution_phases,
    )
    continuation_arguments = observed[1][0]
    assert continuation_arguments[
        continuation_arguments.index("--init-checkpoint") + 1
    ] == str(initial_checkpoint.resolve())
    assert "--resume" not in continuation_arguments
    assert continuation.optimizer_step == 120


def test_rwkv_posttraining_handler_seals_inputs_and_publishes_adapter_bundle(
    tmp_path, monkeypatch
) -> None:
    from pathlib import Path

    from rwkv_lab import posttrain_train

    read_root = tmp_path / "read"
    run_directory = tmp_path / "write" / "run"
    read_root.mkdir()
    run_directory.mkdir(parents=True)
    checkpoint = read_root / "base.pt"
    dataset = read_root / "sft.jsonl"
    checkpoint.write_bytes(b"checkpoint")
    dataset.write_text('{"schema":"fixture"}\n', encoding="utf-8")
    observed = []

    def train(**values):
        observed.append(values)
        output = Path(values["output"])
        adapter = output / "adapter"
        adapter.mkdir()
        (adapter / "weights.safetensors").write_bytes(b"weights")
        (output / "posttrain-result.json").write_text(
            '{"steps":3}\n', encoding="utf-8"
        )
        return {"steps": 3}

    monkeypatch.setattr(posttrain_train, "train", train)
    invocation = SimpleNamespace(
        inputs={
            "config": {
                "checkpoint": str(checkpoint),
                "data": str(dataset),
                "output_dir": str(run_directory),
                "steps": 3,
                "log_every": 1,
            }
        },
        workspace=_sealed_workspace(read_root, run_directory),
        publishes={"adapter": {}},
        resume=None,
    )
    components = SimpleNamespace()
    observability = SimpleNamespace(
        keepalive=lambda *_args: __import__("contextlib").nullcontext()
    )
    controls = SimpleNamespace(effective_values={})

    result = _rwkv_posttraining(
        invocation,
        components,
        observability=observability,
        controls=controls,
    )

    assert result.event_type == "worker.completed"
    assert result.optimizer_step == 3
    assert result.payload == {"reason": "training_complete", "objective": "sft"}
    assert len(result.artifact_requests) == 1
    assert result.artifact_requests[0].output_name == "adapter"
    assert result.artifact_requests[0].source_directory.parent == run_directory
    assert result.artifact_requests[0].source_directory.name.startswith(
        "posttraining-output-"
    )
    assert observed[0]["checkpoint"] == str(checkpoint.resolve())
    assert observed[0]["data"] == str(dataset.resolve())
    assert observed[0]["worker_components"] is components


def test_rlvr_handler_seals_verifier_and_publishes_terminal_checkpoint(
    tmp_path, monkeypatch
) -> None:
    from pathlib import Path

    from rwkv_lab import rlvr_train

    read_root = tmp_path / "read"
    run_directory = tmp_path / "write" / "run"
    read_root.mkdir()
    run_directory.mkdir(parents=True)
    checkpoint = read_root / "base.pt"
    vocab = read_root / "vocab.txt"
    tasks = read_root / "tasks.jsonl"
    verifier = read_root / "verifier"
    checkpoint.write_bytes(b"checkpoint")
    vocab.write_text("vocab\n", encoding="utf-8")
    tasks.write_text(
        '{"id":"one","split":"train","prompt":"1+1",'
        '"verifier":{"kind":"numeric","expected":2}}\n',
        encoding="utf-8",
    )
    verifier.write_bytes(b"verifier")
    observed = []

    def fake_run(arguments, **hooks):
        observed.append((arguments, hooks))
        candidate = Path(arguments.out) / "rlvr.pt"
        candidate.write_bytes(b"candidate")
        return {
            "status": "complete",
            "steps_completed": 4,
            "training_status": "complete",
            "checkpoint": str(candidate.resolve()),
            "promotion": {"eligible": True},
        }

    monkeypatch.setattr(rlvr_train, "run", fake_run)
    invocation = SimpleNamespace(
        inputs={
            "config": {
                "checkpoint": str(checkpoint),
                "output_dir": str(run_directory),
                "vocab": str(vocab),
                "tasks": str(tasks),
                "steps": 4,
                "verifier_executable": str(verifier),
                "verifier_arguments": ("--profile", "math"),
            }
        },
        workspace=_sealed_workspace(read_root, run_directory),
        publishes={"checkpoint": {}},
        resume=None,
    )
    components = SimpleNamespace()
    profiler = SimpleNamespace()
    observability = SimpleNamespace(
        keepalive=lambda *_args: __import__("contextlib").nullcontext()
    )
    controls = SimpleNamespace(effective_values={})

    result = _rlvr(
        invocation,
        components,
        step_profiler=profiler,
        observability=observability,
        controls=controls,
    )

    arguments, hooks = observed[0]
    assert arguments.ckpt == str(checkpoint.resolve())
    assert arguments.tasks == str(tasks.resolve())
    assert arguments.verifier_command == (
        str(verifier.resolve()),
        "--profile",
        "math",
    )
    assert hooks["worker_components"] is components
    assert hooks["worker_step_profiler"] is profiler
    assert result.event_type == "worker.completed"
    assert result.optimizer_step == 4
    assert result.payload["promotion_eligible"] is True
    request = result.checkpoint_requests[0]
    assert request.resume_grade == "terminal_checkpoint"
    assert request.source_directory == run_directory / "checkpoint-final"
    assert (request.source_directory / "state.pt").read_bytes() == b"candidate"


def test_vision_compressor_handler_seals_inputs_and_publishes_checkpoint(
    tmp_path, monkeypatch
) -> None:
    from rwkv_lab import vision_teacher_compressor

    read_root = tmp_path / "read"
    run_directory = tmp_path / "write" / "run"
    image = read_root / "image.png"
    moon_cache = read_root / "moon-cache"
    fusion_cache = read_root / "fusion-cache"
    model_directories = [read_root / name for name in ("siglip", "dino", "sam")]
    read_root.mkdir()
    run_directory.mkdir(parents=True)
    moon_cache.mkdir()
    fusion_cache.mkdir()
    for directory in model_directories:
        directory.mkdir()
    image.write_bytes(b"image")
    manifest_row = json.dumps({"image": str(image)}) + "\n"
    train_manifest = read_root / "train.jsonl"
    eval_manifest = read_root / "eval.jsonl"
    train_manifest.write_text(manifest_row, encoding="utf-8")
    eval_manifest.write_text(manifest_row, encoding="utf-8")
    moonvit = read_root / "moonvit.safetensors"
    moonvit.write_bytes(b"weights")
    observed = []

    def fake_train(arguments, **hooks):
        observed.append((arguments, hooks))
        checkpoint = arguments.out / "checkpoint-current"
        checkpoint.mkdir()
        (checkpoint / "state.pt").write_bytes(b"checkpoint")
        return {
            "status": "complete",
            "step": 7,
            "checkpoint": str(checkpoint.resolve()),
            "best_eval": 0.5,
        }

    monkeypatch.setattr(vision_teacher_compressor, "train", fake_train)
    invocation = SimpleNamespace(
        inputs={
            "config": {
                "train_manifest": str(train_manifest),
                "eval_manifest": str(eval_manifest),
                "moon_cache": str(moon_cache),
                "fusion_cache": str(fusion_cache),
                "output_dir": str(run_directory),
                "moonvit_checkpoint": str(moonvit),
                "siglip2_model": str(model_directories[0]),
                "dinov2_model": str(model_directories[1]),
                "sam_model": str(model_directories[2]),
                "steps": 7,
                "eval_every": 7,
                "checkpoint_every": 7,
                "log_every": 1,
            }
        },
        workspace=_sealed_workspace(read_root, run_directory),
        publishes={"checkpoint": {}},
        resume=None,
    )
    components = SimpleNamespace()
    profiler = SimpleNamespace()
    observability = SimpleNamespace(
        keepalive=lambda *_args: __import__("contextlib").nullcontext()
    )
    controls = SimpleNamespace(effective_values={})

    result = _vision_teacher_compressor(
        invocation,
        components,
        step_profiler=profiler,
        observability=observability,
        controls=controls,
    )

    arguments, hooks = observed[0]
    assert arguments.data == train_manifest.resolve()
    assert arguments.eval_data == eval_manifest.resolve()
    assert arguments.resume == "none"
    assert arguments.resume_from is None
    assert arguments.preflight_only is False
    assert hooks["worker_components"] is components
    assert hooks["worker_step_profiler"] is profiler
    assert result.event_type == "worker.completed"
    assert result.optimizer_step == 7
    assert len(result.checkpoint_requests) == 1
    request = result.checkpoint_requests[0]
    assert request.source_directory == run_directory / "checkpoint-current"
    assert request.resume_grade == "compatible"
    assert "data_cursor" in request.state_components


def test_frozen_vision_handler_lowers_compressor_arm_to_sealed_cache_only_argv(
    tmp_path, monkeypatch
) -> None:
    from rwkv_lab import vision_train

    read_root = tmp_path / "read"
    run_directory = tmp_path / "write" / "run"
    read_root.mkdir()
    run_directory.mkdir(parents=True)
    image = read_root / "image.png"
    image.write_bytes(b"image")
    row = json.dumps({"image": str(image), "text": "caption"}) + "\n"
    train_manifest = read_root / "train.jsonl"
    eval_manifest = read_root / "eval.jsonl"
    train_manifest.write_text(row, encoding="utf-8")
    eval_manifest.write_text(row, encoding="utf-8")
    files = {
        name: read_root / name
        for name in ("rwkv.pt", "moonvit.safetensors", "vocab.txt", "compressor.pt")
    }
    for path in files.values():
        path.write_bytes(b"sealed")
    directories = {
        name: read_root / name
        for name in ("moon-cache", "fusion-cache", "siglip2", "dinov2", "sam")
    }
    for path in directories.values():
        path.mkdir()
    observed = []
    node_run_directory = run_directory / "nodes" / "compressor_arm"

    def fake_train(argv, **hooks):
        observed.append((argv, hooks))
        checkpoint = node_run_directory / "checkpoint-current"
        checkpoint.mkdir(parents=True)
        (checkpoint / "state.pt").write_bytes(b"checkpoint")
        return {
            "status": "complete",
            "step": 1,
            "checkpoint": str(checkpoint.resolve()),
            "best_eval_loss": 1.25,
        }

    monkeypatch.setattr(vision_train, "train", fake_train)
    invocation = SimpleNamespace(
        inputs={
            "config": {
                "arm": "compressor",
                "train_manifests": (str(train_manifest),),
                "eval_manifests": (str(eval_manifest),),
                "rwkv_checkpoint": str(files["rwkv.pt"]),
                "moonvit_checkpoint": str(files["moonvit.safetensors"]),
                "vocab": str(files["vocab.txt"]),
                "moon_cache": str(directories["moon-cache"]),
                "compressor_checkpoint": str(files["compressor.pt"]),
                "fusion_cache": str(directories["fusion-cache"]),
                "siglip2_model": str(directories["siglip2"]),
                "dinov2_model": str(directories["dinov2"]),
                "sam_model": str(directories["sam"]),
                "steps": 1,
                "checkpoint_every": 1,
                "eval_every": 1,
                "log_every": 1,
                "require_fused_ce": False,
            }
        },
        workspace=_sealed_workspace(read_root, run_directory),
        publishes={"checkpoint": {}, "result": {}},
        resume=None,
        node_id="compressor_arm",
    )
    components = SimpleNamespace()
    profiler = SimpleNamespace()
    observability = SimpleNamespace(
        keepalive=lambda *_args: __import__("contextlib").nullcontext()
    )
    controls = SimpleNamespace(effective_values={})

    result = _vision_frozen_adapter(
        invocation,
        components,
        step_profiler=profiler,
        observability=observability,
        controls=controls,
    )

    argv, hooks = observed[0]
    assert argv[argv.index("--rwkv") + 1] == str(files["rwkv.pt"].resolve())
    assert argv[argv.index("--vocab") + 1] == str(files["vocab.txt"].resolve())
    assert argv[argv.index("--resume") + 1] == "none"
    assert "--feature-cache-only" in argv
    assert "--fusion-cache-only" in argv
    assert "--vision-compressor-checkpoint" in argv
    assert "--operator-profile-steps" in argv
    assert "--no-require-fused-ce" in argv
    assert hooks["worker_components"] is components
    assert hooks["worker_step_profiler"] is profiler
    assert result.event_type == "worker.completed"
    assert result.optimizer_step == 1
    assert result.payload["arm"] == "compressor"
    request = result.checkpoint_requests[0]
    assert request.source_directory == node_run_directory / "checkpoint-current"
    assert request.resume_grade == "compatible"
    assert "data_cursor" in request.state_components
    assert len(result.artifact_requests) == 1
    scalar_result = result.artifact_requests[0]
    assert scalar_result.output_name == "result"
    assert json.loads((scalar_result.source_directory / "result.json").read_text()) == {
        "api_version": "rwkv-lab.scalar-metric-result/v1",
        "direction": "minimize",
        "metric": "eval.loss",
        "optimizer_step": 1,
        "subject": "compressor",
        "value": 1.25,
    }


def test_vision_native_head_handler_seals_lineage_and_publishes_checkpoint(
    tmp_path, monkeypatch
) -> None:
    from rwkv_lab import vision_native_train

    read_root = tmp_path / "read"
    run_directory = tmp_path / "write" / "run"
    read_root.mkdir()
    run_directory.mkdir(parents=True)
    image = read_root / "image.png"
    image.write_bytes(b"image")
    row = json.dumps({"image": str(image), "caption": "fixture"}) + "\n"
    train_manifest = read_root / "train.jsonl"
    eval_manifest = read_root / "eval.jsonl"
    train_manifest.write_text(row, encoding="utf-8")
    eval_manifest.write_text(row, encoding="utf-8")
    files = {
        name: read_root / name
        for name in (
            "baseline.pt",
            "compressor.pt",
            "moonvit.safetensors",
            "vocab.txt",
        )
    }
    for path in files.values():
        path.write_bytes(b"sealed")
    directories = {
        name: read_root / name
        for name in ("moon-cache", "fusion-cache", "siglip2", "dinov2", "sam")
    }
    for path in directories.values():
        path.mkdir()
    observed = []

    def fake_train(arguments, **hooks):
        observed.append((arguments, hooks))
        checkpoint = Path(arguments.out) / "checkpoint-current"
        checkpoint.mkdir()
        (checkpoint / "state.pt").write_bytes(b"checkpoint")
        return {
            "status": "complete",
            "step": 9,
            "checkpoint": str(checkpoint.resolve()),
            "best_eval": 1.25,
        }

    from pathlib import Path

    monkeypatch.setattr(vision_native_train, "train", fake_train)
    invocation = SimpleNamespace(
        inputs={
            "config": {
                "baseline_checkpoint": str(files["baseline.pt"]),
                "compressor_checkpoint": str(files["compressor.pt"]),
                "train_manifest": str(train_manifest),
                "eval_manifest": str(eval_manifest),
                "moon_cache": str(directories["moon-cache"]),
                "fusion_cache": str(directories["fusion-cache"]),
                "moonvit_checkpoint": str(files["moonvit.safetensors"]),
                "siglip2_model": str(directories["siglip2"]),
                "dinov2_model": str(directories["dinov2"]),
                "sam_model": str(directories["sam"]),
                "vocab": str(files["vocab.txt"]),
                "output_dir": str(run_directory),
                "steps": 9,
                "eval_every": 9,
                "checkpoint_every": 9,
            }
        },
        workspace=_sealed_workspace(read_root, run_directory),
        publishes={"checkpoint": {}},
        resume=None,
    )
    components = SimpleNamespace()
    profiler = SimpleNamespace()
    observability = SimpleNamespace(
        keepalive=lambda *_args: __import__("contextlib").nullcontext()
    )
    controls = SimpleNamespace(effective_values={})

    result = _vision_native_head(
        invocation,
        components,
        step_profiler=profiler,
        observability=observability,
        controls=controls,
    )

    arguments, hooks = observed[0]
    assert arguments.baseline == str(files["baseline.pt"].resolve())
    assert arguments.compressor == str(files["compressor.pt"].resolve())
    assert arguments.resume_from == ""
    assert hooks["worker_components"] is components
    assert result.event_type == "worker.completed"
    assert result.optimizer_step == 9
    request = result.checkpoint_requests[0]
    assert request.resume_grade == "compatible"
    assert "data_cursor" not in request.state_components


def test_scalar_metric_decision_publishes_lineage_bound_winner(
    tmp_path, monkeypatch
) -> None:
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    artifacts = {
        "left": SimpleNamespace(artifact_id="artifact-moonvit"),
        "right": SimpleNamespace(artifact_id="artifact-compressor"),
    }
    documents = {
        "artifact-moonvit": {
            "api_version": "rwkv-lab.scalar-metric-result/v1",
            "direction": "minimize",
            "metric": "eval.loss",
            "optimizer_step": 2000,
            "subject": "moonvit",
            "value": 1.5,
        },
        "artifact-compressor": {
            "api_version": "rwkv-lab.scalar-metric-result/v1",
            "direction": "minimize",
            "metric": "eval.loss",
            "optimizer_step": 2000,
            "subject": "compressor",
            "value": 1.25,
        },
    }
    monkeypatch.setattr(
        "rwkv_lab.trainvm_adapters.handlers.resolve_input_artifact",
        lambda _invocation, name, **_requirements: artifacts[name],
    )
    monkeypatch.setattr(
        "rwkv_lab.trainvm_adapters.handlers.load_input_artifact_json",
        lambda artifact, _path, **_bounds: documents[artifact.artifact_id],
    )
    invocation = SimpleNamespace(
        inputs={
            "config": {
                "metric": "eval.loss",
                "direction": "minimize",
                "left_subject": "moonvit",
                "right_subject": "compressor",
                "absolute_tolerance": 0.01,
            },
            "left": {},
            "right": {},
        },
        workspace={
            "run_directory": str(run_directory),
            "allowed_read_roots": [str(tmp_path)],
            "allowed_write_roots": [str(tmp_path)],
        },
        publishes={"decision": {}},
        resume=None,
        node_id="compare_arms",
        attempt_id="compare_arms@1",
    )

    result = _scalar_metric_decision(
        invocation,
        None,
        observability=SimpleNamespace(
            keepalive=lambda *_args: __import__("contextlib").nullcontext()
        ),
        controls=SimpleNamespace(effective_values={}),
    )

    assert result.event_type == "operation.completed"
    assert result.payload == {
        "outcome": "selected",
        "selected_subject": "compressor",
    }
    request = result.artifact_requests[0]
    assert request.output_name == "decision"
    assert request.parent_artifact_ids == (
        "artifact-moonvit",
        "artifact-compressor",
    )
    decision = json.loads((request.source_directory / "decision.json").read_text())
    assert decision["selected_subject"] == "compressor"
    assert decision["selected_artifact_id"] == "artifact-compressor"
    assert decision["signed_right_minus_left"] == -0.25


def test_vision_student_handler_seals_lineage_and_publishes_checkpoint(
    tmp_path, monkeypatch
) -> None:
    from rwkv_lab import vision_rwkv_student_train

    read_root = tmp_path / "read"
    run_directory = tmp_path / "write" / "run"
    read_root.mkdir()
    run_directory.mkdir(parents=True)
    image = read_root / "image.png"
    image.write_bytes(b"image")
    row = json.dumps({"image": str(image), "caption": "fixture"}) + "\n"
    train_manifest = read_root / "train.jsonl"
    eval_manifest = read_root / "eval.jsonl"
    train_manifest.write_text(row, encoding="utf-8")
    eval_manifest.write_text(row, encoding="utf-8")
    files = {
        name: read_root / name
        for name in (
            "baseline.pt",
            "compressor.pt",
            "native.pt",
            "moonvit.safetensors",
            "vocab.txt",
        )
    }
    for path in files.values():
        path.write_bytes(b"sealed")
    directories = {
        name: read_root / name
        for name in ("moon-cache", "fusion-cache", "siglip2", "dinov2", "sam")
    }
    for path in directories.values():
        path.mkdir()
    observed = []

    def fake_train(arguments, **hooks):
        observed.append((arguments, hooks))
        checkpoint = Path(arguments.out) / "checkpoint-current"
        checkpoint.mkdir()
        (checkpoint / "state.pt").write_bytes(b"checkpoint")
        return {
            "status": "complete",
            "step": 11,
            "checkpoint": str(checkpoint.resolve()),
            "best_ppl": 1.5,
        }

    from pathlib import Path

    monkeypatch.setattr(vision_rwkv_student_train, "train", fake_train)
    invocation = SimpleNamespace(
        inputs={
            "config": {
                "baseline_checkpoint": str(files["baseline.pt"]),
                "compressor_checkpoint": str(files["compressor.pt"]),
                "native_head_checkpoint": str(files["native.pt"]),
                "train_manifest": str(train_manifest),
                "eval_manifest": str(eval_manifest),
                "moon_cache": str(directories["moon-cache"]),
                "fusion_cache": str(directories["fusion-cache"]),
                "moonvit_checkpoint": str(files["moonvit.safetensors"]),
                "siglip2_model": str(directories["siglip2"]),
                "dinov2_model": str(directories["dinov2"]),
                "sam_model": str(directories["sam"]),
                "vocab": str(files["vocab.txt"]),
                "output_dir": str(run_directory),
                "steps": 11,
                "eval_every": 11,
                "checkpoint_every": 11,
            }
        },
        workspace=_sealed_workspace(read_root, run_directory),
        publishes={"checkpoint": {}},
        resume=None,
    )
    components = SimpleNamespace()
    profiler = SimpleNamespace()
    observability = SimpleNamespace(
        keepalive=lambda *_args: __import__("contextlib").nullcontext()
    )
    controls = SimpleNamespace(effective_values={})

    result = _vision_rwkv_student(
        invocation,
        components,
        step_profiler=profiler,
        observability=observability,
        controls=controls,
    )

    arguments, hooks = observed[0]
    assert arguments.baseline == str(files["baseline.pt"].resolve())
    assert arguments.compressor == str(files["compressor.pt"].resolve())
    assert arguments.native_head == str(files["native.pt"].resolve())
    assert arguments.resume_from == ""
    assert hooks["worker_components"] is components
    assert result.event_type == "worker.completed"
    assert result.optimizer_step == 11
    request = result.checkpoint_requests[0]
    assert request.resume_grade == "compatible"
    assert "data_cursor" in request.state_components


def test_transformer_mla_handler_binds_paths_profile_and_compatible_checkpoint(
    tmp_path, monkeypatch
) -> None:
    from rwkv_lab import train_mla

    read_root = tmp_path / "read"
    model_dir = read_root / "model"
    patch_dir = read_root / "patch"
    engram_dir = read_root / "engram"
    run_directory = tmp_path / "write" / "run"
    model_dir.mkdir(parents=True)
    patch_dir.mkdir()
    engram_dir.mkdir()
    run_directory.mkdir(parents=True)
    tokens = read_root / "tokens.bin"
    tokens.write_bytes(b"\x00" * 4096)
    observed = []

    def fake_train(config, **kwargs):
        observed.append((config, kwargs))
        checkpoint = run_directory / "step_000010"
        checkpoint.mkdir(exist_ok=True)
        (checkpoint / "ckpt.pt").write_bytes(b"checkpoint")
        return {"status": "complete", "step": 10, "checkpoint": str(checkpoint)}

    monkeypatch.setattr(train_mla, "train", fake_train)
    invocation = SimpleNamespace(
        adapter={
            "adapter": "rwkv-lab.transformer-mla",
            "version": "1.0.0",
            "operation": "train",
            "contract": "rwkv_lab.transformer_mla.v1.Train",
        },
        inputs={
            "config": {
                "profile": "mla",
                "model_dir": str(model_dir),
                "patch_dir": str(patch_dir),
                "tokens_bin": str(tokens),
                "output_dir": str(run_directory),
                "total_tokens_in_bin": 1024,
                "eval_tokens": 64,
                "max_steps": 10,
                "warmup_steps": 2,
                "eval_every_steps": 5,
                "eval_batches": 2,
                "save_every_steps": 10,
            }
        },
        workspace=_sealed_workspace(read_root, run_directory),
        publishes={"checkpoint": {}},
    )
    required_implementations = []
    components = SimpleNamespace(
        require_implementation=lambda slot, **values: required_implementations.append(
            (slot, values)
        )
    )
    profiler = SimpleNamespace()
    observability = SimpleNamespace()
    controls = SimpleNamespace()

    result = _transformer_mla(
        invocation,
        components,
        step_profiler=profiler,
        observability=observability,
        controls=controls,
    )

    lowered, keyword_arguments = observed[0]
    assert lowered.model_dir == str(model_dir.resolve())
    assert lowered.patch_dir == str(patch_dir.resolve())
    assert lowered.tokens_bin == str(tokens.resolve())
    assert lowered.out_dir == str(run_directory.resolve())
    assert lowered.optimizer == "trainvm"
    assert required_implementations == [
        (
            "optimizer",
            {
                "category": "optimizer",
                "allowed": frozenset(
                    {
                        "rwkv_lab.optimizer.torch_adamw.v1",
                        "rwkv_lab.optimizer.torch_adamw_no_decay.v2",
                    }
                ),
            },
        )
    ]
    assert keyword_arguments == {
        "worker_components": components,
        "worker_step_profiler": profiler,
        "worker_observability": observability,
        "worker_controls": controls,
    }
    assert result.optimizer_step == 10
    assert result.checkpoint_requests[0].resume_grade == "compatible"
    assert "topology" in result.checkpoint_requests[0].state_components

    wrong_profile = SimpleNamespace(**vars(invocation))
    wrong_profile.inputs = {
        "config": {**invocation.inputs["config"], "profile": "full_backbone"}
    }
    with pytest.raises(AdapterDispatchError, match="does not match"):
        _transformer_mla(wrong_profile, components)

    with pytest.raises(AdapterDispatchError, match="do not declare initial controls"):
        _transformer_mla(
            invocation,
            components,
            observability=observability,
            controls=SimpleNamespace(effective_values={"learning_rate": 1.0e-5}),
        )

    bad_observability = SimpleNamespace(
        declaration=SimpleNamespace(
            metrics={
                "train.loss": SimpleNamespace(step_domain="token"),
            }
        )
    )
    with pytest.raises(AdapterDispatchError, match="optimizer_step"):
        _transformer_mla(
            invocation,
            components,
            observability=bad_observability,
            controls=SimpleNamespace(effective_values={}),
        )

    engram_invocation = SimpleNamespace(**vars(invocation))
    engram_invocation.adapter = {
        **invocation.adapter,
        "adapter": "rwkv-lab.transformer-mla-engram",
        "contract": "rwkv_lab.transformer_mla_engram.v1.Train",
    }
    engram_invocation.inputs = {
        "config": {
            **invocation.inputs["config"],
            "profile": "engram",
            "engram_patch_dir": str(engram_dir),
        }
    }
    required_implementations.clear()
    _transformer_mla(
        engram_invocation,
        components,
        observability=observability,
        controls=controls,
    )
    assert required_implementations == [
        (
            "optimizer",
            {
                "category": "optimizer",
                "allowed": frozenset(
                    {
                        "rwkv_lab.optimizer.torch_adamw.v1",
                        "rwkv_lab.optimizer.torch_adamw_no_decay.v2",
                    }
                ),
            },
        ),
        (
            "host_optimizer",
            {
                "category": "optimizer",
                "allowed": frozenset(
                    {"rwkv_lab.optimizer.torch_sparse_adam.v1"}
                ),
            },
        ),
    ]

    monkeypatch.setattr(
        train_mla,
        "train",
        lambda config, **kwargs: {
            "status": "interrupted",
            "step": 10,
            "checkpoint": str(run_directory / "step_000010"),
        },
    )
    interrupted = _transformer_mla(
        invocation,
        components,
        observability=observability,
        controls=controls,
    )
    assert interrupted.event_type == "operation.failed"
    assert interrupted.payload["reason"] == "checkpointed_interruption"


def test_dispatch_table_is_closed_and_training_composition_is_required() -> None:
    expected = {
        (
            "rwkv-lab.hf-multimodal-sft",
            "1.0.0",
            "train",
            "rwkv_lab.hf_multimodal_sft.v1.Train",
        ),
        (
            "rwkv-lab.mageflow-appearance-expert",
            "1.0.0",
            "train",
            "rwkv_lab.mageflow_appearance_expert.v1.Train",
        ),
        (
            "rwkv-lab.mageflow-terminal-expert",
            "1.0.0",
            "train",
            "rwkv_lab.mageflow_terminal_expert.v1.Train",
        ),
        (
            "rwkv-lab.mageflow-full-backbone",
            "1.0.0",
            "train",
            "rwkv_lab.mageflow_full_backbone.v1.Train",
        ),
        (
            "rwkv-lab.qwen-ao3",
            "1.0.0",
            "train",
            "rwkv_lab.qwen_ao3.v1.Train",
        ),
        (
            "rwkv-lab.scalar-metric-decision",
            "1.0.0",
            "decide",
            "rwkv_lab.scalar_metric_decision.v1.Decide",
        ),
        (
            "rwkv-lab.rwkv-posttraining",
            "1.0.0",
            "train",
            "rwkv_lab.rwkv_posttraining.v1.Train",
        ),
        (
            "rwkv-lab.rwkv-scratch",
            "1.0.0",
            "train",
            "rwkv_lab.rwkv_scratch.v1.Train",
        ),
        (
            "rwkv-lab.rwkv-rlvr",
            "1.0.0",
            "train",
            "rwkv_lab.rwkv_rlvr.v1.Train",
        ),
        (
            "rwkv-lab.vision-teacher-compressor",
            "1.0.0",
            "train",
            "rwkv_lab.vision_teacher_compressor.v1.Train",
        ),
        (
            "rwkv-lab.vision-frozen-adapter",
            "1.0.0",
            "train",
            "rwkv_lab.vision_frozen_adapter.v1.Train",
        ),
        (
            "rwkv-lab.vision-native-head",
            "1.0.0",
            "train",
            "rwkv_lab.vision_native_head.v1.Train",
        ),
        (
            "rwkv-lab.vision-rwkv-student",
            "1.0.0",
            "train",
            "rwkv_lab.vision_rwkv_student.v1.Train",
        ),
    }
    expected.update(
        {
            ("rwkv-lab.transformer-mla", "1.0.0", "train", "rwkv_lab.transformer_mla.v1.Train"),
            ("rwkv-lab.transformer-mla-mtp", "1.0.0", "train", "rwkv_lab.transformer_mla_mtp.v1.Train"),
            ("rwkv-lab.transformer-mla-mutor", "1.0.0", "train", "rwkv_lab.transformer_mla_mutor.v1.Train"),
            ("rwkv-lab.transformer-mla-fsp", "1.0.0", "train", "rwkv_lab.transformer_mla_fsp.v1.Train"),
            ("rwkv-lab.transformer-mla-parallel", "1.0.0", "train", "rwkv_lab.transformer_mla_parallel.v1.Train"),
            ("rwkv-lab.transformer-mla-rwkv8", "1.0.0", "train", "rwkv_lab.transformer_mla_rwkv8.v1.Train"),
            ("rwkv-lab.transformer-mla-engram", "1.0.0", "train", "rwkv_lab.transformer_mla_engram.v1.Train"),
            ("rwkv-lab.transformer-mla-full-backbone", "1.0.0", "train", "rwkv_lab.transformer_mla_full_backbone.v1.Train"),
        }
    )
    assert supported_adapter_keys() == expected
    unknown = SimpleNamespace(
        adapter={
            "adapter": "user.supplied.module",
            "version": "1.0.0",
            "operation": "train",
            "contract": "arbitrary.Import",
        },
        training=None,
    )
    with pytest.raises(AdapterDispatchError, match="closed adapter handler"):
        execute_invocation(unknown)

    supported = SimpleNamespace(
        adapter={
            "adapter": "rwkv-lab.qwen-ao3",
            "version": "1.0.0",
            "operation": "train",
            "contract": "rwkv_lab.qwen_ao3.v1.Train",
        },
        training=None,
    )
    with pytest.raises(AdapterDispatchError, match="no resolved composition"):
        execute_invocation(supported)


def test_a_declared_post_training_arm_no_handler_honours_is_refused() -> None:
    """An arm nobody reads must be refused, not carried and ignored.

    The arm declares what ends the run and what reproducibility it may claim,
    and the composition digest covers it — so a handler that does not read it
    would run the arm unbounded while the digest still asserts the run matched
    its declaration. Silently ignoring it is the exact failure the post-training
    authority exists to prevent, one layer below where it is checked.
    """
    arm = {
        "api_version": "trainvm.post-training-arm/v1",
        "arm_id": "arm.finetune-a",
        "kind": "supervised_finetune",
        "bounds": [{"kind": "optimizer_steps", "magnitude": 10000}],
        "reproducibility_claim": "exact",
    }
    invocation = SimpleNamespace(
        adapter={
            "adapter": "rwkv-lab.qwen-ao3",
            "version": "1.0.0",
            "operation": "train",
            "contract": "rwkv_lab.qwen_ao3.v1.Train",
        },
        training=SimpleNamespace(post_training=arm),
    )
    with pytest.raises(AdapterDispatchError) as failure:
        execute_invocation(invocation)
    message = str(failure.value)
    # Name the arm and the adapter: "refused" without either sends the reader
    # looking through the document for which of them was at fault.
    assert "arm.finetune-a" in message
    assert "rwkv-lab.qwen-ao3" in message

    # A composition carrying no arm gets past this check. It fails later for an
    # unrelated reason — this fixture is not a full composition — so assert on
    # what the failure is NOT, rather than pinning an incidental error further
    # down that would break the moment anything else changes.
    without = SimpleNamespace(
        adapter=dict(invocation.adapter), training=SimpleNamespace(post_training=None)
    )
    with pytest.raises(Exception) as unrelated:
        execute_invocation(without)
    assert "does not honour" not in str(unrelated.value), (
        "the refusal fired for a composition that declares no arm"
    )


def test_nonadopted_adapter_receipts_enabled_phase_failure_before_dispatch() -> None:
    receipts = []
    channel = SimpleNamespace(
        execution_phase_receipt=lambda request, disposition, **values: (
            receipts.append((request, disposition, values)) or 1
        )
    )
    phases = WorkerExecutionPhases(
        channel,
        (
            ExecutionPhaseRequest(
                phase=ExecutionPhase.COMPILE,
                enabled=True,
                steps=None,
                request_digest="sha256:" + "1" * 64,
            ),
        ),
    )
    invocation = SimpleNamespace(
        adapter={
            "adapter": "rwkv-lab.qwen-ao3",
            "version": "1.0.0",
            "operation": "train",
            "contract": "rwkv_lab.qwen_ao3.v1.Train",
        },
        invocation_digest="sha256:" + "2" * 64,
        training=SimpleNamespace(),
    )

    with pytest.raises(AdapterDispatchError, match="enabled compile"):
        execute_invocation(invocation, execution_phases=phases)
    assert len(receipts) == 1
    assert receipts[0][1].value == "failed"


def test_runner_reports_success_with_optimizer_step() -> None:
    bootstrap = SimpleNamespace(run_id="run-1")
    sessions: list[FakeSession] = []

    def factory(value: object) -> FakeSession:
        assert value is bootstrap
        session = FakeSession(value)
        sessions.append(session)
        return session

    status = run_worker(
        bootstrap_reader=lambda descriptor: (
            bootstrap
            if descriptor == WORKER_BOOTSTRAP_DESCRIPTOR
            else pytest.fail("wrong descriptor")
        ),
        session_factory=factory,
        executor=lambda invocation, _profiler, _observability, _controls, _phases: HandlerResult(
            "worker.completed", {"reason": "training_complete"}, 41
        ),
    )
    assert status == 0
    assert sessions[0].finished == [
        ("worker.completed", {"reason": "training_complete"}, 41)
    ]
    assert sessions[0].heartbeats == [(0, "initializing")]
    assert sessions[0].closed


def test_runner_publishes_handler_checkpoints_before_terminal_event(
    monkeypatch, tmp_path
) -> None:
    from rwkv_lab.trainvm_worker import CheckpointPublicationRequest

    bootstrap = SimpleNamespace(run_id="run-1")
    session = FakeSession(bootstrap)
    request = CheckpointPublicationRequest(
        tmp_path,
        optimizer_step=19,
        resume_grade="compatible",
        state_components=("model",),
    )
    observed = []

    def publish(actual_session, requests, *, progress):
        assert actual_session is session
        observed.extend(requests)
        progress(19)
        return (SimpleNamespace(artifact_id="checkpoint-artifact-19"),)

    monkeypatch.setattr(
        "rwkv_lab.trainvm_adapters.entrypoint.publish_checkpoint_requests", publish
    )
    status = run_worker(
        bootstrap_reader=lambda _descriptor: bootstrap,
        session_factory=lambda _bootstrap: session,
        executor=lambda _invocation, _profiler, _observability, _controls, _phases: (
            HandlerResult(
                "worker.completed",
                {"reason": "training_complete"},
                19,
                (request,),
            )
        ),
    )
    assert status == 0
    assert observed == [request]
    assert session.finished == [
        (
            "worker.completed",
            {
                "reason": "training_complete",
                "checkpoint_artifact_ids": ["checkpoint-artifact-19"],
            },
            19,
        )
    ]


def test_runner_publishes_handler_artifacts_before_terminal_event(
    monkeypatch, tmp_path
) -> None:
    from rwkv_lab.trainvm_worker import ArtifactPublicationRequest

    bootstrap = SimpleNamespace(run_id="run-1")
    session = FakeSession(bootstrap)
    request = ArtifactPublicationRequest(tmp_path, "adapter")
    observed = []

    def publish(actual_session, requests, *, progress):
        assert actual_session is session
        observed.extend(requests)
        progress()
        return (SimpleNamespace(artifact_id="adapter-artifact-19"),)

    monkeypatch.setattr(
        "rwkv_lab.trainvm_adapters.entrypoint.publish_artifact_requests", publish
    )
    status = run_worker(
        bootstrap_reader=lambda _descriptor: bootstrap,
        session_factory=lambda _bootstrap: session,
        executor=lambda _invocation, _profiler, _observability, _controls, _phases: (
            HandlerResult(
                "worker.completed",
                {"reason": "training_complete"},
                19,
                artifact_requests=(request,),
            )
        ),
    )
    assert status == 0
    assert observed == [request]
    assert session.finished == [
        (
            "worker.completed",
            {
                "reason": "training_complete",
                "artifact_ids": ["adapter-artifact-19"],
            },
            19,
        )
    ]


def test_runner_publishes_handler_eval_galleries_before_terminal_event(
    monkeypatch,
) -> None:
    from rwkv_lab.trainvm_worker import EvalGalleryPublicationRequest

    bootstrap = SimpleNamespace(run_id="run-1")
    session = FakeSession(bootstrap)
    request = EvalGalleryPublicationRequest(
        output_name="eval_gallery",
        step=23,
        step_domain="optimizer_step",
        checkpoint_manifest_digest="sha256:" + "c" * 64,
        evaluator_profile_digest="sha256:" + "d" * 64,
        use_policy_digest="sha256:" + "e" * 64,
        items=(),
        parent_artifact_ids=("checkpoint-23",),
    )
    observed = []

    def publish(actual_session, requests, *, progress):
        assert actual_session is session
        observed.extend(requests)
        progress(23)
        return (SimpleNamespace(artifact_id="eval-gallery-23"),)

    monkeypatch.setattr(
        "rwkv_lab.trainvm_adapters.entrypoint.publish_eval_gallery_requests",
        publish,
    )
    status = run_worker(
        bootstrap_reader=lambda _descriptor: bootstrap,
        session_factory=lambda _bootstrap: session,
        executor=lambda _invocation, _profiler, _observability, _controls, _phases: (
            HandlerResult(
                "worker.completed",
                {"reason": "evaluation_complete"},
                23,
                eval_gallery_requests=(request,),
            )
        ),
    )

    assert status == 0
    assert observed == [request]
    assert session.finished == [
        (
            "worker.completed",
            {
                "reason": "evaluation_complete",
                "eval_gallery_artifact_ids": ["eval-gallery-23"],
            },
            23,
        )
    ]


def test_runner_reports_sanitized_failure_and_skips_completed_replay() -> None:
    bootstrap = SimpleNamespace(run_id="run-1")
    failed = FakeSession(bootstrap)

    def raise_secret(
        _invocation: object,
        _profiler: object,
        _observability: object,
        _controls: object,
        _phases: object,
    ) -> HandlerResult:
        raise RuntimeError("secret dataset path")

    assert (
        run_worker(
            bootstrap_reader=lambda _descriptor: bootstrap,
            session_factory=lambda _bootstrap: failed,
            executor=raise_secret,
        )
        == 1
    )
    assert failed.finished == [
        (
            "operation.failed",
            {"code": "adapter_execution_failed", "error_type": "RuntimeError"},
            None,
        )
    ]
    assert "secret" not in repr(failed.finished)

    completed = FakeSession(bootstrap, completed=True)
    assert (
        run_worker(
            bootstrap_reader=lambda _descriptor: bootstrap,
            session_factory=lambda _bootstrap: completed,
            executor=lambda _invocation, _profiler, _observability, _controls, _phases: (
                pytest.fail("replayed completed work")
            ),
        )
        == 0
    )
    assert completed.finished == []


@pytest.mark.parametrize(
    "lifecycle_exit",
    [WorkerCancellationRequested, WorkerResourcesReleasedPause],
)
def test_runner_treats_acknowledged_lifecycle_exit_as_clean_process_exit(
    lifecycle_exit: type[RuntimeError],
) -> None:
    bootstrap = SimpleNamespace(run_id="run-1")
    session = FakeSession(bootstrap)

    def cancel_at_safe_point(*_args: object) -> HandlerResult:
        raise lifecycle_exit("operator requested")

    assert (
        run_worker(
            bootstrap_reader=lambda _descriptor: bootstrap,
            session_factory=lambda _bootstrap: session,
            executor=cancel_at_safe_point,
        )
        == 0
    )
    assert session.finished == []
    assert session.closed


def test_cli_accepts_only_the_authority_descriptor(monkeypatch) -> None:
    observed: list[int] = []
    monkeypatch.setattr(
        "rwkv_lab.trainvm_adapters.entrypoint.run_worker",
        lambda descriptor: observed.append(descriptor) or 0,
    )
    assert main(["--trainvm-bootstrap-fd=4"]) == 0
    assert observed == [4]
    for arguments in ([], ["--trainvm-bootstrap-fd=5"], ["--help"]):
        with pytest.raises(WorkerEntrypointError, match="only its fixed"):
            main(arguments)
