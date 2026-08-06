from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from rwkv_lab.training_components import (
    AssistantOnlyMapperConfiguration,
    BatchingImplementation,
    BucketedBatchingConfiguration,
    CausalTokensMapperConfiguration,
    CollatorImplementation,
    DataPipelineError,
    DataSourceImplementation,
    DeterministicHoldoutConfiguration,
    DeterministicSamplerConfiguration,
    FixedBatchingConfiguration,
    ImageCaptionProcessorConfiguration,
    JsonlImageCaptionConfiguration,
    JsonlTokenCorpusConfiguration,
    MappedSample,
    PackedTokenCollatorConfiguration,
    PaddedCollatorConfiguration,
    RegisteredBatching,
    RegisteredCollator,
    RegisteredDataSource,
    RegisteredSampleMapper,
    RegisteredSampleProcessor,
    RegisteredSampler,
    RegisteredSplitSelector,
    SampleMapperImplementation,
    SampleProcessorImplementation,
    SamplerImplementation,
    SplitSelectorImplementation,
    TokenIdsProcessorConfiguration,
    batching_from_resolved_component,
    build_data_pipeline,
    data_source_from_resolved_component,
)
from rwkv_lab.trainvm_adapters import WorkerTrainingComponents
from rwkv_lab.trainvm_worker import load_resolved_training_composition


def _fingerprint(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value: object) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest()
    )


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records)
    )


def _token_pipeline(manifest: Path, *, held_out_count: int = 1):
    source = RegisteredDataSource(
        DataSourceImplementation.JSONL_TOKEN_CORPUS_V1,
        JsonlTokenCorpusConfiguration(
            str(manifest),
            _fingerprint(manifest),
            ("id", "tokens"),
            "tokens",
            "id",
        ),
    )
    processor = RegisteredSampleProcessor(
        SampleProcessorImplementation.TOKEN_IDS_V1,
        TokenIdsProcessorConfiguration("tokens", 1, 32, 1024),
    )
    mapper = RegisteredSampleMapper(
        SampleMapperImplementation.CAUSAL_TOKENS_V1,
        CausalTokensMapperConfiguration("tokens", 32),
    )
    return build_data_pipeline(
        source=source,
        processor=processor,
        mapper=mapper,
        collator=RegisteredCollator(
            CollatorImplementation.PADDED_V1,
            PaddedCollatorConfiguration(0, -100, 4, 32),
        ),
        sampler=RegisteredSampler(
            SamplerImplementation.DETERMINISTIC_V1,
            DeterministicSamplerConfiguration(19, True),
        ),
        batching=RegisteredBatching(
            BatchingImplementation.FIXED_V1,
            FixedBatchingConfiguration(2, False, 3),
        ),
        split_selector=RegisteredSplitSelector(
            SplitSelectorImplementation.DETERMINISTIC_HOLDOUT_V1,
            DeterministicHoldoutConfiguration(27, held_out_count, "train"),
        ),
    )


def test_token_corpus_preflight_proves_target_and_batch_shape(tmp_path: Path) -> None:
    manifest = tmp_path / "tokens.jsonl"
    _write_jsonl(
        manifest,
        [
            {"id": f"sample-{index}", "tokens": list(range(1, index + 3))}
            for index in range(6)
        ],
    )
    pipeline = _token_pipeline(manifest)

    evidence = pipeline.preflight(sample_limit=6)
    repeated = pipeline.preflight(sample_limit=6)

    assert evidence.samples_checked == 6
    assert len(evidence.decoded_sample_ids) == 5
    assert evidence.supervised_token_counts == tuple(
        len(
            next(
                record["tokens"]
                for record in json.loads(
                    "[" + manifest.read_text().replace("\n", ",")[:-1] + "]"
                )
                if record["id"] == sample_id
            )
        )
        for sample_id in evidence.decoded_sample_ids
    )
    assert sum(rows for rows, _ in evidence.batch_shapes) == 5
    assert all(columns % 4 == 0 for _, columns in evidence.batch_shapes)
    assert evidence.evidence_digest.startswith("sha256:")
    assert repeated == evidence
    assert pipeline.sampler.component_state()["cursor"] == 0
    assert pipeline.batching.component_state()["batches_emitted"] == 0
    assert pipeline.batching.prefetch_workers == 3


def test_packed_token_collator_combines_short_samples_with_separator() -> None:
    collator = RegisteredCollator(
        CollatorImplementation.PACKED_TOKENS_V1,
        PackedTokenCollatorConfiguration(0, -100, 1, 8, 2),
    )
    batch = collator.collate(
        (
            MappedSample("first", (10, 11), (10, 11)),
            MappedSample("second", (20, 21, 22), (20, 21, 22)),
        ),
        tensor_output=False,
    )

    assert len(batch["sample_ids"]) == 1
    assert batch["sample_ids"][0].startswith("sha256:")
    assert batch["packed_sample_members"] == (("first", "second"),)
    assert batch["input_ids"] == [[10, 11, 2, 20, 21, 22, 0, 0]]
    assert batch["labels"] == [[10, 11, 2, 20, 21, 22, -100, -100]]
    assert batch["attention_mask"] == [[1, 1, 1, 1, 1, 1, 0, 0]]


class _Tokenizer:
    eos_token_id = 2

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        prefix = [1] if add_special_tokens else []
        return prefix + [3 + (byte % 101) for byte in text.encode()]


def test_image_caption_preflight_builds_assistant_only_targets(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    records: list[dict[str, object]] = []
    for index in range(4):
        name = f"{index}.png"
        Image.new("RGB", (48 + index, 40 + index), (index, 20, 30)).save(
            image_root / name
        )
        records.append(
            {"id": f"image-{index}", "image": name, "teacher": f"caption {index}"}
        )
    manifest = tmp_path / "images.jsonl"
    _write_jsonl(manifest, records)
    pipeline = build_data_pipeline(
        source=RegisteredDataSource(
            DataSourceImplementation.JSONL_IMAGE_CAPTION_V1,
            JsonlImageCaptionConfiguration(
                str(manifest),
                str(image_root),
                _fingerprint(manifest),
                ("id", "image", "teacher"),
                "image",
                ("teacher",),
                "id",
            ),
        ),
        processor=RegisteredSampleProcessor(
            SampleProcessorImplementation.IMAGE_CAPTION_V1,
            ImageCaptionProcessorConfiguration(
                "image", ("teacher",), 1024, 4096, 64, "downscale"
            ),
        ),
        mapper=RegisteredSampleMapper(
            SampleMapperImplementation.ASSISTANT_ONLY_V1,
            AssistantOnlyMapperConfiguration(
                "", "Describe this image.", "teacher", 256, True
            ),
        ),
        collator=RegisteredCollator(
            CollatorImplementation.PADDED_V1,
            PaddedCollatorConfiguration(0, -100, 8, 256),
        ),
        sampler=RegisteredSampler(
            SamplerImplementation.DETERMINISTIC_V1,
            DeterministicSamplerConfiguration(4, False),
        ),
        batching=RegisteredBatching(
            BatchingImplementation.BUCKETED_V1,
            BucketedBatchingConfiguration(
                "image_area", (2048, 4096), (2, 2, 1), False, 2
            ),
        ),
        split_selector=RegisteredSplitSelector(
            SplitSelectorImplementation.DETERMINISTIC_HOLDOUT_V1,
            DeterministicHoldoutConfiguration(8, 1, "train"),
        ),
    )

    evidence = pipeline.preflight(tokenizer=_Tokenizer(), sample_limit=4)

    assert evidence.samples_checked == 4
    assert len(evidence.decoded_sample_ids) == 3
    assert all(count > 0 for count in evidence.supervised_token_counts)
    assert sum(rows for rows, _ in evidence.batch_shapes) == 3
    assert evidence.split_membership_digest.startswith("sha256:")


def test_worker_bridge_consumes_qwen_caption_data_without_dataset_loop_code(
    tmp_path: Path,
) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    records = []
    for index in range(3):
        name = f"{index}.png"
        Image.new("RGB", (32, 32), (index, 10, 20)).save(image_root / name)
        records.append({"id": str(index), "image": name, "teacher": f"target {index}"})
    manifest = tmp_path / "caption-data.jsonl"
    _write_jsonl(manifest, records)
    registry = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "docs/experiment-vm/examples/training-components.v1.json"
        ).read_text()
    )
    requests = {
        "data": (
            "data_source",
            "jsonl_image_caption",
            {
                "caption_columns": ["teacher"],
                "content_fingerprint": _fingerprint(manifest),
                "declared_columns": ["id", "image", "teacher"],
                "id_column": "id",
                "image_column": "image",
                "image_root": str(image_root),
                "manifest_path": str(manifest),
            },
        ),
        "processor": (
            "sample_processor",
            "image_caption",
            {
                "caption_columns": ["teacher"],
                "image_column": "image",
                "maximum_edge": 64,
                "maximum_pixels": 4096,
                "minimum_pixels": 1,
                "oversize_policy": "reject",
            },
        ),
        "sample_mapping": (
            "sample_mapper",
            "assistant_only",
            {
                "append_eos": True,
                "fixed_prompt": "Describe this image.",
                "maximum_tokens": 128,
                "prompt_column": "",
                "target_column": "teacher",
            },
        ),
        "collation": (
            "collator",
            "padded",
            {
                "label_pad_token_id": -100,
                "maximum_sequence_length": 128,
                "pad_to_multiple": 8,
                "pad_token_id": 0,
            },
        ),
        "sampler": (
            "sampler",
            "deterministic",
            {"seed": 4, "shuffle": False},
        ),
        "batching": (
            "batching",
            "fixed",
            {"batch_size": 2, "drop_last": False, "prefetch_workers": 1},
        ),
        "split": (
            "split_selector",
            "deterministic_holdout",
            {"held_out_count": 1, "seed": 6, "selection": "train"},
        ),
    }
    components = {}
    for slot, (category, name, configuration) in requests.items():
        descriptor = next(
            item
            for item in registry["components"]
            if item["key"]["category"] == category and item["key"]["name"] == name
        )
        components[slot] = {
            "configuration": configuration,
            "descriptor": descriptor,
            "descriptor_digest": _digest(descriptor),
        }
    body = {
        "api_version": "trainvm.resolved-training-composition/v1",
        "components": components,
        "model_family": "transformer",
        "registry_digest": _digest(registry),
    }
    resolved = load_resolved_training_composition(
        {**body, "composition_digest": _digest(body)}
    )

    pipeline = WorkerTrainingComponents(resolved, "transformer").data_pipeline()
    evidence = pipeline.preflight(tokenizer=_Tokenizer(), sample_limit=3)

    assert len(evidence.decoded_sample_ids) == 2
    assert evidence.batch_shapes == ((2, 32),)


def test_schema_validation_rejects_missing_columns_and_modalities(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "tokens.jsonl"
    _write_jsonl(manifest, [{"id": "a", "tokens": [1]}, {"id": "b", "tokens": [2]}])
    pipeline = _token_pipeline(manifest)
    object.__setattr__(
        pipeline.mapper,
        "configuration",
        CausalTokensMapperConfiguration("not_declared", 8),
    )
    with pytest.raises(DataPipelineError, match="undeclared columns"):
        pipeline.validate_schema()
    object.__setattr__(
        pipeline.mapper,
        "configuration",
        CausalTokensMapperConfiguration("tokens", 8),
    )

    with pytest.raises(DataPipelineError, match="modalities are incompatible"):
        build_data_pipeline(
            source=pipeline.source,
            processor=RegisteredSampleProcessor(
                SampleProcessorImplementation.IMAGE_CAPTION_V1,
                ImageCaptionProcessorConfiguration(
                    "tokens", ("id",), 1, 32, 32, "reject"
                ),
            ),
            mapper=pipeline.mapper,
            collator=pipeline.collator,
            sampler=pipeline.sampler,
            batching=pipeline.batching,
            split_selector=pipeline.split_selector,
        )


def test_sampler_resume_restores_the_exact_next_sample() -> None:
    configuration = DeterministicSamplerConfiguration(1234, True)
    uninterrupted = RegisteredSampler(
        SamplerImplementation.DETERMINISTIC_V1, configuration
    )
    uninterrupted.bind(tuple(f"sample-{index}" for index in range(11)))
    first = uninterrupted.take(7)
    state = dict(uninterrupted.component_state())
    expected = uninterrupted.take(9)

    resumed = RegisteredSampler(SamplerImplementation.DETERMINISTIC_V1, configuration)
    resumed.bind(tuple(f"sample-{index}" for index in range(11)))
    resumed.restore_component_state(state)

    assert first
    assert resumed.take(9) == expected
    with pytest.raises(DataPipelineError, match="disagrees"):
        resumed.restore_component_state({**state, "order_digest": "sha256:" + "0" * 64})


def test_data_source_cursor_resumes_at_exact_jsonl_record(tmp_path: Path) -> None:
    manifest = tmp_path / "tokens.jsonl"
    _write_jsonl(
        manifest,
        [{"id": f"sample-{index}", "tokens": [index + 1]} for index in range(7)],
    )
    source = _token_pipeline(manifest).source
    assert tuple(sample.sample_id for sample in source.take(3)) == (
        "sample-0",
        "sample-1",
        "sample-2",
    )
    state = dict(source.component_state())

    resumed = _token_pipeline(manifest).source
    resumed.restore_component_state(state)

    assert tuple(sample.sample_id for sample in resumed.take(2)) == (
        "sample-3",
        "sample-4",
    )
    with pytest.raises(DataPipelineError, match="fingerprint disagrees"):
        resumed.restore_component_state(
            {**state, "content_fingerprint": "sha256:" + "0" * 64}
        )


def test_bucket_resume_preserves_pending_order_and_assignment() -> None:
    configuration = BucketedBatchingConfiguration(
        "token_length", (8, 16), (4, 2, 1), False, 0
    )
    original = RegisteredBatching(BatchingImplementation.BUCKETED_V1, configuration)
    assert original.add("a", measurement=5) is None
    assert original.add("b", measurement=11) is None
    state = dict(original.component_state())

    resumed = RegisteredBatching(BatchingImplementation.BUCKETED_V1, configuration)
    resumed.restore_component_state(state, measurements={"a": 5, "b": 11})

    assert dict(resumed.component_state()) == state
    assert resumed.add("c", measurement=12) == ("b", "c")


def test_resolved_factories_reject_wrong_categories_and_preserve_versioned_ids(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "tokens.jsonl"
    _write_jsonl(manifest, [{"id": "a", "tokens": [1]}])
    descriptor = {
        "key": {
            "category": "data_source",
            "name": "jsonl_token_corpus",
            "version": "1.0.0",
        },
        "implementation": DataSourceImplementation.JSONL_TOKEN_CORPUS_V1.value,
    }
    source = data_source_from_resolved_component(
        {
            "configuration": {
                "manifest_path": str(manifest),
                "content_fingerprint": _fingerprint(manifest),
                "declared_columns": ("id", "tokens"),
                "token_column": "tokens",
                "id_column": "id",
            },
            "descriptor": descriptor,
            "descriptor_digest": "sha256:" + "a" * 64,
        }
    )
    assert source.implementation is DataSourceImplementation.JSONL_TOKEN_CORPUS_V1
    with pytest.raises(DataPipelineError, match="not a batching"):
        batching_from_resolved_component(
            {
                "configuration": {},
                "descriptor": descriptor,
                "descriptor_digest": "sha256:" + "a" * 64,
            }
        )


def test_preflight_is_bounded_and_detects_content_drift(tmp_path: Path) -> None:
    manifest = tmp_path / "tokens.jsonl"
    _write_jsonl(
        manifest, [{"id": str(index), "tokens": [index + 1]} for index in range(5)]
    )
    pipeline = _token_pipeline(manifest)
    manifest.write_text(
        manifest.read_text() + json.dumps({"id": "late", "tokens": [8]}) + "\n"
    )
    with pytest.raises(DataPipelineError, match="fingerprint disagrees"):
        pipeline.preflight(sample_limit=4)
    with pytest.raises(DataPipelineError, match="between 2 and 256"):
        pipeline.preflight(sample_limit=257)
