from __future__ import annotations

import hashlib
import json
from collections import defaultdict, deque
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, TypeAlias

_DIGEST_PREFIX = "sha256:"
_MAXIMUM_COLUMNS = 64
_MAXIMUM_COLUMN_BYTES = 256
_MAXIMUM_PREFLIGHT_SAMPLES = 256


class DataPipelineError(ValueError):
    """A declarative data contract could not be honored exactly."""


class DataSourceImplementation(str, Enum):
    JSONL_IMAGE_CAPTION_V1 = "rwkv_lab.data_source.jsonl_image_caption.v1"
    JSONL_FROZEN_IMAGE_SPLITS_V1 = (
        "rwkv_lab.data_source.jsonl_frozen_image_splits.v1"
    )
    JSONL_TOKEN_CORPUS_V1 = "rwkv_lab.data_source.jsonl_token_corpus.v1"
    JSONL_FROZEN_TOKEN_SPLITS_V1 = (
        "rwkv_lab.data_source.jsonl_frozen_token_splits.v1"
    )


class SampleProcessorImplementation(str, Enum):
    IMAGE_CAPTION_V1 = "rwkv_lab.sample_processor.image_caption.v1"
    TOKEN_IDS_V1 = "rwkv_lab.sample_processor.token_ids.v1"


class SampleMapperImplementation(str, Enum):
    ASSISTANT_ONLY_V1 = "rwkv_lab.sample_mapper.assistant_only.v1"
    ASSISTANT_CONVERSATION_V2 = (
        "rwkv_lab.sample_mapper.assistant_conversation.v2"
    )
    CAUSAL_TOKENS_V1 = "rwkv_lab.sample_mapper.causal_tokens.v1"


class CollatorImplementation(str, Enum):
    PADDED_V1 = "rwkv_lab.collator.padded.v1"
    PACKED_TOKENS_V1 = "rwkv_lab.collator.packed_tokens.v1"


class SamplerImplementation(str, Enum):
    DETERMINISTIC_V1 = "rwkv_lab.sampler.deterministic.v1"


class BatchingImplementation(str, Enum):
    FIXED_V1 = "rwkv_lab.batching.fixed.v1"
    BUCKETED_V1 = "rwkv_lab.batching.bucketed.v1"


class SplitSelectorImplementation(str, Enum):
    DETERMINISTIC_HOLDOUT_V1 = "rwkv_lab.split_selector.deterministic_holdout.v1"
    FROZEN_NAMED_V1 = "rwkv_lab.split_selector.frozen_named.v1"


def _digest_bytes(value: bytes) -> str:
    return _DIGEST_PREFIX + hashlib.sha256(value).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return _DIGEST_PREFIX + digest.hexdigest()


def _canonical_digest(value: Any) -> str:
    return _digest_bytes(
        json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )


def _require_digest(value: Any, label: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 71
        and value.startswith(_DIGEST_PREFIX)
        and all(character in "0123456789abcdef" for character in value[7:])
    ):
        raise DataPipelineError(f"{label} must be a lowercase sha256 digest")
    return value


def _require_path(value: Any, label: str, *, directory: bool = False) -> Path:
    path = Path(value) if isinstance(value, str) else Path()
    if not path.is_absolute() or not path.exists():
        raise DataPipelineError(f"{label} must name an existing absolute asset")
    if directory and not path.is_dir():
        raise DataPipelineError(f"{label} must name a directory")
    if not directory and not path.is_file():
        raise DataPipelineError(f"{label} must name a file")
    return path


def _column(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or len(value.encode("utf-8")) > _MAXIMUM_COLUMN_BYTES:
        raise DataPipelineError(f"{label} is not a bounded column name")
    if not value and not allow_empty:
        raise DataPipelineError(f"{label} must not be empty")
    if "\x00" in value or "\n" in value or "\r" in value:
        raise DataPipelineError(f"{label} contains a forbidden character")
    return value


def _columns(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= _MAXIMUM_COLUMNS:
        raise DataPipelineError(f"{label} must be a bounded nonempty column list")
    columns = tuple(_column(item, label) for item in value)
    if tuple(sorted(set(columns))) != columns:
        raise DataPipelineError(f"{label} must be a canonical sorted column set")
    return columns


def _ordered_columns(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= _MAXIMUM_COLUMNS:
        raise DataPipelineError(f"{label} must be a bounded nonempty column list")
    columns = tuple(_column(item, label) for item in value)
    if len(set(columns)) != len(columns):
        raise DataPipelineError(f"{label} must not contain duplicate columns")
    return columns


def _exact(value: Mapping[str, Any], fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise DataPipelineError(f"resolved {label} configuration is inexact")
    return dict(value)


@dataclass(frozen=True, slots=True)
class JsonlImageCaptionConfiguration:
    manifest_path: str
    image_root: str
    content_fingerprint: str
    declared_columns: tuple[str, ...]
    image_column: str
    caption_columns: tuple[str, ...]
    id_column: str

    def __post_init__(self) -> None:
        _require_path(self.manifest_path, "manifest_path")
        _require_path(self.image_root, "image_root", directory=True)
        _require_digest(self.content_fingerprint, "content_fingerprint")
        declared = _columns(self.declared_columns, "declared_columns")
        image = _column(self.image_column, "image_column")
        captions = _ordered_columns(self.caption_columns, "caption_columns")
        identifier = _column(self.id_column, "id_column", allow_empty=True)
        required = {image, *captions, *([identifier] if identifier else [])}
        if not required.issubset(declared):
            raise DataPipelineError(
                "image-caption columns are absent from declared_columns"
            )

    @classmethod
    def from_resolved(cls, value: Mapping[str, Any]) -> JsonlImageCaptionConfiguration:
        value = _exact(
            value,
            {
                "manifest_path",
                "image_root",
                "content_fingerprint",
                "declared_columns",
                "image_column",
                "caption_columns",
                "id_column",
            },
            "JSONL image-caption source",
        )
        return cls(
            manifest_path=value["manifest_path"],
            image_root=value["image_root"],
            content_fingerprint=value["content_fingerprint"],
            declared_columns=_columns(value["declared_columns"], "declared_columns"),
            image_column=value["image_column"],
            caption_columns=_ordered_columns(
                value["caption_columns"], "caption_columns"
            ),
            id_column=value["id_column"],
        )


@dataclass(frozen=True, slots=True)
class JsonlTokenCorpusConfiguration:
    manifest_path: str
    content_fingerprint: str
    declared_columns: tuple[str, ...]
    token_column: str
    id_column: str

    def __post_init__(self) -> None:
        _require_path(self.manifest_path, "manifest_path")
        _require_digest(self.content_fingerprint, "content_fingerprint")
        declared = _columns(self.declared_columns, "declared_columns")
        token = _column(self.token_column, "token_column")
        identifier = _column(self.id_column, "id_column", allow_empty=True)
        required = {token, *([identifier] if identifier else [])}
        if not required.issubset(declared):
            raise DataPipelineError(
                "token-corpus columns are absent from declared_columns"
            )

    @classmethod
    def from_resolved(cls, value: Mapping[str, Any]) -> JsonlTokenCorpusConfiguration:
        value = _exact(
            value,
            {
                "manifest_path",
                "content_fingerprint",
                "declared_columns",
                "token_column",
                "id_column",
            },
            "JSONL token-corpus source",
        )
        return cls(
            manifest_path=value["manifest_path"],
            content_fingerprint=value["content_fingerprint"],
            declared_columns=_columns(value["declared_columns"], "declared_columns"),
            token_column=value["token_column"],
            id_column=value["id_column"],
        )


@dataclass(frozen=True, slots=True)
class JsonlFrozenImageSplitsConfiguration:
    dataset_root: str
    content_fingerprint: str
    declared_columns: tuple[str, ...]
    image_column: str
    caption_columns: tuple[str, ...]
    id_column: str

    def __post_init__(self) -> None:
        root = _require_path(self.dataset_root, "dataset_root", directory=True)
        _require_digest(self.content_fingerprint, "content_fingerprint")
        for name in ("manifest.json", "train.jsonl", "validation.jsonl", "test.jsonl"):
            _require_path(str(root / name), name)
        declared = _columns(self.declared_columns, "declared_columns")
        image = _column(self.image_column, "image_column")
        captions = _ordered_columns(self.caption_columns, "caption_columns")
        identifier = _column(self.id_column, "id_column")
        if not {image, identifier, *captions}.issubset(declared):
            raise DataPipelineError(
                "frozen image-split columns are absent from declared_columns"
            )

    @classmethod
    def from_resolved(
        cls, value: Mapping[str, Any]
    ) -> JsonlFrozenImageSplitsConfiguration:
        value = _exact(
            value,
            {
                "dataset_root",
                "content_fingerprint",
                "declared_columns",
                "image_column",
                "caption_columns",
                "id_column",
            },
            "JSONL frozen image-splits source",
        )
        return cls(
            **{
                **value,
                "declared_columns": _columns(
                    value["declared_columns"], "declared_columns"
                ),
                "caption_columns": _ordered_columns(
                    value["caption_columns"], "caption_columns"
                ),
            }
        )

    @property
    def image_root(self) -> str:
        return self.dataset_root

    @property
    def manifest_path(self) -> str:
        return str(Path(self.dataset_root) / "train.jsonl")

    @property
    def split_manifest(self) -> str:
        return str(Path(self.dataset_root) / "manifest.json")


@dataclass(frozen=True, slots=True)
class JsonlFrozenTokenSplitsConfiguration:
    dataset_root: str
    content_fingerprint: str
    declared_columns: tuple[str, ...]
    token_column: str
    id_column: str

    def __post_init__(self) -> None:
        root = _require_path(self.dataset_root, "dataset_root", directory=True)
        _require_digest(self.content_fingerprint, "content_fingerprint")
        for name in ("manifest.json", "train.jsonl", "validation.jsonl", "test.jsonl"):
            _require_path(str(root / name), name)
        declared = _columns(self.declared_columns, "declared_columns")
        token = _column(self.token_column, "token_column")
        identifier = _column(self.id_column, "id_column")
        if not {identifier, token}.issubset(declared):
            raise DataPipelineError(
                "frozen token-split columns are absent from declared_columns"
            )

    @classmethod
    def from_resolved(
        cls, value: Mapping[str, Any]
    ) -> JsonlFrozenTokenSplitsConfiguration:
        value = _exact(
            value,
            {
                "dataset_root",
                "content_fingerprint",
                "declared_columns",
                "token_column",
                "id_column",
            },
            "JSONL frozen token-splits source",
        )
        return cls(
            **{
                **value,
                "declared_columns": _columns(
                    value["declared_columns"], "declared_columns"
                ),
            }
        )

    @property
    def manifest_path(self) -> str:
        return str(Path(self.dataset_root) / "train.jsonl")

    @property
    def split_manifest(self) -> str:
        return str(Path(self.dataset_root) / "manifest.json")


DataSourceConfiguration: TypeAlias = (
    JsonlFrozenImageSplitsConfiguration
    | JsonlFrozenTokenSplitsConfiguration
    | JsonlImageCaptionConfiguration
    | JsonlTokenCorpusConfiguration
)


FrozenSplitConfiguration: TypeAlias = (
    JsonlFrozenImageSplitsConfiguration | JsonlFrozenTokenSplitsConfiguration
)


@dataclass(frozen=True, slots=True)
class RawSample:
    sample_id: str
    ordinal: int
    values: Mapping[str, Any]


@dataclass(slots=True)
class RegisteredDataSource:
    implementation: DataSourceImplementation
    configuration: DataSourceConfiguration
    _cursor: int = 0

    @property
    def declared_columns(self) -> tuple[str, ...]:
        return self.configuration.declared_columns

    def verify_content(
        self, *, authority_content_fingerprint: str | None = None
    ) -> None:
        if isinstance(self.configuration, FrozenSplitConfiguration):
            configuration = self.configuration
            if authority_content_fingerprint != configuration.content_fingerprint:
                raise DataPipelineError(
                    "frozen dataset fingerprint lacks matching workspace authority"
                )
            with Path(configuration.split_manifest).open(
                "r", encoding="utf-8"
            ) as handle:
                receipt = json.load(handle)
            counts = receipt.get("counts") if isinstance(receipt, dict) else None
            files = receipt.get("files") if isinstance(receipt, dict) else None
            if (
                not isinstance(receipt.get("schema"), str)
                or not isinstance(receipt.get("dataset_digest"), str)
                or not isinstance(counts, dict)
                or set(counts) != {"train", "validation", "test"}
                or not isinstance(files, dict)
            ):
                raise DataPipelineError("frozen split receipt is incomplete")
            seen_ids: set[str] = set()
            total_rows = 0
            for split, path in self._split_paths().items():
                entry = files.get(path.name)
                if (
                    not isinstance(entry, dict)
                    or set(entry) != {"rows", "sha256"}
                    or entry["rows"] != counts[split]
                    or _file_digest(path) != "sha256:" + entry["sha256"]
                ):
                    raise DataPipelineError(
                        f"frozen {split} manifest disagrees with its receipt"
                    )
                rows = 0
                with path.open("r", encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError as error:
                            raise DataPipelineError(
                                f"malformed frozen {split} row {line_number}"
                            ) from error
                        if not isinstance(row, dict) or row.get("split") != split:
                            raise DataPipelineError(
                                f"frozen {split} row has wrong split identity"
                            )
                        sample_id = row.get(configuration.id_column)
                        if (
                            not isinstance(sample_id, str)
                            or not sample_id
                            or sample_id in seen_ids
                        ):
                            raise DataPipelineError(
                                "frozen manifests contain an empty or duplicate ID"
                            )
                        if isinstance(
                            configuration, JsonlFrozenImageSplitsConfiguration
                        ):
                            caption = row.get(configuration.caption_columns[0])
                            image_value = row.get(configuration.image_column)
                            if not isinstance(caption, str) or not caption.strip():
                                raise DataPipelineError(
                                    "frozen manifest contains an empty caption"
                                )
                            if not isinstance(image_value, str) or not image_value:
                                raise DataPipelineError(
                                    "frozen manifest contains an invalid image path"
                                )
                            image = Path(image_value)
                            if not image.is_absolute():
                                image = Path(configuration.dataset_root) / image
                            try:
                                image.resolve(strict=True).relative_to(
                                    Path(configuration.dataset_root).resolve(strict=True)
                                )
                            except (OSError, ValueError) as error:
                                raise DataPipelineError(
                                    "frozen manifest image is absent or outside dataset root"
                                ) from error
                            if not image.is_file():
                                raise DataPipelineError(
                                    "frozen manifest image is not a file"
                                )
                        else:
                            tokens = row.get(configuration.token_column)
                            if not isinstance(tokens, list) or not tokens or any(
                                not isinstance(token, int)
                                or isinstance(token, bool)
                                or token < 0
                                for token in tokens
                            ):
                                raise DataPipelineError(
                                    "frozen manifest contains invalid token IDs"
                                )
                        seen_ids.add(sample_id)
                        rows += 1
                if rows != counts[split] or rows != entry["rows"]:
                    raise DataPipelineError(
                        f"frozen {split} row count disagrees with its receipt"
                    )
                total_rows += rows
            unique_content = receipt.get("unique_content_hashes")
            if isinstance(unique_content, int) and unique_content != total_rows:
                raise DataPipelineError(
                    "frozen dataset unique-content receipt disagrees with split rows"
                )
            return
        path = Path(self.configuration.manifest_path)
        actual = _file_digest(path)
        if actual != self.configuration.content_fingerprint:
            raise DataPipelineError("data-source content fingerprint disagrees")

    def _split_paths(self) -> Mapping[str, Path]:
        configuration = self.configuration
        if not isinstance(configuration, FrozenSplitConfiguration):
            raise DataPipelineError("data source has no frozen named splits")
        return MappingProxyType(
            {
                "train": Path(configuration.dataset_root) / "train.jsonl",
                "validation": Path(configuration.dataset_root) / "validation.jsonl",
                "test": Path(configuration.dataset_root) / "test.jsonl",
            }
        )

    def records(
        self,
        *,
        start_cursor: int = 0,
        limit: int | None = None,
        split: str = "train",
    ) -> Iterator[RawSample]:
        if not isinstance(start_cursor, int) or start_cursor < 0:
            raise DataPipelineError("data-source cursor must be a nonnegative integer")
        if limit is not None and (not isinstance(limit, int) or limit < 0):
            raise DataPipelineError("data-source limit must be a nonnegative integer")
        configuration = self.configuration
        path = (
            self._split_paths().get(split)
            if isinstance(configuration, FrozenSplitConfiguration)
            else Path(configuration.manifest_path)
        )
        if path is None:
            raise DataPipelineError("frozen split name is invalid")
        emitted = 0
        with path.open("r", encoding="utf-8") as handle:
            for ordinal, line in enumerate(handle):
                if ordinal < start_cursor:
                    continue
                if limit is not None and emitted >= limit:
                    return
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise DataPipelineError(
                        f"malformed JSONL record at line {ordinal + 1}"
                    ) from error
                if not isinstance(value, dict):
                    raise DataPipelineError(
                        f"JSONL record at line {ordinal + 1} is not an object"
                    )
                missing = set(configuration.declared_columns).difference(value)
                if missing:
                    raise DataPipelineError(
                        f"JSONL record at line {ordinal + 1} lacks declared columns: "
                        + ", ".join(sorted(missing))
                    )
                id_column = configuration.id_column
                sample_id = str(value[id_column]) if id_column else f"line:{ordinal}"
                if not sample_id:
                    raise DataPipelineError("data source produced an empty sample id")
                if isinstance(configuration, FrozenSplitConfiguration) and value.get(
                    "split"
                ) != split:
                    raise DataPipelineError(
                        f"frozen {split} manifest contains a cross-split row"
                    )
                yield RawSample(
                    sample_id=sample_id,
                    ordinal=ordinal,
                    values=MappingProxyType(value),
                )
                emitted += 1

    def records_for_split(self, split: str) -> tuple[RawSample, ...]:
        if not isinstance(self.configuration, FrozenSplitConfiguration):
            raise DataPipelineError("data source has no frozen named splits")
        return tuple(self.records(split=split))

    def take(self, count: int) -> tuple[RawSample, ...]:
        """Read and advance the exact streaming cursor.

        Epoch reshuffling remains the sampler's responsibility; this cursor is
        the source-line boundary needed to resume a partially consumed stream.
        """

        if not isinstance(count, int) or count < 1:
            raise DataPipelineError("data-source take count must be positive")
        records = tuple(self.records(start_cursor=self._cursor, limit=count))
        self._cursor += len(records)
        return records

    def component_state(self) -> Mapping[str, int | str]:
        return MappingProxyType(
            {
                "content_fingerprint": self.configuration.content_fingerprint,
                "cursor": self._cursor,
            }
        )

    def restore_component_state(self, state: Mapping[str, Any]) -> None:
        state = _exact(
            dict(state),
            {"content_fingerprint", "cursor"},
            "data-source resume state",
        )
        if state["content_fingerprint"] != self.configuration.content_fingerprint:
            raise DataPipelineError("data-source resume fingerprint disagrees")
        cursor = state["cursor"]
        if not isinstance(cursor, int) or cursor < 0:
            raise DataPipelineError("data-source resume cursor is invalid")
        self._cursor = cursor


@dataclass(frozen=True, slots=True)
class ImageCaptionProcessorConfiguration:
    image_column: str
    caption_columns: tuple[str, ...]
    minimum_pixels: int
    maximum_pixels: int
    maximum_edge: int
    oversize_policy: str

    def __post_init__(self) -> None:
        _column(self.image_column, "image_column")
        _ordered_columns(self.caption_columns, "caption_columns")
        if not 1 <= self.minimum_pixels <= self.maximum_pixels <= 268_435_456:
            raise DataPipelineError("image pixel bounds are invalid")
        if not 1 <= self.maximum_edge <= 65536:
            raise DataPipelineError("maximum_edge is invalid")
        if self.oversize_policy not in {"reject", "downscale"}:
            raise DataPipelineError("oversize_policy is invalid")

    @classmethod
    def from_resolved(
        cls, value: Mapping[str, Any]
    ) -> ImageCaptionProcessorConfiguration:
        value = _exact(
            value,
            {
                "image_column",
                "caption_columns",
                "minimum_pixels",
                "maximum_pixels",
                "maximum_edge",
                "oversize_policy",
            },
            "image-caption processor",
        )
        return cls(
            image_column=value["image_column"],
            caption_columns=_ordered_columns(
                value["caption_columns"], "caption_columns"
            ),
            minimum_pixels=value["minimum_pixels"],
            maximum_pixels=value["maximum_pixels"],
            maximum_edge=value["maximum_edge"],
            oversize_policy=value["oversize_policy"],
        )


@dataclass(frozen=True, slots=True)
class TokenIdsProcessorConfiguration:
    token_column: str
    minimum_tokens: int
    maximum_tokens: int
    vocabulary_size: int

    def __post_init__(self) -> None:
        _column(self.token_column, "token_column")
        if not 1 <= self.minimum_tokens <= self.maximum_tokens <= 1_048_576:
            raise DataPipelineError("token-count bounds are invalid")
        if not 2 <= self.vocabulary_size <= 16_777_216:
            raise DataPipelineError("vocabulary_size is invalid")

    @classmethod
    def from_resolved(cls, value: Mapping[str, Any]) -> TokenIdsProcessorConfiguration:
        value = _exact(
            value,
            {"token_column", "minimum_tokens", "maximum_tokens", "vocabulary_size"},
            "token-ids processor",
        )
        return cls(**value)


SampleProcessorConfiguration: TypeAlias = (
    ImageCaptionProcessorConfiguration | TokenIdsProcessorConfiguration
)


@dataclass(frozen=True, slots=True)
class ProcessedSample:
    sample_id: str
    ordinal: int
    values: Mapping[str, Any]
    image: Any | None = None
    image_size: tuple[int, int] | None = None
    token_length: int | None = None


@dataclass(frozen=True, slots=True)
class RegisteredSampleProcessor:
    implementation: SampleProcessorImplementation
    configuration: SampleProcessorConfiguration

    @property
    def required_columns(self) -> frozenset[str]:
        configuration = self.configuration
        if isinstance(configuration, ImageCaptionProcessorConfiguration):
            return frozenset(
                {configuration.image_column, *configuration.caption_columns}
            )
        return frozenset({configuration.token_column})

    def process(
        self, sample: RawSample, *, image_root: Path | None = None
    ) -> ProcessedSample:
        missing = self.required_columns.difference(sample.values)
        if missing:
            raise DataPipelineError(
                f"sample {sample.sample_id!r} lacks processor columns: "
                + ", ".join(sorted(missing))
            )
        configuration = self.configuration
        if isinstance(configuration, TokenIdsProcessorConfiguration):
            tokens = sample.values[configuration.token_column]
            if not isinstance(tokens, list) or any(
                not isinstance(token, int)
                or isinstance(token, bool)
                or not 0 <= token < configuration.vocabulary_size
                for token in tokens
            ):
                raise DataPipelineError(
                    f"sample {sample.sample_id!r} has invalid token ids"
                )
            if (
                not configuration.minimum_tokens
                <= len(tokens)
                <= configuration.maximum_tokens
            ):
                raise DataPipelineError(
                    f"sample {sample.sample_id!r} violates token-count bounds"
                )
            return ProcessedSample(
                sample_id=sample.sample_id,
                ordinal=sample.ordinal,
                values=sample.values,
                token_length=len(tokens),
            )

        if image_root is None:
            raise DataPipelineError("image-caption processing requires an image root")
        relative = sample.values[configuration.image_column]
        if not isinstance(relative, str) or not relative:
            raise DataPipelineError(
                f"sample {sample.sample_id!r} has an invalid image path"
            )
        image_path = (image_root / relative).resolve()
        try:
            image_path.relative_to(image_root.resolve())
        except ValueError as error:
            raise DataPipelineError(
                "image path escapes the declared image root"
            ) from error
        if not image_path.is_file():
            raise DataPipelineError(f"source image does not exist: {image_path}")
        for caption_column in configuration.caption_columns:
            caption = sample.values[caption_column]
            if not isinstance(caption, str) or not caption.strip():
                raise DataPipelineError(
                    f"sample {sample.sample_id!r} has an empty {caption_column!r} caption"
                )
        from PIL import Image

        with Image.open(image_path) as opened:
            opened.load()
            image = opened.convert("RGB")
        width, height = image.size
        pixels = width * height
        too_large = (
            pixels > configuration.maximum_pixels
            or max(width, height) > configuration.maximum_edge
        )
        if pixels < configuration.minimum_pixels:
            raise DataPipelineError(
                f"sample {sample.sample_id!r} is below image bounds"
            )
        if too_large and configuration.oversize_policy == "reject":
            raise DataPipelineError(f"sample {sample.sample_id!r} exceeds image bounds")
        if too_large:
            scale = min(
                (configuration.maximum_pixels / pixels) ** 0.5,
                configuration.maximum_edge / width,
                configuration.maximum_edge / height,
            )
            resized = (max(1, int(width * scale)), max(1, int(height * scale)))
            image = image.resize(resized, Image.Resampling.LANCZOS)
            width, height = resized
        return ProcessedSample(
            sample_id=sample.sample_id,
            ordinal=sample.ordinal,
            values=sample.values,
            image=image,
            image_size=(width, height),
        )


@dataclass(frozen=True, slots=True)
class AssistantOnlyMapperConfiguration:
    prompt_column: str
    fixed_prompt: str
    target_column: str
    maximum_tokens: int
    append_eos: bool

    def __post_init__(self) -> None:
        _column(self.prompt_column, "prompt_column", allow_empty=True)
        _column(self.target_column, "target_column")
        if (
            not isinstance(self.fixed_prompt, str)
            or len(self.fixed_prompt.encode()) > 4096
        ):
            raise DataPipelineError("fixed_prompt must be bounded text")
        if bool(self.prompt_column) == bool(self.fixed_prompt):
            raise DataPipelineError(
                "select exactly one of prompt_column and fixed_prompt"
            )
        if not 1 <= self.maximum_tokens <= 1_048_576:
            raise DataPipelineError("maximum_tokens is invalid")
        if not isinstance(self.append_eos, bool):
            raise DataPipelineError("append_eos must be boolean")

    @classmethod
    def from_resolved(
        cls, value: Mapping[str, Any]
    ) -> AssistantOnlyMapperConfiguration:
        return cls(
            **_exact(
                value,
                {
                    "prompt_column",
                    "fixed_prompt",
                    "target_column",
                    "maximum_tokens",
                    "append_eos",
                },
                "assistant-only mapper",
            )
        )


@dataclass(frozen=True, slots=True)
class AssistantConversationMapperConfiguration:
    system_prompt: str
    user_prompt: str
    target_column: str
    maximum_tokens: int
    append_eos: bool

    def __post_init__(self) -> None:
        _column(self.target_column, "target_column")
        for value, label in (
            (self.system_prompt, "system_prompt"),
            (self.user_prompt, "user_prompt"),
        ):
            if (
                not isinstance(value, str)
                or not value.strip()
                or len(value.encode("utf-8")) > 16_384
            ):
                raise DataPipelineError(f"{label} must be bounded nonempty text")
        if not 1 <= self.maximum_tokens <= 1_048_576:
            raise DataPipelineError("maximum_tokens is invalid")
        if not isinstance(self.append_eos, bool):
            raise DataPipelineError("append_eos must be boolean")

    @classmethod
    def from_resolved(
        cls, value: Mapping[str, Any]
    ) -> AssistantConversationMapperConfiguration:
        return cls(
            **_exact(
                value,
                {
                    "system_prompt",
                    "user_prompt",
                    "target_column",
                    "maximum_tokens",
                    "append_eos",
                },
                "assistant-conversation mapper",
            )
        )


@dataclass(frozen=True, slots=True)
class CausalTokensMapperConfiguration:
    token_column: str
    maximum_tokens: int

    def __post_init__(self) -> None:
        _column(self.token_column, "token_column")
        if not 1 <= self.maximum_tokens <= 1_048_576:
            raise DataPipelineError("maximum_tokens is invalid")

    @classmethod
    def from_resolved(cls, value: Mapping[str, Any]) -> CausalTokensMapperConfiguration:
        return cls(
            **_exact(value, {"token_column", "maximum_tokens"}, "causal-token mapper")
        )


SampleMapperConfiguration: TypeAlias = (
    AssistantConversationMapperConfiguration
    | AssistantOnlyMapperConfiguration
    | CausalTokensMapperConfiguration
)


class TokenizerProtocol(Protocol):
    eos_token_id: int | None

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...


def _tokenizer_backend(value: Any) -> TokenizerProtocol:
    backend = getattr(value, "tokenizer", value)
    if not callable(getattr(backend, "encode", None)):
        raise DataPipelineError("assistant-only mapper received no tokenizer backend")
    return backend


@dataclass(frozen=True, slots=True)
class MappedSample:
    sample_id: str
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    image: Any | None = None
    image_size: tuple[int, int] | None = None

    @property
    def token_length(self) -> int:
        return len(self.input_ids)


@dataclass(frozen=True, slots=True)
class RegisteredSampleMapper:
    implementation: SampleMapperImplementation
    configuration: SampleMapperConfiguration

    @property
    def required_columns(self) -> frozenset[str]:
        configuration = self.configuration
        if isinstance(configuration, CausalTokensMapperConfiguration):
            return frozenset({configuration.token_column})
        if isinstance(configuration, AssistantConversationMapperConfiguration):
            return frozenset({configuration.target_column})
        return frozenset(
            {
                configuration.target_column,
                *([configuration.prompt_column] if configuration.prompt_column else []),
            }
        )

    def map(
        self, sample: ProcessedSample, *, tokenizer: TokenizerProtocol | None = None
    ) -> MappedSample:
        missing = self.required_columns.difference(sample.values)
        if missing:
            raise DataPipelineError(
                f"sample {sample.sample_id!r} lacks mapper columns: "
                + ", ".join(sorted(missing))
            )
        configuration = self.configuration
        if isinstance(configuration, CausalTokensMapperConfiguration):
            tokens = tuple(sample.values[configuration.token_column])[
                : configuration.maximum_tokens
            ]
            if not tokens:
                raise DataPipelineError("causal-token target is empty")
            return MappedSample(
                sample.sample_id, tokens, tokens, sample.image, sample.image_size
            )
        if tokenizer is None:
            raise DataPipelineError("assistant-only mapping requires a tokenizer")
        if isinstance(configuration, AssistantConversationMapperConfiguration):
            render = getattr(tokenizer, "apply_chat_template", None)
            if not callable(render):
                raise DataPipelineError(
                    "assistant-conversation mapping requires apply_chat_template"
                )
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": configuration.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": configuration.user_prompt},
                    ],
                },
            ]
            target = sample.values[configuration.target_column]
            if not isinstance(target, str) or not target.strip():
                raise DataPipelineError(
                    "assistant-conversation target must be nonempty text"
                )
            prompt = render(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            full = render(
                [*messages, {"role": "assistant", "content": target}],
                tokenize=False,
                add_generation_prompt=False,
            )
            if (
                not isinstance(prompt, str)
                or not isinstance(full, str)
                or not full.startswith(prompt)
            ):
                raise DataPipelineError(
                    "assistant-conversation template has no stable target boundary"
                )
            backend = _tokenizer_backend(tokenizer)
            prompt_ids = list(backend.encode(prompt, add_special_tokens=False))
            full_ids = list(backend.encode(full, add_special_tokens=False))
            if full_ids[: len(prompt_ids)] != prompt_ids:
                raise DataPipelineError(
                    "assistant-conversation tokens have no stable target boundary"
                )
            input_ids = full_ids[: configuration.maximum_tokens]
            supervised = max(0, len(input_ids) - len(prompt_ids))
            labels = (
                [-100] * (len(input_ids) - supervised) + input_ids[-supervised:]
                if supervised
                else [-100] * len(input_ids)
            )
            if not input_ids or not any(label != -100 for label in labels):
                raise DataPipelineError(
                    "assistant-conversation mapping produced no supervised tokens"
                )
            if configuration.append_eos:
                eos = backend.eos_token_id
                if not isinstance(eos, int) or input_ids[-1] != eos:
                    raise DataPipelineError(
                        "assistant-conversation template must append exact EOS"
                    )
            return MappedSample(
                sample.sample_id,
                tuple(input_ids),
                tuple(labels),
                sample.image,
                sample.image_size,
            )
        tokenizer = _tokenizer_backend(tokenizer)
        prompt = (
            sample.values[configuration.prompt_column]
            if configuration.prompt_column
            else configuration.fixed_prompt
        )
        target = sample.values[configuration.target_column]
        if (
            not isinstance(prompt, str)
            or not isinstance(target, str)
            or not target.strip()
        ):
            raise DataPipelineError("assistant-only prompt and target must be text")
        prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=True))
        target_ids = list(tokenizer.encode(target, add_special_tokens=False))
        if configuration.append_eos:
            eos = tokenizer.eos_token_id
            if not isinstance(eos, int):
                raise DataPipelineError("append_eos requires an integer eos_token_id")
            target_ids.append(eos)
        input_ids = (prompt_ids + target_ids)[: configuration.maximum_tokens]
        supervised = max(0, len(input_ids) - len(prompt_ids))
        labels = (
            [-100] * (len(input_ids) - supervised) + input_ids[-supervised:]
            if supervised
            else [-100] * len(input_ids)
        )
        if not input_ids or not any(label != -100 for label in labels):
            raise DataPipelineError(
                "assistant-only mapping produced no supervised tokens"
            )
        return MappedSample(
            sample.sample_id,
            tuple(input_ids),
            tuple(labels),
            sample.image,
            sample.image_size,
        )


@dataclass(frozen=True, slots=True)
class PaddedCollatorConfiguration:
    pad_token_id: int
    label_pad_token_id: int
    pad_to_multiple: int
    maximum_sequence_length: int

    def __post_init__(self) -> None:
        if not 0 <= self.pad_token_id <= 16_777_215:
            raise DataPipelineError("pad_token_id is invalid")
        if not -1_000_000 <= self.label_pad_token_id <= 16_777_215:
            raise DataPipelineError("label_pad_token_id is invalid")
        if not 1 <= self.pad_to_multiple <= 4096:
            raise DataPipelineError("pad_to_multiple is invalid")
        if not 1 <= self.maximum_sequence_length <= 1_048_576:
            raise DataPipelineError("maximum_sequence_length is invalid")

    @classmethod
    def from_resolved(cls, value: Mapping[str, Any]) -> PaddedCollatorConfiguration:
        return cls(
            **_exact(
                value,
                {
                    "pad_token_id",
                    "label_pad_token_id",
                    "pad_to_multiple",
                    "maximum_sequence_length",
                },
                "padded collator",
            )
        )


@dataclass(frozen=True, slots=True)
class PackedTokenCollatorConfiguration:
    pad_token_id: int
    label_pad_token_id: int
    pad_to_multiple: int
    maximum_sequence_length: int
    separator_token_id: int

    def __post_init__(self) -> None:
        PaddedCollatorConfiguration(
            self.pad_token_id,
            self.label_pad_token_id,
            self.pad_to_multiple,
            self.maximum_sequence_length,
        )
        if not 0 <= self.separator_token_id <= 16_777_215:
            raise DataPipelineError("separator_token_id is invalid")

    @classmethod
    def from_resolved(
        cls, value: Mapping[str, Any]
    ) -> PackedTokenCollatorConfiguration:
        return cls(
            **_exact(
                value,
                {
                    "pad_token_id",
                    "label_pad_token_id",
                    "pad_to_multiple",
                    "maximum_sequence_length",
                    "separator_token_id",
                },
                "packed-token collator",
            )
        )


@dataclass(frozen=True, slots=True)
class RegisteredCollator:
    implementation: CollatorImplementation
    configuration: PaddedCollatorConfiguration | PackedTokenCollatorConfiguration

    def __post_init__(self) -> None:
        if (
            self.implementation is CollatorImplementation.PADDED_V1
        ) != isinstance(self.configuration, PaddedCollatorConfiguration):
            raise DataPipelineError(
                "collator implementation and configuration disagree"
            )

    def collate(
        self, samples: Sequence[MappedSample], *, tensor_output: bool = True
    ) -> Mapping[str, Any]:
        if not samples:
            raise DataPipelineError("cannot collate an empty batch")
        if self.implementation is CollatorImplementation.PACKED_TOKENS_V1:
            return self._collate_packed_tokens(samples, tensor_output=tensor_output)
        maximum = min(
            max(sample.token_length for sample in samples),
            self.configuration.maximum_sequence_length,
        )
        multiple = self.configuration.pad_to_multiple
        padded_length = min(
            ((maximum + multiple - 1) // multiple) * multiple,
            self.configuration.maximum_sequence_length,
        )
        input_ids: list[list[int]] = []
        labels: list[list[int]] = []
        attention: list[list[int]] = []
        for sample in samples:
            ids = list(sample.input_ids[:padded_length])
            target = list(sample.labels[:padded_length])
            padding = padded_length - len(ids)
            input_ids.append(ids + [self.configuration.pad_token_id] * padding)
            labels.append(target + [self.configuration.label_pad_token_id] * padding)
            attention.append([1] * len(ids) + [0] * padding)
        result: dict[str, Any] = {
            "sample_ids": tuple(sample.sample_id for sample in samples),
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention,
        }
        if any(sample.image is not None for sample in samples):
            if any(sample.image is None for sample in samples):
                raise DataPipelineError(
                    "a batch cannot mix image and non-image samples"
                )
            result["images"] = [sample.image for sample in samples]
        if tensor_output:
            import torch

            for key in ("input_ids", "labels", "attention_mask"):
                result[key] = torch.tensor(result[key], dtype=torch.long)
        return MappingProxyType(result)

    def _collate_packed_tokens(
        self, samples: Sequence[MappedSample], *, tensor_output: bool
    ) -> Mapping[str, Any]:
        configuration = self.configuration
        if not isinstance(configuration, PackedTokenCollatorConfiguration):
            raise DataPipelineError("packed-token collator configuration is invalid")
        if any(sample.image is not None for sample in samples):
            raise DataPipelineError("packed-token collation does not accept images")
        maximum = configuration.maximum_sequence_length
        rows: list[tuple[list[int], list[int], list[str]]] = []
        row_ids: list[int] = []
        row_labels: list[int] = []
        row_samples: list[str] = []

        def flush() -> None:
            nonlocal row_ids, row_labels, row_samples
            if row_ids:
                rows.append((row_ids, row_labels, row_samples))
                row_ids, row_labels, row_samples = [], [], []

        for sample in samples:
            ids = list(sample.input_ids)
            labels = list(sample.labels)
            if len(ids) != len(labels) or not ids:
                raise DataPipelineError("packed-token sample shape is invalid")
            if len(ids) > maximum:
                raise DataPipelineError(
                    "packed-token sample exceeds maximum_sequence_length"
                )
            required = len(ids) + (1 if row_ids else 0)
            if row_ids and len(row_ids) + required > maximum:
                flush()
            if row_ids:
                row_ids.append(configuration.separator_token_id)
                row_labels.append(configuration.separator_token_id)
            row_ids.extend(ids)
            row_labels.extend(labels)
            row_samples.append(sample.sample_id)
        flush()
        input_ids: list[list[int]] = []
        labels: list[list[int]] = []
        attention: list[list[int]] = []
        packed_sample_ids: list[str] = []
        for ids, target, members in rows:
            padding = maximum - len(ids)
            input_ids.append(ids + [configuration.pad_token_id] * padding)
            labels.append(target + [configuration.label_pad_token_id] * padding)
            attention.append([1] * len(ids) + [0] * padding)
            packed_sample_ids.append("+".join(members))
        result: dict[str, Any] = {
            "sample_ids": tuple(packed_sample_ids),
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention,
        }
        if tensor_output:
            import torch

            for key in ("input_ids", "labels", "attention_mask"):
                result[key] = torch.tensor(result[key], dtype=torch.long)
        return MappingProxyType(result)


@dataclass(frozen=True, slots=True)
class DeterministicSamplerConfiguration:
    seed: int
    shuffle: bool

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or not 0 <= self.seed <= 2**63 - 1:
            raise DataPipelineError("sampler seed is invalid")
        if not isinstance(self.shuffle, bool):
            raise DataPipelineError("sampler shuffle must be boolean")

    @classmethod
    def from_resolved(
        cls, value: Mapping[str, Any]
    ) -> DeterministicSamplerConfiguration:
        value = _exact(value, {"seed", "shuffle"}, "deterministic sampler")
        if not isinstance(value["seed"], int) or not 0 <= value["seed"] <= 2**63 - 1:
            raise DataPipelineError("sampler seed is invalid")
        if not isinstance(value["shuffle"], bool):
            raise DataPipelineError("sampler shuffle must be boolean")
        return cls(**value)


@dataclass(slots=True)
class RegisteredSampler:
    implementation: SamplerImplementation
    configuration: DeterministicSamplerConfiguration
    _membership: tuple[str, ...] = ()
    _order: tuple[str, ...] = ()
    _cursor: int = 0
    _epoch: int = 0

    def bind(self, sample_ids: Sequence[str]) -> None:
        membership = tuple(sample_ids)
        if len(set(membership)) != len(membership) or any(
            not item for item in membership
        ):
            raise DataPipelineError(
                "sampler membership requires unique nonempty sample ids"
            )
        self._membership = membership
        self._cursor = 0
        self._epoch = 0
        self._rebuild_order()

    def _rebuild_order(self) -> None:
        if not self.configuration.shuffle:
            self._order = self._membership
            return
        seed = self.configuration.seed
        epoch = self._epoch
        self._order = tuple(
            sorted(
                self._membership,
                key=lambda sample_id: hashlib.sha256(
                    f"{seed}:{epoch}:{sample_id}".encode()
                ).digest(),
            )
        )

    @property
    def order_digest(self) -> str:
        return _canonical_digest(self._order)

    @property
    def rng_state_digest(self) -> str:
        return _canonical_digest(
            {
                "algorithm": "sha256-sort-v1",
                "epoch": self._epoch,
                "seed": self.configuration.seed,
            }
        )

    def take(self, count: int) -> tuple[str, ...]:
        if not isinstance(count, int) or count < 1:
            raise DataPipelineError("sampler take count must be positive")
        result: list[str] = []
        while len(result) < count and self._order:
            if self._cursor == len(self._order):
                self._epoch += 1
                self._cursor = 0
                self._rebuild_order()
            available = min(count - len(result), len(self._order) - self._cursor)
            result.extend(self._order[self._cursor : self._cursor + available])
            self._cursor += available
        return tuple(result)

    def component_state(self) -> Mapping[str, int | str]:
        return MappingProxyType(
            {
                "cursor": self._cursor,
                "epoch": self._epoch,
                "order_digest": self.order_digest,
                "rng_state_digest": self.rng_state_digest,
            }
        )

    def restore_component_state(self, state: Mapping[str, Any]) -> None:
        state = _exact(
            dict(state),
            {"cursor", "epoch", "order_digest", "rng_state_digest"},
            "sampler resume state",
        )
        cursor, epoch = state["cursor"], state["epoch"]
        if (
            not isinstance(cursor, int)
            or not isinstance(epoch, int)
            or cursor < 0
            or epoch < 0
        ):
            raise DataPipelineError(
                "sampler cursor and epoch must be nonnegative integers"
            )
        self._epoch = epoch
        self._rebuild_order()
        if (
            cursor > len(self._order)
            or state["order_digest"] != self.order_digest
            or state["rng_state_digest"] != self.rng_state_digest
        ):
            raise DataPipelineError(
                "sampler resume state disagrees with deterministic order"
            )
        self._cursor = cursor


@dataclass(frozen=True, slots=True)
class DeterministicHoldoutConfiguration:
    seed: int
    held_out_count: int
    selection: str

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or not 0 <= self.seed <= 2**63 - 1:
            raise DataPipelineError("split-selector seed is invalid")
        if (
            not isinstance(self.held_out_count, int)
            or not 1 <= self.held_out_count <= 1_000_000_000
        ):
            raise DataPipelineError("held_out_count is invalid")
        if self.selection not in {"train", "held_out"}:
            raise DataPipelineError("split selection is invalid")

    @classmethod
    def from_resolved(
        cls, value: Mapping[str, Any]
    ) -> DeterministicHoldoutConfiguration:
        value = _exact(
            value,
            {"seed", "held_out_count", "selection"},
            "deterministic holdout selector",
        )
        if not isinstance(value["seed"], int) or not 0 <= value["seed"] <= 2**63 - 1:
            raise DataPipelineError("split-selector seed is invalid")
        if (
            not isinstance(value["held_out_count"], int)
            or not 1 <= value["held_out_count"] <= 1_000_000_000
        ):
            raise DataPipelineError("held_out_count is invalid")
        if value["selection"] not in {"train", "held_out"}:
            raise DataPipelineError("split selection is invalid")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class FrozenNamedSplitConfiguration:
    selection: str

    def __post_init__(self) -> None:
        if self.selection not in {"test", "train", "validation"}:
            raise DataPipelineError("frozen named split selection is invalid")

    @classmethod
    def from_resolved(
        cls, value: Mapping[str, Any]
    ) -> FrozenNamedSplitConfiguration:
        return cls(**_exact(value, {"selection"}, "frozen named split selector"))


@dataclass(frozen=True, slots=True)
class SplitSelection:
    selected_ids: tuple[str, ...]
    held_out_ids: tuple[str, ...]
    membership_digest: str


@dataclass(frozen=True, slots=True)
class RegisteredSplitSelector:
    implementation: SplitSelectorImplementation
    configuration: DeterministicHoldoutConfiguration | FrozenNamedSplitConfiguration

    def select(self, sample_ids: Sequence[str]) -> SplitSelection:
        membership = tuple(sample_ids)
        if len(set(membership)) != len(membership):
            raise DataPipelineError("split selection requires unique sample ids")
        if isinstance(self.configuration, FrozenNamedSplitConfiguration):
            return SplitSelection(
                membership,
                (),
                _canonical_digest(
                    {
                        "algorithm": "manifested-jsonl-split-v1",
                        "selection": self.configuration.selection,
                        "source_membership": membership,
                    }
                ),
            )
        if self.configuration.held_out_count >= len(membership):
            raise DataPipelineError(
                "held_out_count must leave at least one training sample"
            )
        ranked = sorted(
            membership,
            key=lambda sample_id: hashlib.sha256(
                f"{self.configuration.seed}:{sample_id}".encode()
            ).digest(),
        )
        held_out_set = frozenset(ranked[: self.configuration.held_out_count])
        held_out = tuple(item for item in membership if item in held_out_set)
        training = tuple(item for item in membership if item not in held_out_set)
        selected = training if self.configuration.selection == "train" else held_out
        identity = {
            "algorithm": "sha256-rank-v1",
            "held_out": held_out,
            "seed": self.configuration.seed,
            "source_membership": membership,
        }
        return SplitSelection(selected, held_out, _canonical_digest(identity))


def _positive_integer_strings(
    value: Any, label: str, *, strictly_increasing: bool = True
) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise DataPipelineError(f"{label} must be a nonempty integer-string list")
    parsed: list[int] = []
    for item in value:
        if (
            not isinstance(item, str)
            or not item.isascii()
            or not item.isdigit()
            or item.startswith("0")
        ):
            raise DataPipelineError(f"{label} contains a non-canonical integer")
        number = int(item)
        if not 1 <= number <= 1_000_000_000:
            raise DataPipelineError(f"{label} contains an out-of-range integer")
        parsed.append(number)
    if strictly_increasing and tuple(sorted(set(parsed))) != tuple(parsed):
        raise DataPipelineError(f"{label} must be strictly increasing")
    return tuple(parsed)


@dataclass(frozen=True, slots=True)
class FixedBatchingConfiguration:
    batch_size: int
    drop_last: bool
    prefetch_workers: int

    def __post_init__(self) -> None:
        if not isinstance(self.batch_size, int) or not 1 <= self.batch_size <= 257:
            raise DataPipelineError("batch_size is invalid")
        if not isinstance(self.drop_last, bool):
            raise DataPipelineError("drop_last must be boolean")
        if (
            not isinstance(self.prefetch_workers, int)
            or not 0 <= self.prefetch_workers <= 1024
        ):
            raise DataPipelineError("prefetch_workers is invalid")

    @classmethod
    def from_resolved(cls, value: Mapping[str, Any]) -> FixedBatchingConfiguration:
        value = _exact(
            value, {"batch_size", "drop_last", "prefetch_workers"}, "fixed batching"
        )
        if (
            not isinstance(value["batch_size"], int)
            or not 1 <= value["batch_size"] <= 257
        ):
            raise DataPipelineError("batch_size is invalid")
        if not isinstance(value["drop_last"], bool):
            raise DataPipelineError("drop_last must be boolean")
        if (
            not isinstance(value["prefetch_workers"], int)
            or not 0 <= value["prefetch_workers"] <= 1024
        ):
            raise DataPipelineError("prefetch_workers is invalid")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class BucketedBatchingConfiguration:
    bucket_by: str
    bucket_boundaries: tuple[int, ...]
    batch_sizes: tuple[int, ...]
    drop_last: bool
    prefetch_workers: int

    def __post_init__(self) -> None:
        if self.bucket_by not in {"image_area", "token_length"}:
            raise DataPipelineError("bucket_by is invalid")
        if (
            not self.bucket_boundaries
            or tuple(sorted(set(self.bucket_boundaries))) != self.bucket_boundaries
        ):
            raise DataPipelineError("bucket_boundaries must be strictly increasing")
        if len(self.batch_sizes) != len(self.bucket_boundaries) + 1 or any(
            not 1 <= size <= 257 for size in self.batch_sizes
        ):
            raise DataPipelineError("batch_sizes are invalid")
        if sum(size - 1 for size in self.batch_sizes) > 256:
            raise DataPipelineError("bucket queues exceed exact resume-state capacity")
        if not isinstance(self.drop_last, bool):
            raise DataPipelineError("drop_last must be boolean")
        if (
            not isinstance(self.prefetch_workers, int)
            or not 0 <= self.prefetch_workers <= 1024
        ):
            raise DataPipelineError("prefetch_workers is invalid")

    @classmethod
    def from_resolved(cls, value: Mapping[str, Any]) -> BucketedBatchingConfiguration:
        value = _exact(
            value,
            {
                "bucket_by",
                "bucket_boundaries",
                "batch_sizes",
                "drop_last",
                "prefetch_workers",
            },
            "bucketed batching",
        )
        if value["bucket_by"] not in {"image_area", "token_length"}:
            raise DataPipelineError("bucket_by is invalid")
        boundaries = _positive_integer_strings(
            value["bucket_boundaries"], "bucket_boundaries"
        )
        sizes = _positive_integer_strings(
            value["batch_sizes"], "batch_sizes", strictly_increasing=False
        )
        if len(sizes) != len(boundaries) + 1:
            raise DataPipelineError(
                "batch_sizes must have one more entry than bucket_boundaries"
            )
        if not isinstance(value["drop_last"], bool):
            raise DataPipelineError("drop_last must be boolean")
        if (
            not isinstance(value["prefetch_workers"], int)
            or not 0 <= value["prefetch_workers"] <= 1024
        ):
            raise DataPipelineError("prefetch_workers is invalid")
        return cls(
            value["bucket_by"],
            boundaries,
            sizes,
            value["drop_last"],
            value["prefetch_workers"],
        )


BatchingConfiguration: TypeAlias = (
    FixedBatchingConfiguration | BucketedBatchingConfiguration
)


@dataclass(slots=True)
class RegisteredBatching:
    implementation: BatchingImplementation
    configuration: BatchingConfiguration
    _pending: dict[int, deque[str]] = field(default_factory=lambda: defaultdict(deque))
    _batches_emitted: int = 0
    _sample_measurements: dict[str, int] = field(default_factory=dict)

    @property
    def prefetch_workers(self) -> int:
        return self.configuration.prefetch_workers

    def _bucket(self, measurement: int) -> int:
        configuration = self.configuration
        if isinstance(configuration, FixedBatchingConfiguration):
            return 0
        for index, boundary in enumerate(configuration.bucket_boundaries):
            if measurement <= boundary:
                return index
        return len(configuration.bucket_boundaries)

    def _batch_size(self, bucket: int) -> int:
        configuration = self.configuration
        if isinstance(configuration, FixedBatchingConfiguration):
            return configuration.batch_size
        return configuration.batch_sizes[bucket]

    def add(self, sample_id: str, *, measurement: int) -> tuple[str, ...] | None:
        if not sample_id or sample_id in self._sample_measurements:
            raise DataPipelineError(
                "batching requires unique nonempty pending sample ids"
            )
        if not isinstance(measurement, int) or measurement < 1:
            raise DataPipelineError("batch measurement must be a positive integer")
        bucket = self._bucket(measurement)
        self._pending[bucket].append(sample_id)
        self._sample_measurements[sample_id] = measurement
        size = self._batch_size(bucket)
        if len(self._pending[bucket]) < size:
            return None
        batch = tuple(self._pending[bucket].popleft() for _ in range(size))
        for item in batch:
            del self._sample_measurements[item]
        self._batches_emitted += 1
        return batch

    def flush(self) -> tuple[tuple[str, ...], ...]:
        if self.configuration.drop_last:
            self._pending.clear()
            self._sample_measurements.clear()
            return ()
        batches: list[tuple[str, ...]] = []
        for bucket in sorted(self._pending):
            queue = self._pending[bucket]
            if queue:
                batch = tuple(queue)
                batches.append(batch)
                self._batches_emitted += 1
        self._pending.clear()
        self._sample_measurements.clear()
        return tuple(batches)

    @property
    def assignment_digest(self) -> str:
        return _canonical_digest(
            {
                "implementation": self.implementation.value,
                "pending": [
                    (item, self._sample_measurements[item])
                    for bucket in sorted(self._pending)
                    for item in self._pending[bucket]
                ],
            }
        )

    def component_state(self) -> Mapping[str, Any]:
        pending = tuple(
            item for bucket in sorted(self._pending) for item in self._pending[bucket]
        )
        return MappingProxyType(
            {
                "batches_emitted": self._batches_emitted,
                "pending_sample_ids": pending,
                "bucket_assignment_digest": self.assignment_digest,
            }
        )

    def restore_component_state(
        self, state: Mapping[str, Any], *, measurements: Mapping[str, int]
    ) -> None:
        state = _exact(
            dict(state),
            {"batches_emitted", "pending_sample_ids", "bucket_assignment_digest"},
            "batching resume state",
        )
        emitted, pending = state["batches_emitted"], state["pending_sample_ids"]
        if (
            not isinstance(emitted, int)
            or emitted < 0
            or not isinstance(pending, (tuple, list))
            or len(set(pending)) != len(pending)
        ):
            raise DataPipelineError("batching resume state is invalid")
        self._pending.clear()
        self._sample_measurements.clear()
        self._batches_emitted = emitted
        for sample_id in pending:
            if sample_id not in measurements:
                raise DataPipelineError(
                    "batching resume state lacks a pending measurement"
                )
            measurement = measurements[sample_id]
            bucket = self._bucket(measurement)
            self._pending[bucket].append(sample_id)
            self._sample_measurements[sample_id] = measurement
        if state["bucket_assignment_digest"] != self.assignment_digest:
            raise DataPipelineError("batching resume bucket assignment disagrees")


@dataclass(frozen=True, slots=True)
class DataPreflightEvidence:
    samples_checked: int
    decoded_sample_ids: tuple[str, ...]
    split_membership_digest: str
    supervised_token_counts: tuple[int, ...]
    batch_shapes: tuple[tuple[int, int], ...]
    evidence_digest: str


@dataclass(frozen=True, slots=True)
class DeclarativeDataPipeline:
    source: RegisteredDataSource
    processor: RegisteredSampleProcessor
    mapper: RegisteredSampleMapper
    collator: RegisteredCollator
    sampler: RegisteredSampler
    batching: RegisteredBatching
    split_selector: RegisteredSplitSelector

    def validate_schema(self) -> None:
        declared = frozenset(self.source.declared_columns)
        missing_processor = self.processor.required_columns - declared
        missing_mapper = self.mapper.required_columns - declared
        if missing_processor or missing_mapper:
            missing = sorted(missing_processor | missing_mapper)
            raise DataPipelineError(
                "pipeline references undeclared columns: " + ", ".join(missing)
            )
        if isinstance(
            self.source.configuration,
            (JsonlFrozenImageSplitsConfiguration, JsonlImageCaptionConfiguration),
        ) != isinstance(
            self.processor.configuration, ImageCaptionProcessorConfiguration
        ):
            raise DataPipelineError(
                "data source and sample processor modalities are incompatible"
            )
        if isinstance(
            self.processor.configuration, TokenIdsProcessorConfiguration
        ) != isinstance(self.mapper.configuration, CausalTokensMapperConfiguration):
            raise DataPipelineError(
                "sample processor and sample mapper modalities are incompatible"
            )

    def preflight(
        self,
        *,
        tokenizer: TokenizerProtocol | None = None,
        sample_limit: int = 8,
        tensor_output: bool = False,
    ) -> DataPreflightEvidence:
        if (
            not isinstance(sample_limit, int)
            or not 2 <= sample_limit <= _MAXIMUM_PREFLIGHT_SAMPLES
        ):
            raise DataPipelineError("preflight sample_limit must be between 2 and 256")
        self.validate_schema()
        self.source.verify_content()
        raw = tuple(self.source.records(limit=sample_limit))
        if len(raw) < 2:
            raise DataPipelineError(
                "bounded preflight requires at least two source samples"
            )
        split = self.split_selector.select(tuple(sample.sample_id for sample in raw))
        selected = frozenset(split.selected_ids)
        configuration = self.source.configuration
        image_root = (
            Path(configuration.image_root)
            if isinstance(
                configuration,
                (JsonlFrozenImageSplitsConfiguration, JsonlImageCaptionConfiguration),
            )
            else None
        )
        mapped: list[MappedSample] = []
        for sample in raw:
            if sample.sample_id not in selected:
                continue
            processed = self.processor.process(sample, image_root=image_root)
            mapped.append(self.mapper.map(processed, tokenizer=tokenizer))
        if not mapped:
            raise DataPipelineError("preflight split selected no samples")
        # Preflight must never advance the live training cursor. It exercises
        # fresh instances with the exact locked configurations so calling it
        # once or repeatedly cannot skip the first training samples.
        sampler = RegisteredSampler(
            self.sampler.implementation, self.sampler.configuration
        )
        batching = RegisteredBatching(
            self.batching.implementation, self.batching.configuration
        )
        sampler.bind(tuple(sample.sample_id for sample in mapped))
        ordered_ids = sampler.take(len(mapped))
        by_id = {sample.sample_id: sample for sample in mapped}
        batch_ids: list[tuple[str, ...]] = []
        for sample_id in ordered_ids:
            sample = by_id[sample_id]
            if (
                isinstance(batching.configuration, BucketedBatchingConfiguration)
                and batching.configuration.bucket_by == "image_area"
            ):
                if sample.image_size is None:
                    raise DataPipelineError(
                        "image_area bucketing requires decoded images"
                    )
                measurement = sample.image_size[0] * sample.image_size[1]
            else:
                measurement = sample.token_length
            emitted = batching.add(sample_id, measurement=measurement)
            if emitted:
                batch_ids.append(emitted)
        batch_ids.extend(batching.flush())
        shapes: list[tuple[int, int]] = []
        for identifiers in batch_ids:
            batch = self.collator.collate(
                [by_id[item] for item in identifiers], tensor_output=tensor_output
            )
            values = batch["input_ids"]
            shape = (
                tuple(values.shape) if tensor_output else (len(values), len(values[0]))
            )
            shapes.append((int(shape[0]), int(shape[1])))
        supervised = tuple(
            sum(label != -100 for label in sample.labels) for sample in mapped
        )
        body = {
            "batch_shapes": shapes,
            "decoded_sample_ids": [sample.sample_id for sample in mapped],
            "samples_checked": len(raw),
            "split_membership_digest": split.membership_digest,
            "supervised_token_counts": supervised,
        }
        return DataPreflightEvidence(
            samples_checked=len(raw),
            decoded_sample_ids=tuple(body["decoded_sample_ids"]),
            split_membership_digest=split.membership_digest,
            supervised_token_counts=supervised,
            batch_shapes=tuple(shapes),
            evidence_digest=_canonical_digest(body),
        )


DataRuntimeComponent: TypeAlias = (
    RegisteredDataSource
    | RegisteredSampleProcessor
    | RegisteredSampleMapper
    | RegisteredCollator
    | RegisteredSampler
    | RegisteredBatching
    | RegisteredSplitSelector
)


def _resolved(
    component: Mapping[str, Any], category: str
) -> tuple[str, Mapping[str, Any]]:
    if set(component) != {"configuration", "descriptor", "descriptor_digest"}:
        raise DataPipelineError(f"resolved {category} envelope has unknown fields")
    descriptor = component["descriptor"]
    if not isinstance(descriptor, Mapping) or descriptor["key"]["category"] != category:
        raise DataPipelineError(f"resolved component is not a {category}")
    return descriptor["implementation"], component["configuration"]


def data_source_from_resolved_component(
    component: Mapping[str, Any],
) -> RegisteredDataSource:
    implementation_value, value = _resolved(component, "data_source")
    implementation = DataSourceImplementation(implementation_value)
    configuration: DataSourceConfiguration
    if implementation is DataSourceImplementation.JSONL_IMAGE_CAPTION_V1:
        configuration = JsonlImageCaptionConfiguration.from_resolved(value)
    elif implementation is DataSourceImplementation.JSONL_TOKEN_CORPUS_V1:
        configuration = JsonlTokenCorpusConfiguration.from_resolved(value)
    elif implementation is DataSourceImplementation.JSONL_FROZEN_IMAGE_SPLITS_V1:
        configuration = JsonlFrozenImageSplitsConfiguration.from_resolved(value)
    else:
        configuration = JsonlFrozenTokenSplitsConfiguration.from_resolved(value)
    return RegisteredDataSource(implementation, configuration)


def sample_processor_from_resolved_component(
    component: Mapping[str, Any],
) -> RegisteredSampleProcessor:
    implementation_value, value = _resolved(component, "sample_processor")
    implementation = SampleProcessorImplementation(implementation_value)
    configuration: SampleProcessorConfiguration
    if implementation is SampleProcessorImplementation.IMAGE_CAPTION_V1:
        configuration = ImageCaptionProcessorConfiguration.from_resolved(value)
    else:
        configuration = TokenIdsProcessorConfiguration.from_resolved(value)
    return RegisteredSampleProcessor(implementation, configuration)


def sample_mapper_from_resolved_component(
    component: Mapping[str, Any],
) -> RegisteredSampleMapper:
    implementation_value, value = _resolved(component, "sample_mapper")
    implementation = SampleMapperImplementation(implementation_value)
    configuration: SampleMapperConfiguration
    if implementation is SampleMapperImplementation.ASSISTANT_ONLY_V1:
        configuration = AssistantOnlyMapperConfiguration.from_resolved(value)
    elif implementation is SampleMapperImplementation.ASSISTANT_CONVERSATION_V2:
        configuration = AssistantConversationMapperConfiguration.from_resolved(value)
    else:
        configuration = CausalTokensMapperConfiguration.from_resolved(value)
    return RegisteredSampleMapper(implementation, configuration)


def collator_from_resolved_component(
    component: Mapping[str, Any],
) -> RegisteredCollator:
    implementation_value, value = _resolved(component, "collator")
    implementation = CollatorImplementation(implementation_value)
    configuration: PaddedCollatorConfiguration | PackedTokenCollatorConfiguration
    if implementation is CollatorImplementation.PADDED_V1:
        configuration = PaddedCollatorConfiguration.from_resolved(value)
    else:
        configuration = PackedTokenCollatorConfiguration.from_resolved(value)
    return RegisteredCollator(implementation, configuration)


def sampler_from_resolved_component(component: Mapping[str, Any]) -> RegisteredSampler:
    implementation_value, value = _resolved(component, "sampler")
    implementation = SamplerImplementation(implementation_value)
    return RegisteredSampler(
        implementation, DeterministicSamplerConfiguration.from_resolved(value)
    )


def batching_from_resolved_component(
    component: Mapping[str, Any],
) -> RegisteredBatching:
    implementation_value, value = _resolved(component, "batching")
    implementation = BatchingImplementation(implementation_value)
    configuration: BatchingConfiguration
    if implementation is BatchingImplementation.FIXED_V1:
        configuration = FixedBatchingConfiguration.from_resolved(value)
    else:
        configuration = BucketedBatchingConfiguration.from_resolved(value)
    return RegisteredBatching(implementation, configuration)


def split_selector_from_resolved_component(
    component: Mapping[str, Any],
) -> RegisteredSplitSelector:
    implementation_value, value = _resolved(component, "split_selector")
    implementation = SplitSelectorImplementation(implementation_value)
    configuration = (
        DeterministicHoldoutConfiguration.from_resolved(value)
        if implementation is SplitSelectorImplementation.DETERMINISTIC_HOLDOUT_V1
        else FrozenNamedSplitConfiguration.from_resolved(value)
    )
    return RegisteredSplitSelector(implementation, configuration)


def build_data_pipeline(
    *,
    source: RegisteredDataSource,
    processor: RegisteredSampleProcessor,
    mapper: RegisteredSampleMapper,
    collator: RegisteredCollator,
    sampler: RegisteredSampler,
    batching: RegisteredBatching,
    split_selector: RegisteredSplitSelector,
) -> DeclarativeDataPipeline:
    pipeline = DeclarativeDataPipeline(
        source, processor, mapper, collator, sampler, batching, split_selector
    )
    pipeline.validate_schema()
    return pipeline
