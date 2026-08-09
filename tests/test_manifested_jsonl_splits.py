from __future__ import annotations

import hashlib
import json

import pytest
from PIL import Image

from rwkv_lab.training_components import (
    DataPipelineError,
    DataSourceImplementation,
    JsonlFrozenImageSplitsConfiguration,
    JsonlFrozenTokenSplitsConfiguration,
    RegisteredDataSource,
)


def _write_dataset(root):
    (root / "images").mkdir()
    for index in range(4):
        Image.new("RGB", (8, 8), (index, 0, 0)).save(
            root / "images" / f"{index}.png"
        )
    rows = {
        "train": [
            {"id": "a", "split": "train", "caption": "a", "image": "images/0.png"},
            {"id": "b", "split": "train", "caption": "b", "image": "images/1.png"},
        ],
        "validation": [
            {"id": "c", "split": "validation", "caption": "c", "image": "images/2.png"}
        ],
        "test": [
            {"id": "d", "split": "test", "caption": "d", "image": "images/3.png"}
        ],
    }
    for split, values in rows.items():
        (root / f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in values), encoding="utf-8"
        )
    _refresh_receipt(root)


def _refresh_receipt(root):
    files = {}
    counts = {}
    for split in ("train", "validation", "test"):
        path = root / f"{split}.jsonl"
        encoded = path.read_bytes()
        counts[split] = len(encoded.splitlines())
        files[path.name] = {
            "rows": counts[split],
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "fixture.manifested-jsonl-splits.v1",
                "dataset_digest": "fixture",
                "counts": counts,
                "files": files,
                "unique_content_hashes": sum(counts.values()),
            }
        ),
        encoding="utf-8",
    )


def _source(root, fingerprint="sha256:" + "f" * 64):
    return RegisteredDataSource(
        DataSourceImplementation.JSONL_FROZEN_IMAGE_SPLITS_V1,
        JsonlFrozenImageSplitsConfiguration(
            dataset_root=str(root),
            content_fingerprint=fingerprint,
            declared_columns=("caption", "id", "image", "split"),
            image_column="image",
            caption_columns=("caption",),
            id_column="id",
        ),
    )


def test_manifested_splits_validate_closed_receipt_and_authority(tmp_path) -> None:
    _write_dataset(tmp_path)
    source = _source(tmp_path)
    source.verify_content(authority_content_fingerprint="sha256:" + "f" * 64)
    assert [len(source.records_for_split(name)) for name in ("train", "validation", "test")] == [2, 1, 1]
    with pytest.raises(DataPipelineError, match="workspace authority"):
        source.verify_content(authority_content_fingerprint="sha256:" + "e" * 64)


@pytest.mark.parametrize("fault", ["truncated", "duplicate", "missing", "empty"])
def test_manifested_splits_fail_closed_on_bad_membership(tmp_path, fault) -> None:
    _write_dataset(tmp_path)
    train = tmp_path / "train.jsonl"
    rows = [json.loads(line) for line in train.read_text(encoding="utf-8").splitlines()]
    if fault == "truncated":
        train.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    elif fault == "duplicate":
        rows[1]["id"] = "c"
        train.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        _refresh_receipt(tmp_path)
    elif fault == "missing":
        rows[0]["image"] = "images/absent.png"
        train.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        _refresh_receipt(tmp_path)
    else:
        rows[0]["caption"] = " "
        train.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        _refresh_receipt(tmp_path)
    with pytest.raises(DataPipelineError):
        _source(tmp_path).verify_content(
            authority_content_fingerprint="sha256:" + "f" * 64
        )


def _write_token_dataset(root) -> None:
    rows = {
        "train": [
            {"id": "a", "split": "train", "tokens": [1, 2, 3]},
            {"id": "b", "split": "train", "tokens": [4, 5, 6]},
        ],
        "validation": [
            {"id": "c", "split": "validation", "tokens": [7, 8, 9]}
        ],
        "test": [{"id": "d", "split": "test", "tokens": [10, 11, 12]}],
    }
    for split, values in rows.items():
        (root / f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in values), encoding="utf-8"
        )
    _refresh_receipt(root)


def _token_source(root):
    return RegisteredDataSource(
        DataSourceImplementation.JSONL_FROZEN_TOKEN_SPLITS_V1,
        JsonlFrozenTokenSplitsConfiguration(
            dataset_root=str(root),
            content_fingerprint="sha256:" + "a" * 64,
            declared_columns=("id", "split", "tokens"),
            token_column="tokens",
            id_column="id",
        ),
    )


def test_manifested_token_splits_validate_all_three_frozen_partitions(
    tmp_path,
) -> None:
    _write_token_dataset(tmp_path)
    source = _token_source(tmp_path)
    source.verify_content(authority_content_fingerprint="sha256:" + "a" * 64)
    assert [
        tuple(sample.sample_id for sample in source.records_for_split(name))
        for name in ("train", "validation", "test")
    ] == [("a", "b"), ("c",), ("d",)]


@pytest.mark.parametrize(
    "tokens",
    [[], [1, -1], [1, True], [1, "2"]],
)
def test_manifested_token_splits_reject_invalid_token_rows(tmp_path, tokens) -> None:
    _write_token_dataset(tmp_path)
    train = tmp_path / "train.jsonl"
    rows = [json.loads(line) for line in train.read_text().splitlines()]
    rows[0]["tokens"] = tokens
    train.write_text("".join(json.dumps(row) + "\n" for row in rows))
    _refresh_receipt(tmp_path)
    with pytest.raises(DataPipelineError, match="invalid token IDs"):
        _token_source(tmp_path).verify_content(
            authority_content_fingerprint="sha256:" + "a" * 64
        )
