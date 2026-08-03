from __future__ import annotations

import importlib.util
import json
import sys
from array import array
from pathlib import Path

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/prepare_trainvm_production_qualification_data.py"
SPEC = importlib.util.spec_from_file_location("trainvm_production_data_snapshot", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _tokenizer(tmp_path: Path) -> Path:
    root = tmp_path / "tokenizer"
    root.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(
        models.WordLevel(
            {"[UNK]": 0, "hello": 1, "world": 2, "training": 3, "dashboard": 4},
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(root / "tokenizer.json"))
    (root / "config.json").write_text(
        json.dumps({"text_config": {"eos_token_id": 5}}), encoding="utf-8"
    )
    return root


def _manifest(tmp_path: Path, name: str, *, rows_per_domain: int = 4) -> Path:
    path = tmp_path / f"{name}.jsonl"
    rows = []
    for domain_index, domain in enumerate(MODULE.DOMAINS):
        for index in range(rows_per_domain):
            image = tmp_path / f"{name}-{domain}-{index}.png"
            image.write_bytes(f"{name}-{domain}-{index}".encode())
            rows.append(
                {
                    "image": str(image),
                    "image_id": f"{name}-{domain}-{index}",
                    "domain": domain,
                    "caption": f"{domain} caption {index}",
                    "train_width": 64,
                    "train_height": 64,
                    "latent_tokens": 20 - index + domain_index,
                    "source": "test",
                }
            )
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in reversed(rows)),
        encoding="utf-8",
    )
    return path


def _prepare(tmp_path: Path, destination: Path) -> Path:
    text = tmp_path / "source.txt"
    text.write_text("hello world training dashboard\n", encoding="utf-8")
    return MODULE.prepare(
        train_manifest=_manifest(tmp_path, "train"),
        eval_manifest=_manifest(tmp_path, "eval"),
        tokenizer_directory=_tokenizer(tmp_path),
        text_sources=[text],
        destination=destination,
        train_per_domain=2,
        eval_per_domain=1,
        transformer_tokens=32,
    )


def test_snapshot_is_balanced_content_bound_and_uses_final_paths(tmp_path) -> None:
    destination = tmp_path / "snapshot"
    report_path = _prepare(tmp_path, destination)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    train_rows = [
        json.loads(line)
        for line in (destination / "mageflow/train.jsonl").read_text().splitlines()
    ]
    eval_rows = [
        json.loads(line)
        for line in (destination / "mageflow/eval.jsonl").read_text().splitlines()
    ]

    assert report["api_version"] == MODULE.SNAPSHOT_VERSION
    assert [row["domain"] for row in train_rows] == [
        "animation",
        "photo",
        "animation",
        "photo",
    ]
    assert [row["domain"] for row in eval_rows] == ["animation", "photo"]
    assert {row["image_id"] for row in train_rows}.isdisjoint(
        row["image_id"] for row in eval_rows
    )
    for row in [*train_rows, *eval_rows]:
        image = Path(row["image"])
        assert destination in image.parents
        assert ".tmp-" not in str(image)
        assert image.is_file()
        assert len(row["qualification_source_image_sha256"]) == 64
    assert report["mageflow"]["train_manifest"] == str(
        (destination / "mageflow/train.jsonl").resolve()
    )


def test_snapshot_writes_exact_little_endian_uint32_stream(tmp_path) -> None:
    destination = tmp_path / "snapshot"
    report = json.loads(_prepare(tmp_path, destination).read_text(encoding="utf-8"))
    tokens = array("I")
    with (destination / "transformer/tokens.bin").open("rb") as handle:
        tokens.fromfile(handle, 32)

    assert len(tokens) == 32
    assert set(tokens).issubset({0, 1, 2, 3, 4, 5})
    assert report["transformer"]["dtype"] == "uint32-le"
    assert report["transformer"]["total_tokens"] == 32
    assert report["transformer"]["tokens_path"] == str(
        (destination / "transformer/tokens.bin").resolve()
    )
    assert len(report["transformer"]["tokens_sha256"]) == 64


def test_snapshot_is_new_only_and_rejects_missing_domain(tmp_path) -> None:
    destination = tmp_path / "snapshot"
    _prepare(tmp_path, destination)
    with pytest.raises(MODULE.SnapshotError, match="already exists"):
        _prepare(tmp_path, destination)

    train = _manifest(tmp_path, "bad-train", rows_per_domain=1)
    rows = [json.loads(line) for line in train.read_text().splitlines()]
    train.write_text(
        "".join(json.dumps(row) + "\n" for row in rows if row["domain"] == "photo"),
        encoding="utf-8",
    )
    text = tmp_path / "bad-source.txt"
    text.write_text("hello", encoding="utf-8")
    with pytest.raises(MODULE.SnapshotError, match="eligible animation"):
        MODULE.prepare(
            train_manifest=train,
            eval_manifest=_manifest(tmp_path, "bad-eval", rows_per_domain=1),
            tokenizer_directory=_tokenizer(tmp_path / "bad-tokenizer-root"),
            text_sources=[text],
            destination=tmp_path / "bad-snapshot",
            train_per_domain=1,
            eval_per_domain=1,
            transformer_tokens=8,
        )
