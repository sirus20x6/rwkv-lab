"""Tests for the caption before/after gallery builder.

The script's whole claim is in its name: the selection is *deterministic*. A
gallery whose membership drifts between runs cannot be used to compare a model
against itself, which is the only thing it is for — so determinism is the
behaviour worth pinning, not the JSON shape.

The guards are the other half. This script reads two independently produced
caption files and pairs them by image id. Every way that pairing can be wrong
silently — an id present in one file only, an image path that changed under a
stable id, a subset relabelled between runs — produces a gallery that looks
fine and compares the wrong things. Each of those raises here.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

REPOSITORY = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = REPOSITORY / "scripts" / "build_caption_before_after_gallery.py"


def _write_records(path: pathlib.Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


def _corpus(
    tmp_path: pathlib.Path, count: int = 4, subset: str = "alpha"
) -> tuple[pathlib.Path, pathlib.Path]:
    """A minimal before/after pair over real image files on disk.

    The script stats every image, so these have to exist; contents are never
    read, only the path identity is compared.
    """
    images = tmp_path / "images"
    images.mkdir(exist_ok=True)
    before, after = [], []
    for index in range(count):
        image = images / f"{index}.png"
        image.write_bytes(b"")
        identifier = f"id-{index}"
        before.append(
            {
                "id": identifier,
                "image": str(image),
                "response": f"before caption {index}",
                "i1_subset": subset,
            }
        )
        after.append(
            {
                "id": identifier,
                "image": str(image),
                "response": f"after caption {index}",
                "i1_subset": subset,
                "generated_tokens": 12,
            }
        )
    before_path, after_path = tmp_path / "before.jsonl", tmp_path / "after.jsonl"
    _write_records(before_path, before)
    _write_records(after_path, after)
    return before_path, after_path


def _run(
    before: pathlib.Path,
    after: pathlib.Path,
    output: pathlib.Path,
    *,
    per_group: int = 2,
    seed: str | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(SCRIPT),
        "--before", str(before),
        "--after", str(after),
        "--output", str(output),
        "--step", "100",
        "--ppl", "1.5",
        "--per-group", str(per_group),
    ]
    if seed is not None:
        command += ["--seed", seed]
    return subprocess.run(command, capture_output=True, text=True, check=False)


def test_selection_is_deterministic_across_runs(tmp_path: pathlib.Path) -> None:
    before, after = _corpus(tmp_path, count=8)
    first, second = tmp_path / "a.json", tmp_path / "b.json"

    assert _run(before, after, first).returncode == 0
    assert _run(before, after, second).returncode == 0

    assert json.loads(first.read_text()) == json.loads(second.read_text())


def test_a_different_seed_selects_a_different_set(tmp_path: pathlib.Path) -> None:
    """Determinism must come from the seed, not from the file order.

    Without this, a builder that simply took the first N records every time
    would pass the determinism test above while ignoring the seed entirely.
    """
    before, after = _corpus(tmp_path, count=8)
    first, second = tmp_path / "a.json", tmp_path / "b.json"

    assert _run(before, after, first, seed="seed-one").returncode == 0
    assert _run(before, after, second, seed="seed-two").returncode == 0

    chosen_first = {item["caption"] for item in json.loads(first.read_text())["items"]}
    chosen_second = {item["caption"] for item in json.loads(second.read_text())["items"]}
    assert chosen_first != chosen_second


def test_refuses_to_replace_an_existing_gallery(tmp_path: pathlib.Path) -> None:
    before, after = _corpus(tmp_path)
    output = tmp_path / "gallery.json"
    output.write_text("{}", encoding="utf-8")

    result = _run(before, after, output)

    assert result.returncode != 0
    assert "refusing to replace" in result.stderr
    assert output.read_text() == "{}", "the existing gallery was overwritten"


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        pytest.param(
            {"id": "unpaired"},
            "identical image IDs",
            id="id_present_in_only_one_file",
        ),
        pytest.param(
            {"image": "elsewhere.png"},
            "image identity changed",
            id="image_swapped_under_a_stable_id",
        ),
        pytest.param(
            {"i1_subset": "beta"},
            "source group changed",
            id="subset_relabelled_between_runs",
        ),
        pytest.param(
            {"response": "   "},
            "empty comparison caption",
            id="blank_caption",
        ),
    ],
)
def test_mismatched_pairing_is_refused(
    tmp_path: pathlib.Path, mutation: dict[str, str], expected: str
) -> None:
    """Each of these would otherwise yield a gallery comparing the wrong pair."""
    before, after = _corpus(tmp_path)

    records = [json.loads(line) for line in after.read_text().splitlines()]
    records[0].update(mutation)
    _write_records(after, records)

    result = _run(before, after, tmp_path / "gallery.json")

    assert result.returncode != 0
    assert expected in result.stderr, result.stderr


def test_too_few_records_in_a_group_is_refused(tmp_path: pathlib.Path) -> None:
    """Silently emitting a short group would understate what was compared."""
    before, after = _corpus(tmp_path, count=2)

    result = _run(before, after, tmp_path / "gallery.json", per_group=6)

    assert result.returncode != 0
    assert "fewer than" in result.stderr
