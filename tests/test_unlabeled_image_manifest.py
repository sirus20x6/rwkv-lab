import importlib.util
import json
import os
import sys
from pathlib import Path

from PIL import Image


SCRIPT = Path(__file__).parents[1] / "scripts/build_unlabeled_image_manifest.py"
SPEC = importlib.util.spec_from_file_location("unlabeled_manifest", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def make_image(path: Path, color: tuple[int, int, int], size=(320, 280)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, quality=91)


def test_resumable_exact_perceptual_pipeline(tmp_path: Path):
    root = tmp_path / "images"
    original = root / "set" / "a.jpg"
    exact = root / "set" / "a-copy.jpg"
    resized = root / "set" / "a-resized.jpg"
    other = root / "set" / "b.jpg"
    excluded = root / "barely legal" / "excluded.jpg"
    make_image(original, (220, 30, 20))
    exact.write_bytes(original.read_bytes())
    make_image(resized, (220, 30, 20), size=(640, 560))
    make_image(other, (10, 30, 220))
    make_image(excluded, (20, 220, 30))

    db_path = tmp_path / "dedup.sqlite"
    db = module.connection(db_path)
    assert module.inventory(root, db, commit_every=2) == 5
    assert module.inventory(root, db, commit_every=2) == 5
    assert db.execute("SELECT count(*) FROM files").fetchone()[0] == 5

    module.exact_hashes(db, workers=2, commit_every=2)
    assert db.execute(
        "SELECT count(*) FROM files WHERE duplicate_of IS NOT NULL"
    ).fetchone()[0] == 1
    module.perceptual_hashes(db, workers=1, commit_every=2)
    kept, near = module.cluster_near_duplicates(
        db, distance=4, min_side=256, commit_every=2)
    assert kept == 2
    assert near == 1

    manifest = tmp_path / "manifest.jsonl"
    assert module.export_manifest(db, manifest) == 2
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    paths = {record["image"] for record in records}
    assert str(resized.resolve()) in paths
    assert str(other.resolve()) in paths
    assert all(record["task"] == "vision_distillation" for record in records)


def test_bands_cover_full_256_bit_hash():
    value = "0123456789abcdef" * 4
    pieces = module.bands(value)
    band_width = len(value) * 4 // len(pieces)
    rebuilt = sum(part << (band * band_width) for band, part in pieces)
    assert len(pieces) == 8
    assert rebuilt == int(value, 16)


def test_inventory_skips_non_utf8_filesystem_names(tmp_path: Path):
    root = tmp_path / "images"
    valid = root / "valid.jpg"
    make_image(valid, (30, 60, 90))
    invalid = os.fsencode(root) + b"/invalid-\x82.jpg"
    with open(invalid, "wb") as handle:
        handle.write(valid.read_bytes())

    db = module.connection(tmp_path / "dedup.sqlite")
    assert module.inventory(root, db, commit_every=1) == 2
    assert db.execute("SELECT count(*) FROM files").fetchone()[0] == 1
    assert module.statistics(db)["invalid_utf8_paths"] == 1
