import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/assemble_vision_cache_overlay.py"
SPEC = importlib.util.spec_from_file_location("vision_cache_overlay", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_hard_link_overlay_is_idempotent(tmp_path):
    first, second, output = (tmp_path / name for name in ("first", "second", "out"))
    first.mkdir()
    second.mkdir()
    (first / "a.pt").write_bytes(b"a")
    (second / "b.pt").write_bytes(b"b")
    assert MODULE.link_cache_sources([first, second], output) == 2
    assert MODULE.link_cache_sources([first, second], output) == 2
    assert (output / "a.pt").samefile(first / "a.pt")
    assert (output / "b.pt").samefile(second / "b.pt")


def test_overlay_rejects_non_link_collision(tmp_path):
    source, output = tmp_path / "source", tmp_path / "out"
    source.mkdir()
    output.mkdir()
    (source / "same.pt").write_bytes(b"source")
    (output / "same.pt").write_bytes(b"other")
    with pytest.raises(RuntimeError, match="collision"):
        MODULE.link_cache_sources([source], output)


def test_manifest_images_resolves_and_deduplicates(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"image")
    manifest = tmp_path / "rows.jsonl"
    manifest.write_text(
        json.dumps({"image": str(image), "text": "one"}) + "\n"
        + json.dumps({"image": str(image), "text": "two"}) + "\n")
    assert MODULE.manifest_images([manifest]) == {image.resolve()}
