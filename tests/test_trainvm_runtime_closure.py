from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_trainvm_runtime_closure.py"
)
SPEC = importlib.util.spec_from_file_location("trainvm_runtime_closure", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_digest_cache_reuses_only_an_unchanged_attested_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "runtime.bin"
    source.write_bytes(b"first-runtime")
    source.chmod(0o600)
    cache: dict[str, str] = {}
    calls = 0
    original = MODULE._hash_descriptor

    def counted(descriptor: int) -> str:
        nonlocal calls
        calls += 1
        return original(descriptor)

    monkeypatch.setattr(MODULE, "_hash_descriptor", counted)
    first = MODULE._entry(source, cache)
    second = MODULE._entry(source, cache)
    assert first == second
    assert calls == 1

    source.write_bytes(b"other-runtime")
    source.chmod(0o600)
    changed = MODULE._entry(source, cache)
    assert changed["sha256"] != first["sha256"]
    assert calls == 2


def test_digest_cache_round_trip_is_owner_only_and_closed(tmp_path: Path) -> None:
    path = tmp_path / "digests.json"
    entries = {"1:2:3:4:5:6:7:8": "sha256:" + "a" * 64}
    MODULE._publish_digest_cache(path, entries)
    assert MODULE._load_digest_cache(path) == entries
    path.chmod(0o660)
    with pytest.raises(ValueError, match="owner-only"):
        MODULE._load_digest_cache(path)
