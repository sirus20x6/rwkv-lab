"""The runtime-closure builder's inode digest cache.

The closure this builder writes is what decides whether a host may run a sealed
worker, so a cache that serves one stale digest is an incorrect authorisation
rather than a slow test. These tests are about the conditions under which a
cached digest may be reused at all.
"""

from __future__ import annotations

import importlib.util
import os
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


def _counted(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count actual byte reads, which is the observable the cache changes."""
    calls = [0]
    original = MODULE._hash_descriptor

    def counting(descriptor: int) -> str:
        calls[0] += 1
        return original(descriptor)

    monkeypatch.setattr(MODULE, "_hash_descriptor", counting)
    return calls


def test_digest_cache_reuses_only_an_unchanged_attested_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "runtime.bin"
    source.write_bytes(b"first-runtime")
    source.chmod(0o600)
    cache: dict[str, str] = {}
    calls = _counted(monkeypatch)

    first = MODULE._entry(source, cache)
    second = MODULE._entry(source, cache)
    assert first == second
    assert calls[0] == 1

    source.write_bytes(b"other-runtime")
    source.chmod(0o600)
    changed = MODULE._entry(source, cache)
    assert changed["sha256"] != first["sha256"]
    assert calls[0] == 2


def test_digest_cache_misses_when_only_ctime_separates_the_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rewrite that restores mtime and size must still miss the cache.

    This is the attack the key exists to stop, and it needs no privilege: the
    file's own owner rewrites the bytes and then calls ``utimensat`` (here via
    ``os.utime``) to put mtime back exactly where it was. Size is held equal by
    choosing a replacement of the same length, and the inode is preserved by
    writing in place rather than replacing the file. Every field of the
    identity except ``st_ctime_ns`` is therefore identical across the mutation,
    so if the digest still changes, it changed *because of* ``st_ctime_ns``.
    """
    source = tmp_path / "runtime.bin"
    source.write_bytes(b"first-runtime")
    source.chmod(0o600)
    before = source.lstat()
    cache: dict[str, str] = {}
    calls = _counted(monkeypatch)

    original = MODULE._entry(source, cache)
    assert calls[0] == 1

    # In place: same inode, same length, mtime restored to the nanosecond.
    descriptor = os.open(source, os.O_WRONLY)
    try:
        os.pwrite(descriptor, b"other-runtime", 0)
    finally:
        os.close(descriptor)
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))

    after = source.lstat()
    assert after.st_ino == before.st_ino
    assert after.st_size == before.st_size
    assert after.st_mtime_ns == before.st_mtime_ns
    assert after.st_ctime_ns != before.st_ctime_ns

    mutated = MODULE._entry(source, cache)
    assert calls[0] == 2, "a forged mtime served a cached digest"
    assert mutated["sha256"] != original["sha256"]


def test_digest_cache_round_trip_is_owner_only_and_closed(tmp_path: Path) -> None:
    path = tmp_path / "digests.json"
    entries = {"1:2:3:4:5:6:7:8": "sha256:" + "a" * 64}
    MODULE._publish_digest_cache(path, entries)
    assert MODULE._load_digest_cache(path) == entries
    path.chmod(0o660)
    with pytest.raises(ValueError, match="owner-only"):
        MODULE._load_digest_cache(path)


@pytest.mark.parametrize(
    "document",
    [
        {"api_version": "trainvm.runtime-closure-digest-cache/v2", "entries": {}},
        {"api_version": MODULE.DIGEST_CACHE_SCHEMA},
        {"api_version": MODULE.DIGEST_CACHE_SCHEMA, "entries": {}, "extra": 1},
        {"api_version": MODULE.DIGEST_CACHE_SCHEMA, "entries": {"k": "sha256:zz"}},
        {"api_version": MODULE.DIGEST_CACHE_SCHEMA, "entries": {"": "sha256:" + "a" * 64}},
        {
            "api_version": MODULE.DIGEST_CACHE_SCHEMA,
            "entries": {"k": "sha256:" + "A" * 64},
        },
    ],
)
def test_digest_cache_rejects_a_malformed_document(
    tmp_path: Path, document: dict[str, object]
) -> None:
    path = tmp_path / "digests.json"
    MODULE._publish(path, MODULE._canonical(document) + b"\n")
    with pytest.raises(ValueError, match="runtime digest cache"):
        MODULE._load_digest_cache(path)


def test_missing_digest_cache_is_cold_rather_than_an_error(tmp_path: Path) -> None:
    assert MODULE._load_digest_cache(tmp_path / "absent.json") == {}
    assert MODULE._load_digest_cache(None) == {}
