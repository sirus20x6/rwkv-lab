from __future__ import annotations

import argparse
import http.client
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"test_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_hf_revision_is_persisted_before_repository_indexing(tmp_path, monkeypatch):
    fetch = load_script("fetch_i1_sources.py")
    output = tmp_path / "sources"
    args = argparse.Namespace(
        output=output, sources=["pexels"], chunk_gib=1.0, reserve_tib=0.0,
        max_chunks=1, wait_for_space=False, space_poll_seconds=1,
        dry_run=False,
    )

    class Info:
        sha = "fixed-revision"

    class API:
        def dataset_info(self, repo_id):
            return Info()

    monkeypatch.setattr(fetch, "parse_args", lambda: args)
    monkeypatch.setattr(fetch, "HfApi", API)
    monkeypatch.setattr(
        fetch, "repository_files",
        lambda *unused: (_ for _ in ()).throw(RuntimeError("index failed")),
    )
    with pytest.raises(RuntimeError, match="index failed"):
        fetch.main()

    state = json.loads((output / fetch.STATE_NAME).read_text())
    assert state["sources"]["pexels"]["revision"] == "fixed-revision"


@pytest.mark.parametrize("items, message", [
    ([], "tree is empty"),
    ([type("Item", (), {"path": "../escape", "size": 1})()], "unsafe"),
    ([type("Item", (), {"path": "same", "size": 1})(),
      type("Item", (), {"path": "same", "size": 1})()], "duplicate"),
])
def test_hf_repository_index_fails_closed_on_unusable_tree(items, message):
    fetch = load_script("fetch_i1_sources.py")

    class API:
        def list_repo_tree(self, *args, **kwargs):
            return items

    with pytest.raises(RuntimeError, match=message):
        fetch.repository_files(API(), fetch.HF_SOURCES["pexels"], "revision")


def test_archive_publication_rejects_wrong_size_and_renames_exact_payload(tmp_path):
    fetch = load_script("fetch_i1_archives.py")
    partial = tmp_path / "archive.part"
    final = tmp_path / "archive.tar"
    partial.write_bytes(b"short")
    with pytest.raises(RuntimeError, match="size verification failed"):
        fetch.publish_complete_archive(partial, final, 10)
    assert partial.is_file() and not final.exists()

    fetch.publish_complete_archive(partial, final, 5)
    assert final.read_bytes() == b"short"
    assert not partial.exists()


def test_acquisition_state_rejects_destination_remapping():
    hf = load_script("fetch_i1_sources.py")
    state = {"sources": {
        "pexels": {"repo_id": "repo", "directory": "old"},
    }}
    with pytest.raises(RuntimeError, match="state source mismatch"):
        hf.pin_source(
            object(), "pexels", hf.Source("repo", "new"), state)

    archives = load_script("fetch_i1_archives.py")
    state = {"sources": {
        "images": {"url": "https://example.test/data", "path": "old.tar",
                   "expected_bytes": 10},
    }}
    with pytest.raises(RuntimeError, match="state mismatch"):
        archives.archive_state_entry(
            state, "images",
            archives.Archive("https://example.test/data", "new.tar", 10),
        )


@pytest.mark.parametrize("script", ["fetch_i1_sources.py", "fetch_i1_archives.py"])
def test_acquisition_state_rejects_unknown_or_malformed_schema(tmp_path, script):
    fetch = load_script(script)
    path = tmp_path / "state.json"
    for payload in (
        {"version": 2, "sources": {}, "chunks": []},
        {"version": 1, "sources": [], "chunks": []},
        {"version": 1, "sources": {}, "chunks": {}},
    ):
        path.write_text(json.dumps(payload))
        with pytest.raises(RuntimeError, match="unsupported or corrupt"):
            fetch.read_state(path)


class _RangeResponse:
    def __init__(self, payload: bytes, *, start: int, end: int, total: int,
                 etag: str | None):
        self.status = 206
        self.headers = {"Content-Range": f"bytes {start}-{end}/{total}"}
        if etag is not None:
            self.headers["ETag"] = etag
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        return False

    def read(self, unused_size):
        payload, self._payload = self._payload, b""
        return payload


def test_archive_ranges_reject_same_size_remote_generation_change(
        tmp_path, monkeypatch):
    fetch = load_script("fetch_i1_archives.py")
    output = tmp_path / "archive.part"
    responses = iter((
        _RangeResponse(b"aaaa", start=0, end=3, total=8, etag='"generation-a"'),
        _RangeResponse(b"bbbb", start=4, end=7, total=8, etag='"generation-b"'),
    ))
    monkeypatch.setattr(fetch.urllib.request, "urlopen", lambda *args, **kwargs: next(responses))
    pinned = []

    received, etag = fetch.fetch_range(
        "https://example.test/archive", output, 0, 3, expected_total=8,
        pin_etag=pinned.append)
    assert (received, etag, pinned) == (4, '"generation-a"', ['"generation-a"'])
    assert output.read_bytes() == b"aaaa"

    with pytest.raises(RuntimeError, match="ETag changed"):
        fetch.fetch_range(
            "https://example.test/archive", output, 4, 7,
            expected_total=8, expected_etag=etag)
    assert output.read_bytes() == b"aaaa"


@pytest.mark.parametrize("etag", [None, 'W/"weak"'])
def test_archive_ranges_fail_closed_without_strong_etag(
        tmp_path, monkeypatch, etag):
    fetch = load_script("fetch_i1_archives.py")
    response = _RangeResponse(b"data", start=0, end=3, total=4, etag=etag)
    monkeypatch.setattr(fetch.urllib.request, "urlopen", lambda *args, **kwargs: response)

    with pytest.raises(RuntimeError, match="strong ETag"):
        fetch.fetch_range(
            "https://example.test/archive", tmp_path / "archive.part",
            0, 3, expected_total=4)
    assert not (tmp_path / "archive.part").exists()


def test_archive_range_validates_total_object_size(tmp_path, monkeypatch):
    fetch = load_script("fetch_i1_archives.py")
    response = _RangeResponse(
        b"data", start=0, end=3, total=5, etag='"generation"')
    monkeypatch.setattr(fetch.urllib.request, "urlopen", lambda *args, **kwargs: response)

    with pytest.raises(RuntimeError, match="Content-Range"):
        fetch.fetch_range(
            "https://example.test/archive", tmp_path / "archive.part",
            0, 3, expected_total=4)


def test_archive_range_rejects_an_overlong_response_body(tmp_path, monkeypatch):
    fetch = load_script("fetch_i1_archives.py")
    output = tmp_path / "archive.part"
    response = _RangeResponse(
        b"five!", start=0, end=3, total=4, etag='"generation"')
    monkeypatch.setattr(fetch.urllib.request, "urlopen", lambda *args, **kwargs: response)

    with pytest.raises(RuntimeError, match="overran"):
        fetch.fetch_range(
            "https://example.test/archive", output, 0, 3,
            expected_total=4)
    assert output.stat().st_size == 0


def test_archive_short_eof_reissues_remaining_range_with_if_range(
        tmp_path, monkeypatch):
    fetch = load_script("fetch_i1_archives.py")
    output = tmp_path / "archive.part"
    responses = iter((
        _RangeResponse(b"ab", start=0, end=3, total=4, etag='"generation"'),
        _RangeResponse(b"cd", start=2, end=3, total=4, etag='"generation"'),
    ))
    requests = []

    def open_response(request, **unused):
        requests.append(request)
        return next(responses)

    monkeypatch.setattr(fetch.urllib.request, "urlopen", open_response)
    received, etag = fetch.fetch_range(
        "https://example.test/archive", output, 0, 3, expected_total=4)

    assert (received, etag, output.read_bytes()) == (4, '"generation"', b"abcd")
    assert requests[0].get_header("If-range") is None
    assert requests[1].get_header("If-range") == '"generation"'
    assert requests[1].get_header("Range") == "bytes=2-3"


@pytest.mark.parametrize("code, reason", [
    (404, "Not Found"), (403, "Forbidden"), (410, "Gone"),
])
def test_archive_permanent_http_failures_are_not_retried(
        tmp_path, monkeypatch, code, reason):
    fetch = load_script("fetch_i1_archives.py")
    calls = {"count": 0}

    def open_error(request, **unused):
        calls["count"] += 1
        raise fetch.urllib.error.HTTPError(
            request.full_url, code, reason, None, None)

    monkeypatch.setattr(fetch.urllib.request, "urlopen", open_error)
    with pytest.raises(RuntimeError, match="permanent HTTP failure"):
        fetch.fetch_range(
            "https://example.test/archive", tmp_path / "archive.part",
            0, 3, expected_total=4)
    assert calls["count"] == 1


@pytest.mark.parametrize("code", [503, 429, 408])
def test_archive_transient_failures_are_capped_at_six(
        tmp_path, monkeypatch, code):
    fetch = load_script("fetch_i1_archives.py")
    calls = {"count": 0}

    def open_error(request, **unused):
        calls["count"] += 1
        raise fetch.urllib.error.HTTPError(
            request.full_url, code, "unavailable", None, None)

    monkeypatch.setattr(fetch.urllib.request, "urlopen", open_error)
    monkeypatch.setattr(fetch.time, "sleep", lambda unused: None)
    with pytest.raises(RuntimeError, match="consecutive transient failures"):
        fetch.fetch_range(
            "https://example.test/archive", tmp_path / "archive.part",
            0, 3, expected_total=4)
    assert calls["count"] == 6


def test_archive_transient_cap_resets_when_bytes_advance(tmp_path, monkeypatch):
    fetch = load_script("fetch_i1_archives.py")
    output = tmp_path / "archive.part"

    class _ShortRead(_RangeResponse):
        """Yield one byte, then die: progress, so retries must not be capped."""

        def __init__(self, start):
            payload = bytes([65 + start])
            super().__init__(payload, start=start, end=7, total=8,
                             etag='"generation"')
            self.calls = 0

        def read(self, unused_size):
            self.calls += 1
            if self.calls == 1:
                return self._payload
            raise http.client.IncompleteRead(b"")

    responses = iter(_ShortRead(start) for start in range(8))
    monkeypatch.setattr(
        fetch.urllib.request, "urlopen", lambda *args, **kwargs: next(responses))
    monkeypatch.setattr(fetch.time, "sleep", lambda unused: None)
    received, _ = fetch.fetch_range(
        "https://example.test/archive", output, 0, 7, expected_total=8)
    assert received == 8
    assert output.read_bytes() == b"ABCDEFGH"


class _InterruptedRangeResponse(_RangeResponse):
    def __init__(self):
        super().__init__(b"", start=0, end=3, total=4, etag='"generation"')
        self.calls = 0

    def read(self, unused_size):
        self.calls += 1
        if self.calls == 1:
            return b"ab"
        raise http.client.IncompleteRead(b"")


def test_archive_transient_partial_is_fsynced_before_resume(
        tmp_path, monkeypatch):
    fetch = load_script("fetch_i1_archives.py")
    output = tmp_path / "archive.part"
    responses = iter((
        _InterruptedRangeResponse(),
        _RangeResponse(b"cd", start=2, end=3, total=4, etag='"generation"'),
    ))
    monkeypatch.setattr(fetch.urllib.request, "urlopen", lambda *args, **kwargs: next(responses))
    monkeypatch.setattr(fetch.time, "sleep", lambda unused: None)
    real_fsync = fetch.os.fsync
    fsync_sizes = []

    def observe_fsync(descriptor):
        fsync_sizes.append(output.stat().st_size)
        real_fsync(descriptor)

    monkeypatch.setattr(fetch.os, "fsync", observe_fsync)
    received, _ = fetch.fetch_range(
        "https://example.test/archive", output, 0, 3, expected_total=4)

    assert received == 4 and output.read_bytes() == b"abcd"
    assert 2 in fsync_sizes
