"""Exercise the native experiment content-lock authoring command."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def run(
    trainvm: Path,
    experiment: Path,
    roots: Path,
    content_cache: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    argv = [str(trainvm), "lock-input-content", str(experiment), str(roots)]
    if content_cache is not None:
        argv += ["--content-cache", str(content_cache)]
    return subprocess.run(argv, check=False, capture_output=True, text=True)


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: verify_input_content_lock.py TRAINVM SOURCE_ROOT")
    trainvm = Path(sys.argv[1]).resolve(strict=True)
    source_root = Path(sys.argv[2]).resolve(strict=True)
    fixture = source_root / "docs/experiment-vm/examples/mageflow-cache-resume.json"
    with tempfile.TemporaryDirectory(prefix="trainvm-content-lock-") as raw:
        temporary = Path(raw)
        input_directory = temporary / "input"
        input_directory.mkdir()
        first = input_directory / "a.txt"
        second = input_directory / "z.txt"
        first.write_text("first", encoding="utf-8")
        second.write_text("second", encoding="utf-8")
        experiment = json.loads(fixture.read_text(encoding="utf-8"))
        experiment["metadata"]["name"] = "content-lock-fixture"
        experiment["spec"]["workspace"].update(
            {
                "root": str(temporary),
                "run_directory": str(temporary / "run"),
                "allowed_read_roots": [str(input_directory)],
                "allowed_write_roots": [str(temporary / "run")],
            }
        )
        experiment["spec"]["workspace"].pop("input_content_roots", None)
        experiment_path = temporary / "experiment.json"
        experiment_path.write_text(
            json.dumps(experiment, indent=2) + "\n", encoding="utf-8"
        )
        roots_path = temporary / "roots.json"
        roots_path.write_text(
            json.dumps(
                {
                    "api_version": "trainvm.input-content-root-set/v1",
                    "paths": [str(second), str(first)],
                }
            ),
            encoding="utf-8",
        )

        locked_process = run(trainvm, experiment_path, roots_path)
        if locked_process.returncode != 0:
            raise SystemExit(locked_process.stderr)
        locked = json.loads(locked_process.stdout)
        identities = locked["spec"]["workspace"]["input_content_roots"]
        if [identity["path"] for identity in identities] != [str(first), str(second)]:
            raise SystemExit("content lock did not canonicalize root order")
        if any(
            identity["api_version"] != "trainvm.input-content-root/v1"
            or identity["kind"] != "file"
            or identity["file_count"] != 1
            or not identity["tree_sha256"].startswith("sha256:")
            for identity in identities
        ):
            raise SystemExit("content lock emitted an invalid reflected identity")
        locked_path = temporary / "locked.json"
        locked_path.write_text(locked_process.stdout, encoding="utf-8")
        subprocess.run(
            [str(trainvm), "validate", str(locked_path)],
            check=True,
            capture_output=True,
            text=True,
        )

        # A digest store must never be able to change the answer, whatever it
        # holds. Locking through one has to reproduce the identity measured
        # without it, and has to keep reproducing it once the store has been
        # written, rewritten with rubbish, and left owner-only on disk.
        store = temporary / "digests.store"
        cached_process = run(trainvm, experiment_path, roots_path, store)
        if cached_process.returncode != 0:
            raise SystemExit(cached_process.stderr)
        if json.loads(cached_process.stdout) != locked:
            raise SystemExit("a cached content lock produced a different document")
        if not store.exists() or store.stat().st_mode & 0o077:
            raise SystemExit("the digest store was not published owner-only")
        repeated = run(trainvm, experiment_path, roots_path, store)
        if repeated.returncode != 0 or json.loads(repeated.stdout) != locked:
            raise SystemExit("a warm content lock produced a different document")
        store.write_bytes(b"not a digest store")
        store.chmod(0o600)
        corrupted = run(trainvm, experiment_path, roots_path, store)
        if corrupted.returncode != 0 or json.loads(corrupted.stdout) != locked:
            raise SystemExit("a corrupt digest store changed the locked identity")
        # A store anyone can write is not evidence of anything, and saying so
        # is more useful than quietly measuring from bytes.
        store.chmod(0o666)
        exposed = run(trainvm, experiment_path, roots_path, store)
        if exposed.returncode == 0:
            raise SystemExit("a world-writable digest store was accepted")
        store.unlink()

        # A store that cannot be written must not discard a lock that already
        # succeeded. The command publishes after it has measured every root and
        # recompiled the document, so a throw there costs the whole run: the
        # operator saw an uncaught exception and lost a locked document that was
        # already correct. Forced here by a directory the process may read and
        # traverse but not create in, which is a real permission mistake and
        # needs no disk to be filled. The store's own safety rules are satisfied
        # -- it is owner-only and not group- or world-writable -- so this fails
        # at the write, which is exactly where a full disk would fail too.
        sealed_directory = temporary / "sealed"
        sealed_directory.mkdir()
        sealed_store = sealed_directory / "digests.store"
        sealed_directory.chmod(0o500)
        try:
            unpublishable = run(trainvm, experiment_path, roots_path, sealed_store)
        finally:
            sealed_directory.chmod(0o700)
        if unpublishable.returncode != 0:
            raise SystemExit(
                "an unpublishable digest store failed a lock that had succeeded: "
                + unpublishable.stderr
            )
        if json.loads(unpublishable.stdout) != locked:
            raise SystemExit(
                "an unpublishable digest store changed or withheld the locked document"
            )
        if "not published" not in unpublishable.stderr:
            raise SystemExit(
                "a store that could not be written was not reported to the operator"
            )
        if sealed_store.exists():
            raise SystemExit("the sealed store was written after all")

        roots_path.write_text(
            json.dumps(
                {
                    "api_version": "trainvm.input-content-root-set/v1",
                    "paths": [str(first)],
                    "unknown": True,
                }
            ),
            encoding="utf-8",
        )
        malformed = run(trainvm, experiment_path, roots_path)
        if malformed.returncode == 0 or "reflected schema" not in malformed.stderr:
            raise SystemExit("content lock accepted an open root-set document")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
