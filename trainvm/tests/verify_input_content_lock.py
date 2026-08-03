"""Exercise the native experiment content-lock authoring command."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def run(trainvm: Path, experiment: Path, roots: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(trainvm), "lock-input-content", str(experiment), str(roots)],
        check=False,
        capture_output=True,
        text=True,
    )


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
