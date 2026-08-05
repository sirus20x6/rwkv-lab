#!/usr/bin/env python3
"""Choose the smallest honest native-CI tier for a Git change.

The classifier is intentionally conservative.  Paths known to affect only the
catalog/source-identity boundary select a CLI-only build; paths known to be
covered by other jobs select no native build; native and unfamiliar paths
select the full build and ctest suite.  An unknown path can therefore cost
time, but it can never silently remove coverage.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import PurePosixPath
import subprocess
import sys
from typing import Iterable


MODES = {"none": 0, "catalog": 1, "full": 2}
NATIVE_SUFFIXES = {".c", ".cc", ".cmake", ".cpp", ".cxx", ".h", ".hpp", ".proto"}

FULL_FILES = {
    ".github/workflows/tests.yml",
    "CMakeLists.txt",
    "docs/experiment-vm/native-ci-exclusions.v1.json",
    "pyproject.toml",
    "scripts/acceptance.sh",
    "scripts/build_trainvm_runtime_closure.py",
    "scripts/build_trainvm_worker_artifact.py",
    "scripts/materialize_trainvm_worker_deployment.py",
    "scripts/validate_native_ci_exclusions.py",
}
NONE_FILES = {
    ".gitignore",
    "CLAUDE.md",
    "LICENSE",
    "README.md",
}


def _valid_path(value: str) -> bool:
    if not value or "\\" in value or "\n" in value or "\r" in value:
        return False
    path = PurePosixPath(value)
    return (
        not path.is_absolute()
        and str(path) == value
        and all(part not in {"", ".", ".."} for part in path.parts)
    )


def path_mode(value: str) -> tuple[str, str]:
    """Return ``(mode, reason)`` for one repository-relative path."""

    if not _valid_path(value):
        return "full", "malformed or non-normalized path"
    path = PurePosixPath(value)
    if value in FULL_FILES:
        return "full", "native CI contract or build input"
    if value.startswith("trainvm/") or value.startswith(".github/docker/"):
        return "full", "native source, test, build, or toolchain input"
    if path.suffix in NATIVE_SUFFIXES:
        return "full", "native-language or CMake input"
    if value.startswith("docs/experiment-vm/examples/"):
        return "full", "native compiler/registry fixture"
    if value.startswith(".github/workflows/"):
        return "full", "unclassified workflow change"

    # These trees can contain bytes referenced by compatibility-workflows.v1.
    # The light tier builds the real CLI and validates that catalog against the
    # checkout, while their own behavior is covered by Python/Go/schema jobs.
    if value.startswith(("src/", "scripts/", "dashboard/", "docs/")):
        return "catalog", "catalog/source-identity input"

    if value in NONE_FILES or value.startswith("tests/"):
        return "none", "covered by non-native CI"

    # This is the safety property: adding a new top-level tree, build language,
    # or artifact type cannot teach CI to skip itself.
    return "full", "unrecognized path (fail closed)"


def classify(paths: Iterable[str]) -> dict[str, object]:
    decisions = []
    mode = "none"
    for value in sorted(set(paths)):
        selected, reason = path_mode(value)
        decisions.append({"path": value, "mode": selected, "reason": reason})
        if MODES[selected] > MODES[mode]:
            mode = selected
    return {"api_version": "trainvm.native-ci-change-scope/v1", "mode": mode, "paths": decisions}


def git_changed_paths(base: str, head: str) -> list[str]:
    completed = subprocess.run(
        ["git", "diff", "--name-only", "-z", base, head, "--"],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        # A missing/shallow base is not permission to skip.  The synthetic path
        # is deliberately unknown and therefore selects the full tier.
        sys.stderr.write(completed.stderr.decode("utf-8", errors="replace"))
        return ["<git-diff-unavailable>"]
    return [item.decode("utf-8", errors="surrogateescape") for item in completed.stdout.split(b"\0") if item]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--github-output")
    arguments = parser.parse_args()

    receipt = classify(git_changed_paths(arguments.base, arguments.head))
    encoded = json.dumps(receipt, sort_keys=True, separators=(",", ":"))
    print(encoded)
    if arguments.github_output:
        output = os.path.abspath(arguments.github_output)
        with open(output, "a", encoding="utf-8") as stream:
            stream.write(f"mode={receipt['mode']}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
