#!/usr/bin/env python3
"""Real Python-produced driver identities, for the C++ validator to judge.

`fixed_public_identity` in trainvm/src/cache_namespace.cpp decides which
characters may appear in a cache namespace claim. `driver_identity()` in
src/rwkv_lab/trainvm_runtime_guard.py produces a string that has to satisfy it,
and until PR #162 it did not: the identity was the whole first line of
/proc/driver/nvidia/version, which carries a space, `(`, `)` and `@`. On any
real NVIDIA host `validate_evidence` threw and the claim could not be made.

Nothing caught that in either language, because each half was tested against
its own assumption about the other: the native tests wrote
`.driver_version = "610.43.03"`, a value no Python code emits, and the Python
tests asserted the producer's shape without knowing what the validator accepts.
The first fix restated the character set a second time, in Python, which pins
today's answer rather than the rule.

So this file is a producer, not an assertion. It runs the *real*
`driver_identity()` over report text it plants itself, and prints what came
out. `trainvm/tests/driver_identity_namespace_tests.cpp` reads that and pushes
each identity through the *real* `derive_cache_namespace_claim`. The character
set is stated once, in C++, and the Python side never restates it: widen or
narrow `fixed_public_identity` and the native test moves, change the shape of
the identity and the same native test moves, because both sides are driven over
one input.

Two constraints shaped it, and both are still live:

* The guard imports nothing outside the standard library and runs before the
  closure is trusted, so it cannot consult the native binary at runtime. This
  is a test-time agreement; nothing here is imported by production code.
* No GPU and no NVIDIA host. Every report line below is planted in a temporary
  file and the guard's two path constants are pointed at it, so the identities
  are the same on a machine with no driver as on the training host. A test that
  skipped without a driver would make the whole arrangement decorative.

Scope note, deliberately not papered over: every case here is a report line
whose version token `driver_identity()` recognises. A line it does *not*
recognise falls back to keeping the whole line, which reproduces the original
defect in miniature -- the fallback identity carries spaces and is rejected
downstream. That is a real gap, filed as `card-e1a0a6eb` rather than hidden by
dropping the assertion, and it is why no unrecognised-report case appears below.
When that card is decided, its case belongs here, where the real validator will
judge it.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from typing import Any

REPOSITORY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GUARD_PATH = os.path.join(REPOSITORY, "src", "rwkv_lab", "trainvm_runtime_guard.py")

_NT_GNU_BUILD_ID = 3


# The report lines are verbatim shapes, not paraphrases. The first is what this
# host serves; the parenthesised build user and host at the end of it are the
# characters that made the whole line illegal downstream.
CASES: tuple[dict[str, Any], ...] = (
    {
        "case": "open_kernel_module",
        "report": (
            "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  610.43.03  "
            "Release Build  (root@neuromancer)"
        ),
        "build_id": "85084b3554f6473f1b2cd8dc94f54f11dd60977e",
    },
    {
        "case": "open_kernel_module_rebuilt_elsewhere",
        "report": (
            "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  610.43.03  "
            "Release Build  (builder@some-distribution-farm)"
        ),
        "build_id": "85084b3554f6473f1b2cd8dc94f54f11dd60977e",
    },
    {
        "case": "proprietary_module",
        "report": (
            "NVRM version: NVIDIA UNIX x86_64 Kernel Module  580.82.07  "
            "Tue Aug  5 17:24:00 UTC 2025"
        ),
        "build_id": "d41d8cd98f00b204e9800998ecf8427e11223344",
    },
    {
        # A module linked with --build-id=none exposes no note, and the
        # identity is the version token alone. Legal for a different reason
        # than the others, so it is worth its own trip through the validator.
        "case": "module_without_a_build_id",
        "report": (
            "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  610.43.03  "
            "Release Build  (root@neuromancer)"
        ),
        "build_id": None,
    },
)


def load_guard() -> Any:
    """The guard module, loaded by path rather than as `rwkv_lab.*`.

    Importing the package would pull its `__init__`, and this runs from a
    native test with nothing installed. The guard itself imports only the
    standard library, so a bare file load is enough.
    """

    specification = importlib.util.spec_from_file_location(
        "trainvm_runtime_guard_fixture", GUARD_PATH
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"could not load the runtime guard from {GUARD_PATH}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def build_id_note(description: bytes) -> bytes:
    """An `NT_GNU_BUILD_ID` note as /sys/module/<name>/notes/ serves it."""

    return (
        len(b"GNU\x00").to_bytes(4, sys.byteorder)
        + len(description).to_bytes(4, sys.byteorder)
        + _NT_GNU_BUILD_ID.to_bytes(4, sys.byteorder)
        + b"GNU\x00"
        + description
    )


def identities(guard: Any | None = None) -> list[dict[str, str]]:
    """Run the real `driver_identity()` over every case.

    Returns one record per case carrying the planted report line and the
    identity the guard produced from it. The guard's two path constants are
    restored afterwards, so a caller that passes its own already-imported
    module is not left with them pointing at a deleted directory.
    """

    module = load_guard() if guard is None else guard
    original = (
        module.NVIDIA_DRIVER_VERSION_FILE,
        module.NVIDIA_MODULE_BUILD_ID_FILE,
    )
    produced: list[dict[str, str]] = []
    try:
        with tempfile.TemporaryDirectory() as directory:
            version_file = os.path.join(directory, "nvidia-version")
            note_file = os.path.join(directory, "nvidia-build-id")
            module.NVIDIA_DRIVER_VERSION_FILE = version_file
            module.NVIDIA_MODULE_BUILD_ID_FILE = note_file
            for case in CASES:
                with open(version_file, "w", encoding="utf-8") as stream:
                    stream.write(case["report"] + "\n")
                if case["build_id"] is None:
                    if os.path.exists(note_file):
                        os.unlink(note_file)
                else:
                    with open(note_file, "wb") as binary:
                        binary.write(build_id_note(bytes.fromhex(case["build_id"])))
                report = module.driver_report()
                identity = module.driver_identity()
                if report is None or identity is None:
                    raise RuntimeError(f"case {case['case']} produced no identity")
                produced.append(
                    {"case": case["case"], "report": report, "identity": identity}
                )
    finally:
        (
            module.NVIDIA_DRIVER_VERSION_FILE,
            module.NVIDIA_MODULE_BUILD_ID_FILE,
        ) = original
    return produced


def main() -> int:
    json.dump(identities(), sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
