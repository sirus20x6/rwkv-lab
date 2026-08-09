"""Resolve the trainvm binary that grades benchmark and qualification evidence.

The tests that call `trainvm qualify-evidence` are graders: they let the native
authority decide whether a piece of evidence qualifies. Which binary answers is
therefore part of the result, not an implementation detail.

This module exists because it used to be neither recorded nor controlled. The
resolver preferred `shutil.which("trainvm")` over the build tree, so on a host
with a global install at /usr/local/bin/trainvm the suite graded new source with
an old binary. Three tests failed against a stale global install and passed
against a fresh build of the same tree. Hosted CI has no global install, so
`which` found nothing, the build-tree fallback was taken, and CI could never
surface it — the failure mode existed only on developer and deployment hosts,
which is exactly where a qualification verdict gets read.

Two rules follow, and they are the whole point of this file:

1. PATH is never consulted. A binary on PATH is by construction not the checkout
   under test. There is no case where grading this tree with some other tree's
   binary is the right answer, so the option is removed rather than demoted —
   a deprioritised lookup is still a lookup that fires the day the build tree is
   missing, which is precisely when someone is least likely to check.
2. Whichever binary is chosen is reported, on every run, in the pytest header —
   including the case where none was found. An unrecorded tool choice makes a
   surprising verdict unreproducible by construction.

Not building is not silently fatal, deliberately. The hosted Python jobs cannot
build trainvm at all (it needs GCC 16 with -freflection; only the containerised
native job has that toolchain), so failing whenever no build exists would leave
CI permanently red, which teaches people to ignore it rather than to build. The
loud path is available where it is meaningful: set TRAINVM_BINARY to the binary
that must be used, and a resolution that cannot honour it raises instead of
degrading into a skip.
"""

from __future__ import annotations

import os
import pathlib

REPOSITORY = pathlib.Path(__file__).resolve().parents[1]

# In-checkout build directories, in the order a build is most likely to be
# current. `trainvm/build` is what CLAUDE.md tells you to build; `build-verify`
# and `build-acceptance` are the verification and acceptance trees. All three
# are gitignored, so their presence says nothing about their age — but they are
# at least builds *of this checkout*, which PATH never is.
BUILD_DIRECTORIES = (
    "trainvm/build",
    "trainvm/build-verify",
    "trainvm/build-acceptance",
)

BUILD_INSTRUCTION = (
    "cmake -S trainvm -B trainvm/build -G Ninja -DCMAKE_BUILD_TYPE=Debug && "
    "cmake --build trainvm/build -j \"$(nproc)\" --target trainvm"
)


class TrainvmBinaryError(RuntimeError):
    """An explicitly requested trainvm binary could not be used."""


def _candidate_directories() -> list[str]:
    directories = list(BUILD_DIRECTORIES)
    # scripts/acceptance.sh honours TRAINVM_BUILD_DIR, so a host that builds
    # somewhere else still grades with the binary it just built rather than
    # skipping past it.
    override = os.environ.get("TRAINVM_BUILD_DIR")
    if override:
        directories.insert(0, override)
    return directories


def resolve_trainvm() -> tuple[str | None, str]:
    """Return (binary path or None, a sentence naming how it was resolved).

    Raises TrainvmBinaryError when TRAINVM_BINARY names something unusable: an
    explicit request that cannot be honoured must not quietly become a skip.
    """
    requested = os.environ.get("TRAINVM_BINARY")
    if requested:
        path = pathlib.Path(requested)
        if not path.exists():
            raise TrainvmBinaryError(
                f"TRAINVM_BINARY={requested} does not exist")
        if not os.access(path, os.X_OK):
            raise TrainvmBinaryError(
                f"TRAINVM_BINARY={requested} is not executable")
        return str(path), f"TRAINVM_BINARY={path}"

    for directory in _candidate_directories():
        candidate = pathlib.Path(directory)
        if not candidate.is_absolute():
            candidate = REPOSITORY / candidate
        binary = candidate / "trainvm"
        if binary.exists():
            try:
                relative: str = str(binary.relative_to(REPOSITORY))
            except ValueError:
                relative = str(binary)
            return str(binary), f"built in this checkout at {relative}"

    return None, (
        "not built in this checkout (PATH is deliberately not consulted); "
        f"build it with: {BUILD_INSTRUCTION}"
    )


def report_line() -> str:
    """One line for the pytest header, on every run, whatever the outcome."""
    try:
        binary, description = resolve_trainvm()
    except TrainvmBinaryError as error:
        return f"trainvm gate binary: unusable -- {error}"
    if binary is None:
        return f"trainvm gate binary: none -- {description}"
    return f"trainvm gate binary: {binary} ({description})"
