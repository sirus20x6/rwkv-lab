#!/usr/bin/env python3
"""Fail when a gpu-marked test has no record of whether it has ever run.

A skip is green. `vars.TRAINVM_SELF_HOSTED` is unset, so the `gpu` job in
`.github/workflows/tests.yml` reports "skipping" on every pull request, and the
only thing that runs a gpu-marked test is `scripts/acceptance.sh --gpu` on a
real host with `--gpu` passed by hand. So on every pull request a gpu test that
has never produced a number and one that ran and passed are the same colour.
`docs/experiment-vm/gpu-test-observations.v2.json` carries the difference, and
this gate is what stops a test from leaving the receipt.

Why this file exists rather than the check it replaces
------------------------------------------------------
The predecessor lived in `tests/test_benchmark_runner.py` and enumerated the
tests it governed with::

    {name for name, value in globals().items() if ...}

`globals()` of one module. Its name promised a repository-wide property and its
implementation delivered a single-module one, so the 32 gpu-marked test
functions defined in the other eleven modules were invisible to the very gate
whose job was to notice unobserved GPU tests. It decided what counted as
observed, and it counted almost nothing.

This is the defect this repository keeps finding -- an instrument returning a
confident answer to a question it was never asked -- so the enumeration is now
the one pytest itself performs::

    pytest tests --collect-only -m gpu -q

Not a re-implementation of marker resolution. A module-level `pytestmark`, a
decorator, a marker inherited from a class and a parametrized case all reach
this gate the same way they reach the runner, because it is the runner that
reports them. `scripts/ci_coverage_gate.py` already collects the whole suite in
CI, so every module named here is known to import on a host with no
accelerator; a collection error is a real failure and is reported as one.

Two kinds of entry, and why the kind is not written down
--------------------------------------------------------
The old entry schema required `fixture`, `measurement.steady_state_step_seconds
> 0` and `measurement.qualified` of every non-null entry. That schema fits a
benchmark grader and nothing else. `test_the_portable_benchmark_receipt_reports
_the_devices_it_opened` reads `/proc` for open accelerator device files: it has
no fixture and returns no qualification verdict, so it could only be enrolled by
inventing both, and an entry whose fields were invented is worse than no entry.

So an entry is one of two shapes:

    graded        last_observed, commit, host, accelerator_device_name,
                  fixture, measurement{steady_state_step_seconds, qualified}
    observation   last_observed, commit, host, accelerator_device_name

and which shape is required is **derived from the test's source**, not declared
in the receipt: a test is graded exactly when its body calls
`record_gpu_grader_observation`, which is the only function that writes a
measurement. A declared kind would be a second copy of a fact the code already
states, free to disagree with it. The observation shape *rejects* `fixture` and
`measurement` rather than merely not requiring them -- the point is that a test
which produces evidence instead of a number cannot be padded into looking like
one.

Either shape may instead be `{"last_observed": null}`, and nothing else. A test
that has never been observed producing anything is a legitimate recorded state
and the one this gate exists to make visible. Expect most entries to be null:
that is the finding, not a blank to be filled in.

There is no exemption list
--------------------------
`scripts/ci_unwired_module_gate.py` documents the failure mode of a gate that
is mostly allowlist, and this gate has no allowlist at all -- not as discipline,
but because there is nothing an exemption could honestly say. Every gpu-marked
test can carry a bare observation, so "exempt" would only ever mean "we chose
not to look at this one", which is the state the receipt exists to expose. If a
future test genuinely cannot be represented, the honest change is a third entry
shape with its own required fields, not a line saying to skip it.

What this gate does not cover
-----------------------------
- **Staleness.** It never asserts that an observation is recent. An age
  threshold can never be satisfied in hosted CI, which has no accelerator, so
  it would fail for a reason no author of any pull request could fix, and it
  would fire on changes that have nothing to do with GPUs. The receipt records;
  it does not police the calendar. `last_observed` and `commit` are stated so a
  reviewer can judge the age themselves.
- **Truth of a recorded number.** It checks that a measurement is present and
  positive. Whether it is a good number is the benchmark authority's question,
  not this one's.
- **Tests outside `tests/`.** The native suites under `trainvm/tests/` have no
  pytest markers and are not enumerated here.
- **Whether a gpu-marked test is meaningfully a GPU test.** The marker is taken
  at face value. A test that is marked `gpu` but never touches a device is
  enrolled like any other, and this gate will not notice.
- **That anyone ever runs the GPU job.** It makes the omission visible in a
  committed file. It cannot cause a run to happen.

Usage:
    python scripts/ci_gpu_observation_gate.py [--tests-dir tests]
"""

from __future__ import annotations

import argparse
import ast
import datetime
import json
import pathlib
import re
import socket
import subprocess
import sys
import warnings

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from scripts.gate_verdict import verdict_line  # noqa: E402

REPOSITORY = pathlib.Path(__file__).resolve().parent.parent
OBSERVATIONS = REPOSITORY / "docs/experiment-vm/gpu-test-observations.v2.json"
API_VERSION = "trainvm.gpu-test-observations/v2"

# The only function that writes a measurement into the receipt. A test that
# calls it is a graded entry; a test that does not is an observation entry.
GRADER_RECORDER = "record_gpu_grader_observation"

GRADED_KEYS = {
    "last_observed", "commit", "host", "accelerator_device_name",
    "fixture", "measurement",
}
OBSERVATION_KEYS = {
    "last_observed", "commit", "host", "accelerator_device_name",
}


def collect_gpu_tests(tests_dir: pathlib.Path) -> set[str]:
    """Enumerate gpu-marked tests as pytest itself resolves them.

    Returns ``tests/<file>.py::<function>`` keys with any parametrization
    stripped, so a parametrized grader is one entry rather than one per case:
    the receipt records that a test produced a number, and the cases of one
    test are observed together.
    """
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", str(tests_dir), "--collect-only",
         "-q", "-p", "no:cacheprovider", "-m", "gpu"],
        capture_output=True, text=True, check=False, cwd=REPOSITORY,
    )
    if completed.returncode not in (0, 5):  # 5 == no tests collected
        sys.stderr.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        raise SystemExit(
            f"gpu collection failed with exit code {completed.returncode}")
    node = re.compile(r"^(tests/[\w/]+\.py)::([\w]+)")
    return {
        f"{match.group(1)}::{match.group(2)}"
        for line in completed.stdout.splitlines()
        if (match := node.match(line.strip()))
    }


def graded_tests(tests_dir: pathlib.Path) -> set[str]:
    """The tests whose own source records a measurement, by reading it."""
    graded: set[str] = set()
    for path in sorted(tests_dir.rglob("test_*.py")):
        try:
            with warnings.catch_warnings():
                # Parsing every test file surfaces each one's own lint
                # warnings, which say nothing about this gate's verdict.
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - a broken test file
            continue
        relative = path.relative_to(REPOSITORY) if path.is_absolute() else path
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not node.name.startswith("test_"):
                continue
            if any(_is_grader_call(call) for call in ast.walk(node)):
                graded.add(f"{relative.as_posix()}::{node.name}")
    return graded


def _is_grader_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    target = node.func
    if isinstance(target, ast.Name):
        return target.id == GRADER_RECORDER
    if isinstance(target, ast.Attribute):
        return target.attr == GRADER_RECORDER
    return False


def check_entry(key: str, entry: object, is_graded: bool) -> list[str]:
    """Every way one entry can be dishonest, reported by name."""
    if not isinstance(entry, dict):
        return [f"{key}: entry is not an object"]
    if "last_observed" not in entry:
        return [f"{key}: entry has no last_observed"]

    observed = entry["last_observed"]
    if observed is None:
        if set(entry) != {"last_observed"}:
            return [f"{key}: an unobserved entry carries fields it cannot "
                    f"have measured: {sorted(set(entry) - {'last_observed'})}"]
        return []

    failures: list[str] = []
    if not isinstance(observed, str) or not re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", observed):
        failures.append(f"{key}: last_observed is not a UTC timestamp")
    else:
        try:
            datetime.datetime.strptime(observed, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            failures.append(f"{key}: last_observed is not a real instant")

    allowed = GRADED_KEYS if is_graded else OBSERVATION_KEYS
    extra = sorted(set(entry) - allowed)
    if extra:
        kind = "a graded" if is_graded else "an observation-only"
        failures.append(
            f"{key}: {kind} entry may not carry {extra}"
            + ("" if is_graded else
               f"; {key} does not call {GRADER_RECORDER}, so it has no "
               "fixture and no qualification verdict to state"))
    for field in ("commit", "host", "accelerator_device_name"):
        if not entry.get(field):
            failures.append(f"{key}: observed entry has no {field}")

    if is_graded:
        if not entry.get("fixture"):
            failures.append(f"{key}: graded entry has no fixture")
        measurement = entry.get("measurement")
        if not isinstance(measurement, dict):
            failures.append(f"{key}: graded entry has no measurement")
        else:
            seconds = measurement.get("steady_state_step_seconds")
            if not isinstance(seconds, (int, float)) or isinstance(
                    seconds, bool) or seconds <= 0:
                failures.append(
                    f"{key}: steady_state_step_seconds is not a positive "
                    "number")
            if not isinstance(measurement.get("qualified"), bool):
                failures.append(f"{key}: qualified is not a boolean")
    return failures


def check(document: dict, collected: set[str], graded: set[str]) -> list[str]:
    """Compare the receipt against the tests that exist, and report drift."""
    failures: list[str] = []
    if document.get("api_version") != API_VERSION:
        failures.append(
            f"api_version is {document.get('api_version')!r}, "
            f"expected {API_VERSION!r}")
    recorded = document.get("tests")
    if not isinstance(recorded, dict):
        return failures + ["the receipt has no tests object"]

    for key in sorted(collected - set(recorded)):
        failures.append(
            f"UNOBSERVED: {key} is gpu-marked and has no entry in "
            f"{OBSERVATIONS.relative_to(REPOSITORY)}; a skip is green, so "
            "without an entry nothing records whether it has ever run")
    for key in sorted(set(recorded) - collected):
        failures.append(
            f"STALE: {key} has an entry but is no longer collected as a "
            "gpu-marked test")
    for key in sorted(set(recorded) & collected):
        failures.extend(check_entry(key, recorded[key], key in graded))
    return failures


def load() -> dict:
    return json.loads(OBSERVATIONS.read_text(encoding="utf-8"))


def _record(key: str, device_name: str, extra: dict) -> None:
    document = load()
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
        cwd=REPOSITORY, check=False)
    document["tests"][key] = {
        "last_observed": datetime.datetime.now(
            datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": revision.stdout.strip() or "unknown",
        "host": socket.gethostname(),
        "accelerator_device_name": device_name,
        **extra,
    }
    OBSERVATIONS.write_text(
        json.dumps(document, indent=2) + "\n", encoding="utf-8")


def record_measurement(
    key: str, fixture: str, device_name: str, measurement: dict,
) -> None:
    """Write down that this grader produced a number, here, just now.

    Rewrites the committed receipt in place. The caller is expected to commit
    the result: the file being in the tree is what makes "never ran" visible,
    and a run on a host without an accelerator leaves it untouched, which is
    the state the receipt exists to expose.
    """
    _record(key, device_name, {"fixture": fixture, "measurement": measurement})


def record_observation(key: str, device_name: str) -> None:
    """Write down that this test ran to completion on a device, just now.

    For a gpu-marked test that produces evidence rather than a number. It
    records the same four facts a graded entry records and stops there, so
    nothing here has to be invented.
    """
    _record(key, device_name, {})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests-dir", default="tests")
    arguments = parser.parse_args()

    tests_dir = pathlib.Path(arguments.tests_dir)
    if not tests_dir.is_absolute():
        tests_dir = REPOSITORY / tests_dir
    if not tests_dir.is_dir():
        raise SystemExit(f"{tests_dir} is not a directory")

    collected = collect_gpu_tests(tests_dir)
    graded = graded_tests(tests_dir)
    document = load()
    failures = check(document, collected, graded)
    for failure in failures:
        print(failure)

    recorded = document.get("tests", {})
    observed = sum(
        1 for entry in recorded.values()
        if isinstance(entry, dict) and entry.get("last_observed"))
    print(verdict_line(
        "gpu observation gate",
        failures,
        f"{len(collected)} gpu-marked tests collected, {len(recorded)} "
        f"enrolled, {observed} ever observed producing anything, "
        f"{len(collected & graded)} graded by measurement",
    ))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
