#!/usr/bin/env python3
"""Regenerate the adapter-registry half of the step-zero arming authority.

`docs/experiment-vm/step-zero-arming.v1.json` records, per registered adapter
contract, the three registry facts that decide whether a route *can* arm the
universal step-zero eval gate:

- whether its resolved training composition carries an `evaluator` slot,
- whether its authoring declaration publishes a checkpoint artifact,
- whether it declares an `eval_examples` output port, and with what
  `required` / `artifact_type` / `artifact_schema`.

Those facts live in `trainvm/src/rwkv_lab_worker_contract.cpp`, behind builder
functions, inheritance (`checkpoint_authoring()` then `.emplace`) and a loop
over eight Transformer MLA contracts. Reading them out of the source text would
be a parser pretending to be a compiler, so this reads them out of the built
registry -- `trainvm inspect-rwkv-lab-worker` -- exactly like
`trainvm/tests/verify_rwkv_lab_worker_contract.py` does.

Why a pin at all
----------------
The gate that consumes these facts, `scripts/ci_step_zero_arming_gate.py`, runs
in the seconds-fast schema job, which has no GCC 16 and cannot build trainvm.
A gate that lived only where the binary exists would run behind an ~8 minute
build the PR-tier classifier can skip -- and the pull request most likely to
change an arming answer is a JSON edit to a recipe-profile catalog, which is
the one least likely to be waiting on a native build.

So the registry half is pinned here and checked in the native job (ctest
`step_zero_arming_pin`), and the composition-document half is recomputed from
the documents on every pull request. Neither half is hand-maintained.

Usage:
    python scripts/print_step_zero_arming_pin.py TRAINVM [--check | --write]

`--check` is the ctest mode: it fails when the pin disagrees with the registry
the binary was built from. `--write` regenerates the pin. Without either it
prints what the registry says.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from scripts.gate_verdict import verdict_line  # noqa: E402

PIN = "docs/experiment-vm/step-zero-arming.v1.json"
API_VERSION = "trainvm.step-zero-arming/v1"

# The schema `invocation_requires_step_zero_eval_gate` compares against. Named
# once here and once in the gate, both quoting `kEvalExamplesSchema` in
# `trainvm/include/trainvm/eval_examples_contract.hpp`; the pin carries the
# value it observed, so a change in the C++ constant reddens `--check` rather
# than silently agreeing with a new spelling.
EVAL_EXAMPLES_SCHEMA = "rwkv-lab.eval-examples.v1"

# A fingerprint is a required positional argument of the inspect subcommand and
# does not affect any field this pin reads.
INSPECT_FINGERPRINT = "sha256:" + "a" * 64


def registry_profiles(binary: str) -> list[dict]:
    document = json.loads(
        subprocess.check_output(
            [binary, "inspect-rwkv-lab-worker", INSPECT_FINGERPRINT], text=True
        )
    )
    return document["adapter_registry"]["profiles"]


def profile_facts(profile: dict) -> dict:
    """The arming-relevant projection of one registry profile.

    Deliberately a projection and not a copy: pinning the whole profile would
    make every unrelated registry edit -- a description reworded, a lifecycle
    flag flipped -- a diff in this document, and a pin that churns is a pin
    people regenerate without reading.
    """
    composition = profile.get("training_composition") or {}
    slots = composition.get("slots") or {}
    outputs = (profile.get("authoring") or {}).get("outputs") or {}
    port = None
    for name, descriptor in sorted(outputs.items()):
        if descriptor.get("artifact_type") == "eval_examples":
            port = {
                "output_name": name,
                "required": bool(descriptor.get("required", False)),
                "artifact_schema": descriptor.get("artifact_schema"),
            }
            break
    checkpoint = sorted(
        name
        for name, descriptor in outputs.items()
        if descriptor.get("artifact_type") == "checkpoint"
    )
    return {
        "contract": profile["key"]["contract"],
        "adapter": profile["key"]["adapter"],
        "stateful": bool((profile.get("lifecycle") or {}).get("stateful", False)),
        # The slot map is name -> category. The evaluator is identified by its
        # category, not by a slot happening to be spelled "evaluator": a route
        # may name the slot anything, and `validate_eval_examples_gate_provenance`
        # requires exactly one resolved component of category `evaluator`.
        "evaluator_slots": sorted(
            name for name, category in slots.items() if category == "evaluator"
        ),
        "checkpoint_outputs": checkpoint,
        "eval_examples_output": port,
    }


def build(binary: str) -> dict:
    profiles = [profile_facts(profile) for profile in registry_profiles(binary)]
    profiles.sort(key=lambda entry: entry["contract"])
    return {
        "api_version": API_VERSION,
        "authority": "trainvm inspect-rwkv-lab-worker -> adapter_registry.profiles",
        "regenerated_by": "python scripts/print_step_zero_arming_pin.py TRAINVM --write",
        "eval_examples_schema": EVAL_EXAMPLES_SCHEMA,
        "profiles": profiles,
    }


def differences(pinned: dict, observed: dict) -> list[str]:
    problems: list[str] = []
    if pinned.get("api_version") != observed["api_version"]:
        problems.append(
            f"{PIN}: api_version is {pinned.get('api_version')!r}, "
            f"the generator emits {observed['api_version']!r}"
        )
    if pinned.get("eval_examples_schema") != observed["eval_examples_schema"]:
        problems.append(
            f"{PIN}: eval_examples_schema is "
            f"{pinned.get('eval_examples_schema')!r}, the registry uses "
            f"{observed['eval_examples_schema']!r}"
        )
    by_contract = {entry["contract"]: entry for entry in pinned.get("profiles", [])}
    live = {entry["contract"]: entry for entry in observed["profiles"]}
    for contract in sorted(set(live) - set(by_contract)):
        problems.append(
            f"{PIN}: the registry has {contract} and the pin does not; "
            "regenerate with --write"
        )
    for contract in sorted(set(by_contract) - set(live)):
        problems.append(
            f"{PIN}: pins {contract}, which the registry no longer declares"
        )
    for contract in sorted(set(by_contract) & set(live)):
        for field in sorted(set(live[contract]) | set(by_contract[contract])):
            pinned_value = by_contract[contract].get(field)
            live_value = live[contract].get(field)
            if pinned_value != live_value:
                problems.append(
                    f"{PIN}: {contract}.{field} is pinned as {pinned_value!r}, "
                    f"the registry declares {live_value!r}"
                )
    return problems


def population_summary(pinned: dict, observed: dict) -> str:
    """Say which population each number counts, in the line that gets quoted.

    This sentence used to read `{len(observed['profiles'])} registry profiles
    compared field by field against {PIN}` over *every* profile the registry
    declares. `differences()` field-compares only the intersection of the pin
    and the registry: a contract the registry has and the pin does not is
    reported and then skipped, never compared field by field. So the claim
    overstated the checking precisely on the failing path -- the one whose
    number gets quoted into a card.

    The first two counts partition the registry, so they sum to its total and a
    reader can check that they do. The third is a different population -- pin
    entries with no live contract -- and is stated separately for that reason
    rather than folded in. It is empty on a passing run, which is exactly when
    one number under two names is indistinguishable from the truth.

    Both sides are keyed by contract, the same way `differences()` keys them, so
    the counts describe what was actually compared rather than what was read.
    """
    by_contract = {entry["contract"] for entry in pinned.get("profiles", [])}
    live = {entry["contract"] for entry in observed["profiles"]}
    compared = len(live & by_contract)
    unpinned = len(live - by_contract)
    orphaned = len(by_contract - live)
    return (
        f"{len(live)} registry "
        f"{'profile' if len(live) == 1 else 'profiles'}; "
        f"{compared} compared field by field against {PIN}, "
        f"{unpinned} absent from the pin and reported without being compared; "
        f"the pin holds {orphaned} "
        f"{'contract' if orphaned == 1 else 'contracts'} the registry no "
        "longer declares"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trainvm", help="path to a built trainvm binary")
    parser.add_argument(
        "--repository",
        default=str(pathlib.Path(__file__).resolve().parent.parent),
        help="repository root holding the pin",
    )
    parser.add_argument("--check", action="store_true", help="fail on drift")
    parser.add_argument("--write", action="store_true", help="regenerate the pin")
    arguments = parser.parse_args()
    repository = pathlib.Path(arguments.repository).resolve()
    path = repository / PIN

    observed = build(arguments.trainvm)
    rendered = json.dumps(observed, indent=2, sort_keys=False) + "\n"

    if arguments.write:
        path.write_text(rendered, encoding="utf-8")
        print(f"WROTE: {len(observed['profiles'])} profiles into {PIN}")
        return 0

    if not arguments.check:
        print(rendered, end="")
        return 0

    if not path.exists():
        print(f"FAIL: {PIN} does not exist; regenerate it with --write")
        print(verdict_line("step-zero arming pin", ["missing pin"], "0 profiles"))
        return 1

    pinned = json.loads(path.read_text(encoding="utf-8"))
    problems = differences(pinned, observed)
    for problem in problems:
        print(f"FAIL: {problem}")
    print(
        verdict_line(
            "step-zero arming pin",
            problems,
            population_summary(pinned, observed),
        )
    )
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
