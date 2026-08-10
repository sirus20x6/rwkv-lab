#!/usr/bin/env python3
"""Fail when a checked-in experiment document names an adapter that does not exist.

`experiments/` holds the documents someone would actually launch. Nothing
validated it. `scripts/validate_experiment_documents.py` resolves component and
operation invocations against the pinned registry and scans
`docs/experiment-vm/examples/` -- so the directory whose name says it holds
experiments was the one directory of experiment documents nothing looked at.

Measured on `origin/main` at 35e62bb6: of the 21 JSON documents there, ten name
an adapter and **nine of those name an adapter that exists nowhere on main**.

    rwkv-lab.ztok-superposition      6 documents
    rwkv-lab.qwen-caption-finetune   3 documents

Neither key appears in `trainvm/src/rwkv_lab_worker_contract.cpp`, and
`git log origin/main -- src/rwkv_lab/qwen_caption_finetune.py` returns nothing:
that path has never existed on main. Both implementations are real but unmerged
-- `git rev-list --all` finds them on `integration/parity-candidate` -- so this
is "not on trunk", not "lost".

Those nine are archival, and that is a decision rather than a guess. PR #139
deliberately *preserved* the ztok six onto main;
`docs/experiment-vm/QWEN_CAPTION_DECLARATIVE_MIGRATION.md` says the old bespoke
implementation and its artifacts "remain historical evidence"; and
`docs/experiment-vm/PRESERVED_WORKTREE_ARTIFACTS.md` already records the Qwen
bundle with per-file content hashes and the branch they also live on. What was
missing was anything *mechanical* connecting that prose to the documents, so a
reader opening `experiments/` could not tell a launchable document from a
preserved one, and a tenth unrunnable document could be added tomorrow with
nothing to say so.

Why a countdown and not a scope exclusion
-----------------------------------------
`docs/experiment-vm/unrunnable-experiment-documents.v1.json` starts at exactly
those nine. It can only shrink: a listed document whose adapter becomes
resolvable **fails**, so porting an implementation forces the entry's deletion
rather than leaving a stale exemption behind. A listed document that no longer
exists fails too.

Nine is small enough to be an honest countdown. The same instrument over the
`__all__` surface was measured at 79 entries and deliberately not shipped
(card-d198cc09), because an allowlist that large is an instrument tuned until
the number looks comfortable rather than a debt anybody intends to pay.

What this does NOT check
------------------------
Only that a named adapter is one the registry declares. It says nothing about
whether the rest of a document is valid, whether its inputs exist, or whether
it would run -- `validate_experiment_documents.py` does the deep work, and
widening *that* to this directory would go red on nine documents immediately.
A document naming a real adapter can still be wrong in every other way.

Usage:
    python scripts/ci_experiment_adapter_gate.py [--repository .]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from scripts.gate_verdict import verdict_line  # noqa: E402

SCOPE = "experiments"
REGISTRY_PIN = "docs/experiment-vm/step-zero-arming.v1.json"
UNRUNNABLE = "docs/experiment-vm/unrunnable-experiment-documents.v1.json"

# Adapters the authority implements itself rather than dispatching to a worker.
# `trainvm.core` is not in the rwkv-lab registry pin and never will be, so it is
# named here rather than treated as an absence.
BUILTIN_ADAPTERS = frozenset({"trainvm.core"})


def declared_adapters(repository: pathlib.Path) -> set[str]:
    """Adapter keys the pinned registry declares.

    Read from the pin rather than from `rwkv_lab_worker_contract.cpp`: the pin
    is regenerated from the native registry and is the artifact the other gates
    already resolve against, so this asks the same authority the same way.
    """
    pin = json.loads((repository / REGISTRY_PIN).read_text(encoding="utf-8"))
    return {profile["adapter"] for profile in pin.get("profiles", [])
            if profile.get("adapter")}


def adapters_named_by(document: object) -> set[str]:
    """Every `adapter` string anywhere in a document.

    Walked rather than read from a fixed path, because these documents nest
    adapters at several depths and a path-specific reader would silently see
    none in a shape it did not expect -- which is indistinguishable from a
    document that names none.
    """
    found: set[str] = set()

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "adapter" and isinstance(value, str):
                    found.add(value)
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(document)
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository",
        default=str(pathlib.Path(__file__).resolve().parent.parent),
        help="repository root to analyse")
    arguments = parser.parse_args()
    repository = pathlib.Path(arguments.repository).resolve()

    problems: list[str] = []
    declared = declared_adapters(repository) | BUILTIN_ADAPTERS

    listed: dict[str, str] = {}
    unrunnable_path = repository / UNRUNNABLE
    if unrunnable_path.exists():
        document = json.loads(unrunnable_path.read_text(encoding="utf-8"))
        listed = {entry["path"]: entry.get("why", "")
                  for entry in document.get("documents", [])}

    scope = repository / SCOPE
    scanned = 0
    still_unrunnable = 0
    for path in sorted(scope.rglob("*.json")) if scope.is_dir() else []:
        relative = str(path.relative_to(repository))
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            problems.append(f"{relative}: is not valid JSON ({error})")
            continue
        scanned += 1

        missing = sorted(adapters_named_by(document) - declared)
        if missing and relative not in listed:
            problems.append(
                f"{relative}: names {', '.join(missing)}, which the registry "
                f"does not declare. Either the adapter needs declaring in "
                f"trainvm/src/rwkv_lab_worker_contract.cpp, or the document is "
                f"preserved rather than launchable and belongs in {UNRUNNABLE} "
                f"with a reason and where its implementation lives.")
        elif missing:
            still_unrunnable += 1
        elif relative in listed:
            problems.append(
                f"{relative}: is listed in {UNRUNNABLE} but every adapter it "
                f"names now resolves. Delete the entry -- this list is a "
                f"countdown, and an exemption that no longer applies is one "
                f"nobody will delete later.")

    for relative in sorted(listed):
        if not (repository / relative).exists():
            problems.append(
                f"{UNRUNNABLE}: lists {relative}, which does not exist. Remove "
                f"the entry; an exemption for a deleted file implies coverage "
                f"of something that is not there.")

    if scanned == 0:
        # Not a pass. An empty scope means nothing was checked, and a renamed
        # directory would otherwise retire this gate in silence.
        problems.append(
            f"no JSON documents found under {SCOPE}/, so nothing was checked")

    for problem in problems:
        print(f"FAIL: {problem}")

    print(verdict_line(
        "experiment adapter gate",
        problems,
        f"{scanned} documents under {SCOPE}/ checked against "
        f"{len(declared)} declared adapters; {still_unrunnable} recorded as "
        f"preserved rather than launchable",
    ))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
