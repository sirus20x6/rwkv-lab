#!/usr/bin/env python3
"""Validate the representative benchmark fixture matrix.

The qualification contract selects parity evidence by operation effect class:
a training kernel must prove gradient, optimizer-update, state and
resumed-trajectory parity, while a serving kernel must not invent a backward
requirement, and a deterministic producer is graded on content/ordering/manifest
parity instead. Written as prose that rule is easy to drift from, so this
enforces it against the checked-in matrix.

It also fails when a family the roadmap names has no fixture, when a fixture
omits any required measurement axis, and when the document acquires anything
that would carry executable authority. A benchmark document declares what must
be measured; it never carries argv, environment, an executable identity, or a
measured result.

Usage:
    python scripts/validate_benchmark_matrix.py [path]
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from scripts.gate_verdict import verdict_line  # noqa: E402

DEFAULT = pathlib.Path("docs/experiment-vm/benchmark-matrix.v1.json")

# The roadmap requires fixtures "for MageFlow/flow, RWKV LM, transformer LM,
# vision/RWKV, fine-tuning, distillation, and post-training rather than one
# synthetic GEMM". Conversion and serving are included because both are already
# distinct effect classes in the qualification contract.
REQUIRED_FAMILIES = {
    "mageflow_diffusion", "rwkv_scratch", "rwkv_pretrained", "transformer_lm",
    "vision_multimodal", "fine_tuning", "distillation", "conversion",
    "post_training", "serving",
}
PORTABILITY = {"portable", "machine_native"}

# The nine parity booleans actually implemented on CacheQualificationEvidence
# in trainvm/include/trainvm/cache_artifact_authority.hpp, plus the throughput
# evidence the same receipt carries. A fixture may only require evidence the
# qualification gate can actually produce, so the matrix cannot invent a parity
# dimension that nothing computes.
IMPLEMENTED_EVIDENCE = {
    "output_parity", "gradient_parity", "optimizer_update_parity",
    "state_parity", "resumed_trajectory_parity", "determinism_parity",
    "content_parity", "ordering_parity", "manifest_parity",
    "throughput_evidence", "ownership_parity", "trajectory_parity",
    "resume_parity",
}

# A benchmark fixture is a declaration, never a launcher.
FORBIDDEN_KEYS = {
    "argv", "arguments", "command", "env", "environment", "executable",
    "executable_path", "interpreter", "module", "working_directory",
    # A declaration must not carry an outcome either; results belong in a
    # receipt produced by a run, not in the checked-in matrix.
    "measured_throughput", "measured_step_time", "result", "results",
}


def main() -> int:
    path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    document = json.loads(path.read_text())
    failures: list[str] = []

    if document.get("api_version") != "trainvm.benchmark-matrix/v1":
        failures.append("api_version must be trainvm.benchmark-matrix/v1")
    if document.get("authority") != "benchmark_fixture_evidence_only":
        failures.append(
            "authority must be benchmark_fixture_evidence_only; this document "
            "cannot grant execution authority")

    evidence = {
        name: value
        for name, value in document.get("effect_class_evidence", {}).items()
        if name != "comment"
    }
    for effect_class, required in evidence.items():
        unknown = sorted(set(required) - IMPLEMENTED_EVIDENCE)
        if unknown:
            failures.append(
                f"effect class '{effect_class}' requires evidence the "
                f"qualification receipt does not implement: {unknown}")
    axes = document.get("required_measurements", {}).get("axes", [])
    if not axes:
        failures.append("required_measurements.axes must be declared")

    fixtures = document.get("fixtures", [])
    if not fixtures:
        failures.append("no fixtures declared")

    seen_ids: set[str] = set()
    families: set[str] = set()
    transition_families: set[str] = set()

    for fixture in fixtures:
        identifier = fixture.get("id", "<missing id>")
        if identifier in seen_ids:
            failures.append(f"{identifier}: duplicate fixture id")
        seen_ids.add(identifier)

        for key in fixture:
            if key in FORBIDDEN_KEYS:
                failures.append(f"{identifier}: reserved key '{key}'")

        family = fixture.get("family")
        families.add(family)

        effect_class = fixture.get("effect_class")
        if effect_class not in evidence:
            failures.append(
                f"{identifier}: unknown effect_class {effect_class!r}")
        else:
            # The fixture restates its class's evidence, and it must match
            # exactly. This is the gate that stops a serving fixture acquiring
            # a gradient-parity claim it cannot have, or a training fixture
            # quietly dropping resumed-trajectory parity to look qualified.
            expected = list(evidence[effect_class])
            declared = fixture.get("parity_evidence")
            if declared != expected:
                extra = sorted(set(declared or []) - set(expected))
                missing_evidence = sorted(set(expected) - set(declared or []))
                detail = []
                if extra:
                    detail.append(f"claims {extra} which {effect_class} cannot")
                if missing_evidence:
                    detail.append(f"omits required {missing_evidence}")
                if not detail:
                    detail.append("declares them in a different order")
                failures.append(
                    f"{identifier}: parity_evidence " + "; ".join(detail))

        if fixture.get("portability") not in PORTABILITY:
            failures.append(
                f"{identifier}: portability must be one of {sorted(PORTABILITY)}")

        # A portable fixture that needs an accelerator is not portable. This
        # catches the quiet case where a "portable baseline" silently requires
        # the one machine that has the device.
        if (fixture.get("portability") == "portable"
                and fixture.get("accelerator_required")):
            failures.append(
                f"{identifier}: a portable fixture cannot require an accelerator")

        buckets = fixture.get("shape_buckets") or []
        if not buckets:
            failures.append(f"{identifier}: shape_buckets must be non-empty")
        if len(set(buckets)) != len(buckets):
            failures.append(f"{identifier}: duplicate shape buckets")

        if not fixture.get("quality_gate"):
            failures.append(
                f"{identifier}: a quality gate is required before any speed claim")

        if not str(fixture.get("notes", "")).strip():
            failures.append(f"{identifier}: notes must explain the fixture")

        if fixture.get("declares_curriculum_transition"):
            transition_families.add(family)

    missing = REQUIRED_FAMILIES - families
    for family in sorted(missing):
        failures.append(f"no benchmark fixture covers family '{family}'")
    unexpected = families - REQUIRED_FAMILIES - {None}
    for family in sorted(unexpected):
        failures.append(f"fixture declares unreviewed family '{family}'")

    # The contract requires "representative shape-bucket coverage, including a
    # transition between curriculum stages". At least one fixture must actually
    # exercise a transition, or that clause is decorative.
    if not transition_families:
        failures.append(
            "no fixture declares a curriculum-stage transition")

    for failure in failures:
        print(f"FAIL: {failure}")
    print(verdict_line(
        "benchmark matrix",
        failures,
        f"{len(fixtures)} fixtures over {len(families)} families, "
        f"{len(transition_families)} with curriculum transitions",
    ))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
