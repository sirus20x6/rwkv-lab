#!/usr/bin/env python3
"""Pin the rwkv-lab adapter profile set reported by the native registry.

Split out of verify_rwkv_lab_worker_artifact.py so this can run in hosted CI.
That test builds and replays a sealed worker artifact, which is why it is
excluded from the hosted native job; this check needs the trainvm binary and
nothing else, so excluding it cost real coverage. An adapter was added and the
pin was left stale for hours because the only thing that checked it never ran
on a pull request.

The pin is by identity and order, not by count. A bare length said only that
something had changed, and said it as "not canonical", which is why locating
the last addition cost a bisect. It also exists to force a deliberate
confirmation that a newly registered profile belongs here with the right root
distributions, so updating it should be an act of review rather than of
silencing a red test.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

API_VERSION = "trainvm.rwkv-lab-worker-runtime-requirements/v1"

# Registry order, because the deployment materializer groups adapters that
# share one sealed Python closure and the MageFlow routes must stay adjacent.
EXPECTED_ADAPTERS = (
    "rwkv-lab.mageflow-appearance-expert",
    "rwkv-lab.mageflow-full-backbone",
    "rwkv-lab.mageflow-terminal-expert",
    "rwkv-lab.rwkv-posttraining",
    "rwkv-lab.scalar-metric-decision",
    "rwkv-lab.qwen-ao3",
    "rwkv-lab.hf-multimodal-sft",
    "rwkv-lab.transformer-mla",
    "rwkv-lab.transformer-mla-mtp",
    "rwkv-lab.transformer-mla-mutor",
    "rwkv-lab.transformer-mla-fsp",
    "rwkv-lab.transformer-mla-parallel",
    "rwkv-lab.transformer-mla-rwkv8",
    "rwkv-lab.transformer-mla-engram",
    "rwkv-lab.transformer-mla-full-backbone",
    "rwkv-lab.vision-teacher-compressor",
    "rwkv-lab.vision-frozen-adapter",
    "rwkv-lab.vision-native-head",
    "rwkv-lab.vision-rwkv-student",
    "rwkv-lab.rwkv-rlvr",
    "rwkv-lab.rwkv-scratch",
)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_rwkv_lab_runtime_requirements.py TRAINVM")
    trainvm = Path(sys.argv[1]).resolve(strict=True)
    requirements = json.loads(
        subprocess.check_output(
            [str(trainvm), "inspect-rwkv-lab-runtime-requirements"], text=True
        )
    )
    if requirements["api_version"] != API_VERSION:
        raise SystemExit("native runtime requirements schema drifted")

    adapters = tuple(profile["adapter"] for profile in requirements["profiles"])
    if adapters != EXPECTED_ADAPTERS:
        added = sorted(set(adapters) - set(EXPECTED_ADAPTERS))
        removed = sorted(set(EXPECTED_ADAPTERS) - set(adapters))
        if added or removed:
            raise SystemExit(
                "native runtime requirement profiles drifted: "
                f"added={added} removed={removed}. Confirm each new profile's "
                "root_distributions before pinning it here."
            )
        raise SystemExit(
            "native runtime requirement profile order changed: "
            f"{list(adapters)}. The deployment materializer groups adapters "
            "that share a sealed closure, so order is part of the contract."
        )

    if len(set(adapters)) != len(adapters):
        raise SystemExit("native runtime requirements repeat an adapter id")

    shared_distributions = tuple(requirements["shared_root_distributions"])
    if shared_distributions != tuple(sorted(set(shared_distributions))):
        raise SystemExit(
            "shared root distributions must be sorted and free of duplicates"
        )
    unshared = [
        profile["adapter"]
        for profile in requirements["profiles"]
        if not set(shared_distributions).issubset(profile["root_distributions"])
    ]
    if unshared:
        raise SystemExit(
            "shared root distributions are not a subset of every profile: "
            f"{unshared}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
