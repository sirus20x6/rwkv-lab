"""Cross the native/Python boundary with a resolved training composition.

The native side attaches optional blocks to the resolved-composition envelope
(topologies since PR #34, post_training since PR #61) and the Python worker
validated that envelope with an exact set comparison, so either block made the
whole composition unloadable. The topologies chain was complete natively and
dead at this boundary, with CI green throughout, because every test on each
side used a fixture built by that same side and neither crossed over carrying
an optional block.

The last test here is the one that would have caught it: it reads the native
source of truth rather than trusting a fixture to agree with it.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from rwkv_lab.trainvm_worker import load_resolved_training_composition
from rwkv_lab.trainvm_worker.training import (
    _COMPOSITION_OPTIONAL_FIELDS,
    TrainingCompositionError,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_SOURCE = (
    REPOSITORY_ROOT / "trainvm" / "src" / "training_component_registry.cpp"
)
COMPONENT_REGISTRY = (
    REPOSITORY_ROOT / "docs" / "experiment-vm" / "examples" / "training-components.v1.json"
)

TOPOLOGY_BLOCK = {
    "api_version": "trainvm.rwkv-scratch-training/v1",
    "flags": {"--engram": "1"},
}
POST_TRAINING_BLOCK = {
    "api_version": "trainvm.post-training-arm/v1",
    "arm_id": "arm.finetune-a",
    "kind": "supervised_finetune",
    "bounds": [{"kind": "optimizer_steps", "magnitude": 10000}],
    "reproducibility_claim": "exact",
    "claims_trajectory_preserving_resume": False,
    "seed": 7,
}


def _canonical(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def _digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def envelope(**blocks: object) -> dict:
    """An envelope shaped like composition_body(), with a real descriptor.

    The descriptor comes from the checked-in component registry rather than
    being written out here, so this fixture cannot quietly disagree with what
    the registry actually declares.
    """
    registry = json.loads(COMPONENT_REGISTRY.read_text(encoding="utf-8"))
    descriptor = next(
        item
        for item in registry["components"]
        if item["key"]["category"] == "optimizer"
        and item["key"]["name"] == "torch_adamw_no_decay"
    )
    body = {
        "api_version": "trainvm.resolved-training-composition/v1",
        "components": {
            "optimizer": {
                "configuration": {
                    "learning_rate": 1e-3,
                    "beta1": 0.9,
                    "beta2": 0.999,
                    "epsilon": 1e-8,
                    "foreach": True,
                    "fused": False,
                },
                "descriptor": descriptor,
                "descriptor_digest": _digest(_canonical(descriptor)),
            }
        },
        "model_family": "rwkv",
        "registry_digest": _digest(_canonical(registry)),
    }
    body.update(blocks)
    return {**body, "composition_digest": _digest(_canonical(body))}


def test_a_composition_without_optional_blocks_still_loads() -> None:
    loaded = load_resolved_training_composition(envelope())
    assert loaded.topologies is None
    assert loaded.post_training is None


@pytest.mark.parametrize(
    "block, value",
    [("topologies", TOPOLOGY_BLOCK), ("post_training", POST_TRAINING_BLOCK)],
)
def test_optional_blocks_load_and_are_preserved(block: str, value: dict) -> None:
    loaded = load_resolved_training_composition(envelope(**{block: value}))
    carried = getattr(loaded, block)
    assert carried is not None, f"{block} was dropped rather than carried"
    assert carried["api_version"] == value["api_version"]


def test_both_blocks_together_load() -> None:
    loaded = load_resolved_training_composition(
        envelope(topologies=TOPOLOGY_BLOCK, post_training=POST_TRAINING_BLOCK)
    )
    assert loaded.topologies is not None
    assert loaded.post_training is not None


def test_an_unknown_block_is_still_refused() -> None:
    """The envelope check stays exact about what it does not know.

    Relaxing it to "ignore anything unrecognised" would fix the reported bug
    and reintroduce it silently the next time the native side adds a block: an
    unrecognised key inside a digest-bound envelope is precisely the drift this
    check exists to catch.
    """
    with pytest.raises(TrainingCompositionError):
        load_resolved_training_composition(envelope(qualification={"passed": True}))


def test_python_knows_every_optional_block_the_native_side_emits() -> None:
    """Fail when the native side grows a block Python has not learned about.

    Reads composition_body() in the registry source and extracts the keys it
    conditionally attaches. This is the check the original defect needed: it
    fails when the two sides disagree, rather than when someone runs a worker.
    """
    source = REGISTRY_SOURCE.read_text(encoding="utf-8")
    start = source.index("Json composition_body(")
    end = source.index("\n}", start)
    emitted = set(re.findall(r'body\["([a-z_]+)"\]\s*=', source[start:end]))
    assert emitted, "could not read composition_body; this test needs updating"
    unknown = emitted - _COMPOSITION_OPTIONAL_FIELDS
    assert not unknown, (
        f"the native composition emits {sorted(unknown)}, which the Python "
        "worker will refuse. Add them to _COMPOSITION_OPTIONAL_FIELDS and "
        "carry them onto ResolvedTrainingComposition."
    )
