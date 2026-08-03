from __future__ import annotations

import pytest

from rwkv_lab.trainvm_adapters.posttraining import RWKVPostTrainConfig


def config(**updates: object) -> RWKVPostTrainConfig:
    values = {
        "checkpoint": "/models/base.pt",
        "data": "/datasets/sft.jsonl",
        "output_dir": "/runs/posttrain",
    }
    values.update(updates)
    return RWKVPostTrainConfig(**values)


def test_posttraining_config_closes_objectives_paths_and_budget() -> None:
    value = config(objective="dpo", targets=["key", "value"], steps=20)
    assert value.targets == ("key", "value")
    assert value.weight_decay == pytest.approx(0.01)
    with pytest.raises(ValueError, match="objective"):
        config(objective="arbitrary_import")
    with pytest.raises(ValueError, match="unique"):
        config(targets=["key", "key"])
    with pytest.raises(ValueError, match="warmup_steps"):
        config(steps=10, warmup_steps=11)
    with pytest.raises(ValueError, match="device"):
        config(device="remote:worker")


def test_posttraining_config_rejects_unknown_fields() -> None:
    with pytest.raises(TypeError):
        RWKVPostTrainConfig(
            checkpoint="/models/base.pt",
            data="/datasets/sft.jsonl",
            output_dir="/runs/posttrain",
            python_import="attacker.module",  # type: ignore[call-arg]
        )
