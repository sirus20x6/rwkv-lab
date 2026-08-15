from __future__ import annotations

import json

import pytest

from rwkv_lab.prototype_suite import run_prototype_suite


@pytest.mark.parametrize("architecture", ["rwkv8", "kan_rwkv", "blt_rwkv7", "rosa_blt"])
def test_declarative_model_probe(tmp_path, architecture):
    cfg = {
        "name": "probe",
        "prototype": {
            "output_dir": str(tmp_path),
            "tasks": [
                {
                    "name": architecture.replace("_", "-"),
                    "kind": "model_probe",
                    "architecture": architecture,
                    "model": {
                        "vocab_size": 32,
                        "d_model": 16,
                        "n_layers": 2,
                        "head_size": 16,
                    },
                    "batch": 1,
                    "sequence_length": 4,
                }
            ],
        },
    }
    summary_path = run_prototype_suite(cfg)
    summary = json.loads(summary_path.read_text())
    result = next(iter(summary["tasks"].values()))["result"]
    assert result["architecture"] == architecture
    assert result["parameters"] > 0
    assert result["logits_shape"] == [1, 4, 32]
    assert result["latency_ms"] > 0


def test_prototype_suite_rejects_unknown_task(tmp_path):
    cfg = {
        "prototype": {
            "output_dir": str(tmp_path),
            "tasks": [{"name": "bad", "kind": "unknown"}],
        }
    }
    with pytest.raises(ValueError, match="unknown prototype task"):
        run_prototype_suite(cfg)
