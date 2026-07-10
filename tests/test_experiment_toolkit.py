import tempfile

import numpy as np
import torch

from rwkv_lab import registry
from rwkv_lab.experiment import _seeded_batch, factorial_configs
from rwkv_lab.experiment_analysis import holm_adjust, paired_stats, pareto_front
from rwkv_lab.synthetic_tasks import make_task


def test_seeded_batch_is_independent_of_model_rng():
    task = make_task("recall:4")
    a = _seeded_batch(task, 3, "cpu", 123)
    torch.randn(1000)
    b = _seeded_batch(task, 3, "cpu", 123)
    assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_paired_statistics_and_holm():
    base = np.arange(8, dtype=float)
    better = base + 1
    st = paired_stats(base, better, bootstrap=500, seed=4)
    assert st["delta"] == 1 and st["ci_low"] == 1 and st["ci_high"] == 1
    adjusted = holm_adjust({"a": st, "b": paired_stats(base, base, bootstrap=100, seed=5)})
    assert adjusted["a"]["p_adjusted"] <= 0.05
    assert adjusted["a"]["significant"]


def test_factorial_and_pareto_helpers():
    generated = factorial_configs(["loop3", "engram"], 2)
    assert generated["loop3+engram"]["n_loops"] == 3
    flags = pareto_front([
        {"acc": .8, "train_seconds": 10, "peak_alloc_mb": 10},
        {"acc": .7, "train_seconds": 12, "peak_alloc_mb": 12},
    ])
    assert flags == [True, False]


def test_trial_registry_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        db = td + "/experiments.db"
        cid = registry.create_campaign("copy:4", {"steps": 2}, capsule={"test": True}, db=db)
        aid = registry.ensure_arm(cid, "baseline", {}, db=db)
        registry.record_trial(cid, aid, 0, 0, 2, {"acc": .5},
                              series=[{"step": 1, "loss": 2}], profile={"tok_per_sec": 10}, db=db)
        rows = registry.campaign_rows(cid, db=db)
        assert rows[0]["arm"] == "baseline" and rows[0]["metrics"]["acc"] == .5
        registry.finish_campaign(cid, db=db)
