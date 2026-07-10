import tempfile
import json

import numpy as np
import torch

from rwkv_lab import registry
from rwkv_lab.experiment import _block_source, _seeded_batch, factorial_configs
from rwkv_lab.experiment_analysis import (alpha_spending, holm_adjust, paired_stats,
                                          pareto_front, sequential_holm)
from rwkv_lab.synthetic_tasks import make_task


def test_seeded_batch_is_independent_of_model_rng():
    task = make_task("recall:4")
    a = _seeded_batch(task, 3, "cpu", 123)
    torch.randn(1000)
    b = _seeded_batch(task, 3, "cpu", 123)
    assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_resumed_data_tape_starts_at_exact_next_batch():
    task = make_task("recall:4")
    full = _block_source(task, 2, "cpu", 3, data_seed=91)
    expected = [next(full) for _ in range(5)][-1]
    resumed = next(_block_source(task, 2, "cpu", 3, data_seed=91, start_step=4))
    assert all(torch.equal(x, y) for x, y in zip(expected, resumed))


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


def test_alpha_spending_is_monotone_and_exhausts_family_alpha():
    looks = [alpha_spending(i, 4) for i in range(1, 5)]
    assert all(a["increment"] > 0 for a in looks)
    assert all(a["cumulative"] < b["cumulative"] for a, b in zip(looks, looks[1:]))
    assert abs(sum(a["increment"] for a in looks) - 0.05) < 1e-12
    assert looks[0]["increment"] < looks[-1]["increment"]  # OBF is conservative early


def test_sequential_holm_records_the_look_boundary():
    base = np.arange(12, dtype=float)
    raw = {"better": paired_stats(base, base + 1, bootstrap=200, alpha=0.01)}
    out = sequential_holm(raw, 2, 4)
    assert out["better"]["sequential"]["look"] == 2
    assert out["better"]["p_adjusted"] <= 1


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


def test_trainer_logs_normalize_for_lm_and_conversion(tmp_path):
    from rwkv_lab.config import _read_train_log
    lm = tmp_path / "lm.jsonl"
    lm.write_text("\n".join(json.dumps(r) for r in [
        {"kind": "train", "step": 2, "loss": 3.0, "tok_per_sec": 99},
        {"kind": "eval", "step": 2, "loss": 2.0, "val_loss": 2.0, "ppl": 7.39},
    ]))
    metrics, series, profile = _read_train_log(str(lm), "lm")
    assert metrics["acc"] == -2.0 and metrics["val_loss"] == 2.0
    assert series[0]["step"] == 2 and profile["tok_per_sec"] == 99

    conv = tmp_path / "conversion.jsonl"
    conv.write_text(json.dumps({"kind": "eval", "step": 9, "loss": 1.2,
                                "ppl": 3.3, "top1_acc": 0.42}) + "\n")
    metrics, _, _ = _read_train_log(str(conv), "conversion")
    assert metrics["acc"] == metrics["top1_acc"] == 0.42


def test_lm_subprocesses_land_in_one_normalized_campaign(tmp_path, monkeypatch):
    from pathlib import Path
    from rwkv_lab import config
    monkeypatch.chdir(tmp_path)
    db = str(tmp_path / "lm.db")
    real_run = config.subprocess.run

    def fake_run(cmd, **_):
        if "--out" not in cmd:
            return real_run(cmd, **_)
        out = Path(cmd[cmd.index("--out") + 1]); out.mkdir(parents=True, exist_ok=True)
        seed = int(cmd[cmd.index("--seed") + 1]); arm = out.parts[-2]
        loss = 2.0 + seed * .01 - (.1 if arm == "candidate" else 0)
        (out / "train.jsonl").write_text("\n".join(json.dumps(r) for r in [
            {"kind": "train", "step": 2, "loss": loss + .2, "tok_per_sec": 100},
            {"kind": "eval", "step": 2, "loss": loss, "val_loss": loss,
             "ppl": float(np.exp(loss))},
        ]) + "\n")
        Path(cmd[cmd.index("--save") + 1]).write_bytes(b"checkpoint")

    monkeypatch.setattr(config.subprocess, "run", fake_run)
    cid = config._run_lm_campaign({"baseline": {}, "candidate": {"n_loops": 2}},
                                  {"d_model": 16, "n_layers": 1, "head_size": 8},
                                  {"steps": 2}, ["--data", "fake.bin"], None,
                                  task="lm:test", seeds=2, db=db)
    con = registry._con(db)
    assert con.execute("select count(*) from trials where campaign_id=?", (cid,)).fetchone()[0] == 4
    assert con.execute("select count(*) from comparisons where campaign_id=?", (cid,)).fetchone()[0] == 1
    assert con.execute("select count(*) from artifacts where campaign_id=?", (cid,)).fetchone()[0] == 8
    con.close()


def test_conversion_layers_and_seeds_are_paired_campaign_units(tmp_path, monkeypatch):
    from pathlib import Path
    from rwkv_lab import config
    monkeypatch.chdir(tmp_path)
    db = str(tmp_path / "conversion.db")
    real_run = config.subprocess.run

    def fake_run(cmd, **_):
        if "--out" not in cmd:
            return real_run(cmd, **_)
        out = Path(cmd[cmd.index("--out") + 1]); out.mkdir(parents=True, exist_ok=True)
        arm = out.parts[-2]; steps = int(cmd[cmd.index("--steps") + 1])
        acc = .6 + (.1 if arm == "candidate" else 0)
        (out / "train.jsonl").write_text(json.dumps(
            {"kind": "eval", "step": steps - 1, "loss": 1.0 - acc,
             "ppl": 2.0, "top1_acc": acc}) + "\n")
        final = out / f"step_{steps - 1:06d}" / "ckpt.pt"
        final.parent.mkdir(); final.write_bytes(b"final")
        best = out / "best" / "ckpt.pt"; best.parent.mkdir(); best.write_bytes(b"best")

    monkeypatch.setattr(config.subprocess, "run", fake_run)
    cid = config._run_conversion({
        "name": "test conversion", "registry_db": db, "seeds": 1,
        "conversion": {"model_dir": "model", "data": "tokens", "layers": [0, 2]},
        "train": {"steps": 2},
        "configs": {"baseline": {"loop_count": 1}, "candidate": {"loop_count": 2}},
    })
    con = registry._con(db)
    units = [r[0] for r in con.execute("select distinct seed from trials where campaign_id=? order by seed",
                                      (cid,))]
    assert units == [0, 2_000_000]
    assert con.execute("select count(*) from trials where campaign_id=?", (cid,)).fetchone()[0] == 4
    assert con.execute("select count(*) from comparisons where campaign_id=?", (cid,)).fetchone()[0] == 1
    assert con.execute("select count(*) from artifacts where campaign_id=?", (cid,)).fetchone()[0] == 12
    con.close()
