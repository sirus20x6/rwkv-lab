"""CPU tests for the Tier-1 synthetic diagnostic tasks + experiment aggregation."""
import numpy as np
import torch

from rwkv_lab.synthetic_tasks import make_task, Task, PAD, SEP


def _shapes_and_mask(spec):
    t = make_task(spec)
    rng = np.random.default_rng(0)
    x, y, m = t.batch(4, "cpu", rng)
    assert x.shape == y.shape == m.shape
    assert x.dtype == torch.long and m.dtype == torch.float32
    assert x.max().item() < t.vocab and x.min().item() >= 0
    assert m.sum().item() > 0                         # some answer tokens are scored
    assert set(m.unique().tolist()) <= {0.0, 1.0}     # mask is binary
    return t, x, y, m


def test_copy_shapes():
    t, x, y, m = _shapes_and_mask("copy:8")
    # copy scores the whole reproduced sequence (length L) per example
    assert m.sum().item() == 4 * 8


def test_recall_answer_is_determined_by_context():
    # the queried key's value must appear in the pairs => the task is learnable (not random)
    t = make_task("recall:5")
    rng = np.random.default_rng(1)
    x, y, m = t.batch(8, "cpu", rng)
    for b in range(8):
        row = x[b].tolist()                           # [k1 v1 … kn vn SEP kq]
        kq = row[-1]
        keys = row[0:2 * t.n:2]; vals = row[1:2 * t.n:2]
        answer = y[b][m[b] > 0].item()
        assert kq in keys, "query key must be one of the stored keys"
        assert vals[keys.index(kq)] == answer, "answer must be the queried key's value"
    assert m.sum().item() == 8                          # exactly one answer token per example


def test_induction_scores_final_continuation():
    t, x, y, m = _shapes_and_mask("induction:16")
    assert m.sum().item() == 4                          # one continuation per example
    assert (m[:, -1] == 1).all()                        # scored at the last position


def test_accuracy_metric():
    # perfect logits -> accuracy 1.0; scrambled -> < 1.0
    t = make_task("copy:6")
    rng = np.random.default_rng(2)
    x, y, m = t.batch(4, "cpu", rng)
    V = t.vocab
    onehot = torch.zeros(*y.shape, V); onehot.scatter_(-1, y.unsqueeze(-1), 10.0)
    assert Task.accuracy(onehot, y, m) == 1.0
    wrong = torch.zeros(*y.shape, V); wrong[..., 0] = 10.0   # always predict PAD
    assert Task.accuracy(wrong, y, m) < 0.5


def test_length_generalization_spec():
    # a task's name must round-trip to a 2x-longer spec (used by experiment.train_eval)
    t = make_task("copy:16")
    long = make_task(f"copy:{2 * t.L}")
    assert long.L == 32


def test_experiment_significance_logic():
    from rwkv_lab.experiment import _agg
    m, s = _agg([0.9, 0.92, 0.88])
    assert 0.88 <= m <= 0.92 and s > 0
    # |Δmean| > pooled std => significant; a tiny gap within noise => not
    base_m, base_s = _agg([0.90, 0.91])
    hi_m, hi_s = _agg([0.50, 0.44])                     # clearly worse, low overlap
    assert abs(hi_m - base_m) > (hi_s + base_s)          # SIGNIFICANT
    near_m, near_s = _agg([0.89, 0.92])
    assert abs(near_m - base_m) <= (near_s + base_s)     # within noise


def test_registry_roundtrip(tmp_path):
    from rwkv_lab import registry
    db = str(tmp_path / "t.db")
    registry.record("copy:8", "baseline", 3, 500, {"acc": [0.90, 0.02]}, db=db)
    registry.record("copy:8", "loop3", 3, 500, {"acc": [0.50, 0.10]}, db=db)
    d = registry.latest_by_config("copy:8", "acc", db=db)
    assert d["baseline"][0] == 0.90 and d["loop3"][0] == 0.50
    registry.record("copy:8", "baseline", 3, 500, {"acc": [0.95, 0.01]}, db=db)   # latest wins
    assert registry.latest_by_config("copy:8", "acc", db=db)["baseline"][0] == 0.95
