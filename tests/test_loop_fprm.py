"""CPU tests for the two FPRM levers in LoopedRWKV (arXiv:2606.18206):
  - k-window truncated BPTT (deq_window): generalizes --loop-deq from Neumann-1 to Neumann-k.
  - fixed-point-residual halting (fixed_point_halt): stop when the iterate reaches a fixed point.

Run: pytest tests/test_loop_fprm.py
"""
import copy
import torch
import torch.nn as nn
from rwkv_lab.looped_rwkv import LoopedRWKV


class TinyCore(nn.Module):
    def __init__(self, H=32, G=4, std=0.3):
        super().__init__()
        self.hidden_size = H
        self.num_heads = G
        self.receptance = nn.Linear(H, H)
        self.mix = nn.Linear(H, H)
        nn.init.normal_(self.mix.weight, std=std)
    def forward(self, x, *a, **k):
        return torch.tanh(self.mix(x))


# ---------------- Lever A: k-window truncated BPTT ----------------

def test_kwindow_forward_value_equals_full_bptt():
    # forward VALUE must be identical for any window (detach/no_grad don't change values)
    torch.manual_seed(0); core = TinyCore()
    x = torch.randn(2, 5, 32)
    ref = LoopedRWKV(copy.deepcopy(core), n_loops=5, loop_deq=False)
    ref.residual_weight.data.fill_(0.5)                   # un-zero the gate -> the loop actually moves
    for w in (1, 2, 3):
        m = LoopedRWKV(copy.deepcopy(core), n_loops=5, loop_deq=True, deq_window=w)
        m.load_state_dict(ref.state_dict())
        with torch.enable_grad():
            assert torch.allclose(m(x.clone()), ref(x.clone()), atol=1e-6), f"window={w} value != full-BPTT"
    print("[fprm] k-window DEQ forward value == full-BPTT for k∈{1,2,3} — OK")


def test_kwindow_gradient_is_truncated_bptt():
    # deq_window=2 gradient must equal a hand-rolled 2-step truncated BPTT from the detached state
    torch.manual_seed(1); core = TinyCore()
    x = torch.randn(1, 4, 32)
    m = LoopedRWKV(copy.deepcopy(core), n_loops=5, loop_deq=True, deq_window=2)
    m.residual_weight.data.fill_(0.5)                    # non-zero gate so increments carry gradient
    m(x).sum().backward(); g = m.core.mix.weight.grad.clone()

    ref = LoopedRWKV(copy.deepcopy(core), n_loops=5, loop_deq=False); ref.load_state_dict(m.state_dict())
    with torch.no_grad():                                  # detached passes 1,2 (n_loops-1-w = 2)
        out = ref._t(ref.core(x))
        for i in (1, 2):
            out = out + ref._gate(i) * ref._t(ref.core(ref.iter_norm(x + out)))
    out = out.detach()
    for i in (3, 4):                                       # graded window of 2
        out = out + ref._gate(i) * ref._t(ref.core(ref.iter_norm(x + out)))
    out.sum().backward()
    assert torch.allclose(g, ref.core.mix.weight.grad, atol=1e-5), "k=2 grad != hand-rolled 2-window BPTT"
    # and it differs from the Neumann-1 gradient
    m1 = LoopedRWKV(copy.deepcopy(core), n_loops=5, loop_deq=True, deq_window=1)
    m1.load_state_dict(m.state_dict())                   # includes the 0.5 gate
    m1(x).sum().backward()
    assert not torch.allclose(g, m1.core.mix.weight.grad, atol=1e-5), "k=2 grad identical to k=1 (window ignored)"
    print("[fprm] k-window gradient == truncated 2-step BPTT, ≠ Neumann-1 — OK")


def test_window_clamped():
    core = TinyCore()
    m = LoopedRWKV(core, n_loops=3, loop_deq=True, deq_window=99)   # clamp to n_loops-1=2
    x = torch.randn(1, 4, 32)
    assert torch.isfinite(m(x)).all()
    print("[fprm] deq_window clamped to n_loops-1 — OK")


# ---------------- Lever B: fixed-point-residual halting ----------------

def test_fp_halt_off_runs_all_loops():
    torch.manual_seed(2); core = TinyCore()
    a = LoopedRWKV(copy.deepcopy(core), n_loops=5, fixed_point_halt=False)
    b = LoopedRWKV(copy.deepcopy(core), n_loops=5, fixed_point_halt=True, fp_tol=1e-9)  # tol so tight it never halts
    b.load_state_dict(a.state_dict())
    x = torch.randn(2, 5, 32)
    with torch.no_grad():
        assert torch.allclose(a(x.clone()), b(x.clone()), atol=1e-6)
    print("[fprm] fixed_point_halt with impossible tol ≡ full loop — OK")


def test_fp_halt_early_exits_on_fixed_point():
    # zero-init core -> increment 0 -> the iterate doesn't move -> residual 0 -> halt at fp_min_iters
    core = TinyCore(); nn.init.zeros_(core.mix.weight); nn.init.zeros_(core.mix.bias)
    m = LoopedRWKV(core, n_loops=8, fixed_point_halt=True, fp_tol=1e-3, fp_min_iters=1)
    with torch.no_grad():
        m(torch.randn(2, 4, 32))
    assert m.last_halt_iters is not None and m.last_halt_iters < 8, f"did not halt early ({m.last_halt_iters})"
    print(f"[fprm] converged loop halts early at {m.last_halt_iters}/8 passes — OK")


def test_fp_halt_runs_full_when_not_converged():
    torch.manual_seed(4); core = TinyCore(std=0.8)              # lively core
    m = LoopedRWKV(core, n_loops=5, fixed_point_halt=True, fp_tol=1e-4, fp_min_iters=1)
    m.residual_weight.data.fill_(0.6)                          # real gate -> the loop keeps moving
    with torch.no_grad():
        m(torch.randn(2, 5, 32))
    assert m.last_halt_iters == 5, f"halted early on a non-converging loop ({m.last_halt_iters})"
    print("[fprm] non-converging loop runs all n_loops — OK")


def test_fp_halt_exclusivity():
    for kw in (dict(adaptive_halt=True), dict(hyper_lanes=4), dict(loop_deq=True)):
        try:
            LoopedRWKV(TinyCore(), n_loops=4, fixed_point_halt=True, **kw); assert False
        except ValueError as e:
            assert "fixed_point_halt" in str(e)
    try:
        LoopedRWKV(TinyCore(), n_loops=1, fixed_point_halt=True); assert False
    except ValueError as e:
        assert "n_loops" in str(e)
    print("[fprm] fixed_point_halt rejects adaptive_halt/hyper/loop_deq/n_loops<2 — OK")


if __name__ == "__main__":
    test_kwindow_forward_value_equals_full_bptt()
    test_kwindow_gradient_is_truncated_bptt()
    test_window_clamped()
    test_fp_halt_off_runs_all_loops()
    test_fp_halt_early_exits_on_fixed_point()
    test_fp_halt_runs_full_when_not_converged()
    test_fp_halt_exclusivity()
    print("\nall FPRM lever tests passed")
