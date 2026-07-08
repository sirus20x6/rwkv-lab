"""CPU test for the DEQ / 1-step-gradient loop mode in LoopedRWKV (HRM 2506.21734).

Run: pytest tests/test_loop_deq.py
"""
import copy
import torch
import torch.nn as nn
from looped_rwkv import LoopedRWKV


class TinyCore(nn.Module):
    def __init__(self, H=32, G=4, std=0.3):
        super().__init__()
        self.hidden_size = H
        self.num_heads = G
        self.receptance = nn.Linear(H, H)
        self.mix = nn.Linear(H, H)
        nn.init.normal_(self.mix.weight, std=std)     # live core -> real refinement increments
    def forward(self, x, *a, **k):
        return torch.tanh(self.mix(x))


def _pair(seed=0, **kw):
    """Two identical models differing ONLY in loop_deq."""
    torch.manual_seed(seed); a = LoopedRWKV(TinyCore(), n_loops=4, loop_deq=False, **kw)
    torch.manual_seed(seed); b = LoopedRWKV(TinyCore(), n_loops=4, loop_deq=True, **kw)
    b.load_state_dict(a.state_dict())
    return a, b


def test_deq_forward_value_equals_full_bptt():
    # The defining invariant: no_grad/detach don't change VALUES, so the DEQ forward output
    # must be bit-identical to the full-BPTT loop (only the gradient graph differs).
    a, b = _pair(seed=1)
    x = torch.randn(2, 6, 32)
    with torch.enable_grad():
        oa = a(x.clone()); ob = b(x.clone())
    assert torch.allclose(oa, ob, atol=1e-6), "DEQ forward value diverged from full-BPTT"
    print("[deq] forward value == full-BPTT (only the gradient graph is cheaper) — OK")


def test_deq_matches_full_bptt_value_with_cart():
    a, b = _pair(seed=2, cart_anchor=True)     # DEQ composes with the CART contractive gate
    x = torch.randn(2, 5, 32)
    with torch.enable_grad():
        assert torch.allclose(a(x.clone()), b(x.clone()), atol=1e-6)
    print("[deq] composes with cart_anchor (value identical) — OK")


def test_deq_gradient_is_one_step():
    # The DEQ gradient must equal the gradient of ONLY the final refine step taken from the
    # detached fixed point — i.e. gradients differ from full-BPTT, and match a hand-rolled 1-step.
    torch.manual_seed(3)
    core = TinyCore()
    m = LoopedRWKV(copy.deepcopy(core), n_loops=4, loop_deq=True)
    x = torch.randn(1, 4, 32)
    m(x).sum().backward()
    g_deq = m.core.mix.weight.grad.clone()

    # hand-rolled reference: run 2 detached refine passes, then 1 graded pass, same as the impl
    ref = LoopedRWKV(copy.deepcopy(core), n_loops=4, loop_deq=False)
    ref.load_state_dict(m.state_dict())
    with torch.no_grad():
        out = ref._t(ref.core(x))
        for i in (1, 2):
            out = out + ref._gate(i) * ref._t(ref.core(ref.iter_norm(x + out)))
    out = out.detach()
    graded = out + ref._gate(3) * ref._t(ref.core(ref.iter_norm(x + out)))
    graded.sum().backward()
    g_ref = ref.core.mix.weight.grad
    assert torch.allclose(g_deq, g_ref, atol=1e-5), "DEQ gradient != hand-rolled 1-step gradient"
    print("[deq] gradient == one graded step from the detached fixed point (Neumann-1) — OK")


def test_deq_is_eval_transparent():
    # Under no_grad (eval), DEQ takes the normal loop path -> identical to loop_deq off.
    a, b = _pair(seed=4)
    x = torch.randn(2, 5, 32)
    with torch.no_grad():
        assert torch.equal(a(x.clone()), b(x.clone()))
    print("[deq] eval (no_grad) path identical to the plain loop — OK")


def test_deq_exclusivity():
    for kw, msg in [(dict(adaptive_halt=True), "adaptive_halt"), (dict(hyper_lanes=4), "hyper")]:
        try:
            LoopedRWKV(TinyCore(), n_loops=4, loop_deq=True, **kw); assert False
        except ValueError as e:
            assert "loop_deq" in str(e)
    m = LoopedRWKV(TinyCore(), n_loops=4, loop_deq=True)
    m.iter_consist = True                       # trainer-set attr -> forward must reject
    try:
        m(torch.randn(1, 4, 32)); assert False
    except ValueError as e:
        assert "iter_consist" in str(e)
    try:
        LoopedRWKV(TinyCore(), n_loops=1, loop_deq=True); assert False
    except ValueError as e:
        assert "n_loops" in str(e)
    print("[deq] rejects halt/hyper/iter-consist/n_loops<2 — OK")


if __name__ == "__main__":
    test_deq_forward_value_equals_full_bptt()
    test_deq_matches_full_bptt_value_with_cart()
    test_deq_gradient_is_one_step()
    test_deq_is_eval_transparent()
    test_deq_exclusivity()
    print("\nall DEQ loop-mode tests passed")
