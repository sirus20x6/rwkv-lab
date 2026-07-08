"""CPU test for the CART contractive LTI gate in LoopedRWKV (arXiv:2606.01495).

Run: pytest tests/test_loop_cart.py
"""
import copy
import torch
import torch.nn as nn
from looped_rwkv import LoopedRWKV


class TinyCore(nn.Module):
    """Minimal RWKV-core stand-in: a channel mixer with the attrs LoopedRWKV reads."""
    def __init__(self, H=32, G=4):
        super().__init__()
        self.hidden_size = H
        self.num_heads = G
        self.receptance = nn.Linear(H, H)
        self.mix = nn.Linear(H, H)
        nn.init.zeros_(self.mix.weight); nn.init.zeros_(self.mix.bias)  # zero-init => near-identity loop
    def forward(self, x, *a, **k):
        return torch.tanh(self.mix(x))


def _model(cart, seed=0, **kw):
    torch.manual_seed(seed)
    return LoopedRWKV(TinyCore(), n_loops=4, cart_anchor=cart, **kw)


def test_off_is_bit_identical():
    torch.manual_seed(1); x = torch.randn(2, 6, 32)
    a = _model(False, seed=3); b = _model(True, seed=3)
    # copy shared params so the ONLY difference is the cart path (which is off in `a`)
    b.load_state_dict(a.state_dict(), strict=False)
    with torch.no_grad():
        oa = a(x.clone())
        # force cart off on b too -> must match a exactly
        b.cart_anchor = False
        ob = b(x.clone())
    assert torch.equal(oa, ob), "cart-off path diverged from plain loop"
    print("[cart] off ≡ plain loop (bit-identical) — OK")


def test_gate_is_contractive():
    m = _model(True, seed=2)
    g = torch.sigmoid(m.cart_gate)
    assert (g > 0).all() and (g < 1).all(), "sigmoid gate not strictly in (0,1)"
    assert float(g.mean()) > 0.9, "cart_gate_init=4 should start near-identity (σ≈0.98)"
    print(f"[cart] gate σ(g)∈(0,1), starts near-identity (mean {float(g.mean()):.3f}) — OK")


def test_deep_loop_converges_to_fixed_point():
    # With a contractive gate, the carried state should converge as loop depth grows (Cauchy):
    # successive-depth outputs get closer. Give the increment real magnitude via a non-zero core.
    torch.manual_seed(5); x = torch.randn(1, 4, 32)
    core = TinyCore(); nn.init.normal_(core.mix.weight, std=0.3)
    diffs = []
    prev = None
    for n in range(2, 9):
        m = LoopedRWKV(copy.deepcopy(core), n_loops=n, cart_anchor=True, cart_gate_init=0.0)  # σ=0.5
        with torch.no_grad():
            out = m(x.clone())
        if prev is not None:
            diffs.append((out - prev).norm().item())
        prev = out
    # the depth-to-depth change should shrink (contraction) — later diffs < earlier
    assert diffs[-1] < diffs[0], f"deep loop not converging (diffs {diffs})"
    assert all(torch.isfinite(torch.tensor(diffs)))
    print(f"[cart] deep loop converges to a fixed point (Δ {diffs[0]:.3f}→{diffs[-1]:.3f}) — OK")


def test_hyper_mutually_exclusive():
    try:
        LoopedRWKV(TinyCore(), n_loops=4, cart_anchor=True, hyper_lanes=4)
        assert False, "expected ValueError for cart_anchor + hyper_lanes"
    except ValueError as e:
        assert "alternative loop-dynamics" in str(e)
    print("[cart] cart_anchor + hyper_lanes rejected (mutually exclusive) — OK")


def test_grad_flows_to_gate():
    torch.manual_seed(7)
    core = TinyCore(); nn.init.normal_(core.mix.weight, std=0.3)   # live core -> nonzero increment
    m = LoopedRWKV(core, n_loops=4, cart_anchor=True)
    x = torch.randn(2, 5, 32, requires_grad=True)
    m(x).sum().backward()
    assert m.cart_gate.grad is not None and m.cart_gate.grad.abs().sum() > 0
    print("[cart] gradient reaches the cart gate — OK")


if __name__ == "__main__":
    test_off_is_bit_identical()
    test_gate_is_contractive()
    test_deep_loop_converges_to_fixed_point()
    test_hyper_mutually_exclusive()
    test_grad_flows_to_gate()
    print("\nall CART LTI-gate tests passed")
