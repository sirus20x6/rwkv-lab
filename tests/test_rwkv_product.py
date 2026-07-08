"""CPU test for RWKVProduct (multi-substep delta-rule core).

Run: RWKV8_FORCE_PYREF=1 python test_rwkv_product.py
"""
import os
os.environ.setdefault("RWKV8_FORCE_PYREF", "1")
import torch
from rwkv_product import RWKVProduct

C, H, N, T, B = 64, 4, 16, 10, 2


def _mk(M):
    torch.manual_seed(0)
    m = RWKVProduct(C, num_heads=H, head_size=N, M=M, lora_rank=8, layer_idx=0)
    with torch.no_grad():
        m.output.weight.copy_(torch.randn(C, C) * 0.1)          # defeat zero-init readout
    return m


def test_interleave_roundtrip():
    # the stack(dim=2).reshape / reshape[...,M-1] plumbing must select the last sub-step
    M = 3
    subs = [torch.full((B, T, H, N), float(j)) for j in range(M)]
    inter = torch.stack(subs, dim=2).reshape(B, T * M, H, N)
    last = inter.reshape(B, T, M, H, N)[:, :, M - 1]
    assert torch.equal(last, torch.full((B, T, H, N), float(M - 1))), "interleave/deinterleave wrong"
    # and the interleave order is token-major (t0:[s0,s1,s2], t1:[...])
    assert torch.equal(inter[0, :M, 0, 0], torch.tensor([0.0, 1.0, 2.0]))
    print("[interleave] token-major order + last-substep selection — OK")


def test_finite_and_grad():
    x = torch.randn(B, T, C)
    for M in (1, 2, 3):
        m = _mk(M)
        y = m(x)
        assert y.shape == (B, T, C) and torch.isfinite(y).all(), f"M={M} bad output"
        y.pow(2).mean().backward()
        # grads reach the per-substep LoRA (B side carries grad when B is zero-init) + base proj
        gB = sum(float(Bm.grad.abs().sum()) for Bm in m.B_b if Bm.grad is not None)
        gr = float(m.receptance.weight.grad.abs().sum())
        assert gB > 0 and gr > 0, f"M={M} grads not flowing (B={gB}, r={gr})"
    print("[finite+grad] M=1,2,3 finite, shape ok, grads flow to LoRA + projections — OK")


def test_m_changes_function():
    x = torch.randn(B, T, C)
    torch.manual_seed(0); y1 = _mk(1)(x)
    torch.manual_seed(0); y2 = _mk(2)(x)
    # even with zero-init LoRA, M=2 runs two delta sub-steps/token => different recurrence
    assert not torch.allclose(y1, y2), "M=2 identical to M=1 (extra sub-step had no effect)"
    print(f"[expressiveness] M=2 differs from M=1 (maxΔ {float((y1-y2).abs().max()):.3e}) — OK")


def test_return_state_raises():
    m = _mk(2)
    try:
        m(torch.randn(B, T, C), return_state=True)
    except NotImplementedError:
        print("[scope] return_state raises NotImplementedError — OK")
        return
    raise AssertionError("return_state should raise")


if __name__ == "__main__":
    test_interleave_roundtrip()
    test_finite_and_grad()
    test_m_changes_function()
    test_return_state_raises()
    print("\nall RWKV-Product tests passed")
