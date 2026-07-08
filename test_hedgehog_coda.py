"""CPU tests for the portable pieces of Attention-to-Mamba (hedgehog.py, B3) and CODA (coda.py, B7).

Run: CODA_NO_COMPILE=1 python test_hedgehog_coda.py
"""
import os
os.environ.setdefault("CODA_NO_COMPILE", "1")   # keep the test CPU/eager and deterministic
import torch
import torch.nn.functional as F
from hedgehog import HedgehogFeatureMap, linear_attn_map, attn_map_ce_loss
from coda import compile_module, SwiGLU

B, H, T, d = 2, 4, 10, 16


def test_hedgehog():
    fm = HedgehogFeatureMap(d)
    phi = fm(torch.randn(B, H, T, d))
    assert (phi >= 0).all() and torch.allclose(phi.sum(-1), torch.ones(B, H, T)), "phi not a distribution"
    A = linear_attn_map(fm(torch.randn(B, H, T, d)), fm(torch.randn(B, H, T, d)))
    assert A.triu(1).abs().max() < 1e-6, "attention map not causal"
    assert torch.allclose(A.sum(-1), torch.ones(B, H, T), atol=1e-5), "rows do not sum to 1"
    teach = F.softmax(torch.randn(B, H, T, T).masked_fill(
        ~torch.tril(torch.ones(T, T, dtype=torch.bool)), -1e9), dim=-1)
    x = torch.randn(B, H, T, d)
    l = attn_map_ce_loss(teach, fm(x), fm(x))
    assert torch.isfinite(l)
    l.backward()
    assert fm.w.weight.grad.abs().sum() > 0
    print("[hedgehog] phi distribution, causal map, attn-map CE + grads — OK")


def test_coda():
    m = SwiGLU(64, 128)
    mc = compile_module(m)                          # CODA_NO_COMPILE=1 -> eager fallback, no crash
    x = torch.randn(2, 8, 64)
    y = mc(x)
    g, u = m.up(x).chunk(2, -1)
    ref = m.down(F.silu(g) * u)
    assert torch.allclose(y, ref), "SwiGLU numerics changed"
    assert torch.isfinite(y).all()
    print("[coda] compile_module safe fallback + SwiGLU numerics — OK")


if __name__ == "__main__":
    test_hedgehog()
    test_coda()
    print("\nall hedgehog + coda tests passed")
