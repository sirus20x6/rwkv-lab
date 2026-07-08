"""CPU test for Comba decoupled-removal in RWKV8TimeMixDeltaNet (comba_decouple).

Run: RWKV8_FORCE_PYREF=1 python test_comba.py
"""
import os
os.environ.setdefault("RWKV8_FORCE_PYREF", "1")
import torch
from rwkv8_deltanet import RWKV8TimeMixDeltaNet

C, H, N = 64, 4, 16


def _mk(**kw):
    torch.manual_seed(1)
    return RWKV8TimeMixDeltaNet(C, num_heads=H, head_size=N, layer_idx=0,
                                depth_layer_id=0, depth_n_layer=32, **kw)


def test_comba():
    torch.manual_seed(0)
    x = torch.randn(2, 10, C)
    base = _mk(); off = _mk(comba_decouple=False); on = _mk(comba_decouple=True)
    sd = base.state_dict()
    off.load_state_dict(sd, strict=False); on.load_state_dict(sd, strict=False)
    with torch.no_grad():
        w = torch.randn(C, C) * 0.1
        for m in (base, off, on):
            m.output.weight.copy_(w)
        yb, yo, yon = base(x), off(x), on(x)
    assert torch.equal(yo, yb), "comba_decouple=False not identical"
    assert on.comba_b.shape == (H,), "comba_b should be per-head"
    # b_fb = sigmoid(0) = 0.5 halves the removal term -> output differs from base
    assert not torch.allclose(yon, yb), "comba on had no effect"
    assert torch.isfinite(yon).all()
    on.zero_grad(); on(x).pow(2).mean().backward()
    assert float(on.comba_b.grad.abs().sum()) > 0, "no grad to comba_b"
    print(f"[comba] off=vanilla (Δ=0); on (b_fb=0.5) Δ={float((yon-yb).abs().max()):.2e}; grads flow — OK")


if __name__ == "__main__":
    test_comba()
    print("\nComba test passed")
