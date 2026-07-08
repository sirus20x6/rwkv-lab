"""CPU test for RAD-RWKV7 RoPE in RWKV8TimeMixDeltaNet (use_rope).

Guarantees:
  1. use_rope=False is BIT-IDENTICAL to the pre-RoPE path (no behavior change when off).
  2. use_rope=True changes the output (rotary is actually applied to r/k) and is
     position-sensitive (shifting position_ids changes the output).
  3. Output stays finite.

The mixer's output projection is zero-init (near-identity swap at step 0), so the test
sets a shared non-zero output weight to make the internal rotary effect observable.

Run: RWKV8_FORCE_PYREF=1 python test_rwkv_rope.py   (CPU, no fla/GPU needed)
"""
import os
os.environ.setdefault("RWKV8_FORCE_PYREF", "1")
import torch
from rwkv8_deltanet import RWKV8TimeMixDeltaNet

C, H, N, T, B = 64, 4, 16, 12, 2


def _mk(**kw):
    torch.manual_seed(1)
    return RWKV8TimeMixDeltaNet(C, num_heads=H, head_size=N, layer_idx=0,
                                depth_layer_id=0, depth_n_layer=32, **kw).eval()


def test_rope():
    torch.manual_seed(0)
    x = torch.randn(B, T, C)
    base = _mk(); off = _mk(use_rope=False); on = _mk(use_rope=True, rope_theta=1e7, rope_frac=0.25)
    sd = base.state_dict()
    off.load_state_dict(sd, strict=False); on.load_state_dict(sd, strict=False)
    with torch.no_grad():
        w = torch.randn(C, C) * 0.1
        for m in (base, off, on):
            m.output.weight.copy_(w)                      # defeat the zero-init readout
        yb, yo, yon = base(x), off(x), on(x)
        pa = torch.arange(T)[None].expand(B, T)
        pb = (torch.arange(T) + 5)[None].expand(B, T)
        ypa, ypb = on(x, position_ids=pa), on(x, position_ids=pb)

    assert yb.abs().max() > 0, "degenerate test: readout is zero"
    assert torch.equal(yo, yb), f"use_rope=False not identical: maxΔ={(yo-yb).abs().max():.2e}"
    assert on.rope_dim == int(N * 0.25), f"rope_dim={on.rope_dim}"
    assert not torch.allclose(yon, yb), "RoPE on did not change the output"
    assert not torch.allclose(ypa, ypb), "RoPE output is not position-sensitive"
    assert torch.isfinite(yon).all(), "RoPE output non-finite"
    print(f"[rope] off≡vanilla (Δ=0); on Δ={float((yon-yb).abs().max()):.2e}; "
          f"pos-shift Δ={float((ypa-ypb).abs().max()):.2e}; rope_dim={on.rope_dim} — OK")


if __name__ == "__main__":
    test_rope()
    print("\nRoPE test passed")
