"""CPU test for PonderNet/ACT adaptive loop halting in LoopedRWKV.

Run: RWKV8_FORCE_PYREF=1 python test_adaptive_halt.py
"""
import os
os.environ.setdefault("RWKV8_FORCE_PYREF", "1")
import torch
from rwkv8_deltanet import RWKV8TimeMixDeltaNet
from looped_rwkv import LoopedRWKV

C, H, N = 64, 4, 16


def _core():
    torch.manual_seed(1)
    return RWKV8TimeMixDeltaNet(C, num_heads=H, head_size=N, layer_idx=0, depth_layer_id=0, depth_n_layer=32)


def test_off_identical():
    x = torch.randn(2, 10, C)
    torch.manual_seed(1); off = LoopedRWKV(_core(), n_loops=3)
    torch.manual_seed(1); base = LoopedRWKV(_core(), n_loops=3)
    with torch.no_grad():
        assert torch.equal(off(x), base(x)), "adaptive_halt=False not identical to plain loop"
    assert off.last_ponder is None
    print("[halt] off = plain loop (bit-identical), no ponder state — OK")


def test_on_halt_weighted():
    x = torch.randn(2, 10, C)
    torch.manual_seed(1); lp = LoopedRWKV(_core(), n_loops=3, adaptive_halt=True)
    with torch.no_grad():
        for p in lp.parameters():
            if p.shape == lp.residual_weight.shape:
                p.copy_(torch.full_like(p, 0.3))
    y = lp(x)
    assert torch.isfinite(y).all() and y.shape == x.shape
    assert lp.last_ponder is not None and torch.isfinite(lp.last_ponder)
    (y.pow(2).mean() + lp.last_ponder).backward()
    gh = float(lp.halt_head.weight.grad.abs().sum()) + float(lp.halt_head.bias.grad.abs().sum())
    assert gh > 0, "no gradient to the halt head"
    print(f"[halt] on: halt-weighted output finite, ponder {float(lp.last_ponder):.3f}, halt-head grads — OK")


def test_needs_multiloop():
    try:
        LoopedRWKV(_core(), n_loops=1, adaptive_halt=True)
    except ValueError:
        print("[halt] n_loops=1 rejected — OK")
        return
    raise AssertionError("adaptive_halt with n_loops=1 should raise")


if __name__ == "__main__":
    test_off_identical()
    test_on_halt_weighted()
    test_needs_multiloop()
    print("\nall adaptive-halt tests passed")
