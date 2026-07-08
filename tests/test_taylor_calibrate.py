"""CPU test for taylor_calibrate.py (half-life decay inversion + look-back distance).

Run: RWKV8_FORCE_PYREF=1 python test_taylor_calibrate.py
"""
import os, math
os.environ.setdefault("RWKV8_FORCE_PYREF", "1")
import torch
import torch.nn.functional as F
from rwkv8_deltanet import RWKV8TimeMixDeltaNet
from taylor_calibrate import teacher_lookback_distance, set_halflife_decay, value_rms_match

C, H, N = 64, 4, 16


def _core():
    torch.manual_seed(0)
    return RWKV8TimeMixDeltaNet(C, num_heads=H, head_size=N, layer_idx=0, depth_layer_id=0, depth_n_layer=32)


def test_halflife_roundtrip():
    core = _core()
    d_target = torch.tensor([2.0, 8.0, 32.0, 128.0])
    set_halflife_decay(core, d_target)
    w0 = core.w0.data.view(H, N)[:, 0].float()            # per-head; all N channels equal
    w = -F.softplus(-w0) - 0.5                            # data-independent init (w1=0)
    decay = torch.exp(-torch.exp(w))
    halflife = math.log(2) / (-torch.log(decay))          # decay^hl = 1/2
    assert torch.allclose(halflife, d_target, rtol=2e-3), f"{halflife.tolist()} != {d_target.tolist()}"
    # all N channels of a head identical
    assert torch.allclose(core.w0.data.view(H, N), core.w0.data.view(H, N)[:, :1].expand(H, N))
    print(f"[halflife] target {d_target.tolist()} -> realized {[round(float(x),2) for x in halflife]} — OK")


def test_lookback():
    B, T = 2, 24
    A = torch.zeros(B, H, T, T)
    for t in range(T):                                    # each query attends 4 tokens back
        A[:, :, t, max(0, t - 4)] = 1.0
    d = teacher_lookback_distance(A)
    assert (d > 1.0).all() and d.max() < 4.5, d.tolist()  # ~4 for t>=4, less for early t
    # uniform attention should give a larger look-back than the local one
    U = torch.tril(torch.ones(T, T)); U = (U / U.sum(-1, keepdim=True))[None, None].expand(B, H, T, T)
    du = teacher_lookback_distance(U)
    assert (du > d).all(), "uniform attn should look back farther than local"
    print(f"[lookback] local~{float(d.mean()):.2f}  uniform~{float(du.mean()):.2f} — OK")


def test_value_rms_match():
    core = _core()
    torch.manual_seed(1)
    tout = torch.randn(2, 10, C) * 3.0                    # teacher output (large amplitude)
    sout = torch.randn(2, 10, C) * 1.0                    # student output (small)
    w_before = core.value.weight.data.clone()
    s = value_rms_match(core, tout, sout)
    assert (s >= 0.2).all() and (s <= 5.0).all(), s.tolist()
    assert not torch.equal(core.value.weight.data, w_before), "value weight not scaled"
    print(f"[value-rms] per-head scale {[round(float(x),2) for x in s]} — OK")


if __name__ == "__main__":
    test_halflife_roundtrip()
    test_lookback()
    test_value_rms_match()
    print("\nall Taylor-Calibrate tests passed")
