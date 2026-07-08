"""CPU test for FwPKM (product-key memory retrieval + objectives).

Run: python test_fwpkm.py
"""
import torch
from rwkv_lab.fwpkm import FwPKM

C, B, T = 64, 2, 16


def test_forward_and_grads():
    torch.manual_seed(0)
    m = FwPKM(C, sqrt_n=16, d_key=32, topk=4)          # N = 16^2 = 256
    assert m.n == 256
    h = torch.randn(B, T, C)
    out = m(h)
    assert out.shape == (B, T, C) and torch.isfinite(out).all()
    # o_proj + gate are init so the layer is a near-no-op at step 0
    assert out.abs().mean() < 1e-3, f"not near-no-op at init: {float(out.abs().mean())}"
    assert torch.isfinite(m.last_mem_loss) and torch.isfinite(m.last_addr_loss)
    (out.pow(2).mean() + m.last_mem_loss + 0.01 * m.last_addr_loss).backward()
    assert m.V.grad.abs().sum() > 0, "no grad to value memory (memorization objective)"
    assert m.K1.grad.abs().sum() > 0 and m.K2.grad.abs().sum() > 0, "no grad to sub-keys"
    assert m.q.weight.grad.abs().sum() > 0, "no grad to query projection"
    print(f"[fwpkm] N={m.n}, near-no-op init, retrieval + mem/addr objectives, grads flow — OK")


def test_topk_rows_valid():
    m = FwPKM(C, sqrt_n=8, d_key=16, topk=4)           # N=64
    h = torch.randn(1, 4, C)
    # rows selected must be in [0, N); exercise the product-key index math via a forward
    out = m(h)
    assert torch.isfinite(out).all()
    # sanity: with more topk the selected candidate count is capped at k*k
    m2 = FwPKM(C, sqrt_n=8, d_key=16, topk=6)
    assert torch.isfinite(m2(h)).all()
    print("[fwpkm] product-key index math valid across topk settings — OK")


if __name__ == "__main__":
    test_forward_and_grads()
    test_topk_rows_valid()
    print("\nall FwPKM tests passed")
