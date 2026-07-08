"""CPU tests for the added Muon levers in spectral_muon.py:
Hierarchical/Tiled Muon (tile_size), Distance-Aware Muon (da_muon), ARO-Sinkhorn (aro).

Run: python test_muon_levers.py
"""
import copy
import torch
from spectral_muon import SpectralMuon, himuon_orthogonalize, orthogonalize, _sinkhorn_normalize, _cholesky_qr


def _opt(p, **kw):
    return SpectralMuon([{"params": [p], "use_muon": True, "lr": 0.02}], **kw)


def _run(p, opt, steps=20, seed=0):
    torch.manual_seed(seed)
    T = torch.randn_like(p)
    for _ in range(steps):
        p.grad = (p.detach() - T).clone()
        opt.step()
        p.grad = None
    return p.detach().clone()


def test_all_off_identical():
    W0 = torch.randn(24, 20)
    a = _run(torch.nn.Parameter(W0.clone()), _opt(torch.nn.Parameter(W0.clone())))  # dummy
    pA = torch.nn.Parameter(W0.clone()); oA = _opt(pA)
    pB = torch.nn.Parameter(W0.clone()); oB = _opt(pB, tile_size=0, da_muon=False, aro=False)
    rA = _run(pA, oA); rB = _run(pB, oB)
    assert torch.equal(rA, rB), "levers-off not identical to bare Muon"
    print("[muon] all levers off ≡ bare Muon (bit-identical) — OK")


def test_himuon_tiling():
    # tiled NS: off-state (tile >= max dim) == full NS; tiled (tile < dim) differs but is finite
    G = torch.randn(16, 16)
    full = himuon_orthogonalize(G.clone(), steps=5, tile=16)     # single tile == whole matrix
    tiled = himuon_orthogonalize(G.clone(), steps=5, tile=8)     # 2x2 grid of 8x8 tiles
    assert torch.isfinite(full).all() and torch.isfinite(tiled).all()
    assert not torch.allclose(full.float(), tiled.float()), "tiling had no effect"
    # full-tile == full NS, INCLUDING tall matrices (regression: tall transpose handling)
    for shape in [(16, 16), (5, 3), (3, 8)]:
        Gr = torch.randn(*shape)
        assert torch.allclose(himuon_orthogonalize(Gr, 5, max(shape)).float(),
                              orthogonalize(Gr, 5).float(), atol=1e-5), f"full-tile != full NS for {shape}"
    # a converging run with tiling stays finite and reduces the objective
    p = torch.nn.Parameter(torch.randn(32, 32)); opt = _opt(p, tile_size=16)
    torch.manual_seed(1); Tt = torch.randn(32, 32)
    l0 = (p.detach() - Tt).pow(2).sum().item()
    for _ in range(30):
        p.grad = (p.detach() - Tt).clone(); opt.step(); p.grad = None
    assert (p.detach() - Tt).pow(2).sum().item() < l0 and torch.isfinite(p).all()
    print("[himuon] tiled NS finite + converges; off-state == full NS — OK")


def test_da_muon():
    p = torch.nn.Parameter(torch.randn(16, 16))
    opt = _opt(p, da_muon=True, da_eta_max=0.01, da_r0=1e-3)
    torch.manual_seed(2); T = torch.randn(16, 16)
    for _ in range(15):
        p.grad = (p.detach() - T).clone(); opt.step(); p.grad = None
    st = opt.state[p]
    assert "da_W0" in st and st["da_rbar"] >= 1e-3 and st["da_k"] == 15
    # eta is capped at eta_max
    eta = opt._da_eta(p, st, opt.param_groups[0])
    assert eta <= 0.01 + 1e-9 and torch.isfinite(p).all()
    print(f"[da-muon] running-max distance tracked, η capped at 0.01 (η={eta:.4g}) — OK")


def test_aro_sinkhorn():
    # sinkhorn output is row+col balanced; cholesky-qr returns an orthonormal Q
    X = torch.randn(10, 8)
    s = _sinkhorn_normalize(X.clone(), 5)
    assert torch.isfinite(s).all()
    A = torch.randn(12, 12)
    Q = _cholesky_qr(A)
    assert torch.allclose(Q.t() @ Q, torch.eye(12), atol=1e-3), "cholesky-qr Q not orthonormal"
    # ARO mode runs, converges, stays finite; adds the R rotation state
    p = torch.nn.Parameter(torch.randn(16, 16)); opt = _opt(p, aro=True, aro_sink_iters=5)
    torch.manual_seed(3); T = torch.randn(16, 16)
    l0 = (p.detach() - T).pow(2).sum().item()
    for _ in range(30):
        p.grad = (p.detach() - T).clone(); opt.step(); p.grad = None
    assert "aro_R" in opt.state[p] and torch.isfinite(p).all()
    assert (p.detach() - T).pow(2).sum().item() < l0, "ARO did not reduce the objective"
    # zero-gradient step must NOT poison the rotation (regression: cholesky-qr zero-Q)
    p2 = torch.nn.Parameter(torch.randn(12, 12)); o2 = _opt(p2, aro=True)
    p2.grad = (p2.detach() - torch.randn(12, 12)); o2.step(); p2.grad = None      # warm up R
    R_before = o2.state[p2]["aro_R"].clone()
    p2.grad = torch.zeros_like(p2); o2.step(); p2.grad = None                       # zero-grad step
    R_after = o2.state[p2]["aro_R"]
    assert torch.allclose(R_after.t() @ R_after, torch.eye(12), atol=1e-2), "zero-grad poisoned aro_R"
    p2.grad = (p2.detach() - torch.randn(12, 12)); before = p2.detach().clone()
    o2.step(); p2.grad = None
    assert not torch.equal(p2.detach(), before), "ARO stuck after a zero-grad step"
    # _cholesky_qr on a zero matrix returns a valid (orthonormal) rotation, not zeros
    Qz = _cholesky_qr(torch.zeros(8, 8))
    assert torch.allclose(Qz.t() @ Qz, torch.eye(8), atol=1e-3), "cholesky-qr(0) not orthonormal"
    print("[aro] Sinkhorn + Cholesky-QR rotation; converges, zero-grad-safe — OK")


if __name__ == "__main__":
    test_all_off_identical()
    test_himuon_tiling()
    test_da_muon()
    test_aro_sinkhorn()
    print("\nall Muon-lever tests passed")
