"""CPU tests for the RSAV (SpecMuon-inspired) lever in spectral_muon.py.

Guarantees checked:
  1. INERT-WHEN-OFF: rsav=False produces a byte-identical trajectory to a run with
     rsav=True but rsav_cap=0 (ξ forced to 1) — i.e. the RSAV plumbing is a no-op on
     the weights unless ξ is actually allowed to move. Also rsav=False leaves no r state.
  2. ACTIVE-SANE: rsav=True with a real cap converges a toy quadratic, keeps ξ inside
     [1-cap, 1+cap], and keeps r finite/positive (no NaN/Inf).
  3. DAMPING: on a deliberately spiking-energy problem, RSAV actually pulls ξ below 1.

Run: python test_spectral_muon_rsav.py   (CPU-only, no GPU/fla needed)
"""
import copy
import torch

from spectral_muon import SpectralMuon

torch.manual_seed(0)


def _make(W_init, **kw):
    p = torch.nn.Parameter(W_init.clone())
    grp = dict(params=[p], use_muon=True, lr=0.02)
    opt = SpectralMuon([grp], **kw)
    return p, opt


def test_inert_when_off():
    W0 = torch.randn(8, 6)
    T = torch.randn(8, 6)
    # A: rsav off ; B: rsav on but cap=0 -> xi clamped to exactly 1.0
    pA, oA = _make(W0, rsav=False)
    pB, oB = _make(W0, rsav=True, rsav_cap=0.0, rsav_c=1.0)
    for _ in range(40):
        gA = (pA.detach() - T)
        gB = (pB.detach() - T)
        pA.grad = gA.clone(); pB.grad = gB.clone()
        oA.step(); oB.step()
        pA.grad = None; pB.grad = None
    diff = (pA.detach() - pB.detach()).abs().max().item()
    assert diff < 1e-6, f"rsav cap=0 not inert vs rsav-off: max|Δ|={diff:.2e}"
    assert oA._rsav_r is None, "rsav=False should create no r state"
    assert oB._rsav_r is not None, "rsav=True should have r state"
    print(f"[1] inert-when-off OK (max|Δ|={diff:.2e}, r_off={oA._rsav_r})")


def test_active_sane():
    W0 = torch.randn(10, 10)
    T = torch.randn(10, 10)
    cap = 0.2
    p, opt = _make(W0, rsav=True, rsav_cap=cap, rsav_c=5.0, rsav_relax=0.1)
    loss0 = 0.5 * (p.detach() - T).pow(2).sum().item()
    xis = []
    for _ in range(80):
        p.grad = (p.detach() - T).clone()
        opt.step()
        p.grad = None
        xis.append(opt._rsav_last_xi)
        assert torch.isfinite(opt._rsav_r).all(), "r went non-finite"
        assert opt._rsav_r.item() > 0, "r went non-positive"
    loss1 = 0.5 * (p.detach() - T).pow(2).sum().item()
    lo, hi = min(xis), max(xis)
    assert loss1 < loss0, f"did not converge: {loss0:.3f} -> {loss1:.3f}"
    assert 1 - cap - 1e-6 <= lo and hi <= 1 + cap + 1e-6, f"xi out of band: [{lo:.3f},{hi:.3f}]"
    assert torch.isfinite(p).all(), "weights went non-finite"
    print(f"[2] active-sane OK (loss {loss0:.3f}->{loss1:.3f}, xi∈[{lo:.3f},{hi:.3f}])")


def test_damps_on_energy_spike():
    # Inject a growing-gradient (rising-energy) schedule; RSAV's r lags, so xi should dip <1.
    W0 = torch.zeros(6, 6)
    p, opt = _make(W0, rsav=True, rsav_cap=0.5, rsav_c=1.0, rsav_relax=0.0)
    saw_below_1 = False
    for k in range(30):
        # energy climbs each step -> sqrt(E+C) outruns the lagging r -> xi<1
        p.grad = torch.full((6, 6), 0.1 * (k + 1))
        opt.step(); p.grad = None
        if opt._rsav_last_xi < 0.999:
            saw_below_1 = True
    assert saw_below_1, "RSAV never damped (xi stayed >=1) under a rising-energy schedule"
    print(f"[3] damping OK (xi dipped below 1 on rising energy; last xi={opt._rsav_last_xi:.3f})")


if __name__ == "__main__":
    test_inert_when_off()
    test_active_sane()
    test_damps_on_energy_spike()
    print("\nall RSAV tests passed")
