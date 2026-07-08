"""CPU test for L-MTP leap multi-token prediction (lookahead_module).

Run: python test_lmtp.py
"""
import torch
import torch.nn as nn
from rwkv_lab.lookahead_module import LeapMTPHead, lmtp_loss, LookaheadSystem

D, V, B, T = 64, 500, 2, 24


def test_offsets_and_loss():
    torch.manual_seed(0)
    h = torch.randn(B, T, D, requires_grad=True)
    ids = torch.randint(0, V, (B, T + 8))
    lm = nn.Linear(D, V, bias=False)
    head = LeapMTPHead(D, n_heads=3, k=2)
    assert [head.offset(j) for j in range(3)] == [3, 5, 7], "leap offsets wrong"
    l = lmtp_loss(head, h, ids, lm)
    assert torch.isfinite(l), "loss non-finite"
    # zero-init adapters => head == backbone next-token predictor at that shift, so the
    # loss equals plain CE(lm_head(h_shifted), target) at init
    with torch.no_grad():
        ref = 0.0; nt = 0
        for j in range(3):
            off = head.offset(j); tc = min(T, ids.shape[1] - off)   # same coverage as lmtp_loss
            lg = lm(h[:, :tc]).reshape(-1, V)
            tg = ids[:, off:off + tc].reshape(-1)
            ref += torch.nn.functional.cross_entropy(lg, tg, reduction="sum"); nt += tg.numel()
        ref = ref / nt
    assert torch.allclose(l, ref, rtol=1e-4), f"init loss {float(l)} != backbone-shift CE {float(ref)}"
    l.backward()
    assert sum(float(a.weight.grad.abs().sum()) for a in head.adapters) > 0, "no adapter grad"
    print(f"[lmtp] offsets {[head.offset(j) for j in range(3)]}, loss {float(l):.3f} == init CE — OK")


def test_system_wiring():
    lm = nn.Linear(D, V, bias=False)
    assert not LookaheadSystem(D, V, lm_head=lm).enabled
    sys_on = LookaheadSystem(D, V, lmtp_weight=0.5, lmtp_k=2, lmtp_heads=3, lm_head=lm)
    assert sys_on.enabled and sys_on.lmtp is not None
    h = torch.randn(B, T, D); ids = torch.randint(0, V, (B, T + 8))
    out = sys_on.compute(h, ids, nn.Embedding(V, D), lm)
    assert "lmtp" in out and torch.isfinite(out["aux_total"])
    print("[lmtp] system enable/disable + compute wiring — OK")


if __name__ == "__main__":
    test_offsets_and_loss()
    test_system_wiring()
    print("\nall L-MTP tests passed")
