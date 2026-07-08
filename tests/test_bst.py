"""CPU test for the Belief State Transformer aux head (lookahead_module).

Run: python test_bst.py
"""
import torch
import torch.nn as nn
from rwkv_lab.lookahead_module import BeliefStateHead, bst_loss, LookaheadSystem

D, V, B, T = 64, 500, 2, 24


def test_bst_grads():
    torch.manual_seed(0)
    H = torch.randn(B, T, D, requires_grad=True)
    ids = torch.randint(0, V, (B, T + 2))
    emb = nn.Embedding(V, D); lm = nn.Linear(D, V, bias=False)
    head = BeliefStateHead(D, backward_layers=1)
    l = bst_loss(head, H, ids, emb, lm, lam=0.25, n_pairs=16)
    assert torch.isfinite(l)
    l.backward()
    gB = sum(float(p.grad.abs().sum()) for p in head.benc.parameters() if p.grad is not None)
    gW = float(head.wn.weight.grad.abs().sum()) + float(head.wp.weight.grad.abs().sum())
    assert gB > 0, "no gradient to the backward GRU (suffix encoder)"
    assert gW > 0, "no gradient to the fusion heads"
    assert H.grad.abs().sum() > 0, "no gradient to the forward hidden (base model not trained)"
    print(f"[bst] loss {float(l):.3f}; grads -> backward GRU, wn/wp, and forward hidden — OK")


def test_short_sequence_noop():
    # T<3 has no valid (i,j) pair with a gap -> zero loss, no crash
    head = BeliefStateHead(D)
    l = bst_loss(head, torch.randn(1, 2, D), torch.randint(0, V, (1, 2)),
                 nn.Embedding(V, D), nn.Linear(D, V, bias=False))
    assert float(l) == 0.0
    print("[bst] T<3 -> zero loss (no valid pair) — OK")


def test_system_wiring():
    lm = nn.Linear(D, V, bias=False)
    assert not LookaheadSystem(D, V, lm_head=lm).enabled
    s = LookaheadSystem(D, V, bst_weight=0.1, bst_pairs=8, lm_head=lm)
    assert s.enabled and s.bst is not None
    out = s.compute(torch.randn(B, T, D), torch.randint(0, V, (B, T + 2)), nn.Embedding(V, D), lm)
    assert "bst" in out and torch.isfinite(out["aux_total"])
    print("[bst] system enable/disable + compute wiring — OK")


if __name__ == "__main__":
    test_bst_grads()
    test_short_sequence_noop()
    test_system_wiring()
    print("\nall Belief State tests passed")
