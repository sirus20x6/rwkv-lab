"""CPU test for JTP joint multi-token prediction and its composition with the Belief State
Transformer (lookahead_module).

Run: python test_jtp.py
"""
import torch
import torch.nn as nn
from lookahead_module import JTPHead, jtp_loss, LookaheadSystem

C, V, B, T = 64, 500, 2, 24


def test_jtp_grads():
    torch.manual_seed(0)
    h = torch.randn(B, T, C, requires_grad=True)
    ids = torch.randint(0, V, (B, T + 6))
    emb = nn.Embedding(V, C); lm = nn.Linear(C, V, bias=False)
    head = JTPHead(C, D=4, gamma=0.5)
    l = jtp_loss(head, h, ids, emb, lm)
    assert torch.isfinite(l)
    l.backward()
    # grad reaches the FORWARD hidden (the whole point: enrich h) and the Fetch output side
    assert h.grad.abs().sum() > 0, "no gradient to the forward hidden"
    assert head.o.weight.grad.abs().sum() > 0, "no gradient to Fetch output (o)"
    # q/k/v are zero-grad at init (o zero-init => Fetch no-op) but train once o is active
    with torch.no_grad():
        head.o.weight.normal_(0, 0.02)
    head.zero_grad()
    jtp_loss(head, h.detach().requires_grad_(True), ids, emb, lm).backward()
    assert sum(float(p.grad.abs().sum()) for p in (head.q.weight, head.k.weight, head.v.weight)) > 0
    print("[jtp] loss finite; grads -> forward hidden + Fetch self-attn — OK")


def test_short_sequence_noop():
    head = JTPHead(C, D=4)
    l = jtp_loss(head, torch.randn(1, 3, C), torch.randint(0, V, (1, 3)),
                 nn.Embedding(V, C), nn.Linear(C, V, bias=False))
    assert float(l) == 0.0     # T too short for D=4 targets
    print("[jtp] short seq -> zero loss — OK")


def test_composes_with_bst():
    lm = nn.Linear(C, V, bias=False)
    s = LookaheadSystem(C, V, jtp_weight=1.0, jtp_d=4, bst_weight=0.1, bst_lambda=0.0,
                        bst_pairs=8, lm_head=lm)
    assert s.enabled and s.jtp is not None and s.bst is not None
    out = s.compute(torch.randn(B, T, C), torch.randint(0, V, (B, T + 6)), nn.Embedding(V, C), lm)
    assert "jtp" in out and "bst" in out and torch.isfinite(out["aux_total"])
    # both contribute to the aggregate (forward-joint + backward-prev on one shared hidden)
    assert out["jtp"] > 0 and out["bst"] > 0
    print(f"[jtp+bst] both objectives fire on one hidden (jtp {out['jtp']:.2f}, bst {out['bst']:.2f}) — OK")


if __name__ == "__main__":
    test_jtp_grads()
    test_short_sequence_noop()
    test_composes_with_bst()
    print("\nall JTP (+ BST composition) tests passed")
