"""CPU test for L³ Large Lookup Layers (arXiv:2601.21461).

Run: pytest tests/test_l3_lookup.py
"""
import torch
from l3_lookup import LargeLookupLayer, allocate_slots_by_frequency

V, D, E, B, T = 50, 32, 24, 3, 7


def test_residual_off_state_is_identity():
    m = LargeLookupLayer(V, D, E, max_slots=4, integration="residual")
    x = torch.randn(B, T, D); tok = torch.randint(0, V, (B, T))
    assert torch.allclose(m(x, tok), x, atol=1e-6), "zero-init residual L³ not identity at init"
    print("[l3] residual integration is exact identity at init (safe off-state) — OK")


def test_context_dependent_read():
    # The defining L³ property: SAME token, DIFFERENT hidden state -> DIFFERENT read (x is the query).
    # A plain embedding table would return the same vector regardless of x.
    m = LargeLookupLayer(V, D, E, max_slots=6, integration="residual")
    torch.nn.init.normal_(m.out.weight, std=0.05)     # un-zero the output so the read is observable
    tok = torch.zeros(2, dtype=torch.long)            # same token id for both
    x = torch.randn(2, D) * 3.0                       # two different hidden states
    y = m(x.unsqueeze(1), tok.unsqueeze(1)).squeeze(1)
    read = y - x                                      # the L³ contribution
    assert not torch.allclose(read[0], read[1], atol=1e-4), "read not context-dependent (x not querying)"
    print("[l3] same token, different x -> different read (context-dependent softmax) — OK")


def test_grad_to_tables():
    m = LargeLookupLayer(V, D, E, max_slots=4, integration="residual")
    torch.nn.init.normal_(m.out.weight, std=0.05)
    x = torch.randn(B, T, D, requires_grad=True); tok = torch.randint(0, V, (B, T))
    m(x, tok).pow(2).sum().backward()
    assert m.W_K.weight.grad.abs().sum() > 0 and m.W_V.weight.grad.abs().sum() > 0
    print("[l3] gradient reaches the K and V lookup tables — OK")


def test_concat_integration_and_up():
    m = LargeLookupLayer(V, D, E, max_slots=4, up_dim=40, integration="concat")
    x = torch.randn(B, T, D); tok = torch.randint(0, V, (B, T))
    y = m(x, tok)
    assert y.shape == (B, T, D) and torch.isfinite(y).all()
    assert m.W_up is not None
    print("[l3] concat integration + paper-faithful W_up path runs — OK")


def test_tie_kv():
    m = LargeLookupLayer(V, D, d_emb=D, max_slots=4, tie_kv=True)   # d_emb must == d_in
    assert m.W_V is m.W_K
    x = torch.randn(B, T, D); tok = torch.randint(0, V, (B, T))
    assert torch.isfinite(m(x, tok)).all()
    print("[l3] tied K=V table (halved storage) works — OK")


def test_variable_allocation():
    counts = torch.arange(V).float()                  # token freq ~ id
    alloc = allocate_slots_by_frequency(counts, total_slots=V * 3, cap=6, floor=1)
    assert (alloc >= 1).all() and (alloc <= 6).all()
    assert alloc[V - 1] >= alloc[0]                    # frequent token gets >= slots
    m = LargeLookupLayer(V, D, E, max_slots=6, integration="residual")
    m.set_allocation(alloc)
    assert int(m.slot_mask.sum()) == int(alloc.sum())  # table reflects the allocation
    torch.nn.init.normal_(m.out.weight, std=0.05)
    x = torch.randn(B, T, D); tok = torch.randint(0, V, (B, T))
    assert torch.isfinite(m(x, tok)).all()
    print("[l3] variable per-token allocation (freq-proportional, capped) — OK")


if __name__ == "__main__":
    test_residual_off_state_is_identity()
    test_context_dependent_read()
    test_grad_to_tables()
    test_concat_integration_and_up()
    test_tie_kv()
    test_variable_allocation()
    print("\nall L³ tests passed")
