"""Tests for lookahead_module.py (NextLat + TOP auxiliary objectives).

Run:  python test_lookahead.py        (no pytest needed)
  or: python -m pytest test_lookahead.py -q
"""
from __future__ import annotations

import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rwkv_lab.lookahead_module import (
    LookaheadSystem,
    NextLatPredictor,
    TOPHead,
    add_lookahead_cli,
    frozen_head_kl,
    lookahead_from_args,
    nextlat_loss,
    top_loss,
    top_targets,
)

torch.manual_seed(0)

V, D = 50, 16
B, T, W = 2, 24, 4


def _args(**over):
    ap = argparse.ArgumentParser()
    add_lookahead_cli(ap)
    ns = ap.parse_args([])
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


# ---------------------------------------------------------------------------
# TOP
# ---------------------------------------------------------------------------

def test_top_targets_hand_computed():
    ids = torch.tensor([7, 3, 5, 3, 9, 1, 2, 4])
    y = top_targets(ids, t0=0, t1=1, window=4, vocab=V)  # position 0, window ids[1..4]=3,5,3,9
    assert y.shape == (1, V)
    assert y[0, 3] == 3.0      # nearest occurrence at distance 1 -> W-1
    assert y[0, 5] == 2.0      # distance 2
    assert y[0, 9] == 0.0      # distance 4 -> W-4
    assert torch.isinf(y[0, 7]) and y[0, 7] < 0   # not in window (only at t=0 itself)
    finite = torch.isfinite(y[0])
    assert finite.sum() == 3


def test_top_targets_nearest_occurrence_wins():
    # token 3 appears at distances 1 and 3; the nearer (higher score) must win
    ids = torch.tensor([0, 3, 8, 3, 8, 8])
    y = top_targets(ids, 0, 1, window=4, vocab=V)
    assert y[0, 3] == 3.0      # W - 1, not W - 3


def test_top_loss_matches_dense_reference():
    torch.manual_seed(1)
    head = TOPHead(D, V)
    h = torch.randn(B, T, D)
    ids = torch.randint(0, V, (B, T + W))
    got = top_loss(head, h, ids, W, chunk=5)   # odd chunk to exercise boundaries
    # naive dense reference
    ref = 0.0
    for b in range(B):
        logits = head(h[b]).float()
        y = torch.full((T, V), float("-inf"))
        for t in range(T):
            for j in range(W, 0, -1):
                y[t, ids[b, t + j]] = float(W - j)
        p = torch.softmax(y, -1)
        ref = ref + -(p * F.log_softmax(logits, -1)).sum(-1).sum()
    ref = ref / (B * T)
    assert torch.allclose(got, ref, rtol=1e-5, atol=1e-6), (float(got), float(ref))


def test_top_loss_chunk_invariance():
    torch.manual_seed(2)
    head = TOPHead(D, V)
    h = torch.randn(B, T, D)
    ids = torch.randint(0, V, (B, T + W))
    a = top_loss(head, h, ids, W, chunk=3)
    b = top_loss(head, h, ids, W, chunk=1024)
    assert torch.allclose(a, b, rtol=1e-5, atol=1e-6)


def test_top_loss_grads_flow_to_hidden_and_head():
    head = TOPHead(D, V)
    h = torch.randn(B, T, D, requires_grad=True)
    ids = torch.randint(0, V, (B, T + W))
    top_loss(head, h, ids, W).backward()
    assert h.grad is not None and h.grad.abs().sum() > 0
    assert head.proj.weight.grad is not None and head.proj.weight.grad.abs().sum() > 0


def test_top_head_lmhead_init_and_rank():
    lm = nn.Linear(D, V, bias=False)
    head = TOPHead.from_lm_head(lm)
    assert torch.equal(head.proj.weight, lm.weight)
    assert head.proj.weight.data_ptr() != lm.weight.data_ptr()   # a clone, not a tie
    lr = TOPHead(D, V, rank=4)
    assert lr(torch.randn(3, D)).shape == (3, V)
    n_full = sum(p.numel() for p in head.parameters())
    n_rank = sum(p.numel() for p in lr.parameters())
    assert n_rank < n_full


def test_top_loss_rejects_short_ids():
    head = TOPHead(D, V)
    try:
        top_loss(head, torch.randn(1, T, D), torch.randint(0, V, (1, T)), W)
    except ValueError:
        return
    raise AssertionError("expected ValueError for ids shorter than T+window")


# ---------------------------------------------------------------------------
# NextLat
# ---------------------------------------------------------------------------

def test_nextlat_identity_at_init():
    pred = NextLatPredictor(D)
    h = torch.randn(B, T, D)
    act = torch.randn(B, T, D)
    out = pred(h, act)
    assert torch.equal(out, h), "zero-init output layer must make p_psi the identity transition"


def test_nextlat_targets_are_stop_grad():
    # at d=1 the last position only ever serves as a TARGET -> zero grad there
    pred = NextLatPredictor(D)
    with torch.no_grad():   # break identity so the loss is non-trivial
        pred.net[-1].weight.normal_(0, 0.1)
    h = torch.randn(B, T, D, requires_grad=True)
    act = torch.randn(B, T, D)
    lm = nn.Linear(D, V, bias=False)
    l_h, l_kl = nextlat_loss(pred, h, act, lm.weight, d=1, kl_weight=1.0)
    (l_h + l_kl).backward()
    assert h.grad is not None
    assert h.grad[:, -1].abs().sum() == 0, "targets must be stop-grad"
    assert h.grad[:, :-1].abs().sum() > 0, "prediction path must shape the backbone"


def test_nextlat_kl_leaves_head_frozen():
    pred = NextLatPredictor(D)
    with torch.no_grad():
        pred.net[-1].weight.normal_(0, 0.1)
    h = torch.randn(B, T, D)
    lm = nn.Linear(D, V, bias=False)
    kl = frozen_head_kl(pred(h[:, :-1], torch.randn(B, T - 1, D)), h[:, 1:], lm.weight)
    kl.backward()
    assert lm.weight.grad is None, "lm_head must receive no gradient from the KL term"
    assert kl >= 0


def test_nextlat_kl_zero_for_equal_states():
    lm = nn.Linear(D, V, bias=False)
    h = torch.randn(B, T, D)
    kl = frozen_head_kl(h, h, lm.weight)
    assert abs(float(kl)) < 1e-5


def test_nextlat_rollout_d2():
    pred = NextLatPredictor(D)
    h = torch.randn(B, T, D)
    act = torch.randn(B, T, D)
    lm = nn.Linear(D, V, bias=False)
    l_h, l_kl = nextlat_loss(pred, h, act, lm.weight, d=2, kl_weight=1.0)
    assert math.isfinite(float(l_h)) and math.isfinite(float(l_kl))
    # identity-at-init: step i loss is SmoothL1(h_t, h_{t+i}) averaged; must be > 0
    assert float(l_h) > 0
    try:
        nextlat_loss(pred, h[:, :2], act[:, :2], lm.weight, d=2)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when d >= T")


def test_nextlat_rollout_alignment_exact():
    """Stub predictor returns the action embedding; with h[t] == emb[t] == t,
    every rollout step is exact iff cur/act/target indices line up, so the
    SmoothL1 loss is 0 for all d. Any off-by-one makes it > 0."""

    class Echo(nn.Module):
        def forward(self, h, act):
            return act

    ramp = torch.arange(T, dtype=torch.float32).view(1, T, 1).expand(1, T, D).contiguous()
    for d in (1, 2, 3):
        l_h, _ = nextlat_loss(Echo(), ramp, ramp, None, d=d, kl_weight=0.0)
        assert float(l_h) == 0.0, f"misaligned rollout at d={d}: loss={float(l_h)}"
    # sanity: a shifted target makes it non-zero (the test can actually fail)
    l_bad, _ = nextlat_loss(Echo(), ramp + 1.0, ramp, None, d=1, kl_weight=0.0)
    assert float(l_bad) > 0


def test_nextlat_jump_alignment_exact():
    """Echo predictor returns the pooled action; craft emb so the window mean equals
    the target exactly: emb[j] = j + (k-1)/2 makes mean(emb[t+1..t+k]) = t+k = h[t+k].
    Loss is 0 iff every index in the jump path lines up."""
    from rwkv_lab.lookahead_module import nextlat_jump_loss

    class Echo(nn.Module):
        def forward(self, h, act):
            return act

    k = 2
    ramp = torch.arange(T, dtype=torch.float32).view(1, T, 1).expand(1, T, D).contiguous()
    emb = ramp + (k - 1) / 2.0
    l_h, _ = nextlat_jump_loss(Echo(), ramp, emb, k, None, kl_weight=0.0)
    assert float(l_h) == 0.0, f"misaligned jump: loss={float(l_h)}"
    l_bad, _ = nextlat_jump_loss(Echo(), ramp + 1.0, emb, k, None, kl_weight=0.0)
    assert float(l_bad) > 0


def test_nextlat_jump_guards_and_system():
    from rwkv_lab.lookahead_module import nextlat_jump_loss
    pred = NextLatPredictor(D)
    h = torch.randn(B, T, D)
    for bad_k in (0, 1, T):
        try:
            nextlat_jump_loss(pred, h, torch.randn(B, T, D), bad_k, None)
        except ValueError:
            continue
        raise AssertionError(f"jump k={bad_k} must be rejected")
    try:
        lookahead_from_args(_args(nextlat_jump_weight=1.0, nextlat_jump_k=1), D, V, None)
    except ValueError:
        pass
    else:
        raise AssertionError("jump_k=1 must be rejected at system level")
    lm = nn.Linear(D, V, bias=False)
    emb = nn.Embedding(V, D)
    sys_ = lookahead_from_args(_args(nextlat_jump_weight=1.0, nextlat_jump_k=4), D, V, lm)
    assert sys_ is not None and sys_.nextlat_jump is not None and sys_.nextlat is None
    ids = torch.randint(0, V, (B, T))
    out = sys_.compute(torch.randn(B, T, D), ids, emb, lm)
    assert "nextlat_jump_h" in out and math.isfinite(float(out["aux_total"]))
    assert float(out["aux_total"]) > 0


def test_nextlat_kl_weight_zero_skips_head():
    pred = NextLatPredictor(D)
    h = torch.randn(B, T, D)
    l_h, l_kl = nextlat_loss(pred, h, torch.randn(B, T, D), None, d=1, kl_weight=0.0)
    assert float(l_kl) == 0.0 and math.isfinite(float(l_h))


# ---------------------------------------------------------------------------
# System / CLI
# ---------------------------------------------------------------------------

def test_from_args_disabled_is_none():
    assert lookahead_from_args(_args(), D, V, None) is None


def test_invalid_hyperparams_rejected():
    for bad in ({"nextlat_weight": 1.0, "nextlat_d": 0},
                {"top_weight": 1.0, "top_window": 0},
                {"top_weight": 1.0, "top_chunk": 0}):
        try:
            lookahead_from_args(_args(**bad), D, V, None)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")


def test_system_components_and_extra_tokens():
    lm = nn.Linear(D, V, bias=False)
    both = lookahead_from_args(_args(nextlat_weight=1.0, top_weight=0.5), D, V, lm)
    assert both.enabled and both.nextlat is not None and both.top is not None
    assert both.extra_tokens == both.top_window
    nl_only = lookahead_from_args(_args(nextlat_weight=1.0), D, V, lm)
    assert nl_only.top is None and nl_only.extra_tokens == 0


def test_system_compute_and_weighting():
    torch.manual_seed(3)
    lm = nn.Linear(D, V, bias=False)
    emb = nn.Embedding(V, D)
    sys_ = LookaheadSystem(D, V, nextlat_weight=0.7, nextlat_kl_weight=0.5,
                           top_weight=0.3, top_window=W, lm_head=lm)
    with torch.no_grad():
        sys_.nextlat.net[-1].weight.normal_(0, 0.1)
    h = torch.randn(B, T, D)
    ids = torch.randint(0, V, (B, T + W))
    out = sys_.compute(h, ids, emb, lm)
    expect = 0.7 * out["nextlat_h"] + 0.7 * 0.5 * out["nextlat_kl"] + 0.3 * out["top"]
    assert abs(float(out["aux_total"]) - expect) < 1e-4 * max(1.0, abs(expect))
    assert float(out["aux_total"]) > 0


def test_concept_head_targets_and_stopgrad():
    from rwkv_lab.lookahead_module import ConceptHead
    torch.manual_seed(6)
    head = ConceptHead(D, chunk=4, segments=4, codes=8)
    h = torch.randn(B, T, D, requires_grad=True)
    l_ncp, l_vq, frac = head.loss(h)
    assert 0.0 < frac <= 1.0
    (l_ncp + l_vq).backward()
    assert h.grad is not None
    # positions in [T-chunk, T) are target-only (predictor covers [0, T-chunk)) and
    # targets are stop-grad -> zero gradient there; predictor path must have gradient
    assert h.grad[:, T - 4:].abs().sum() == 0, "concept targets must be stop-grad"
    assert h.grad[:, : T - 4].abs().sum() > 0
    assert head.codebook.grad is not None and head.codebook.grad.abs().sum() > 0
    assert head.predictor[1].weight.grad.abs().sum() > 0


def test_concept_pooled_target_alignment():
    """With h[t] = t, the pooled concept target at t is mean(t+1..t+k) = t+(k+1)/2.
    Craft a codebook-free check by comparing against the cumsum formula directly."""
    from rwkv_lab.lookahead_module import ConceptHead
    k = 4
    head = ConceptHead(D, chunk=k, segments=4, codes=8)
    ramp = torch.arange(T, dtype=torch.float32).view(1, T, 1).expand(1, T, D)
    cs = ramp.float().cumsum(1)
    c = (cs[:, k:] - cs[:, :-k]) / k
    expect = torch.arange(T - k, dtype=torch.float32) + (k + 1) / 2.0
    assert torch.allclose(c[0, :, 0], expect, atol=1e-4), "pooled window must be x[t+1..t+k]"
    l_ncp, l_vq, _ = head.loss(ramp.contiguous())
    assert math.isfinite(float(l_ncp)) and math.isfinite(float(l_vq))


def test_concept_head_bf16():
    """Trainers cast the lookahead system to model dtype (bf16); the concept head
    must survive that (codex tier-3 #1: fp32 codebook into bf16 cb_mlp crashed)."""
    from rwkv_lab.lookahead_module import ConceptHead
    head = ConceptHead(D, chunk=4, segments=4, codes=8).to(dtype=torch.bfloat16)
    h = torch.randn(B, T, D, dtype=torch.bfloat16)
    l_ncp, l_vq, frac = head.loss(h)
    assert math.isfinite(float(l_ncp)) and math.isfinite(float(l_vq)) and 0 < frac <= 1
    (l_ncp + l_vq).backward()


def test_concept_guards_and_system_integration():
    from rwkv_lab.lookahead_module import ConceptHead
    for bad in (dict(segments=5), dict(chunk=1)):
        try:
            ConceptHead(D, **bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")
    lm = nn.Linear(D, V, bias=False)
    emb = nn.Embedding(V, D)
    sys_ = lookahead_from_args(_args(concept_weight=1.0, concept_segments=4,
                                     concept_codes=8), D, V, lm)
    assert sys_ is not None and sys_.concept is not None and sys_.nextlat is None
    out = sys_.compute(torch.randn(B, T, D), torch.randint(0, V, (B, T)), emb, lm)
    assert "concept_ncp" in out and "concept_vq" in out and "concept_codes" in out
    assert math.isfinite(float(out["aux_total"])) and float(out["aux_total"]) > 0


def test_smoke_train_step_shapes_backbone():
    """One AdamW step on a mini backbone: aux loss must move BACKBONE params."""
    torch.manual_seed(4)

    class Mini(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(V, D)
            self.mix = nn.Linear(D, D)
            self.lm_head = nn.Linear(D, V, bias=False)

        def hidden(self, ids):
            return torch.tanh(self.mix(self.emb(ids)))

    m = Mini()
    sys_ = LookaheadSystem(D, V, nextlat_weight=1.0, top_weight=0.5,
                           top_window=W, lm_head=m.lm_head)
    opt = torch.optim.AdamW(list(m.mix.parameters()) + list(sys_.parameters()), lr=1e-2)
    ids = torch.randint(0, V, (B, T + W))
    before = m.mix.weight.detach().clone()
    losses = []
    for _ in range(3):
        out = sys_.compute(m.hidden(ids[:, :T]), ids, m.emb, m.lm_head)
        opt.zero_grad()
        out["aux_total"].backward()
        opt.step()
        losses.append(float(out["aux_total"]))
    assert not torch.equal(before, m.mix.weight), "backbone must be shaped by the aux loss"
    assert all(math.isfinite(x) for x in losses)


def test_state_dict_roundtrip():
    lm = nn.Linear(D, V, bias=False)
    a = LookaheadSystem(D, V, nextlat_weight=1.0, top_weight=0.5, top_window=W, lm_head=lm)
    b = LookaheadSystem(D, V, nextlat_weight=1.0, top_weight=0.5, top_window=W, lm_head=None,
                        top_init="random")
    b.load_state_dict(a.state_dict())
    for (ka, pa), (kb, pb) in zip(sorted(a.state_dict().items()), sorted(b.state_dict().items())):
        assert ka == kb and torch.equal(pa, pb)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
