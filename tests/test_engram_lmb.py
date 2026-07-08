"""Tests for engram_lmb.py (+ the alloc builder in engram_lmb_build.py).

Run:  python test_engram_lmb.py        (no pytest needed)
  or: python -m pytest test_engram_lmb.py -q
"""
from __future__ import annotations

import torch
import torch.nn as nn

from rwkv_lab.engram_lmb import (
    LearnedTable,
    LexicalMemoryBank,
    StreamingRecall,
    attach_engram,
    detach_engram,
    effective_depth_profile,
    engram_parameters,
    float_growth_params,
    install_input_ids_hook,
    pick_sites,
    token_rosa_recall,
)

torch.manual_seed(0)

V, C, H = 1000, 64, 4
B, T = 2, 48


# ---------------------------------------------------------------------------
# Stub model mimicking the convert_train seams:
# model.model.layers[i].linear_attn.core.value  (looped: core called n times)
# ---------------------------------------------------------------------------

class MiniTimeMix(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.value = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(torch.tanh(self.value(x)))


class MiniLooped(nn.Module):
    def __init__(self, dim: int, n_loops: int = 3) -> None:
        super().__init__()
        self.core = MiniTimeMix(dim)
        self.n_loops = n_loops
        self.gate = nn.Parameter(torch.full((n_loops,), 0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.core(x)
        for i in range(1, self.n_loops):
            out = out + self.gate[i] * self.core(x + out)
        return out


class MiniLayer(nn.Module):
    def __init__(self, dim: int, looped: bool = True) -> None:
        super().__init__()
        if looped:
            self.linear_attn = MiniLooped(dim)
        self.mlp = nn.Linear(dim, dim)

    def forward(self, hidden_states: torch.Tensor):
        h = hidden_states
        la = getattr(self, "linear_attn", None)
        if la is not None:
            h = h + la(h)
        return (h + self.mlp(h),)  # HF-style tuple output


class MiniModel(nn.Module):
    def __init__(self, dim: int, n_layers: int = 3) -> None:
        super().__init__()
        inner = nn.Module()
        inner.layers = nn.ModuleList(
            MiniLayer(dim, looped=(i != 2)) for i in range(n_layers))
        self.model = inner
        self.emb = nn.Embedding(V, dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.emb(input_ids)
        for layer in self.model.layers:
            h = layer(h)[0]
        return h


def _make_lmb(**kw) -> LexicalMemoryBank:
    defaults = dict(hidden_size=C, vocab_size=V, layer_sites=[0, 2],
                    d_row=32, kernels=(3, 5), num_heads=H, max_loops=4)
    defaults.update(kw)
    return LexicalMemoryBank(**defaults)


def _repeat_ids(b: int = B, t: int = T) -> torch.Tensor:
    """ids whose second half repeats the first -> guaranteed recalls."""
    half = torch.randint(0, V, (b, t // 2))
    return torch.cat([half, half], dim=1)


# ---------------------------------------------------------------------------
# token-level ROSA recall (Emb_ROSA)
# ---------------------------------------------------------------------------

def test_token_recall_explicit_pattern():
    # X A B C Y ... A B C -> at the second C, the longest suffix (A,B,C) matched
    # the earlier run; successor of its occurrence is Y.
    ids = torch.tensor([[9, 1, 2, 3, 7, 8, 8, 1, 2, 3]])
    rr = token_rosa_recall(ids, vocab_size=10)
    assert bool(rr.valid[0, 9]) and int(rr.recalled[0, 9]) == 7, \
        f"expected recall of Y=7, got {int(rr.recalled[0, 9])}"
    assert int(rr.mlen[0, 9]) == 3
    assert int(rr.dist[0, 9]) == 5, "tau=4 (position of Y) -> dist = 9-4"
    assert not bool(rr.valid[0, 0]), "first position can never recall"
    # early positions with no earlier repeat are invalid
    assert not bool(rr.valid[0, 1]) and not bool(rr.valid[0, 2])


def test_token_recall_matches_are_real_and_causal():
    torch.manual_seed(1)
    ids = torch.randint(0, 6, (3, 120))  # small alphabet -> many matches
    rr = token_rosa_recall(ids, vocab_size=6)
    n_checked = 0
    for b in range(3):
        seq = ids[b].tolist()
        for t in range(120):
            if not bool(rr.valid[b, t]):
                continue
            L = int(rr.mlen[b, t])
            s = t - int(rr.dist[b, t])  # tau
            assert 0 < s < t, "recall must be strictly historical"
            assert int(rr.recalled[b, t]) == seq[s]
            assert seq[s - L: s] == seq[t - L + 1: t + 1], \
                f"b={b} t={t}: recall is not a real suffix match"
            n_checked += 1
    assert n_checked > 50, "test should exercise many recalls"


def test_token_recall_agrees_with_rosa_sam_kernel():
    try:
        from rwkv_lab.rosa_sam import sam_retrieve, HAVE_NUMBA
    except Exception:
        return  # rosa_sam not importable; skip
    if not HAVE_NUMBA:
        return
    torch.manual_seed(2)
    K = 8  # alphabet small enough for rosa_sam's dense transition table
    ids = torch.randint(0, K, (2, 200))
    tau_ref = sam_retrieve(ids.unsqueeze(-1).numpy(), ids.unsqueeze(-1).numpy(), K)
    tau_ref = torch.from_numpy(tau_ref)[:, 0, :]  # [B, T]
    rr = token_rosa_recall(ids, vocab_size=K)
    assert torch.equal(rr.valid, tau_ref >= 0), "hit mask must match rosa_sam"
    ref_recalled = torch.gather(ids, 1, tau_ref.clamp_min(0)) * (tau_ref >= 0)
    assert torch.equal(rr.recalled, ref_recalled), \
        "sparse token-SAM must reproduce rosa_sam's dense kernel exactly"


def test_boundary_blocks_cross_document_recall():
    EOD = 99
    # doc1 = [1,2,3,4], doc2 repeats it: without a boundary the second doc
    # recalls from the first; with it, nothing crosses.
    ids = torch.tensor([[1, 2, 3, 4, EOD, 1, 2, 3, 4]])
    rr_open = token_rosa_recall(ids, vocab_size=100)
    rr_seg = token_rosa_recall(ids, vocab_size=100, boundary_id=EOD)
    assert bool(rr_open.valid[0, 6]), "sanity: without boundary, doc2 recalls doc1"
    assert not rr_seg.valid[0].any(), \
        "with boundary_id, no position may recall across the document break"
    assert not bool(rr_seg.valid[0, 4]), "the boundary position itself never recalls"
    # boundary equals manual per-segment recall, stitched
    torch.manual_seed(3)
    seg1 = torch.randint(0, 6, (1, 40))
    seg2 = torch.randint(0, 6, (1, 40))
    packed = torch.cat([seg1, torch.tensor([[EOD]]), seg2], dim=1)
    rr = token_rosa_recall(packed, vocab_size=100, boundary_id=EOD)
    ra = token_rosa_recall(seg1, vocab_size=100)
    rb = token_rosa_recall(seg2, vocab_size=100)
    assert torch.equal(rr.valid[0, :40], ra.valid[0])
    assert torch.equal(rr.recalled[0, :40], ra.recalled[0])
    assert torch.equal(rr.valid[0, 41:], rb.valid[0])
    assert torch.equal(rr.recalled[0, 41:], rb.recalled[0])
    assert torch.equal(rr.mlen[0, 41:], rb.mlen[0])
    assert torch.equal(rr.dist[0, 41:], rb.dist[0])


def test_streaming_matches_batch():
    torch.manual_seed(4)
    EOD = 7
    ids = torch.randint(0, 8, (1, 150))
    rr = token_rosa_recall(ids, vocab_size=8, boundary_id=EOD)
    sr = StreamingRecall(boundary_id=EOD)
    for t in range(150):
        rec, L, d = sr.extend(int(ids[0, t]))
        if bool(rr.valid[0, t]):
            assert rec == int(rr.recalled[0, t]) and L == int(rr.mlen[0, t]) \
                and d == int(rr.dist[0, t]), f"streaming diverges at t={t}"
        else:
            assert rec == -1, f"streaming false recall at t={t}"


# ---------------------------------------------------------------------------
# LearnedTable
# ---------------------------------------------------------------------------

def test_table_causality():
    table = LearnedTable(V, d_row=16, kernels=(3, 5))
    for p in table.parameters():  # move off tiny init so effects are visible
        nn.init.normal_(p, std=0.1)
    ids = torch.randint(0, V, (1, T))
    out1 = table(ids)
    ids2 = ids.clone()
    ids2[0, T // 2] = (ids2[0, T // 2] + 1) % V
    out2 = table(ids2)
    assert torch.allclose(out1[0, : T // 2], out2[0, : T // 2]), \
        "future token perturbation leaked backward"
    assert not torch.allclose(out1[0, T // 2:], out2[0, T // 2:])


def test_table_uniform_fallback_shapes():
    table = LearnedTable(V, d_row=16, kernels=(3,), table_rows=100)
    assert table.n_rows == 100
    out = table(torch.randint(0, V, (B, T)))
    assert out.shape == (B, T, 16)


def test_read_recalled_invalid_neighbor_cannot_leak():
    lmb = _make_lmb()
    for p in lmb.table.parameters():
        nn.init.normal_(p, std=0.1)
    rec = torch.tensor([[5, 3, 7, 4]])
    valid = torch.tensor([[True, False, True, True]])
    out1 = lmb.read_recalled(rec, valid)
    rec2 = rec.clone()
    rec2[0, 1] = 8  # change the INVALID position's id
    out2 = lmb.read_recalled(rec2, valid)
    assert torch.equal(out1, out2), \
        "an invalid recall id must not affect any output (incl. conv neighbors)"
    assert float(out1[0, 1].detach().abs().sum()) == 0.0


# ---------------------------------------------------------------------------
# Allocation builder
# ---------------------------------------------------------------------------

def test_freq_allocation():
    import numpy as np
    from rwkv_lab.engram_lmb_build import build_freq_allocation
    rng = np.random.RandomState(0)
    counts = rng.zipf(1.3, size=V).astype(np.int64)
    idx, w = build_freq_allocation(counts, rho=0.5, k_vip=10, n_buckets=8)
    S = int(idx.max()) + 1
    assert S <= int(0.5 * V) + 8
    vip = np.argsort(-counts)[:10]
    assert len(np.unique(idx[vip, 0])) == 10, "VIP tokens must own dedicated rows"
    assert np.all(w[vip, 0] == 1.0) and np.all(w[vip, 1:] == 0.0)
    sums = w.sum(1)
    assert np.allclose(sums[sums > 0], 1.0, atol=1e-5), "weights normalize per token"
    assert idx.min() >= 0
    # table built from this allocation round-trips
    table = LearnedTable(V, d_row=8, kernels=(3,),
                         access_idx=torch.from_numpy(idx),
                         access_w=torch.from_numpy(w))
    out = table(torch.randint(0, V, (1, 16)))
    assert out.shape == (1, 16, 8)


# ---------------------------------------------------------------------------
# No-op at init + injection behavior
# ---------------------------------------------------------------------------

def test_attach_noop_byte_exact():
    model = MiniModel(C)
    ids = _repeat_ids()
    with torch.no_grad():
        ref = model(ids).clone()
    lmb = _make_lmb()
    handles = attach_engram(model, lmb)
    ids_handle = install_input_ids_hook(model, lmb)
    with torch.no_grad():
        out = model(ids)
    assert torch.equal(ref, out), "attach must be byte-exact no-op at init"
    assert lmb.sites["0"].stats["rosa_valid_rate"] > 0.3
    # loop counter advanced once per core pass on the looped site layer
    assert lmb.sites["0"].loop_i == model.model.layers[0].linear_attn.n_loops
    # residual-only site (no linear_attn) never counts v passes
    assert lmb.sites["2"].loop_i == 0
    detach_engram(handles)
    ids_handle.remove()
    with torch.no_grad():
        out2 = model(ids)
    assert torch.equal(ref, out2)


def test_noop_without_ids():
    model = MiniModel(C)
    ids = _repeat_ids()
    with torch.no_grad():
        ref = model(ids).clone()
    lmb = _make_lmb()
    attach_engram(model, lmb)  # no ids hook, no set_input_ids
    with torch.no_grad():
        out = model(ids)
    assert torch.equal(ref, out)


def test_injection_changes_output_and_grads_flow():
    model = MiniModel(C)
    ids = _repeat_ids()
    lmb = _make_lmb()
    attach_engram(model, lmb)
    install_input_ids_hook(model, lmb)
    with torch.no_grad():
        ref = model(ids).clone()  # projections still zero -> no-op baseline
    for site in lmb.sites.values():  # open the zero-init output paths
        nn.init.normal_(site.v_c.weight, std=0.05)
        nn.init.normal_(site.h_c.weight, std=0.05)
    lmb.set_warmup(1.0)
    out = model(ids)
    assert not torch.allclose(ref, out), "opened projections must change output"
    out.square().mean().backward()
    g_table = lmb.table.tables[0].grad
    assert g_table is not None and float(g_table.abs().sum()) > 0, \
        "gradient must reach learned table rows through the recall read"
    site = lmb.sites["0"]
    assert site.loop_scale.grad is not None and \
        float(site.loop_scale.grad.abs().sum()) > 0, \
        "loop-index scale must receive gradient through v-stream injections"
    assert site.len_scale_vc.grad is not None and \
        float(site.len_scale_vc.grad.abs().sum()) > 0, \
        "match-length gate modulation must train"
    assert site.dist_scale_vc.grad is not None and \
        float(site.dist_scale_vc.grad.abs().sum()) > 0, \
        "recall-distance gate modulation must train"


def test_injection_localized_to_valid_recalls():
    lmb = _make_lmb(layer_sites=[0])
    ids = torch.randint(20, V, (1, T))
    ids[0, 10:13] = torch.tensor([1, 2, 3])
    ids[0, 30:33] = torch.tensor([1, 2, 3])  # recall valid inside/after 2nd run
    lmb.set_input_ids(ids)
    site = lmb.sites["0"]
    nn.init.normal_(site.v_c.weight, std=0.5)
    assert lmb.ensure_features()
    with torch.no_grad():
        inj = site.inj_v(torch.randn(1, T, C))
    rr = token_rosa_recall(ids, V)
    assert bool(rr.valid[0, 31]) and float(inj[0, 31].abs().sum()) > 0, \
        "valid recall position must inject"
    assert not bool(rr.valid[0, 5]) and float(inj[0, 5].abs().sum()) == 0, \
        "positions with no recall must stay silent"


def test_logit_bias_copy_head():
    lmb = _make_lmb(layer_sites=[0])
    ids = _repeat_ids(1, T)
    lmb.set_input_ids(ids)
    assert lmb.ensure_features()
    logits = torch.randn(1, T, V)
    out = lmb.logit_bias(logits)
    assert torch.equal(out, logits), "copy head must be exact no-op at init"
    with torch.no_grad():
        lmb.logit_scale.fill_(2.0)
    out = lmb.logit_bias(logits)
    rr = lmb.last_recall
    changed = (out != logits).any(-1)
    assert torch.equal(changed, rr.valid), \
        "bonus lands exactly at positions with a valid recall"
    t = int(rr.valid[0].nonzero()[0])
    tok = int(rr.recalled[0, t])
    diff = out[0, t] - logits[0, t]
    assert float(diff[tok]) > 0, "recalled token's logit must increase"
    assert float(diff.abs().sum() - diff[tok].abs()) == 0, \
        "only the recalled token's logit changes"
    # gradient reaches the copy-head params
    lmb.logit_scale.grad = None
    lmb.logit_bias(logits).square().mean().backward()
    assert float(lmb.logit_scale.grad.abs().sum()) > 0
    assert float(lmb.logit_feat.grad.abs().sum()) > 0
    # disabled context -> pass-through
    lmb.ctx.enabled = False
    assert torch.equal(lmb.logit_bias(logits), logits)
    lmb.ctx.enabled = True


def test_recall_telemetry():
    lmb = _make_lmb(layer_sites=[0])
    half = torch.randint(0, V, (1, 40))
    ids = torch.cat([half, half], dim=1)  # recalls at distance 40 (> 32)
    lmb.set_input_ids(ids)
    assert lmb.ensure_features()
    tel = lmb.telemetry()
    rs = tel["recall"]
    assert rs["valid_rate"] > 0.3
    assert rs["frac_beyond_32"] > 0.9, \
        "distance-40 recalls must register as beyond ROSA-soft's window"
    assert rs["mlen_p50"] >= 1 and rs["dist_p50"] > 32
    assert "rosa_valid_rate" in tel["0"]


def test_warmup_zero_silences_injection():
    model = MiniModel(C)
    ids = _repeat_ids()
    with torch.no_grad():
        ref = model(ids).clone()
    lmb = _make_lmb()
    attach_engram(model, lmb)
    install_input_ids_hook(model, lmb)
    for site in lmb.sites.values():
        nn.init.normal_(site.v_c.weight, std=0.05)
        nn.init.normal_(site.h_c.weight, std=0.05)
    lmb.set_warmup(0.0)
    with torch.no_grad():
        out = model(ids)
    assert torch.equal(ref, out)


def test_ctx_disable_for_isolation_stages():
    model = MiniModel(C)
    ids = _repeat_ids()
    with torch.no_grad():
        ref = model(ids).clone()
    lmb = _make_lmb()
    attach_engram(model, lmb)
    install_input_ids_hook(model, lmb)
    for site in lmb.sites.values():
        nn.init.normal_(site.v_c.weight, std=0.05)
    lmb.ctx.enabled = False  # SMT/DMT per-layer stages
    with torch.no_grad():
        out = model(ids)
    assert torch.equal(ref, out), "ctx.enabled=False must silence injection"


# ---------------------------------------------------------------------------
# dtype hygiene, placement, checkpointing
# ---------------------------------------------------------------------------

def test_float_growth_params_bf16_safe():
    lmb = _make_lmb().to(torch.bfloat16)
    float_growth_params(lmb)
    site = lmb.sites["0"]
    assert site.loop_scale.dtype == torch.float32
    assert site.out_scale_v.dtype == torch.float32
    assert site.len_scale_vc.dtype == torch.float32
    assert lmb.table.row_scale[0].dtype == torch.float32
    for view in lmb.table.views:  # convs must stay uniform dtype
        assert view.conv_g.bias.dtype == view.conv_g.weight.dtype
    lmb.set_input_ids(_repeat_ids(1, 16))
    assert lmb.ensure_features()  # forward path survives mixed precision


def test_effective_depth_and_pick_sites():
    n_layers = 32
    gates = {i: torch.tensor([0.0, 0.5, 0.5, 0.5]) for i in range(4, 14)}
    d_eff = effective_depth_profile(gates, n_layers)
    assert d_eff.shape == (n_layers,)
    assert torch.all(d_eff[1:] >= d_eff[:-1])
    assert abs(float(d_eff[-1]) - (32 + 10 * 1.5)) < 1e-5
    sites = pick_sites(d_eff)
    assert len(sites) == len(set(sites)) == 3
    flat = pick_sites(torch.cumsum(torch.ones(n_layers), 0))
    # loop mass concentrated at L4-13 pulls the later sites EARLIER
    assert sites[1] <= flat[1] and sites[2] <= flat[2]


def test_state_dict_roundtrip():
    lmb = _make_lmb()
    sd = lmb.state_dict()
    assert any(k.startswith("table.tables") for k in sd)
    assert any("loop_scale" in k for k in sd)
    lmb2 = _make_lmb()
    lmb2.load_state_dict(sd)
    assert len(engram_parameters(lmb)) > 0


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
