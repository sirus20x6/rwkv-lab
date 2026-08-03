"""Tests for engram_lmb.py (+ the alloc builder in engram_lmb_build.py).

Run:  python test_engram_lmb.py        (no pytest needed)
  or: python -m pytest test_engram_lmb.py -q
"""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from rwkv_lab.engram_lmb import (
    BatchedStreamingEngramState,
    LearnedTable,
    LexicalMemoryBank,
    RecallResult,
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

    def forward(self, input_ids: torch.Tensor | None = None, *,
                inputs_embeds: torch.Tensor | None = None) -> torch.Tensor:
        if inputs_embeds is None:
            assert input_ids is not None
            h = self.emb(input_ids)
        else:
            h = inputs_embeds
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


def test_eval_hooks_keep_gate_telemetry_device_resident():
    model = MiniModel(C)
    lmb = _make_lmb(layer_sites=[0])
    handles = attach_engram(model, lmb)
    ids_handle = install_input_ids_hook(model, lmb)
    lmb.eval()
    with torch.no_grad():
        model(_repeat_ids())
    site = lmb.sites["0"]
    assert torch.is_tensor(site.stats["gate_vc_mean"])
    assert torch.is_tensor(site.stats["gate_hc_mean"])
    telemetry = lmb.telemetry()
    assert isinstance(telemetry["0"]["gate_vc_mean"], float)
    detach_engram(handles)
    ids_handle.remove()


def test_attach_fla_v_proj_seam():
    class FLATimeMix(nn.Module):
        def __init__(self):
            super().__init__()
            self.v_proj = nn.Linear(C, C, bias=False)
            self.o_proj = nn.Linear(C, C, bias=False)

        def forward(self, x):
            return self.o_proj(torch.tanh(self.v_proj(x)))

    class FLALayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.att = FLATimeMix()

        def forward(self, hidden_states):
            return hidden_states + self.att(hidden_states)

    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([FLALayer()])
    ids = _repeat_ids()
    hidden = torch.randn(B, T, C)
    lmb = _make_lmb(layer_sites=[0])
    attach_engram(model, lmb)
    lmb.set_input_ids(ids)
    with torch.no_grad():
        ref = model.model.layers[0](hidden)
        for table in lmb.table.tables:
            table.fill_(0.25)
        lmb.sites["0"].v_c.weight.fill_(0.1)
        lmb.set_input_ids(ids)
        out = model.model.layers[0](hidden)
    assert not torch.equal(ref, out), "FLA v_proj must receive Engram injection"


def test_attach_real_fla_attn_attribute_v_proj_seam():
    """FLA RWKV7DecoderLayer exposes time-mix as `.attn`, not `.linear_attn`."""
    C, V, T = 16, 64, 12

    class AutocastProjection(nn.Module):
        def forward(self, x):
            return x.to(torch.bfloat16)

    class FLAAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.v_proj = AutocastProjection()
            self.seen_v_dtype = None

        def forward(self, x):
            v = self.v_proj(x)
            self.seen_v_dtype = v.dtype
            return v.float()

    class FLALayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = FLAAttention()

        def forward(self, x):
            return x + self.attn(x)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([FLALayer()])

        def forward(self, x):
            return self.model.layers[0](x)

    model = Model()
    lmb = LexicalMemoryBank(hidden_size=C, vocab_size=V, layer_sites=[0],
                            d_row=8, table_rows=V, num_heads=4, max_loops=1)
    attach_engram(model, lmb)
    ids = torch.arange(T)[None] % 4
    lmb.set_input_ids(ids)
    x = torch.randn(1, T, C)
    ref = model(x).detach().clone()
    with torch.no_grad():
        for table in lmb.table.tables:
            table.fill_(0.25)
        lmb.sites["0"].v_c.weight.fill_(0.01)
    lmb.set_input_ids(ids)
    out = model(x)
    assert not torch.equal(ref, out), "real FLA `.attn.v_proj` must receive Engram injection"
    assert model.model.layers[0].attn.seen_v_dtype == torch.bfloat16
    assert lmb.sites["0"].loop_i == 1
    assert lmb.sites["0"].last_inj_v_rms is not None


def test_prefetched_recall_matches_inline_path():
    model = MiniModel(C)
    ids = _repeat_ids()
    lmb = _make_lmb()
    attach_engram(model, lmb)
    install_input_ids_hook(model, lmb)
    for site in lmb.sites.values():
        nn.init.normal_(site.v_c.weight, std=0.05)
        nn.init.normal_(site.h_c.weight, std=0.05)
    with torch.no_grad():
        inline = model(ids)
        rr = token_rosa_recall(ids.cpu(), V)
        prefetched = model(ids, precomputed_recall=rr)
    torch.testing.assert_close(prefetched, inline, rtol=0, atol=0)


def test_full_context_active_slice_matches_last_full_engram_features():
    """One-token decode keeps exact recall and causal ShortConv context."""
    lmb = _make_lmb(layer_sites=[0])
    ids = _repeat_ids()
    lmb.set_input_ids(ids)
    assert lmb.ensure_features()
    full_recall = RecallResult(*(value.clone() for value in lmb.last_recall))
    full_site = lmb.sites["0"]
    full_k = full_site._k_r[:, -1:].clone()
    full_v = full_site._v_r[:, -1:].clone()
    full_h = full_site._h_r[:, -1:].clone()

    lmb.set_input_ids(ids, active_slice=slice(-1, None))
    assert lmb.ensure_features()
    assert lmb.last_recall.valid.shape == (ids.shape[0], 1)
    for actual, expected in zip(lmb.last_recall, full_recall):
        torch.testing.assert_close(actual, expected[:, -1:], rtol=0, atol=0)
    # CPU grouped-convolution/linear kernels may differ by a few ULP between
    # repeated calls; the sliced and full paths use the same learned features.
    torch.testing.assert_close(full_site._k_r, full_k, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(full_site._v_r, full_v, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(full_site._h_r, full_h, rtol=1e-5, atol=1e-6)


def test_streaming_engram_matches_full_recall_and_shortconv_features():
    """Rolling recall/table windows equal full-history evaluation at every step."""
    boundary = 7
    lmb = _make_lmb(layer_sites=[0], boundary_id=boundary)
    with torch.no_grad():
        for parameter in lmb.parameters():
            parameter.normal_(std=0.05)
    ids = torch.tensor([[
        9, 4, 9, 4, boundary, 3, 8, 3, 8, 6, 3, 8, 6, 5, 3, 8, 6, 5,
    ]])
    split = 10
    full_recall = token_rosa_recall(ids, V, boundary)
    with torch.no_grad():
        full_features = lmb.read_recalled(full_recall.recalled, full_recall.valid)

        prefill_ids = ids[:, 4:split]  # begin at the last document boundary
        prefill_recall = RecallResult(*(
            value[:, 4:split] for value in full_recall))
        lmb.set_input_ids(prefill_ids, recall=prefill_recall)
        assert lmb.ensure_features()
        stream = lmb.begin_streaming(prefill_ids, recall=prefill_recall)

        for position in range(split, ids.shape[1]):
            actual_recall = stream.step(int(ids[0, position]))
            for actual, expected in zip(actual_recall, full_recall):
                torch.testing.assert_close(
                    actual, expected[:, position:position + 1], rtol=0, atol=0)
            assert stream.last_feature is not None
            torch.testing.assert_close(
                stream.last_feature, full_features[:, position:position + 1],
                rtol=1e-5, atol=1e-6)
            # Layer hooks must consume the prepared rolling feature rather than
            # falling back to a full-history token_rosa_recall call.
            assert lmb.ensure_features()
            assert lmb.last_recall is actual_recall


def test_streaming_engram_requires_single_matching_history():
    lmb = _make_lmb(layer_sites=[0])
    ids = _repeat_ids()[:1]
    recall = token_rosa_recall(ids, V)
    with pytest.raises(ValueError, match="one.*history"):
        lmb.begin_streaming(ids.expand(2, -1), recall=RecallResult(*(
            value.expand(2, -1) for value in recall)))
    with pytest.raises(ValueError, match="must match"):
        lmb.begin_streaming(ids[:, :-1], recall=recall)


def test_batched_streaming_engram_matches_unequal_scalar_streams():
    """Batched rows preserve scalar recall/features, boundaries, and pauses."""
    boundary = 7
    scalar_lmb = _make_lmb(layer_sites=[0], boundary_id=boundary)
    batched_lmb = _make_lmb(layer_sites=[0], boundary_id=boundary)
    with torch.no_grad():
        for parameter in scalar_lmb.parameters():
            parameter.normal_(std=0.05)
    batched_lmb.load_state_dict(scalar_lmb.state_dict())

    # Row 2 is intentionally shorter than the largest ShortConv history. The
    # batched state must left-pad it without changing causal semantics.
    prefills = [
        torch.tensor([[boundary, 1, 2, 1]]),
        torch.tensor([[boundary, 4, 5, 4, 5, 6, 4]]),
        torch.tensor([[boundary, 8]]),
    ]
    scalar_streams, batch_rows = [], []
    for ids in prefills:
        recall = token_rosa_recall(ids, V, boundary)
        scalar_streams.append(scalar_lmb.begin_streaming(ids, recall=recall))
        batch_rows.append(batched_lmb.begin_streaming(
            ids, recall=RecallResult(*(value.clone() for value in recall))))
    stream = BatchedStreamingEngramState(batch_rows)
    assert stream.batch_size == 3
    assert [history.shape[:2] for history in stream.histories] == [(3, 2), (3, 4)]

    tokens = [
        [2, 5, 9],
        [3, 6, 999],       # row 2 is paused; its token must be ignored
        [1, boundary, 8],  # an active boundary resets only row 1 recall
        [2, 4, 9],
        [3, 5, 0],
        [boundary, 4, 8],  # independently reset row 0 later
        [1, 5, 9],
    ]
    active = [
        [True, True, True],
        [True, True, False],
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [True, True, True],
    ]
    saw_valid = False
    with torch.no_grad():
        for iteration, (step_tokens, step_active) in enumerate(zip(tokens, active)):
            paused_histories = ([history[2].clone() for history in stream.histories]
                                if not step_active[2] else None)
            paused_time = (stream.recallers[2]._t if not step_active[2] else None)

            expected_recall, expected_features = [], []
            for row, (scalar, token, enabled) in enumerate(zip(
                    scalar_streams, step_tokens, step_active)):
                if enabled:
                    rr = scalar.step(token)
                    expected_recall.append(rr)
                    expected_features.append(scalar.last_feature.clone())
                    saw_valid = saw_valid or bool(rr.valid.any())
                else:
                    expected_recall.append(RecallResult(
                        recalled=torch.zeros((1, 1), dtype=torch.long),
                        valid=torch.zeros((1, 1), dtype=torch.bool),
                        mlen=torch.zeros((1, 1), dtype=torch.long),
                        dist=torch.zeros((1, 1), dtype=torch.long)))
                    expected_features.append(torch.zeros(
                        (1, 1, scalar_lmb.table.d_row),
                        dtype=scalar_lmb.table.tables[0].dtype))

            actual = stream.step(
                torch.tensor(step_tokens), active=torch.tensor(step_active))
            expected_batched = RecallResult(*(
                torch.cat([rr[field] for rr in expected_recall], dim=0)
                for field in range(4)))
            for got, want in zip(actual, expected_batched):
                torch.testing.assert_close(got.cpu(), want.cpu(), rtol=0, atol=0)
            torch.testing.assert_close(
                stream.last_feature.cpu(), torch.cat(expected_features).cpu(),
                rtol=1e-5, atol=1e-6)

            if paused_histories is not None:
                assert stream.recallers[2]._t == paused_time
                for history, previous in zip(stream.histories, paused_histories):
                    torch.testing.assert_close(history[2], previous, rtol=0, atol=0)
                assert not bool(actual.valid[2, 0])
                assert torch.count_nonzero(stream.last_feature[2]) == 0

            # Shared model hooks see one feature per row and must not rebuild
            # recall from the clamped recalled-token IDs.
            assert batched_lmb.last_recall is actual
            assert batched_lmb.sites["0"]._shape == (3, 1)
            assert batched_lmb.ensure_features()
            assert batched_lmb.last_recall is actual
    assert saw_valid, "parity exercise must include nonzero recalled features"


def test_batched_streaming_engram_validates_rows_and_step_width():
    lmb = _make_lmb(layer_sites=[0])
    ids = torch.tensor([[1, 2, 1]])
    recall = token_rosa_recall(ids, V)
    row = lmb.begin_streaming(ids, recall=recall)
    with pytest.raises(ValueError, match="at least one"):
        BatchedStreamingEngramState([])
    other_lmb = _make_lmb(layer_sites=[0])
    other = other_lmb.begin_streaming(ids, recall=recall)
    with pytest.raises(ValueError, match="share one"):
        BatchedStreamingEngramState([row, other])

    stream = BatchedStreamingEngramState([row])
    with pytest.raises(ValueError, match="one token per row"):
        stream.step([1, 2])
    with pytest.raises(ValueError, match="one active flag per row"):
        stream.step([1], active=[True, False])


def _make_stream(lmb, prompts):
    rows = []
    for ids in prompts:
        recall = token_rosa_recall(ids, V)
        rows.append(lmb.begin_streaming(ids, recall=recall))
    return BatchedStreamingEngramState(rows)


def test_compacting_a_decode_batch_equals_decoding_those_rows_alone():
    """Dropping a finished row must not disturb the rows that remain."""
    lmb = _make_lmb(layer_sites=[0])
    prompts = [
        torch.tensor([[1, 2, 1]]),
        torch.tensor([[3, 4, 3, 4]]),
        torch.tensor([[5, 6, 5, 6, 5]]),
    ]
    warm = [2, 4, 6]
    tail = [[5, 1], [6, 2], [5, 1]]

    with torch.no_grad():
        stream = _make_stream(lmb, prompts)
        stream.step(warm)
        stream.select_rows([2, 0])
        compacted = []
        for step_tokens in tail:
            recall = stream.step(step_tokens)
            compacted.append((
                RecallResult(*(value.clone() for value in recall)),
                stream.last_feature.clone()))

        # The same two rows, in the same order, never batched with the dropped
        # row. Identical output here covers history layout, per-row automaton
        # donation and the shared bank's rolling features in one shot.
        reference_stream = _make_stream(lmb, [prompts[2], prompts[0]])
        reference_stream.step([warm[2], warm[0]])
        reference = []
        for step_tokens in tail:
            recall = reference_stream.step(step_tokens)
            reference.append((
                RecallResult(*(value.clone() for value in recall)),
                reference_stream.last_feature.clone()))

    assert stream.batch_size == 2
    for (got_recall, got_feature), (want_recall, want_feature) in zip(
            compacted, reference):
        for got, want in zip(got_recall, want_recall):
            torch.testing.assert_close(got.cpu(), want.cpu(), rtol=0, atol=0)
        torch.testing.assert_close(
            got_feature.cpu(), want_feature.cpu(), rtol=0, atol=0)

    with pytest.raises(ValueError, match="zero rows"):
        stream.select_rows([])
    with pytest.raises(ValueError, match="unique"):
        stream.select_rows([0, 0])
    with pytest.raises(IndexError, match="out of range"):
        stream.select_rows([0, 2])


def test_compaction_invalidates_the_shared_banks_stale_batch_state():
    lmb = _make_lmb(layer_sites=[0])
    stream = _make_stream(lmb, [
        torch.tensor([[1, 2, 1]]),
        torch.tensor([[3, 4, 3]]),
        torch.tensor([[5, 6, 5]]),
    ])
    with torch.no_grad():
        stream.step([2, 4, 6])
        assert lmb.sites["0"]._shape == (3, 1)

        stream.select_rows([2, 0])

        # Nothing may still describe the pre-compaction batch: a forward here
        # would otherwise inject a [3,1,D] feature into a two-row stream.
        assert lmb.ctx.ids is None
        assert lmb.last_recall is None
        assert lmb.sites["0"]._shape is None
        assert not lmb.ensure_features()

        stream.step([5, 1])
        assert lmb.sites["0"]._shape == (2, 1)


def test_engram_active_slice_rejects_empty_selection():
    lmb = _make_lmb(layer_sites=[0])
    ids = _repeat_ids()
    lmb.set_input_ids(ids, active_slice=slice(0, 0))
    with pytest.raises(ValueError, match="selected no positions"):
        lmb.ensure_features()


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


def test_inputs_embeds_forward_clears_prior_hook_ids():
    model = MiniModel(C)
    ids = _repeat_ids()
    lmb = _make_lmb()
    attach_engram(model, lmb)
    install_input_ids_hook(model, lmb)
    model(ids)
    assert lmb.ctx.ids is not None and lmb.last_recall is not None
    model(inputs_embeds=model.emb(ids))
    assert lmb.ctx.ids is None and lmb.last_recall is None


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


def test_sparse_logit_bias_matches_full_copy_head():
    lmb = _make_lmb(layer_sites=[0])
    ids = _repeat_ids(2, T)
    lmb.set_input_ids(ids)
    assert lmb.ensure_features()
    with torch.no_grad():
        lmb.logit_scale.fill_(1.25)
        lmb.logit_feat.copy_(torch.tensor([0.2, 0.1, -0.05]))
    logits = torch.randn(2, T, V)
    batch = torch.tensor([0, 0, 1, 1])
    positions = torch.tensor([3, T - 2, 7, T - 1])
    sparse = lmb.logit_bias_at(logits[batch, positions], batch, positions)
    full = lmb.logit_bias(logits)
    torch.testing.assert_close(sparse, full[batch, positions])

    selected_leaf = logits[batch, positions].clone().requires_grad_()
    selected = selected_leaf * 1.0
    storage = selected.data_ptr()
    inplace = lmb.logit_bias_at(selected, batch, positions, inplace=True)
    assert inplace.data_ptr() == storage
    torch.testing.assert_close(inplace, sparse)
    inplace.square().mean().backward()
    assert selected_leaf.grad is not None


def test_sparse_logit_bias_preserves_bfloat16_destination_dtype():
    lmb = _make_lmb(layer_sites=[0])
    ids = _repeat_ids(1, T)
    lmb.set_input_ids(ids)
    assert lmb.ensure_features()
    with torch.no_grad():
        lmb.logit_scale.fill_(1.0)
    batch = torch.tensor([0, 0])
    positions = torch.tensor([T - 2, T - 1])
    logits = torch.randn(2, V, dtype=torch.bfloat16)
    result = lmb.logit_bias_at(logits.clone(), batch, positions, inplace=True)
    assert result.dtype == torch.bfloat16


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
