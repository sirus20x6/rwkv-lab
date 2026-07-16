"""Lexical Memory Bank (LMB) — ROSA-recall embedding memory for the looped RWKV
conversion (Emb_ROSA at scale).

Design: research/embedding-memory/ENGRAM_DESIGN.md.  2026-07-01 decision: the
frozen teacher n-gram bank (Path A) and the current-token Engram read (Path B
channel) were REMOVED — the sole channel is Path C, chosen for its complementarity
with ROSA-soft (which is a learned, soft, ~32-token-window recaller over hidden
bit-routes; this is exact, parameter-free, whole-sequence, token-level recall).
The removed paths live in git history and the design doc if ever wanted back.

Mechanism per position t:
  1. token_rosa_recall — online suffix automaton over the RAW TOKEN IDS finds the
     longest suffix of ids[:t+1] seen earlier in the sequence and returns the id
     of the token that FOLLOWED it ("what came next last time"), the match
     length, and the recall distance.  Parameter-free; sparse-transition SAM
     (rosa_sam.py's algorithm, but its dense [2T, K] transition table is
     infeasible at K=152k — here only the root gets a dense row).  `boundary_id`
     (the EOD token of packed training windows) segments recall so it never
     crosses document boundaries.  ~2 ms/step CPU at B=8 T=4096 V=152k.
  2. The recalled id reads a frequency-allocated learned table (X-GRAM style:
     VIP rows + bucketed local hashing from an offline allocation) refined by
     multi-view gated causal ShortConvs — consecutive recalls are consecutive
     positions of the historical match, so the convs recompose the historical
     PHRASE, not just one token.
  3. Per-site gated injection: a v-stream delta added to the RWKV time-mix
     `value` Linear output (persists in WKV state; re-fires every loop pass with
     a loop-index-conditioned scale) plus an inter-layer residual delta.
     Elementwise sigmoid QK gates at per-head + per-channel granularity; match
     LENGTH and recall DISTANCE modulate the gate (learn e.g. "defer to
     ROSA-soft inside its ~32-token window, take over beyond it").
  4. Copy head (optional wiring): lmb.logit_bias(logits) adds a gated bonus to
     the recalled token's logit — the Head-QK/pointer pattern; recall at t is
     exactly the next-token slot logits[t] predicts.  Zero-init, exact no-op.
  5. StreamingRecall — incremental per-sequence recall for generation loops.

Telemetry: lmb.telemetry() -> per-site gate means + recall stats including
frac_beyond_32, the fraction of recalls past ROSA-soft's window (the metric that
decides whether Path C does non-overlapping work on your corpus).

No-op-at-init guarantee (rosa.py convention): the output projections (`v_c`,
`h_c`) are zero-initialized, so injection is exactly zero until training moves
them.  test_engram_lmb.py asserts byte-exactness on the real
RWKV8TimeMixDeltaNet + LoopedRWKV stack.

Integration with convert_train.py (no edits to rwkv8_deltanet.py/looped_rwkv.py):

    lmb = LexicalMemoryBank(hidden_size=4096, vocab_size=151936,
                            layer_sites=[3, 10, 15],
                            num_heads=64, max_loops=args.loop_count,
                            boundary_id=EOD_TOKEN_ID)   # packed-window separator
    lmb.to(device=_p0.device, dtype=_p0.dtype)
    float_growth_params(lmb)                      # keep 1-D gates/scales fp32
    handles = attach_engram(model, lmb)           # hooks; exact no-op at init
    ids_handle = install_input_ids_hook(model, lmb)   # or lmb.set_input_ids(batch)
    # per step:   lmb.set_warmup(min(1.0, step / warmup_steps))
    # copy head:  logits = lmb.logit_bias(logits)   # after the LM head (optional)
    # optimizer:  append engram_parameters(lmb) as a named AdamW group LAST
    #             (never Muon — embedding LR-unit trap)
    # checkpoint: blob["engram"] = lmb.state_dict()
    # SMT/DMT per-layer stages: lmb.ctx.enabled = False  (hooks become no-ops)

Gradient checkpointing caveat: the loop-pass counter in inj_v assumes each core
forward runs once; activation recomputation would double-count loop indices.
"""
from __future__ import annotations

import math
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "EngramContext",
    "LearnedTable",
    "EngramSite",
    "LexicalMemoryBank",
    "RecallResult",
    "StreamingRecall",
    "attach_engram",
    "detach_engram",
    "install_input_ids_hook",
    "engram_parameters",
    "float_growth_params",
    "effective_depth_profile",
    "pick_sites",
    "token_rosa_recall",
]


def _rms(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps).to(x.dtype)


# ---------------------------------------------------------------------------
# Path C recall — token-alphabet online suffix automaton.
#
# rosa_sam.py's kernel is the same algorithm but allocates a DENSE [2T, K]
# transition table (fine for its 2^M=16 bit-route alphabet, ~5GB/sequence at the
# 152k token alphabet).  Here transitions are sparse: a dense row only for the
# root (the sole high-out-degree state), linked lists elsewhere.  Semantics match
# rosa_sam exactly (firstpos convention, match-before-extend, causal s < t) and
# are cross-checked against it in test_engram_lmb.py.
# ---------------------------------------------------------------------------

class StreamingRecall:
    """Incremental token-level ROSA recall for one sequence (generation-time API,
    and the single source of truth for the pure-python batch fallback).

    Feed tokens one at a time with extend(); each call returns the recall FOR THE
    POSITION JUST FED: (recalled_id, match_len, distance), recalled_id = -1 when
    there is no usable historical match.  `boundary_id` (e.g. the EOD token)
    resets recall at document boundaries: the boundary position itself never
    recalls or extends, and no match may cross it (root transitions are cleared;
    older states become unreachable).
    """

    def __init__(self, boundary_id: Optional[int] = None) -> None:
        self._bid = -1 if boundary_id is None else int(boundary_id)
        self._trans: List[dict] = [dict()]
        self._slen = [0]
        self._slink = [-1]
        self._sfirst = [0]
        self._last = 0
        self._mstate = 0
        self._mlen = 0
        self._t = 0
        self._ids: List[int] = []

    def extend(self, c: int) -> Tuple[int, int, int]:
        c = int(c)
        t = self._t
        self._t += 1
        self._ids.append(c)
        if self._bid >= 0 and c == self._bid:
            self._trans[0] = {}  # document boundary: root cleared -> old states unreachable
            self._mstate = 0
            self._mlen = 0
            self._last = 0
            return -1, 0, 0
        trans, slen, slink, sfirst = self._trans, self._slen, self._slink, self._sfirst
        mstate, mlen = self._mstate, self._mlen
        while True:  # match token t against the SAM of tokens [0:t]
            nxt = trans[mstate].get(c, -1)
            if nxt != -1:
                mstate = nxt
                mlen += 1
                break
            if mstate == 0:
                mlen = 0
                break
            mstate = slink[mstate]
            mlen = slen[mstate]
        rec, L, d = -1, 0, 0
        if mlen > 0:
            s = sfirst[mstate] + 1
            if s < t:
                rec, L, d = self._ids[s], mlen, t - s
        cur = len(slen)  # extend with token t
        trans.append({})
        slen.append(slen[self._last] + 1)
        slink.append(-1)
        sfirst.append(t)
        pp = self._last
        while pp != -1 and c not in trans[pp]:
            trans[pp][c] = cur
            pp = slink[pp]
        if pp == -1:
            slink[cur] = 0
        else:
            q = trans[pp][c]
            if slen[pp] + 1 == slen[q]:
                slink[cur] = q
            else:
                clone = len(slen)
                trans.append(dict(trans[q]))
                slen.append(slen[pp] + 1)
                slink.append(slink[q])
                sfirst.append(sfirst[q])
                while pp != -1 and trans[pp].get(c, -1) == q:
                    trans[pp][c] = clone
                    pp = slink[pp]
                slink[q] = clone
                slink[cur] = clone
        self._last = cur
        self._mstate, self._mlen = mstate, mlen
        return rec, L, d


def _tok_sam_one_py(sym, tau, mlen_out, bid: int = -1):  # batch fallback (no numba)
    sr = StreamingRecall(None if bid < 0 else bid)
    for t in range(len(sym)):
        rec, L, d = sr.extend(int(sym[t]))
        if rec >= 0:
            tau[t] = t - d
            mlen_out[t] = L


try:  # numba fast path (same structure, array-based)
    import numpy as _np
    from numba import njit as _njit, prange as _prange

    @_njit(cache=True)
    def _tok_sam_one_nb(sym, V, bid, tau, mlen_out):  # pragma: no cover (jitted)
        T = sym.shape[0]
        maxs = 2 * T + 5
        cap = 8 * T + 16
        slen = _np.zeros(maxs, _np.int32)
        slink = _np.full(maxs, -1, _np.int32)
        sfirst = _np.zeros(maxs, _np.int32)
        root_trans = _np.full(V, -1, _np.int32)
        # document-boundary reset without clearing the V-sized root row: root
        # transitions are only valid when their stamp matches the current epoch
        root_stamp = _np.full(V, -1, _np.int32)
        epoch = 0
        e_sym = _np.empty(cap, _np.int32)
        e_dst = _np.empty(cap, _np.int32)
        e_nxt = _np.empty(cap, _np.int32)
        head = _np.full(maxs, -1, _np.int32)
        ne = 0
        size = 1
        last = 0
        mstate = 0
        mlen = 0
        for t in range(T):
            c = sym[t]
            if bid >= 0 and c == bid:  # boundary: no match, no extend, reset
                epoch += 1
                mstate = 0
                mlen = 0
                last = 0
                continue
            while True:  # match before extend
                if mstate == 0:
                    nxt = root_trans[c] if root_stamp[c] == epoch else -1
                else:
                    nxt = -1
                    e = head[mstate]
                    while e != -1:
                        if e_sym[e] == c:
                            nxt = e_dst[e]
                            break
                        e = e_nxt[e]
                if nxt != -1:
                    mstate = nxt
                    mlen += 1
                    break
                if mstate == 0:
                    mlen = 0
                    break
                mstate = slink[mstate]
                mlen = slen[mstate]
            if mlen > 0:
                s = sfirst[mstate] + 1
                if s < t:
                    tau[t] = s
                    mlen_out[t] = mlen
            # extend with sym[t]
            cur = size
            size += 1
            slen[cur] = slen[last] + 1
            sfirst[cur] = t
            slink[cur] = -1
            pp = last
            while pp != -1:
                if pp == 0:
                    if root_stamp[c] == epoch and root_trans[c] != -1:
                        break
                    root_trans[c] = cur
                    root_stamp[c] = epoch
                else:
                    found = -1
                    e = head[pp]
                    while e != -1:
                        if e_sym[e] == c:
                            found = e
                            break
                        e = e_nxt[e]
                    if found != -1:
                        break
                    e_sym[ne] = c
                    e_dst[ne] = cur
                    e_nxt[ne] = head[pp]
                    head[pp] = ne
                    ne += 1
                pp = slink[pp]
            if pp == -1:
                slink[cur] = 0
            else:
                if pp == 0:
                    q = root_trans[c]
                else:
                    q = -1
                    e = head[pp]
                    while e != -1:
                        if e_sym[e] == c:
                            q = e_dst[e]
                            break
                        e = e_nxt[e]
                if slen[pp] + 1 == slen[q]:
                    slink[cur] = q
                else:
                    clone = size
                    size += 1
                    slen[clone] = slen[pp] + 1
                    slink[clone] = slink[q]
                    sfirst[clone] = sfirst[q]
                    e = head[q]  # q is never the root: copy its (short) edge list
                    while e != -1:
                        e_sym[ne] = e_sym[e]
                        e_dst[ne] = e_dst[e]
                        e_nxt[ne] = head[clone]
                        head[clone] = ne
                        ne += 1
                        e = e_nxt[e]
                    while pp != -1:  # redirect pp-chain c-transitions q -> clone
                        if pp == 0:
                            if root_stamp[c] == epoch and root_trans[c] == q:
                                root_trans[c] = clone
                            else:
                                break
                        else:
                            found = -1
                            e = head[pp]
                            while e != -1:
                                if e_sym[e] == c:
                                    found = e
                                    break
                                e = e_nxt[e]
                            if found == -1 or e_dst[found] != q:
                                break
                            e_dst[found] = clone
                        pp = slink[pp]
                    slink[q] = clone
                    slink[cur] = clone
            last = cur

    @_njit(parallel=True, cache=True)
    def _tok_sam_all_nb(ids, V, bid, tau, mlen):  # pragma: no cover (jitted)
        B = ids.shape[0]
        for b in _prange(B):
            _tok_sam_one_nb(ids[b], V, bid, tau[b], mlen[b])

    _HAVE_NUMBA = True
except Exception:  # pragma: no cover
    _HAVE_NUMBA = False


class RecallResult(NamedTuple):
    """Per-position recall.  All [B, T]; recalled/mlen/dist are 0 where invalid."""
    recalled: torch.Tensor  # long — token id that followed the match historically
    valid: torch.Tensor     # bool
    mlen: torch.Tensor      # long — suffix match length
    dist: torch.Tensor      # long — t - tau (how far back the recall reached)


def token_rosa_recall(ids: torch.Tensor, vocab_size: int,
                      boundary_id: Optional[int] = None) -> RecallResult:
    """Parameter-free ROSA recall over raw token ids.

    For each position t: find the longest suffix of ids[:t+1] that occurred
    earlier in the sequence; return the token id at the successor of that
    occurrence — "what followed this context last time" — plus the match length
    and the recall distance.

    boundary_id (e.g. the EOD token in packed training windows) segments the
    sequence: the boundary position never recalls or extends, and no match may
    cross it — without it, recall bleeds across unrelated documents and repeated
    separator tokens fabricate matches.

    Runs on CPU (numba kernel when available); O(T) amortized per sequence.
    """
    import numpy as np
    ids_np = ids.detach().cpu().numpy().astype(np.int32)
    B, T = ids_np.shape
    tau = np.full((B, T), -1, np.int32)
    mlen = np.zeros((B, T), np.int32)
    bid = -1 if boundary_id is None else int(boundary_id)
    if _HAVE_NUMBA:
        _tok_sam_all_nb(ids_np, int(vocab_size), bid, tau, mlen)
    else:
        for b in range(B):
            _tok_sam_one_py(ids_np[b], tau[b], mlen[b], bid)
    tau_t = torch.from_numpy(tau.astype(np.int64)).to(ids.device)
    valid = tau_t >= 0
    recalled = torch.gather(ids, 1, tau_t.clamp_min(0)) * valid
    pos = torch.arange(T, device=ids.device).unsqueeze(0)
    dist = (pos - tau_t).clamp_min(0) * valid
    return RecallResult(recalled, valid,
                        torch.from_numpy(mlen.astype(np.int64)).to(ids.device), dist)


# ---------------------------------------------------------------------------
# Shared per-forward context
# ---------------------------------------------------------------------------

class EngramContext:
    """Mutable per-forward state shared between the trainer and the hooks."""

    def __init__(self) -> None:
        self.ids: Optional[torch.Tensor] = None  # [B, T] long
        self.recall: Optional[RecallResult] = None
        self.version: int = 0
        self.enabled: bool = True
        self._warned: bool = False

    def set_input_ids(self, ids: torch.Tensor,
                      recall: Optional[RecallResult] = None) -> None:
        self.ids = ids
        self.recall = recall
        self.version += 1

    def clear(self) -> None:
        self.ids = None
        self.recall = None
        self.version += 1


# ---------------------------------------------------------------------------
# Learned recall table — frequency-allocated rows + gated causal ShortConv views
# ---------------------------------------------------------------------------

class _ShortConvView(nn.Module):
    """RMSNorm -> depthwise causal SwiGLU conv -> residual add of the raw retrieval
    (X-GRAM Eq. 8-10).  One kernel size per view."""

    def __init__(self, dim: int, kernel: int) -> None:
        super().__init__()
        self.kernel = int(kernel)
        self.norm_w = nn.Parameter(torch.ones(dim))
        self.conv_c = nn.Conv1d(dim, dim, kernel, groups=dim, bias=False)
        self.conv_g = nn.Conv1d(dim, dim, kernel, groups=dim, bias=True)
        nn.init.ones_(self.conv_g.bias)  # SiLU(1) ~= 0.73: gate starts open

    def forward(self, e: torch.Tensor) -> torch.Tensor:  # e: [B, T, d]
        x = (_rms(e) * self.norm_w.to(e.dtype)).transpose(1, 2)  # [B, d, T]
        x = F.pad(x, (self.kernel - 1, 0))  # causal
        c = self.conv_c(x) * F.silu(self.conv_g(x))
        return e + c.transpose(1, 2)


class LearnedTable(nn.Module):
    """Frequency-allocated 1-gram table with multi-view ShortConv extraction.

    The token -> physical-row mapping comes from an offline allocation
    (engram_lmb_build.py alloc): access_idx [V, A] int32 and access_w [V, A]
    float32 (VIP rows, bucketed local hashing, alias paths with decayed weights).
    Without an allocation, a uniform modulo mapping (A=1) is used so the module
    works standalone.  Row init is +/-1e-4 (SmallInitEmb: the RMSNorm inside each
    view renormalizes, so training moves rows quickly relative to init).
    """

    def __init__(self, vocab_size: int, d_row: int = 512,
                 kernels: Sequence[int] = (3, 5, 7, 9),
                 table_rows: Optional[int] = None,
                 access_idx: Optional[torch.Tensor] = None,
                 access_w: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.d_row = int(d_row)
        if access_idx is None:
            rows = int(table_rows) if table_rows else self.vocab_size
            access_idx = (torch.arange(self.vocab_size, dtype=torch.int64) % rows).unsqueeze(1)
            access_w = torch.ones(self.vocab_size, 1)
        if access_idx.shape != access_w.shape:
            raise ValueError("access_idx/access_w shape mismatch")
        rows = int(access_idx.max().item()) + 1
        self.n_rows = rows
        self.register_buffer("access_idx", access_idx.to(torch.int32))
        self.register_buffer("access_w", access_w.to(torch.float32))
        self.views = nn.ModuleList(_ShortConvView(self.d_row, k) for k in kernels)
        self.tables = nn.ParameterList(
            nn.Parameter(torch.empty(rows, self.d_row).uniform_(-1e-4, 1e-4))
            for _ in kernels)
        self.row_scale = nn.ParameterList(
            nn.Parameter(torch.zeros(rows)) for _ in kernels)
        self.view_weight = nn.Parameter(torch.ones(len(self.views)))

    def _retrieve(self, table: torch.Tensor, row_scale: torch.Tensor,
                  ids: torch.Tensor) -> torch.Tensor:
        al_idx = self.access_idx[ids].long()          # [B, T, A]
        al_w = self.access_w[ids]                     # [B, T, A]
        rows = table[al_idx]                          # [B, T, A, d]
        w = (al_w * torch.sigmoid(row_scale[al_idx])).to(rows.dtype)
        return (rows * w.unsqueeze(-1)).sum(dim=2)    # [B, T, d]

    def forward(self, ids: torch.Tensor,
                valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ids [B, T] long -> fused feature [B, T, d_row].

        `valid` [B, T] bool (optional): invalid positions are zeroed BEFORE the
        conv views (so garbage rows can't leak into neighbors through the causal
        window) and again after."""
        mask = valid.unsqueeze(-1) if valid is not None else None
        outs = []
        for v, (view, table, rs) in enumerate(zip(self.views, self.tables, self.row_scale)):
            e = self._retrieve(table, rs, ids)
            if mask is not None:
                e = e * mask.to(e.dtype)
            outs.append(self.view_weight[v].to(e.dtype) * view(e))
        out = torch.stack(outs).sum(0) / math.sqrt(len(outs))
        if mask is not None:
            out = out * mask.to(out.dtype)
        return out


# ---------------------------------------------------------------------------
# Per-layer injection site (single channel: the ROSA-recalled table read)
# ---------------------------------------------------------------------------

class EngramSite(nn.Module):
    """Projections + gates for one injection layer.

    Two zero-initialized output paths:
      inj_v  -> added to the time-mix `value` Linear output (enters WKV state);
                re-fires per loop pass with a loop-index-conditioned scale
      inj_h  -> added to the decoder-layer output (inter-layer residual anchor)

    Gate: sigmoid( rms(query)*w_q * rms(key)*w_k + per-channel bias + per-head
    bias + len_scale*log1p(match_len) ), elementwise — granularity matches
    per-head/channel loop gates; longer suffix matches earn more gate.  Recall
    features are loop-invariant, so projections are computed once per model
    forward in prepare() and cached.
    """

    def __init__(self, hidden_size: int, d_row: int,
                 num_heads: int = 64, max_loops: int = 8) -> None:
        super().__init__()
        C, H = int(hidden_size), int(num_heads)
        if C % H:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.C, self.H, self.head_dim = C, H, C // H
        self.max_loops = int(max_loops)

        def _zero_linear(d_in: int) -> nn.Linear:
            lin = nn.Linear(d_in, C, bias=False)
            nn.init.zeros_(lin.weight)
            return lin

        self.k_c = nn.Linear(d_row, C, bias=False)
        self.v_c = _zero_linear(d_row)
        self.h_c = _zero_linear(d_row)
        # gates
        self.q_scale_v = nn.Parameter(torch.ones(C))
        self.q_scale_h = nn.Parameter(torch.ones(C))
        self.k_scale = nn.Parameter(torch.ones(C))
        self.gate_bias_vc = nn.Parameter(torch.zeros(C))
        self.gate_bias_hc = nn.Parameter(torch.zeros(C))
        self.head_bias_v = nn.Parameter(torch.zeros(H))
        self.head_bias_h = nn.Parameter(torch.zeros(H))
        # recall-evidence gate modulation, zero-init: match length (longer suffix
        # -> more trust) and recall distance (lets the gate learn near/far policy,
        # e.g. defer to ROSA-soft inside its window and take over beyond it)
        self.len_scale_vc = nn.Parameter(torch.zeros(1))
        self.len_scale_hc = nn.Parameter(torch.zeros(1))
        self.dist_scale_vc = nn.Parameter(torch.zeros(1))
        self.dist_scale_hc = nn.Parameter(torch.zeros(1))
        # per-channel out scale; overwrite via set_depth_gain (effective-depth init)
        self.out_scale_v = nn.Parameter(torch.ones(C))
        self.out_scale_h = nn.Parameter(torch.ones(C))
        # loop-index-conditioned multiplier (1 + loop_scale[i]); zero-init
        self.loop_scale = nn.Parameter(torch.zeros(self.max_loops))

        # runtime (non-persistent)
        self.loop_i = 0
        self.warmup = 1.0
        self._k_r = None
        self._v_r = None
        self._h_r = None
        self._len_r = None
        self._dist_r = None
        self._shape = None
        self.last_inj_v_rms = None
        self.last_inj_h_rms = None
        self.last_gate_v_mean = None
        self.last_gate_h_mean = None
        self.stats: Dict[str, float | torch.Tensor] = {}

    def set_depth_gain(self, gain: torch.Tensor) -> None:
        """Init out scales from an effective-depth profile (per-channel or scalar)."""
        with torch.no_grad():
            g = torch.as_tensor(gain, dtype=self.out_scale_v.dtype,
                                device=self.out_scale_v.device).expand(self.C).clone()
            self.out_scale_v.copy_(g)
            self.out_scale_h.copy_(g)

    def prepare(self, m_c: torch.Tensor, recall: "RecallResult",
                warmup: float) -> None:
        """Cache loop-invariant projected recall features for this model forward."""
        vmask = recall.valid.unsqueeze(-1).to(m_c.dtype)
        kc = _rms(self.k_c(m_c)) * self.k_scale.to(m_c.dtype)
        self._k_r = kc * vmask
        self._v_r = self.v_c(m_c) * vmask
        self._h_r = self.h_c(m_c) * vmask
        self._len_r = torch.log1p(recall.mlen.to(m_c.dtype)).unsqueeze(-1)
        self._dist_r = torch.log1p(recall.dist.to(m_c.dtype)).unsqueeze(-1)
        # Keep telemetry device-resident in the hot forward path. Converting a
        # CUDA scalar to Python here would synchronize before RWKV can launch.
        self.stats["rosa_valid_rate"] = recall.valid.float().mean()
        self._shape = tuple(m_c.shape[:2])
        self.warmup = float(warmup)
        self.loop_i = 0

    def begin_layer(self) -> None:
        self.loop_i = 0

    def invalidate(self) -> None:
        self._k_r = self._v_r = self._h_r = self._len_r = self._dist_r = None
        self._shape = None
        self.last_inj_v_rms = self.last_inj_h_rms = None
        self.last_gate_v_mean = self.last_gate_h_mean = None

    def _gate(self, query: torch.Tensor, q_scale: torch.Tensor,
              bias_c: torch.Tensor, bias_h: torch.Tensor,
              len_scale: torch.Tensor, dist_scale: torch.Tensor) -> torch.Tensor:
        q = _rms(query) * q_scale.to(query.dtype)
        pre = q * self._k_r.to(query.dtype)
        pre = pre + bias_c.to(query.dtype) \
                  + bias_h.to(query.dtype).repeat_interleave(self.head_dim) \
                  + len_scale.to(query.dtype) * self._len_r.to(query.dtype) \
                  + dist_scale.to(query.dtype) * self._dist_r.to(query.dtype)
        return torch.sigmoid(pre)

    def inj_v(self, xv: torch.Tensor) -> Optional[torch.Tensor]:
        """Delta for the time-mix value output; call once per loop pass."""
        if self._k_r is None or tuple(xv.shape[:2]) != self._shape:
            return None
        i = min(self.loop_i, self.max_loops - 1)
        self.loop_i += 1
        gate = self._gate(xv, self.q_scale_v, self.gate_bias_vc, self.head_bias_v,
                          self.len_scale_vc, self.dist_scale_vc)
        scale = self.warmup * (1.0 + self.loop_scale[i].to(xv.dtype))
        out = scale * gate * self._v_r.to(xv.dtype) * self.out_scale_v.to(xv.dtype)
        # Keep only scalar device tensors here. Consumers can synchronize when
        # they log, rather than forcing a CPU/GPU barrier inside the layer.
        self.last_inj_v_rms = out.detach().float().square().mean().sqrt()
        self.last_gate_v_mean = gate.detach().float().mean()
        if not self.training:
            self.stats["gate_vc_mean"] = gate.detach().float().mean()
        return out

    def inj_h(self, h: torch.Tensor) -> Optional[torch.Tensor]:
        """Delta for the decoder-layer output (applied once per layer forward)."""
        if self._h_r is None or tuple(h.shape[:2]) != self._shape:
            return None
        gate = self._gate(h, self.q_scale_h, self.gate_bias_hc, self.head_bias_h,
                          self.len_scale_hc, self.dist_scale_hc)
        out = self.warmup * gate * self._h_r.to(h.dtype) * self.out_scale_h.to(h.dtype)
        self.last_inj_h_rms = out.detach().float().square().mean().sqrt()
        self.last_gate_h_mean = gate.detach().float().mean()
        if not self.training:
            self.stats["gate_hc_mean"] = gate.detach().float().mean()
        return out


# ---------------------------------------------------------------------------
# Top-level module
# ---------------------------------------------------------------------------

class LexicalMemoryBank(nn.Module):
    """Recall table + per-layer sites.  See module docstring for wiring."""

    def __init__(self, hidden_size: int, vocab_size: int,
                 layer_sites: Sequence[int],
                 d_row: int = 512, kernels: Sequence[int] = (3, 5, 7, 9),
                 table_rows: Optional[int] = None,
                 access_idx: Optional[torch.Tensor] = None,
                 access_w: Optional[torch.Tensor] = None,
                 num_heads: int = 64, max_loops: int = 8,
                 boundary_id: Optional[int] = None) -> None:
        super().__init__()
        self.ctx = EngramContext()
        self.boundary_id = boundary_id
        self.table = LearnedTable(vocab_size, d_row=d_row, kernels=kernels,
                                  table_rows=table_rows,
                                  access_idx=access_idx, access_w=access_w)
        self.sites = nn.ModuleDict({
            str(layer): EngramSite(hidden_size, d_row=d_row,
                                   num_heads=num_heads, max_loops=max_loops)
            for layer in sorted(set(int(l) for l in layer_sites))})
        # copy head: gated logit bonus on the recalled token (Head-QK/pointer
        # pattern).  logit_scale zero-init -> exact no-op; logit_feat = (bias,
        # len coeff, dist coeff) feeding a sigmoid confidence.
        self.logit_scale = nn.Parameter(torch.zeros(1))
        self.logit_feat = nn.Parameter(torch.zeros(3))
        self._warmup = 1.0
        self._feat_version = -1
        self.last_recall: Optional[RecallResult] = None
        self.recall_stats: Dict[str, float] = {}

    # -- trainer API ----------------------------------------------------------

    def set_input_ids(self, ids: torch.Tensor,
                      recall: Optional[RecallResult] = None) -> None:
        """Stash ids and an optional CPU-prefetched recall result for this forward."""
        self.ctx.set_input_ids(ids, recall)

    def set_warmup(self, w: float) -> None:
        self._warmup = float(min(max(w, 0.0), 1.0))

    def read_recalled(self, recalled_ids: torch.Tensor,
                      valid: torch.Tensor) -> torch.Tensor:
        """Read the learned table at recalled token ids.

        recalled_ids [B, T] long (any value where invalid), valid [B, T] bool.
        Returns [B, T, d_row], zeroed where invalid (masked before the conv views
        so garbage rows can't leak into neighbors).  Public so external recall
        sources can reuse the table.
        """
        ids = recalled_ids.clamp_min(0).clamp_max(self.table.vocab_size - 1)
        return self.table(ids, valid=valid)

    # -- hook plumbing ----------------------------------------------------------

    def ensure_features(self) -> bool:
        """Run recall + table read once per forward; fan out to all sites."""
        if not self.ctx.enabled or self.ctx.ids is None:
            if self.ctx.ids is None and not self.ctx._warned and self.ctx.enabled:
                self.ctx._warned = True
                print("[engram_lmb] input ids not set; injection inactive "
                      "(call lmb.set_input_ids or install_input_ids_hook)")
            for s in self.sites.values():
                s.invalidate()
            self.last_recall = None
            return False
        if self.ctx.version == self._feat_version:
            return True
        ids = self.ctx.ids
        rr = self.ctx.recall
        if rr is None:
            rr = token_rosa_recall(ids, self.table.vocab_size, self.boundary_id)
        elif rr.recalled.device != ids.device:
            rr = RecallResult(*(x.to(ids.device, non_blocking=True) for x in rr))
        self.last_recall = rr
        m_c = self.read_recalled(rr.recalled, rr.valid)
        for site in self.sites.values():
            site.prepare(m_c, rr, self._warmup)
        self._feat_version = self.ctx.version
        return True

    def logit_bias(self, logits: torch.Tensor) -> torch.Tensor:
        """Copy head: add a gated bonus to the RECALLED token's logit.

        Call after the LM head: `logits = lmb.logit_bias(logits)`.  Position t's
        recall is the token that historically followed the suffix ending at t —
        the same next-token slot logits[t] predicts, so the alignment is exact.
        No-op while logit_scale == 0 (init) or before ensure_features ran.
        """
        rr = self.last_recall
        if rr is None or not self.ctx.enabled:
            return logits
        if rr.valid.shape != logits.shape[:2]:
            return logits
        batch, positions = torch.meshgrid(
            torch.arange(logits.shape[0], device=logits.device),
            torch.arange(logits.shape[1], device=logits.device),
            indexing="ij",
        )
        biased = self.logit_bias_at(
            logits.flatten(0, 1), batch.flatten(), positions.flatten())
        return biased.view_as(logits)

    def logit_bias_at(self, logits: torch.Tensor, batch: torch.Tensor,
                      positions: torch.Tensor, *, inplace: bool = False) -> torch.Tensor:
        """Apply the copy head to selected sequence positions only.

        ``logits`` is ``[N,V]`` and each row corresponds to
        ``last_recall[batch[i], positions[i]]``. This lets trainers score only
        supervised caption tokens instead of materializing a full ``[B,T,V]``
        tensor while keeping the copy-head alignment identical to
        :meth:`logit_bias`.
        """
        rr = self.last_recall
        if rr is None or not self.ctx.enabled:
            return logits
        if logits.ndim != 2:
            raise ValueError("selected Engram logits must have shape [N,V]")
        batch = batch.to(device=rr.valid.device, dtype=torch.long)
        positions = positions.to(device=rr.valid.device, dtype=torch.long)
        if batch.ndim != 1 or positions.ndim != 1 or len(batch) != len(logits) \
                or len(positions) != len(logits):
            raise ValueError("Engram logit selectors must be one row per logit")
        if len(logits) == 0:
            return logits
        valid = rr.valid[batch, positions]
        recalled = rr.recalled[batch, positions]
        mlen = rr.mlen[batch, positions]
        dist = rr.dist[batch, positions]
        f = self.logit_feat.to(logits.dtype)
        conf = torch.sigmoid(f[0]
                             + f[1] * torch.log1p(mlen.to(logits.dtype))
                             + f[2] * torch.log1p(dist.to(logits.dtype)))
        bonus = self._warmup * self.logit_scale.to(logits.dtype) \
            * conf * valid.to(logits.dtype)
        # Autocast may promote the elementwise copy-head expression even when
        # its operands were explicitly converted. scatter_add requires an
        # exact dtype match between the destination and source tensors.
        bonus = bonus.to(dtype=logits.dtype)
        if inplace:
            return logits.scatter_add_(-1, recalled.unsqueeze(-1),
                                       bonus.unsqueeze(-1))
        return logits.scatter_add(-1, recalled.unsqueeze(-1),
                                  bonus.unsqueeze(-1))

    def _update_recall_stats(self, rr: RecallResult) -> None:
        """Materialize Python telemetry only when a consumer asks for it."""
        # Stack every scalar and materialize with one .tolist(): per-field
        # float() calls each forced a separate device synchronization.
        v = rr.valid
        names = ["valid_rate"]
        values = [v.float().mean()]
        if bool(v.any()):
            d = rr.dist[v].float()
            m = rr.mlen[v].float()
            names += ["dist_p50", "dist_p90",
                      # the decision metric vs ROSA-soft: recalls beyond its window
                      "frac_beyond_32", "mlen_p50", "mlen_max"]
            values += [d.median(), d.quantile(0.9),
                       (d > 32).float().mean(), m.median(), m.max()]
        rendered = torch.stack([value.float() for value in values]).tolist()
        self.recall_stats = dict(zip(names, rendered))

    def telemetry(self) -> Dict[str, Dict[str, float]]:
        if self.last_recall is not None:
            self._update_recall_stats(self.last_recall)
        out = {layer: {
            name: float(value)
            for name, value in site.stats.items()
        } for layer, site in self.sites.items()}
        out["recall"] = dict(self.recall_stats)
        return out


# ---------------------------------------------------------------------------
# Attach / detach
# ---------------------------------------------------------------------------

def _resolve(model: nn.Module, path: str):
    mod = model
    for part in path.split("."):
        mod = getattr(mod, part)
    return mod


def attach_engram(model: nn.Module, lmb: LexicalMemoryBank,
                  resolve: str = "model.layers") -> List[torch.utils.hooks.RemovableHandle]:
    """Register hooks on each site layer.  Exact no-op at init (zero output projs).

    Per site layer:
      pre-hook          ensure_features() + reset loop counter
      value-Linear hook v += inj_v(xv)   (found at linear_attn[.core].value or
                        FLA attn.v_proj;
                        fires once per loop pass; skipped when the layer has no
                        RWKV time-mix — e.g. full-attention layers get residual-only)
      output hook       out += inj_h(layer_input)   (tuple-safe, rosa pattern)
    """
    layers = _resolve(model, resolve)
    handles: List[torch.utils.hooks.RemovableHandle] = []
    for l_str, site in lmb.sites.items():
        layer = layers[int(l_str)]

        def make_pre(site: EngramSite):
            def pre(module, args, kwargs):
                lmb.ensure_features()
                site.begin_layer()
                return None
            return pre

        handles.append(layer.register_forward_pre_hook(make_pre(site), with_kwargs=True))

        la = getattr(layer, "linear_attn", None)
        if la is None:                       # FLA RWKV7DecoderLayer names the time-mix `attn`
            la = getattr(layer, "attn", None)
        if la is None:                       # RWKV7Small Block (rwkv_pretrain) names the time-mix `att`
            la = getattr(layer, "att", None)
        core = getattr(la, "core", la) if la is not None else None
        vmod = getattr(core, "value", None) if core is not None else None
        if vmod is None and core is not None:
            # FLA RWKV-7 follows HF projection names. This is the same value
            # seam as native RWKV's `.value` and must be hooked before the
            # module is placed inside the factored-loop wrapper.
            vmod = getattr(core, "v_proj", None)
        if vmod is not None:
            def make_v(site: EngramSite):
                def vhook(module, inputs, output):
                    inj = site.inj_v(inputs[0])
                    # Under CUDA autocast FLA's projection input can remain
                    # fp32 while the Linear output (and v_first) is bf16. Keep
                    # the injected value stream in the projection's dtype.
                    return None if inj is None else output + inj.to(
                        device=output.device, dtype=output.dtype)
                return vhook
            handles.append(vmod.register_forward_hook(make_v(site)))
        else:
            print(f"[engram_lmb] layer {l_str}: no linear_attn.value seam; "
                  "residual-only injection")

        def make_post(site: EngramSite):
            def post(module, args, kwargs, output):
                h = args[0] if args else kwargs.get("hidden_states")
                inj = site.inj_h(h) if h is not None else None
                if inj is None:
                    return None
                if isinstance(output, tuple):
                    return (output[0] + inj,) + tuple(output[1:])
                return output + inj
            return post

        handles.append(layer.register_forward_hook(make_post(site), with_kwargs=True))
    return handles


def detach_engram(handles: List[torch.utils.hooks.RemovableHandle]) -> None:
    for h in handles:
        h.remove()


def install_input_ids_hook(model: nn.Module, lmb: LexicalMemoryBank
                           ) -> torch.utils.hooks.RemovableHandle:
    """Model-level pre-hook that stashes input_ids into the context each forward."""
    def pre(module, args, kwargs):
        recall = kwargs.pop("precomputed_recall", None)
        ids = kwargs.get("input_ids")
        if ids is None and args and torch.is_tensor(args[0]) \
                and args[0].dtype in (torch.int64, torch.int32):
            ids = args[0]
        if ids is not None:
            lmb.set_input_ids(ids.long(), recall=recall)
        else:
            # An inputs_embeds-only call must not reuse lexical state from the
            # previous token-ID forward. Stale recall would inject a different
            # sequence whenever its shape happened to match.
            lmb.ctx.clear()
        return args, kwargs
    return model.register_forward_pre_hook(pre, with_kwargs=True)


# ---------------------------------------------------------------------------
# Optimizer helpers (convention: rosa_soft_layer.float_growth_params)
# ---------------------------------------------------------------------------

def engram_parameters(lmb: LexicalMemoryBank) -> List[nn.Parameter]:
    return [p for p in lmb.parameters() if p.requires_grad]


_GROWTH_PARAM_KEYS = ("row_scale", "view_weight", "q_scale", "k_scale",
                      "gate_bias", "head_bias", "out_scale", "loop_scale",
                      "len_scale", "dist_scale", "logit_scale", "logit_feat")


def float_growth_params(lmb: LexicalMemoryBank) -> None:
    """Keep zero-origin gates/scales fp32 after a module-wide .to(bf16):
    bf16 ULP swallows the tiny first steps away from zero.  Conv/norm params are
    deliberately excluded (mixed dtypes would break F.conv1d); their forwards
    cast explicitly."""
    for name, p in lmb.named_parameters():
        if any(k in name for k in _GROWTH_PARAM_KEYS):
            p.data = p.data.float()


# ---------------------------------------------------------------------------
# Effective-depth placement (per ENGRAM_DESIGN.md; loop gates from looped_rwkv)
# ---------------------------------------------------------------------------

def effective_depth_profile(loop_gates: Dict[int, torch.Tensor],
                            n_layers: int) -> torch.Tensor:
    """Cumulative effective depth per layer.

    loop_gates: {layer_idx: effective_rw()} — [n_loops] or [n_loops, C]; pass 0
    is the un-gated first pass.  Layers absent from the dict count as depth 1.
    Returns d_eff [n_layers] where d_eff[l] is the depth AFTER layer l.
    """
    w = torch.ones(n_layers)
    for l, g in loop_gates.items():
        g = g.detach().float()
        if g.ndim > 1:
            g = g.abs().mean(dim=tuple(range(1, g.ndim)))
        else:
            g = g.abs()
        w[int(l)] = 1.0 + float(g[1:].sum())
    return torch.cumsum(w, 0)


def pick_sites(d_eff: torch.Tensor,
               fractions: Sequence[float] = (0.094, 0.28, 0.469)) -> List[int]:
    """Layer indices whose effective-depth fraction is closest to each target.

    Defaults are the paper-faithful fractions behind {L3, +mid, L15} on a
    32-layer stack; recompute per checkpoint as loop gates evolve.
    """
    total = float(d_eff[-1])
    out: List[int] = []
    for f in fractions:
        idx = int(torch.argmin((d_eff / total - f).abs()))
        while idx in out and idx + 1 < len(d_eff):
            idx += 1
        out.append(idx)
    return out


# ---------------------------------------------------------------------------
# Smoke test (mirrors rosa.py's __main__ convention)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    V, C, B, T = 1000, 64, 2, 32
    lmb = LexicalMemoryBank(hidden_size=C, vocab_size=V, layer_sites=[0],
                            d_row=32, kernels=(3, 5), num_heads=4, max_loops=4)
    ids = torch.randint(0, V, (B, T))
    ids[:, T // 2:] = ids[:, : T // 2]  # repeats -> recalls
    lmb.set_input_ids(ids)
    assert lmb.ensure_features()
    site = lmb.sites["0"]
    assert site.stats["rosa_valid_rate"] > 0.3, "repeat sequence must recall"
    rr = token_rosa_recall(ids, V)
    # at position T//2+1 the suffix (ids[0], ids[1]) matches its first occurrence
    # ending at position 1; the recall is the SUCCESSOR token ids[2]
    assert bool(rr.valid[0, T // 2 + 1]) and \
        int(rr.recalled[0, T // 2 + 1]) == int(ids[0, 2]), "recall = successor of match"
    assert int(rr.dist[0, T // 2 + 1]) == T // 2 - 1, "distance = t - tau"
    xv = torch.randn(B, T, C)
    inj = site.inj_v(xv)
    assert inj is not None and float(inj.abs().max()) == 0.0, "no-op at init"
    assert float(site.inj_h(xv).abs().max()) == 0.0
    logits = torch.randn(B, T, V)
    assert torch.equal(lmb.logit_bias(logits), logits), "copy head no-op at init"
    print("engram_lmb smoke test OK")
