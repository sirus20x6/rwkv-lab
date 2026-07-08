"""lookahead_module.py — future-prediction auxiliary objectives (training-only).

Two retrofit-proven objectives that shape the backbone's hidden states toward a
belief state (all future-relevant info) WITHOUT touching the architecture or
inference path — both heads are dropped after training:

  NextLat  (arXiv:2511.05963)  A small MLP p_psi predicts the model's own next
      final hidden state from (h_t, emb(x_{t+1})) as a residual delta. Trained
      with SmoothL1 against stop-grad targets over a d-step teacher-forced
      rollout, plus a KL term through the FROZEN lm_head (self-distillation in
      token space). Gradients reach the backbone only through the prediction
      path (h_t), never the targets. Preserves next-token quality; forces the
      residual stream toward transition-consistency (belief-state pressure) —
      the natural complement to a fixed-state recurrent (RWKV) conversion.
      NOTE: the authors report Muon instability with this objective; keep p_psi
      (and ideally the whole run) on AdamW.

  TOP  (arXiv:2508.19228)  One extra unembedding ranks upcoming tokens by
      proximity: target score y[t,v] = W - d where d is the distance to v's
      next occurrence in (t, t+W]; absent tokens get -inf. ListNet loss =
      CE(softmax(y), log_softmax(U_top h_t)). Far cheaper signal than exact
      far-token prediction and proven in continued training at 1.8B/7B.

Both losses assume h is the POST-final-norm hidden (what lm_head consumes).
Doc-boundary masking is intentionally not applied: the conversion pipeline
trains on a contiguous token stream with no document map.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# NextLat
# ---------------------------------------------------------------------------

class NextLatPredictor(nn.Module):
    """p_psi: (h_t, emb(x_{t+1})) -> h_{t+1}, as a residual delta on h_t.

    3 linear layers with GELU, LayerNorm on the concatenated input (paper's
    Fig 11 layout). The output layer is zero-initialized so the predictor
    starts as the identity transition (delta = 0) — the SmoothL1 loss then
    begins at the natural h_{t+1}-vs-h_t gap instead of MLP noise.
    """

    def __init__(self, d_model: int, hidden: int = 0, n_layers: int = 3) -> None:
        super().__init__()
        hidden = hidden or 2 * d_model
        self.ln = nn.LayerNorm(2 * d_model)
        dims = [2 * d_model] + [hidden] * (n_layers - 1) + [d_model]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
        self.net = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return h + self.net(self.ln(torch.cat([h, act], dim=-1)))


def frozen_head_kl(pred: torch.Tensor, tgt: torch.Tensor, weight: torch.Tensor,
                   bias: Optional[torch.Tensor] = None, chunk: int = 2048) -> torch.Tensor:
    """KL( p(.|sg[h]) || p(.|h_hat) ) through a DETACHED lm_head, chunked over
    positions so the fp32 [N,V] buffers stay transient. Gradients flow into
    `pred` only (targets are computed under no_grad, head weights detached)."""
    W = weight.detach()
    b = bias.detach() if bias is not None else None
    pf = pred.reshape(-1, pred.shape[-1])
    tf = tgt.reshape(-1, tgt.shape[-1])
    n = pf.shape[0]
    total = pred.new_zeros((), dtype=torch.float32)
    for i in range(0, n, chunk):
        lp = F.log_softmax(F.linear(pf[i:i + chunk], W, b).float(), dim=-1)
        with torch.no_grad():
            lt = F.log_softmax(F.linear(tf[i:i + chunk], W, b).float(), dim=-1)
        total = total + (lt.exp() * (lt - lp)).sum(-1).sum()
    return total / max(n, 1)


def nextlat_loss(predictor: NextLatPredictor, h: torch.Tensor, act_emb: torch.Tensor,
                 head_weight: Optional[torch.Tensor], head_bias: Optional[torch.Tensor] = None,
                 d: int = 1, kl_weight: float = 1.0, kl_chunk: int = 2048):
    """d-step teacher-forced rollout of p_psi with stop-grad targets.

    h        [B,T,D] post-norm final hiddens (h_t for t in [0,T)).
    act_emb  [B,T,D] embeddings of the INPUT tokens x_t; the action carrying
             h_{t+i-1} -> h_{t+i} is emb(x_{t+i}) = act_emb[:, i:].
    Returns (smooth_l1_mean_over_steps, kl_mean_over_steps) as fp32 scalars.
    """
    T = h.shape[1]
    if not 1 <= d < T:
        raise ValueError(f"nextlat d={d} must be in [1, T) for T={T}")
    cur = h
    l_h = h.new_zeros((), dtype=torch.float32)
    l_kl = h.new_zeros((), dtype=torch.float32)
    for i in range(1, d + 1):
        cur = cur[:, : T - i]                      # states approximating h_{t+i-1}
        cur = predictor(cur, act_emb[:, i:])       # -> approximations of h_{t+i}
        tgt = h[:, i:].detach()
        l_h = l_h + F.smooth_l1_loss(cur.float(), tgt.float())
        if kl_weight > 0 and head_weight is not None:
            l_kl = l_kl + frozen_head_kl(cur, tgt, head_weight, head_bias, chunk=kl_chunk)
    return l_h / d, l_kl / d


def nextlat_jump_loss(predictor: NextLatPredictor, h: torch.Tensor, act_emb: torch.Tensor,
                      k: int, head_weight: Optional[torch.Tensor],
                      head_bias: Optional[torch.Tensor] = None,
                      kl_weight: float = 1.0, kl_chunk: int = 2048):
    """Direct k-step jump: predict h_{t+k} from (h_t, mean(emb(x_{t+1..t+k}))).

    Complements the recursive rollout: supervision from the deeper future without
    compounding one-step errors, and the pooled intervening embeddings play the
    teacher-forcing role (disambiguate WHICH future happened) while keeping the
    dynamics burden on h_t. Same stop-grad targets + frozen-head KL as the rollout.
    """
    T = h.shape[1]
    if not 2 <= k < T:
        raise ValueError(f"jump k={k} must be in [2, T) for T={T} (k=1 is the d=1 rollout)")
    cs = act_emb.float().cumsum(dim=1)
    act = (cs[:, k:] - cs[:, :-k]) / k              # row t = mean emb of x_{t+1..t+k}
    pred = predictor(h[:, : T - k], act.to(h.dtype))
    tgt = h[:, k:].detach()
    l_h = F.smooth_l1_loss(pred.float(), tgt.float())
    l_kl = h.new_zeros((), dtype=torch.float32)
    if kl_weight > 0 and head_weight is not None:
        l_kl = frozen_head_kl(pred, tgt, head_weight, head_bias, chunk=kl_chunk)
    return l_h, l_kl


# ---------------------------------------------------------------------------
# TOP
# ---------------------------------------------------------------------------

class TOPHead(nn.Module):
    """Extra unembedding for token-order prediction. rank=0 -> full [D,V]
    linear (paper-faithful; ~D*V params — at Qwen scale this is an lm_head-
    sized matrix, mind optimizer memory). rank>0 -> D->rank->V factorization."""

    def __init__(self, d_model: int, vocab: int, rank: int = 0) -> None:
        super().__init__()
        self.vocab = vocab
        if rank > 0:
            self.proj = nn.Sequential(nn.Linear(d_model, rank, bias=False),
                                      nn.Linear(rank, vocab, bias=False))
        else:
            self.proj = nn.Linear(d_model, vocab, bias=False)

    @classmethod
    def from_lm_head(cls, lm_head: nn.Linear) -> "TOPHead":
        """Full-rank head warm-started as a clone of lm_head — a sensible init
        when retrofitting (proximity ranking correlates with next-token logits)."""
        V, D = lm_head.weight.shape
        head = cls(D, V, rank=0)
        with torch.no_grad():
            head.proj.weight.copy_(lm_head.weight)
        return head

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h)


def top_targets(ids_row: torch.Tensor, t0: int, t1: int, window: int, vocab: int,
                device=None) -> torch.Tensor:
    """Dense proximity targets for positions [t0, t1) of one sequence.

    ids_row is the 1-D token stream and must extend to at least t1-1+window.
    Row t gets y[v] = window - j for the SMALLEST j in [1, window] with
    ids_row[t+j] == v (descending-j writes let the nearest occurrence win),
    -inf for tokens absent from the lookahead window.
    """
    c = t1 - t0
    y = torch.full((c, vocab), float("-inf"), dtype=torch.float32,
                   device=device if device is not None else ids_row.device)
    rows = torch.arange(c, device=y.device)
    for j in range(window, 0, -1):
        tok = ids_row[t0 + j: t1 + j].to(y.device)
        y[rows, tok] = float(window - j)
    return y


def top_loss(head: TOPHead, h: torch.Tensor, ids_full: torch.Tensor, window: int,
             chunk: int = 256) -> torch.Tensor:
    """ListNet loss between proximity targets and the TOP head's logits.

    h        [B,T,D] post-norm final hiddens.
    ids_full [B, >= T+window] token ids (the model input plus `window` future
             tokens so end-of-window positions keep a full lookahead).
    Chunked over positions: the transient fp32 [chunk,V] target/logit buffers
    bound memory; the fused-Triton variant in the paper is not needed here.
    """
    B, T, _ = h.shape
    if window < 1:
        raise ValueError(f"top window={window} must be >= 1 (0 would softmax all--inf rows to NaN)")
    if ids_full.shape[1] < T + window:
        raise ValueError(f"ids_full length {ids_full.shape[1]} < T+window={T + window}")
    total = h.new_zeros((), dtype=torch.float32)
    for b in range(B):
        for i in range(0, T, chunk):
            e = min(i + chunk, T)
            logits = head(h[b, i:e])
            y = top_targets(ids_full[b], i, e, window, head.vocab, device=h.device)
            p = torch.softmax(y, dim=-1)
            total = total + -(p * F.log_softmax(logits.float(), dim=-1)).sum(-1).sum()
    return total / (B * T)


# ---------------------------------------------------------------------------
# L-MTP: leap multi-token prediction (arXiv:2505.17505)
# ---------------------------------------------------------------------------

class LeapMTPHead(nn.Module):
    """Medusa-style leap heads: head j predicts the token at offset (j+1)*k + 1, i.e.
    {k+1, 2k+1, ...} — non-adjacent future positions. Each head is a zero-init residual
    adapter (z' = z + SiLU(W z + b)) reusing the backbone unembedding, so at init every
    head == the backbone's next-token predictor evaluated at that leap. The next-token
    (offset 1) term is the base CE, so these are the n leap heads BEYOND it."""

    def __init__(self, d_model: int, n_heads: int, k: int) -> None:
        super().__init__()
        self.k = int(k)
        self.adapters = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_heads)])
        for a in self.adapters:
            nn.init.zeros_(a.weight); nn.init.zeros_(a.bias)

    def offset(self, j: int) -> int:
        return (j + 1) * self.k + 1                     # j=0 -> k+1, j=1 -> 2k+1, ...


def lmtp_loss(head: LeapMTPHead, h: torch.Tensor, ids_full: torch.Tensor,
              lm_head: nn.Linear, chunk: int = 256) -> torch.Tensor:
    """Mean CE of the leap heads. h [B,T,D] post-norm; ids_full [B, >=T] token ids.
    Positions without a valid target at a head's offset are simply dropped (like TOP)."""
    B, T, D = h.shape
    hw = lm_head.weight
    hb = getattr(lm_head, "bias", None)
    tot = h.new_zeros((), dtype=torch.float32)
    ntok = 0
    for j, ad in enumerate(head.adapters):
        off = head.offset(j)
        tcov = min(T, ids_full.shape[1] - off)         # positions with a target at this leap
        if tcov < 1:
            continue
        zp = h[:, :tcov] + F.silu(ad(h[:, :tcov]))      # [B,tcov,D]
        tgt = ids_full[:, off:off + tcov]               # [B,tcov]
        for s in range(0, tcov, chunk):
            e = min(s + chunk, tcov)
            lg = F.linear(zp[:, s:e].reshape(-1, D).float(), hw.float(),
                          hb.float() if hb is not None else None)
            tt = tgt[:, s:e].reshape(-1)
            tot = tot + F.cross_entropy(lg, tt, reduction="sum")
            ntok += tt.numel()
    return tot / max(ntok, 1)


# ---------------------------------------------------------------------------
# Belief State Transformer (ICLR 2025) — cheap forward-only-decoder adapter
# ---------------------------------------------------------------------------

class BeliefStateHead(nn.Module):
    """BST aux objective. The decoder's own per-position hidden H[:,i] IS the forward belief
    state f_i (zero extra params). A SHALLOW backward GRU over reversed token embeddings gives
    the suffix state b_j = summary of x_{j:T}. A fused head over concat[f_i, b_j] predicts the
    token AFTER the prefix (x_{i+1}) and the token BEFORE the suffix (x_{j-1}) — the backward
    'prev' signal is the load-bearing part (its ablation collapses to forward-only). The base
    unembedding is reused/tied for both logits. Off by default; inference is unchanged."""

    def __init__(self, d_model: int, backward_layers: int = 1) -> None:
        super().__init__()
        self.benc = nn.GRU(d_model, d_model, num_layers=int(backward_layers), batch_first=True)
        self.wn = nn.Linear(2 * d_model, d_model)
        self.wp = nn.Linear(2 * d_model, d_model)


def bst_loss(head: BeliefStateHead, H: torch.Tensor, ids_full: torch.Tensor,
             embed_tokens: nn.Module, lm_head: nn.Linear, lam: float = 0.25,
             n_pairs: int = 16) -> torch.Tensor:
    """H [B,T,C] forward states (reused as f). ids_full [B,>=T]. Samples n_pairs valid (i,j),
    j>=i+2, and returns lam*CE_next + (1-lam)*CE_prev over them. Backward GRU is the only
    extra compute; the forward path is untouched."""
    B, T, C = H.shape
    if T < 3:
        return H.new_zeros((), dtype=torch.float32)
    with torch.no_grad():
        e = embed_tokens(ids_full[:, :T]).to(H.dtype)     # frozen token features
    b, _ = head.benc(e.flip(1))
    b = b.flip(1)                                          # b[:,j] summarizes x_j..x_T
    i = torch.randint(0, T - 2, (n_pairs,), device=H.device)          # 0..T-3
    span = (T - 1) - (i + 2)                                          # >= 0
    j = (i + 2 + (torch.rand(n_pairs, device=H.device) * (span + 1)).long()).clamp(max=T - 1)
    f_i = H[:, i]                                          # [B,P,C]
    b_j = b[:, j]                                          # [B,P,C]
    pair = torch.cat([f_i, b_j], dim=-1)                   # [B,P,2C]
    hw = lm_head.weight
    hb = getattr(lm_head, "bias", None)
    nx = F.linear(F.silu(head.wn(pair)).reshape(-1, C).float(), hw.float(),
                  hb.float() if hb is not None else None)
    pv = F.linear(F.silu(head.wp(pair)).reshape(-1, C).float(), hw.float(),
                  hb.float() if hb is not None else None)
    tgt_n = ids_full[:, i + 1].reshape(-1)                 # token after prefix
    tgt_p = ids_full[:, j - 1].reshape(-1)                 # token before suffix (unseen by b_j)
    return lam * F.cross_entropy(nx, tgt_n) + (1.0 - lam) * F.cross_entropy(pv, tgt_p)


# ---------------------------------------------------------------------------
# JTP: joint multi-token prediction (arXiv:2503.21801) — complements Belief State
# ---------------------------------------------------------------------------

class JTPHead(nn.Module):
    """Joint multi-token prediction (Ahn/Lamb/Langford — the same group as the Belief State
    Transformer). Predicts the JOINT distribution of D future tokens via chain-rule factoring
    with teacher-forcing routed through a lightweight "Fetch" bottleneck, so the FORWARD hidden
    h_{t-1} is forced to encode multi-step planning info (unlike Gloeckle-MTP's independent
    marginals). Fetch is a single-layer causal self-attention over the window vectors
        h^(j) = gamma*h_{t-1} + Emb(x_{t+j-1}),  j=0..D
    with a skip connection: o_i = h_{t-1} + SelfAttn(h^(0..i)); head(o_i) predicts x_{t+i}.
    Composes with BeliefStateHead: JTP enriches the forward state, BST adds the backward/prev
    signal, both sharing the same decoder hidden and unembedding (forward-only at inference)."""

    def __init__(self, d_model: int, D: int = 4, gamma: float = 0.5):
        super().__init__()
        self.D = int(D)
        self.gamma = float(gamma)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.o.weight)                      # skip-dominant at init: Fetch ~ 0
        self.scale = d_model ** -0.5

    def fetch(self, hwin):
        """Causal single-layer self-attention over the window dim of hwin [.., L, C]."""
        L = hwin.shape[-2]
        s = (self.q(hwin) @ self.k(hwin).transpose(-1, -2)) * self.scale     # [..,L,L]
        mask = torch.triu(torch.ones(L, L, device=hwin.device, dtype=torch.bool), 1)
        a = torch.softmax(s.masked_fill(mask, float("-inf")), dim=-1)
        return self.o(a @ self.v(hwin))                    # [..,L,C]


def jtp_loss(head: JTPHead, h: torch.Tensor, ids_full: torch.Tensor,
             embed_tokens: nn.Module, lm_head: nn.Linear, chunk: int = 256) -> torch.Tensor:
    """Mean CE of the JTP joint future heads (offsets 1..D). h [B,T,C]; ids_full [B,>=T+D]."""
    B, T, C = h.shape
    D = head.D
    S = min(T, ids_full.shape[1] - 1 - D)                  # sources s with target x[s+1+D] valid
    if S < 1:
        return h.new_zeros((), dtype=torch.float32)
    hs = h[:, :S]                                          # [B,S,C] forward states
    with torch.no_grad():                                  # teacher-forced token features (frozen)
        e = torch.stack([embed_tokens(ids_full[:, kk:kk + S]) for kk in range(D + 1)], dim=2)
    hwin = head.gamma * hs.unsqueeze(2) + e.to(hs.dtype)   # [B,S,D+1,C]
    o = hs.unsqueeze(2) + head.fetch(hwin.reshape(B * S, D + 1, C)).reshape(B, S, D + 1, C)  # skip
    hw = lm_head.weight
    hb = getattr(lm_head, "bias", None)
    tot = h.new_zeros((), dtype=torch.float32)
    ntok = 0
    for i in range(1, D + 1):                              # joint future tokens (i=0 is base NTP)
        tgt = ids_full[:, i + 1:i + 1 + S]                 # predict x[s+1+i]
        for st in range(0, S, chunk):
            en = min(st + chunk, S)
            lg = F.linear(o[:, st:en, i].reshape(-1, C).float(), hw.float(),
                          hb.float() if hb is not None else None)
            tt = tgt[:, st:en].reshape(-1)
            tot = tot + F.cross_entropy(lg, tt, reduction="sum")
            ntok += tt.numel()
    return tot / max(ntok, 1)


# ---------------------------------------------------------------------------
# ConceptLM-style next-concept prediction (adapted as a pure aux)
# ---------------------------------------------------------------------------

class ConceptHead(nn.Module):
    """Next-concept prediction in a discrete latent space (ConceptLM, 2602.08984),
    adapted as a RETROFIT AUXILIARY: the only paper in the bundle with continued-
    pretraining evidence near 9B (Llama-3.1-8B +0.4 avg from 9.6B tokens).

    Adaptation vs the paper: ConceptLM inserts a concept module into the residual
    stream and CONDITIONS generation on the predicted concept (an architecture
    change that survives to inference). Here the concept machinery is a pure aux
    head on the post-norm hidden — dropped at inference like NextLat/TOP — and the
    concept targets are stop-grad (the NextLat collapse discipline).

    Mechanism: the target concept at position t is the mean-pooled hidden over the
    NEXT k positions (sliding, not strided — every position gets a target). It is
    product-quantized: split into S segments, each snapped to the nearest of N
    learnable codes. Two collapse fixes copied from the paper: the codebook is
    passed through a shared 2-layer MLP+ReLU before use (SimVQ; gave ~100% code
    utilization where vanilla VQ collapsed) and the VQ loss is isolated from the
    representation path. The predictor maps h_t to per-segment logits; the
    prediction is the softmax-weighted codebook combination, regressed (MSE) onto
    the CONTINUOUS concept — the paper's L_NCP. Gradients reach the backbone only
    through the predictor's input h_t.
    """

    def __init__(self, d_model: int, chunk: int = 4, segments: int = 8,
                 codes: int = 64) -> None:
        super().__init__()
        if segments < 1 or codes < 1:
            raise ValueError(f"concept segments={segments}/codes={codes} must be >= 1")
        if d_model % segments:
            raise ValueError(f"concept segments={segments} must divide d_model={d_model}")
        if chunk < 2:
            raise ValueError(f"concept chunk={chunk} must be >= 2 (k=1 is NextLat territory)")
        self.chunk, self.S, self.N = int(chunk), int(segments), int(codes)
        self.dseg = d_model // segments
        self.codebook = nn.Parameter(torch.randn(self.S, self.N, self.dseg) * 0.02)
        self.cb_mlp = nn.Sequential(nn.Linear(self.dseg, self.dseg), nn.ReLU(),
                                    nn.Linear(self.dseg, self.dseg))
        self.predictor = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 2 * d_model), nn.GELU(),
            nn.Linear(2 * d_model, self.S * self.N))

    def loss(self, h: torch.Tensor):
        """h [B,T,D] post-norm. Returns (l_ncp, l_vq, code_frac) — fp32 scalars +
        the fraction of the codebook used this batch (collapse telemetry)."""
        k, S, N, ds = self.chunk, self.S, self.N, self.dseg
        B, T, D = h.shape
        if T <= k:
            raise ValueError(f"concept chunk={k} needs T > k (got T={T})")
        cs = h.float().cumsum(dim=1)
        c = ((cs[:, k:] - cs[:, :-k]) / k).detach()        # [B,T-k,D]: mean h[t+1..t+k], sg
        cseg = c.view(B, T - k, S, ds)
        # transform in the MODULE dtype (bf16 after .to()), then upcast: feeding fp32
        # into bf16 Linear weights raises a dtype mismatch (codex tier-3 #1)
        E = self.cb_mlp(self.codebook).float()             # [S,N,ds] transformed codes
        # nearest code per segment via ||x||^2 - 2xE + ||E||^2 (never materialize [.,.,S,N,ds])
        d2 = (cseg.pow(2).sum(-1, keepdim=True)
              - 2.0 * torch.einsum("btsd,snd->btsn", cseg, E)
              + E.pow(2).sum(-1).view(1, 1, S, N))
        idx = d2.argmin(-1)                                # [B,T-k,S]
        eq = E[torch.arange(S, device=h.device).view(1, 1, S), idx]   # [B,T-k,S,ds]
        l_vq = F.mse_loss(eq, cseg)                        # codebook/MLP chase the (sg) data
        logits = self.predictor(h[:, : T - k]).view(B, T - k, S, N)
        chat = torch.einsum("btsn,snd->btsd", torch.softmax(logits.float(), dim=-1), E)
        l_ncp = F.mse_loss(chat, cseg)                     # paper's L_NCP: regress the soft
        with torch.no_grad():                              # combo onto the CONTINUOUS concept
            code_frac = float(torch.unique(
                idx + torch.arange(S, device=idx.device).view(1, 1, S) * N).numel()) / (S * N)
        return l_ncp, l_vq, code_frac


# ---------------------------------------------------------------------------
# Bundle + CLI
# ---------------------------------------------------------------------------

class LookaheadSystem(nn.Module):
    """Owns whichever of the two objectives are enabled and computes the total
    weighted auxiliary loss from one student forward. Training-only: nothing
    here is consulted at inference, and the caller drops it after training."""

    def __init__(self, d_model: int, vocab: int, *,
                 nextlat_weight: float = 0.0, nextlat_kl_weight: float = 1.0,
                 nextlat_d: int = 1, nextlat_hidden: int = 0,
                 nextlat_jump_k: int = 0, nextlat_jump_weight: float = 0.0,
                 top_weight: float = 0.0, top_window: int = 16, top_rank: int = 0,
                 top_chunk: int = 256, kl_chunk: int = 2048,
                 concept_weight: float = 0.0, concept_chunk: int = 4,
                 concept_segments: int = 8, concept_codes: int = 64,
                 concept_vq_weight: float = 1.0,
                 lmtp_weight: float = 0.0, lmtp_k: int = 2, lmtp_heads: int = 3,
                 bst_weight: float = 0.0, bst_lambda: float = 0.25, bst_pairs: int = 16,
                 bst_layers: int = 1,
                 jtp_weight: float = 0.0, jtp_d: int = 4, jtp_gamma: float = 0.5,
                 lm_head: Optional[nn.Linear] = None, top_init: str = "lmhead") -> None:
        super().__init__()
        self.nextlat_weight = float(nextlat_weight)
        self.nextlat_kl_weight = float(nextlat_kl_weight)
        self.nextlat_d = int(nextlat_d)
        self.nextlat_jump_k = int(nextlat_jump_k)
        self.nextlat_jump_weight = float(nextlat_jump_weight)
        self.top_weight = float(top_weight)
        self.top_window = int(top_window)
        self.top_chunk = int(top_chunk)
        self.kl_chunk = int(kl_chunk)
        self.lmtp_weight = float(lmtp_weight)
        self.lmtp = (LeapMTPHead(d_model, int(lmtp_heads), int(lmtp_k))
                     if self.lmtp_weight > 0 else None)
        self.bst_weight = float(bst_weight)
        self.bst_lambda = float(bst_lambda)
        self.bst_pairs = int(bst_pairs)
        self.bst = (BeliefStateHead(d_model, int(bst_layers)) if self.bst_weight > 0 else None)
        self.jtp_weight = float(jtp_weight)
        self.jtp = (JTPHead(d_model, int(jtp_d), float(jtp_gamma)) if self.jtp_weight > 0 else None)
        if self.nextlat_weight > 0 and self.nextlat_d < 1:
            raise ValueError(f"--nextlat-d must be >= 1, got {self.nextlat_d}")
        if self.nextlat_jump_weight > 0 and self.nextlat_jump_k < 2:
            raise ValueError(f"--nextlat-jump-k must be >= 2 (k=1 is the d=1 rollout), "
                             f"got {self.nextlat_jump_k}")
        if self.top_weight > 0 and self.top_window < 1:
            raise ValueError(f"--top-window must be >= 1, got {self.top_window}")
        if self.top_chunk < 1 or self.kl_chunk < 1:
            raise ValueError("--top-chunk and kl chunk must be >= 1")
        self.nextlat = (NextLatPredictor(d_model, hidden=nextlat_hidden)
                        if self.nextlat_weight > 0 else None)
        # a SEPARATE predictor: the k-jump map is a different function from the
        # one-step transition; sharing weights would force one MLP to be both
        self.nextlat_jump = (NextLatPredictor(d_model, hidden=nextlat_hidden)
                             if self.nextlat_jump_weight > 0 else None)
        self.concept_weight = float(concept_weight)
        self.concept_vq_weight = float(concept_vq_weight)
        self.concept = (ConceptHead(d_model, chunk=concept_chunk,
                                    segments=concept_segments, codes=concept_codes)
                        if self.concept_weight > 0 else None)
        if self.top_weight > 0:
            if top_init == "lmhead" and top_rank == 0 and lm_head is not None:
                self.top = TOPHead.from_lm_head(lm_head)
            else:
                self.top = TOPHead(d_model, vocab, rank=top_rank)
        else:
            self.top = None

    @property
    def enabled(self) -> bool:
        return (self.nextlat is not None or self.nextlat_jump is not None
                or self.top is not None or self.concept is not None
                or self.lmtp is not None or self.bst is not None or self.jtp is not None)

    @property
    def extra_tokens(self) -> int:
        """Future tokens the caller must fetch beyond the model input window."""
        return self.top_window if self.top is not None else 0

    def compute(self, h_final: torch.Tensor, ids_full: torch.Tensor,
                embed_tokens: nn.Module, lm_head: Optional[nn.Linear]) -> dict:
        """h_final [B,T,D] post-norm; ids_full [B, >=T] token ids.

        TOP covers the first min(T, len(ids)-W) positions — with ids_full of
        length T+W every position gets a full lookahead window; with a shorter
        fetch (e.g. convert_train's [B,T+1]) the last W-1 positions are simply
        left out of the TOP loss instead of forcing extra data plumbing.
        Returns {"aux_total": weighted fp32 scalar, <component>: float, ...}."""
        T = h_final.shape[1]
        out: dict = {}
        total = h_final.new_zeros((), dtype=torch.float32)
        act = None
        if self.nextlat is not None or self.nextlat_jump is not None:
            with torch.no_grad():          # embeddings are frozen action inputs
                act = embed_tokens(ids_full[:, :T])
        hw = lm_head.weight if lm_head is not None else None
        hb = getattr(lm_head, "bias", None) if lm_head is not None else None
        if self.nextlat is not None:
            l_h, l_kl = nextlat_loss(
                self.nextlat, h_final, act, hw, hb,
                d=self.nextlat_d, kl_weight=self.nextlat_kl_weight,
                kl_chunk=self.kl_chunk)
            total = total + self.nextlat_weight * l_h + \
                (self.nextlat_weight * self.nextlat_kl_weight) * l_kl
            out["nextlat_h"] = float(l_h)
            out["nextlat_kl"] = float(l_kl)
        if self.nextlat_jump is not None:
            l_jh, l_jkl = nextlat_jump_loss(
                self.nextlat_jump, h_final, act, self.nextlat_jump_k, hw, hb,
                kl_weight=self.nextlat_kl_weight, kl_chunk=self.kl_chunk)
            total = total + self.nextlat_jump_weight * l_jh + \
                (self.nextlat_jump_weight * self.nextlat_kl_weight) * l_jkl
            out["nextlat_jump_h"] = float(l_jh)
            out["nextlat_jump_kl"] = float(l_jkl)
        if self.concept is not None:
            l_ncp, l_vq, code_frac = self.concept.loss(h_final)
            total = total + self.concept_weight * l_ncp + \
                (self.concept_weight * self.concept_vq_weight) * l_vq
            out["concept_ncp"] = float(l_ncp)
            out["concept_vq"] = float(l_vq)
            out["concept_codes"] = code_frac
        if self.top is not None:
            t_cov = min(T, ids_full.shape[1] - self.top_window)
            if t_cov < 1:
                raise ValueError(f"TOP needs ids beyond position window={self.top_window}; "
                                 f"got ids len {ids_full.shape[1]}")
            l_top = top_loss(self.top, h_final[:, :t_cov], ids_full, self.top_window,
                             chunk=self.top_chunk)
            total = total + self.top_weight * l_top
            out["top"] = float(l_top)
        if self.lmtp is not None and lm_head is not None:
            l_lmtp = lmtp_loss(self.lmtp, h_final, ids_full, lm_head, chunk=self.top_chunk)
            total = total + self.lmtp_weight * l_lmtp
            out["lmtp"] = float(l_lmtp)
        if self.bst is not None and lm_head is not None:
            l_bst = bst_loss(self.bst, h_final, ids_full, embed_tokens, lm_head,
                             lam=self.bst_lambda, n_pairs=self.bst_pairs)
            total = total + self.bst_weight * l_bst
            out["bst"] = float(l_bst)
        if self.jtp is not None and lm_head is not None:
            l_jtp = jtp_loss(self.jtp, h_final, ids_full, embed_tokens, lm_head, chunk=self.top_chunk)
            total = total + self.jtp_weight * l_jtp
            out["jtp"] = float(l_jtp)
        out["aux_total"] = total
        return out


def add_lookahead_cli(ap) -> None:
    g = ap.add_argument_group("lookahead (NextLat + TOP; all default OFF)")
    g.add_argument("--nextlat-weight", type=float, default=0.0,
                   help="lambda for the NextLat SmoothL1 term; >0 enables NextLat (paper: 1.0)")
    g.add_argument("--nextlat-kl-weight", type=float, default=1.0,
                   help="KL-through-frozen-head weight, relative to --nextlat-weight (paper: 1.0)")
    g.add_argument("--nextlat-d", type=int, default=1,
                   help="rollout horizon in latent steps (paper: 1-2 suffice)")
    g.add_argument("--nextlat-hidden", type=int, default=0,
                   help="p_psi MLP hidden width; 0 = 2*d_model")
    g.add_argument("--nextlat-jump-k", type=int, default=0,
                   help="direct k-step jump target h_t -> h_{t+k} (separate predictor, "
                        "pooled intervening embeddings as action); >=2, 0 = off")
    g.add_argument("--nextlat-jump-weight", type=float, default=0.0,
                   help="weight for the jump SmoothL1 term; >0 enables the jump head "
                        "(KL rides --nextlat-kl-weight)")
    g.add_argument("--top-weight", type=float, default=0.0,
                   help="weight for the TOP ListNet term; >0 enables TOP (paper favors high, up to 0.9 vs 0.1 CE)")
    g.add_argument("--top-window", type=int, default=16,
                   help="proximity-ranking lookahead window W (paper sweeps 4..4096; all beat NTP)")
    g.add_argument("--top-rank", type=int, default=0,
                   help="0 = full D->V head (paper-faithful, lm_head-sized); >0 = D->rank->V factorization to save memory")
    g.add_argument("--top-init", choices=["lmhead", "random"], default="lmhead",
                   help="full-rank head init; lmhead clones the frozen lm_head (retrofit warm start)")
    g.add_argument("--top-chunk", type=int, default=256,
                   help="positions per chunk for the fp32 [chunk,V] TOP buffers")
    g.add_argument("--concept-weight", type=float, default=0.0,
                   help="ConceptLM-style next-concept prediction weight (2602.08984, adapted as a "
                        "pure aux); >0 enables. Only bundle paper with ~8B retrofit evidence.")
    g.add_argument("--concept-chunk", type=int, default=4,
                   help="tokens per concept (paper k=4); target = mean-pooled next-chunk hidden")
    g.add_argument("--concept-segments", type=int, default=8,
                   help="product-quantization segments S (must divide d_model; paper uses n_heads)")
    g.add_argument("--concept-codes", type=int, default=64,
                   help="codes per segment N (paper 64; discrete space = N^S)")
    g.add_argument("--concept-vq-weight", type=float, default=1.0,
                   help="VQ codebook-fit loss weight, relative to --concept-weight (paper: equal)")
    g.add_argument("--lmtp-weight", type=float, default=0.0,
                   help="L-MTP (2505.17505): weight for the leap multi-token-prediction heads; >0 enables it")
    g.add_argument("--lmtp-k", type=int, default=2,
                   help="L-MTP leap stride: head j predicts offset (j+1)*k+1, i.e. {k+1,2k+1,...} (k=1 = dense MTP)")
    g.add_argument("--lmtp-heads", type=int, default=3,
                   help="number of L-MTP leap heads beyond next-token")
    g.add_argument("--bst-weight", type=float, default=0.0,
                   help="Belief State Transformer (ICLR25): weight for the fwd+bwd next/prev aux; >0 enables it")
    g.add_argument("--bst-lambda", type=float, default=0.25,
                   help="BST next-vs-prev mix (lower = more of the load-bearing backward 'prev' signal)")
    g.add_argument("--bst-pairs", type=int, default=16,
                   help="BST sampled (prefix,suffix) pairs per step")
    g.add_argument("--jtp-weight", type=float, default=0.0,
                   help="JTP (2503.21801): joint multi-token prediction via a Fetch bottleneck; >0 enables it. "
                        "Complements --bst-weight (JTP enriches the forward state, BST adds the backward signal)")
    g.add_argument("--jtp-d", type=int, default=4, help="JTP future window D")
    g.add_argument("--jtp-gamma", type=float, default=0.5, help="JTP scale on h in h^(j)=gamma*h+Emb(x)")
    g.add_argument("--lookahead-lr", type=float, default=0.0,
                   help="LR for the aux heads' AdamW param group; 0 = same as --lr")


def lookahead_from_args(args, d_model: int, vocab: int,
                        lm_head: Optional[nn.Linear]) -> Optional[LookaheadSystem]:
    if (getattr(args, "nextlat_weight", 0.0) <= 0 and getattr(args, "top_weight", 0.0) <= 0
            and getattr(args, "nextlat_jump_weight", 0.0) <= 0
            and getattr(args, "concept_weight", 0.0) <= 0
            and getattr(args, "lmtp_weight", 0.0) <= 0
            and getattr(args, "bst_weight", 0.0) <= 0
            and getattr(args, "jtp_weight", 0.0) <= 0):
        return None
    return LookaheadSystem(
        d_model, vocab,
        nextlat_weight=args.nextlat_weight, nextlat_kl_weight=args.nextlat_kl_weight,
        nextlat_d=args.nextlat_d, nextlat_hidden=args.nextlat_hidden,
        nextlat_jump_k=getattr(args, "nextlat_jump_k", 0),
        nextlat_jump_weight=getattr(args, "nextlat_jump_weight", 0.0),
        top_weight=args.top_weight, top_window=args.top_window,
        top_rank=args.top_rank, top_chunk=args.top_chunk,
        concept_weight=getattr(args, "concept_weight", 0.0),
        concept_chunk=getattr(args, "concept_chunk", 4),
        concept_segments=getattr(args, "concept_segments", 8),
        concept_codes=getattr(args, "concept_codes", 64),
        concept_vq_weight=getattr(args, "concept_vq_weight", 1.0),
        lmtp_weight=getattr(args, "lmtp_weight", 0.0),
        lmtp_k=getattr(args, "lmtp_k", 2), lmtp_heads=getattr(args, "lmtp_heads", 3),
        bst_weight=getattr(args, "bst_weight", 0.0),
        bst_lambda=getattr(args, "bst_lambda", 0.25), bst_pairs=getattr(args, "bst_pairs", 16),
        jtp_weight=getattr(args, "jtp_weight", 0.0),
        jtp_d=getattr(args, "jtp_d", 4), jtp_gamma=getattr(args, "jtp_gamma", 0.5),
        lm_head=lm_head, top_init=args.top_init)
