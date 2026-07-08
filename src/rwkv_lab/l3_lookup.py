"""L³ — Large Lookup Layers (arXiv:2601.21461, Tseng & De Sa).

A per-token bank of *multiple* learned key/value embeddings, read by a CONTEXT-DEPENDENT single-query
softmax: the hidden state `x` is the query that attends over the token's own rows. That context-
dependence is the whole delta from a plain embedding table (which returns one fixed row per token)
and from our single-vector Engram — here each token owns `d_t` slots and `x` picks a soft mixture:

    s = softmax(K_t · x)            # over the token's d_t rows   (K_t: [d_t, d_in])
    a = V_tᵀ s                      # context-dependent read      (V_t: [d_t, d_emb])
    y = x + out( LayerNorm(a) )     # integrate (residual form, our safe off-state)

Distinct from our other memories: Engram/DeepEmbed = one (or hash-pooled) vector per token, no intra-
token attention; FwPKM = learned content routing + factorized product keys. L³ = STATIC token-id
routing (no n-gram hashing / tokenizer compression / pooling — the paper explicitly avoids those) +
a dense softmax read within the token's small slot set. The load-bearing quality lever in the paper
is variable per-token allocation `d_t` (frequent/high-entropy tokens get more slots, capped) — see
`allocate_slots_by_frequency`.

Deviations from the paper, flagged: (1) the paper's `W_up` projection before LayerNorm is UN-ablated;
the RWKV-community read (BlinkDL/Smerky) is that it's wasteful — so we default `up_dim=0` (normalize
the read directly) and expose `up_dim>0` for the paper-faithful form. (2) The paper gives no no-op
init; we default to a residual add with a zero-init output projection, so at init L³ is exact identity
(the repo's off≡no-op convention). `integration="concat"` gives the paper's concat-then-mix instead.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def allocate_slots_by_frequency(counts, total_slots: int, cap: int, floor: int = 1):
    """Variable per-token slot allocation `d_t` (L³'s main quality lever). The paper uses an LZW /
    entropy criterion; this is the cheaper frequency-proportional approximation with a per-token cap
    (the paper found the cap essential — uncapped, one token grabs a huge share). counts [vocab] →
    LongTensor [vocab] of slot counts in [floor, cap] summing to ≈ total_slots."""
    counts = torch.as_tensor(counts, dtype=torch.float64)
    V = counts.numel()
    alloc = torch.full((V,), floor, dtype=torch.long)
    remaining = total_slots - floor * V
    if remaining > 0 and counts.sum() > 0:
        share = (counts / counts.sum() * remaining).long()
        alloc = (alloc + share).clamp(max=cap)
    return alloc.clamp(min=floor, max=cap)


class LargeLookupLayer(nn.Module):
    """L³ lookup layer. Insert between decoder blocks at mid-stack (the paper: after ~layer 4 and
    ~16; too early = no context to exploit, too late = no downstream impact)."""

    def __init__(self, vocab: int, d_in: int, d_emb: int = 512, max_slots: int = 8,
                 up_dim: int = 0, integration: str = "residual", tie_kv: bool = False):
        super().__init__()
        self.vocab, self.d_in, self.d_emb = int(vocab), int(d_in), int(d_emb)
        self.max_slots = int(max_slots)
        self.integration = integration
        self.tie_kv = bool(tie_kv)
        if self.tie_kv and d_emb != d_in:
            raise ValueError(f"tie_kv needs d_emb == d_in (got {d_emb} vs {d_in})")

        total = self.vocab * self.max_slots            # uniform allocation by default
        self.W_K = nn.Embedding(total, d_in)           # each row is a key
        self.W_V = self.W_K if self.tie_kv else nn.Embedding(total, d_emb)   # tie: no effect, halves storage
        nn.init.normal_(self.W_K.weight, std=0.02)
        if not self.tie_kv:
            nn.init.normal_(self.W_V.weight, std=0.02)

        # slot_table[t] = row indices this token owns (right-padded with -1); slot_mask marks valid.
        idx = torch.arange(total).view(self.vocab, self.max_slots)
        self.register_buffer("slot_table", idx, persistent=True)
        self.register_buffer("slot_mask", torch.ones(self.vocab, self.max_slots, dtype=torch.bool),
                             persistent=True)

        read_dim = d_emb
        self.W_up = None
        if up_dim > 0:                                 # paper-faithful (un-ablated) up-projection
            self.W_up = nn.Linear(d_emb, up_dim, bias=False)
            read_dim = up_dim
        self.ln = nn.LayerNorm(read_dim)
        if integration == "residual":                  # our safe off-state: zero-init -> identity
            self.out = nn.Linear(read_dim, d_in, bias=False)
            nn.init.zeros_(self.out.weight)
        elif integration == "concat":                  # paper's concat-then-mix
            self.W_mix = nn.Linear(d_in + read_dim, d_in, bias=False)
        else:
            raise ValueError(f"integration must be 'residual' or 'concat', got {integration!r}")

    @torch.no_grad()
    def set_allocation(self, slot_counts):
        """Apply a variable per-token allocation (from `allocate_slots_by_frequency`). Rebuilds
        slot_table/slot_mask as a contiguous ragged layout over a table sized to the total; resizes
        the K/V embeddings to match. slot_counts [vocab] LongTensor, each in [0, max_slots]."""
        counts = torch.as_tensor(slot_counts, dtype=torch.long).clamp(0, self.max_slots)
        assert counts.numel() == self.vocab
        total = int(counts.sum().item())
        table = torch.full((self.vocab, self.max_slots), 0, dtype=torch.long)
        mask = torch.zeros(self.vocab, self.max_slots, dtype=torch.bool)
        cur = 0
        for t in range(self.vocab):
            d = int(counts[t])
            if d:
                table[t, :d] = torch.arange(cur, cur + d)
                mask[t, :d] = True
                cur += d
        # resize embeddings to `total` rows (fresh init; call before training)
        self.W_K = nn.Embedding(max(total, 1), self.d_in)
        nn.init.normal_(self.W_K.weight, std=0.02)
        if self.tie_kv:
            self.W_V = self.W_K
        else:
            self.W_V = nn.Embedding(max(total, 1), self.d_emb)
            nn.init.normal_(self.W_V.weight, std=0.02)
        self.slot_table.copy_(table) if self.slot_table.shape == table.shape else \
            self.register_buffer("slot_table", table, persistent=True)
        self.register_buffer("slot_mask", mask, persistent=True)

    def forward(self, x: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        """x [..., d_in] hidden states; token_ids [...] matching leading dims. Returns [..., d_in]."""
        lead = x.shape[:-1]
        fx = x.reshape(-1, self.d_in)                  # [N, d_in]
        ft = token_ids.reshape(-1).long()              # [N]
        slots = self.slot_table[ft]                    # [N, max_slots] row indices (pad rows -> row 0)
        mask = self.slot_mask[ft]                      # [N, max_slots]
        K = self.W_K(slots)                            # [N, max_slots, d_in]
        V = (self.W_K if self.tie_kv else self.W_V)(slots)   # [N, max_slots, d_emb]
        scores = (K * fx.unsqueeze(1)).sum(-1)         # [N, max_slots]  == K_t · x  (unscaled, per paper)
        scores = scores.masked_fill(~mask, float("-inf"))
        s = torch.softmax(scores, dim=-1)
        s = torch.nan_to_num(s, nan=0.0)               # tokens with 0 slots -> zero read (no NaN)
        a = (V * s.unsqueeze(-1)).sum(1)               # [N, d_emb]  context-dependent read
        r = self.W_up(a) if self.W_up is not None else a
        r = self.ln(r)
        if self.integration == "residual":
            y = fx + self.out(r)                       # out zero-init => identity at init
        else:
            y = self.W_mix(torch.cat([r, fx], dim=-1))
        return y.reshape(*lead, self.d_in)
