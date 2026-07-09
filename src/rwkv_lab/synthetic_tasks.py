"""Synthetic diagnostic tasks for RWKV-Lab — low-noise, capability-relevant accuracy signals
that fractional-nat web-text ppl cannot give.

Each task generates (input_ids, target_ids, loss_mask) batches over a small vocab; accuracy is
scored ONLY where loss_mask==1 (the answer region). These probe exactly the capabilities the
levers are for:
  - CopyTask            state capacity   (reproduce a length-L sequence after a delimiter)
  - AssocRecallTask     retrieval        (k1 v1 … kn vn ? kq -> vq)
  - InductionTask       in-context copy  (…A B… A -> B, the induction head)
and support length-generalization (train at length L, eval at 2L/4L) since each task is
parameterised by length. Accuracy on these is a clean, high-signal metric: a lever either
extends the capability or it doesn't, no ±0.1-nat noise floor.

Batches are generated DIRECTLY ON `device` with tensor ops (no numpy / no host->device copy) so
tiny models stay GPU-bound instead of stalling on Python batch construction — the small-vocab
tasks run at very high step rates where a per-row numpy loop was the bottleneck. Determinism comes
from the caller's torch.manual_seed(); the legacy `rng` arg is accepted but unused.

Vocab layout: 0=PAD, 1=SEP (delimiter / query marker), 2.. = content symbols.
"""
from __future__ import annotations
import torch

PAD, SEP = 0, 1


class Task:
    name: str
    vocab: int

    def batch(self, B: int, device, rng) -> tuple:
        """Return (x[B,T] long, y[B,T] long, mask[B,T] float) on `device`. y is next-token; loss
        and accuracy are taken only where mask==1."""
        raise NotImplementedError

    @staticmethod
    def _pack(ids, mask_from: int):
        """ids: [B,W] long on device -> (x, y, mask) with y (next-token) scored from mask_from on."""
        x = ids[:, :-1].contiguous()
        y = ids[:, 1:].contiguous()
        m = torch.zeros_like(y, dtype=torch.float32)
        m[:, mask_from:] = 1.0                      # score the region after the delimiter
        return x, y, m

    @staticmethod
    def _pack_last(ids):
        """ids: [B,W] long -> (x, y, mask) scoring ONLY the final target token."""
        x = ids[:, :-1].contiguous()
        y = ids[:, 1:].contiguous()
        m = torch.zeros_like(y, dtype=torch.float32)
        m[:, -1] = 1.0
        return x, y, m

    @staticmethod
    def accuracy(logits, y, mask) -> float:
        pred = logits.argmax(-1)
        ok = ((pred == y).float() * mask).sum()
        return float(ok / mask.sum().clamp_min(1))


class CopyTask(Task):
    """[S] SEP [S] — reproduce a random length-L symbol sequence after the delimiter."""
    def __init__(self, length: int = 32, n_symbols: int = 32):
        self.L, self.ns = length, n_symbols
        self.vocab = 2 + n_symbols
        self.name = f"copy{length}"

    def batch(self, B, device, rng=None):
        S = torch.randint(2, 2 + self.ns, (B, self.L), device=device)
        sep = torch.full((B, 1), SEP, dtype=torch.long, device=device)
        ids = torch.cat([S, sep, S], dim=1)                          # [B, 2L+1]
        return self._pack(ids, mask_from=self.L)                     # score the second S


class AssocRecallTask(Task):
    """k1 v1 k2 v2 … kn vn SEP kq -> vq. Keys distinct within an example; retrieve the queried value."""
    def __init__(self, n_pairs: int = 16, n_keys: int = 64, n_vals: int = 64):
        assert n_keys >= n_pairs
        self.n, self.nk, self.nv = n_pairs, n_keys, n_vals
        self.k0 = 2
        self.v0 = 2 + n_keys
        self.vocab = 2 + n_keys + n_vals
        self.name = f"recall{n_pairs}"

    def batch(self, B, device, rng=None):
        n = self.n
        # distinct keys per row: argsort of iid uniforms = uniform random permutation (Fisher-Yates)
        keys = torch.rand(B, self.nk, device=device).argsort(dim=1)[:, :n] + self.k0   # [B,n]
        vals = torch.randint(0, self.nv, (B, n), device=device) + self.v0
        seq = torch.empty(B, 2 * n, dtype=torch.long, device=device)
        seq[:, 0::2] = keys; seq[:, 1::2] = vals                     # k1 v1 … kn vn
        qi = torch.randint(0, n, (B,), device=device)               # which pair is queried
        br = torch.arange(B, device=device)
        kq, vq = keys[br, qi][:, None], vals[br, qi][:, None]
        sep = torch.full((B, 1), SEP, dtype=torch.long, device=device)
        ids = torch.cat([seq, sep, kq, vq], dim=1)                   # … SEP kq vq  [B, 2n+3]
        return self._pack_last(ids)                                  # score only vq


class InductionTask(Task):
    """A length-L random stream with a trigger bigram (t, c) inserted once early and the trigger t
    repeated at the end -> the model must emit c (the induction-head capability)."""
    def __init__(self, length: int = 64, n_symbols: int = 32):
        self.L, self.ns = length, n_symbols
        self.vocab = 2 + n_symbols
        self.name = f"induction{length}"

    def batch(self, B, device, rng=None):
        L = self.L
        s = torch.randint(2, 2 + self.ns, (B, L), device=device)
        t = torch.randint(2, 2 + self.ns, (B,), device=device)
        c = torch.randint(2, 2 + self.ns, (B,), device=device)
        j = torch.randint(1, L - 2, (B,), device=device)            # trigger position (early)
        br = torch.arange(B, device=device)
        s[br, j] = t; s[br, j + 1] = c                              # plant trigger -> continuation
        s[:, -1] = t                                                # repeat trigger at the end
        ids = torch.cat([s, c[:, None]], dim=1)                     # [B, L+1], target continuation = c
        return self._pack_last(ids)                                 # score the final continuation


REGISTRY = {"copy": CopyTask, "recall": AssocRecallTask, "induction": InductionTask}


def make_task(spec: str) -> Task:
    """spec = 'copy:32', 'recall:16', 'induction:64' (name:length/pairs)."""
    name, _, arg = spec.partition(":")
    cls = REGISTRY[name]
    if not arg:
        return cls()
    return cls(int(arg))
