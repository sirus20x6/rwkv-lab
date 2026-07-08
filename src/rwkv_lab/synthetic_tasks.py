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

Vocab layout: 0=PAD, 1=SEP (delimiter / query marker), 2.. = content symbols.
"""
from __future__ import annotations
import numpy as np
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
    def _pack(ids: np.ndarray, mask_from: int, device):
        x = torch.from_numpy(ids[:, :-1]).long().to(device)
        y = torch.from_numpy(ids[:, 1:]).long().to(device)
        m = torch.zeros_like(y, dtype=torch.float32)
        m[:, mask_from:] = 1.0                      # score the region after the delimiter
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

    def batch(self, B, device, rng):
        S = rng.integers(2, 2 + self.ns, size=(B, self.L))
        ids = np.concatenate([S, np.full((B, 1), SEP), S], axis=1)   # [B, 2L+1]
        return self._pack(ids, mask_from=self.L, device=device)      # score the second S


class AssocRecallTask(Task):
    """k1 v1 k2 v2 … kn vn SEP kq -> vq. Keys distinct within an example; retrieve the queried value."""
    def __init__(self, n_pairs: int = 16, n_keys: int = 64, n_vals: int = 64):
        assert n_keys >= n_pairs
        self.n, self.nk, self.nv = n_pairs, n_keys, n_vals
        self.k0 = 2
        self.v0 = 2 + n_keys
        self.vocab = 2 + n_keys + n_vals
        self.name = f"recall{n_pairs}"

    def batch(self, B, device, rng):
        rows = []
        ans_pos = []
        for _ in range(B):
            keys = rng.choice(self.nk, size=self.n, replace=False) + self.k0
            vals = rng.integers(0, self.nv, size=self.n) + self.v0
            seq = np.empty(2 * self.n, dtype=np.int64)
            seq[0::2] = keys; seq[1::2] = vals
            qi = rng.integers(0, self.n)
            row = np.concatenate([seq, [SEP], [keys[qi]], [vals[qi]]])  # … SEP kq vq
            rows.append(row); ans_pos.append(len(row) - 1)             # vq is the last token
        ids = np.stack(rows)
        x = torch.from_numpy(ids[:, :-1]).long().to(device)
        y = torch.from_numpy(ids[:, 1:]).long().to(device)
        m = torch.zeros_like(y, dtype=torch.float32)
        m[:, -1] = 1.0                                                 # score only vq
        return x, y, m


class InductionTask(Task):
    """A length-L random stream with a trigger bigram (t, c) inserted once early and the trigger t
    repeated at the end -> the model must emit c (the induction-head capability)."""
    def __init__(self, length: int = 64, n_symbols: int = 32):
        self.L, self.ns = length, n_symbols
        self.vocab = 2 + n_symbols
        self.name = f"induction{length}"

    def batch(self, B, device, rng):
        rows = []
        for _ in range(B):
            s = rng.integers(2, 2 + self.ns, size=self.L)
            t, c = rng.integers(2, 2 + self.ns, size=2)
            j = rng.integers(1, self.L - 2)
            s[j] = t; s[j + 1] = c                                     # plant trigger->continuation
            s[-1] = t                                                  # repeat trigger at the end
            rows.append(np.concatenate([s, [c]]))                      # target continuation
        ids = np.stack(rows)
        x = torch.from_numpy(ids[:, :-1]).long().to(device)
        y = torch.from_numpy(ids[:, 1:]).long().to(device)
        m = torch.zeros_like(y, dtype=torch.float32)
        m[:, -1] = 1.0                                                 # score the final continuation
        return x, y, m


REGISTRY = {"copy": CopyTask, "recall": AssocRecallTask, "induction": InductionTask}


def make_task(spec: str) -> Task:
    """spec = 'copy:32', 'recall:16', 'induction:64' (name:length/pairs)."""
    name, _, arg = spec.partition(":")
    cls = REGISTRY[name]
    if not arg:
        return cls()
    return cls(int(arg))
