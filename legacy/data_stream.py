#!/usr/bin/env python
"""Non-repeating window sampler.

Every training step must see fresh tokens — reusing windows means epochs, and
epochs mean memorization. WindowStream walks a token array in shuffled,
non-overlapping `seqlen`-windows and hands each out at most once (raises when
exhausted), so no step ever trains on data a prior step already saw.

With ~1e9 tokens in fwedu_train (~976k windows of 1024) and runs drawing <2e5
windows, exhaustion never happens in practice — but raising (vs wrapping) keeps
the no-repeat guarantee honest.
"""
from __future__ import annotations

import numpy as np


class WindowStream:
    def __init__(self, toks, seqlen, seed=0, margin=0):
        self.toks = toks
        self.seqlen = int(seqlen)
        self.margin = int(margin)                       # extra tokens past seqlen (e.g. +1 for targets)
        n = (len(toks) - self.margin) // self.seqlen
        if n <= 0:
            raise ValueError("token stream shorter than one window")
        self.starts = (np.arange(n, dtype=np.int64) * self.seqlen)
        np.random.default_rng(seed).shuffle(self.starts)
        self.cursor = 0
        self.n_windows = n

    def remaining(self):
        return self.n_windows - self.cursor

    def next_starts(self, batch):
        if self.cursor + batch > self.n_windows:
            raise RuntimeError(
                f"WindowStream exhausted: {self.n_windows} unique windows, "
                f"asked for {self.cursor + batch}. Use more shards or fewer steps.")
        s = self.starts[self.cursor:self.cursor + batch]
        self.cursor += batch
        return s

    def next_batch(self, batch, length=None, device=None):
        """Return a [batch, length] int64 token tensor of fresh windows."""
        import torch
        T = int(length or self.seqlen)
        s = self.next_starts(batch)
        arr = np.stack([np.asarray(self.toks[int(i):int(i) + T], dtype=np.int64) for i in s])
        t = torch.as_tensor(arr)
        return t.to(device) if device is not None else t
