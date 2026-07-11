"""Witten-Bell fallback distribution for unmatched ROSA suffixes.

Inspired by https://github.com/bcml-labs/rosa-plus (GPL implementation is not
copied). Community lead:
https://discord.com/channels/992359628979568762/992362722035507270/1426732439266656359
"""
from __future__ import annotations
from collections import Counter, defaultdict


class WittenBellFallback:
    """Independent, token-generic interpolated Witten-Bell n-gram oracle."""
    def __init__(self, max_order: int = 5):
        if max_order < 1: raise ValueError("max_order must be positive")
        self.max_order = max_order; self.counts = defaultdict(Counter); self.vocab = set()

    def observe(self, tokens):
        seq = list(tokens); self.vocab.update(seq)
        for i, token in enumerate(seq):
            for order in range(min(self.max_order, i) + 1):
                self.counts[tuple(seq[i-order:i])][token] += 1

    def probability(self, context, token) -> float:
        vocab = max(len(self.vocab), 1); backoff = 1.0 / vocab
        ctx = list(context)
        for order in range(0, min(self.max_order, len(ctx)) + 1):
            counts = self.counts[tuple(ctx[-order:]) if order else ()]
            total, types = sum(counts.values()), len(counts)
            if total:
                backoff = (counts[token] + types * backoff) / (total + types)
        return backoff

    def distribution(self, context) -> dict:
        if not self.vocab: return {}
        raw = {token: self.probability(context, token) for token in self.vocab}
        z = sum(raw.values()); return {token: value / z for token, value in raw.items()}


def rosa_or_fallback(tau: int, rosa_token, fallback: WittenBellFallback, context):
    return ({rosa_token: 1.0}, "rosa") if tau >= 0 else (fallback.distribution(context), "witten-bell")
