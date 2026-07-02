"""Offline tools for the Lexical Memory Bank (ENGRAM_DESIGN.md, Path-C-only).

Subcommands (read the uint32 token corpus, default /thearray/data/engram_tokens.bin):

  freq   token frequency counts -> counts.npy
  alloc  frequency-aware recall-table row allocation (VIP + buckets + local
         hashing, X-GRAM style) -> alloc.npz with access_idx [V, A] / access_w [V, A]

Examples:
  python engram_lmb_build.py freq  --out engram_lmb_assets/counts.npy --max-tokens 4e9
  python engram_lmb_build.py alloc --counts engram_lmb_assets/counts.npy \\
         --out engram_lmb_assets/alloc.npz --rho 0.5

Then:
  data = np.load("engram_lmb_assets/alloc.npz")
  lmb = LexicalMemoryBank(..., access_idx=torch.from_numpy(data["access_idx"]),
                          access_w=torch.from_numpy(data["access_w"]))

Counting is O(corpus); use --max-tokens / --stride to sample — frequency ranks
stabilize long before the full 79.9B tokens.

(The n-gram mining / teacher rank-check / frozen-bank builders that served the
removed Paths A/B live in git history.)
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Tuple

import numpy as np

DEFAULT_CORPUS = "/thearray/data/engram_tokens.bin"
DEFAULT_VOCAB = 151936


# ---------------------------------------------------------------------------
# Corpus iteration
# ---------------------------------------------------------------------------

def _corpus(path: str) -> np.memmap:
    return np.memmap(path, dtype=np.uint32, mode="r")


def _chunks(tokens: np.memmap, chunk: int, max_tokens: int, stride: int):
    n = min(len(tokens), max_tokens) if max_tokens > 0 else len(tokens)
    pos = 0
    while pos < n:
        yield pos, np.asarray(tokens[pos: min(n, pos + chunk)], dtype=np.int64)
        pos += chunk * stride


# ---------------------------------------------------------------------------
# freq
# ---------------------------------------------------------------------------

def cmd_freq(args: argparse.Namespace) -> None:
    tokens = _corpus(args.corpus)
    counts = np.zeros(args.vocab_size, dtype=np.int64)
    total = 0
    for _pos, buf in _chunks(tokens, args.chunk, int(args.max_tokens), args.stride):
        counts += np.bincount(buf, minlength=args.vocab_size)[: args.vocab_size]
        total += len(buf)
        if total % (100 * args.chunk) < args.chunk:
            print(f"  {total/1e9:.2f}B tokens", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, counts)
    nz = int((counts > 0).sum())
    print(f"freq: {total/1e9:.2f}B tokens, {nz}/{args.vocab_size} vocab seen -> {args.out}")


# ---------------------------------------------------------------------------
# alloc (importable: build_freq_allocation)
# ---------------------------------------------------------------------------

def build_freq_allocation(counts: np.ndarray, rho: float = 0.5, alpha: float = 0.5,
                          k_vip: int = 200, n_buckets: int = 32, h_paths: int = 2,
                          alias_decay: float = 0.8, seed: int = 1234,
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """X-GRAM frequency-aware row allocation.

    Returns access_idx [V, A] int32 and access_w [V, A] float32 (weights sum to 1
    per token over its non-padded entries).  VIP head tokens get dedicated rows;
    remaining tokens are bucketed by smoothed mass p^alpha; sparse buckets map
    direct (spares recycled as decayed alias rows), dense buckets use h_paths
    local hash paths with geometric decay.
    """
    V = len(counts)
    S = max(int(round(rho * V)), k_vip + n_buckets)
    rng = np.random.RandomState(seed)
    order = np.argsort(-counts, kind="stable")
    vip = order[:k_vip]
    rest = order[k_vip:]

    A = max(2, h_paths)
    access_idx = np.zeros((V, A), dtype=np.int64)
    access_w = np.zeros((V, A), dtype=np.float32)

    access_idx[vip, 0] = np.arange(k_vip)
    access_w[vip, 0] = 1.0

    mass = np.power(np.maximum(counts[rest].astype(np.float64), 1.0), alpha)
    cum = np.cumsum(mass) / mass.sum()
    bucket_of = np.minimum((cum * n_buckets).astype(np.int64), n_buckets - 1)
    rows_left = S - k_vip
    span = rows_left // n_buckets
    salts = rng.randint(1, 2**31, size=h_paths)

    for b in range(n_buckets):
        toks = rest[bucket_of == b]
        start = k_vip + b * span
        r = span if b < n_buckets - 1 else rows_left - span * (n_buckets - 1)
        if len(toks) == 0 or r <= 0:
            continue
        if len(toks) <= r:  # direct + spare-alias
            access_idx[toks, 0] = start + np.arange(len(toks))
            access_w[toks, 0] = 1.0
            spare = r - len(toks)
            if spare > 0:
                access_idx[toks, 1] = start + len(toks) + np.arange(len(toks)) % spare
                access_w[toks, 1] = alias_decay
        else:  # multi-path local hashing
            for p in range(h_paths):
                h = (toks.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)
                     + np.uint64(salts[p]))
                h ^= h >> np.uint64(29)
                access_idx[toks, p] = start + (h % np.uint64(r)).astype(np.int64)
                access_w[toks, p] = alias_decay ** p

    wsum = access_w.sum(1, keepdims=True)
    access_w = np.where(wsum > 0, access_w / np.maximum(wsum, 1e-9), access_w)
    return access_idx.astype(np.int32), access_w


def cmd_alloc(args: argparse.Namespace) -> None:
    counts = np.load(args.counts)
    idx, w = build_freq_allocation(counts, rho=args.rho, alpha=args.alpha,
                                   k_vip=args.k_vip, n_buckets=args.buckets,
                                   h_paths=args.h_paths, alias_decay=args.alias_decay)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, access_idx=idx, access_w=w,
             meta=json.dumps({"rho": args.rho, "alpha": args.alpha, "k_vip": args.k_vip,
                              "buckets": args.buckets, "h_paths": args.h_paths,
                              "rows": int(idx.max()) + 1}))
    print(f"alloc: {int(idx.max())+1} rows for vocab {len(counts)} -> {args.out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("freq")
    sp.add_argument("--corpus", default=DEFAULT_CORPUS)
    sp.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB)
    sp.add_argument("--chunk", type=int, default=50_000_000)
    sp.add_argument("--max-tokens", type=float, default=4e9)
    sp.add_argument("--stride", type=int, default=1,
                    help="process every Nth chunk (subsampling)")
    sp.add_argument("--out", default="engram_lmb_assets/counts.npy")
    sp.set_defaults(fn=cmd_freq)

    sp = sub.add_parser("alloc")
    sp.add_argument("--counts", default="engram_lmb_assets/counts.npy")
    sp.add_argument("--out", default="engram_lmb_assets/alloc.npz")
    sp.add_argument("--rho", type=float, default=0.5)
    sp.add_argument("--alpha", type=float, default=0.5)
    sp.add_argument("--k-vip", type=int, default=200)
    sp.add_argument("--buckets", type=int, default=32)
    sp.add_argument("--h-paths", type=int, default=2)
    sp.add_argument("--alias-decay", type=float, default=0.8)
    sp.set_defaults(fn=cmd_alloc)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
