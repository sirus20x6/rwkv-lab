"""Golden brute-force reference for ROSA suffix-matching — validates the fast SAM kernel.

ROSA (Rapid Online Suffix Automaton) — confirmed by the RWKV community as RWKV-8's sequence-mixing
primitive — is: over a symbol stream, at each position find the longest suffix of the query prefix
that occurs earlier in the key stream, and return the successor of that earlier occurrence. The FPGA
reference (KakaruHayate/ROSA-FPGA) realises this as an online suffix automaton (INSERT/QUERY/ENDPOS)
over a 1-bit sequence.

`rosa_sam.py` implements this with a numba-jitted online SAM (O(T) amortized) for speed. THIS file is
the dead-simple O(T²·L) brute force that is "obviously correct" — used only to check the fast kernel.
It replicates rosa_sam's exact semantics so the two must agree bit-for-bit:
  * at step t the automaton holds keys[0:t] (positions 0..t-1); the query symbol at t is matched first,
    then key[t] is inserted (so a match at t is strictly causal),
  * mlen[t] = length of the LONGEST suffix of q[0..t] that occurs as a substring of keys[0..t-1],
  * tau[t] = firstpos+1 = successor of the EARLIEST occurrence of that longest suffix,
  * causal gate: report (tau, mlen) only if that successor s < t; NO fallback to a shorter suffix when
    the longest suffix's earliest successor is not causal (rosa_sam does not fall back either).
"""
from __future__ import annotations

import numpy as np


def _one(q: np.ndarray, k: np.ndarray):
    """Brute-force reference for one route. q, k: int[T]. Returns (tau[T], mlen[T])."""
    T = q.shape[0]
    tau = np.full(T, -1, np.int64)
    mlen = np.zeros(T, np.int32)
    for t in range(T):
        # longest suffix of q[0..t] that occurs as a substring of keys[0..t-1] (max length t)
        L_star, firstpos = 0, -1
        for L in range(t, 0, -1):                       # longest first, no shorter fallback
            pat = q[t - L + 1: t + 1]                    # query suffix of length L, ending at t
            earliest_e = -1
            for e in range(L - 1, t):                    # occurrence ends at e, within keys[0..t-1]
                if np.array_equal(k[e - L + 1: e + 1], pat):
                    earliest_e = e
                    break                                # earliest occurrence
            if earliest_e >= 0:
                L_star, firstpos = L, earliest_e
                break
        if L_star > 0:
            s = firstpos + 1                             # successor of the earliest occurrence
            if s < t:                                    # causal gate (matches rosa_sam), no fallback
                tau[t] = s
                mlen[t] = L_star
    return tau, mlen


def rosa_reference(aq: np.ndarray, ak: np.ndarray):
    """Brute-force ROSA over batched routes. aq, ak: int[B,T,R]. Returns tau[B,R,T], mlen[B,R,T],
    matching the shapes and semantics of rosa_sam.sam_retrieve(..., return_mlen=True)."""
    aq = np.asarray(aq); ak = np.asarray(ak)
    B, T, R = aq.shape
    tau = np.full((B, R, T), -1, np.int64)
    mlen = np.zeros((B, R, T), np.int32)
    for b in range(B):
        for r in range(R):
            tau[b, r], mlen[b, r] = _one(aq[b, :, r].astype(np.int64), ak[b, :, r].astype(np.int64))
    return tau, mlen
