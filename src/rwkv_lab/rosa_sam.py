"""Fast online-suffix-automaton retrieval for ROSA — the v2 kernel.

Replaces rosa.py's naive O(R*T^2) Python reference with an O(R*T) amortized,
numba-jitted ONLINE suffix automaton, run in parallel over routes. This is what
makes ROSA runnable at real scale (hidden C=4096 -> R=C/M routes, T=1024, several
layers): pure Python was ~1e13 symbol-ops/forward (days); this is milliseconds.

Per (batch b, route r): build the suffix automaton of the KEY symbol stream ONLINE
(one symbol at a time) and, in lock-step, match the QUERY stream against it with
suffix-link fallback, so at each step we hold the longest suffix of q[0:t] that is a
substring of k[0:t]. We read tau = firstpos(match)+1 (the successor position of the
match's earliest occurrence in the keys).

CONVENTION — firstpos (earliest) vs the paper's most-recent:
  We use FIRSTPOS. It is online, O(T), and CAUSAL BY CONSTRUCTION (firstpos < t
  always, so a query at t can never read a future key). The paper (Eq.16) reads the
  MOST-RECENT occurrence; maintaining max-endpos online is O(T^2) (the endpos set
  grows as keys stream in and can't be max'd in O(1)), and on the full SAM it would
  leak future positions. firstpos still reads a *valid historical successor of a real
  suffix match*; the model learns to use whichever source. A most-recent variant
  would need a per-prefix offline suffix-link-tree pass (deferred).
"""
import numpy as np

try:
    from numba import njit, prange
    HAVE_NUMBA = True
except Exception:                                              # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        def deco(fn):
            return fn
        return deco

    prange = range


@njit(cache=True)
def _retrieve_one(qsym, ksym, K, tau, mlen_out):
    """One route. qsym/ksym: int32[T]. Writes tau[T] (= firstpos+1, -1 if no match)
    and mlen_out[T] (matched length, for validation). Online SAM over keys + query
    match. Standard SAM extend (Blumer et al.) with firstpos tracking."""
    T = qsym.shape[0]
    maxs = 2 * T + 5
    slen = np.zeros(maxs, np.int32)
    slink = np.full(maxs, -1, np.int32)
    sfirst = np.zeros(maxs, np.int32)
    strans = np.full((maxs, K), -1, np.int32)
    size = 1                                                   # state 0 = root (len 0, link -1)
    last = 0
    mstate = 0
    mlen = 0
    for t in range(T):
        c = qsym[t]
        # --- match query symbol c against the SAM of keys[0:t] ---
        while mstate != 0 and strans[mstate, c] == -1:
            mstate = slink[mstate]
            mlen = slen[mstate]
        if strans[mstate, c] != -1:
            mstate = strans[mstate, c]
            mlen = mlen + 1
        else:
            mstate = 0
            mlen = 0
        if mlen > 0:
            s = sfirst[mstate] + 1                             # successor of earliest occurrence
            if s < t:                                          # causal: strictly historical
                tau[t] = s
                mlen_out[t] = mlen
        # --- extend the SAM with key symbol ksym[t] at position t ---
        ck = ksym[t]
        cur = size
        size += 1
        slen[cur] = slen[last] + 1
        sfirst[cur] = t
        slink[cur] = -1
        pp = last
        while pp != -1 and strans[pp, ck] == -1:
            strans[pp, ck] = cur
            pp = slink[pp]
        if pp == -1:
            slink[cur] = 0
        else:
            q = strans[pp, ck]
            if slen[pp] + 1 == slen[q]:
                slink[cur] = q
            else:
                clone = size
                size += 1
                slen[clone] = slen[pp] + 1
                slink[clone] = slink[q]
                sfirst[clone] = sfirst[q]
                for k in range(K):
                    strans[clone, k] = strans[q, k]
                while pp != -1 and strans[pp, ck] == q:
                    strans[pp, ck] = clone
                    pp = slink[pp]
                slink[q] = clone
                slink[cur] = clone
        last = cur


@njit(parallel=True, cache=True)
def _retrieve_all(aq, ak, K, tau, mlen):
    B, T, R = aq.shape
    for i in prange(B * R):
        b = i // R
        r = i % R
        _retrieve_one(np.ascontiguousarray(aq[b, :, r]),
                      np.ascontiguousarray(ak[b, :, r]), K, tau[b, r], mlen[b, r])


@njit(cache=True)
def _match_step(strans, slink, slen, mstate, mlen, c):
    """One online match-step (with suffix-link fallback) WITHOUT mutating anything.
    Returns (new_state, new_len). Used both for the real query stream and for the
    counterfactual query-bit flips (Eq. 25)."""
    while mstate != 0 and strans[mstate, c] == -1:
        mstate = slink[mstate]
        mlen = slen[mstate]
    if strans[mstate, c] != -1:
        return strans[mstate, c], mlen + 1
    return 0, 0


@njit(cache=True)
def _retrieve_one_cf(qsym, ksym, K, M, tau, mlen_out, tau0, tau1):
    """Like _retrieve_one, but ALSO emits the counterfactual destination tables for
    the query-side gradient (Eq. 25): for each step t and query-bit j, tau0[t,j] /
    tau1[t,j] are the retrieval positions had query bit j been 0 / 1 (other bits as is),
    computed by re-matching from the SAME pre-step state. -1 = no match."""
    T = qsym.shape[0]
    maxs = 2 * T + 5
    slen = np.zeros(maxs, np.int32)
    slink = np.full(maxs, -1, np.int32)
    sfirst = np.zeros(maxs, np.int32)
    strans = np.full((maxs, K), -1, np.int32)
    size = 1
    last = 0
    mstate = 0
    mlen = 0
    for t in range(T):
        c = qsym[t]
        ms_prev = mstate
        ml_prev = mlen
        # real match (advances the persistent state)
        mstate, mlen = _match_step(strans, slink, slen, mstate, mlen, c)
        if mlen > 0:
            s = sfirst[mstate] + 1
            if s < t:
                tau[t] = s
                mlen_out[t] = mlen
        # counterfactual query-bit flips, re-matched from (ms_prev, ml_prev)
        for j in range(M):
            bitj = 1 << j
            c0 = c ^ (c & bitj)          # bit j forced 0
            c1 = c | bitj                # bit j forced 1
            s0, l0 = _match_step(strans, slink, slen, ms_prev, ml_prev, c0)
            s1, l1 = _match_step(strans, slink, slen, ms_prev, ml_prev, c1)
            if l0 > 0:
                p0 = sfirst[s0] + 1
                if p0 < t:
                    tau0[t, j] = p0
            if l1 > 0:
                p1 = sfirst[s1] + 1
                if p1 < t:
                    tau1[t, j] = p1
        # extend SAM with key symbol ksym[t]
        ck = ksym[t]
        cur = size
        size += 1
        slen[cur] = slen[last] + 1
        sfirst[cur] = t
        slink[cur] = -1
        pp = last
        while pp != -1 and strans[pp, ck] == -1:
            strans[pp, ck] = cur
            pp = slink[pp]
        if pp == -1:
            slink[cur] = 0
        else:
            q = strans[pp, ck]
            if slen[pp] + 1 == slen[q]:
                slink[cur] = q
            else:
                clone = size
                size += 1
                slen[clone] = slen[pp] + 1
                slink[clone] = slink[q]
                sfirst[clone] = sfirst[q]
                for k in range(K):
                    strans[clone, k] = strans[q, k]
                while pp != -1 and strans[pp, ck] == q:
                    strans[pp, ck] = clone
                    pp = slink[pp]
                slink[q] = clone
                slink[cur] = clone
        last = cur


@njit(parallel=True, cache=True)
def _retrieve_all_cf(aq, ak, K, M, tau, mlen, tau0, tau1):
    B, T, R = aq.shape
    for i in prange(B * R):
        b = i // R
        r = i % R
        _retrieve_one_cf(np.ascontiguousarray(aq[b, :, r]),
                         np.ascontiguousarray(ak[b, :, r]), K, M,
                         tau[b, r], mlen[b, r], tau0[b, r], tau1[b, r])


def sam_retrieve_cf(aq, ak, K, M):
    """Forward retrieval + counterfactual query-bit destination tables for the Eq.25
    gradient. Returns (tau[B,R,T], tau0[B,R,T,M], tau1[B,R,T,M]) — tau0/tau1[...,j] are
    the destinations had query bit j been 0/1. All int64, -1 = no match."""
    aq = np.ascontiguousarray(np.asarray(aq, dtype=np.int32))
    ak = np.ascontiguousarray(np.asarray(ak, dtype=np.int32))
    B, T, R = aq.shape
    tau = np.full((B, R, T), -1, np.int64)
    mlen = np.zeros((B, R, T), np.int32)
    tau0 = np.full((B, R, T, M), -1, np.int64)
    tau1 = np.full((B, R, T, M), -1, np.int64)
    _retrieve_all_cf(aq, ak, K, M, tau, mlen, tau0, tau1)
    return tau, tau0, tau1


def sam_retrieve(aq, ak, K, return_mlen=False):
    """aq, ak: int symbol streams [B,T,R]. Returns tau [B,R,T] (int64; firstpos+1, or
    -1 if no usable historical match). Optionally also the matched length [B,R,T]."""
    aq = np.ascontiguousarray(np.asarray(aq, dtype=np.int32))
    ak = np.ascontiguousarray(np.asarray(ak, dtype=np.int32))
    B, T, R = aq.shape
    tau = np.full((B, R, T), -1, np.int64)
    mlen = np.zeros((B, R, T), np.int32)
    _retrieve_all(aq, ak, K, tau, mlen)
    return (tau, mlen) if return_mlen else tau


if __name__ == "__main__":
    import time
    rng = np.random.default_rng(0)

    # --- correctness: every retrieved tau must be a REAL, CAUSAL suffix match ---
    B, T, R, K = 2, 80, 4, 16
    aq = rng.integers(0, K, (B, T, R))
    ak = rng.integers(0, K, (B, T, R))
    tau, mlen = sam_retrieve(aq, ak, K, return_mlen=True)
    hits = 0
    bad = 0
    for b in range(B):
        for r in range(R):
            q = aq[b, :, r]
            k = ak[b, :, r]
            for t in range(T):
                s = int(tau[b, r, t])
                if s < 0:
                    continue
                hits += 1
                L = int(mlen[b, r, t])
                e = s - 1                                      # match ends at e in keys
                ok = (s < t) and (e - L + 1 >= 0) and (L >= 1) and \
                     (list(k[e - L + 1:e + 1]) == list(q[t - L + 1:t + 1]))
                bad += 0 if ok else 1
    print(f"correctness: {hits} retrievals, {bad} INVALID (want 0)  numba={HAVE_NUMBA}")

    # --- benchmark at real ROSA scale (C=4096,M=4 -> R=1024 routes; T=1024) ---
    B, T, R, K = 1, 1024, 1024, 16
    aq = rng.integers(0, K, (B, T, R))
    ak = rng.integers(0, K, (B, T, R))
    sam_retrieve(aq[:, :16], ak[:, :16], K)                    # warm up the JIT
    t0 = time.time()
    sam_retrieve(aq, ak, K)
    dt = time.time() - t0
    print(f"benchmark R={R} T={T}: {dt * 1000:.1f} ms / route-set "
          f"(one attention layer's retrieval per forward)")
