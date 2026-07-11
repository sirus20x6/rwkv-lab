"""Fast online-suffix-automaton retrieval for ROSA — the v2 kernel.

Replaces rosa.py's naive O(R*T^2) Python reference with O(R*T) amortized online
suffix automata: a device-native Numba-CUDA kernel for GPU training and a parallel
Numba CPU oracle/fallback. This is what
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
from collections import OrderedDict
import threading

import numpy as np

try:
    from numba import cuda, njit, prange
    HAVE_NUMBA = True
    HAVE_CUDA = cuda.is_available()
except Exception:                                              # pragma: no cover
    HAVE_NUMBA = False
    HAVE_CUDA = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        def deco(fn):
            return fn
        return deco

    prange = range


if HAVE_CUDA:
    @cuda.jit(device=True, inline=True)
    def _cuda_match_step(strans, slink, slen, route, state, mlen, symbol):
        while state != 0 and strans[route, state, symbol] == -1:
            state = slink[route, state]
            mlen = slen[route, state]
        nxt = strans[route, state, symbol]
        if nxt != -1:
            return nxt, mlen + 1
        return 0, 0


    @cuda.jit
    def _cuda_retrieve_all_cf(aq, ak, K, M, tau, tau0, tau1,
                              slen, slink, sfirst, strans):
        """ROSA online SAM (arXiv:2602.02499), one independent route per GPU thread."""
        route = cuda.grid(1)
        B, T, R = aq.shape
        routes = B * R
        if route >= routes:
            return
        b, r = route // R, route % R
        maxs = 2 * T + 5
        for state in range(maxs):
            slen[route, state] = 0
            slink[route, state] = -1
            sfirst[route, state] = 0
            for symbol in range(K):
                strans[route, state, symbol] = -1
        size, last, mstate, mlen = 1, 0, 0, 0
        for t in range(T):
            tau[b, r, t] = -1
            for j in range(M):
                tau0[b, r, t, j] = -1
                tau1[b, r, t, j] = -1

            symbol = aq[b, t, r]
            old_state, old_len = mstate, mlen
            mstate, mlen = _cuda_match_step(
                strans, slink, slen, route, mstate, mlen, symbol)
            if mlen > 0:
                source = sfirst[route, mstate] + 1
                if source < t:
                    tau[b, r, t] = source

            for j in range(M):
                bit = 1 << j
                symbol0 = symbol ^ (symbol & bit)
                symbol1 = symbol | bit
                state0, len0 = _cuda_match_step(
                    strans, slink, slen, route, old_state, old_len, symbol0)
                state1, len1 = _cuda_match_step(
                    strans, slink, slen, route, old_state, old_len, symbol1)
                if len0 > 0:
                    source0 = sfirst[route, state0] + 1
                    if source0 < t:
                        tau0[b, r, t, j] = source0
                if len1 > 0:
                    source1 = sfirst[route, state1] + 1
                    if source1 < t:
                        tau1[b, r, t, j] = source1

            key_symbol = ak[b, t, r]
            cur = size
            size += 1
            slen[route, cur] = slen[route, last] + 1
            sfirst[route, cur] = t
            slink[route, cur] = -1
            parent = last
            while parent != -1 and strans[route, parent, key_symbol] == -1:
                strans[route, parent, key_symbol] = cur
                parent = slink[route, parent]
            if parent == -1:
                slink[route, cur] = 0
            else:
                target = strans[route, parent, key_symbol]
                if slen[route, parent] + 1 == slen[route, target]:
                    slink[route, cur] = target
                else:
                    clone = size
                    size += 1
                    slen[route, clone] = slen[route, parent] + 1
                    slink[route, clone] = slink[route, target]
                    sfirst[route, clone] = sfirst[route, target]
                    for k in range(K):
                        strans[route, clone, k] = strans[route, target, k]
                    while parent != -1 and strans[route, parent, key_symbol] == target:
                        strans[route, parent, key_symbol] = clone
                        parent = slink[route, parent]
                    slink[route, target] = clone
                    slink[route, cur] = clone
            last = cur


_CUDA_WORKSPACES = OrderedDict()
_CUDA_WORKSPACE_LOCK = threading.Lock()
_CUDA_WORKSPACE_LIMIT = 4


def cuda_sam_workspace_bytes(batch: int, length: int, routes: int, alphabet: int) -> int:
    """Bytes required for the four int32 online-SAM workspace tables."""
    states = 2 * int(length) + 5
    route_count = int(batch) * int(routes)
    return route_count * states * (3 + int(alphabet)) * 4


def _cuda_workspace(torch, *, device, stream_id: int, routes: int, states: int, alphabet: int):
    key = (str(device), int(stream_id), int(routes), int(states), int(alphabet))
    with _CUDA_WORKSPACE_LOCK:
        workspace = _CUDA_WORKSPACES.pop(key, None)
        if workspace is None:
            base = torch.empty((routes, states), dtype=torch.int32, device=device)
            workspace = (base, torch.empty_like(base), torch.empty_like(base),
                         torch.empty((routes, states, alphabet), dtype=torch.int32, device=device))
        _CUDA_WORKSPACES[key] = workspace
        while len(_CUDA_WORKSPACES) > _CUDA_WORKSPACE_LIMIT:
            _CUDA_WORKSPACES.popitem(last=False)
        return workspace


def cuda_sam_retrieve_cf(aq, ak, K, M):
    """Device-native ROSA retrieval on PyTorch's current CUDA stream.

    The workspace stores the independent online automaton for each route.  Outputs match
    :func:`sam_retrieve_cf`; the CPU/Numba implementation remains the golden fallback.
    """
    if not HAVE_CUDA or not getattr(aq, "is_cuda", False):
        raise RuntimeError("CUDA suffix-automaton retrieval is unavailable")
    import torch
    aq = aq.to(dtype=torch.int32).contiguous()
    ak = ak.to(dtype=torch.int32).contiguous()
    B, T, R = aq.shape
    routes, maxs = B * R, 2 * T + 5
    required = cuda_sam_workspace_bytes(B, T, R, K)
    free_bytes, _ = torch.cuda.mem_get_info(aq.device)
    if required > int(free_bytes * 0.5):
        raise RuntimeError(
            f"ROSA CUDA SAM needs {required / 2**30:.2f} GiB workspace, more than half of "
            f"the currently free {free_bytes / 2**30:.2f} GiB; reduce context/batch/routes")
    tau = torch.empty((B, R, T), dtype=torch.long, device=aq.device)
    tau0 = torch.empty((B, R, T, M), dtype=torch.long, device=aq.device)
    tau1 = torch.empty_like(tau0)
    current_stream = torch.cuda.current_stream(aq.device)
    slen, slink, sfirst, strans = _cuda_workspace(
        torch, device=aq.device, stream_id=current_stream.cuda_stream,
        routes=routes, states=maxs, alphabet=K)
    stream = cuda.external_stream(current_stream.cuda_stream)
    # Each route performs a long sequential SAM update and uses substantial register state; small
    # blocks distribute route work across more SMs than the usual elementwise-kernel block size.
    threads = 8
    _cuda_retrieve_all_cf[(routes + threads - 1) // threads, threads, stream](
        cuda.as_cuda_array(aq), cuda.as_cuda_array(ak), K, M,
        cuda.as_cuda_array(tau), cuda.as_cuda_array(tau0), cuda.as_cuda_array(tau1),
        cuda.as_cuda_array(slen), cuda.as_cuda_array(slink), cuda.as_cuda_array(sfirst),
        cuda.as_cuda_array(strans))
    return tau, tau0, tau1


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
