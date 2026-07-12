"""ROSA-Tuning — "RWKV Online Suffix Automaton"-Tuning (arXiv 2602.02499v2).

IMPORTANT: this is NOT a low-rank / LoRA-style adapter. ROSA is a retrieval-and-
recall SIDE MODULE injected additively into the residual stream:

  1. project hidden H -> Q^vec,K^vec,V^vec via its OWN W_q,W_k,W_v (decoupled from
     the backbone attention), on LN(H)                                   (Eqs. 8-9)
  2. sign-binarize each to 1 bit/dim (Theorem 1: binary is provably optimal) (10-12)
  3. split C channels into R routes of width M, bit-pack each route into an
     integer symbol in {0..2^M-1}                                        (13-15)
  4. per (batch, route): maintain an ONLINE suffix automaton over the KEY symbol
     stream; for each query find the longest suffix of the query stream that
     occurs in the keys, and read the SUCCESSOR position of its most-recent
     occurrence:  tau = endpos(match)+1  (causal, < t, else -1)             (16)
  5. read the VALUE symbol at tau, unpack its bits, map through a learned affine
     y = m * (e0 + (e1-e0)*bits), then inj = W_out @ y                   (17-22)
  6. fuse:  post-attn  H' = H + Attn_W(LN H) + inj                        (1-3)
            pre-attn   H' = H + Attn_W(LN((1-a)H + a*inj)),  a=sigmoid(a0) (4-7)

Init-to-identity: e0=e1=0  =>  y==0  =>  inj==0 for any input, so grafting onto a
pretrained model is an exact no-op at step 0; the recall path activates during
training (§3.4). W_out=I is a clean start.

INTEGRATION (per the paper, §6, and our project):
 * Complementary to Engram — run both as parallel additive injections:
       H' = H + A + inj_engram + inj_rosa
   (Engram retrieves pretrained-knowledge from a static table; ROSA retrieves
   in-context history. Different sources, no conflict.)
 * Attach ONLY to windowed/softmax-attention layers, NOT the RWKV-7-converted
   layers (RWKV-7 already carries rich global state, so ROSA there is redundant).

-------------------------------------------------------------------------------
THIS FILE keeps the reference and production paths together. What is faithful vs deferred:
  FAITHFUL forward: binarize -> route-pack -> longest-suffix retrieval -> read ->
    unpack -> affine -> W_out. CUDA runs a device-native online suffix automaton;
    CPU uses the Numba implementation as its exact oracle, with the O(T^2) matcher
    retained only as a dependency-free fallback.
  EXACT gradients: e0, e1, W_out (autograd), W_v via destination scatter (Eq. 24),
    and W_q via counterfactual query-bit destinations (Eq. 25).
  DEFERRED: K counterfactual gradients (Eq. 26), whose mutation changes the global
    automaton rather than one local query transition.
Paper: Zheng/Wang/Ren/Chen, "ROSA-Tuning: Enhancing Long-Context Modeling via
Suffix Matching", 2602.02499v2; ref impl github.com/zyaaa-ux/ROSA-Tuning.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Retrieval (v1: naive, correct). Returns tau[b,r,t] = source position to read,
# or -1 if there is no usable historical match. SAM kernel is the v2 perf path.
# ---------------------------------------------------------------------------
def retrieve_routes(aq: torch.Tensor, ak: torch.Tensor, max_match: int = 64) -> torch.Tensor:
    """aq, ak: int symbol streams [B, T, R] (query / key). Returns tau [B, R, T] long.

    Mirrors rosa_sam / rosa_reference exactly: for each (b, r, t) find the LONGEST L
    (<= max_match) such that the query suffix aq[b, t-L+1 : t+1, r] occurs as a
    contiguous run in the keys ak[b, 0:t-1+1, r] ending at some e in [L-1, t-1]; take
    the EARLIEST such e (firstpos); tau = e + 1 (the successor, Eq. 16), reported only
    if it is STRICTLY causal (tau < t) — no fallback to a shorter suffix otherwise.
    tau is left at -1 when no match exists. max_match caps L for the naive search
    (the SAM finds the true longest; the cap only matters for very long exact repeats)."""
    B, T, R = aq.shape
    aq_l = aq.tolist()
    ak_l = ak.tolist()
    tau = [[[-1] * T for _ in range(R)] for _ in range(B)]
    for b in range(B):
        for r in range(R):
            qcol = [aq_l[b][t][r] for t in range(T)]
            kcol = [ak_l[b][t][r] for t in range(T)]
            for t in range(1, T):
                best_e = -1
                upper = min(max_match, t)
                # longest suffix first; stop at the first L that has any occurrence
                for L in range(upper, 0, -1):
                    pat = qcol[t - L + 1:t + 1]
                    # scan key end-positions e in [L-1, t-1] for the EARLIEST match
                    for e in range(L - 1, t):
                        if kcol[e - L + 1:e + 1] == pat:
                            best_e = e
                            break
                    if best_e >= 0:
                        break
                if best_e >= 0:
                    s = best_e + 1            # successor of the earliest occurrence
                    if s < t:                 # causal gate: strictly historical (matches rosa_sam)
                        tau[b][r][t] = s
    return torch.tensor(tau, dtype=torch.long, device=aq.device)


try:
    from .rosa_sam import (HAVE_CUDA as _HAVE_CUDA_SAM,
                           cuda_sam_retrieve_cf as _cuda_sam_retrieve_cf,
                           sam_retrieve as _sam_retrieve,
                           sam_retrieve_cf as _sam_retrieve_cf)
    _HAVE_SAM = True                                          # fast numba online-SAM kernel (v2)
except Exception:                                            # pragma: no cover
    _HAVE_SAM = False
    _HAVE_CUDA_SAM = False

_WARNED_FALLBACK = False                                     # one-time naive-fallback warning


def _retrieve(aq: torch.Tensor, ak: torch.Tensor, M: int) -> torch.Tensor:
    """tau [B,R,T] long on aq.device. Uses the fast numba online-SAM kernel (O(R*T))
    when available — the only thing that makes ROSA runnable at scale — and falls back
    to the naive O(R*T^2) reference otherwise (small/CPU-less only). The SAM round-trips
    through CPU (D2H -> retrieve -> H2D); the paper overlaps this with GPU attention."""
    if _HAVE_SAM:
        tau_np = _sam_retrieve(aq.detach().cpu().numpy(), ak.detach().cpu().numpy(), 1 << M)
        return torch.from_numpy(tau_np).to(aq.device)
    return retrieve_routes(aq, ak)


def _pack_bits(bits: torch.Tensor, M: int) -> torch.Tensor:
    """bits: [B, T, C] in {0,1}. -> symbols [B, T, R] with R=C//M, each route's M
    bits packed little-endian into an integer in {0..2^M-1}  (Eqs. 13-15)."""
    B, T, C = bits.shape
    R = C // M
    # Sum in an exact integer dtype: activation dtypes like bf16 cannot represent
    # all integers > 256, which would corrupt packed symbols for M >= 9.
    w = 2 ** torch.arange(M, device=bits.device, dtype=torch.long)      # [M]
    return (bits.view(B, T, R, M).long() * w).sum(-1)                   # [B, T, R]


class _RosaRetrieve(torch.autograd.Function):
    """Discrete forward (binarize -> pack -> retrieve -> read value -> unpack to
    bits) with the counterfactual VALUE gradient (Eq. 24). Returns the read bit
    pattern `bits` [B,T,C] (float 0/1) and the validity mask `m` [B,T,1]. The
    learned affine (e0,e1,W_out) is applied OUTSIDE so autograd handles it exactly."""

    @staticmethod
    def forward(ctx, q_vec, k_vec, v_vec, M, max_match, needs_grad=False):
        B, T, C = q_vec.shape
        R = C // M
        qb = (q_vec > 0).to(q_vec.dtype)
        kb = (k_vec > 0).to(k_vec.dtype)
        vb = (v_vec > 0).to(v_vec.dtype)
        aq = _pack_bits(qb, M)
        ak = _pack_bits(kb, M)
        av = _pack_bits(vb, M)                                  # [B,T,R] value symbols
        if q_vec.is_cuda and _HAVE_CUDA_SAM:
            # Device-native online SAM; exact CPU parity is covered against the cited ROSA
            # construction (arXiv:2602.02499) in the regression suite.
            tau, tau0, tau1 = _cuda_sam_retrieve_cf(aq, ak, 1 << M, M)
        elif _HAVE_SAM:                                         # forward + Eq.25 counterfactual tables
            t_np, t0_np, t1_np = _sam_retrieve_cf(aq.detach().cpu().numpy(),
                                                  ak.detach().cpu().numpy(), 1 << M, M)
            tau = torch.from_numpy(t_np).to(q_vec.device)
            tau0 = torch.from_numpy(t0_np).to(q_vec.device)    # [B,R,T,M] dest if query bit j = 0
            tau1 = torch.from_numpy(t1_np).to(q_vec.device)    # [B,R,T,M] dest if query bit j = 1
        else:
            # The naive matcher has no counterfactual (Eq. 25) tables: recomputing the
            # retrieval with each query bit flipped would be another O(M*R*T^2) pass.
            # Refuse silently-zero W_q gradients in training; allow inference only.
            # (needs_grad is computed by the caller: inside Function.forward grad mode
            # is off and inputs are detached, so requires_grad can't be read here.)
            if needs_grad:
                raise RuntimeError(
                    "ROSA fallback matcher cannot provide counterfactual W_q gradients "
                    "(Eq. 25); install numba so the SAM kernel (rosa_sam) is available, "
                    "or run under torch.no_grad() for inference.")
            global _WARNED_FALLBACK
            if not _WARNED_FALLBACK:
                _WARNED_FALLBACK = True
                import warnings
                warnings.warn("ROSA: numba SAM kernel unavailable; using the naive "
                              "O(T^2) matcher (forward-only, no counterfactual "
                              "gradients — NOT equivalent to the SAM for training).")
            tau = retrieve_routes(aq, ak)
            tau0 = tau.new_full((B, R, T, M), -1)              # inference-only: Q grad path unused
            tau1 = tau.new_full((B, R, T, M), -1)
        m = (tau >= 0)                                          # [B,R,T]
        av_rt = torch.gather(av.transpose(1, 2), 2, tau.clamp(min=0)) * m   # a^(v) at tau
        shifts = torch.arange(M, device=q_vec.device)
        bits = ((av_rt.unsqueeze(-1) >> shifts) & 1).to(q_vec.dtype)        # [B,R,T,M]
        bits = bits.permute(0, 2, 1, 3).reshape(B, T, C)
        mC = m.transpose(1, 2).reshape(B, T, R, 1).expand(B, T, R, M).reshape(B, T, C).to(q_vec.dtype)
        ctx.save_for_backward(q_vec, v_vec, tau, tau0, tau1)
        ctx.dims = (B, T, C, R, M)
        return bits, mC

    @staticmethod
    def backward(ctx, g_bits, g_m):
        # g_bits = dL/d(bits) = G^y * m * Delta  (= theta in the paper, already masked).
        q_vec, v_vec, tau, tau0, tau1 = ctx.saved_tensors
        B, T, C, R, M = ctx.dims
        sig_v = torch.sigmoid(v_vec)
        # --- V gradient (Eq. 24): destination scatter to v^vec at tau ---
        gv = torch.zeros_like(v_vec)
        src_c = tau.transpose(1, 2).repeat_interleave(M, dim=2)             # [B,T,C] source pos/channel
        valid = (src_c >= 0)
        gv.scatter_add_(1, src_c.clamp(min=0), torch.where(valid, g_bits, torch.zeros_like(g_bits)))
        gv = gv * (sig_v * (1 - sig_v))
        # --- Q gradient (Eq. 25): counterfactual query-bit differencing ---
        # dL/dq^vec[t,(r,j)] = sigma'(q) * sum_m g_bits[t,(r,m)] * (Pv[tau1_j] - Pv[tau0_j])[r,m]
        Pv = sig_v.reshape(B, T, R, M).permute(0, 2, 1, 3).contiguous()     # [B,R,T,M_m]

        def _gather_cf(tau_cf):                                # [B,R,T,M_j] -> [B,R,T,M_j,M_m]
            ok = (tau_cf >= 0)
            idx = tau_cf.clamp(min=0).reshape(B, R, T * M, 1).expand(B, R, T * M, M)
            g = torch.gather(Pv, 2, idx).reshape(B, R, T, M, M)
            return g * ok.reshape(B, R, T, M, 1).to(g.dtype)

        delta = _gather_cf(tau1) - _gather_cf(tau0)                          # [B,R,T,M_j,M_m]
        g_bits_r = g_bits.reshape(B, T, R, M).permute(0, 2, 1, 3)            # [B,R,T,M_m]
        contrib = torch.einsum('brtm,brtjm->brtj', g_bits_r, delta)         # [B,R,T,M_j]
        contrib = contrib.permute(0, 2, 1, 3).reshape(B, T, C)              # [B,T,C]
        sig_q = torch.sigmoid(q_vec)
        gq = contrib * (sig_q * (1 - sig_q))
        # K gradient (Eq. 26) deferred: run-level counterfactual mutates the SAM globally (App. C.6)
        return gq, None, gv, None, None, None


class RosaLayer(nn.Module):
    """Per-decoder-block ROSA module (no rank, no resample, no merge — those are
    LoRA concepts that do NOT exist in ROSA). M=4 default => alphabet 2^4=16 (§5.1)."""

    def __init__(self, hidden_size: int, M: int = 4, mode: str = "post", max_match: int = 64):
        super().__init__()
        assert hidden_size % M == 0, "hidden_size must be divisible by route width M"
        assert mode in ("post", "pre")
        C = hidden_size
        self.M, self.mode, self.max_match = M, mode, max_match
        self.norm = nn.RMSNorm(C)                               # U = LN(H)            (Eq. 8)
        self.Wq = nn.Linear(C, C, bias=False)                  #                       (Eq. 9)
        self.Wk = nn.Linear(C, C, bias=False)
        self.Wv = nn.Linear(C, C, bias=False)
        self.Wout = nn.Linear(C, C, bias=False)                #                       (Eq. 22)
        self.e0 = nn.Parameter(torch.zeros(C))                 # init 0 -> inj==0      (§3.4)
        self.e1 = nn.Parameter(torch.zeros(C))                 # init 0
        nn.init.eye_(self.Wout.weight)                         # W_out = I (clean start)
        if mode == "pre":
            self.alpha0 = nn.Parameter(torch.full((C,), -4.0))  # a=sigmoid(a0) small
        # grokking diagnostics (default off): injection() stashes its RMS (relative
        # to H) and the affine spread |e1-e0| so a trainer can chart ROSA "grokking
        # on" — the recall path is a no-op until these grow. See grokking_metrics.py.
        self.log_grok = False
        self.last_stats: dict = {}

    def injection(self, H: torch.Tensor) -> torch.Tensor:
        """inj = ROSA(H), shape [B,T,C]. Exact no-op while e0==e1 (start of training)."""
        U = self.norm(H)
        qv, kv, vv = self.Wq(U), self.Wk(U), self.Wv(U)
        # Only the counterfactual W_q gradient (Eq. 25, via tau0/tau1) is
        # missing on the fallback path; value-path grads (e0/e1/Wout/Wv) are
        # exact there. Gate the fallback's hard error on qv alone so
        # value-only training still runs without numba.
        needs_grad = torch.is_grad_enabled() and qv.requires_grad
        bits, mC = _RosaRetrieve.apply(qv, kv, vv, self.M, self.max_match, needs_grad)
        delta = self.e1 - self.e0
        y = mC * (self.e0 + delta * bits)                      # affine readout       (Eqs. 20-21)
        out = self.Wout(y)                                     #                       (Eq. 22)
        if self.log_grok:
            from . import grokking_metrics as gm
            self.last_stats = {"rosa_inj_rms": gm.injection_rms(out, H),
                               "rosa_e_gap": gm.affine_gap(self.e0, self.e1)}
        return out

    def forward(self, H: torch.Tensor, attn) -> torch.Tensor:
        """`attn` is a callable: attn(x)->[B,T,C], the block's windowed attention on a
        normalized input. We pass LN(H) (the block's own norm should be used in real
        wiring; here we reuse self.norm for the standalone reference)."""
        if self.mode == "post":                                # additive            (Eqs. 1-3)
            A = attn(self.norm(H))
            return H + A + self.injection(H)
        a = torch.sigmoid(self.alpha0)                         # time-mix            (Eqs. 4-7)
        inj = self.injection(H)
        M = (1 - a) * H + a * inj
        return H + attn(self.norm(M))
    # NOTE: deliberately NO resample()/merge(): ROSA's "periodic" op is the online
    # SAM build inside retrieve_routes (every forward), not an every-N-steps refresh,
    # and the data-dependent SAM lookup cannot be folded into a weight matrix.


# ---------------------------------------------------------------------------
# Drop-in integration. To TURN ROSA ON in any decoder-stack trainer:
#     rosa = attach_rosa(model, ATTN_LAYERS, hidden_size, device=dev, dtype=dt)
#     model.rosa = rosa                          # keep a reference (so .to()/save see it)
#     opt = Optimizer([*base_params, *rosa_parameters(rosa)], ...)
#     # ... save/load rosa.state_dict() alongside the run ckpt ...
# Because e0=e1=0 at init, the model is BIT-FOR-BIT unchanged until ROSA trains.
#
# PLACEMENT: attach to the windowed/softmax-ATTENTION layers (here 3,7,11,...,31),
# NOT the RWKV-converted ones (RWKV-7 already carries global state). It makes those
# layers trainable, so it does NOT belong in the per-layer ISOLATION convert_train
# (frozen backbone) — use it in a full-model / consolidation / long-context stage.
# Run it alongside Engram as a SECOND additive injection (the paper says they're
# complementary: Engram=pretrained-knowledge table, ROSA=in-context history).
# ---------------------------------------------------------------------------
def attach_rosa(model, layer_indices, hidden_size, *, M=4, mode="post",
                max_match=64, resolve="model.layers", device=None, dtype=None):
    """Install a RosaLayer on each decoder layer in `layer_indices` via a forward hook
    that ADDS rosa.injection(layer_input) to that layer's output hidden states. Returns
    an nn.ModuleDict {str(idx): RosaLayer}. No-op at init (inj==0 while e0==e1==0).

    The hook adds inj to the layer OUTPUT (after attn+MLP); the paper's exact spot is
    after attn, before MLP (Eqs.1-3) — fold into the block forward if you want that.

    Only mode="post" is supported here: the pre-attn time-mix (Eqs. 4-7, RosaLayer.forward
    with mode="pre") must rewrite the ATTENTION INPUT, which a forward (output) hook cannot
    do — fold RosaLayer into the block forward for that."""
    if mode != "post":
        raise NotImplementedError(
            "attach_rosa only supports mode='post'; mode='pre' needs the alpha0 time-mix "
            "applied to the attention input (RosaLayer.forward), which cannot be done from "
            "a forward hook — integrate RosaLayer into the block forward instead.")
    import torch.nn as _nn
    mod = model
    for part in resolve.split("."):
        mod = getattr(mod, part)
    layers = mod
    rosa = _nn.ModuleDict()
    for i in layer_indices:
        rl = RosaLayer(hidden_size, M=M, mode=mode, max_match=max_match)
        if device is not None or dtype is not None:
            rl = rl.to(device=device, dtype=dtype)
        rosa[str(i)] = rl

        def make_hook(rl):
            def hook(module, args, kwargs, output):
                H = args[0] if args else kwargs.get("hidden_states")
                inj = rl.injection(H)
                if isinstance(output, tuple):
                    return (output[0] + inj,) + tuple(output[1:])
                return output + inj
            return hook
        layers[int(i)].register_forward_hook(make_hook(rl), with_kwargs=True)
    return rosa


def rosa_parameters(rosa):
    """All trainable ROSA params (pass to the optimizer). `rosa` = attach_rosa's ModuleDict."""
    return [p for p in rosa.parameters() if p.requires_grad]


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, C, M = 2, 24, 32, 4
    H = torch.randn(B, T, C, requires_grad=True)
    rosa = RosaLayer(C, M=M, mode="post", max_match=16)
    # init-to-identity: inj must be exactly 0 while e0==e1==0
    inj0 = rosa.injection(H)
    assert inj0.abs().max().item() == 0.0, "ROSA must be an exact no-op at init"
    print("[ok] init-to-identity: inj == 0")
    # activate the readout, check a gradient flows to e0/e1/W_out/W_v
    with torch.no_grad():
        rosa.e1 += 1.0
    inj = rosa.injection(H)
    loss = inj.pow(2).mean()
    loss.backward()
    gnz = lambda p: (p is not None) and p.grad is not None and p.grad.abs().sum().item() > 0
    print(f"[ok] inj active: |inj|max={inj.abs().max().item():.4f}  loss={loss.item():.4f}")
    print(f"     grads -> e0:{gnz(rosa.e0)}  e1:{gnz(rosa.e1)}  "
          f"Wout:{gnz(rosa.Wout.weight)}  Wv:{gnz(rosa.Wv.weight)}  "
          f"Wq:{gnz(rosa.Wq.weight)}(Eq.25)  Wk:{gnz(rosa.Wk.weight)}(K=Eq.26 TODO)  H:{H.grad is not None}")
    print("[ok] backward ran (e0/e1/Wout/Wv/Wq train via Eqs.24-25; Wk frozen — K=Eq.26 deferred)")

    # --- attach_rosa: drop-in onto a mock decoder stack, prove no-op-at-init ---
    class _MockLayer(nn.Module):
        def __init__(self, C): super().__init__(); self.lin = nn.Linear(C, C)
        def forward(self, hidden_states): return (hidden_states,)   # identity, tuple out
    class _MockBackbone(nn.Module):
        def __init__(self, C, n): super().__init__(); self.layers = nn.ModuleList(_MockLayer(C) for _ in range(n))
    class _MockModel(nn.Module):
        def __init__(self, C, n): super().__init__(); self.model = _MockBackbone(C, n)
        def forward(self, h):
            for L in self.model.layers: h = L(h)[0]
            return h
    torch.manual_seed(0)
    mm = _MockModel(C, 4)
    h = torch.randn(B, T, C)
    base = mm(h).clone()
    rd = attach_rosa(mm, [1, 3], C, M=M, mode="post", max_match=16)
    assert torch.allclose(mm(h), base), "attach_rosa must be a no-op at init"
    print(f"[ok] attach_rosa: stack output UNCHANGED at init over layers {list(rd.keys())}")
    for rl in rd.values():
        with torch.no_grad(): rl.e1 += 1.0
    delta = (mm(h) - base).abs().max().item()
    print(f"[ok] attach_rosa: output changes after activation, max|delta|={delta:.4f}")
    print(f"[ok] rosa_parameters: {sum(p.numel() for p in rosa_parameters(rd))} trainable params "
          f"over {len(rd)} layers — pass these to the optimizer to turn ROSA ON")
