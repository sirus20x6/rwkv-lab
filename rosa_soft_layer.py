"""ROSA-soft — RosaLayer's retrieval mechanism swapped for research/rosa_soft's
`rosa_anchor_ops`, a CUDA `torch.autograd.Function` that analytically differentiates
through a softmax-relaxed suffix-match (weighted bit-mismatch -> multiplicative
suffix-score recurrence -> softmax over candidates + a null/sink candidate ->
output = softmax-weighted average of sign(v)). Unlike rosa.py's `_RosaRetrieve`
(hard-SAM lookup + Eq.24/25 counterfactual-gradient trick, K frozen), gradients here
flow analytically through Q AND K.

Shape mapping is exact for Q/K: rosa_anchor_ops' `(B,T,H,D)` convention IS
rosa.py's own `[B,T,C] -> [B,T,R,M]` route-packing (R routes == H heads,
M bits/route == D). V may use fewer grouped value heads, matching the reference
operator's `H % H_v == 0` contract; the op still returns `[B,T,R,M]` when
`value_dim == M`, so the readout shape remains `[B,T,C]`.
Raw float Q/K/V projections are passed straight in (no `>0` binarization first) —
the op needs magnitude for its confidence weighting and binarizes internally.

Readout reuses RosaLayer's own affine (e0, e1, Wout): the op's output s in [-1,1]
(a softmax-weighted average of +-1 values) is remapped via (s+1)/2 into the same
`e0 + (e1-e0)*bits` formula RosaLayer uses for hard bits in {0,1}, so e0=e1=0 still
gives an exact no-op at init regardless of s. No mode="pre"/alpha0 machinery and no
.forward(H, attn) wrapper -- the only consumer is a forward hook that calls
.injection(H) directly (mirrors how attach_rosa uses RosaLayer in rosa.py).

CUDA-only (rosa_anchor_ops requires CUDA tensors). Requires `research/rosa_soft`
installed: `pip install -e research/rosa_soft`.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from rosa_soft import rosa_anchor_ops


class RosaAnchorLayer(nn.Module):
    """Per-decoder-block ROSA-soft module. M=4 default matches rosa.py's RosaLayer."""

    def __init__(
        self,
        hidden_size: int,
        M: int = 4,
        window_size: int = 32,
        scale: float | None = None,
        value_heads: int | None = None,
        logit_epsilon: float = 0.0,
        qk_damper_strength: float = 0.0,
        route_dim: int | None = None,
    ):
        super().__init__()
        assert 1 <= M <= 32, "rosa_anchor_ops requires the per-route bit dim in [1, 32]"
        C = hidden_size
        # Low-rank retrieval subspace: route a d-dim Q/K projection instead of the full hidden.
        # rosa_anchor_ops' CUDA kernel launches B*T*R_query blocks (R_query = d/M query routes),
        # so cost is LINEAR in d -- d << C is the throughput lever, orthogonal to window (reach)
        # and to value_heads (which only groups the V projection). d=None/0 -> full width (R=C/M),
        # the rosa_soft package's own convention (num_q_heads = C/qk_bits).
        d = int(route_dim) if route_dim else C
        assert d % M == 0, "route_dim (or hidden_size when route_dim is None) must be divisible by M"
        R = d // M                               # query routes == block-count driver (B*T*R)
        if value_heads is None:
            value_heads = R
        assert 1 <= value_heads <= R, "value_heads must be in [1, route_dim // M]"
        assert R % value_heads == 0, "route head count must be divisible by value_heads"
        assert logit_epsilon >= 0.0, "logit_epsilon must be non-negative"
        assert 0.0 <= qk_damper_strength <= 1.0, "qk_damper_strength must be in [0, 1]"
        self.M = M
        self.window_size = window_size
        self.scale = scale
        self.value_heads = value_heads
        self.logit_epsilon = logit_epsilon
        self.qk_damper_strength = qk_damper_strength
        self.C = C
        self.d = d
        self.norm = nn.RMSNorm(C)
        self.Wq = nn.Linear(C, d, bias=False)
        self.Wk = nn.Linear(C, d, bias=False)
        self.Wv = nn.Linear(C, value_heads * M, bias=False)
        self.Wout = nn.Linear(d, C, bias=False)
        self.e0 = nn.Parameter(torch.zeros(d))   # init 0 -> inj==0 (in the routed subspace)
        self.e1 = nn.Parameter(torch.zeros(d))   # init 0
        # telemetry probe channel (plain attrs, NOT buffers -- state_dict/warm-start shape
        # must stay unchanged). Trainer sets want_telemetry (one-shot); the next injection()
        # also captures AttentionTelemetry into last_telemetry for the reference recipe's
        # RosaAnchorScaleController (top_prob-targeted scale calibration) + train.jsonl.
        self.want_telemetry = False
        self.last_telemetry = None
        # NOTE: `scale` above is also a plain attr (not a buffer) for the same reason;
        # convert_train persists it separately (blob["rosa_soft_scale"]) so a calibrated
        # retrieval temperature survives warm-restarts.
        if d == C:
            nn.init.eye_(self.Wout.weight)       # W_out = I clean start (square, full-width path)
        # else: keep default (nonzero) init so gradients still reach e0/e1; y==0 at init (e0==e1)
        # already gives an exact no-op regardless of Wout, so zero-init is unnecessary and would
        # starve e0/e1 of gradient.

    def float_growth_params(self):
        """Keep e0/e1 in fp32 after a module-wide .to(dtype=bf16). They grow from
        zero by small optimizer steps that bf16's ~3 significant digits quantize
        away once the params have magnitude (same fix as LoopedRWKV.float_gates).
        injection() computes the affine in e0's dtype and casts back to the stream
        dtype before Wout, so the residual stream is unaffected."""
        self.e0.data = self.e0.data.float()
        self.e1.data = self.e1.data.float()
        return self

    def injection(self, H: torch.Tensor) -> torch.Tensor:
        """inj = ROSA-soft(H), shape [B,T,C]. Exact no-op while e0==e1 (start of training)."""
        B, T, C = H.shape
        R = self.d // self.M
        U = self.norm(H)
        q = self.Wq(U).view(B, T, R, self.M)
        k = self.Wk(U).view(B, T, R, self.M)
        v = self.Wv(U).view(B, T, self.value_heads, self.M)
        probe = self.want_telemetry
        self.want_telemetry = False
        s = rosa_anchor_ops(
            q,
            k,
            v,
            window_size=self.window_size,
            scale=self.scale,
            return_telemetry=probe,
            logit_epsilon=self.logit_epsilon,
            qk_damper_strength=self.qk_damper_strength,
        )  # [B,T,R,M] in [-1,1]
        if probe:
            s, self.last_telemetry = s
        s = s.reshape(B, T, self.d)
        delta = self.e1 - self.e0
        # bit=0 -> e0, bit=1 -> e1, continuous between; [B,T,d]. Computed in e0's dtype
        # (fp32 under float_growth_params) then cast back so Wout sees the stream dtype.
        y = self.e0 + delta * (s.to(self.e0.dtype) + 1) * 0.5
        return self.Wout(y.to(s.dtype))           # [B,T,d] -> [B,T,C]


def rosa_anchor_parameters(rosa):
    return [p for p in rosa.parameters() if p.requires_grad]


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda"
    B, T, C, M = 2, 48, 256, 4
    rosa = RosaAnchorLayer(C, M=M, window_size=32).to(device)
    H = torch.randn(B, T, C, device=device)

    inj0 = rosa.injection(H)
    print(f"init-to-identity: inj0.abs().max()={inj0.abs().max().item():.3e}")
    assert inj0.abs().max().item() == 0.0, "ROSA-soft must be an exact no-op at init"

    with torch.no_grad():
        rosa.e1 += 1.0
    inj1 = rosa.injection(H)
    loss = inj1.pow(2).sum()
    loss.backward()

    def gnz(p):
        return "None" if p.grad is None else f"{p.grad.norm().item():.3e}"

    print(f"after e1+=1: inj1.abs().max()={inj1.abs().max().item():.3e}")
    print(f"grad norms: e0={gnz(rosa.e0)} e1={gnz(rosa.e1)} Wout={gnz(rosa.Wout.weight)} "
          f"Wv={gnz(rosa.Wv.weight)} Wq={gnz(rosa.Wq.weight)} Wk={gnz(rosa.Wk.weight)}")
    for name, p in [("e0", rosa.e0), ("e1", rosa.e1), ("Wout", rosa.Wout.weight),
                    ("Wv", rosa.Wv.weight), ("Wq", rosa.Wq.weight), ("Wk", rosa.Wk.weight)]:
        assert p.grad is not None and p.grad.norm().item() > 0, f"{name} did not receive gradient"
    print("OK: all of e0,e1,Wout,Wq,Wk,Wv receive gradient (Wk trains here, unlike rosa.py v1)")

    rosa.want_telemetry = True
    inj2 = rosa.injection(H)
    assert inj2.shape == inj1.shape
    assert not rosa.want_telemetry and rosa.last_telemetry is not None, "telemetry probe did not fire"
    tf = rosa.last_telemetry.as_float_dict()
    print(f"telemetry probe: top_prob={tf['top_prob']:.3f} null_prob={tf['null_prob']:.3f} "
          f"entropy={tf['entropy_norm']:.3f} trunc={tf.get('truncated_fraction', float('nan')):.3f}")
    rosa.last_telemetry = None
    rosa.injection(H)  # one-shot: next call must NOT probe
    assert rosa.last_telemetry is None, "telemetry probed again without want_telemetry"
    print("OK: telemetry is one-shot (want_telemetry auto-clears)")

    # fp32 growth params under a bf16 stream (the trainer path: .to(bf16) then float)
    rb = RosaAnchorLayer(C, M=M, window_size=32).to(device, torch.bfloat16).float_growth_params()
    assert rb.e0.dtype == rb.e1.dtype == torch.float32
    Hb = H.to(torch.bfloat16)
    ib = rb.injection(Hb)
    assert ib.dtype == torch.bfloat16 and ib.abs().max().item() == 0.0, "fp32 e0/e1 broke the init no-op"
    with torch.no_grad():
        rb.e1 += 0.5
    rb.injection(Hb).float().pow(2).sum().backward()
    assert rb.e0.grad is not None and rb.e0.grad.dtype == torch.float32 and rb.e0.grad.norm() > 0
    print("OK: float_growth_params keeps e0/e1 fp32, bf16 stream + grads intact")
