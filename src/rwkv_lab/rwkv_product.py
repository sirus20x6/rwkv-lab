"""RWKV-Product (ChainGPT, arXiv:2606.16167-adjacent registry paper) — a drop-in RWKV-7 core
that performs M low-rank delta-rule SUB-STEPS per token inside a single wkv7 kernel call.

The transition becomes an ordered product of M "diagonal − rank-1" delta operators, giving an
effective rank-M state update per token (F_1 ⊊ F_M for M≥2), at the cost of only a few LoRA
pairs per sub-step (~0.1M params for M=2). Mechanism (paper Appendix D.1): build M sets of
low-rank (r, gk, k, v, a, b), INTERLEAVE them on the time axis (length T·M), run the standard
gated-delta wkv7 kernel ONCE, then keep every M-th (last) output.

This is a drop-in replacement for RWKV8TimeMixDeltaNet's forward (same y = mixer(hidden)), so
LoopedRWKV wraps it unchanged. M=1 reduces to an ordinary single-sub-step RWKV-7 mixer (the
interleave is a no-op), which the test pins against a direct one-shot kernel call.

Three ambiguities the paper leaves open, resolved here (documented so they can be revisited):
  (a) DECAY on intermediate sub-steps. Paper says w=−∞ for j>0. In OUR kernel gk is LOG-decay
      (decay = exp(gk)), so "no decay" is gk = 0 (decay 1), NOT −∞ (which would WIPE the state).
      => first sub-step carries the token's real gk; sub-steps j>0 use gk = 0.
  (b) value-LoRA index. Main-text eq 7 uses A^v_j; D.1 writes A^v_{j-1}. We follow eq 7 (index j).
  (c) k_ac / rk. Undefined dims in the paper; we use the base RWKV-7 semantics — a learned per-
      channel k_a gate on the c-key and a learned per-head r_k for the current-token bonus.

SCOPE: full-sequence training/eval path only (return_state / SMT-DMT rollout not supported for
the interleaved recurrence yet — raises NotImplementedError, mirroring the base core's cache
guard). LoRA B-matrices are zero-init so each sub-step starts from the shared base projection.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rwkv8_deltanet import _rwkv7_python_ref, _fla_chunk_rwkv7, _HAS_FLA
import os


def _wkv7(r, gk, k, v, a, b):
    """[B,T,H,N] gated-delta kernel, matching RWKV8TimeMixDeltaNet._wkv7 (gk = log-decay)."""
    if _HAS_FLA and os.environ.get("RWKV8_FORCE_PYREF") != "1":
        out, _ = _fla_chunk_rwkv7(r, gk, k, v, a, b, scale=1.0,
                                  initial_state=None, output_final_state=False)
        return out
    out, _ = _rwkv7_python_ref(r, gk, k, v, a, b, initial_state=None)
    return out


class RWKVProduct(nn.Module):
    def __init__(self, hidden_size: int, *, num_heads: int, head_size: int,
                 M: int = 2, lora_rank: int = 32, beta_lora: int = 16,
                 decay_lora: int = 8, gate_lora: int = 8, layer_idx: Optional[int] = None):
        super().__init__()
        if num_heads * head_size != hidden_size:
            raise ValueError("num_heads*head_size must equal hidden_size")
        C, H, N = hidden_size, num_heads, head_size
        self.hidden_size, self.num_heads, self.head_size = C, H, N
        self.M = int(M)
        self.layer_idx = layer_idx

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        for nm in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g"):
            self.register_parameter(nm, nn.Parameter(torch.ones(1, 1, C) * 0.5))
        self.receptance = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)
        nn.init.zeros_(self.output.weight)                       # near-identity swap at step 0

        self.w0 = nn.Parameter(torch.zeros(1, 1, C) + 0.5)       # decay bias (data-indep at init)
        self.w1 = nn.Parameter(torch.zeros(C, decay_lora))       # w1=0 => decay data-independent
        self.w2 = nn.Parameter(torch.randn(decay_lora, C) * (0.1 / math.sqrt(decay_lora)))
        self.g1 = nn.Parameter(torch.zeros(C, gate_lora))        # g1=0 => constant gate at init
        self.g2 = nn.Parameter(torch.randn(gate_lora, C) * (1.0 / math.sqrt(gate_lora)))  # active gate
        self.k_k = nn.Parameter(torch.ones(1, 1, C) * 0.71)
        self.k_a = nn.Parameter(torch.ones(1, 1, C))
        self.r_k = nn.Parameter(torch.zeros(H, N))
        self.ln_x = nn.GroupNorm(H, C, eps=64e-5)

        # per-sub-step LoRA (b-key, c-key, value) + per-sub-step beta gates (b and c paths)
        r_ = lora_rank
        self.A_b = nn.ParameterList([nn.Parameter(torch.zeros(C, r_)) for _ in range(M)])
        self.B_b = nn.ParameterList([nn.Parameter(torch.zeros(r_, C)) for _ in range(M)])
        self.A_c = nn.ParameterList([nn.Parameter(torch.zeros(C, r_)) for _ in range(M)])
        self.B_c = nn.ParameterList([nn.Parameter(torch.zeros(r_, C)) for _ in range(M)])
        self.A_v = nn.ParameterList([nn.Parameter(torch.zeros(C, r_)) for _ in range(M)])
        self.B_v = nn.ParameterList([nn.Parameter(torch.zeros(r_, C)) for _ in range(M)])
        for A in list(self.A_b) + list(self.A_c) + list(self.A_v):
            nn.init.normal_(A, std=1.0 / math.sqrt(C))           # A random, B zero => zero increment
        self.beta0_b = nn.ParameterList([nn.Parameter(torch.zeros(1, 1, C)) for _ in range(M)])
        self.beta0_c = nn.ParameterList([nn.Parameter(torch.zeros(1, 1, C)) for _ in range(M)])
        self.beta1_b = nn.ParameterList([nn.Parameter(torch.zeros(C, beta_lora)) for _ in range(M)])
        self.beta2_b = nn.ParameterList([nn.Parameter(torch.zeros(beta_lora, C)) for _ in range(M)])
        self.beta1_c = nn.ParameterList([nn.Parameter(torch.zeros(C, beta_lora)) for _ in range(M)])
        self.beta2_c = nn.ParameterList([nn.Parameter(torch.zeros(beta_lora, C)) for _ in range(M)])

    def forward(self, hidden_states, *args, return_state: bool = False,
                shift_state=None, initial_state=None, **kwargs):
        if return_state or initial_state is not None or shift_state is not None:
            # Chunked use would need BOTH the token-shift carry AND the interleaved WKV matrix
            # state threaded across the boundary; only full-sequence training is supported, so
            # reject all three rather than silently drop the recurrent state (audit finding).
            raise NotImplementedError("RWKVProduct supports full-sequence training only "
                                      "(no return_state / initial_state / shift_state / SMT-DMT yet)")
        x = hidden_states
        B, T, C = x.shape
        H, N, M = self.num_heads, self.head_size, self.M
        xx = self.time_shift(x) - x

        def mix(p):
            return x + xx * p
        xr, xw, xk, xv, xa, xg = (mix(self.x_r), mix(self.x_w), mix(self.x_k),
                                  mix(self.x_v), mix(self.x_a), mix(self.x_g))
        R = self.receptance(xr)
        W = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5
        gk_tok = (-torch.exp(W))                                 # token log-decay (first sub-step)
        K0, V0 = self.key(xk), self.value(xv)
        zero_gk = torch.zeros_like(gk_tok)                       # no-decay for sub-steps j>0

        r_s, gk_s, k_s, v_s, a_s, b_s = [], [], [], [], [], []
        R_last = k_last = v_last = None
        for j in range(M):
            kb = K0 + (xk @ self.A_b[j]) @ self.B_b[j]
            kc = K0 + (xk @ self.A_c[j]) @ self.B_c[j]
            vj = V0 + (xv @ self.A_v[j]) @ self.B_v[j]
            beta_b = torch.sigmoid(self.beta0_b[j] + (xa @ self.beta1_b[j]) @ self.beta2_b[j])
            beta_c = torch.sigmoid(self.beta0_c[j] + (xa @ self.beta1_c[j]) @ self.beta2_c[j])
            # normalize in fp32: default eps underflows in fp16 and an all-zero head key -> NaN
            kb_n = F.normalize((kb * self.k_k).float().view(B, T, H, N), dim=-1, p=2.0,
                               eps=1e-6).to(kb.dtype).view(B, T, C)
            kc_hat = kc * (1 + (beta_c - 1) * self.k_a)          # gated c-key (k_a analog)
            r_j = R if j == M - 1 else torch.zeros_like(R)       # read only last sub-step
            gk_j = gk_tok if j == 0 else zero_gk                 # decay only first sub-step (amb. a)
            r_s.append(r_j.view(B, T, H, N)); gk_s.append(gk_j.view(B, T, H, N))
            k_s.append(kc_hat.view(B, T, H, N)); v_s.append(vj.view(B, T, H, N))
            a_s.append((-kb_n * beta_b).view(B, T, H, N)); b_s.append(kb_n.view(B, T, H, N))
            if j == M - 1:
                R_last, k_last, v_last = R.view(B, T, H, N), kc_hat.view(B, T, H, N), vj.view(B, T, H, N)

        def interleave(lst):                                    # [M]x[B,T,H,N] -> [B,T*M,H,N]
            return torch.stack(lst, dim=2).reshape(B, T * M, H, N)
        Xi = _wkv7(interleave(r_s), interleave(gk_s), interleave(k_s),
                   interleave(v_s), interleave(a_s), interleave(b_s))   # [B,T*M,H,N]
        Xatt = Xi.reshape(B, T, M, H, N)[:, :, M - 1]           # keep last sub-step -> [B,T,H,N]
        Xgn = self.ln_x(Xatt.reshape(B * T, C)).view(B, T, C)
        bonus = ((R_last * k_last * self.r_k.view(1, 1, H, N)).sum(-1, keepdim=True)
                 * v_last).reshape(B, T, C)                      # RWKV-7 current-token term
        g = torch.sigmoid(xg @ self.g1) @ self.g2
        return self.output((Xgn + bonus) * g)
