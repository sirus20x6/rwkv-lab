"""
RWKV-8-style replacements for Qwen3.5/Qwen3-Next DeltaNet linear-attention
layers. Two flavors live here, both as drop-in replacements for the
``linear_attn`` slot on a decoder layer:

* ``RWKV8ChannelMixDeltaNet``: the cheap ChannelMix (FFN-like) stand-in,
  ``y_t = relu((x_t + (x_{t-1} - x_t)*x_k) W_k)^2 W_v``. Smallest-possible
  experiment, kept as a legacy mode.

* ``RWKV8TimeMixDeltaNet``: a faithful port of BlinkDL's RWKV-7/v8
  ``RWKV_Tmix_x070`` time-mix block. Uses ``fla.ops.rwkv7.chunk_rwkv7``
  (Triton) when available, falls back to a slow recurrent Python reference
  when ``RWKV8_FORCE_PYREF=1`` is set or fla is missing. The output linear
  is zero-initialised so a single-layer swap is near-identity at step 0.

The DeepEmbed multiplier used in BlinkDL's demo needs token ids at the mixer
call site, but HF's ``linear_attn(...)`` forward does not receive
``input_ids``. Keep that as a later wrapper-level extension; both modules
here only need ``hidden_states``.
"""

from __future__ import annotations

import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# fla 0.4.x ships a Triton implementation of RWKV-7's wkv7 kernel under
# ``fla.ops.rwkv7.chunk_rwkv7``. Signature (verified at v0.4.1):
#     chunk_rwkv7(r, w, k, v, a, b,
#                 scale=1.0, initial_state=None, output_final_state=True,
#                 cu_seqlens=None, head_first=False) -> (out, final_state)
# All inputs are (B, T, H, K) when head_first=False. We discard final_state.
try:
    from fla.ops.rwkv7 import chunk_rwkv7 as _fla_chunk_rwkv7  # type: ignore
    _HAS_FLA = True
    _FLA_IMPORT_ERROR: Exception | None = None
except Exception as _e:  # pragma: no cover - import-time fallback
    _fla_chunk_rwkv7 = None
    _HAS_FLA = False
    _FLA_IMPORT_ERROR = _e

_PYREF_WARNED = False  # warn ONCE at first pyref dispatch, not at import (tools may never forward)


class RWKV8ChannelMixDeltaNet(nn.Module):
    """A tensor-returning replacement for Qwen3.5/Qwen3-Next `linear_attn`.

    The output projection is initialized small so swapping a single layer starts
    close to the residual-only path instead of injecting a large random mixer.
    """

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: Optional[int] = None,
        *,
        layer_idx: Optional[int] = None,
        initializer_range: float = 0.02,
        init_output_scale: float = 1e-3,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size or hidden_size * 4
        self.layer_idx = layer_idx
        self.init_output_scale = init_output_scale

        self.x_k = nn.Parameter(torch.zeros(hidden_size))
        self.key = nn.Linear(hidden_size, self.ffn_hidden_size, bias=False)
        self.value = nn.Linear(self.ffn_hidden_size, hidden_size, bias=False)

        nn.init.normal_(self.key.weight, mean=0.0, std=initializer_range)
        nn.init.normal_(
            self.value.weight,
            mean=0.0,
            std=initializer_range * init_output_scale,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params=None,
        cache_position=None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if cache_params is not None:
            has_previous = getattr(cache_params, "has_previous_state", None)
            if callable(has_previous) and has_previous():
                raise NotImplementedError(
                    "RWKV8ChannelMixDeltaNet supports full-sequence training/eval "
                    "only; recurrent cache decoding is not implemented."
                )

        prev = torch.zeros_like(hidden_states)
        if hidden_states.shape[1] > 1:
            prev[:, 1:] = hidden_states[:, :-1]

        xk = self.x_k.to(dtype=hidden_states.dtype).view(1, 1, -1)
        mixed = hidden_states + (prev - hidden_states) * xk
        k = torch.square(F.relu(self.key(mixed)))
        return self.value(k)


# ---------------------------------------------------------------------------
# RWKV-8 / RWKV-7 "Goose" time-mix (`RWKV_Tmix_x070`) port.
# ---------------------------------------------------------------------------

def _ortho_init(x: torch.Tensor, scale: float) -> torch.Tensor:
    """BlinkDL's `ortho_init`: scaled orthogonal for 2D, slice-wise for 3D."""
    with torch.no_grad():
        shape = x.shape
        if len(shape) == 2:
            gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1.0
            nn.init.orthogonal_(x, gain=gain * scale)
        elif len(shape) == 3:
            gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1.0
            for i in range(shape[0]):
                nn.init.orthogonal_(x[i], gain=gain * scale)
        else:
            raise AssertionError(f"unsupported shape for ortho_init: {tuple(shape)}")
        return x


def _rwkv7_python_ref(
    r: torch.Tensor, gk: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    a: torch.Tensor, b: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-Python recurrent reference for the RWKV-7 wkv7 kernel (DPLR form).

    Convention: ``gk`` is the log-decay (``decay = exp(gk)``), matching fla's
    ``chunk_dplr_delta_rule`` (``gk shape [B,T,H,K]``, decay term in log
    space). ``forward()`` is responsible for converting BlinkDL's
    ``w = -softplus(-α) - 0.5`` to ``gk = -exp(w)`` before calling either
    backend.

    State has shape (B, H, K_dim, V_dim) where:
        out[V] = sum_K state[K, V] * r[K]
        state[K, V] += k[K] * v[V]                            (write)
        state[K, V] += b[K] * sum_K' (state[K', V] * a[K'])   (rank-1 delta)
        state[K, V] *= exp(gk[K])                             (decay, per-K)
    For RWKV-7's wkv7 we have K_dim == V_dim == head_size, but the role
    of each axis still matters. (Verified against fla's
    fused_recurrent_dplr_delta_rule Triton kernel: ``b_h`` is (BK, BV),
    ``b_h += b_k[:, None] * b_v[None, :]``.)

    All inputs are (B, T, H, K). Slow O(B*T*H*K^2) but exact-match to fla.
    """
    B, T, H, Kd = r.shape
    Vd = v.shape[-1]
    if initial_state is None:
        state = torch.zeros(B, H, Kd, Vd, device=r.device, dtype=torch.float32)
    else:
        state = initial_state.to(device=r.device, dtype=torch.float32).clone()
    out = torch.empty_like(v, dtype=torch.float32)
    for t in range(T):
        rt = r[:, t].float()                                    # (B, H, Kd)
        gkt = gk[:, t].float()                                  # (B, H, Kd)
        kt = k[:, t].float()                                    # (B, H, Kd)
        vt = v[:, t].float()                                    # (B, H, Vd)
        at = a[:, t].float()                                    # (B, H, Kd)
        bt = b[:, t].float()                                    # (B, H, Kd)

        # RWKV-7 / DPLR step. The low-rank "remove" term uses the PRE-decay
        # state, matching fla's chunk_rwkv7 / fused_recurrent_dplr_delta_rule
        # (verified equal to chunk_rwkv7 below). Order: sa from old state ->
        # decay -> rank-1 delta add -> rank-1 k⊗v write -> read.
        old = state
        sa = (old * at.unsqueeze(-1)).sum(dim=-2)              # (B,H,Vd) pre-decay
        state = old * torch.exp(gkt).unsqueeze(-1)            # decay along K
        state = state + bt.unsqueeze(-1) * sa.unsqueeze(-2)    # (B, H, Kd, Vd)
        # Write: state[K, V] += k[K] * v[V]
        state = state + kt.unsqueeze(-1) * vt.unsqueeze(-2)    # (B, H, Kd, Vd)
        # Read: out[V] = sum_K state[K, V] * r[K]
        out_t = (state * rt.unsqueeze(-1)).sum(dim=-2)          # (B, H, Vd)
        out[:, t] = out_t.to(out.dtype)
    return out.to(v.dtype), state


class RWKV8TimeMixDeltaNet(nn.Module):
    """A drop-in replacement for Qwen3.5/Qwen3-Next ``linear_attn`` based on
    BlinkDL's RWKV-7 ``RWKV_Tmix_x070`` time-mix block (also shipped in the
    RWKV-v8 directory as the time-mix component of the ROSA hybrids).

    Notes for single-layer swaps:

    * ``v_first`` is treated as the layer-0 raw value (``v_first = v``) since
      we have only one RWKV layer in an otherwise-non-RWKV stack. The
      cross-layer value-residual mechanism is therefore inert for a single
      swap; it becomes meaningful once two or more RWKV layers exist.
    * ``output.weight`` is zero-initialised (BlinkDL convention), so the
      module returns near-zero at step 0 and only contributes to the
      residual stream once gradients flow.
    * ``depth_layer_id`` and ``depth_n_layer`` drive the per-layer init
      scaling formulas from BlinkDL — defaults model L30 of the 32-layer
      Qwen3.5-9B-Base. Override if dropping into a different position.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        num_heads: int = 64,
        head_size: int = 64,
        layer_idx: Optional[int] = None,
        depth_layer_id: int = 30,
        depth_n_layer: int = 32,
        decay_lora: int = 8,
        a_lora: int = 8,
        v_lora: int = 8,
        gate_lora: int = 8,
        decay_cap_delta: float = 0.0,
        allow_neg_eigval: bool = False,
        is_first_rwkv_layer: bool = True,
        use_rope: bool = False,
        rope_theta: float = 1e7,
        rope_frac: float = 0.25,
        comba_decouple: bool = False,
        out_correct: bool = True,
    ) -> None:
        super().__init__()
        if num_heads * head_size != hidden_size:
            raise ValueError(
                f"num_heads*head_size must equal hidden_size; "
                f"got {num_heads}*{head_size} != {hidden_size}"
            )
        if depth_n_layer < 2:
            raise ValueError("depth_n_layer must be >= 2 for paper init formulas")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = head_size
        self.layer_idx = layer_idx
        self.is_first_rwkv_layer = bool(is_first_rwkv_layer)
        # RAD-RWKV7 (RADLADS 2505.03005): for converting FULL-ATTENTION layers, graft the
        # teacher's RoPE onto the receptance r and write-key k. The delta-rule removal keys
        # kk / a / b are derived from k *after* rotation, so they inherit it automatically
        # (matches recursal/QRWKV7's forward). v, decay gk, and the iclr gate a are NOT
        # rotated. Off by default => bit-identical to the GDN-subset path. Partial rotary +
        # theta are inherited from the teacher (Qwen3.5: frac 1/4, theta 1e7).
        self.use_rope = bool(use_rope)
        if self.use_rope:
            rd = int(head_size * rope_frac)
            rd = max(2, (rd // 2) * 2)                 # even, <= head_size
            rd = min(rd, head_size - (head_size % 2))
            self.rope_dim = rd
            inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rd, 2, dtype=torch.float32) / rd))
            self.register_buffer("rope_inv_freq", inv_freq, persistent=False)
        else:
            self.rope_dim = 0
        # Comba (2506.02475) decoupled removal: scale the delta-rule REMOVAL term by a learned
        # per-head factor b_fb in (0,1) so removal can be WEAKER than the write (Comba's SPLR
        # β̃ = b·β decoupling; our out_correct_d already provides Comba's output feedback r-d·k).
        # When off, the removal term is unchanged => bit-identical. On => b_fb init 0.5 (Comba's
        # "weaker than write" setting).
        self.comba_decouple = bool(comba_decouple)
        # Comba r-d*k output-correction is a conversion lever, NOT part of native g070.
        # out_correct=False omits it entirely => clean native RWKV-7 forward.
        self.out_correct = bool(out_correct)
        if self.comba_decouple:
            self.comba_b = nn.Parameter(torch.zeros(num_heads))    # sigmoid(0)=0.5
        # Structural stability cap: floor w so the effective per-step decay
        # exp(gk) = exp(-exp(w)) stays <= 1 - decay_cap_delta. 0.0 disables it
        # (bit-identical to prior behavior). Prevents the decay->integrator
        # drift that erodes the RWKV-7 transition's contraction margin and is
        # the most likely driver of the delayed training divergence.
        self.decay_cap_delta = float(decay_cap_delta)
        # allow_neg_eigval: scale the in-context removal gate a from (0,1)->(0,2)
        # so the rank-1 DPLR term b=kk*a can reach reflection strength (transition
        # eigenvalue < 0), matching DeltaNet's beta*2 trick. Default off (no-op).
        self.allow_neg_eigval = bool(allow_neg_eigval)
        self._a_scale = 2.0 if self.allow_neg_eigval else 1.0
        self._w_floor = (
            math.log(-math.log(1.0 - self.decay_cap_delta))
            if self.decay_cap_delta > 0.0
            else float("-inf")
        )
        H = num_heads
        N = head_size
        C = hidden_size

        # BlinkDL's depth-scaled init: layer_id / (n_layer-1) and 1 - layer_id/n_layer.
        # We use ``depth_layer_id``/``depth_n_layer`` so we can place a single
        # RWKV layer "as if" it sat in a 32-layer RWKV stack at depth 30.
        with torch.no_grad():
            ratio_0_to_1 = depth_layer_id / (depth_n_layer - 1)         # 0..1
            ratio_1_to_almost0 = 1.0 - (depth_layer_id / depth_n_layer)  # 1..~0

            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C
            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_v = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

            www = torch.zeros(C)
            zigzag = torch.zeros(C)
            linear = torch.zeros(C)
            for n in range(C):
                linear[n] = n / (C - 1) - 0.5
                zigzag[n] = ((n % N) - ((N - 1) / 2)) / ((N - 1) / 2)
                zigzag[n] = zigzag[n] * abs(zigzag[n])
                www[n] = -6 + 6 * (n / (C - 1)) ** (1 + 1 * ratio_0_to_1 ** 0.3)

            self.w1 = nn.Parameter(torch.zeros(C, decay_lora))
            self.w2 = nn.Parameter(_ortho_init(torch.zeros(decay_lora, C), 0.1))
            self.w0 = nn.Parameter(www.reshape(1, 1, C) + 0.5 + zigzag * 2.5)

            self.a1 = nn.Parameter(torch.zeros(C, a_lora))
            self.a2 = nn.Parameter(_ortho_init(torch.zeros(a_lora, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1, 1, C) - 0.19 + zigzag * 0.3 + linear * 0.4)

            self.v1 = nn.Parameter(torch.zeros(C, v_lora))
            self.v2 = nn.Parameter(_ortho_init(torch.zeros(v_lora, C), 0.1))
            self.v0 = nn.Parameter(torch.zeros(1, 1, C) + 0.73 - linear * 0.4)

            self.g1 = nn.Parameter(torch.zeros(C, gate_lora))
            self.g2 = nn.Parameter(_ortho_init(torch.zeros(gate_lora, C), 0.1))

            self.k_k = nn.Parameter(torch.zeros(1, 1, C) + 0.71 - linear * 0.1)
            self.k_a = nn.Parameter(torch.zeros(1, 1, C) + 1.02)
            self.r_k = nn.Parameter(torch.zeros(H, N) - 0.04)
            # Comba (arXiv:2506.02475) output-correction: query the state with
            # (r - d*k) instead of r. Per-head scalar d, init 0 => exact no-op so
            # existing checkpoints/inits are unchanged until distillation trains it.
            # Omitted entirely for the clean-native g070 forward (out_correct=False).
            if self.out_correct:
                self.out_correct_d = nn.Parameter(torch.zeros(H))

            # Token-shift: pads ZERO at position 0 along the time dim, drops the
            # last position. With 3D input (B,T,C) ``ZeroPad2d`` pads/crops the
            # last two dims, so we get x shifted right by 1 along T.
            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(C, C, bias=False)
            self.key = nn.Linear(C, C, bias=False)
            self.value = nn.Linear(C, C, bias=False)
            self.output = nn.Linear(C, C, bias=False)
            self.ln_x = nn.GroupNorm(H, C, eps=64e-5)

            self.receptance.weight.data.uniform_(-0.5 / (C ** 0.5), 0.5 / (C ** 0.5))
            self.key.weight.data.uniform_(-0.05 / (C ** 0.5), 0.05 / (C ** 0.5))
            self.value.weight.data.uniform_(-0.5 / (C ** 0.5), 0.5 / (C ** 0.5))
            self.output.weight.data.zero_()

    # ------------------------------------------------------------------
    # DeltaNet weight inheritance for shape-matching tensors. Optional;
    # silently skips any tensor it can't slot in cleanly.
    # ------------------------------------------------------------------
    def init_from_deltanet(self, deltanet_state_dict: dict) -> dict:
        """Seed RWKV-7 params from a Gated-DeltaNet ``linear_attn.state_dict()``.

        Copies what transfers cleanly and leaves the rest to the functional
        (codec-prefit / block-MSE) init -- receptance/key/gate don't map cleanly
        (GDN q/k are 2048-wide vs RWKV 4096; the gate is full-rank vs low-rank):

          * output  <- out_proj.
          * value   <- V-slice of in_proj_qkv, head-split-aligned. GDN value
            head h (Hg=num_v_heads, dim Dvg) maps onto the (Hr//Hg)x-finer RWKV
            heads via the contiguous channel layout (Option 2).
          * decay w0 <- A_log/dt_bias so the per-step decay matches GDN's
            data-independent baseline d_h = exp(-exp(A_log[h])*softplus(dt_bias[h])),
            clamped to RWKV-7's reachable band [exp(-exp(-0.5))~0.545,
            1-decay_cap_delta]. w1 is zeroed so decay is data-independent at init;
            training learns the data-dependence (Option 1). GDN's fast-forget
            heads (d<0.545) saturate at RWKV-7's floor -- a known representational
            limit of the -0.5 in w=-softplus(-.)-0.5.

        Returns a report dict (which slots were filled vs left at paper init).
        """
        report: dict[str, str] = {}
        with torch.no_grad():
            out_w = deltanet_state_dict.get("out_proj.weight")
            if out_w is not None and out_w.shape == self.output.weight.shape:
                self.output.weight.data.copy_(out_w.to(self.output.weight.dtype))
                report["output"] = "from out_proj"
            else:
                report["output"] = "paper init (zeros)"

            # --- Option 2: value head-split (contiguous layout == split) ---
            qkv_w = deltanet_state_dict.get("in_proj_qkv.weight")
            if qkv_w is not None:
                # Standard Qwen3.5 DeltaNet layout: rows are [Q | K | V] along
                # dim 0. For 9B: Q=2048, K=2048, V=4096 -> V slice = last 4096.
                v_dim = self.value.weight.shape[0]
                if qkv_w.shape[0] >= v_dim:
                    v_slice = qkv_w[qkv_w.shape[0] - v_dim:, :]
                    if v_slice.shape == self.value.weight.shape:
                        # Contiguous channels: GDN value-head h's block
                        # [per*h:per*(h+1)] lands on RWKV heads [r*h:r*(h+1)],
                        # i.e. each GDN value-head splits across r=Hr//Hg RWKV
                        # heads -- the same head mapping the decay init uses.
                        self.value.weight.data.copy_(v_slice.to(self.value.weight.dtype))
                        report["value"] = "from in_proj_qkv V (head-split)"
                    else:
                        report["value"] = "paper init (slice shape mismatch)"
                else:
                    report["value"] = "paper init (qkv too small)"
            else:
                report["value"] = "paper init (no in_proj_qkv)"

            # --- Option 1: decay init from A_log/dt_bias (clamped) ---
            A_log = deltanet_state_dict.get("A_log")
            dt_bias = deltanet_state_dict.get("dt_bias")
            if A_log is not None and dt_bias is not None and self.hidden_size % int(A_log.shape[0]) == 0:
                Hg = int(A_log.shape[0])
                C = self.hidden_size
                # Decay inversion in float64: avoids catastrophic cancellation in
                # (-w_eff - 0.5) for near-floor heads (Codex: was ~3e-5 on the
                # floor-saturated heads). 32 values -> float64 is free.
                A64, dt64 = A_log.double(), dt_bias.double()
                d_raw = (-A64.exp() * F.softplus(dt64)).exp()           # GDN baseline decay (0,1) [Hg]
                floor = math.exp(-math.exp(-0.5))                        # ~0.5452 (RWKV-7 floor)
                cap = (1.0 - self.decay_cap_delta) if self.decay_cap_delta > 0 else 0.999999
                d = d_raw.clamp(min=floor + 1e-9, max=cap)
                w_eff = torch.log(-torch.log(d))                         # effective w in (-inf, -0.5]
                s = (-w_eff - 0.5).clamp_min(1e-9)                       # softplus(-w0) > 0 (d>floor)
                w0_head = -torch.log(torch.expm1(s))                     # -softplus_inv(s)  [Hg]
                w0_full = w0_head.repeat_interleave(C // Hg)             # [C]: per-head broadcast
                self.w0.data.copy_(w0_full.reshape(1, 1, C).to(self.w0.dtype))
                self.w1.data.zero_()                                     # decay data-independent at init
                n_sat = int((d_raw < floor).sum())
                report["decay_w0"] = (f"from A_log/dt_bias, clamp[{floor:.3f},{cap:.3f}], "
                                      f"{Hg} heads, {n_sat} saturated at floor")
            else:
                report["decay_w0"] = "paper init (no A_log/dt_bias or head mismatch)"
        return report

    # ------------------------------------------------------------------
    # Forward.
    # ------------------------------------------------------------------
    def _wkv7(
        self,
        r: torch.Tensor, gk: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        a: torch.Tensor, b: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Dispatch to fla's chunk_rwkv7 (Triton) or the slow Python ref.

        ``gk`` is the LOG-DECAY (``decay = exp(gk)``), matching fla's
        convention. ``forward()`` performs the BlinkDL-to-fla transform
        ``gk = -exp(w_blinkdl)`` before calling this method.

        Returns ``(out, final_state)``. ``final_state`` is the recurrent state
        ``[B, H, K, V]`` after the chunk, used for closed-loop rollout (DMT) and
        one-step memory supervision (SMT). ``initial_state`` seeds the
        recurrence (zeros when None). The fla path may return ``None`` for the
        state when ``output_final_state`` is False.
        """
        if _HAS_FLA and os.environ.get("RWKV8_FORCE_PYREF") != "1":
            out, final = _fla_chunk_rwkv7(
                r, gk, k, v, a, b, scale=1.0,
                initial_state=initial_state,
                output_final_state=output_final_state,
            )
            return out, final
        global _PYREF_WARNED
        if not _PYREF_WARNED:
            _PYREF_WARNED = True
            if os.environ.get("RWKV8_FORCE_PYREF") == "1":
                print("[rwkv8] RWKV8_FORCE_PYREF=1: using the slow Python wkv7 reference", flush=True)
            else:
                # A broken fla install must not degrade silently: the T-step Python
                # loop is ~100x slower and the only symptom would be terrible tok/s.
                print(f"[rwkv8] WARNING: fla unavailable ({_FLA_IMPORT_ERROR!r}) — falling back to the "
                      f"T-step Python wkv7 reference (~100x slower). Fix the fla install; convert_train "
                      f"refuses to launch in this state.", flush=True)
        out, final = _rwkv7_python_ref(
            r, gk, k, v, a, b, initial_state=initial_state
        )
        return out, (final if output_final_state else None)

    def _rope_cos_sin(self, B, T, device, dtype, position_ids):
        """Build (cos, sin) of shape [B, T, rope_dim] for partial rotary."""
        if position_ids is None:
            pos = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(0).expand(B, T)
        else:
            pos = position_ids.to(device=device, dtype=torch.float32)
            if pos.dim() == 1:
                pos = pos.unsqueeze(0).expand(B, T)
        freqs = pos[..., None] * self.rope_inv_freq.to(device)   # [B, T, rope_dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)                  # [B, T, rope_dim]
        return emb.cos().to(dtype), emb.sin().to(dtype)

    @staticmethod
    def _apply_partial_rope(x, cos, sin, rd):
        """Rotate the first ``rd`` channels of each head. x: [B,T,H,N]; cos/sin: [B,T,rd]."""
        xr, xp = x[..., :rd], x[..., rd:]
        half = rd // 2
        rot = torch.cat((-xr[..., half:], xr[..., :half]), dim=-1)
        c, s = cos.unsqueeze(2), sin.unsqueeze(2)                # [B,T,1,rd]
        xr = xr * c + rot * s
        return torch.cat((xr, xp), dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params=None,
        cache_position=None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        *,
        initial_state: Optional[torch.Tensor] = None,
        shift_state: Optional[torch.Tensor] = None,
        return_state: bool = False,
        v_first: Optional[torch.Tensor] = None,
        return_v_first: bool = False,
        **kwargs,
    ):
        if cache_params is not None:
            has_previous = getattr(cache_params, "has_previous_state", None)
            if callable(has_previous) and has_previous():
                raise NotImplementedError(
                    "RWKV8TimeMixDeltaNet supports full-sequence training/eval "
                    "only; recurrent cache decoding is not implemented."
                )

        x = hidden_states
        B, T, C = x.shape
        H = self.num_heads
        N = self.head_size

        # Token-shift reaches one token back. Across a chunk boundary the
        # "previous token" must come from the prior chunk (shift_state), or the
        # rollout would diverge from a full-sequence forward. shift_state is the
        # last hidden of the previous chunk, shape [B, 1, C] (or [B, C]).
        if shift_state is None:
            xx = self.time_shift(x) - x
        else:
            prev = shift_state.to(dtype=x.dtype).reshape(B, 1, C)
            shifted = torch.cat([prev, x[:, :-1]], dim=1)
            xx = shifted - x
        new_shift_state = x[:, -1:]

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5
        if self.decay_cap_delta > 0.0:
            # Floor w so exp(gk) = exp(-exp(w)) <= 1 - decay_cap_delta, keeping
            # the per-step decay strictly contractive (no integrator drift).
            w = w.clamp_min(self._w_floor)
        k = self.key(xk)
        v = self.value(xv)
        # RWKV-7 cross-layer value residual: layer 0 defines the shared v_first; every later
        # layer lerps its own v toward that layer-0 value, gated by the v-LoRA. This is the
        # native g070 mechanism; v0/v1/v2 are the residual gate's params.
        if self.is_first_rwkv_layer:
            v_first_out = v
        else:
            if v_first is None:
                raise ValueError(
                    "is_first_rwkv_layer=False requires v_first from layer 0 — thread the "
                    "layer-0 value through the stack (see RWKV7Small.forward / LoopedRWKV)."
                )
            a_v = torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
            v = torch.lerp(v, v_first.to(v.dtype), a_v.to(v.dtype))
            v_first_out = v_first
        # Comba output-correction (d init 0 => no-op): r_eff = r - d*k, per head.
        if self.out_correct:
            r = r - self.out_correct_d.repeat_interleave(N).view(1, 1, C) * k
        # RAD-RWKV7 RoPE: rotate r and the write-key k (same cos/sin). kk / k_eff / a / b
        # are computed from k below, so they inherit the rotation (QRWKV7 semantics). The
        # Comba mix above is rotation-equivariant (rotate(r - d*k) == rotate(r) - d*rotate(k)),
        # so applying rotary here is equivalent to rotating before the correction.
        if self.use_rope:
            cos, sin = self._rope_cos_sin(B, T, x.device, x.dtype, position_ids)
            r = self._apply_partial_rope(r.view(B, T, H, N), cos, sin, self.rope_dim).reshape(B, T, C)
            k = self._apply_partial_rope(k.view(B, T, H, N), cos, sin, self.rope_dim).reshape(B, T, C)
        a = self._a_scale * torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, H, N), dim=-1, p=2.0).view(B, T, C)
        k_eff = k * (1 + (a - 1) * self.k_a)

        # BlinkDL's CUDA kernel uses w as a "pre-log decay" and applies
        # ``decay = exp(-exp(w))`` internally. fla's chunk_rwkv7 instead
        # treats its second arg as log-decay (``decay = exp(gk)``). To match
        # BlinkDL's effective decay through fla we transform w first.
        gk = -torch.exp(w)

        # Reshape to (B, T, H, N) for fla / python ref.
        r_h = r.view(B, T, H, N)
        gk_h = gk.view(B, T, H, N)
        k_h = k_eff.view(B, T, H, N)
        v_h = v.view(B, T, H, N)
        a_h = (-kk).view(B, T, H, N)
        b_h = (kk * a).view(B, T, H, N)
        if self.comba_decouple:                                # Comba: decouple removal strength
            b_h = b_h * torch.sigmoid(self.comba_b).view(1, 1, H, 1)

        out, final_state = self._wkv7(
            r_h, gk_h, k_h, v_h, a_h, b_h,
            initial_state=initial_state,
            output_final_state=return_state,
        )                                                      # (B, T, H, N)

        # GroupNorm over channels (groups = H).
        out = out.reshape(B * T, C)
        out = self.ln_x(out).view(B, T, C)

        # r_k bonus residual. Uses k_eff (the k_a-updated key), matching BlinkDL x070 which
        # reassigns k = k*(1+(a-1)*k_a) before the bonus — NOT the raw key (was a port bug).
        bonus = (
            (r.view(B, T, H, N) * k_eff.view(B, T, H, N) * self.r_k.view(1, 1, H, N))
            .sum(dim=-1, keepdim=True) * v.view(B, T, H, N)
        ).view(B, T, C)

        y = self.output((out + bonus) * g)
        if return_v_first:                                     # native-stack threading (opt-in)
            if return_state:
                return y, final_state, new_shift_state, v_first_out
            return y, v_first_out
        if return_state:
            # Recurrent state for rollout/SMT/DMT is the pair
            # (wkv matrix state [B,H,K,V], last-token shift carry [B,1,C]).
            return y, final_state, new_shift_state
        return y


def rwkv8_timemix_from_config(
    config,
    *,
    layer_idx: Optional[int] = None,
    init_from_deltanet: Optional[dict] = None,
    num_heads: int = 64,
    head_size: int = 64,
    depth_layer_id: Optional[int] = None,
    depth_n_layer: Optional[int] = None,
    decay_cap_delta: float = 0.0,
    allow_neg_eigval: bool = False,
) -> RWKV8TimeMixDeltaNet:
    """Build an RWKV-8 time-mix module from an HF-style config.

    If ``init_from_deltanet`` is provided (a state_dict from the layer's
    original ``linear_attn`` module), shape-matching tensors are copied in
    after construction.
    """
    cfg = _text_config(config)
    H = int(getattr(cfg, "hidden_size"))
    n_layer = int(getattr(cfg, "num_hidden_layers", 32))
    d_layer_id = int(layer_idx) if depth_layer_id is None and layer_idx is not None else (depth_layer_id or 30)
    d_n_layer = depth_n_layer if depth_n_layer is not None else max(2, n_layer)
    rwkv = RWKV8TimeMixDeltaNet(
        hidden_size=H,
        num_heads=num_heads,
        head_size=head_size,
        layer_idx=layer_idx,
        depth_layer_id=d_layer_id,
        depth_n_layer=d_n_layer,
        decay_cap_delta=decay_cap_delta,
        allow_neg_eigval=allow_neg_eigval,
    )
    if init_from_deltanet is not None:
        rwkv.init_from_deltanet(init_from_deltanet)
    return rwkv


def _text_config(config):
    return getattr(config, "text_config", config)


def rwkv8_from_config(
    config,
    *,
    layer_idx: Optional[int] = None,
    ffn_hidden_size: Optional[int] = None,
    init_output_scale: float = 1e-3,
) -> RWKV8ChannelMixDeltaNet:
    cfg = _text_config(config)
    return RWKV8ChannelMixDeltaNet(
        hidden_size=int(cfg.hidden_size),
        ffn_hidden_size=ffn_hidden_size,
        layer_idx=layer_idx,
        initializer_range=float(getattr(cfg, "initializer_range", 0.02)),
        init_output_scale=init_output_scale,
    )


def parse_layer_list(spec: str | list[int] | tuple[int, ...] | None) -> list[int]:
    """Parse comma-separated layer indices. Empty means no replacement."""
    if spec is None or spec == "":
        return []
    if isinstance(spec, (list, tuple)):
        return [int(x) for x in spec]
    out: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def linear_attention_layer_indices_from_config(config) -> list[int]:
    cfg = _text_config(config)
    return [
        i for i, layer_type in enumerate(getattr(cfg, "layer_types", []))
        if layer_type == "linear_attention"
    ]
