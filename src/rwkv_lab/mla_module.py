"""
DeepSeek-V2/V3 style Multi-head Latent Attention module.

Parameter names match svd_init.gqa_to_mla_svd's output so a state_dict produced
there loads directly via load_state_dict.

Forward signature mirrors HF transformers attention modules so it can be
hot-swapped into a decoder layer: accepts hidden_states, position_embeddings
(cos/sin tuple), attention_mask, and returns (attn_output, attn_weights).
Cached decoding stores the normalized latent KV representation plus the small
rotary key and per-head K-normalization scale.  It never expands cached K/V to
all attention heads.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [..., D_rope]; cos/sin: [B, T, D_rope] (broadcastable to x)
    return x * cos + _rotate_half(x) * sin


class MLAAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        kv_lora_rank: int,
        q_lora_rank: Optional[int] = None,
        rms_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        softmax_scale: Optional[float] = None,
        use_latent_norm: bool = False,
        has_output_gate: bool = False,     # Qwen-style sigmoid gate on attn output
        has_qk_norm: bool = False,         # per-head-dim RMSNorm on Q and K
        rope_position: str = "last",       # "first" = Qwen, "last" = DeepSeek
        num_kv_rope_heads: int = 1,        # 1 = canonical shared k_rope, >1 = per-KV-head
        layer_idx: Optional[int] = None,    # cache slot when used inside an HF decoder
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.attention_dropout = attention_dropout
        self.softmax_scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(self.qk_head_dim)
        self.has_output_gate = has_output_gate
        self.has_qk_norm = has_qk_norm
        assert rope_position in ("first", "last")
        self.rope_position = rope_position
        assert num_heads % num_kv_rope_heads == 0, (
            f"num_heads ({num_heads}) must be a multiple of num_kv_rope_heads ({num_kv_rope_heads})"
        )
        self.num_kv_rope_heads = num_kv_rope_heads
        self.layer_idx = layer_idx
        self.xsa_enabled = False

        def _norm(dim: int) -> nn.Module:
            return nn.RMSNorm(dim, eps=rms_norm_eps) if use_latent_norm else nn.Identity()

        if q_lora_rank is None:
            self.q_proj = nn.Linear(hidden_size, num_heads * self.qk_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=False)
            self.q_a_layernorm = _norm(q_lora_rank)
            self.q_b_proj = nn.Linear(q_lora_rank, num_heads * self.qk_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(
            hidden_size, kv_lora_rank + num_kv_rope_heads * qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = _norm(kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim), bias=False
        )

        self.o_proj = nn.Linear(num_heads * v_head_dim, hidden_size, bias=False)

        if has_output_gate:
            # Per-head gate of size head_dim (same as V's head_dim so it fits the
            # pre-o_proj tensor of shape [B, T, Nh, v_head_dim]).
            self.gate_proj = nn.Linear(hidden_size, num_heads * v_head_dim, bias=False)

        if has_qk_norm:
            # One RMSNorm each, applied to the full per-head vector of size head_dim.
            # RMS is computed over the full concatenated [nope|rope] or [rope|nope],
            # matching Qwen's semantics; splitting would change normalization scale.
            self.q_norm = nn.RMSNorm(self.qk_head_dim, eps=rms_norm_eps)
            self.k_norm = nn.RMSNorm(self.qk_head_dim, eps=rms_norm_eps)

    def _project_q(self, x: torch.Tensor) -> torch.Tensor:
        if self.q_lora_rank is None:
            return self.q_proj(x)
        return self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))

    def _cache_components(self, kv_lat, k_rope_shared, k_raw, cos, sin):
        """Return compact cache tensors: latent KV, weighted/rotated rope K, K RMS scale."""
        B, T, Nh, _ = k_raw.shape
        Dn, Dr = self.qk_nope_head_dim, self.qk_rope_head_dim
        if self.has_qk_norm:
            inv = torch.rsqrt(k_raw.float().square().mean(-1) + self.k_norm.eps).to(k_raw.dtype)
            w = self.k_norm.weight.to(k_raw.dtype)
            w_rope = w[:Dr] if self.rope_position == "first" else w[Dn:]
        else:
            inv = k_raw.new_ones(B, T, Nh)
            w_rope = k_raw.new_ones(Dr)
        rope = _apply_rope(
            k_rope_shared * w_rope.view(1, 1, 1, Dr),
            cos.unsqueeze(2), sin.unsqueeze(2),
        )
        return kv_lat, rope, inv

    def _update_cache(self, past_key_value, kv_lat, rope, inv):
        if self.layer_idx is None:
            raise ValueError("MLAAttention needs layer_idx when past_key_value is supplied")
        B, T, _, _ = rope.shape
        packed = torch.cat((rope.reshape(B, T, -1), inv), dim=-1).unsqueeze(1)
        latent = kv_lat.unsqueeze(1)
        packed, latent = past_key_value.update(packed, latent, self.layer_idx)
        packed, latent = packed.squeeze(1), latent.squeeze(1)
        rope_width = self.num_kv_rope_heads * self.qk_rope_head_dim
        rope_all = packed[..., :rope_width].view(
            B, packed.shape[-2], self.num_kv_rope_heads, self.qk_rope_head_dim
        )
        inv_all = packed[..., rope_width:]
        return latent, rope_all, inv_all

    def _latent_cached_attention(self, q, kv_lat, rope, inv, attention_mask, past_len):
        """Exact absorbed MLA attention over the compact latent cache."""
        B, Q, Nh, _ = q.shape
        Dn, Dr, R = self.qk_nope_head_dim, self.qk_rope_head_dim, self.kv_lora_rank
        if self.rope_position == "first":
            q_rope, q_nope = q[..., :Dr], q[..., Dr:]
        else:
            q_nope, q_rope = q[..., :Dn], q[..., Dn:]

        weights = self.kv_b_proj.weight.view(Nh, Dn + self.v_head_dim, R)
        wk, wv = weights[:, :Dn], weights[:, Dn:]
        if self.has_qk_norm:
            kw = self.k_norm.weight.to(q.dtype)
            kw_nope = kw[Dr:] if self.rope_position == "first" else kw[:Dn]
            q_nope = q_nope * kw_nope.view(1, 1, 1, Dn)
        q_lat = torch.einsum("bthd,hdr->bhtr", q_nope, wk.to(q.dtype))
        score = torch.einsum("bhqr,bkr->bhqk", q_lat, kv_lat.to(q.dtype))

        repeat = Nh // self.num_kv_rope_heads
        rope_h = rope.unsqueeze(3).expand(
            B, rope.shape[1], self.num_kv_rope_heads, repeat, Dr
        ).reshape(B, rope.shape[1], Nh, Dr)
        score = score + torch.einsum("bqhd,bkhd->bhqk", q_rope, rope_h.to(q.dtype))
        score = score * inv.transpose(1, 2).unsqueeze(2).to(score.dtype)
        score = score * self.softmax_scale

        if attention_mask is not None:
            mask = attention_mask[..., -Q:, :score.shape[-1]]
            if mask.dtype == torch.bool:
                score = score.masked_fill(~mask, float("-inf"))
            else:
                score = score + mask.to(score.dtype)
        elif Q > 1:
            qi = torch.arange(Q, device=q.device) + past_len
            ki = torch.arange(score.shape[-1], device=q.device)
            score = score.masked_fill(ki.view(1, 1, 1, -1) > qi.view(1, 1, -1, 1), float("-inf"))
        prob = torch.softmax(score.float(), dim=-1).to(q.dtype)
        if self.training and self.attention_dropout:
            prob = F.dropout(prob, p=self.attention_dropout)
        ctx = torch.einsum("bhqk,bkr->bhqr", prob, kv_lat.to(prob.dtype))
        return torch.einsum("bhqr,hdr->bqhd", ctx, wv.to(ctx.dtype))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        past_key_value = kwargs.get("past_key_value")
        use_cache = bool(kwargs.get("use_cache", False) or past_key_value is not None)
        B, T, _ = hidden_states.shape
        Nh = self.num_heads
        D_nope = self.qk_nope_head_dim
        D_rope = self.qk_rope_head_dim
        D_v = self.v_head_dim
        R = self.kv_lora_rank

        cos, sin = position_embeddings
        if cos.shape[-1] != D_rope:
            cos = cos[..., :D_rope]
            sin = sin[..., :D_rope]
        if cos.dim() == 2:
            cos = cos.unsqueeze(0).expand(B, -1, -1)
            sin = sin.unsqueeze(0).expand(B, -1, -1)

        # --- Q --- project to per-head vector, then norm over full head_dim
        q = self._project_q(hidden_states).view(B, T, Nh, self.qk_head_dim)

        # --- KV ---
        Nkr = self.num_kv_rope_heads
        kv_a = self.kv_a_proj_with_mqa(hidden_states)
        kv_lat, k_rope_flat = kv_a.split([R, Nkr * D_rope], dim=-1)
        kv_lat = self.kv_a_layernorm(kv_lat)
        k_rope_shared = k_rope_flat.view(B, T, Nkr, D_rope)
        kv = self.kv_b_proj(kv_lat).view(B, T, Nh, D_nope + D_v)
        k_nope, v = kv.split([D_nope, D_v], dim=-1)

        # Expand k_rope_shared to Nh and assemble full per-head K (pre-RoPE)
        n_rep = Nh // Nkr
        k_rope_per_head = k_rope_shared.unsqueeze(3).expand(B, T, Nkr, n_rep, D_rope).reshape(B, T, Nh, D_rope)
        if self.rope_position == "first":
            k = torch.cat([k_rope_per_head, k_nope], dim=-1)   # [B, T, Nh, head_dim]
        else:
            k = torch.cat([k_nope, k_rope_per_head], dim=-1)

        # QK-norm on the full per-head vector (matches Qwen's RMS semantics)
        k_raw = k
        if self.has_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # RoPE on the rope slice within each head
        if self.rope_position == "first":
            q_rope = _apply_rope(q[..., :D_rope], cos.unsqueeze(2), sin.unsqueeze(2))
            q = torch.cat([q_rope, q[..., D_rope:]], dim=-1)
            k_rope = _apply_rope(k[..., :D_rope], cos.unsqueeze(2), sin.unsqueeze(2))
            k = torch.cat([k_rope, k[..., D_rope:]], dim=-1)
        else:
            q_rope = _apply_rope(q[..., D_nope:], cos.unsqueeze(2), sin.unsqueeze(2))
            q = torch.cat([q[..., :D_nope], q_rope], dim=-1)
            k_rope = _apply_rope(k[..., D_nope:], cos.unsqueeze(2), sin.unsqueeze(2))
            k = torch.cat([k[..., :D_nope], k_rope], dim=-1)

        # Cache only c_KV, the shared rotary key, and one K RMS scalar/head/token.
        # Prefill keeps the highly optimized SDPA path; calls with an existing
        # cache use the exact absorbed formulation and never expand cached V.
        cached = None
        if use_cache:
            c_lat, c_rope, c_inv = self._cache_components(
                kv_lat, k_rope_shared, k_raw, cos, sin
            )
            past_len = 0
            if past_key_value is not None:
                try:
                    past_len = past_key_value.get_seq_length(self.layer_idx)
                except (AttributeError, TypeError):
                    if self.layer_idx is not None and len(getattr(past_key_value, "key_cache", [])) > self.layer_idx:
                        old = past_key_value.key_cache[self.layer_idx]
                        past_len = 0 if not torch.is_tensor(old) or not old.numel() else old.shape[-2]
                cached = self._update_cache(past_key_value, c_lat, c_rope, c_inv)
            else:
                cached = (c_lat, c_rope, c_inv)

        if past_key_value is not None and past_len > 0:
            attn_h = self._latent_cached_attention(q, *cached, attention_mask, past_len)
            attn = attn_h.reshape(B, T, Nh * D_v)
        else:
            q_full = q.transpose(1, 2)
            k_full = k.transpose(1, 2)
            v_t = v.transpose(1, 2)

            attn = F.scaled_dot_product_attention(
                q_full, k_full, v_t,
                attn_mask=attention_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=(attention_mask is None),
                scale=self.softmax_scale,
            )
            attn = attn.transpose(1, 2).contiguous().view(B, T, Nh * D_v)

        if self.xsa_enabled:
            attn_h = attn.view(B, T, Nh, D_v)
            v_n = F.normalize(v, dim=-1)
            attn_h = attn_h - (attn_h * v_n).sum(dim=-1, keepdim=True) * v_n
            attn = attn_h.reshape(B, T, Nh * D_v)

        if self.has_output_gate:
            gate = self.gate_proj(hidden_states)  # [B, T, Nh*D_v]
            attn = attn * torch.sigmoid(gate)

        return self.o_proj(attn), None


# ---------------------------------------------------------------------------
# Sanity: load an SVD-initialized state_dict and verify forward doesn't crash
# and that at full rank it matches a reference MHA output (no RoPE).
# ---------------------------------------------------------------------------

def _smoke() -> None:
    from .svd_init import GQAConfig, MLAConfig, gqa_to_mla_svd, _expand_gqa

    # num_kv_heads=1 so all heads already share K/V -> k_rope averaging is lossless.
    gqa = GQAConfig(hidden_size=512, num_q_heads=8, num_kv_heads=1, head_dim=64)
    mla = MLAConfig(
        hidden_size=512,
        num_heads=8,
        qk_nope_head_dim=32,
        qk_rope_head_dim=32,
        v_head_dim=64,
        kv_lora_rank=min(512, 8 * (32 + 64)),  # full rank for exact-match test
    )

    torch.manual_seed(0)
    gqa_sd = {
        "q_proj.weight": torch.randn(gqa.num_q_heads * gqa.head_dim, gqa.hidden_size) * 0.02,
        "k_proj.weight": torch.randn(gqa.num_kv_heads * gqa.head_dim, gqa.hidden_size) * 0.02,
        "v_proj.weight": torch.randn(gqa.num_kv_heads * gqa.head_dim, gqa.hidden_size) * 0.02,
        "o_proj.weight": torch.randn(gqa.hidden_size, gqa.num_q_heads * gqa.head_dim) * 0.02,
    }
    mla_sd = gqa_to_mla_svd(gqa_sd, gqa, mla)

    module = MLAAttention(
        hidden_size=mla.hidden_size,
        num_heads=mla.num_heads,
        qk_nope_head_dim=mla.qk_nope_head_dim,
        qk_rope_head_dim=mla.qk_rope_head_dim,
        v_head_dim=mla.v_head_dim,
        kv_lora_rank=mla.kv_lora_rank,
    ).to(torch.float32)
    # RMSNorm weight init to 1 -> identity, matching SVD init assumption.
    missing, unexpected = module.load_state_dict(mla_sd, strict=False)
    assert not unexpected, unexpected

    B, T = 2, 8
    x = torch.randn(B, T, mla.hidden_size)
    # No-RoPE sanity: cos = 1, sin = 0 => RoPE is identity.
    cos = torch.ones(B, T, mla.qk_rope_head_dim)
    sin = torch.zeros(B, T, mla.qk_rope_head_dim)
    with torch.no_grad():
        y, _ = module(x, position_embeddings=(cos, sin))
    print(f"MLA output: {tuple(y.shape)}   norm={y.norm().item():.4f}")

    # Reference: original GQA forward (RoPE-free), same softmax scale.
    Nh = gqa.num_q_heads
    d = gqa.head_dim
    W_Q = gqa_sd["q_proj.weight"].T
    W_K = _expand_gqa(gqa_sd["k_proj.weight"].T, gqa.num_kv_heads, gqa.num_q_heads, d)
    W_V = _expand_gqa(gqa_sd["v_proj.weight"].T, gqa.num_kv_heads, gqa.num_q_heads, d)
    W_O = gqa_sd["o_proj.weight"]
    q = (x @ W_Q).view(B, T, Nh, d).transpose(1, 2)
    k = (x @ W_K).view(B, T, Nh, d).transpose(1, 2)
    v = (x @ W_V).view(B, T, Nh, d).transpose(1, 2)
    with torch.no_grad():
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0 / math.sqrt(d))
    ref = ref.transpose(1, 2).contiguous().view(B, T, Nh * d) @ W_O.T
    rel = (y - ref).norm().item() / (ref.norm().item() + 1e-9)
    print(f"vs reference GQA (RoPE-free, full-rank init): rel err = {rel:.3e}")


if __name__ == "__main__":
    _smoke()
