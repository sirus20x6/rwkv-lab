"""
End-to-end sanity check on a real Qwen3.6-35B-A3B full-attention layer.

Loads layer 3's attention weights, runs SVD init to MLA (balanced config:
R=1024, 4x heads, gate/qk_norm/rope-first plumbed), and compares the MLA
forward to a reference Qwen-style forward (q_norm/k_norm, partial rope on
first 64 dims, sigmoid output gate) on a random batch.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from svd_init import GQAConfig, MLAConfig, gqa_to_mla_svd, _expand_gqa
from mla_module import MLAAttention, _apply_rope
from layer_swap import gqa_config_from_hf

MODEL_DIR = Path("/thearray/git/moe-mla/Qwen3.6-35B-A3B")
LAYER_IDX = 3
KEYS = ["q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
        "q_norm.weight", "k_norm.weight"]


def load_layer_sd(layer_idx: int) -> dict:
    idx = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())
    wmap = idx["weight_map"]
    prefix = f"model.language_model.layers.{layer_idx}.self_attn."
    out = {}
    for k in KEYS:
        full = prefix + k
        with safe_open(MODEL_DIR / wmap[full], framework="pt") as f:
            out[k] = f.get_tensor(full).float()
    return out


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    v = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(v + eps) * weight


def reference_forward(sd: dict, cfg: dict, x: torch.Tensor,
                      cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Replicate Qwen3_5Moe attention forward exactly (no KV cache, causal)."""
    B, T, H = x.shape
    Nh = cfg["num_attention_heads"]
    Nkv = cfg["num_key_value_heads"]
    d = cfg["head_dim"]
    D_rope = int(d * cfg["partial_rotary_factor"])

    # Q path: projected, viewed per-head with 2*d last, chunked into (q, gate)
    q_all = x @ sd["q_proj.weight"].T          # [B, T, Nh*2*d]
    q_all = q_all.view(B, T, Nh, 2 * d)
    q, gate = q_all.chunk(2, dim=-1)           # each [B, T, Nh, d]
    k = (x @ sd["k_proj.weight"].T).view(B, T, Nkv, d)
    v = (x @ sd["v_proj.weight"].T).view(B, T, Nkv, d)

    # Norm before RoPE, per-head
    q = rms_norm(q, sd["q_norm.weight"])
    k = rms_norm(k, sd["k_norm.weight"])

    # GQA expansion
    n_rep = Nh // Nkv
    k = k.unsqueeze(3).expand(B, T, Nkv, n_rep, d).reshape(B, T, Nh, d)
    v = v.unsqueeze(3).expand(B, T, Nkv, n_rep, d).reshape(B, T, Nh, d)

    # Partial RoPE on first D_rope dims
    q_rot, q_pass = q[..., :D_rope], q[..., D_rope:]
    k_rot, k_pass = k[..., :D_rope], k[..., D_rope:]
    q_rot = _apply_rope(q_rot, cos.unsqueeze(2), sin.unsqueeze(2))
    k_rot = _apply_rope(k_rot, cos.unsqueeze(2), sin.unsqueeze(2))
    q = torch.cat([q_rot, q_pass], dim=-1)
    k = torch.cat([k_rot, k_pass], dim=-1)

    # Attention
    q_t, k_t, v_t = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    attn = F.scaled_dot_product_attention(
        q_t, k_t, v_t, is_causal=True, scale=1.0 / math.sqrt(d)
    )
    attn = attn.transpose(1, 2).contiguous()   # [B, T, Nh, d]

    # Output gate
    attn = attn * torch.sigmoid(gate)

    # Output projection
    attn = attn.view(B, T, Nh * d)
    return attn @ sd["o_proj.weight"].T


def main() -> None:
    cfg = json.loads((MODEL_DIR / "config.json").read_text())["text_config"]
    sd = load_layer_sd(LAYER_IDX)

    gqa_cfg = gqa_config_from_hf(cfg)
    d = gqa_cfg.head_dim
    D_rope = int(d * cfg["partial_rotary_factor"])
    D_nope = d - D_rope
    print(f"GQAConfig: {gqa_cfg}")
    print(f"head_dim={d}  D_rope={D_rope}  D_nope={D_nope}")

    # Balanced MLA: 4x heads, R=1024, preserve per-KV-head k_rope
    mla_cfg = MLAConfig(
        hidden_size=gqa_cfg.hidden_size,
        num_heads=gqa_cfg.num_q_heads * 4,
        qk_nope_head_dim=D_nope,
        qk_rope_head_dim=D_rope,
        v_head_dim=d,
        kv_lora_rank=1024,
        has_output_gate=True,
        has_qk_norm=True,
        num_kv_rope_heads=gqa_cfg.num_kv_heads,
    )
    print(f"MLAConfig: Nh={mla_cfg.num_heads}  R={mla_cfg.kv_lora_rank}")

    # --- First run with expand=1, noise=0 to isolate SVD + gate + norm + rope plumbing ---
    mla_cfg_same = MLAConfig(**{**mla_cfg.__dict__, "num_heads": gqa_cfg.num_q_heads})

    def build_mla(mcfg: MLAConfig, noise: float) -> MLAAttention:
        mla_sd = gqa_to_mla_svd(sd, gqa_cfg, mcfg, head_expand_noise_std=noise)
        m = MLAAttention(
            hidden_size=mcfg.hidden_size,
            num_heads=mcfg.num_heads,
            qk_nope_head_dim=mcfg.qk_nope_head_dim,
            qk_rope_head_dim=mcfg.qk_rope_head_dim,
            v_head_dim=mcfg.v_head_dim,
            kv_lora_rank=mcfg.kv_lora_rank,
            has_output_gate=mcfg.has_output_gate,
            has_qk_norm=mcfg.has_qk_norm,
            rope_position=gqa_cfg.rope_position,
            num_kv_rope_heads=mcfg.num_kv_rope_heads,
        ).to(torch.float32)
        missing, unexpected = m.load_state_dict(mla_sd, strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected: {unexpected}")
        return m

    B, T = 2, 32
    torch.manual_seed(0)
    x = torch.randn(B, T, gqa_cfg.hidden_size) * 0.5
    cos = torch.ones(B, T, D_rope)     # identity RoPE for baseline
    sin = torch.zeros(B, T, D_rope)

    ref = reference_forward(sd, cfg, x, cos, sin)

    # Configuration sweep
    configs = [
        ("same heads, R=1024, noise=0", mla_cfg_same, 0.0),
        ("same heads, R=full, noise=0",
         MLAConfig(**{**mla_cfg_same.__dict__, "kv_lora_rank": min(2048, gqa_cfg.num_q_heads*(D_nope+d))}),
         0.0),
        ("4x heads, R=1024, noise=0", mla_cfg, 0.0),
        ("4x heads, R=1024, noise=1e-3", mla_cfg, 1e-3),
    ]
    print()
    print(f"{'config':<38s} {'rel_err_vs_ref':>16s}")
    print("-" * 58)
    for name, mcfg, noise in configs:
        m = build_mla(mcfg, noise)
        with torch.no_grad():
            y, _ = m(x, position_embeddings=(cos, sin))
        rel = (y - ref).norm().item() / (ref.norm().item() + 1e-9)
        print(f"{name:<38s} {rel:>16.4e}")

    # Also try a non-trivial RoPE (real cos/sin) to stress the partial-rope path
    print()
    print("with non-trivial RoPE (real sinusoidal cos/sin):")
    freqs = 1.0 / (cfg["rope_parameters"]["rope_theta"] ** (torch.arange(0, D_rope, 2).float() / D_rope))
    pos = torch.arange(T).float()
    theta = torch.outer(pos, freqs)
    cos_real = torch.cat([theta.cos(), theta.cos()], dim=-1).unsqueeze(0).expand(B, -1, -1)
    sin_real = torch.cat([theta.sin(), theta.sin()], dim=-1).unsqueeze(0).expand(B, -1, -1)

    ref2 = reference_forward(sd, cfg, x, cos_real, sin_real)
    m = build_mla(mla_cfg, 0.0)
    with torch.no_grad():
        y2, _ = m(x, position_embeddings=(cos_real, sin_real))
    rel = (y2 - ref2).norm().item() / (ref2.norm().item() + 1e-9)
    print(f"  4x heads, R=1024, noise=0: rel_err = {rel:.4e}")


if __name__ == "__main__":
    main()
