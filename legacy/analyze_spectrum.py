"""
Run KV singular-value spectrum analysis on every full-attention layer of the
Qwen3.6-35B-A3B checkpoint. Uses Qwen's layout: RoPE on the first
`partial_rotary_factor * head_dim` dims per head.

Prints, for each layer, the rank needed to retain various energy thresholds.
This tells us what kv_lora_rank captures ~99% of the trained [K_nope | V]
structure — i.e. how much rank we can drop without meaningful quality loss.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open

MODEL_DIR = Path("/thearray/git/moe-mla/Qwen3.6-35B-A3B")


def load_config() -> dict:
    cfg = json.loads((MODEL_DIR / "config.json").read_text())
    return cfg["text_config"]


def full_attn_indices(cfg: dict) -> list[int]:
    return [i for i, t in enumerate(cfg["layer_types"]) if t == "full_attention"]


def weight_map() -> dict[str, str]:
    idx = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text())
    return idx["weight_map"]


def get_tensor(wmap: dict[str, str], name: str) -> torch.Tensor:
    with safe_open(MODEL_DIR / wmap[name], framework="pt") as f:
        return f.get_tensor(name)


def expand_gqa(w: torch.Tensor, n_kv: int, n_q: int, d: int) -> torch.Tensor:
    # w: [n_kv * d, H] (PyTorch [out, in] layout) -> we'll operate in [H, n_q*d]
    H = w.shape[1]
    wt = w.T.contiguous().view(H, n_kv, d)
    n_rep = n_q // n_kv
    return wt.unsqueeze(2).expand(H, n_kv, n_rep, d).reshape(H, n_q * d).contiguous()


def layer_spectrum(wmap: dict, layer_idx: int, cfg: dict) -> dict:
    H = cfg["hidden_size"]
    Nq = cfg["num_attention_heads"]
    Nkv = cfg["num_key_value_heads"]
    d = cfg["head_dim"]
    D_rope = int(d * cfg["partial_rotary_factor"])   # Qwen: first D_rope dims carry RoPE
    D_nope = d - D_rope
    D_v = d

    prefix = f"model.language_model.layers.{layer_idx}.self_attn."
    W_K = get_tensor(wmap, prefix + "k_proj.weight").float()   # [Nkv*d, H]
    W_V = get_tensor(wmap, prefix + "v_proj.weight").float()   # [Nkv*d, H]

    W_K_full = expand_gqa(W_K, Nkv, Nq, d)  # [H, Nq*d]
    W_V_full = expand_gqa(W_V, Nkv, Nq, d)

    # Qwen layout: rope = first D_rope, nope = last D_nope
    W_K_per_head = W_K_full.view(H, Nq, d)
    W_K_nope = W_K_per_head[:, :, D_rope:].reshape(H, Nq * D_nope).contiguous()
    # V is fully content (not rope-affected)
    W_combined = torch.cat([W_K_nope, W_V_full], dim=1)  # [H, Nq*(D_nope + D_v)]

    _, S, _ = torch.linalg.svd(W_combined, full_matrices=False)
    energy = (S ** 2).cumsum(0) / (S ** 2).sum()
    full_rank = S.numel()

    def rank_at(t: float) -> int:
        return int((energy >= t).nonzero(as_tuple=True)[0][0].item()) + 1

    return {
        "layer": layer_idx,
        "full_rank": full_rank,
        "r50":  rank_at(0.50),
        "r90":  rank_at(0.90),
        "r95":  rank_at(0.95),
        "r99":  rank_at(0.99),
        "r999": rank_at(0.999),
        "top_sv_ratio": (S[0] / S[-1]).item(),  # condition number proxy
    }


def main() -> None:
    cfg = load_config()
    wmap = weight_map()
    idx = full_attn_indices(cfg)
    print(f"hidden_size={cfg['hidden_size']}  num_heads={cfg['num_attention_heads']}  "
          f"num_kv_heads={cfg['num_key_value_heads']}  head_dim={cfg['head_dim']}  "
          f"rope_dim={int(cfg['head_dim']*cfg['partial_rotary_factor'])}")
    print(f"full-attention layers: {idx}")
    print()
    print(f"{'layer':>5} {'full':>6} {'r50':>6} {'r90':>6} {'r95':>6} {'r99':>6} {'r99.9':>7}")
    rs = {k: [] for k in ("r50", "r90", "r95", "r99", "r999")}
    for li in idx:
        row = layer_spectrum(wmap, li, cfg)
        print(f"{row['layer']:>5} {row['full_rank']:>6} "
              f"{row['r50']:>6} {row['r90']:>6} {row['r95']:>6} "
              f"{row['r99']:>6} {row['r999']:>7}")
        for k in rs:
            rs[k].append(row[k])
    print()
    print("Summary (min / median / max across all full-attn layers):")
    for k, vs in rs.items():
        vs = sorted(vs)
        print(f"  {k:6s} min={vs[0]:4d}  median={vs[len(vs)//2]:4d}  max={vs[-1]:4d}")


if __name__ == "__main__":
    main()
