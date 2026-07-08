"""
Build a fresh Engram patch (initial state) for layers [3, 19] of the MLA-
converted Qwen3.6-35B-A3B.

The Engram tables have no SVD-init path (unlike MLA) — they're random-
initialized embeddings that need to be learned from scratch. Per the paper:
  - Conv weights zero-initialized (identity at start -> Engram contributes
    gate*value only, and small random gate*value won't blow up training).
  - Other components (embedding, value_proj, key_proj, norms) use default PyTorch
    inits, which are sensible.

Output:
    /thearray/git/moe-mla/engram_converted/patch.safetensors
    /thearray/git/moe-mla/engram_converted/manifest.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import save_file

_ENGRAM_PATH = Path("/thearray/git/engram/python")
if str(_ENGRAM_PATH) not in sys.path:
    sys.path.insert(0, str(_ENGRAM_PATH))

from engram_ext.engram_module import EngramConfig, EngramModule  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/thearray/git/moe-mla/engram_converted_l2_l19")
    ap.add_argument("--layer-indices", nargs="+", type=int, default=[2, 19])
    ap.add_argument("--hidden-size", type=int, default=2048)
    ap.add_argument("--n-embed-per-ngram", type=int, default=512,
                    help="Engram internal dim per ngram order. 512 -> embed_dim_per_head=64")
    ap.add_argument("--vocab-per-ngram", type=int, default=5_000_000,
                    help="approximate number of slots per ngram order per head "
                         "(actual sizes are 8 distinct primes near this). "
                         "Target: ~5B params per Engram at default dims.")
    ap.add_argument("--host-offload", type=int, default=1,
                    help="1=store embedding tables on host RAM (pinned) for CPU-offloaded lookup. "
                         "0=legacy GPU-resident tables.")
    ap.add_argument("--max-ngram-size", type=int, default=3)
    ap.add_argument("--n-head-per-ngram", type=int, default=8)
    ap.add_argument("--tokenizer", default="/thearray/git/moe-mla/Qwen3.6-35B-A3B",
                    help="Use the local Qwen3.6 tokenizer path; its vocab is identical to Qwen3.5")
    ap.add_argument("--disable-compression", type=int, default=1,
                    help="1=hash raw token-ids (preserve case/punct for code); "
                         "0=paper-default NFKC+lowercase compression.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    engram_cfg = EngramConfig(
        engram_vocab_size=[args.vocab_per_ngram] * (args.max_ngram_size - 1),
        max_ngram_size=args.max_ngram_size,
        n_embed_per_ngram=args.n_embed_per_ngram,
        n_head_per_ngram=args.n_head_per_ngram,
        layer_ids=list(args.layer_indices),
        seed=args.seed,
        tokenizer_name_or_path=args.tokenizer,
        disable_compression=bool(args.disable_compression),
    )
    dtype = getattr(torch, args.dtype)

    print(f"Engram config:")
    print(f"  layers:              {engram_cfg.layer_ids}")
    print(f"  n_embed_per_ngram:   {engram_cfg.n_embed_per_ngram}")
    print(f"  embed_dim_per_head:  {engram_cfg.embed_dim_per_head}")
    print(f"  n_head_per_ngram:    {engram_cfg.n_head_per_ngram}")
    print(f"  max_ngram_size:      {engram_cfg.max_ngram_size}")
    print(f"  vocab per ngram:     {args.vocab_per_ngram:,}")
    print(f"  disable_compression: {engram_cfg.disable_compression}")
    print()

    patch: dict[str, torch.Tensor] = {}
    total_params = 0

    for li in engram_cfg.layer_ids:
        em = EngramModule(layer_id=li, cfg=engram_cfg, hidden_size=args.hidden_size)
        # Paper: convolution parameters zero-initialized so Engram's residual
        # contribution at t=0 is bounded by gate*value only (not the conv branch).
        with torch.no_grad():
            em.short_conv.conv.weight.zero_()

        em = em.to(dtype=dtype)
        layer_params = sum(p.numel() for p in em.parameters())
        total_params += layer_params
        print(f"  layer {li:2d}: {layer_params/1e6:,.1f}M params")
        for k, v in em.state_dict().items():
            patch[f"layer_{li}.{k}"] = v.contiguous().cpu()

    out_path = out_dir / "patch.safetensors"
    save_file(patch, out_path)

    manifest = {
        "output": str(out_path),
        "layer_indices": engram_cfg.layer_ids,
        "hidden_size": args.hidden_size,
        "dtype": args.dtype,
        "engram_config": asdict(engram_cfg),
        "vocab_per_ngram": args.vocab_per_ngram,
        "total_params": total_params,
        "host_offload": bool(args.host_offload),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print()
    print(f"wrote: {out_path} ({out_path.stat().st_size/1e9:.2f} GB)")
    print(f"wrote: {out_dir / 'manifest.json'}")
    print(f"total engram params: {total_params/1e9:.2f} B")


if __name__ == "__main__":
    main()
