"""
GPU-based Engram pre-fill / standalone N-gram LM training.

Takes existing Engram tables (from a training ckpt or the init patch) and
trains them as a standalone N-gram language model:

    slot_embeddings -> value_proj -> final_norm -> lm_head -> CE(next_token)

The backbone isn't used during prefill — we substitute final_norm + lm_head
directly on the Engram output. This gives the Engram embeddings a much
cleaner learning signal than joint training (which has to disentangle
Engram's contribution from 40 layers of backbone mixing).

Gate components (key_proj, norm_key, norm_query, short_conv) are NOT trained
during prefill — those are context-dependent (need a real hidden state) and
will be refined during the subsequent joint training. Prefill just learns:
  * embedding.embedding.weight     — the big hash table
  * value_proj.weight              — 1024 -> hidden_size projection

Multi-corpus schedule: corpora are processed in the order given. Earlier
corpora establish a base; later corpora refine. Example:

    python gpu_engram_prefill.py \\
      --engram-patch-dir /thearray/git/moe-mla/engram_converted_l3_l19 \\
      --resume-from-ckpt /thearray/git/moe-mla/runs/phase3_engram_l3_l19/step_007178/ckpt.pt \\
      --corpus /thearray/data/engram_tokens.bin:79940000000 \\
      --corpus /thearray/data/non_cvevc_tokens.bin:29284583603 \\
      --batch-size 8192 --steps-per-corpus 20000 --lr 5e-4 \\
      --out /thearray/git/moe-mla/engram_prefilled

Output: a safetensors patch in --out plus a manifest.json, loadable by
train_mla.py via --engram-patch-dir (just like the fresh init patch).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from safe_torch import safe_torch_load

_ENGRAM_PATH = Path("/thearray/git/engram/python")
if str(_ENGRAM_PATH) not in sys.path:
    sys.path.insert(0, str(_ENGRAM_PATH))

from engram_ext.engram_module import EngramConfig, EngramModule, NgramHashMapping  # noqa: E402


def _parse_corpus_arg(s: str) -> tuple[Path, int]:
    """--corpus PATH:TOTAL_TOKENS."""
    if ":" not in s:
        raise ValueError(f"--corpus must be PATH:TOTAL_TOKENS, got {s!r}")
    path, total = s.rsplit(":", 1)
    return Path(path), int(total)


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    var = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(var + eps) * weight.float()).to(x.dtype)


class PrefillLens(nn.Module):
    """One Engram-layer's forward, stripped to just the prefill-relevant path:
        embedding lookup -> concat -> value_proj -> final_norm -> lm_head
    Trainable: embedding.weight, value_proj.weight.
    Frozen:    final_norm, lm_head.
    """

    def __init__(self, em: EngramModule, final_norm_weight: torch.Tensor,
                 lm_head_weight: torch.Tensor) -> None:
        super().__init__()
        self.em = em
        self.register_buffer("final_norm_weight", final_norm_weight, persistent=False)
        self.register_buffer("lm_head_weight", lm_head_weight, persistent=False)

    def forward(self, hash_ids: torch.Tensor) -> torch.Tensor:
        """hash_ids: [B, total_heads] int64 — slot ids for one target position each.
        returns logits [B, V]."""
        emb = self.em.embedding(hash_ids)              # [B, H_heads, d_per_head]
        emb = emb.flatten(start_dim=-2)                 # [B, d_engram]
        value = self.em.value_proj(emb)                 # [B, hidden_size]
        normed = _rms_norm(value, self.final_norm_weight)
        logits = normed.float() @ self.lm_head_weight.float().T  # [B, V]
        return logits


def _load_engram_weights(em: EngramModule, layer_id: int, patch: dict[str, torch.Tensor]) -> int:
    sd = {}
    prefix = f"layer_{layer_id}."
    for k, v in patch.items():
        if k.startswith(prefix):
            sd[k[len(prefix):]] = v
    em.load_state_dict(sd, strict=False)
    return len(sd)


def _resume_from_ckpt(em: EngramModule, layer_id: int, ckpt_path: Path) -> int:
    """Load Engram weights from a training ckpt produced by train_mla.py."""
    ck = safe_torch_load(str(ckpt_path), map_location="cpu")
    engs = ck.get("engram_state_dicts", {})
    sd = engs.get(f"layer_{layer_id}")
    if sd is None:
        return 0
    em.load_state_dict(sd, strict=False)
    return len(sd)


def _load_lm_head_and_final_norm(model_dir: Path, device: str,
                                 dtype: torch.dtype
                                 ) -> tuple[torch.Tensor, torch.Tensor]:
    idx = json.loads((model_dir / "model.safetensors.index.json").read_text())
    wm = idx["weight_map"]
    lmh_shard = wm["lm_head.weight"]
    # Possible names for final norm in our two model loading conventions
    norm_key = None
    for k in ("model.norm.weight", "model.language_model.norm.weight"):
        if k in wm:
            norm_key = k
            break
    if norm_key is None:
        raise RuntimeError("could not find model.norm.weight in index")
    norm_shard = wm[norm_key]

    lmh = load_file(str(model_dir / lmh_shard))["lm_head.weight"]
    norms = load_file(str(model_dir / norm_shard))[norm_key]
    lm_head = lmh.to(device=device, dtype=dtype)
    final_norm = norms.to(device=device, dtype=torch.float32)
    return lm_head, final_norm


def _sample_batch(arr: np.memmap, total_tokens: int, batch_size: int,
                  seq_len: int, rng: np.random.Generator) -> np.ndarray:
    """Return [B, seq_len+1] uint32-as-int64 windows."""
    # Clamp total_tokens to the actual mmap size to avoid index-out-of-bounds
    # when the user's declared total overshoots the file (e.g. due to rounding).
    effective_total = min(total_tokens, arr.shape[0])
    max_start = effective_total - seq_len - 1
    starts = rng.integers(0, max_start, size=batch_size)
    offsets = np.arange(seq_len + 1, dtype=np.int64)
    idx = starts[:, None].astype(np.int64) + offsets[None, :]
    return arr[idx.reshape(-1)].astype(np.int64).reshape(batch_size, seq_len + 1)


def _hash_on_cpu(hasher: NgramHashMapping, input_ids_np: np.ndarray,
                 layer_id: int) -> np.ndarray:
    """Returns [B, L, total_heads] of int64 slot ids."""
    return hasher.hash(input_ids_np, layer_id)


def _format_count(n: int) -> str:
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n/div:.2f}{unit}"
    return str(n)


def train_on_corpus(
    lenses: dict[int, PrefillLens],
    hasher_per_layer: dict[int, NgramHashMapping],
    corpus_path: Path,
    total_tokens: int,
    *,
    seq_len: int,
    batch_size: int,
    steps: int,
    log_every: int,
    save_every: int,
    save_cb,  # callable (step_global) -> None
    step_global_start: int,
    optimizers: Iterable[torch.optim.Optimizer],
    device: str,
    rng_seed: int,
) -> int:
    """Run `steps` optimizer updates on the given corpus. Returns new step_global."""
    arr = np.memmap(str(corpus_path), dtype=np.uint32, mode="r")
    rng = np.random.default_rng(rng_seed)
    step_global = step_global_start

    print(f"\n==== corpus {corpus_path.name}  (total={_format_count(total_tokens)} tokens, "
          f"steps={steps}, batch={batch_size}) ====", flush=True)
    t0 = time.time()
    loss_buf: list[float] = []
    for s in range(steps):
        step_t = time.time()
        step_global += 1
        batch_np = _sample_batch(arr, total_tokens, batch_size, seq_len, rng)  # [B, seq_len+1]
        ctx_np = batch_np[:, :seq_len]           # [B, seq_len]
        # Use EVERY position in each sequence as a training example. One disk
        # access per sequence (vs one per token previously) → ~seq_len× fewer
        # random page-faults on the memmap, which was bottlenecking step time.
        target_np = batch_np[:, 1:seq_len + 1]                                     # [B, seq_len]
        target = torch.from_numpy(target_np.reshape(-1)).to(device, dtype=torch.long)  # [B*seq_len]

        total_loss = 0.0
        # Train each Engram layer independently (they have independent params).
        for li, lens in lenses.items():
            hashes_np = _hash_on_cpu(hasher_per_layer[li], ctx_np, li)   # [B, seq_len, 16]
            B, L, H = hashes_np.shape
            hashes_np = hashes_np.reshape(B * L, H)                      # [B*L, 16]
            hashes = torch.from_numpy(hashes_np).to(device)              # [B*L, 16]
            logits = lens(hashes)                                         # [B*L, V]
            loss = F.cross_entropy(logits, target)
            total_loss = total_loss + loss.item()
            # Per-layer backward accumulates grads into its own params
            loss.backward()

        # Lightweight sanity: check a FEW known-touched rows (via the hash_ids
        # we just computed for this step). Doesn't allocate a full table copy.
        # Only runs at s==0.
        if s == 0:
            _sample_slots = {}
            _pre_values = {}
            for li in lenses:
                # hashes_np was the last layer's — just reuse any small sample
                sample = hashes_np[:4, :4].flatten()  # 16 slot ids
                w = lenses[li].em.embedding.embedding.weight
                _sample_slots[li] = sample
                _pre_values[li] = w[sample].detach().clone()

        for opt in optimizers:
            opt.step()
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        if s == 0:
            for li in lenses:
                w = lenses[li].em.embedding.embedding.weight
                post = w[_sample_slots[li]].detach()
                diff = (post.float() - _pre_values[li].float()).abs()
                print(f"  [post-step-0 sanity] layer {li}  "
                      f"sample max |Δ|={diff.max().item():.2e}  "
                      f"dtype={w.dtype}", flush=True)
            del _sample_slots, _pre_values

        loss_buf.append(total_loss / len(lenses))
        # Early visibility: print first few steps to see per-step timing
        if s < 3 or (s + 1) % log_every == 0:
            elapsed = time.time() - t0
            tokens_done = (s + 1) * batch_size * seq_len
            eta = elapsed / (s + 1) * (steps - s - 1)
            recent = loss_buf[-log_every:]
            step_ms = (time.time() - step_t) * 1000
            print(f"  step {s+1:6d}/{steps}  loss={sum(recent)/len(recent):.4f}  "
                  f"tokens={_format_count(tokens_done)}  "
                  f"step_ms={step_ms:.0f}  "
                  f"elapsed={elapsed:5.0f}s  eta={eta:5.0f}s", flush=True)

        if save_every > 0 and (s + 1) % save_every == 0:
            save_cb(step_global)

    return step_global


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/thearray/git/moe-mla/Qwen3.6-35B-A3B")
    ap.add_argument("--engram-patch-dir",
                    default="/thearray/git/moe-mla/engram_converted_l3_l19",
                    help="patch that defines the Engram architecture "
                         "(vocab sizes, dim, layer ids, hash config).")
    ap.add_argument("--resume-from-ckpt", default="",
                    help="training ckpt to load Engram weights from. "
                         "If empty, use the init-patch weights.")
    ap.add_argument("--corpus", action="append", default=[],
                    help="corpus in PATH:TOTAL_TOKENS form. May be repeated; "
                         "processed in order.")
    ap.add_argument("--seq-len", type=int, default=3,
                    help="context length for each training example. must be "
                         ">= max_ngram_size from the config.")
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--steps-per-corpus", type=int, default=20_000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--momentum", type=float, default=0.0,
                    help="SGD momentum for the embedding tables. 0.9 breaks "
                         "the oscillation we see with lr=1e-3 + no momentum.")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=2000,
                    help="save a checkpoint every N steps within a corpus.")
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--out", default="/thearray/git/moe-mla/engram_prefilled")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16",
                    help="dtype for Engram tables. bf16 halves memory; fp32 is safer.")
    args = ap.parse_args()

    if not args.corpus:
        raise SystemExit("at least one --corpus is required")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    corpora = [_parse_corpus_arg(c) for c in args.corpus]

    # ---- Engram config ----
    manifest = json.loads((Path(args.engram_patch_dir) / "manifest.json").read_text())
    cfg = EngramConfig(**manifest["engram_config"])
    hidden_size = manifest["hidden_size"]
    if args.seq_len < cfg.max_ngram_size:
        raise SystemExit(f"--seq-len ({args.seq_len}) must be >= max_ngram_size "
                         f"({cfg.max_ngram_size}) so the hash at the last position "
                         f"uses a full N-gram.")

    print(f"Engram config:  layers={cfg.layer_ids}  max_n={cfg.max_ngram_size}  "
          f"heads_per_order={cfg.n_head_per_ngram}  vocab/head={cfg.engram_vocab_size[0]:,}  "
          f"d_embed={cfg.n_embed_per_ngram}  disable_compression={cfg.disable_compression}")

    # ---- Build Engram modules (one per layer) ----
    engram_modules: dict[int, EngramModule] = {}
    hasher_per_layer: dict[int, NgramHashMapping] = {}
    for li in cfg.layer_ids:
        em = EngramModule(layer_id=li, cfg=cfg, hidden_size=hidden_size)
        em = em.to(device=args.device, dtype=dtype)
        engram_modules[li] = em
        hasher_per_layer[li] = em.hash_mapping

    # ---- Load weights (patch or resume) ----
    if args.resume_from_ckpt:
        print(f"\nresuming Engram weights from {args.resume_from_ckpt}")
        for li, em in engram_modules.items():
            n = _resume_from_ckpt(em, li, Path(args.resume_from_ckpt))
            print(f"  layer {li}: loaded {n} tensors from ckpt")
    else:
        print(f"\nloading Engram init from {args.engram_patch_dir}/patch.safetensors")
        patch_weights = load_file(str(Path(args.engram_patch_dir) / "patch.safetensors"))
        for li, em in engram_modules.items():
            n = _load_engram_weights(em, li, patch_weights)
            print(f"  layer {li}: loaded {n} tensors from patch")

    # ---- Load lm_head + final_norm (frozen) ----
    print(f"\nloading lm_head + final_norm from {args.model_dir}")
    lm_head_w, final_norm_w = _load_lm_head_and_final_norm(
        Path(args.model_dir), device=args.device, dtype=dtype,
    )
    print(f"  lm_head: {tuple(lm_head_w.shape)}  final_norm: {tuple(final_norm_w.shape)}")

    # ---- Build one PrefillLens per layer, wiring in frozen lm_head + norm ----
    lenses: dict[int, PrefillLens] = {}
    for li, em in engram_modules.items():
        lens = PrefillLens(em, final_norm_w, lm_head_w)
        lenses[li] = lens

    # ---- Configure trainable params ----
    # Only embedding.weight (big, sparse) and value_proj.weight (small, dense).
    # Freeze gating components (key_proj, norm_key, norm_query, short_conv).
    sparse_params: list[nn.Parameter] = []
    dense_params: list[nn.Parameter] = []
    for li, em in engram_modules.items():
        # Embedding table — use sparse=True in forward for sparse grads
        # (torch's nn.Embedding.forward uses sparse path when the module's .sparse=True).
        em.embedding.embedding.sparse = True
        sparse_params.append(em.embedding.embedding.weight)
        em.embedding.embedding.weight.requires_grad_(True)
        # value_proj
        em.value_proj.weight.requires_grad_(True)
        dense_params.append(em.value_proj.weight)
        # Freeze gate components + short_conv
        for mod in (em.key_proj, em.norm_key, em.norm_query, em.short_conv):
            for p in mod.parameters():
                p.requires_grad_(False)

    print(f"\ntrainable:")
    print(f"  sparse (embedding tables): {sum(p.numel() for p in sparse_params)/1e9:.2f}B params")
    print(f"  dense  (value_proj):       {sum(p.numel() for p in dense_params)/1e6:.2f}M params")

    # Plain SGD for the huge sparse embeddings — zero optimizer state, constant
    # per-step time regardless of how many rows have been touched. SparseAdam
    # turned out to degrade catastrophically here (step time grew 150x over 20
    # steps). Adam for the small value_proj (tiny state, no issue).
    opt_sparse = torch.optim.SGD(sparse_params, lr=args.lr, momentum=args.momentum)
    opt_dense = torch.optim.AdamW(dense_params, lr=args.lr, weight_decay=args.weight_decay)

    # ---- Output dir + save helper ----
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    def save_patch(tag: str) -> None:
        """Save the Engram state as a safetensors patch (same format as build_engram_patch.py)."""
        patch: dict[str, torch.Tensor] = {}
        for li, em in engram_modules.items():
            sd = em.state_dict()
            for k, v in sd.items():
                patch[f"layer_{li}.{k}"] = v.detach().contiguous().cpu()
        out_path = out_dir / f"patch_{tag}.safetensors"
        save_file(patch, str(out_path))
        # Current "latest" patch (what train_mla.py will load)
        latest = out_dir / "patch.safetensors"
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(out_path.name)
        print(f"  saved patch: {out_path.name}  ({out_path.stat().st_size/1e9:.2f}GB)"
              f"  ->  patch.safetensors")

    def write_manifest() -> None:
        info = dict(manifest)
        info["output"] = str(out_dir / "patch.safetensors")
        info["prefilled"] = True
        info["prefill_corpora"] = [str(c[0]) for c in corpora]
        info["prefill_total_steps_per_corpus"] = args.steps_per_corpus
        info["prefill_batch_size"] = args.batch_size
        info["prefill_seq_len"] = args.seq_len
        info["prefill_lr"] = args.lr
        info["prefill_resume_from_ckpt"] = args.resume_from_ckpt
        (out_dir / "manifest.json").write_text(json.dumps(info, indent=2))

    # Save the starting state under a tag for provenance
    save_patch("init")
    write_manifest()

    # ---- Train across corpora in order ----
    step_global = 0
    for ci, (path, total) in enumerate(corpora):
        if not path.exists():
            raise SystemExit(f"corpus file not found: {path}")
        step_global = train_on_corpus(
            lenses=lenses,
            hasher_per_layer=hasher_per_layer,
            corpus_path=path,
            total_tokens=total,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            steps=args.steps_per_corpus,
            log_every=args.log_every,
            save_every=args.save_every,
            save_cb=lambda s: save_patch(f"c{ci}_s{s}"),
            step_global_start=step_global,
            optimizers=[opt_sparse, opt_dense],
            device=args.device,
            rng_seed=args.seed + 1000 * ci,
        )
        # End-of-corpus snapshot
        save_patch(f"c{ci}_final_s{step_global}")

    # Final canonical patch
    save_patch(f"final_s{step_global}")
    write_manifest()
    print(f"\nDone. Prefilled patch at: {out_dir / 'patch.safetensors'}")


if __name__ == "__main__":
    main()
