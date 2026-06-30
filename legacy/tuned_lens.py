"""
Tuned LogitLens sweep on the ORIGINAL (unconverted) Qwen3.6-35B-A3B.

Naive LogitLens assumes every layer's hidden state lives in the same basis
as the final `lm_head`. For Qwen3.6 this is false for the middle layers:
linear-attention outputs and MoE routing contexts are orthogonal to the
unembedding directions, so naive LogitLens shows mid-layer KL growing to
~14 nats (worse than uniform) which is a reading artifact, not a model bug.

The Tuned Lens (Belrose et al., arXiv:2303.08112) fixes this by learning
a per-layer linear transform A_l so that
    lm_head(final_norm(A_l @ h_l))
approximates the final-layer distribution at each depth. With tuned lens
the KL trajectory becomes monotonic and the honest "where does decodable
information first appear" signal is recoverable.

We do two things here:
1. Train one Linear(H, H) "lens" per layer by minimizing
   KL(final_softmax || lens_softmax) on calibration data.
   Lenses are initialized to identity — at step 0 they match naive LogitLens.
2. Re-run the KL/CE/top1/top5 sweep using the trained lenses and compare
   against naive LogitLens to see which layers are actually doing
   vocab-aligned work.

Output:
    tuned_lens_{name}.pt — dict with:
        lenses:          {l: {"weight": Tensor[H,H], "bias": Tensor[H]}}
        naive_sweep:     dict from logit_lens_sweep (or recomputed)
        tuned_sweep:     same structure as naive_sweep but using trained lenses
        num_layers, seq_len, nsamples_train, nsamples_eval, model_dir

Usage:
    python tuned_lens.py \\
        --nsamples-train 64 --nsamples-eval 32 --epochs 3 \\
        --out tuned_lens_original.pt
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_lens(H: int, dtype: torch.dtype, device: str) -> nn.Linear:
    m = nn.Linear(H, H, bias=True)
    nn.init.eye_(m.weight)
    nn.init.zeros_(m.bias)
    return m.to(device=device, dtype=dtype)


def _cal_windows(arr, eval_start: int, total: int, seq_len: int,
                 nsamples: int, rng) -> torch.Tensor:
    seqp1 = seq_len + 1
    starts = rng.integers(low=eval_start, high=total - seqp1, size=nsamples)
    offsets = np.arange(seqp1, dtype=np.int64)
    idx = starts[:, None].astype(np.int64) + offsets[None, :]
    out = arr[idx.reshape(-1)].astype(np.int64).reshape(nsamples, seqp1)
    return torch.from_numpy(out)


def _sweep_stats(logits_l: torch.Tensor, final_logits: torch.Tensor,
                 labels: torch.Tensor, chunk: int):
    """Compute KL(final||layer), CE vs labels, top1/top5 counts per chunk.
    Returns (sum_kl, sum_ce, top1, top5, n_tok)."""
    V = logits_l.shape[-1]
    flat_l = logits_l.reshape(-1, V)
    flat_f = final_logits.reshape(-1, V)
    flat_y = labels.reshape(-1)
    n_tok = flat_y.numel()
    sum_kl = 0.0
    sum_ce = 0.0
    top1 = 0
    top5 = 0
    for i in range(0, n_tok, chunk):
        end = min(i + chunk, n_tok)
        lc = flat_l[i:end].float()
        fc = flat_f[i:end].float()
        yc = flat_y[i:end]

        lc_logp = F.log_softmax(lc, dim=-1)
        fc_logp = F.log_softmax(fc, dim=-1)
        fc_p = fc_logp.exp()
        sum_kl += (fc_p * (fc_logp - lc_logp)).sum(dim=-1).sum().item()
        sum_ce += F.cross_entropy(lc, yc, reduction="sum").item()

        _, top_idx = lc.topk(5, dim=-1)
        matches = (top_idx == yc.unsqueeze(-1))
        cum = matches.cumsum(dim=-1).clamp(max=1)
        top1 += int(cum[:, 0].sum().item())
        top5 += int(cum[:, 4].sum().item())
    return sum_kl, sum_ce, top1, top5, n_tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/thearray/git/moe-mla/Qwen3.6-35B-A3B")
    ap.add_argument("--tokens-bin", default="/thearray/data/non_cvevc_tokens.bin")
    ap.add_argument("--total-tokens-in-bin", type=int, default=29_284_583_603)
    ap.add_argument("--nsamples-train", type=int, default=64,
                    help="calibration samples for LENS TRAINING (~131K tokens at 2048)")
    ap.add_argument("--nsamples-eval", type=int, default=32,
                    help="held-out samples for final KL/top1 sweep")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--chunk", type=int, default=2048,
                    help="flat tokens per chunk for lm_head materialization")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16",
                    help="model dtype. lens params are always fp32 for stable Adam.")
    ap.add_argument("--seed-train", type=int, default=54321,
                    help="rng seed for TRAINING windows. Eval uses 12345 to match "
                         "eval_original_qwen.py and logit_lens_sweep.py.")
    ap.add_argument("--out", default="/thearray/git/moe-mla/tuned_lens_original.pt")
    args = ap.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    # ---- data ----
    arr = np.memmap(args.tokens_bin, dtype=np.uint32, mode="r")
    N = args.total_tokens_in_bin
    eval_start = N - 100_000_000
    rng_train = np.random.default_rng(args.seed_train)
    rng_eval  = np.random.default_rng(12345)

    cal_train = _cal_windows(arr, eval_start, N, args.seq_len, args.nsamples_train, rng_train)
    cal_eval  = _cal_windows(arr, eval_start, N, args.seq_len, args.nsamples_eval,  rng_eval)

    print(f"model:      {args.model_dir}")
    print(f"samples:    train={args.nsamples_train}  eval={args.nsamples_eval}  "
          f"seq_len={args.seq_len}")
    print(f"epochs:     {args.epochs}  lr={args.lr}")

    # ---- load model ----
    print("loading model...")
    t0 = time.time()
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=dtype, device_map=args.device,
        trust_remote_code=True, attn_implementation="sdpa",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"  loaded in {time.time()-t0:.1f}s")

    text = getattr(model.model, "language_model", model.model)
    final_norm = text.norm
    lm_head = model.lm_head
    num_layers = len(text.layers)
    H = lm_head.weight.shape[1]
    V = lm_head.weight.shape[0]
    print(f"num_layers: {num_layers}  hidden_size: {H}  vocab_size: {V}")
    print(f"# lens states: {num_layers + 1} (embed + {num_layers} layer outputs)")
    print()

    # ---- build lenses (identity init = naive LogitLens at step 0) ----
    # Keep lens params in fp32 for stable Adam; cast to bf16 only for the
    # forward multiply (matches model dtype).
    n_lens = num_layers + 1
    lenses = nn.ModuleList([
        _build_lens(H, dtype=torch.float32, device=args.device) for _ in range(n_lens)
    ])
    # Single shared optimizer over all lens params. Each lens is independent
    # in the compute graph, so param grads don't interfere.
    opt = torch.optim.AdamW(
        [p for L in lenses for p in L.parameters()],
        lr=args.lr, weight_decay=args.weight_decay, fused=False,
    )

    # ---- helper: run tuned lens through final_norm + lm_head, chunked ----
    # Each call does a full-token-range forward for ONE layer and returns KL loss
    # (scalar), with backward handled internally to avoid storing [B*T, V] grads.
    def train_step_for_layer(h_l: torch.Tensor, final_logits_flat: torch.Tensor,
                             lens: nn.Linear) -> float:
        """Train one lens on one batch. Returns mean KL on this batch.
        h_l: [B*T, H]  (detached from model)
        final_logits_flat: [B*T, V]  (detached from model, fp-any)
        """
        n_tok = h_l.shape[0]
        total_kl = 0.0
        for i in range(0, n_tok, args.chunk):
            end = min(i + args.chunk, n_tok)
            h_chunk_fp32 = h_l[i:end].float()
            # Lens in fp32: A @ h + b
            lens_out = lens(h_chunk_fp32)
            # final_norm is RMSNorm; do it in bf16 to match model
            lens_out_bf = lens_out.to(dtype)
            lens_normed = final_norm(lens_out_bf)
            ll = lm_head(lens_normed).float()  # [chunk, V]

            fc = final_logits_flat[i:end].float().detach()
            fc_logp = F.log_softmax(fc, dim=-1)
            fc_p = fc_logp.exp()
            ll_logp = F.log_softmax(ll, dim=-1)
            kl = (fc_p * (fc_logp - ll_logp)).sum(dim=-1).mean()

            # Accumulate grads chunk-by-chunk. Scale by chunk coverage.
            (kl * (end - i) / n_tok).backward()
            total_kl += kl.item() * (end - i) / n_tok
            del lens_out, lens_out_bf, lens_normed, ll, fc, fc_logp, fc_p, ll_logp
        return total_kl

    # ---- train ----
    print("training lenses (identity init → learned transform)...")
    t0 = time.time()
    for epoch in range(args.epochs):
        ep_kl_sum = np.zeros(n_lens, dtype=np.float64)
        ep_n = 0
        for s in range(args.nsamples_train):
            ids = cal_train[s:s+1].to(args.device)
            x = ids[:, :-1]
            with torch.no_grad():
                out = model(input_ids=x, output_hidden_states=True)
            hs = out.hidden_states  # tuple of n_lens tensors [1, T, H]
            final_logits = out.logits.detach()  # [1, T, V]
            del out

            assert len(hs) == n_lens, (
                f"expected {n_lens} hidden states, got {len(hs)}"
            )
            flat_final = final_logits.reshape(-1, V)
            opt.zero_grad(set_to_none=True)
            for l in range(n_lens):
                h_flat = hs[l].reshape(-1, H).detach()
                kl_mean = train_step_for_layer(h_flat, flat_final, lenses[l])
                ep_kl_sum[l] += kl_mean
            opt.step()
            ep_n += 1
            del hs, final_logits, flat_final, x, ids

            elapsed = time.time() - t0
            samples_done = epoch * args.nsamples_train + (s + 1)
            samples_total = args.epochs * args.nsamples_train
            eta = elapsed / samples_done * (samples_total - samples_done)
            if (s + 1) % 8 == 0 or s == args.nsamples_train - 1:
                print(f"  epoch {epoch+1}/{args.epochs} sample {s+1:3d}/{args.nsamples_train}  "
                      f"elapsed={elapsed:5.0f}s  eta={eta:5.0f}s  "
                      f"mean_kl={ep_kl_sum.mean()/ep_n:.3f}")
            torch.cuda.empty_cache()

        # End-of-epoch per-layer summary
        print(f"  epoch {epoch+1} per-layer mean KL (first 5, ..., last 5):")
        avg = ep_kl_sum / ep_n
        idx_sample = list(range(5)) + list(range(n_lens - 5, n_lens))
        for l in idx_sample:
            label = ("embed" if l == 0 else
                    (f"L{l-1}" if l < n_lens - 1 else "final"))
            print(f"    {label:>5s}  kl={avg[l]:.3f}")

    print(f"lens training done in {time.time()-t0:.1f}s")
    print()

    # ---- eval sweep (held-out samples, different seed) ----
    print(f"running eval sweep on {args.nsamples_eval} held-out samples...")
    t0 = time.time()
    sums_kl_naive = np.zeros(n_lens, dtype=np.float64)
    sums_ce_naive = np.zeros(n_lens, dtype=np.float64)
    sums_t1_naive = np.zeros(n_lens, dtype=np.int64)
    sums_t5_naive = np.zeros(n_lens, dtype=np.int64)
    sums_kl_tuned = np.zeros(n_lens, dtype=np.float64)
    sums_ce_tuned = np.zeros(n_lens, dtype=np.float64)
    sums_t1_tuned = np.zeros(n_lens, dtype=np.int64)
    sums_t5_tuned = np.zeros(n_lens, dtype=np.int64)
    total_tok = 0

    for L in lenses:
        L.eval()

    with torch.no_grad():
        for s in range(args.nsamples_eval):
            ids = cal_eval[s:s+1].to(args.device)
            x, y = ids[:, :-1], ids[:, 1:]
            out = model(input_ids=x, output_hidden_states=True)
            hs = out.hidden_states
            final_logits = out.logits
            for l in range(n_lens):
                h = hs[l]
                h_flat = h.reshape(-1, H)
                # naive: h passed through final_norm (skip if last, since model did it)
                if l == n_lens - 1:
                    h_naive = h  # already normed
                else:
                    h_naive = final_norm(h)
                logits_naive = lm_head(h_naive)

                # tuned: A @ h + b, then final_norm, then lm_head
                h_tuned = lenses[l](h_flat.float()).to(dtype).reshape(h.shape)
                h_tuned = final_norm(h_tuned)
                logits_tuned = lm_head(h_tuned)

                kl_n, ce_n, t1_n, t5_n, n_tok = _sweep_stats(
                    logits_naive, final_logits, y, args.chunk)
                kl_t, ce_t, t1_t, t5_t, _ = _sweep_stats(
                    logits_tuned, final_logits, y, args.chunk)

                sums_kl_naive[l] += kl_n; sums_ce_naive[l] += ce_n
                sums_t1_naive[l] += t1_n; sums_t5_naive[l] += t5_n
                sums_kl_tuned[l] += kl_t; sums_ce_tuned[l] += ce_t
                sums_t1_tuned[l] += t1_t; sums_t5_tuned[l] += t5_t
                del logits_naive, logits_tuned, h_naive, h_tuned
            total_tok += n_tok

            elapsed = time.time() - t0
            eta = elapsed / (s + 1) * (args.nsamples_eval - s - 1)
            if (s + 1) % 4 == 0:
                print(f"  eval sample {s+1:3d}/{args.nsamples_eval}  "
                      f"elapsed={elapsed:5.0f}s  eta={eta:5.0f}s")
            del hs, final_logits, out, x, y, ids
            torch.cuda.empty_cache()

    # ---- summarize ----
    kl_n_per = sums_kl_naive / total_tok
    kl_t_per = sums_kl_tuned / total_tok
    ce_n_per = sums_ce_naive / total_tok
    ce_t_per = sums_ce_tuned / total_tok
    t1_n_per = sums_t1_naive / total_tok
    t5_n_per = sums_t5_naive / total_tok
    t1_t_per = sums_t1_tuned / total_tok
    t5_t_per = sums_t5_tuned / total_tok

    print()
    print(f"{'layer':>6s}  {'kl_naive':>9s}  {'kl_tuned':>9s}  "
          f"{'ΔKL':>7s}  {'top1_n':>7s}  {'top1_t':>7s}")
    print("-" * 60)
    for l in range(n_lens):
        label = ("embed" if l == 0 else
                (f"L{l-1}" if l < n_lens - 1 else "final"))
        dkl = kl_n_per[l] - kl_t_per[l]
        print(f"  {label:>4s}  {kl_n_per[l]:>9.3f}  {kl_t_per[l]:>9.3f}  "
              f"{dkl:>7.3f}  {t1_n_per[l]*100:>6.2f}%  {t1_t_per[l]*100:>6.2f}%")

    out_data = {
        "lenses": {l: {k: v.detach().cpu() for k, v in lenses[l].state_dict().items()}
                   for l in range(n_lens)},
        "naive_sweep": {
            "kl_per_layer":   kl_n_per,
            "ce_per_layer":   ce_n_per,
            "top1_per_layer": t1_n_per,
            "top5_per_layer": t5_n_per,
        },
        "tuned_sweep": {
            "kl_per_layer":   kl_t_per,
            "ce_per_layer":   ce_t_per,
            "top1_per_layer": t1_t_per,
            "top5_per_layer": t5_t_per,
        },
        "num_layers":     num_layers,
        "hidden_size":    H,
        "vocab_size":     V,
        "seq_len":        args.seq_len,
        "nsamples_train": args.nsamples_train,
        "nsamples_eval":  args.nsamples_eval,
        "epochs":         args.epochs,
        "lr":             args.lr,
        "model_dir":      args.model_dir,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_data, out_path)
    print(f"\nwrote: {out_path}")

    # --- Engram-placement: rank layers by tuned KL drop, no early-bias ---
    # With a tuned lens, KL is a real measure of "how much of the final
    # prediction is recoverable at this depth". Largest ΔKL (over layer
    # boundary) = where new decodable information is being composed.
    kl_drops_tuned = -np.diff(kl_t_per)
    print()
    print("Engram-placement candidates — layers with largest TUNED-KL drop:")
    top = np.argsort(kl_drops_tuned)[::-1][:10]
    for rank, li in enumerate(top):
        print(f"  rank {rank+1}: layer {li} "
              f"(tuned_kl_drop {kl_drops_tuned[li]:.3f}, "
              f"from {kl_t_per[li]:.3f} to {kl_t_per[li+1]:.3f})")


if __name__ == "__main__":
    main()
