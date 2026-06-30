"""
LogitLens sweep on the ORIGINAL (unconverted) Qwen3.6-35B-A3B.

For each decoder layer, project its output hidden state through the model's
final norm + lm_head, compute per-token KL divergence vs the final
distribution. Averages across calibration samples.

Why:
    Engram paper §6.1.1 shows that Engram reduces KL divergence most strongly
    in EARLY layers. Reading that direction: layers with high-but-descending
    KL are doing "static reconstruction" — composing predictions from local
    context. Those are the best Engram insertion points because Engram's
    static N-gram memory can OFFLOAD that reconstruction, freeing the
    backbone to do deeper reasoning.

    Paper §6.2's 12-layer sweep showed single-injection optimum at layer 2,
    monotonic degradation deeper. LogitLens gives us the same insight more
    cheaply: look for the layer where KL starts its steepest drop (i.e.
    where "feature composition" activity is highest) — that's the Engram
    insertion point.

Output:
    logit_lens_{name}.pt — dict with:
        kl_per_layer:     [num_layers+1] mean KL across tokens/samples
        ce_per_layer:     [num_layers+1] mean CE vs next-token labels
        top1_per_layer:   [num_layers+1] accuracy at each layer
        top5_per_layer:   [num_layers+1] top-5 accuracy
        num_samples, seq_len, model_dir

Usage:
    python logit_lens_sweep.py \\
        --nsamples 32 --seq-len 2048 \\
        --out logit_lens_original.pt
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/thearray/git/moe-mla/Qwen3.6-35B-A3B")
    ap.add_argument("--tokens-bin", default="/thearray/data/non_cvevc_tokens.bin")
    ap.add_argument("--total-tokens-in-bin", type=int, default=29_284_583_603)
    ap.add_argument("--nsamples", type=int, default=32,
                    help="number of calibration samples. 32 is ~65K tokens, "
                         "usually enough to get stable per-layer curves.")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--logits-chunk", type=int, default=2048,
                    help="flat tokens per chunk for KL/CE computation. "
                         "Controls peak fp32 [chunk, V] materialization.")
    ap.add_argument("--out", default="/thearray/git/moe-mla/logit_lens_original.pt")
    args = ap.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    # ---- calibration windows ----
    arr = np.memmap(args.tokens_bin, dtype=np.uint32, mode="r")
    N = args.total_tokens_in_bin
    eval_start = N - 100_000_000  # same held-out tail we use for all evals
    rng = np.random.default_rng(args.seed)

    seqp1 = args.seq_len + 1
    starts = rng.integers(low=eval_start, high=N - seqp1, size=args.nsamples)
    offsets = np.arange(seqp1, dtype=np.int64)
    idx = starts[:, None].astype(np.int64) + offsets[None, :]
    cal = arr[idx.reshape(-1)].astype(np.int64).reshape(args.nsamples, seqp1)
    cal_tensor = torch.from_numpy(cal)

    print(f"model:     {args.model_dir}")
    print(f"samples:   {args.nsamples} × seq_len {args.seq_len} = "
          f"{args.nsamples * args.seq_len / 1e3:.1f}K tokens")

    # ---- load model ----
    print("loading model...")
    t0 = time.time()
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=dtype,
        device_map=args.device,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s")

    # ---- resolve backbone (text) submodule ----
    text = getattr(model.model, "language_model", model.model)
    final_norm = text.norm
    lm_head = model.lm_head
    num_layers = len(text.layers)
    print(f"num_layers: {num_layers}  (output will have {num_layers + 1} entries "
          f"— one per layer output plus one for the norm'd final)")

    V = lm_head.weight.shape[0]
    print(f"vocab_size: {V}")
    print()

    # ---- sweep ----
    # Accumulators [num_layers + 1]. Index 0 is AFTER layer 0, ..., index num_layers-1
    # is AFTER layer num_layers-1 (pre final norm), and index num_layers is
    # POST final norm (== last_hidden_state). Plus an embedded state at index -1?
    # HF convention returns all_hidden_states of length num_layers + 1, with
    # [0] being the embedded input BEFORE any layer. So we end up with
    # num_layers + 1 total entries.
    sums_kl = np.zeros(num_layers + 1, dtype=np.float64)
    sums_ce = np.zeros(num_layers + 1, dtype=np.float64)
    sums_top1 = np.zeros(num_layers + 1, dtype=np.int64)
    sums_top5 = np.zeros(num_layers + 1, dtype=np.int64)
    total_tok = 0

    t0 = time.time()
    with torch.no_grad():
        for s in range(args.nsamples):
            full = cal_tensor[s:s+1].to(args.device)
            x = full[:, :-1]
            y = full[:, 1:]
            B, T = x.shape

            out = model(input_ids=x, output_hidden_states=True)
            # hidden_states: tuple of (num_layers + 1) tensors of shape [B, T, H]
            #   [0]: embedded input
            #   [k]: after layer k-1 (pre-final-norm)
            #   [-1]: after final norm (== last_hidden_state)
            # BUT: HF layers sometimes return slightly different structure.
            # Verify by checking hidden_states[-1] matches last_hidden_state.
            hs = out.hidden_states
            assert len(hs) == num_layers + 1, \
                f"expected {num_layers+1} hidden states, got {len(hs)}"
            # Final layer logits (ground-truth distribution)
            final_logits = out.logits  # [B, T, V] bf16

            # For each hidden state, compute KL vs final + CE vs labels.
            flat_final = final_logits.reshape(-1, V)
            flat_y = y.reshape(-1)
            n_tok = flat_y.numel()

            # Compute final log-probs in chunks to bound fp32 memory
            # Then for each layer's logits, we compute KL and CE in the same chunks.
            for li in range(num_layers + 1):
                h = hs[li]  # [B, T, H]
                # Final index (li == num_layers) is already norm'd by the model.
                # All earlier ones are pre-norm → apply final_norm.
                if li == num_layers:
                    h_normed = h  # already done by model
                else:
                    h_normed = final_norm(h)
                # Project to vocab
                li_logits = lm_head(h_normed)  # [B, T, V]
                flat_li = li_logits.reshape(-1, V)

                # Accumulate per-chunk stats
                layer_kl = 0.0
                layer_ce = 0.0
                layer_top1 = 0
                layer_top5 = 0
                chunk = args.logits_chunk
                for i in range(0, n_tok, chunk):
                    end = min(i + chunk, n_tok)
                    lc = flat_li[i:end].float()
                    fc = flat_final[i:end].float()
                    yc = flat_y[i:end]

                    lc_logp = F.log_softmax(lc, dim=-1)
                    fc_logp = F.log_softmax(fc, dim=-1)
                    fc_p = fc_logp.exp()
                    # KL(final || layer) = sum_v final_p * (final_logp - layer_logp)
                    kl = (fc_p * (fc_logp - lc_logp)).sum(dim=-1).sum().item()
                    layer_kl += kl

                    # CE vs true labels
                    ce = F.cross_entropy(lc, yc, reduction="sum").item()
                    layer_ce += ce

                    # Top-k from layer distribution
                    _, top_idx = lc.topk(5, dim=-1)
                    matches = (top_idx == yc.unsqueeze(-1))
                    cum = matches.cumsum(dim=-1).clamp(max=1)
                    layer_top1 += int(cum[:, 0].sum().item())
                    layer_top5 += int(cum[:, 4].sum().item())

                    del lc, fc, yc, lc_logp, fc_logp, fc_p

                sums_kl[li] += layer_kl
                sums_ce[li] += layer_ce
                sums_top1[li] += layer_top1
                sums_top5[li] += layer_top5
                del li_logits, flat_li, h_normed
            total_tok += n_tok
            del hs, final_logits, flat_final, flat_y, out
            torch.cuda.empty_cache()

            elapsed = time.time() - t0
            eta = elapsed / (s + 1) * (args.nsamples - s - 1)
            print(f"  sample {s+1:3d}/{args.nsamples}  elapsed={elapsed:5.0f}s  eta={eta:5.0f}s")

    # ---- Results ----
    kl_per_layer = sums_kl / total_tok
    ce_per_layer = sums_ce / total_tok
    top1_per_layer = sums_top1 / total_tok
    top5_per_layer = sums_top5 / total_tok

    print()
    print(f"{'layer':>6s}  {'kl':>10s}  {'ce':>8s}  {'ppl':>8s}  {'top1':>8s}  {'top5':>8s}")
    print("-" * 60)
    # Index 0 = embedded input (before any layer); 1 = after layer 0; ...; N = after final norm
    for li in range(num_layers + 1):
        label = "embed" if li == 0 else (f"L{li-1}" if li < num_layers else "final")
        print(f"  {label:>4s}  {kl_per_layer[li]:>10.4f}  "
              f"{ce_per_layer[li]:>8.4f}  "
              f"{np.exp(ce_per_layer[li]):>8.4f}  "
              f"{top1_per_layer[li]*100:>7.2f}%  "
              f"{top5_per_layer[li]*100:>7.2f}%")

    out_data = {
        "kl_per_layer": kl_per_layer,
        "ce_per_layer": ce_per_layer,
        "top1_per_layer": top1_per_layer,
        "top5_per_layer": top5_per_layer,
        "num_samples": args.nsamples,
        "seq_len": args.seq_len,
        "model_dir": args.model_dir,
        "num_layers": num_layers,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_data, out_path)
    print(f"\nwrote: {out_path}")

    # --- Engram-placement recommendations ---
    # Best Engram insertion: early layer where the KL drop per layer is steepest
    # (that's where static reconstruction is doing the most work).
    kl_drops = -np.diff(kl_per_layer)  # layer-over-layer drop (positive = improving)
    # Score = magnitude of drop, weighted by how early the layer is (prefer early).
    # Paper §6.2 shows best at layer 2 of 12, so early_weight = 1 / layer_idx.
    print()
    print("Engram-placement candidates — layers with largest KL drop (early preferred):")
    top_drops = np.argsort(kl_drops)[::-1][:10]
    for rank, li in enumerate(top_drops):
        early_bonus = 1.0 / (li + 1)
        score = kl_drops[li] * early_bonus
        print(f"  rank {rank+1}: layer {li} (kl drop {kl_drops[li]:.4f}, "
              f"early_bonus {early_bonus:.3f}, score {score:.4f})")


if __name__ == "__main__":
    main()
