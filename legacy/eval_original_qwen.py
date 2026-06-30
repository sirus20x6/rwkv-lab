"""
Evaluate the ORIGINAL (unconverted) Qwen3.6-35B-A3B on our held-out token range.

No MLA, no MTP, no aux heads, no Engram — just the base GQA model as shipped.
This establishes the absolute ppl ceiling so we can quantify how much the
MLA-swap + Phase 1/2/3 training cost us. Run before/after BKV conversion to
see if BKV is narrowing the gap.

Usage:
    python eval_original_qwen.py --eval-batches 8
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/thearray/git/moe-mla/Qwen3.6-35B-A3B")
    ap.add_argument("--tokens-bin", default="/thearray/data/non_cvevc_tokens.bin")
    ap.add_argument("--total-tokens-in-bin", type=int, default=29_284_583_603)
    ap.add_argument("--eval-tokens", type=int, default=100_000_000,
                    help="same default as TrainConfig so the eval range matches our runs.")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--micro-batch-size", type=int, default=1)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    # Same sampling recipe as train_mla.eval_loss: deterministic seed 12345 rng,
    # windows from [train_end, eval_end) where train_end = N - eval_tokens.
    arr = np.memmap(args.tokens_bin, dtype=np.uint32, mode="r")
    N = args.total_tokens_in_bin
    train_end = N - args.eval_tokens
    eval_end = N
    rng = np.random.default_rng(12345)

    print(f"model:  {args.model_dir}")
    print(f"data:   train_end={train_end/1e9:.2f}B  eval_range=[{train_end/1e9:.2f}B, {eval_end/1e9:.2f}B)")
    print(f"eval:   {args.eval_batches} batches × {args.seq_len} tokens = "
          f"{args.eval_batches * args.seq_len * args.micro_batch_size / 1e3:.1f}K tokens")
    print()

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

    import torch.nn.functional as F

    def chunked_ce(logits, labels, chunk=2048):
        B, T, V = logits.shape
        flat_l = logits.reshape(-1, V)
        flat_t = labels.reshape(-1)
        n = flat_l.shape[0]
        total_sum = 0.0
        for i in range(0, n, chunk):
            end = min(i + chunk, n)
            lc = flat_l[i:end].float()
            total_sum += F.cross_entropy(lc, flat_t[i:end], reduction="sum").item()
        return total_sum, n

    @torch.no_grad()
    def chunked_topk_counts(logits, labels, chunk=2048, topk=(1, 5)):
        B, T, V = logits.shape
        flat_l = logits.reshape(-1, V)
        flat_t = labels.reshape(-1)
        n = flat_l.shape[0]
        max_k = max(topk)
        counts = {k: 0 for k in topk}
        for i in range(0, n, chunk):
            end = min(i + chunk, n)
            lc = flat_l[i:end]
            yc = flat_t[i:end]
            _, top_idx = lc.topk(max_k, dim=-1)
            matches = (top_idx == yc.unsqueeze(-1))
            cum = matches.cumsum(dim=-1).clamp(max=1)
            for k in topk:
                counts[k] += int(cum[:, k - 1].sum().item())
        return counts, n

    total_loss_sum = 0.0
    total_tokens = 0
    top_counts_agg = {1: 0, 5: 0}

    seqp1 = args.seq_len + 1
    t0 = time.time()
    with torch.no_grad():
        for b in range(args.eval_batches):
            max_start = eval_end - seqp1
            starts = rng.integers(low=train_end, high=max_start, size=args.micro_batch_size)
            offsets = np.arange(seqp1, dtype=np.int64)
            idx = starts[:, None].astype(np.int64) + offsets[None, :]
            batch = arr[idx.reshape(-1)].astype(np.int64).reshape(args.micro_batch_size, seqp1)
            ids = torch.from_numpy(batch).to(args.device)
            x, y = ids[:, :-1], ids[:, 1:]

            logits = model(input_ids=x).logits
            loss_sum, ntoks = chunked_ce(logits, y)
            total_loss_sum += loss_sum
            total_tokens += ntoks
            counts, _ = chunked_topk_counts(logits, y)
            for k in (1, 5):
                top_counts_agg[k] += counts[k]
            print(f"  batch {b+1:2d}/{args.eval_batches}: ce_sum={loss_sum:.1f}  "
                  f"tok={ntoks}  top1={counts[1]/ntoks*100:.2f}%  top5={counts[5]/ntoks*100:.2f}%")

    mean_loss = total_loss_sum / total_tokens
    ppl = math.exp(mean_loss)
    top1 = top_counts_agg[1] / total_tokens * 100
    top5 = top_counts_agg[5] / total_tokens * 100
    dt = time.time() - t0
    print()
    print(f"ORIGINAL Qwen3.6-35B-A3B (no MLA/MTP/aux/Engram):")
    print(f"  eval_loss = {mean_loss:.4f}")
    print(f"  ppl       = {ppl:.4f}")
    print(f"  top1      = {top1:.3f}%")
    print(f"  top5      = {top5:.3f}%")
    print(f"  ({total_tokens} tokens in {dt:.0f}s)")


if __name__ == "__main__":
    main()
