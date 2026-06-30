"""
Eval PPL on the untouched Qwen3.6-35B-A3B, over the exact same held-out slice
that train_mla.py uses. Reports a number directly comparable to the converted-
model baseline eval so we can size how much ground the finetune needs to make up.
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from train_mla import TrainConfig, sample_windows, open_tokens


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/thearray/git/moe-mla/Qwen3.6-35B-A3B")
    ap.add_argument("--eval-batches", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--micro-batch-size", type=int, default=1)
    args = ap.parse_args()

    cfg = TrainConfig(
        seq_len=args.seq_len,
        micro_batch_size=args.micro_batch_size,
        eval_batches=args.eval_batches,
    )

    print(f"loading base model from {args.model_dir} ...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, device_map="cuda:0",
    )
    model.eval()
    print(f"loaded in {time.time()-t0:.0f}s")

    arr, train_end, eval_end = open_tokens(cfg)
    eval_start = train_end
    device = next(model.parameters()).device

    rng = np.random.default_rng(12345)   # same seed as train_mla eval
    total_loss = 0.0
    total_tokens = 0
    t0 = time.time()
    for i in range(cfg.eval_batches):
        ids = sample_windows(arr, eval_start, eval_end, cfg.seq_len,
                             cfg.micro_batch_size, rng).to(device)
        x, y = ids[:, :-1], ids[:, 1:]
        logits = model(input_ids=x).logits
        loss = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            y.reshape(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += y.numel()
        if (i + 1) % 8 == 0:
            print(f"  batch {i+1:3d}/{cfg.eval_batches}  "
                  f"running loss={total_loss/total_tokens:.4f}")

    mean_loss = total_loss / total_tokens
    ppl = math.exp(mean_loss)
    print()
    print(f"base Qwen3.6-35B-A3B  eval_loss={mean_loss:.4f}  ppl={ppl:.3f}")
    print(f"(MLA-converted init was ppl=3.93; dryrun-5steps was ppl=3.82)")
    print(f"elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
