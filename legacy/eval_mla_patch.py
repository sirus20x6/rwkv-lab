"""
Evaluate an MLA patch without any training — load model + patch, run eval on
the same held-out range used in our training runs. Use this to compare:

  original Qwen (eval_original_qwen.py)   ->  best possible
  old-SVD patch + trained Phase 2 ckpt    ->  what we had
  new-BKV patch, UNTRAINED                ->  does BKV alone close the gap?
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import torch
from safe_torch import safe_torch_load


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/thearray/git/moe-mla/Qwen3.6-35B-A3B")
    ap.add_argument("--patch-dir", default="/thearray/git/moe-mla/converted_bkv")
    ap.add_argument("--resume", default="", help="optional trained ckpt to load MLA weights from")
    ap.add_argument("--tokens-bin", default="/thearray/data/non_cvevc_tokens.bin")
    ap.add_argument("--total-tokens-in-bin", type=int, default=29_284_583_603)
    ap.add_argument("--eval-tokens", type=int, default=100_000_000)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--micro-batch-size", type=int, default=1)
    ap.add_argument("--xsa-enabled", type=int, default=0)
    args = ap.parse_args()

    from load_converted import load_converted_model
    print(f"model:  {args.model_dir}")
    print(f"patch:  {args.patch_dir}")
    print(f"resume: {args.resume or '(none — patch-init only)'}")
    print()

    model, mla_modules = load_converted_model(
        model_dir=args.model_dir, patch_dir=args.patch_dir,
        device_map="cuda:0", dtype=torch.bfloat16, freeze_non_mla=True,
        install_mtp=False,
    )

    if args.xsa_enabled:
        for m in mla_modules:
            m.xsa_enabled = True
        print(f"xsa_enabled: toggled on {len(mla_modules)} MLA modules")

    if args.resume:
        print(f"loading MLA state from ckpt...")
        ckpt = safe_torch_load(args.resume, map_location="cpu")
        saved_mla = ckpt.get("mla_state_dicts", {})
        loaded = 0
        for m in mla_modules:
            key = getattr(m, "_save_key", None)
            if key is None or key not in saved_mla:
                continue
            sd = {k: v.to(device=next(m.parameters()).device,
                          dtype=next(m.parameters()).dtype)
                  for k, v in saved_mla[key].items()}
            m.load_state_dict(sd, strict=False)
            loaded += 1
        print(f"  loaded {loaded}/{len(mla_modules)} MLA modules")
        del ckpt

    import torch.nn.functional as F
    @torch.no_grad()
    def do_eval():
        model.eval()
        arr = np.memmap(args.tokens_bin, dtype=np.uint32, mode="r")
        N = args.total_tokens_in_bin
        if N <= 0 or N > int(arr.shape[0]):
            raise ValueError(
                f"total_tokens_in_bin={N} is invalid for {args.tokens_bin}; "
                f"actual length is {int(arr.shape[0])}"
            )
        train_end = N - args.eval_tokens
        if train_end <= 0:
            raise ValueError(
                f"eval_tokens={args.eval_tokens} leaves no training range for N={N}"
            )
        eval_end = N
        rng = np.random.default_rng(12345)
        seqp1 = args.seq_len + 1

        total_ce, total_tok = 0.0, 0
        top1_cnt, top5_cnt = 0, 0
        for b in range(args.eval_batches):
            max_start = eval_end - seqp1
            starts = rng.integers(low=train_end, high=max_start, size=args.micro_batch_size)
            offsets = np.arange(seqp1, dtype=np.int64)
            idx = starts[:, None].astype(np.int64) + offsets[None, :]
            batch = arr[idx.reshape(-1)].astype(np.int64).reshape(args.micro_batch_size, seqp1)
            ids = torch.from_numpy(batch).to("cuda:0")
            x, y = ids[:, :-1], ids[:, 1:]
            logits = model(input_ids=x, use_cache=False).logits
            B, T, V = logits.shape
            flat_l = logits.reshape(-1, V)
            flat_t = y.reshape(-1)
            n = flat_l.shape[0]
            chunk = 2048
            for i in range(0, n, chunk):
                end = min(i + chunk, n)
                lc = flat_l[i:end].float()
                yc = flat_t[i:end]
                total_ce += F.cross_entropy(lc, yc, reduction="sum").item()
                _, top_idx = lc.topk(5, dim=-1)
                matches = (top_idx == yc.unsqueeze(-1))
                cum = matches.cumsum(dim=-1).clamp(max=1)
                top1_cnt += int(cum[:, 0].sum().item())
                top5_cnt += int(cum[:, 4].sum().item())
            total_tok += n
            print(f"  batch {b+1:2d}: running ppl={math.exp(total_ce/total_tok):.4f}  "
                  f"top1={top1_cnt/total_tok*100:.2f}%  top5={top5_cnt/total_tok*100:.2f}%")

        mean = total_ce / total_tok
        return mean, math.exp(mean), top1_cnt / total_tok * 100, top5_cnt / total_tok * 100

    loss, ppl, top1, top5 = do_eval()
    print()
    print(f"MLA patch ({args.patch_dir.split('/')[-1]}):")
    print(f"  loss = {loss:.4f}")
    print(f"  ppl  = {ppl:.4f}")
    print(f"  top1 = {top1:.3f}%")
    print(f"  top5 = {top5:.3f}%")


if __name__ == "__main__":
    main()
