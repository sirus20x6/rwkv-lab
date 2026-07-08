#!/usr/bin/env python
"""Build Qwen3.5-tokenized DCLM + FineWeb-Edu caches for the GDN->RWKV conversion.

Adapted from babyllm/scripts/build_dclm_fwedu.py, with the tweaks the SMT/DMT
pipeline needs (see plan glistening-inventing-garden.md, Stage 0.5):

  * Tokenizer: the exact Qwen3.5-9B-Base tokenizer (id-identical to qwen3.6, but
    we use the real one for exactness).
  * Format: ONE flat uint32 `tokens.bin` per split (train/val/test x dclm/fwedu),
    directly memmap-loadable by train_mla.py's `cfg.tokens_bin`
    (np.memmap(..., dtype=np.uint32)).
  * Document boundaries: the Qwen3.5 EOS id is inserted BETWEEN documents, and a
    sidecar `doc_offsets.bin` (uint64) records the token index where each document
    begins in the flat stream. This lets SMT/DMT rollout windows stay within a
    single document so the teacher recurrent-state targets are not contaminated by
    cross-document context.

Splits are carved test -> val -> train from each pre-shuffled stream (disjoint),
matching the babyllm convention.

Full run streams ~1B+ tokens over the network; run it in the background or
delegate it. Use --max-docs for a fast format smoke test. The writer logic is
covered by test_build_qwen35_data.py (no network).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

DEFAULT_TOKENIZER = "/thearray/git/moe-mla/Qwen3.5-9B-Base"


def flat_writer(out_dir: Path, eos_id: int):
    """Stateful writer: feed whole documents, it streams a flat uint32 token file
    plus a uint64 doc-offset sidecar. EOS is appended after every document."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tok_f = open(out_dir / "tokens.bin", "wb")
    off_f = open(out_dir / "doc_offsets.bin", "wb")
    state = {"total": 0, "ndocs": 0}

    def feed_doc(ids):
        # record where this doc starts, then write ids followed by one EOS
        np.asarray([state["total"]], dtype=np.uint64).tofile(off_f)
        np.asarray(ids, dtype=np.uint32).tofile(tok_f)
        np.asarray([eos_id], dtype=np.uint32).tofile(tok_f)
        state["total"] += len(ids) + 1
        state["ndocs"] += 1

    def finalize(tok_label, vocab_size, source):
        tok_f.flush(); tok_f.close()
        off_f.flush(); off_f.close()
        manifest = {
            "tokenizer": tok_label,
            "vocab_size": int(vocab_size),
            "eos_id": int(eos_id),
            "source": source,
            "format": "flat",
            "dtype": "uint32",
            "tokens_file": "tokens.bin",
            "doc_offsets_file": "doc_offsets.bin",
            "doc_offsets_dtype": "uint64",
            "total_tokens": int(state["total"]),
            "n_docs": int(state["ndocs"]),
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return manifest

    return feed_doc, finalize, state


def build_source(repo, field, out_root, tok, tok_label, eos_id,
                 train_tok, val_tok, test_tok, batch_docs, prefix, max_docs):
    from datasets import load_dataset

    vocab = int(len(tok))  # len(), not vocab_size: covers added/special ids
    splits = {
        "test": flat_writer(out_root / f"{prefix}_test", eos_id),
        "val": flat_writer(out_root / f"{prefix}_val", eos_id),
        "train": flat_writer(out_root / f"{prefix}_train", eos_id),
    }
    budgets = {"test": test_tok, "val": val_tok, "train": train_tok}
    order = ["test", "val", "train"]
    cur = 0

    def route(ids_list):
        nonlocal cur
        for ids in ids_list:
            while cur < len(order):
                feed_doc, _, state = splits[order[cur]]
                if state["total"] >= budgets[order[cur]]:
                    cur += 1
                    continue
                feed_doc(ids)
                break
            else:
                return False  # all splits full
        return True

    ds = load_dataset(repo, split="train", streaming=True)
    t0 = time.time()
    buf, seen, done = [], 0, False
    for ex in ds:
        buf.append(ex[field]); seen += 1
        if max_docs and seen >= max_docs:
            enc = tok(buf, add_special_tokens=False)["input_ids"]
            route(enc); done = True; break
        if len(buf) >= batch_docs:
            enc = tok(buf, add_special_tokens=False)["input_ids"]
            if not route(enc):
                done = True
            buf = []
            if done:
                break
            if seen % (batch_docs * 20) == 0:
                tot = sum(s[2]["total"] for s in splits.values())
                dt = time.time() - t0
                print(f"  [{prefix}] {seen} docs, {tot/1e6:.1f}M tok, "
                      f"{tot/max(dt,1e-9)/1e6:.2f}M tok/s", flush=True)
    if buf and not done:
        enc = tok(buf, add_special_tokens=False)["input_ids"]
        route(enc)

    out = {}
    for name in order:
        _, finalize, _ = splits[name]
        m = finalize(tok_label, vocab, f"{repo}:{name}")
        out[name] = m["total_tokens"]
        print(f"  [{prefix}/{name}] {m['total_tokens']/1e6:.1f}M tok, "
              f"{m['n_docs']} docs", flush=True)
    print(f"  [{prefix}] DONE in {time.time()-t0:.0f}s, {seen} docs", flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer_path", default=DEFAULT_TOKENIZER)
    p.add_argument("--label", default="qwen3.5")
    p.add_argument("--out_root", default="/thearray/git/babyllm/data/cache")
    p.add_argument("--batch_docs", type=int, default=1000)
    p.add_argument("--train_tok", type=int, default=1_000_000_000)
    p.add_argument("--val_tok", type=int, default=20_000_000)
    p.add_argument("--test_tok", type=int, default=20_000_000)
    p.add_argument("--max-docs", dest="max_docs", type=int, default=0,
                   help="smoke test: stop after N docs per source (0 = full run)")
    args = p.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
    try:
        tok.model_max_length = int(1e12)
    except Exception:
        pass
    eos_id = tok.eos_token_id
    if eos_id is None:
        raise SystemExit("tokenizer has no eos_token_id; specify a separator id")
    print(f"tokenizer={args.tokenizer_path} len={len(tok)} eos_id={eos_id} "
          f"({tok.convert_ids_to_tokens([eos_id])[0]!r})", flush=True)

    out_root = Path(args.out_root)
    sources = [
        ("HuggingFaceFW/dclm_100BT-shuffled", "text", f"{args.label}_dclm"),
        ("karpathy/fineweb-edu-100b-shuffle", "text", f"{args.label}_fwedu"),
    ]
    grand = time.time()
    summary = {}
    for repo, field, prefix in sources:
        print(f"=== {repo} -> {prefix}_{{train,val,test}} ===", flush=True)
        summary[prefix] = build_source(
            repo, field, out_root, tok, args.tokenizer_path, eos_id,
            args.train_tok, args.val_tok, args.test_tok,
            args.batch_docs, prefix, args.max_docs,
        )
    print(f"=== ALL DONE in {time.time()-grand:.0f}s ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
