#!/usr/bin/env python3
"""Reproduce one token-budgeted vision epoch and publish its launch receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

from rwkv_lab.generate import WorldVocab
from rwkv_lab.vision_train import (
    EpochBatchSampler,
    _image_file_identity,
    load_examples,
    prepare_examples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default="Describe this image:\n")
    parser.add_argument("--max-text-tokens", type=int, default=768)
    parser.add_argument("--prefix-tokens", type=int, default=64)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--min-batch", type=int, default=4)
    parser.add_argument("--max-batch", type=int, default=32)
    parser.add_argument("--target-batch-tokens", type=int, default=3584)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--stat-workers", type=int, default=64)
    return parser.parse_args()


def percentile(values: list[int], fraction: float) -> int:
    return sorted(values)[min(len(values) - 1, int(len(values) * fraction))]


def main() -> None:
    args = parse_args()
    started = time.time()
    raw = load_examples(args.manifest, stat_workers=args.stat_workers)
    unique = {}
    for row in raw:
        identity = (
            repr(_image_file_identity(row)), row["text"],
            str(row.get("prompt") or args.prompt),
        )
        unique[identity] = row
    rows, lengths = prepare_examples(
        list(unique.values()), WorldVocab(), prompt=args.prompt,
        max_text_tokens=args.max_text_tokens)
    if not rows:
        raise SystemExit("manifest has no usable rows")
    indices = list(range(len(rows)))
    sampler = EpochBatchSampler(
        indices, lengths, batch_size=args.batch, seed=args.seed)
    token_costs = [args.prefix_tokens + length for length in lengths]
    steps = 0
    actual_tokens = 0
    padded_tokens = 0
    examples_per_step = []
    while sampler.position < len(sampler.order):
        batch = sampler.peek_budget_batch(
            token_costs, target_tokens=args.target_batch_tokens,
            min_items=args.min_batch, max_items=args.max_batch)
        if not batch:
            raise RuntimeError("sampler produced an empty in-epoch batch")
        sampler.commit_batch(batch)
        steps += 1
        examples_per_step.append(len(batch))
        actual_tokens += sum(token_costs[index] for index in batch)
        padded_tokens += len(batch) * max(token_costs[index] for index in batch)
    receipt = {
        "schema": 1,
        "manifest": str(args.manifest),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "rows": len(rows),
        "max_text_tokens": args.max_text_tokens,
        "prefix_tokens": args.prefix_tokens,
        "target_batch_tokens": args.target_batch_tokens,
        "min_batch": args.min_batch,
        "max_batch": args.max_batch,
        "sampler_seed": args.seed,
        "steps_per_epoch": steps,
        "min_examples_per_step": min(examples_per_step),
        "max_examples_per_step": max(examples_per_step),
        "mean_examples_per_step": statistics.fmean(examples_per_step),
        "actual_tokens": actual_tokens,
        "padded_tokens": padded_tokens,
        "padding_efficiency": actual_tokens / padded_tokens,
        "mean_text_tokens": statistics.fmean(lengths),
        "p50_text_tokens": percentile(lengths, 0.50),
        "p90_text_tokens": percentile(lengths, 0.90),
        "p99_text_tokens": percentile(lengths, 0.99),
        "truncated_captions": sum(bool(row["truncated"]) for row in rows),
        "tokenization_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(args.output)
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
