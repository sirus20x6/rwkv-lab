"""Comprehensive validation, comparative benchmarking, and proof of dynamic patch-adaptation for BLT-RWKV-7.

This script compares a Standard byte-level RWKV-7 model against our BLT-RWKV-7 model
under the exact same standards on a multilingual Tiny Stories dataset.
It records training convergence, validation perplexity, throughput/latency, parameter counts,
and verifies next-byte entropy decrease with corresponding average patch length increase.
"""

from __future__ import annotations

import os
os.environ.setdefault("RWKV8_FORCE_PYREF", "1")

import json
import time
import argparse
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rwkv_lab.rwkv8_deltanet import RWKV8TimeMixDeltaNet, RWKV8ChannelMixDeltaNet
from rwkv_lab.toy_blt_train import BLTRWKV7LanguageModel, generate_multilingual_dataset


def parse_zip_chars(chars_str: str) -> set[int]:
    """Parse comma-separated characters to a set of byte integers."""
    res = set()
    for segment in chars_str.split(","):
        if not segment:
            continue
        for char in segment:
            res.add(ord(char))
    return res


def zip_compress_bytes(byte_list: list[int], zip_chars: set[int], min_run: int = 3, zip_token: int = 256) -> list[int]:
    """Compress repeating runs of specified characters in a byte list."""
    compressed = []
    i = 0
    n = len(byte_list)
    while i < n:
        val = byte_list[i]
        if val in zip_chars:
            run_len = 1
            while i + run_len < n and byte_list[i + run_len] == val:
                run_len += 1
            if run_len >= min_run:
                remaining = run_len
                while remaining > 0:
                    current_run = min(remaining, 255)
                    compressed.extend([zip_token, val, current_run])
                    remaining -= current_run
                i += run_len
                continue
        compressed.append(val)
        i += 1
    return compressed


def zip_decompress_bytes(token_list: list[int], zip_token: int = 256) -> list[int]:
    """Decompress a list of tokens back to raw bytes."""
    decompressed = []
    i = 0
    n = len(token_list)
    while i < n:
        if token_list[i] == zip_token:
            if i + 2 < n:
                val = token_list[i + 1]
                count = token_list[i + 2]
                decompressed.extend([val] * count)
                i += 3
                continue
        decompressed.append(token_list[i])
        i += 1
    return decompressed


def compress_tensor_batch(data_tensor: torch.Tensor, zip_chars: set[int], min_run: int = 3, zip_token: int = 256) -> torch.Tensor:
    """Compress a batch of byte rows into a padded tensor."""
    compressed_rows = []
    max_len = 0
    for row in data_tensor:
        comp = zip_compress_bytes(row.tolist(), zip_chars, min_run, zip_token)
        compressed_rows.append(comp)
        if len(comp) > max_len:
            max_len = len(comp)
    padded_rows = []
    for comp in compressed_rows:
        padded = comp + [0] * (max_len - len(comp))
        padded_rows.append(padded)
    return torch.tensor(padded_rows, dtype=torch.long)


class StandardRWKV7Block(nn.Module):
    """A standard RWKV-7 block running TimeMix and ChannelMix at the raw byte/character level."""

    def __init__(self, d_model: int, i: int, num_layers: int):
        super().__init__()
        self.i = i
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.att = RWKV8TimeMixDeltaNet(
            hidden_size=d_model,
            num_heads=d_model // 16,
            head_size=16,
            layer_idx=i,
            depth_layer_id=i,
            depth_n_layer=num_layers,
            is_first_rwkv_layer=(i == 0),
            out_correct=False,
        )

        self.ffn = RWKV8ChannelMixDeltaNet(
            hidden_size=d_model,
            ffn_hidden_size=d_model * 2,
            layer_idx=i,
        )

    def forward(self, x: torch.Tensor, v_first: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        ln_x = self.ln1(x)
        att_out, v_first_out = self.att(ln_x, v_first=v_first, return_v_first=True)
        x = x + att_out

        ln_x2 = self.ln2(x)
        ffn_out = self.ffn(ln_x2)
        x = x + ffn_out
        return x, v_first_out


class StandardRWKV7LanguageModel(nn.Module):
    """Standard byte-level RWKV-7 language model."""

    def __init__(self, vocab_size: int = 256, d_model: int = 32, n_layers: int = 2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            StandardRWKV7Block(d_model, i, n_layers) for i in range(n_layers)
        ])
        self.ln_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.emb(ids)
        v_first = None
        for block in self.blocks:
            x, v_first = block(x, v_first=v_first)
        x = self.ln_out(x)
        return self.head(x)


def run_comparative_validation(
    epochs: int = 50,
    lr: float = 3e-3,
    threshold: float = 3.5,
    max_patch: int = 16,
    zip_compress: bool = False,
    zip_chars_str: str = " ,#,=,-",
    zip_min_run: int = 3,
    *,
    d_model: int = 32,
    n_layers: int = 2,
    train_rows: int | None = None,
    sequence_length: int | None = None,
    benchmark_batch: int = 4,
    benchmark_tokens: int = 256,
    benchmark_warmup: int = 3,
    benchmark_iters: int = 15,
    report_path: str | Path = "runs/blt_comparison_report.json",
):
    """Train and compare byte-level and dynamically patched RWKV variants.

    The workload controls make the same pipeline usable for quick CI checks and
    full proof runs. They affect only scale, not the measured code paths.
    """
    if epochs < 1 or d_model < 16 or d_model % 16:
        raise ValueError("epochs must be positive and d_model must be a positive multiple of 16")
    if n_layers < 2 or benchmark_batch < 1 or benchmark_tokens < 2 or benchmark_iters < 1:
        raise ValueError("n_layers must be at least 2 and benchmark dimensions must be positive")
    if benchmark_warmup < 0:
        raise ValueError("benchmark_warmup cannot be negative")
    print("=====================================================================")
    print("      Comparative Validation: Standard RWKV-7 vs BLT-RWKV-7          ")
    print("=====================================================================")

    device = "cpu"
    torch.manual_seed(42)

    # 1. Prepare Datasets (multilingual UTF-8 stories)
    train_data, val_data = generate_multilingual_dataset()
    if train_rows is not None:
        if train_rows < 1:
            raise ValueError("train_rows must be positive")
        train_data = train_data[:train_rows]
    if sequence_length is not None:
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        train_data = train_data[:, :sequence_length]
        val_data = val_data[:, :sequence_length]
    vocab_size = 256
    if zip_compress:
        zip_chars = parse_zip_chars(zip_chars_str)
        train_data = compress_tensor_batch(train_data, zip_chars, zip_min_run, 256)
        val_data = compress_tensor_batch(val_data, zip_chars, zip_min_run, 256)
        vocab_size = 257
        print(f"Zip compression enabled on characters: {zip_chars}")

    print(f"Dataset Details:")
    print(f"  Training batch shape:   {train_data.shape}")
    print(f"  Validation batch shape: {val_data.shape}")

    # 2. Build Models
    std_model = StandardRWKV7LanguageModel(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers).to(device)
    blt_model = BLTRWKV7LanguageModel(
        vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, threshold=threshold, max_patch=max_patch
    ).to(device)

    # Count parameters
    std_params = sum(p.numel() for p in std_model.parameters() if p.requires_grad)
    blt_params = sum(p.numel() for p in blt_model.parameters() if p.requires_grad)

    print(f"\nModel Configuration:")
    print(f"  d_model: {d_model} | n_layers: {n_layers} | threshold: {threshold} | max_patch: {max_patch}")
    print(f"  Standard RWKV-7 parameters: {std_params:,}")
    print(f"  BLT-RWKV-7 parameters:      {blt_params:,} (includes entropy heads)")

    # 3. Train Standard RWKV-7
    print("\n--- Training Standard RWKV-7 Model ---")
    std_optimizer = optim.AdamW(std_model.parameters(), lr=lr, weight_decay=1e-4)
    std_train_losses = []

    for epoch in range(1, epochs + 1):
        std_model.train()
        std_optimizer.zero_grad()
        logits = std_model(train_data)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab_size), train_data[:, 1:].reshape(-1))
        loss.backward()
        std_optimizer.step()
        std_train_losses.append(loss.item())

    std_model.eval()
    with torch.no_grad():
        std_val_logits = std_model(val_data)
        std_val_loss = F.cross_entropy(std_val_logits[:, :-1].reshape(-1, vocab_size), val_data[:, 1:].reshape(-1)).item()
    print(f"Standard Model training complete. Final Train Loss: {std_train_losses[-1]:.4f} | Val Loss: {std_val_loss:.4f}")

    # 4. Train BLT-RWKV-7
    print("\n--- Training BLT-RWKV-7 Model ---")
    blt_optimizer = optim.AdamW(blt_model.parameters(), lr=lr, weight_decay=1e-4)
    blt_train_losses = []
    blt_avg_patch_lengths = []
    blt_entropies = []

    for epoch in range(1, epochs + 1):
        blt_model.train()
        blt_optimizer.zero_grad()
        logits, entropy_losses = blt_model(train_data, target_bytes=train_data)
        ce_loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab_size), train_data[:, 1:].reshape(-1))
        loss = ce_loss + 0.1 * sum(entropy_losses)
        loss.backward()
        blt_optimizer.step()
        blt_train_losses.append(ce_loss.item())

        # Collect metrics at intermediate checkpoints
        if epoch % 5 == 0 or epoch == 1:
            blt_model.eval()
            with torch.no_grad():
                _ = blt_model(val_data)
                avg_lens = []
                entropies = []
                for b_idx, block in enumerate(blt_model.blocks):
                    patch_ids = block.ffn.last_patch_ids
                    num_patches = int(patch_ids[0].max()) + 1
                    avg_len = val_data.shape[1] / num_patches
                    avg_lens.append(avg_len)
                    entropies.append(block.ffn.last_entropy.mean().item())
                blt_avg_patch_lengths.append(avg_lens)
                blt_entropies.append(entropies)
                lens_str = ", ".join([f"L{i}: {l:.2f}" for i, l in enumerate(avg_lens)])
                print(f"Epoch {epoch:02d} | Train CE: {ce_loss.item():.4f} | Avg Patch Lens: [{lens_str}]")

    blt_model.eval()
    with torch.no_grad():
        blt_val_logits = blt_model(val_data)
        blt_val_loss = F.cross_entropy(blt_val_logits[:, :-1].reshape(-1, vocab_size), val_data[:, 1:].reshape(-1)).item()
    print(f"BLT Model training complete. Final Train Loss: {blt_train_losses[-1]:.4f} | Val Loss: {blt_val_loss:.4f}")

    # 5. Throughput/Latency Benchmark on CPU
    print("\n--- Latency and Throughput Benchmarking ---")
    # Simulate a longer context sequence to highlight theoretical FFN performance scaling
    B, T = benchmark_batch, benchmark_tokens
    benchmark_input = torch.randint(0, vocab_size, (B, T)).to(device)

    # Warmup
    for _ in range(benchmark_warmup):
        with torch.no_grad():
            _ = std_model(benchmark_input)
            _ = blt_model(benchmark_input)

    # Benchmark Standard RWKV-7
    t0 = time.perf_counter()
    iters = benchmark_iters
    for _ in range(iters):
        with torch.no_grad():
            _ = std_model(benchmark_input)
    t_std = (time.perf_counter() - t0) / iters

    # Benchmark BLT-RWKV-7
    t1 = time.perf_counter()
    for _ in range(iters):
        with torch.no_grad():
            _ = blt_model(benchmark_input)
    t_blt = (time.perf_counter() - t1) / iters

    std_throughput = (B * T) / t_std
    blt_throughput = (B * T) / t_blt
    ffn_speedup = t_std / t_blt

    print(f"Benchmark Sequence Shape: [{B}, {T}]")
    print(f"Standard RWKV-7 Avg Latency: {t_std*1000:.2f} ms ({std_throughput:.1f} bytes/sec)")
    print(f"BLT-RWKV-7 Avg Latency:      {t_blt*1000:.2f} ms ({blt_throughput:.1f} bytes/sec)")
    print(f"FFN Optimization Speedup:    {ffn_speedup:.2f}x")

    # 6. Verify Dynamic Patch Adaptation (Decreasing next-byte entropy corresponds to increasing patch lengths)
    initial_entropy_mean = blt_entropies[0][0]
    final_entropy_mean = blt_entropies[-1][0]
    initial_patch_len = blt_avg_patch_lengths[0][0]
    final_patch_len = blt_avg_patch_lengths[-1][0]

    print("\n--- Dynamic Patch Adaptation Analysis ---")
    print(f"  Initial next-byte entropy: {initial_entropy_mean:.3f} | Final next-byte entropy: {final_entropy_mean:.3f}")
    print(f"  Initial average patch len: {initial_patch_len:.2f} | Final average patch len: {final_patch_len:.2f}")

    entropy_reduced = final_entropy_mean < initial_entropy_mean
    patch_len_increased = final_patch_len > initial_patch_len
    adaptation_successful = entropy_reduced and patch_len_increased

    print(f"  Next-byte prediction entropy reduced?   {'YES' if entropy_reduced else 'NO'}")
    print(f"  Average patch length scaled upward?     {'YES' if patch_len_increased else 'NO'}")
    print(f"  Context-aware dynamic adaptation proof: {'PASSED' if adaptation_successful else 'FAILED'}")

    # Compile report dictionary
    report = {
        "parameters": {
            "standard_rwkv7": std_params,
            "blt_rwkv7": blt_params,
        },
        "training": {
            "standard_final_train_loss": std_train_losses[-1],
            "standard_val_loss": std_val_loss,
            "blt_final_train_loss": blt_train_losses[-1],
            "blt_val_loss": blt_val_loss,
        },
        "benchmark": {
            "sequence_shape": [B, T],
            "standard_latency_ms": t_std * 1000,
            "blt_latency_ms": t_blt * 1000,
            "standard_throughput_bps": std_throughput,
            "blt_throughput_bps": blt_throughput,
            "speedup": ffn_speedup,
        },
        "dynamic_adaptation": {
            "initial_entropy": initial_entropy_mean,
            "final_entropy": final_entropy_mean,
            "initial_patch_length": initial_patch_len,
            "final_patch_length": final_patch_len,
            "successful": adaptation_successful,
        }
    }

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written successfully to {report_path}\n")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run standardized comparative validation of BLT vs Standard RWKV.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate")
    parser.add_argument("--threshold", type=float, default=3.5, help="Entropy threshold for patch boundary splits")
    parser.add_argument("--max_patch", type=int, default=16, help="Maximum allowed patch size in bytes")
    parser.add_argument("--zip_compress", action="store_true", help="Enable zip compression for repeating bytes")
    parser.add_argument("--zip_chars", type=str, default=" ,#,=,-", help="Comma-separated characters to zip compress")
    parser.add_argument("--zip_min_run", type=int, default=3, help="Minimum repeating run length to zip compress")
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--train-rows", type=int)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--benchmark-batch", type=int, default=4)
    parser.add_argument("--benchmark-tokens", type=int, default=256)
    parser.add_argument("--benchmark-warmup", type=int, default=3)
    parser.add_argument("--benchmark-iters", type=int, default=15)
    parser.add_argument("--report-path", default="runs/blt_comparison_report.json")
    args = parser.parse_args()

    run_comparative_validation(
        epochs=args.epochs,
        lr=args.lr,
        threshold=args.threshold,
        max_patch=args.max_patch,
        zip_compress=args.zip_compress,
        zip_chars_str=args.zip_chars,
        zip_min_run=args.zip_min_run,
        d_model=args.d_model,
        n_layers=args.n_layers,
        train_rows=args.train_rows,
        sequence_length=args.sequence_length,
        benchmark_batch=args.benchmark_batch,
        benchmark_tokens=args.benchmark_tokens,
        benchmark_warmup=args.benchmark_warmup,
        benchmark_iters=args.benchmark_iters,
        report_path=args.report_path,
    )
