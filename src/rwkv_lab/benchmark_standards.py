"""Standardized Comparative Benchmark for RWKV-8, ROSA + BLT, and ROSA Latent Thinking.

This script compares the newly designed architectures under the exact same standards on:
1. Multilingual Tiny Stories (byte-level language modeling next-byte cross-entropy).
2. Logic NIAH (multi-hop logical Needle-in-a-Haystack reasoning accuracy under exact routing standards).

Both training and evaluation are conducted strictly on CPU to ensure deterministic behavior.
"""

from __future__ import annotations

import os
os.environ.setdefault("RWKV8_FORCE_PYREF", "1")

import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rwkv_lab.rwkv8_prototype import RWKV8LanguageModel
from rwkv_lab.rosa_blt_prototype import BLT_ROSA_LanguageModel
from rwkv_lab.rosa_thinking import RosaLatentThinkingModel
from rwkv_lab.logic_niaah import LogicNiahTask, LogicNiahRosaSolver
from rwkv_lab.toy_rwkv8_train import generate_multilingual_dataset


def run_benchmark(
    *,
    epochs: int = 15,
    d_model: int = 32,
    n_layers: int = 2,
    head_size: int = 16,
    train_rows: int | None = None,
    sequence_length: int | None = None,
    report_path: str | Path = "runs/benchmark_report.json",
):
    """Run the shared prototype benchmark at a caller-selected workload scale."""
    if epochs < 1 or d_model < head_size or d_model % head_size:
        raise ValueError("epochs must be positive and d_model must be divisible by head_size")
    if n_layers < 2 or head_size < 1:
        raise ValueError("n_layers must be at least 2 and head_size must be positive")
    print("=====================================================================")
    print("      Unified Benchmark: RWKV-8 vs ROSA + BLT vs ROSA Thinking       ")
    print("=====================================================================")

    # Standard model dimensions to compare against the exact same standards
    vocab_size_stories = 256
    d_model_stories = d_model
    n_layers_stories = n_layers
    head_size_stories = head_size
    lr = 3e-3

    device = "cpu"
    torch.manual_seed(42)

    # 1. Datasets Preparation
    # Multilingual Stories (byte-level next-token prediction)
    train_stories, val_stories = generate_multilingual_dataset()
    if train_rows is not None:
        if train_rows < 1:
            raise ValueError("train_rows must be positive")
        train_stories = train_stories[:train_rows]
    if sequence_length is not None:
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        train_stories = train_stories[:, :sequence_length]
        val_stories = val_stories[:, :sequence_length]

    # Logic NIAH Task (logical reasoning accuracy)
    # Standard vocab size 16 and d_model 8 to prevent unused route retrieval noise
    niah_task = LogicNiahTask(haystack_length=12, vocab_size=16)
    x_niah, y_niah, m_niah = niah_task.generate_batch(8, device)

    # Dictionary to collect all comparative results
    results_stories = {
        "Standard RWKV-8": {"train_loss": 0.0, "val_loss": 0.0},
        "ROSA + BLT": {"train_loss": 0.0, "val_loss": 0.0},
        "ROSA Latent Thinker": {"train_loss": 0.0, "val_loss": 0.0},
    }

    # =========================================================================
    # Task 1: Multilingual Tiny Stories Language Modeling
    # =========================================================================

    # 1.1 Standard RWKV-8
    print("\n--- Training Standard RWKV-8 Model ---")
    model_rwkv8 = RWKV8LanguageModel(
        vocab_size=vocab_size_stories, d_model=d_model_stories, n_layers=n_layers_stories, head_size=head_size_stories, deepembed=True
    ).to(device)
    opt = optim.AdamW(model_rwkv8.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        model_rwkv8.train()
        opt.zero_grad()
        logits = model_rwkv8(train_stories)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab_size_stories), train_stories[:, 1:].reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model_rwkv8.parameters(), 1.0)
        opt.step()

    model_rwkv8.eval()
    with torch.no_grad():
        val_logits = model_rwkv8(val_stories)
        val_loss = F.cross_entropy(val_logits[:, :-1].reshape(-1, vocab_size_stories), val_stories[:, 1:].reshape(-1)).item()
    results_stories["Standard RWKV-8"]["train_loss"] = loss.item()
    results_stories["Standard RWKV-8"]["val_loss"] = val_loss

    # 1.2 ROSA + BLT
    print("\n--- Training ROSA + BLT Hybrid Model ---")
    model_rosa_blt = BLT_ROSA_LanguageModel(
        vocab_size=vocab_size_stories, d_model=d_model_stories, n_layers=n_layers_stories, head_size=head_size_stories, threshold=3.2, max_patch=8, rosa_M=4
    ).to(device)
    # Activate ROSA readout path
    with torch.no_grad():
        for block in model_rosa_blt.blocks:
            block.rosa.e1.fill_(0.1)
    opt = optim.AdamW(model_rosa_blt.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        model_rosa_blt.train()
        opt.zero_grad()
        logits, entropy_losses = model_rosa_blt(train_stories, target_bytes=train_stories)
        ce_loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab_size_stories), train_stories[:, 1:].reshape(-1))
        loss = ce_loss + 0.1 * sum(entropy_losses)
        loss.backward()
        nn.utils.clip_grad_norm_(model_rosa_blt.parameters(), 1.0)
        opt.step()

    model_rosa_blt.eval()
    with torch.no_grad():
        val_logits = model_rosa_blt(val_stories)
        val_loss = F.cross_entropy(val_logits[:, :-1].reshape(-1, vocab_size_stories), val_stories[:, 1:].reshape(-1)).item()
    results_stories["ROSA + BLT"]["train_loss"] = ce_loss.item()
    results_stories["ROSA + BLT"]["val_loss"] = val_loss

    # 1.3 ROSA Latent Thinker
    print("\n--- Training ROSA Latent Thinking Model ---")
    model_thinking_stories = RosaLatentThinkingModel(
        vocab_size=vocab_size_stories, d_model=d_model_stories, rosa_M=4
    ).to(device)
    # Activate ROSA exact readout path
    with torch.no_grad():
        model_thinking_stories.block.rosa.e0.fill_(0.0)
        model_thinking_stories.block.rosa.e1.fill_(1.0)
    opt = optim.AdamW(model_thinking_stories.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        model_thinking_stories.train()
        opt.zero_grad()
        logits = model_thinking_stories(train_stories, n_thoughts=0)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab_size_stories), train_stories[:, 1:].reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model_thinking_stories.parameters(), 1.0)
        opt.step()

    model_thinking_stories.eval()
    with torch.no_grad():
        val_logits = model_thinking_stories(val_stories, n_thoughts=0)
        val_loss = F.cross_entropy(val_logits[:, :-1].reshape(-1, vocab_size_stories), val_stories[:, 1:].reshape(-1)).item()
    results_stories["ROSA Latent Thinker"]["train_loss"] = loss.item()
    results_stories["ROSA Latent Thinker"]["val_loss"] = val_loss

    # =========================================================================
    # Task 2: Logic NIAH Logical Reasoning (Depth vs Thinking Steps)
    # =========================================================================
    print("\n--- Evaluating Logic NIAH Logical Reasoning ---")

    # 2.1 1-layer ROSA with 0 thinking steps (Immediate Speaking)
    model_1l_no_think = RosaLatentThinkingModel(vocab_size=16, d_model=8, rosa_M=4).to(device)
    with torch.no_grad():
        logits_niah_no_think = model_1l_no_think(x_niah, n_thoughts=0)
        acc_niah_no_think = niah_task.accuracy(logits_niah_no_think, y_niah, m_niah)

    # 2.2 1-layer ROSA with 1 thinking step (Latent Thinking)
    # Generates a sequence of continuous thoughts in the latent state before speaking
    with torch.no_grad():
        logits_niah_think = model_1l_no_think(x_niah, n_thoughts=1)
        m_thinking = torch.zeros_like(logits_niah_think[..., 0], dtype=torch.float32)
        m_thinking[:, -1] = 1.0
        acc_niah_think = niah_task.accuracy(logits_niah_think, y_niah[:, -1:], m_thinking)

    # 2.3 2-layer ROSA Solver (Standard Physical depth)
    model_2l_solver = LogicNiahRosaSolver(vocab_size=16, d_model=8, rosa_M=4).to(device)
    with torch.no_grad():
        logits_niah_2l = model_2l_solver(x_niah)
        acc_niah_2l = niah_task.accuracy(logits_niah_2l, y_niah, m_niah)

    # =========================================================================
    # Final Standardized Comparison Display
    # =========================================================================
    print("\n" + "="*80)
    print("                 FINAL STANDARDIZED COMPARISON RESULTS                     ")
    print("="*80)
    print(" 1. Multilingual Tiny Stories byte-level Language Modeling:")
    print("--------------------------------------------------------------------------------")
    print(f"| {'Architecture':<25} | {'Stories Train CE':<23} | {'Stories Val CE':<21} |")
    print(f"|{'-'*27}|{'-'*25}|{'-'*23}|")
    for arch, metrics in results_stories.items():
        print(f"| {arch:<25} | {metrics['train_loss']:23.4f} | {metrics['val_loss']:21.4f} |")

    print("\n 2. Logic Needle-in-a-Haystack (Logical Reasoning Depth vs Thinking):")
    print("--------------------------------------------------------------------------------")
    print(f"| {'Model / Solver':<45} | {'Logic NIAH Accuracy':<25} |")
    print(f"|{'-'*47}|{'-'*27}|")
    print(f"| {'1-Layer ROSA (n_thoughts = 0) - No Thinking':<45} | {acc_niah_no_think*100:23.1f}% |")
    print(f"| {'1-Layer ROSA (n_thoughts = 1) - Latent Thinker':<45} | {acc_niah_think*100:23.1f}% |")
    print(f"| {'2-Layer ROSA Physical Solver (Standard Depth)':<45} | {acc_niah_2l*100:23.1f}% |")
    print("="*80)

    # Also save to a JSON report for archival / automated verification
    results = {
        "multilingual_stories": results_stories,
        "logic_niah": {
            "1l_no_think_acc": acc_niah_no_think,
            "1l_think_acc": acc_niah_think,
            "2l_solver_acc": acc_niah_2l
        }
    }
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w") as handle:
        json.dump(results, handle, indent=2)
    print(f"Benchmark saved to {report_path}\n")
    return results


if __name__ == "__main__":
    run_benchmark()
