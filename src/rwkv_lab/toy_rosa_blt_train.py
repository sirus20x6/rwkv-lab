"""Self-contained training and evaluation demonstration for the ROSA + BLT language model.

This script trains a small ROSA + BLT language model prototype (utilizing RWKV8TimeMixDeltaNet,
RosaLayer suffix-retrieval, and RWKV7BLTChannelMix dynamic-patching) on a small multilingual
Tiny Stories dataset represented as character/byte-level UTF-8 streams (vocab size 256).
"""

from __future__ import annotations

import random
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rwkv_lab.rosa_blt_prototype import BLT_ROSA_LanguageModel
from rwkv_lab.toy_rwkv8_train import generate_multilingual_dataset


def main():
    print("=====================================================================")
    print("        ROSA + BLT Language Model Prototype Multilingual Trainer     ")
    print("=====================================================================")

    # Ensure CPU reference mode is used if FLA is missing
    import os
    os.environ.setdefault("RWKV8_FORCE_PYREF", "1")

    # Hyperparameters
    d_model = 32
    n_layers = 2
    head_size = 16
    threshold = 3.2
    max_patch = 8
    rosa_M = 4
    lr = 3e-3
    epochs = 40

    # 1. Create dataset
    train_data, val_data = generate_multilingual_dataset()
    print(f"Dataset generated:")
    print(f"  Training batch shape:   {train_data.shape}")
    print(f"  Validation batch shape: {val_data.shape}")

    # 2. Build ROSA + BLT model
    model = BLT_ROSA_LanguageModel(
        vocab_size=256,
        d_model=d_model,
        n_layers=n_layers,
        head_size=head_size,
        threshold=threshold,
        max_patch=max_patch,
        rosa_M=rosa_M,
    )

    # We must activate the ROSA readout path by making e1 != e0 (otherwise it's an exact no-op at step 0)
    with torch.no_grad():
        for block in model.blocks:
            block.rosa.e1.fill_(0.1)

    print(f"Model constructed with {n_layers} layers, d_model={d_model}, threshold={threshold}, max_patch={max_patch}, ROSA M={rosa_M}.")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Forward pass: next-token prediction and entropy head training
        logits, entropy_losses = model(train_data, target_bytes=train_data) # [B, T, 256]

        # 1. Standard next-byte prediction loss
        ce_loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, 256),
            train_data[:, 1:].reshape(-1)
        )

        # 2. Entropy prediction head losses (to train the entropy predictors)
        entropy_loss = sum(entropy_losses)

        total_loss = ce_loss + 0.1 * entropy_loss
        total_loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Monitor loss and log stats
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_logits = model(val_data)
                val_loss = F.cross_entropy(
                    val_logits[:, :-1].reshape(-1, 256),
                    val_data[:, 1:].reshape(-1)
                ).item()

                # Calculate average patch length across layers using stored context-aware last_patch_ids
                avg_lens = []
                for b_idx, block in enumerate(model.blocks):
                    patch_ids = block.ffn.last_patch_ids
                    num_patches = int(patch_ids[0].max()) + 1
                    avg_len = val_data.shape[1] / num_patches
                    avg_lens.append(avg_len)

                lens_str = ", ".join([f"L{i}: {l:.2f}" for i, l in enumerate(avg_lens)])
                print(f"Epoch {epoch:02d} | Train CE: {ce_loss.item():.4f} | Val CE: {val_loss:.4f} | Avg Patch Lens: [{lens_str}]")

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
