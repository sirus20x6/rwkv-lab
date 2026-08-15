"""Self-contained toy training and evaluation demonstration for KAN-RWKV language model.

This script trains a small KAN-RWKV language model (utilizing KANLinear-based FFN layers
and DeepEmbed gating) on a small multilingual Tiny Stories dataset represented as character
bytes (using vocab size 256). It showcases training convergence, stable gradient updates,
and autoregressive text generation.
"""

from __future__ import annotations

import random
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rwkv_lab.kan_rwkv import KANRWKVLanguageModel
from rwkv_lab.toy_rwkv8_train import generate_multilingual_dataset, sample_logits


def main():
    print("=====================================================================")
    print("       KAN-RWKV Language Model Prototype Multilingual Trainer        ")
    print("=====================================================================")

    # Ensure CPU reference mode is used if FLA is missing
    import os
    os.environ.setdefault("RWKV8_FORCE_PYREF", "1")

    # Hyperparameters
    d_model = 32
    n_layers = 2
    head_size = 16
    lr = 3e-3
    epochs = 60
    grid_size = 5
    spline_order = 3

    # 1. Create dataset
    train_data, val_data = generate_multilingual_dataset()
    print(f"Dataset generated:")
    print(f"  Training batch shape:   {train_data.shape}")
    print(f"  Validation batch shape: {val_data.shape}")

    # 2. Build KAN-RWKV language model with DeepEmbed
    model = KANRWKVLanguageModel(
        vocab_size=256,
        d_model=d_model,
        n_layers=n_layers,
        head_size=head_size,
        deepembed=True,
        grid_size=grid_size,
        spline_order=spline_order
    )
    print(f"Model constructed with {n_layers} layers, d_model={d_model}, DeepEmbed=True.")
    print(f"FFN layers use Kolmogorov-Arnold Networks (KAN) with grid_size={grid_size}, spline_order={spline_order}.")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Forward pass: next-token prediction
        logits = model(train_data)  # [B, T, 256]

        # Standard next-token prediction loss
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, 256),
            train_data[:, 1:].reshape(-1)
        )

        loss.backward()
        # Gradient clipping to ensure stability
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
                print(f"Epoch {epoch:02d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")

    print("\nKAN-RWKV training completed successfully!")

    # 3. Multilingual Autoregressive Generation Demonstration
    print("\n--- Autoregressive Generation Demo ---")
    model.eval()
    prompt = "El gato se sento "
    generated = list(prompt.encode("utf-8"))

    # Initialize recurrent states (one per block/layer)
    # Each block state is: ((wkv_state, shift_state_att), shift_state_ffn)
    states = None

    print(f"Prompt: '{prompt}'")

    # Feed prompt characters one by one to prime the states
    with torch.no_grad():
        for byte in generated[:-1]:
            input_tensor = torch.tensor([[byte]], dtype=torch.long)
            _, states = model(input_tensor, states=states, return_state=True)

        # Generate 40 new bytes autoregressively
        last_byte = generated[-1]
        for _ in range(40):
            input_tensor = torch.tensor([[last_byte]], dtype=torch.long)
            logits, states = model(input_tensor, states=states, return_state=True)

            # Sample next token
            next_byte = sample_logits(logits[0, 0], temperature=1.0, top_p=0.8)
            generated.append(next_byte)
            last_byte = next_byte

            # Decode and print character as we go
            try:
                char = bytes([next_byte]).decode("utf-8")
                print(char, end="", flush=True)
            except UnicodeDecodeError:
                print(".", end="", flush=True)

    print("\n\nFull Generated Text (decoded):")
    final_text = bytes(generated).decode("utf-8", errors="replace")
    print(f"'{final_text}'")
    print("--------------------------------------")


if __name__ == "__main__":
    main()
