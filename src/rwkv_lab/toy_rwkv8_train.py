"""Self-contained toy training and evaluation demonstration for the RWKV-8 language model.

This script trains a small RWKV-8 language model prototype (utilizing RWKV8TimeMixDeltaNet,
RWKV8ChannelMixDeltaNet, and DeepEmbed gating) on a small multilingual Tiny Stories dataset
represented as character bytes (using vocab size 256). It showcases training convergence,
stable gradient updates, and autoregressive generation.
"""

from __future__ import annotations

import random
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rwkv_lab.rwkv8_prototype import RWKV8LanguageModel


# Multilingual Tiny Stories Generator (English, French, Spanish)
def generate_multilingual_dataset() -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a toy multilingual dataset of simple stories in English, French, and Spanish."""
    english_stories = [
        "The cat sat on the warm mat.",
        "A small girl found a beautiful shiny key.",
        "The big dog barked happily at the bird.",
        "She drank sweet juice from a red cup.",
    ]
    french_stories = [
        "Le chat etait assis sur le tapis chaud.",
        "Une petite fille a trouve une jolie cle brillante.",
        "Le grand chien aboyait joyeusement apres l'oiseau.",
        "Elle a bu du jus sucre dans une tasse rouge.",
    ]
    spanish_stories = [
        "El gato se sento en la alfombra calida.",
        "Una niña pequeña encontro una hermosa llave brillante.",
        "El perro grande ladro alegremente al pajaro.",
        "Ella bebio un jugo dulce de una taza roja.",
    ]

    all_texts = []
    # Mix them to create a multilingual stream
    for en, fr, es in zip(english_stories, french_stories, spanish_stories):
        all_texts.extend([en, fr, es])

    # Convert all texts to UTF-8 bytes
    train_bytes = []
    for text in all_texts:
        # Wrap each story with special start (0) and end (1) markers
        story_bytes = [0] + list(text.encode("utf-8")) + [1]
        train_bytes.extend(story_bytes)

    # Pad/slice into sequences of length 64
    seq_len = 64
    chunks = [train_bytes[i:i + seq_len] for i in range(0, len(train_bytes) - seq_len, seq_len)]

    # Validation data (unseen stories)
    val_stories = [
        "The small bird sang a beautiful song.",
        "L'oiseau a chante une belle chanson.",
        "El pajaro canto una hermosa cancion.",
    ]
    val_bytes = []
    for text in val_stories:
        val_bytes.extend([0] + list(text.encode("utf-8")) + [1])
    while len(val_bytes) < seq_len:
        val_bytes.append(0)
    val_bytes = val_bytes[:seq_len]

    return torch.tensor(chunks, dtype=torch.long), torch.tensor([val_bytes], dtype=torch.long)


# Autoregressive sampling function
def sample_logits(logits, temperature: float = 1.0, top_p: float = 0.8):
    probs = F.softmax(logits.float(), dim=-1)
    sorted_probs, sorted_ids = torch.sort(probs, descending=True)

    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    cutoff_index = torch.searchsorted(cumulative_probs, top_p)
    if cutoff_index < len(sorted_probs):
        cutoff = sorted_probs[cutoff_index]
        probs[probs < cutoff] = 0.0

    if temperature != 1.0 and temperature > 0.0:
        probs = probs ** (1.0 / temperature)
    probs = probs / torch.sum(probs)
    return torch.multinomial(probs, num_samples=1).item()


def main():
    print("=====================================================================")
    print("        RWKV-8 Language Model Prototype Multilingual Trainer         ")
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

    # 1. Create dataset
    train_data, val_data = generate_multilingual_dataset()
    print(f"Dataset generated:")
    print(f"  Training batch shape:   {train_data.shape}")
    print(f"  Validation batch shape: {val_data.shape}")

    # 2. Build RWKV-8 language model with DeepEmbed
    model = RWKV8LanguageModel(
        vocab_size=256,
        d_model=d_model,
        n_layers=n_layers,
        head_size=head_size,
        deepembed=True
    )
    print(f"Model constructed with {n_layers} layers, d_model={d_model}, DeepEmbed=True.")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Forward pass: next-token prediction
        logits = model(train_data) # [B, T, 256]

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

    print("\nTraining completed successfully!")

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
