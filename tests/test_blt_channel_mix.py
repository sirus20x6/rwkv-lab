"""Rigorous correctness, numerical equivalence, gradient flow, and performance tests for RWKV7BLTChannelMix."""

from __future__ import annotations

import time
import torch
import torch.nn as nn
import pytest

from rwkv_lab.blt_channel_mix import RWKV7BLTChannelMix
from rwkv_lab.rwkv8_deltanet import RWKV8ChannelMixDeltaNet


def test_blt_parallel_vs_recurrent_equivalence():
    """Verify that recurrent streaming inference matches parallel training at patch boundaries."""
    torch.manual_seed(42)
    B, T, C = 2, 16, 8
    hidden_states = torch.randn(B, T, C)

    # Initialize BLT ChannelMix
    blt = RWKV7BLTChannelMix(
        hidden_size=C,
        ffn_hidden_size=C * 2,
        threshold=2.5,
        min_patch=1,
        max_patch=4,
        streaming_mode="causal"
    )

    # 1. Parallel forward pass (training mode)
    with torch.no_grad():
        out_parallel = blt(hidden_states)

        # Get the entropy values and determine where the boundaries are triggered
        logits = blt.entropy_head(hidden_states)
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)

        # Step through and find boundaries similarly to internal logic
        from rwkv_lab.byte_patches import entropy_patch_ids
        patch_ids = entropy_patch_ids(entropy, threshold=blt.threshold, min_patch=blt.min_patch, max_patch=blt.max_patch)

    # 2. Recurrent step-by-step forward pass (inference mode)
    shift_state = None
    out_recurrent_list = []
    splits = []

    with torch.no_grad():
        for t in range(T):
            # Recurrent slice of shape [B, 1, C]
            step_input = hidden_states[:, t:t+1]
            step_out, shift_state = blt(step_input, shift_state=shift_state, return_state=True)
            out_recurrent_list.append(step_out)

            # Reconstruct boundary splits for validation
            running_sum, length, prev_patch_state, last_emitted = shift_state
            # A split is triggered if length is reset to 0 in shift_state (meaning it was reset on boundaries)
            splits.append(length == 0)

    out_recurrent = torch.cat(out_recurrent_list, dim=1) # [B, T, C]

    # Validate that at boundary steps, the outputs match exactly
    for b in range(B):
        for t in range(T):
            # If a boundary was triggered at step t
            is_boundary = splits[t][b].item()
            if is_boundary:
                # The recurrent streaming output must match the parallel training output exactly
                diff = torch.abs(out_parallel[b, t] - out_recurrent[b, t]).max().item()
                assert diff < 1e-5, f"Boundary mismatch at batch {b}, step {t}: diff={diff}"


def test_blt_gradient_flow_and_entropy_training():
    """Verify that gradients flow perfectly through all parameters, including the entropy head."""
    torch.manual_seed(42)
    B, T, C = 2, 8, 8
    hidden_states = torch.randn(B, T, C, requires_grad=True)
    target_bytes = torch.randint(0, 256, (B, T))

    blt = RWKV7BLTChannelMix(
        hidden_size=C,
        ffn_hidden_size=C * 2,
        threshold=2.5,
        min_patch=1,
        max_patch=4
    )

    # Verify that the forward pass is differentiable
    out = blt(hidden_states)
    loss_out = out.sum()
    loss_out.backward()

    # Key, value and x_k parameters must have non-zero gradients
    assert blt.key.weight.grad is not None and blt.key.weight.grad.norm() > 0
    assert blt.value.weight.grad is not None and blt.value.weight.grad.norm() > 0
    assert blt.x_k.grad is not None and blt.x_k.grad.norm() > 0

    # Test the compute_entropy_loss method (training the entropy head directly on next-byte targets)
    blt.zero_grad()
    entropy_loss = blt.compute_entropy_loss(hidden_states.detach(), target_bytes)
    assert entropy_loss.item() > 0
    entropy_loss.backward()

    # The entropy head weights must have non-zero gradients
    assert blt.entropy_head.weight.grad is not None and blt.entropy_head.weight.grad.norm() > 0


def test_blt_performance_benchmark():
    """Run an empirical benchmarking pass to compare BLT ChannelMix vs. Standard ChannelMix."""
    torch.manual_seed(42)
    # Larger dimensions to get reliable timings
    B, T, C = 4, 128, 64
    hidden_states = torch.randn(B, T, C)

    blt = RWKV7BLTChannelMix(
        hidden_size=C,
        ffn_hidden_size=C * 4,
        threshold=3.0,
        min_patch=1,
        max_patch=8
    )

    std = RWKV8ChannelMixDeltaNet(
        hidden_size=C,
        ffn_hidden_size=C * 4
    )

    # Warmup
    for _ in range(3):
        _ = blt(hidden_states)
        _ = std(hidden_states)

    # Measure Standard ChannelMix latency
    t0 = time.perf_counter()
    for _ in range(10):
        _ = std(hidden_states)
    t_std = (time.perf_counter() - t0) / 10

    # Measure BLT ChannelMix latency
    t1 = time.perf_counter()
    for _ in range(10):
        _ = blt(hidden_states)
    t_blt = (time.perf_counter() - t1) / 10

    # Print benchmarking report
    print(f"\n--- BLT ChannelMix vs Standard ChannelMix Benchmark ---")
    print(f"Sequence shape: [{B}, {T}, {C}]")
    print(f"Standard ChannelMix average forward latency: {t_std*1000:.3f} ms")
    print(f"BLT ChannelMix average forward latency:      {t_blt*1000:.3f} ms")
    print(f"------------------------------------------------------")

    # Since the sequence length is small, CPU overhead might dominate, but the forward passes should both run correctly
    assert t_blt > 0
    assert t_std > 0


def test_blt_entropy_learning_and_patch_adaptation():
    """Verify that training the entropy head on a simple context-rich sequence
    reduces predicted next-byte entropy and increases average patch lengths.
    """
    torch.manual_seed(42)
    B, T, C = 2, 32, 16

    # Create a highly predictable context-rich sequence (repeating pattern 'abcabc...')
    pattern = [97, 98, 99]  # 'a', 'b', 'c'
    seq = []
    for _ in range(T):
        seq.append(pattern[_ % len(pattern)])
    train_data = torch.tensor([seq, seq], dtype=torch.long)  # [2, 32]

    # Initialize a small BLT block/model with at least 2 layers
    from rwkv_lab.toy_blt_train import BLTRWKV7LanguageModel
    import torch.nn.functional as F
    model = BLTRWKV7LanguageModel(
        vocab_size=256,
        d_model=C,
        n_layers=2,
        threshold=3.0,
        max_patch=16
    )

    # 1. Evaluate untrained model
    model.eval()
    with torch.no_grad():
        _ = model(train_data)
        # Check starting patch IDs
        init_patch_ids = model.blocks[0].ffn.last_patch_ids
        init_num_patches = int(init_patch_ids[0].max()) + 1
        init_avg_len = T / init_num_patches
        init_entropy = model.blocks[0].ffn.last_entropy

    # Untrained model must have high entropy and patch lengths close or equal to 1.0
    assert init_entropy.mean().item() > 4.5
    assert init_avg_len < 1.1, f"Expected initial average patch length close to 1.0, got {init_avg_len}"

    # 2. Train the model for some steps
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    for step in range(30):
        optimizer.zero_grad()
        logits, entropy_losses = model(train_data, target_bytes=train_data)
        ce_loss = F.cross_entropy(logits[:, :-1].reshape(-1, 256), train_data[:, 1:].reshape(-1))
        entropy_loss = sum(entropy_losses)
        total_loss = ce_loss + 0.5 * entropy_loss
        total_loss.backward()
        optimizer.step()

    # 3. Evaluate trained model
    model.eval()
    with torch.no_grad():
        _ = model(train_data)
        trained_patch_ids = model.blocks[0].ffn.last_patch_ids
        trained_num_patches = int(trained_patch_ids[0].max()) + 1
        trained_avg_len = T / trained_num_patches
        trained_entropy = model.blocks[0].ffn.last_entropy

    # The trained model should have learned context-aware predictions, leading to
    # lower next-byte entropy on predictable tokens and larger patch lengths (> 1.0)
    print(f"\n--- Entropy Adaptation Proof ---")
    print(f"Initial entropy: {init_entropy.mean().item():.3f} | Trained entropy: {trained_entropy.mean().item():.3f}")
    print(f"Initial patch length: {init_avg_len:.3f} | Trained patch length: {trained_avg_len:.3f}")
    print(f"--------------------------------")

    assert trained_entropy.mean().item() < init_entropy.mean().item(), "Entropy should decrease after training"
    assert trained_avg_len > 1.1, f"Average patch length should increase after training, got {trained_avg_len}"
