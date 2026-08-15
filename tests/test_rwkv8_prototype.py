"""Unit and integration tests for the RWKV-8 Prototype Language Model."""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from rwkv_lab.rwkv8_prototype import RWKV8LanguageModel


def test_rwkv8_lm_shapes():
    """Verify initialization and forward pass shapes."""
    vocab_size = 128
    d_model = 32
    n_layers = 2
    head_size = 16

    model = RWKV8LanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        head_size=head_size,
        deepembed=True
    )

    # Batch size 2, Sequence length 8
    ids = torch.randint(0, vocab_size, (2, 8))
    logits = model(ids)

    assert logits.shape == (2, 8, vocab_size)
    assert torch.isfinite(logits).all()


def test_rwkv8_lm_backward():
    """Verify gradients flow correctly to active parameters."""
    vocab_size = 128
    d_model = 32
    n_layers = 2
    head_size = 16

    model = RWKV8LanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        head_size=head_size,
        deepembed=True
    )

    ids = torch.randint(0, vocab_size, (2, 8))
    logits = model(ids)
    loss = logits.pow(2).mean()
    loss.backward()

    # Check that parameters have gradients
    for name, param in model.named_parameters():
        # blocks.0.att.v0/v1/v2 are not used in layer 0 due to is_first_rwkv_layer=True.
        # This is expected and standard behavior in RWKV-7/8 cross-layer value-residual design.
        if "blocks.0.att.v" in name and any(suffix in name for suffix in (".v0", ".v1", ".v2")):
            continue

        assert param.grad is not None, f"Parameter {name} has no gradient!"
        assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradient!"


def test_rwkv8_lm_parallel_recurrent_parity():
    """Verify mathematical parity between parallel and recurrent step-by-step passes."""
    torch.manual_seed(42)
    vocab_size = 64
    d_model = 16
    n_layers = 2
    head_size = 8

    model = RWKV8LanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        head_size=head_size,
        deepembed=True
    )
    model.eval()

    B, T = 1, 5
    ids = torch.randint(0, vocab_size, (B, T))

    # 1. Parallel forward pass
    with torch.no_grad():
        logits_parallel = model(ids)

    # 2. Recurrent forward pass step-by-step
    logits_recurrent_list = []
    states = None
    with torch.no_grad():
        for t in range(T):
            token = ids[:, t : t + 1]
            out, states = model(token, states=states, return_state=True)
            logits_recurrent_list.append(out)

    logits_recurrent = torch.cat(logits_recurrent_list, dim=1)

    # Compare parallel and recurrent logits
    # Allow small numerical tolerance due to Python float/accumulation order
    diff = (logits_parallel - logits_recurrent).abs().max().item()
    assert diff < 1e-4, f"Parity mismatch! Max diff: {diff}"


def test_rwkv8_lm_training_convergence():
    """Verify that a small training run successfully decreases loss."""
    torch.manual_seed(42)
    vocab_size = 32
    d_model = 16
    n_layers = 2
    head_size = 8

    model = RWKV8LanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        head_size=head_size,
        deepembed=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    ids = torch.randint(0, vocab_size, (2, 8))

    # Train for a few steps
    initial_loss = None
    for step in range(10):
        model.train()
        optimizer.zero_grad()
        logits = model(ids)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab_size), ids[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()

        if step == 0:
            initial_loss = loss.item()

    final_loss = loss.item()
    assert final_loss < initial_loss, f"Loss did not decrease! {initial_loss} -> {final_loss}"
