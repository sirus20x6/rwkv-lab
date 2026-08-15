"""Unit and integration tests for the KAN-RWKV Language Model."""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from rwkv_lab.kan_rwkv import KANLinear, KANRWKVLanguageModel


def test_kan_linear_shapes():
    """Verify initialization and forward pass shapes for KANLinear."""
    in_features = 16
    out_features = 32
    grid_size = 5
    spline_order = 3

    layer = KANLinear(
        in_features=in_features,
        out_features=out_features,
        grid_size=grid_size,
        spline_order=spline_order,
    )

    # 1D/2D/3D inputs
    x_2d = torch.randn(4, in_features)
    out_2d = layer(x_2d)
    assert out_2d.shape == (4, out_features)

    x_3d = torch.randn(2, 8, in_features)
    out_3d = layer(x_3d)
    assert out_3d.shape == (2, 8, out_features)


def test_kan_linear_grad_flow():
    """Verify that both base and spline parameters receive gradients in KANLinear."""
    in_features = 8
    out_features = 16
    layer = KANLinear(in_features=in_features, out_features=out_features)

    x = torch.randn(4, in_features)
    out = layer(x)
    loss = out.pow(2).mean()
    loss.backward()

    assert layer.base_weight.grad is not None
    assert layer.spline_weight.grad is not None
    assert torch.isfinite(layer.base_weight.grad).all()
    assert torch.isfinite(layer.spline_weight.grad).all()


def test_kan_rwkv_lm_shapes():
    """Verify initialization and forward pass shapes of KANRWKVLanguageModel."""
    vocab_size = 128
    d_model = 32
    n_layers = 2
    head_size = 16

    model = KANRWKVLanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        head_size=head_size,
        deepembed=True,
    )

    # Batch size 2, Sequence length 8
    ids = torch.randint(0, vocab_size, (2, 8))
    logits = model(ids)

    assert logits.shape == (2, 8, vocab_size)
    assert torch.isfinite(logits).all()


def test_kan_rwkv_lm_backward():
    """Verify gradients flow correctly to active parameters in KAN-RWKV."""
    vocab_size = 128
    d_model = 32
    n_layers = 2
    head_size = 16

    model = KANRWKVLanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        head_size=head_size,
        deepembed=True,
    )

    ids = torch.randint(0, vocab_size, (2, 8))
    logits = model(ids)
    loss = logits.pow(2).mean()
    loss.backward()

    # Check that parameters have gradients
    for name, param in model.named_parameters():
        # blocks.0.att.v0/v1/v2 are not used in layer 0 due to is_first_rwkv_layer=True.
        if "blocks.0.att.v" in name and any(suffix in name for suffix in (".v0", ".v1", ".v2")):
            continue

        assert param.grad is not None, f"Parameter {name} has no gradient!"
        assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradient!"


def test_kan_rwkv_lm_parallel_recurrent_parity():
    """Verify mathematical parity between parallel and recurrent step-by-step passes for KAN-RWKV."""
    torch.manual_seed(42)
    vocab_size = 64
    d_model = 16
    n_layers = 2
    head_size = 8

    model = KANRWKVLanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        head_size=head_size,
        deepembed=True,
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
    diff = (logits_parallel - logits_recurrent).abs().max().item()
    assert diff < 1e-4, f"Parity mismatch! Max diff: {diff}"


def test_kan_rwkv_lm_training_convergence():
    """Verify that a small KAN-RWKV training run successfully decreases loss."""
    torch.manual_seed(42)
    vocab_size = 32
    d_model = 16
    n_layers = 2
    head_size = 8

    model = KANRWKVLanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        head_size=head_size,
        deepembed=True,
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
