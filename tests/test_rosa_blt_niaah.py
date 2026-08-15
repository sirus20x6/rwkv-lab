"""Unit and integration tests for ROSA + BLT and ROSA + Logic NIAH (Needle in a Haystack)."""

import os
os.environ.setdefault("RWKV8_FORCE_PYREF", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from rwkv_lab.rosa_blt_prototype import BLT_ROSA_LanguageModel
from rwkv_lab.logic_niaah import LogicNiahTask, LogicNiahRosaSolver


def test_rosa_blt_model_shapes_and_gradients():
    """Verify that the combined ROSA + BLT language model initializes, forwards, and backward-propagates correctly."""
    torch.manual_seed(42)
    vocab_size = 256
    d_model = 32
    n_layers = 2
    head_size = 16

    model = BLT_ROSA_LanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        head_size=head_size,
        threshold=3.0,
        max_patch=8,
    )

    # We must activate ROSA readout by making e1 != e0
    with torch.no_grad():
        for block in model.blocks:
            block.rosa.e1.fill_(0.1)

    # Input batch size 2, sequence length 16
    ids = torch.randint(0, vocab_size, (2, 16))

    # Forward pass with target bytes to train entropy predictor heads
    logits, entropy_losses = model(ids, target_bytes=ids)

    assert logits.shape == (2, 16, vocab_size)
    assert len(entropy_losses) == n_layers
    assert all(loss.shape == () for loss in entropy_losses)

    # Loss backward pass to check gradient flow
    total_loss = logits.pow(2).mean() + sum(entropy_losses)
    total_loss.backward()

    # Check active parameters have gradients
    for name, param in model.named_parameters():
        # Exclude block.0.att.v0/v1/v2 which are inactive in layer 0 due to is_first_rwkv_layer
        if "blocks.0.att.v" in name and any(suffix in name for suffix in (".v0", ".v1", ".v2")):
            continue
        # Exclude ROSA Wk as its gradient is deferred (standard ROSA behavior)
        if "rosa.Wk" in name:
            continue

        assert param.grad is not None, f"Parameter {name} has no gradient!"
        assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradient!"


def test_logic_niaah_task_generation():
    """Verify that LogicNiahTask generates correct shapes and mask boundaries."""
    task = LogicNiahTask(haystack_length=24, vocab_size=16)
    B = 4
    x, y, m = task.generate_batch(B)

    # Sequence length T = L1 + 3 + L2 + 3 = 24 + 6 = 30
    # Inputs/Targets are shifted by 1, so shape is [B, T-1] = [4, 29]
    assert x.shape == (B, 29)
    assert y.shape == (B, 29)
    assert m.shape == (B, 29)

    # Mask is active ONLY at the very last token
    assert (m[:, :-1] == 0.0).all()
    assert (m[:, -1] == 1.0).all()


def test_rosa_solver_achieves_100_percent_on_logic_niaah():
    """Verify that our multi-layer ROSA solver achieves exactly 100% accuracy on LogicNiahTask over various haystack lengths."""
    for L in (12, 36, 72):
        task = LogicNiahTask(haystack_length=L, vocab_size=16)
        B = 8
        x, y, m = task.generate_batch(B)

        # Instantiate 2-layer ROSA solver
        solver = LogicNiahRosaSolver(vocab_size=16, d_model=8, rosa_M=4)

        with torch.no_grad():
            logits = solver(x)

        acc = task.accuracy(logits, y, m)
        assert acc == 1.0, f"Expected 100% accuracy on Logic NIAH with haystack L={L}, but got {acc * 100}%"
        print(f"[logic-niaah] ROSA solver achieved exactly {acc * 100}% accuracy on haystack length {L}! — OK")
