"""Tests for validating 100% accuracy of the ROSA Online Suffix Automaton retrieval model."""

import os
os.environ.setdefault("RWKV8_FORCE_PYREF", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from rwkv_lab.rosa import RosaLayer
from rwkv_lab.synthetic_tasks import AssocRecallTask


def test_rosa_deterministic_100_percent_accuracy_on_recall_small():
    """Verify that a properly parameterized ROSA layer achieves exactly 100% accuracy on AssocRecallTask.

    We use n_pairs = 3, n_keys = 6, n_vals = 6, yielding a vocab_size of 14 (<= 16).
    This guarantees that every vocabulary token is mapped to a unique 4-bit symbol in Route 0,
    completely preventing packed symbol collisions!
    """
    task = AssocRecallTask(n_pairs=3, n_keys=6, n_vals=6)
    vocab_size = task.vocab
    assert vocab_size <= 16, f"Vocab size {vocab_size} exceeds 16, which would cause route-symbol collisions!"

    # C = 8, M = 4, R = 2 routes
    C = 8
    M = 4

    rosa = RosaLayer(hidden_size=C, M=M, max_match=16)

    with torch.no_grad():
        nn.init.eye_(rosa.Wq.weight)
        nn.init.eye_(rosa.Wk.weight)
        nn.init.eye_(rosa.Wv.weight)
        nn.init.eye_(rosa.Wout.weight)
        rosa.e0.fill_(0.0)
        rosa.e1.fill_(1.0)

    B = 4
    device = "cpu"
    x, y, m = task.batch(B, device)
    T = x.shape[1]

    H = torch.zeros(B, T, C)
    for b in range(B):
        for t in range(T):
            token_id = x[b, t].item()
            for i in range(C):
                bit = 1.0 if (token_id & (1 << i)) != 0 else -1.0
                H[b, t, i] = bit

    with torch.no_grad():
        inj = rosa.injection(H)

    head_weight = torch.zeros(vocab_size, C)
    for token_id in range(vocab_size):
        for i in range(C):
            head_weight[token_id, i] = 1.0 if (token_id & (1 << i)) != 0 else -1.0

    logits = F.linear(inj, head_weight)

    acc = task.accuracy(logits, y, m)
    assert acc == 1.0, f"Expected 100% accuracy using ROSA, but got {acc * 100}%"
    print(f"[rosa-accuracy] Deterministic ROSA layer achieved exactly {acc * 100}% accuracy on AssocRecallTask! — OK")


def test_rosa_deterministic_100_percent_accuracy_on_recall_large():
    """Verify that a properly parameterized ROSA layer achieves exactly 100% accuracy on a larger AssocRecallTask."""
    task = AssocRecallTask(n_pairs=4, n_keys=6, n_vals=6)
    vocab_size = task.vocab
    assert vocab_size <= 16

    C = 8
    M = 4

    rosa = RosaLayer(hidden_size=C, M=M, max_match=16)

    with torch.no_grad():
        nn.init.eye_(rosa.Wq.weight)
        nn.init.eye_(rosa.Wk.weight)
        nn.init.eye_(rosa.Wv.weight)
        nn.init.eye_(rosa.Wout.weight)
        rosa.e0.fill_(0.0)
        rosa.e1.fill_(1.0)

    B = 4
    device = "cpu"
    x, y, m = task.batch(B, device)
    T = x.shape[1]

    H = torch.zeros(B, T, C)
    for b in range(B):
        for t in range(T):
            token_id = x[b, t].item()
            for i in range(C):
                bit = 1.0 if (token_id & (1 << i)) != 0 else -1.0
                H[b, t, i] = bit

    with torch.no_grad():
        inj = rosa.injection(H)

    head_weight = torch.zeros(vocab_size, C)
    for token_id in range(vocab_size):
        for i in range(C):
            head_weight[token_id, i] = 1.0 if (token_id & (1 << i)) != 0 else -1.0

    logits = F.linear(inj, head_weight)

    acc = task.accuracy(logits, y, m)
    assert acc == 1.0, f"Expected 100% accuracy using ROSA on large recall, but got {acc * 100}%"
    print(f"[rosa-accuracy] Deterministic ROSA layer achieved exactly {acc * 100}% accuracy on large AssocRecallTask! — OK")


def test_rosa_optimization_to_100_percent_accuracy():
    """Verify that ROSA parameters can be trained to achieve 100% accuracy on AssocRecallTask."""
    torch.manual_seed(42)
    task = AssocRecallTask(n_pairs=1, n_keys=2, n_vals=2)
    vocab_size = task.vocab
    assert vocab_size <= 16

    C = 8
    M = 4

    # Learnable embedding to map token ids to continuous representations
    emb = nn.Embedding(vocab_size, C)
    rosa = RosaLayer(hidden_size=C, M=M, max_match=16)
    head = nn.Linear(C, vocab_size, bias=False)

    # Keep the discrete projections frozen to Identity to ensure stable symbol routing
    with torch.no_grad():
        nn.init.eye_(rosa.Wq.weight)
        nn.init.eye_(rosa.Wk.weight)
        nn.init.eye_(rosa.Wv.weight)
        nn.init.eye_(rosa.Wout.weight)
        rosa.e0.fill_(0.0)
        rosa.e1.fill_(1.0)

    for p in [rosa.Wq.weight, rosa.Wk.weight, rosa.Wv.weight, rosa.Wout.weight]:
        p.requires_grad = False

    # We want to train the model to output the queried value using ROSA's injection
    optimizer = torch.optim.AdamW(
        list(emb.parameters()) + list(rosa.parameters()) + list(head.parameters()),
        lr=0.1
    )

    B = 4
    device = "cpu"
    x, y, m = task.batch(B, device)

    # Train to achieve 100% accuracy
    success = False
    for step in range(30):
        optimizer.zero_grad()
        H = emb(x)
        inj = rosa.injection(H)
        logits = head(inj)

        # Cross entropy loss where mask m is 1.0 (only the final query value)
        active_logits = logits[m > 0]
        active_targets = y[m > 0]
        loss = F.cross_entropy(active_logits, active_targets)
        loss.backward()
        optimizer.step()

        acc = task.accuracy(logits, y, m)
        if acc == 1.0:
            success = True
            print(f"[rosa-accuracy] Trained ROSA model achieved 100% accuracy at step {step}! — OK")
            break

    assert success, "Failed to train ROSA model to 100% accuracy within 30 steps."
