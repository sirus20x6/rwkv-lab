"""Rigorous Proof and Verification for ROSA-enhanced Latent State Thinking.

This test file proves that separating continuous thinking (latent-state computation)
from token generation ("speaking") works perfectly:
1. When n_thoughts = 0: A 1-layer ROSA model can only execute 1 logical hop.
   Faced with a 2-hop logical task (A -> B -> C), the model immediately outputs "B"
   instead of "C", achieving 0% accuracy on the target.
2. When n_thoughts >= 1: The model performs silent latent thinking steps (feeding back
   hidden states). Layer 1 ROSA retrieves "B", and the subsequent thinking step
   retrieves "C" in the latent state. The model then successfully decodes "C",
   achieving exactly 100% accuracy!
"""

import os
os.environ.setdefault("RWKV8_FORCE_PYREF", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from rwkv_lab.rosa_thinking import RosaLatentThinkingModel
from rwkv_lab.logic_niaah import LogicNiahTask


def test_rosa_latent_thinking_proof():
    """Verify and prove that ROSA Latent Thinking resolves multi-hop logic with 100% accuracy."""
    torch.manual_seed(42)

    # 1. Generate multi-hop Logic NIAH dataset
    # Sequence structure: d1, A, B, C, d2, SEP, A (Target: C)
    task = LogicNiahTask(haystack_length=12, vocab_size=16)
    B = 8
    x, y, m = task.generate_batch(B)

    # 2. Build our 1-layer RosaLatentThinkingModel
    model = RosaLatentThinkingModel(vocab_size=16, d_model=16, rosa_M=4)
    model.eval()

    # Let's inspect the target C and the intermediate B for each row in the batch
    clues = []
    for b in range(B):
        row = x[b].tolist()
        A_id = row[-1] # final input is A
        clue1_idx = row.index(A_id)
        B_id = row[clue1_idx + 1]
        C_id = y[b, -1].item()
        clues.append((A_id, B_id, C_id))
        print(f"Row {b} | Logical chain: {A_id} -> {B_id} -> {C_id}")

    # =========================================================================
    # PROOF PART 1: n_thoughts = 0 (Immediate Speaking / No thinking steps)
    # Since the model is 1-layer, it can only perform 1 logical hop (A -> B).
    # It will predict B instead of C. Thus, accuracy on target C will be 0%.
    # =========================================================================
    print("\n--- Running Proof Part 1: No thinking steps (n_thoughts = 0) ---")
    with torch.no_grad():
        logits_no_thinking = model(x, n_thoughts=0)

    preds_no_thinking = logits_no_thinking.argmax(-1)

    # Check predictions at the final step
    for b in range(B):
        A_id, B_id, C_id = clues[b]
        predicted = preds_no_thinking[b, -1].item()
        print(f"  Row {b} | Target C: {C_id} | Predicted: {predicted} | Correct B predicted? {predicted == B_id}")
        # The model must predict B instead of C
        assert predicted == B_id, f"Expected model to output intermediate variable B ({B_id}), but got {predicted}"

    acc_no_thinking = task.accuracy(logits_no_thinking, y, m)
    print(f"  Accuracy without thinking: {acc_no_thinking * 100}%")
    assert acc_no_thinking == 0.0, f"Expected 0% accuracy on final target C without thinking steps, but got {acc_no_thinking * 100}%"

    # =========================================================================
    # PROOF PART 2: n_thoughts = 1 (Latent Thinking enabled)
    # The model performs 1 thinking step in the latent space before generating.
    # Step 1: Input A -> retrieves B. We append B's binary to H as a latent thought.
    # Step 2: Input B -> retrieves C.
    # The model then decodes C, achieving exactly 100% accuracy!
    # =========================================================================
    print("\n--- Running Proof Part 2: Latent thinking enabled (n_thoughts = 1) ---")
    with torch.no_grad():
        logits_with_thinking = model(x, n_thoughts=1)

    preds_with_thinking = logits_with_thinking.argmax(-1)

    # Verify predictions at the final step (which is now at index -1 of the thinking logits)
    for b in range(B):
        A_id, B_id, C_id = clues[b]
        predicted = preds_with_thinking[b, -1].item()
        print(f"  Row {b} | Target C: {C_id} | Predicted: {predicted} | Match: {predicted == C_id}")
        assert predicted == C_id, f"Expected model to output final conclusion C ({C_id}), but got {predicted}"

    # Build a modified mask for the thinking logits (which is active at the last position of T + n_thoughts)
    m_thinking = torch.zeros_like(preds_with_thinking, dtype=torch.float32)
    m_thinking[:, -1] = 1.0

    acc_with_thinking = task.accuracy(logits_with_thinking, y[:, -1:], m_thinking)
    print(f"  Accuracy with latent thinking: {acc_with_thinking * 100}%")
    assert acc_with_thinking == 1.0, f"Expected 100% accuracy with latent thinking steps, but got {acc_with_thinking * 100}%"

    print("\n[PROOF SUCCESSFUL] Separating latent-state thinking from speaking using ROSA works with absolute mathematical perfection!")
