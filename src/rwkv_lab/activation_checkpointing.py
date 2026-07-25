"""Selective activation checkpointing for unusually long decoder sequences.

The ordinary caption batches are kept on the fastest path.  Once a sequence
crosses a configured token threshold, decoder blocks are recomputed during
backward instead of retaining their large intermediate activations.  Stateful
hook sites (visual reinjection and Engram) can be excluded while the remaining
blocks still provide most of the memory saving.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import partial
from typing import Iterator, Sequence

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def checkpointed_layer_indices(layer_count: int, *, sequence_tokens: int,
                               min_tokens: int,
                               excluded_layers: Sequence[int] = (),
                               grad_enabled: bool = True) -> tuple[int, ...]:
    """Return decoder layers to checkpoint for this particular forward."""
    if min_tokens <= 0 or sequence_tokens < min_tokens or not grad_enabled:
        return ()
    excluded = {int(index) for index in excluded_layers}
    return tuple(index for index in range(int(layer_count))
                 if index not in excluded)


@contextmanager
def selective_activation_checkpointing(
        layers: Sequence[nn.Module], *, sequence_tokens: int, min_tokens: int,
        excluded_layers: Sequence[int] = ()) -> Iterator[tuple[int, ...]]:
    """Temporarily checkpoint selected FLA ``GradientCheckpointingLayer``s.

    Non-reentrant checkpointing correctly handles FLA's tensor-valued keyword
    arguments (notably RWKV's cross-layer ``v_first`` stream).  Per-layer state
    is restored immediately after the forward; the already-created autograd
    checkpoint closures remain valid for backward.
    """
    selected = checkpointed_layer_indices(
        len(layers), sequence_tokens=sequence_tokens, min_tokens=min_tokens,
        excluded_layers=excluded_layers, grad_enabled=torch.is_grad_enabled())
    if not selected:
        yield selected
        return

    checkpoint_fn = partial(checkpoint, use_reentrant=False)
    prior: list[tuple[nn.Module, bool, object, bool]] = []
    for index in selected:
        layer = layers[index]
        enabled = bool(getattr(layer, "gradient_checkpointing", False))
        function = getattr(layer, "_gradient_checkpointing_func", None)
        training = bool(layer.training)
        prior.append((layer, enabled, function, training))
        # FLA's GradientCheckpointingLayer gates its checkpoint wrapper on
        # both ``gradient_checkpointing`` and ``training``.  The frozen RWKV
        # base normally remains in eval mode, so enable training mode only for
        # the selected blocks while their forward is constructed.  RWKV7
        # blocks contain no dropout or running-stat modules, and the original
        # state is restored before leaving this context.
        layer.train(True)
        layer.gradient_checkpointing = True
        layer._gradient_checkpointing_func = checkpoint_fn
    try:
        yield selected
    finally:
        for layer, enabled, function, training in prior:
            layer.gradient_checkpointing = enabled
            layer.train(training)
            if function is None:
                try:
                    delattr(layer, "_gradient_checkpointing_func")
                except AttributeError:
                    pass
            else:
                layer._gradient_checkpointing_func = function
