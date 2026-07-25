import torch
from torch import nn

from rwkv_lab.activation_checkpointing import (
    checkpointed_layer_indices,
    selective_activation_checkpointing,
)


class CheckpointAwareLayer(nn.Module):
    gradient_checkpointing = False

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))
        self.calls = 0

    def _body(self, value):
        self.calls += 1
        return value * self.weight

    def forward(self, value):
        # Match FLA GradientCheckpointingLayer's real gating contract.
        if self.gradient_checkpointing and self.training:
            return self._gradient_checkpointing_func(self._body, value)
        return self._body(value)


def test_checkpoint_selection_preserves_excluded_hook_sites():
    assert checkpointed_layer_indices(
        8, sequence_tokens=8704, min_tokens=4096,
        excluded_layers=(3, 5)) == (0, 1, 2, 4, 6, 7)
    assert checkpointed_layer_indices(
        8, sequence_tokens=4095, min_tokens=4096) == ()
    assert checkpointed_layer_indices(
        8, sequence_tokens=8704, min_tokens=4096,
        grad_enabled=False) == ()


def test_selected_layers_recompute_and_restore_state():
    layers = nn.ModuleList([CheckpointAwareLayer() for _ in range(3)])
    layers.eval()
    value = torch.ones(1, requires_grad=True)
    with selective_activation_checkpointing(
            layers, sequence_tokens=8704, min_tokens=4096,
            excluded_layers=(1,)) as selected:
        assert selected == (0, 2)
        for layer in layers:
            value = layer(value)
    value.sum().backward()

    assert layers[0].calls == 2
    assert layers[1].calls == 1
    assert layers[2].calls == 2
    assert all(not layer.gradient_checkpointing for layer in layers)
    assert all(not layer.training for layer in layers)
    assert all(layer.weight.grad is not None for layer in layers)


def test_short_sequence_has_no_recomputation():
    layers = nn.ModuleList([CheckpointAwareLayer() for _ in range(2)])
    value = torch.ones(1, requires_grad=True)
    with selective_activation_checkpointing(
            layers, sequence_tokens=1024, min_tokens=4096) as selected:
        assert selected == ()
        for layer in layers:
            value = layer(value)
    value.sum().backward()
    assert [layer.calls for layer in layers] == [1, 1]
