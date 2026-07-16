import pytest
import torch
from torch import nn

from rwkv_lab.vision_loop import (
    FLAFactoredTimeMix,
    install_factored_timemix,
    load_loop_adapter_state,
    loop_adapter_state,
    loop_training_metrics,
    reset_loop_adapters,
    set_loop_scale,
    loop_telemetry_from_states,
)


class FakeAttention(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.proj = nn.Linear(width, width, bias=False)

    def forward(self, hidden_states, v_first=None, past_key_values=None, **kwargs):
        vf = hidden_states if v_first is None else v_first
        return self.proj(hidden_states), None, past_key_values, vf


def test_disabled_and_zero_gate_enabled_paths_preserve_base_output():
    torch.manual_seed(1)
    base = FakeAttention()
    wrapper = FLAFactoredTimeMix(
        base, hidden_size=8, num_heads=2, n_loops=2, loop_index=True)
    x = torch.randn(2, 5, 8)
    vf = torch.randn_like(x)
    expected = base(x, v_first=vf)
    actual = wrapper(x, v_first=vf)
    assert torch.equal(actual[0], expected[0]) and actual[3] is expected[3]
    wrapper.enabled = True
    enabled = wrapper(x, v_first=vf)
    assert torch.equal(enabled[0], expected[0])
    assert torch.count_nonzero(wrapper.loop.loop_index_embed) == 0


def test_checkpoint_excludes_frozen_core():
    wrapper = FLAFactoredTimeMix(FakeAttention(), hidden_size=8, num_heads=2)
    saved = loop_adapter_state([wrapper])[0]
    assert saved and not any(name.startswith("core.") for name in saved)


def test_loop_metrics_support_disabled_loop_index_embedding():
    wrapper = FLAFactoredTimeMix(
        FakeAttention(), hidden_size=8, num_heads=2, n_loops=2,
        loop_index=False)
    metrics = loop_training_metrics([wrapper])
    assert metrics["loop_index_rms"] == 0.0


def test_checkpoint_loader_rejects_partial_adapter_state():
    wrapper = FLAFactoredTimeMix(
        FakeAttention(), hidden_size=8, num_heads=2, n_loops=2, loop_index=True)
    saved = loop_adapter_state([wrapper])[0]
    saved.pop(next(iter(saved)))
    with pytest.raises(ValueError, match="key mismatch"):
        load_loop_adapter_state([wrapper], [saved])


def test_runtime_scale_and_reset_preserve_safe_loop_start():
    torch.manual_seed(2)
    wrapper = FLAFactoredTimeMix(
        FakeAttention(), hidden_size=8, num_heads=2, n_loops=2, loop_index=True)
    wrapper.enabled = True
    with torch.no_grad():
        wrapper.loop.residual_weight.fill_(0.1)
        wrapper.loop.gate_chan.fill_(0.2)
        wrapper.loop.loop_index_embed.fill_(0.3)
        wrapper.loop.iter_norm.weight.fill_(0.8)
    x = torch.randn(2, 5, 8)
    expected = wrapper.inner(x)[0]
    set_loop_scale([wrapper], 0.0)
    assert torch.equal(wrapper(x)[0], expected)
    reset_loop_adapters([wrapper])
    assert torch.count_nonzero(wrapper.loop.residual_weight) == 0
    assert torch.count_nonzero(wrapper.loop.gate_chan) == 0
    assert torch.count_nonzero(wrapper.loop.loop_index_embed) == 0
    assert torch.equal(wrapper.loop.iter_norm.weight, torch.ones_like(wrapper.loop.iter_norm.weight))


def test_loop_artifact_reports_executed_runtime_scale():
    wrapper = FLAFactoredTimeMix(
        FakeAttention(), hidden_size=8, num_heads=2, n_loops=2)
    with torch.no_grad():
        wrapper.loop.residual_weight[1].fill_(0.1)
    state = loop_adapter_state([wrapper])
    full = loop_telemetry_from_states(
        state, loop_count=2, gate_cap=0.25, step=10, runtime_scale=1.0)
    ramped = loop_telemetry_from_states(
        state, loop_count=2, gate_cap=0.25, step=10, runtime_scale=0.2)
    assert ramped["runtime_scale"] == 0.2
    assert ramped["layers"][0]["max_rw"] == pytest.approx(
        full["layers"][0]["max_rw"] * 0.2)


def test_single_pass_loop_telemetry_has_empty_refinement_rows():
    wrapper = FLAFactoredTimeMix(
        FakeAttention(), hidden_size=8, num_heads=2, n_loops=1)
    artifact = loop_telemetry_from_states(
        loop_adapter_state([wrapper]), loop_count=1, gate_cap=0.25, step=0)

    assert artifact["loop_count"] == 1
    assert artifact["layers"][0]["max_rw"] == 0.0
    assert artifact["layers"][0]["rw"] == []
    assert artifact["layers"][0]["split"]["channel_abs"] == []


def test_installed_adapter_uses_the_base_attention_device():
    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = FakeAttention()
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("Config", (), {"hidden_size": 8, "num_heads": 2})()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([Layer()])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model().to(device)
    model.requires_grad_(False)
    wrappers = install_factored_timemix(model, n_loops=2, loop_index=True)
    wrapper = wrappers[0]
    assert {parameter.device.type for parameter in wrapper.parameters()} == {device.type}
    wrapper.enabled = True
    x = torch.randn(2, 5, 8, device=device, requires_grad=True)
    wrapper(x)[0].sum().backward()
    assert wrapper.loop.residual_weight.grad is not None
