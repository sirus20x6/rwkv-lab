import pytest
import torch
from torch import nn

from rwkv_lab.vision_loop import (
    FLAFactoredTimeMix,
    capture_loop_refinement_caches,
    install_factored_timemix,
    load_loop_adapter_state,
    loop_adapter_state,
    loop_training_metrics,
    reset_loop_adapters,
    reset_loop_inference_cache,
    restore_loop_refinement_caches,
    set_loop_scale,
    stack_legacy_fla_caches,
    stack_loop_refinement_cache_snapshots,
    loop_telemetry_from_states,
)


class FakeAttention(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.proj = nn.Linear(width, width, bias=False)

    def forward(self, hidden_states, v_first=None, past_key_values=None, **kwargs):
        vf = hidden_states if v_first is None else v_first
        return self.proj(hidden_states), None, past_key_values, vf


class FakeRecurrentCache:
    """Small outer-cache stand-in with the protocol FLA TimeMix consumes."""
    def __init__(self):
        self.states = {}

    def __len__(self):
        return max(self.states, default=-1) + 1

    def __getitem__(self, layer_idx):
        return self.states[layer_idx]

    def update(self, *, recurrent_state=None, conv_state=None, layer_idx=0,
               offset=1, **_kwargs):
        state = self.states.setdefault(layer_idx, {
            "recurrent_state": None, "conv_state": None,
            "attn_state": None, "ffn_state": None,
        })
        if recurrent_state is not None:
            state["recurrent_state"] = recurrent_state
        if conv_state is not None:
            state["conv_state"] = conv_state
        return state


class FakeLegacyFLACache:
    """CPU stand-in for FLA's list-backed LegacyFLACache protocol."""
    def __init__(self, seen_tokens=0):
        self.states = []
        self._seen_tokens = int(seen_tokens)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, layer_idx):
        return self.states[layer_idx]

    def update(self, *, recurrent_state=None, attn_state=None, conv_state=None,
               ffn_state=None, layer_idx=0, offset=1, **_kwargs):
        while len(self.states) <= layer_idx:
            self.states.append({
                "recurrent_state": None, "attn_state": None,
                "conv_state": None, "ffn_state": None,
            })
        state = self.states[layer_idx]
        if recurrent_state is not None:
            state["recurrent_state"] = recurrent_state
        if attn_state is not None:
            state["attn_state"] = attn_state
        if conv_state is not None:
            state["conv_state"] = conv_state
        if ffn_state is not None:
            state["ffn_state"] = ffn_state
        if layer_idx == 0:
            self._seen_tokens += int(offset or 0)
        return state

    @classmethod
    def from_legacy_cache(cls, states, seen_tokens=0):
        cache = cls(seen_tokens)
        cache.states = list(states)
        return cache


class FakeRecurrentAttention(nn.Module):
    """Causal recurrence whose chunked and incremental values are identical."""
    def __init__(self, width=8, layer_idx=0):
        super().__init__()
        self.proj = nn.Linear(width, width, bias=False)
        self.layer_idx = layer_idx

    def forward(self, hidden_states, v_first=None, past_key_values=None,
                use_cache=False, **_kwargs):
        batch, _length, width = hidden_states.shape
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            saved = past_key_values[self.layer_idx]
            recurrent = saved["recurrent_state"]
            previous = saved["conv_state"]
        else:
            recurrent = hidden_states.new_zeros(batch, width)
            previous = hidden_states.new_zeros(batch, width)
        outputs = []
        for token in hidden_states.unbind(1):
            mixed = token + 0.2 * previous
            recurrent = 0.6 * recurrent + self.proj(mixed)
            outputs.append(recurrent)
            previous = token
        out = torch.stack(outputs, dim=1)
        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent,
                conv_state=previous,
                layer_idx=self.layer_idx,
                offset=hidden_states.shape[1],
            )
        vf = out if v_first is None else v_first
        return out, None, past_key_values, vf


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


@pytest.mark.parametrize("prefill", [1, 3, 5])
def test_cached_factored_loop_matches_full_sequence(prefill):
    torch.manual_seed(31)
    wrapper = FLAFactoredTimeMix(
        FakeRecurrentAttention(), hidden_size=8, num_heads=2,
        n_loops=2, gate_cap=0.25, loop_index=True,
    )
    wrapper.enabled = True
    with torch.no_grad():
        wrapper.loop.residual_weight[1].copy_(torch.tensor([0.12, -0.08]))
        wrapper.loop.gate_chan[1].copy_(torch.linspace(-0.15, 0.2, 8))
        wrapper.loop.loop_index_embed[1].copy_(torch.linspace(-0.1, 0.1, 8))

    x = torch.randn(2, 7, 8)
    full = wrapper(x, use_cache=False)[0]

    outer = FakeRecurrentCache()
    chunks = [wrapper(x[:, :prefill], past_key_values=outer, use_cache=True)[0]]
    for position in range(prefill, x.shape[1]):
        chunks.append(wrapper(
            x[:, position:position + 1],
            past_key_values=outer,
            use_cache=True,
        )[0])
    incremental = torch.cat(chunks, dim=1)

    torch.testing.assert_close(incremental, full, rtol=2e-6, atol=2e-6)
    # A downstream LM head sees the same logits too, not just locally equal
    # TimeMix values.
    head = nn.Linear(8, 19, bias=False)
    torch.testing.assert_close(head(incremental), head(full), rtol=2e-6, atol=2e-6)
    assert outer.states[0]["recurrent_state"] is not None
    assert wrapper._refinement_caches[0].state["recurrent_state"] is not None


def test_new_outer_cache_resets_refinement_stream_and_reset_helper_clears_it():
    torch.manual_seed(47)
    wrapper = FLAFactoredTimeMix(
        FakeRecurrentAttention(), hidden_size=8, num_heads=2,
        n_loops=2, loop_index=True,
    )
    wrapper.enabled = True
    with torch.no_grad():
        wrapper.loop.residual_weight[1].fill_(0.1)
        wrapper.loop.loop_index_embed[1].fill_(0.05)
    x = torch.randn(1, 4, 8)

    first = wrapper(x, past_key_values=FakeRecurrentCache(), use_cache=True)[0]
    second = wrapper(x, past_key_values=FakeRecurrentCache(), use_cache=True)[0]
    torch.testing.assert_close(second, first, rtol=0, atol=0)

    assert wrapper._refinement_caches
    reset_loop_inference_cache([wrapper])
    assert wrapper._cache_owner is None
    assert wrapper._refinement_caches == []


def test_stack_legacy_fla_caches_concatenates_nested_batch_states():
    first = FakeLegacyFLACache(11)
    second = FakeLegacyFLACache(17)
    for value, cache in ((1.0, first), (2.0, second)):
        cache.states = [{
            "recurrent_state": torch.full((1, 2, 3), value),
            "attn_state": None,
            "conv_state": torch.full((1, 4), value + 0.1),
            "ffn_state": (
                torch.full((1, 5), value + 0.2),
                [torch.full((1, 1), value + 0.3)],
            ),
        }]

    stacked = stack_legacy_fla_caches([first, second])

    assert isinstance(stacked, FakeLegacyFLACache)
    assert stacked._seen_tokens == 17
    assert stacked[0]["recurrent_state"].shape == (2, 2, 3)
    assert stacked[0]["recurrent_state"][:, 0, 0].tolist() == [1.0, 2.0]
    assert stacked[0]["attn_state"] is None
    assert stacked[0]["ffn_state"][0].shape == (2, 5)
    assert stacked[0]["ffn_state"][1][0].shape == (2, 1)


def test_stack_legacy_fla_caches_rejects_nonbatch_shape_mismatch():
    first = FakeLegacyFLACache.from_legacy_cache([{
        "recurrent_state": torch.zeros(1, 2, 3), "attn_state": None,
        "conv_state": None, "ffn_state": None,
    }])
    second = FakeLegacyFLACache.from_legacy_cache([{
        "recurrent_state": torch.zeros(1, 3, 3), "attn_state": None,
        "conv_state": None, "ffn_state": None,
    }])
    with pytest.raises(ValueError, match="shapes disagree"):
        stack_legacy_fla_caches([first, second])


def test_stack_installed_legacy_fla_cache_preserves_layer_states():
    # The installed LegacyFLACache accepts only list (not tuple) in
    # from_legacy_cache. Exercise the real dependency so that silent empty
    # cache regression cannot pass the protocol-only fake.
    from fla.models.utils import Cache as InstalledFLACache

    caches = []
    for value, length in ((1.0, 3), (2.0, 7)):
        cache = InstalledFLACache()
        cache.update(
            recurrent_state=torch.full((1, 2, 3), value),
            conv_state=torch.full((1, 4), value),
            layer_idx=0, offset=length)
        caches.append(cache)

    stacked = stack_legacy_fla_caches(caches)
    assert isinstance(stacked, InstalledFLACache)
    assert len(stacked) == 1
    assert stacked._seen_tokens == 7
    assert stacked[0]["recurrent_state"].shape == (2, 2, 3)
    assert stacked[0]["recurrent_state"][:, 0, 0].tolist() == [1.0, 2.0]


def test_capture_stack_restore_refinement_caches_preserves_outer_owner():
    torch.manual_seed(73)
    wrapper = FLAFactoredTimeMix(
        FakeRecurrentAttention(), hidden_size=8, num_heads=2,
        n_loops=3, gate_cap=0.25, loop_index=True,
    )
    wrapper.enabled = True
    with torch.no_grad():
        wrapper.loop.residual_weight[1:].fill_(0.1)

    rows = [torch.randn(1, length, 8) for length in (3, 5)]
    outer_caches = []
    snapshots = []
    for row in rows:
        outer = FakeLegacyFLACache()
        wrapper(row, past_key_values=outer, use_cache=True)
        outer_caches.append(outer)
        snapshots.append(capture_loop_refinement_caches([wrapper], owner=outer))

    stacked_outer = stack_legacy_fla_caches(outer_caches)
    stacked_refinement = stack_loop_refinement_cache_snapshots(snapshots)
    restore_loop_refinement_caches(
        [wrapper], stacked_refinement, owner=stacked_outer)

    assert wrapper._cache_owner is stacked_outer
    assert len(wrapper._refinement_caches) == 2
    for pass_index, cache in enumerate(wrapper._refinement_caches):
        expected = torch.cat([
            snapshot.wrappers[0][pass_index].state["recurrent_state"]
            for snapshot in snapshots
        ], dim=0)
        torch.testing.assert_close(cache.state["recurrent_state"], expected)
        assert cache.state["recurrent_state"].shape[0] == 2
        assert cache.seen_tokens == 5
    # Reusing the installed owner preserves the restored cache objects. A new
    # outer owner must reset them rather than leaking recurrence across rows.
    restored_ids = tuple(map(id, wrapper._refinement_caches))
    assert tuple(map(id, wrapper._prepare_refinement_caches(stacked_outer))) == restored_ids
    replacement = FakeLegacyFLACache()
    fresh = wrapper._prepare_refinement_caches(replacement)
    assert wrapper._cache_owner is replacement
    assert tuple(map(id, fresh)) != restored_ids
    assert all(cache.state is None for cache in fresh)


def test_capture_refinement_cache_rejects_wrong_owner():
    wrapper = FLAFactoredTimeMix(
        FakeRecurrentAttention(), hidden_size=8, num_heads=2, n_loops=2)
    wrapper.enabled = True
    owner = FakeLegacyFLACache()
    wrapper(torch.randn(1, 2, 8), past_key_values=owner, use_cache=True)
    with pytest.raises(ValueError, match="does not belong"):
        capture_loop_refinement_caches(
            [wrapper], owner=FakeLegacyFLACache())


def test_stacked_outer_and_refinement_caches_match_independent_next_tokens():
    """Variable-length scalar prefills may safely become one decode batch."""
    import copy

    torch.manual_seed(89)
    template = FLAFactoredTimeMix(
        FakeRecurrentAttention(), hidden_size=8, num_heads=2,
        n_loops=3, gate_cap=0.25, loop_index=True)
    template.enabled = True
    with torch.no_grad():
        template.loop.residual_weight[1:].copy_(torch.tensor([
            [0.11, -0.07], [0.04, 0.09]]))
        template.loop.gate_chan[1:].copy_(torch.linspace(
            -0.2, 0.2, 16).reshape(2, 8))
        template.loop.loop_index_embed[1:].copy_(torch.linspace(
            -0.1, 0.1, 16).reshape(2, 8))
    batched = copy.deepcopy(template)
    scalar = copy.deepcopy(template)
    rows = [torch.randn(1, length, 8) for length in (2, 6, 4)]
    next_rows = [torch.randn(1, 1, 8) for _ in rows]

    outer_rows, snapshots = [], []
    for row in rows:
        outer = FakeLegacyFLACache()
        batched(row, past_key_values=outer, use_cache=True)
        outer_rows.append(outer)
        snapshots.append(capture_loop_refinement_caches([batched], owner=outer))
    stacked_outer = stack_legacy_fla_caches(outer_rows)
    restore_loop_refinement_caches(
        [batched], stack_loop_refinement_cache_snapshots(snapshots),
        owner=stacked_outer)
    actual = batched(
        torch.cat(next_rows), past_key_values=stacked_outer, use_cache=True)[0]

    expected_rows = []
    for row, next_row in zip(rows, next_rows):
        outer = FakeLegacyFLACache()
        scalar.reset_inference_cache()
        scalar(row, past_key_values=outer, use_cache=True)
        expected_rows.append(scalar(
            next_row, past_key_values=outer, use_cache=True)[0])
    expected = torch.cat(expected_rows)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
