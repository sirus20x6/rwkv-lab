from __future__ import annotations

import pytest
import torch
from torch import nn

from rwkv_lab.mage_flow_tread_looping import (
    CombinedConfig,
    FactorizationConfig,
    LearnedFactoredLoopAdapter,
    LoopingConfig,
    TreadConfig,
    TreadFactoredLoopController,
    TreadLoopConfig,
    extract_tread_route,
    mageflow_loop_telemetry_payload,
    restore_tread_route,
    write_mageflow_loop_telemetry,
)


def _packed_fixture():
    tokens = torch.arange(9, dtype=torch.float32).view(1, 9, 1)
    rope = torch.arange(18, dtype=torch.float32).view(9, 2)
    cu_lens = torch.tensor([0, 2, 5, 9], dtype=torch.int32)
    return tokens, rope, cu_lens


def test_tread_exact_restoration_with_variable_native_resolutions():
    tokens, rope, cu_lens = _packed_fixture()
    generator = torch.Generator().manual_seed(123)
    active, active_rope, route = extract_tread_route(
        tokens,
        rope,
        cu_lens,
        bypass_fraction=0.5,
        generator=generator,
    )
    assert route.active_cu_lens.tolist() == [0, 1, 3, 5]
    assert route.original_cu_lens.tolist() == [0, 2, 5, 9]
    assert torch.equal(active_rope, rope.index_select(0, route.active_indices))
    assert torch.equal(restore_tread_route(active, route), tokens)
    assert sorted(
        torch.cat((route.active_indices, route.bypass_indices)).tolist()
    ) == list(range(9))


def test_tread_bypass_tokens_skip_middle_computation_but_keep_gradient():
    tokens, rope, cu_lens = _packed_fixture()
    tokens.requires_grad_(True)
    active, _active_rope, route = extract_tread_route(
        tokens,
        rope,
        cu_lens,
        bypass_fraction=0.5,
        generator=torch.Generator().manual_seed(7),
    )
    restored = restore_tread_route(active * 2, route)
    restored.sum().backward()
    gradient = tokens.grad.squeeze()
    assert torch.equal(
        gradient.index_select(0, route.active_indices),
        torch.full_like(route.active_indices, 2, dtype=torch.float32),
    )
    assert torch.equal(
        gradient.index_select(0, route.bypass_indices),
        torch.ones_like(route.bypass_indices, dtype=torch.float32),
    )


def test_tread_never_routes_text_and_does_not_cross_samples():
    tokens, rope, cu_lens = _packed_fixture()
    active, _active_rope, route = extract_tread_route(
        tokens,
        rope,
        cu_lens,
        bypass_fraction=0.25,
        generator=torch.Generator().manual_seed(9),
    )
    assert route.modality_ids.unique().tolist() == [1]
    for sample in range(3):
        original_start = int(cu_lens[sample])
        original_end = int(cu_lens[sample + 1])
        selected = route.active_indices[
            (route.active_indices >= original_start)
            & (route.active_indices < original_end)
        ]
        reduced_start = int(route.active_cu_lens[sample])
        reduced_end = int(route.active_cu_lens[sample + 1])
        assert selected.numel() == reduced_end - reduced_start
        assert torch.equal(
            active[:, reduced_start:reduced_end],
            tokens.index_select(1, selected),
        )


def test_combined_configuration_resolves_against_full_custom_path():
    config = TreadLoopConfig.combined_training_preset()
    assert config.resolve_and_validate(
        path_depth=15, hidden_width=3072, attention_heads=24
    ) == (2, 11)
    assert config.looping.loop_count == 3
    assert config.looping.factorization.adaptive_halt


def test_low_level_default_is_baseline_off():
    config = TreadLoopConfig()
    assert not config.tread.enabled
    assert not config.looping.enabled
    assert not config.combined.enabled
    assert config.resolve_and_validate(
        path_depth=15, hidden_width=3072, attention_heads=24
    ) == (0, 0)


def test_zero_initialized_learned_loops_preserve_first_pass_exactly():
    adapter = LearnedFactoredLoopAdapter(
        hidden_width=8,
        attention_heads=2,
        loop_count=3,
        loop_embedding=True,
        gate_cap=0.25,
        adaptive_halt=True,
        ponder_prior=0.1,
    )
    image = torch.randn(1, 5, 8)
    text = torch.randn(1, 3, 8)
    calls = []

    def runner(block, current_image, current_text, temb):
        del block, temb
        calls.append(1)
        return current_text + 1, current_image + 1

    output_text, output_image = adapter(
        object(),
        image,
        text,
        torch.randn(1, 8),
        runner,
    )
    assert len(calls) == 3
    assert torch.equal(output_image, image + 1)
    assert torch.equal(output_text, text + 1)
    assert adapter.residual_weight.shape == (2, 2)
    assert adapter.gate_chan.shape == (2, 8)


def test_factored_gate_learns_per_head_and_channel():
    adapter = LearnedFactoredLoopAdapter(
        hidden_width=8,
        attention_heads=2,
        loop_count=2,
        loop_embedding=False,
        gate_cap=0.25,
        adaptive_halt=False,
        ponder_prior=0.1,
    )
    with torch.no_grad():
        adapter.residual_weight[0, 0] = 0.1
        adapter.gate_chan[0, 0] = 0.5
    gate = adapter._gate(0, torch.float32).flatten()
    assert gate[0] > gate[1] > 0
    assert torch.count_nonzero(gate[4:]) == 0


def test_executable_factors_skip_inactive_projection_groups():
    adapter = LearnedFactoredLoopAdapter(
        hidden_width=8,
        attention_heads=4,
        loop_count=2,
        loop_embedding=False,
        gate_cap=0.25,
        adaptive_halt=False,
        ponder_prior=0.1,
        inference_gate_threshold=1e-4,
    )
    image = torch.zeros(1, 2, 8)
    text = torch.zeros(1, 1, 8)
    factor_calls = []

    def runner(_block, current_image, current_text, _temb):
        return current_text + 1, current_image + 1

    def run_factors(
        _block,
        current_image,
        current_text,
        _temb,
        factor_count,
        active_factors,
    ):
        factor_calls.append((factor_count, active_factors))
        return current_text + 1, current_image + 1

    runner.run_factors = run_factors
    with torch.no_grad():
        adapter.residual_weight[0, 0] = 0.1
    adapter.eval()
    gate = adapter._gate(0, torch.float32).detach().abs().flatten()
    width = adapter.hidden_width // adapter.factor_count
    adapter._inference_active_factors = [
        tuple(
            factor
            for factor in range(adapter.factor_count)
            if float(gate[factor * width : (factor + 1) * width].max()) >= 1e-4
        )
    ]

    adapter(object(), image, text, torch.zeros(1, 8), runner)

    assert factor_calls == [(4, (0,))]
    assert adapter.last_executed_refinements == 1
    assert adapter.last_executed_factors == 1


def test_mageflow_loop_telemetry_matches_rwkv_dashboard_schema(tmp_path):
    class TinyTransformer:
        transformer_blocks = nn.ModuleList([nn.Identity(), nn.Identity()])
        terminal_expert_blocks = nn.ModuleList([nn.Identity()])
        inner_dim = 8
        num_attention_heads = 2

    config = TreadLoopConfig(
        tread=TreadConfig(enabled=True, route_start=0, route_end=1),
        looping=LoopingConfig(
            enabled=True,
            loop_count=3,
            factorization=FactorizationConfig(enabled=True),
        ),
        combined=CombinedConfig(enabled=True),
    )
    controller = TreadFactoredLoopController(TinyTransformer(), config)
    with torch.no_grad():
        controller.backbone_adapters[0].residual_weight[0, 0] = 0.1
        controller.backbone_adapters[0].gate_chan[0, 0] = 0.5
        for adapter in (
            *controller.backbone_adapters,
            *controller.expert_adapters,
        ):
            adapter.last_expected_loops = torch.tensor(2.25)
            adapter.last_executed_refinements = 2
    controller.last_metrics = {
        "total_image_tokens": 100,
        "active_tread_image_tokens": 50,
        "maximum_refinement_block_calls": 6,
        "executed_refinement_block_calls": 6,
        "mean_gate_rms": 0.01,
    }

    payload = mageflow_loop_telemetry_payload(
        controller,
        step=12,
        resident_domain="animation",
    )
    assert payload["model_family"] == "mageflow"
    assert payload["resident_domain"] == "animation"
    assert payload["mean_expected_loops"] == pytest.approx(2.25)
    assert payload["tread_active_fraction"] == pytest.approx(0.5)
    assert payload["gate_mode"] == "executable-factored-head-channel"
    assert [layer["label"] for layer in payload["layers"]] == ["B0", "B1", "E0"]
    assert payload["layers"][0]["split"]["heads"] == 2
    assert len(payload["layers"][0]["split"]["head_abs"]) == 2
    assert len(payload["layers"][0]["split"]["channel_abs"][0]) == 8

    artifact = tmp_path / "loop_rw.json"
    write_mageflow_loop_telemetry(
        artifact,
        controller,
        step=12,
        resident_domain="animation",
    )
    assert '"model_family":"mageflow"' in artifact.read_text()


def test_inference_skip_calibration_is_explicit_and_telemetry_is_pure():
    class TinyTransformer:
        transformer_blocks = nn.ModuleList([nn.Identity(), nn.Identity()])
        terminal_expert_blocks = nn.ModuleList([nn.Identity()])
        inner_dim = 8
        num_attention_heads = 2

    config = TreadLoopConfig(
        tread=TreadConfig(enabled=True, route_start=0, route_end=1),
        looping=LoopingConfig(
            enabled=True,
            loop_count=3,
            factorization=FactorizationConfig(
                enabled=True,
                inference_gate_threshold=1e-4,
            ),
        ),
        combined=CombinedConfig(enabled=True),
    )
    controller = TreadFactoredLoopController(TinyTransformer(), config)
    controller.refresh_inference_skip_refinements()
    adapter = controller.expert_adapters[0]
    assert adapter._inference_skip_refinements == [True, True]

    with torch.no_grad():
        adapter.residual_weight.fill_(0.01)
    mageflow_loop_telemetry_payload(
        controller,
        step=1,
        resident_domain="photo",
    )
    assert adapter._inference_skip_refinements == [True, True]

    controller.refresh_inference_skip_refinements()
    assert adapter._inference_skip_refinements == [False, False]


@pytest.mark.parametrize(
    "config,match",
    [
        (
            TreadLoopConfig(
                tread=TreadConfig(enabled=True),
                looping=LoopingConfig(
                    enabled=True,
                    factorization=FactorizationConfig(enabled=True),
                ),
                combined=CombinedConfig(enabled=True),
            ),
            "attention heads must divide",
        ),
        (
            TreadLoopConfig(
                tread=TreadConfig(enabled=True, route_end=1),
                looping=LoopingConfig(
                    enabled=True,
                    factorization=FactorizationConfig(enabled=True),
                ),
                combined=CombinedConfig(enabled=True),
            ),
            "nonempty",
        ),
        (
            TreadLoopConfig(
                tread=TreadConfig(enabled=True),
                looping=LoopingConfig(
                    enabled=True,
                    factorization=FactorizationConfig(
                        enabled=True,
                        gate_cap=0,
                    ),
                ),
                combined=CombinedConfig(enabled=True),
            ),
            "gate_cap",
        ),
    ],
)
def test_invalid_combined_configurations_are_rejected(config, match):
    with pytest.raises(ValueError, match=match):
        config.resolve_and_validate(
            path_depth=15,
            hidden_width=(
                3070 if match == "attention heads must divide" else 3072
            ),
            attention_heads=24,
        )
