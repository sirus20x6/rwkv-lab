from __future__ import annotations

import importlib.util
import io

import pytest
import torch

from rwkv_lab.training_components import (
    BFloat16PrecisionConfiguration,
    BFloat16PrecisionPolicy,
    Float8PrecisionConfiguration,
    Float8PrecisionPolicy,
    NVFP4PrecisionConfiguration,
    NVFP4PrecisionPolicy,
    PrecisionImplementation,
    PrecisionUnavailableError,
    build_registered_precision_policy,
)
from rwkv_lab.training_runtime import precision as precision_runtime


def _torchao_fp8_configuration() -> Float8PrecisionConfiguration:
    return Float8PrecisionConfiguration(
        backend="torchao",
        scaling_strategy="dynamic_per_tensor",
        amax_history_length=0,
        checkpoint_state_domains=("scale_factors", "master_weights"),
    )


@pytest.mark.parametrize(
    ("implementation", "configuration", "backend", "label"),
    (
        (
            PrecisionImplementation.FP8_SCALED_V1,
            Float8PrecisionConfiguration(),
            "transformer_engine",
            "FP8",
        ),
        (
            PrecisionImplementation.FP8_SCALED_V1,
            _torchao_fp8_configuration(),
            "torchao",
            "FP8",
        ),
        (
            PrecisionImplementation.NVFP4_MICROSCALING_V1,
            NVFP4PrecisionConfiguration(),
            "transformer_engine",
            "NVFP4",
        ),
    ),
)
def test_scaled_precision_selection_fails_closed_when_backend_is_absent(
    implementation, configuration, backend, label
):
    assert importlib.util.find_spec(backend) is None
    with pytest.raises(
        PrecisionUnavailableError,
        match=rf"^{label} precision requires backend package '{backend}', but it is not installed$",
    ):
        build_registered_precision_policy(implementation, configuration)


def test_failed_fp8_selection_never_returns_a_bf16_policy():
    with pytest.raises(PrecisionUnavailableError):
        build_registered_precision_policy(
            PrecisionImplementation.FP8_SCALED_V1,
            Float8PrecisionConfiguration(),
        )


def test_nvfp4_selection_fails_closed_when_cuda_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        precision_runtime.importlib.util, "find_spec", lambda _: object()
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(
        PrecisionUnavailableError,
        match=(
            r"^NVFP4 precision requires CUDA, but "
            r"torch\.cuda\.is_available\(\) is false$"
        ),
    ):
        build_registered_precision_policy(
            PrecisionImplementation.NVFP4_MICROSCALING_V1,
            NVFP4PrecisionConfiguration(),
        )


@pytest.mark.parametrize(
    ("implementation", "configuration", "label"),
    (
        (
            PrecisionImplementation.FP8_SCALED_V1,
            Float8PrecisionConfiguration(),
            "FP8",
        ),
        (
            PrecisionImplementation.NVFP4_MICROSCALING_V1,
            NVFP4PrecisionConfiguration(),
            "NVFP4",
        ),
    ),
)
def test_scaled_precision_rejects_unqualified_cuda_capability(
    monkeypatch, implementation, configuration, label
):
    monkeypatch.setattr(
        precision_runtime.importlib.util, "find_spec", lambda _: object()
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (9, 0))

    with pytest.raises(
        PrecisionUnavailableError,
        match=(
            rf"^{label} precision has insufficient CUDA capability sm_90; "
            r"qualified capabilities: sm_120$"
        ),
    ):
        build_registered_precision_policy(implementation, configuration)


@pytest.mark.parametrize(
    "factory",
    (
        lambda: Float8PrecisionConfiguration(scaling_strategy="per_block"),
        lambda: Float8PrecisionConfiguration(amax_history_length=0),
        lambda: Float8PrecisionConfiguration(
            retain_fp32_master_weights=False,
        ),
        lambda: NVFP4PrecisionConfiguration(scaling_strategy="per_tensor"),
        lambda: NVFP4PrecisionConfiguration(block_size=32),
        lambda: NVFP4PrecisionConfiguration(
            retain_fp32_master_weights=False,
        ),
    ),
)
def test_scaled_precision_configuration_rejects_invalid_declarations(factory):
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_scaled_precision_configuration_declares_an_excludable_projection_surface():
    configuration = Float8PrecisionConfiguration(
        eligible_projection_patterns=("blocks.*.key.weight",),
        excluded_parameter_patterns=("blocks.0.key.weight", "*.lm_head.weight"),
    )
    assert configuration.eligible_projection_patterns == ("blocks.*.key.weight",)
    assert "*.lm_head.weight" in configuration.excluded_parameter_patterns

    with pytest.raises(ValueError, match="must not overlap"):
        Float8PrecisionConfiguration(
            eligible_projection_patterns=("*.weight",),
            excluded_parameter_patterns=("*.weight",),
        )


@pytest.mark.parametrize(
    ("policy", "checkpoint_state"),
    (
        (
            Float8PrecisionPolicy(Float8PrecisionConfiguration(amax_history_length=4)),
            {
                "scale_factors": {"blocks.0.key": torch.tensor(0.125)},
                "amax_history": {"blocks.0.key": torch.tensor([2.0, 1.5, 1.0, 0.5])},
                "master_weights": {
                    "blocks.0.key.weight": torch.arange(6.0).reshape(2, 3)
                },
            },
        ),
        (
            NVFP4PrecisionPolicy(NVFP4PrecisionConfiguration()),
            {
                "scale_factors": {"blocks.0.value.block.0": torch.tensor(0.0625)},
                "master_weights": {
                    "blocks.0.value.weight": torch.arange(8.0).reshape(2, 4)
                },
            },
        ),
    ),
)
def test_scaled_precision_state_round_trips_through_torch_checkpoint(
    policy, checkpoint_state
):
    policy.load_state_dict(checkpoint_state)
    checkpoint = io.BytesIO()
    torch.save({"precision_policy": policy.state_dict()}, checkpoint)
    checkpoint.seek(0)
    restored_blob = torch.load(checkpoint, weights_only=True)

    resumed = type(policy)(policy.configuration)
    resumed.load_state_dict(restored_blob["precision_policy"])
    restored = resumed.state_dict()
    assert set(restored) == set(policy.configuration.checkpoint_state_domains)
    for domain, values in checkpoint_state.items():
        assert set(restored[domain]) == set(values)
        for name, expected in values.items():
            torch.testing.assert_close(restored[domain][name], expected)


@pytest.mark.parametrize(
    ("policy", "lost_state"),
    (
        (
            Float8PrecisionPolicy(Float8PrecisionConfiguration()),
            {"amax_history": {}, "master_weights": {}},
        ),
        (
            NVFP4PrecisionPolicy(NVFP4PrecisionConfiguration()),
            {"scale_factors": {}, "master_weights": {}},
        ),
    ),
)
def test_scaled_precision_rejects_checkpoint_that_loses_scale_state(policy, lost_state):
    with pytest.raises(ValueError, match="scale"):
        policy.load_state_dict(lost_state)


def test_bfloat16_state_contract_remains_empty():
    policy = build_registered_precision_policy(
        PrecisionImplementation.BF16_PARAMETERS_FP32_REDUCTIONS_V1,
        BFloat16PrecisionConfiguration(),
    )
    assert isinstance(policy, BFloat16PrecisionPolicy)
    assert policy.state_dict() == {}
    with pytest.raises(ValueError, match="must be empty"):
        policy.load_state_dict({"scale_factors": {}})


@pytest.mark.gpu
def test_sm120_is_qualified_without_importing_scaled_backends():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is not exposed to this test process")
    precision_runtime._require_cuda_capability("FP8")
    precision_runtime._require_cuda_capability("NVFP4")
