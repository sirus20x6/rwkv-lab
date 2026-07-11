from __future__ import annotations

import pytest
import torch

from rwkv_lab.sm120_kernels import (
    L2PersistenceController,
    cutlass_sm120_status,
    occupancy_plan,
    persistent_rwkv7_state_step,
    sm120_profile,
)


def test_sm120_helpers_fail_closed_without_cuda():
    profile = sm120_profile("cpu")
    assert not profile.available
    assert occupancy_plan(512, "cpu")["resident_programs"] == 0
    status = cutlass_sm120_status()
    assert isinstance(status["available"], bool)
    assert "source" in status


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_sm120_persistent_state_matches_reference():
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 required")
    from rwkv_lab.megakernel import rwkv7_recurrent_step

    shape = (1, 1, 8, 32)
    tensors = [torch.randn(shape, device="cuda", dtype=torch.bfloat16)
               for _ in range(6)]
    tensors[1] = -tensors[1].float().abs().to(torch.bfloat16)
    state = torch.randn((1, 8, 32, 32), device="cuda", dtype=torch.float32)
    expected, expected_state = rwkv7_recurrent_step(*tensors, state, inplace=False)
    actual_state = state.clone()
    actual = persistent_rwkv7_state_step(actual_state, *tensors)
    torch.testing.assert_close(actual, expected, atol=5e-3, rtol=3e-3)
    torch.testing.assert_close(actual_state, expected_state, atol=5e-3, rtol=3e-3)
    plan = occupancy_plan(512)
    assert plan["resident_programs"] == min(512, plan["sms"])


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_sm120_l2_window_round_trip():
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 required")
    tensor = torch.empty(1 << 20, device="cuda", dtype=torch.uint8)
    controller = L2PersistenceController("cuda")
    receipt = controller.apply(tensor, name="test")
    if not receipt.get("available"):
        pytest.skip(receipt["reason"])
    assert receipt["adopted"]
    assert 0 < receipt["window_bytes"] <= tensor.numel()
    controller.clear()
    assert not controller.active
