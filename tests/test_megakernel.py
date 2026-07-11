from __future__ import annotations

import json
import pytest
import torch

from rwkv_lab.megakernel import (MegakernelBackend, StateCodec,
                                 adopt_megakernel_receipt, file_sha256,
                                 megakernel_incompatibility,
                                 rwkv7_recurrent_step, triton_status)
from rwkv_lab.rwkv8_deltanet import _rwkv7_python_ref
from rwkv_lab.rwkv_pretrain import RWKV7Small


def test_state_codec_round_trips_tensor_only_recurrent_abi():
    state = [{"wkv": torch.ones(1, 2, 4, 4),
              "att_shift": torch.zeros(1, 1, 8), "optional": None}]
    codec, leaves = StateCodec.from_state(state)
    rebuilt = codec.rebuild(leaves)
    assert codec.leaf_count == 2
    assert torch.equal(rebuilt[0]["wkv"], state[0]["wkv"])
    assert rebuilt[0]["optional"] is None
    with pytest.raises(TypeError, match="tensors"):
        StateCodec.from_state({"changing_step": 3})


def test_megakernel_fails_closed_without_cuda():
    model = RWKV7Small(32, 8, 1, 4, {})
    assert "requires CUDA" in megakernel_incompatibility(model, "cpu")


@pytest.mark.skipif(not triton_status()[0], reason=triton_status()[1])
def test_persisted_adoption_is_bound_to_checkpoint_and_runtime(tmp_path):
    checkpoint = tmp_path / "ckpt.pt"
    checkpoint.write_bytes(b"checkpoint identity")
    report = {
        "schema": "rwkv-lab.megakernel-qualification.v1", "adopted": True,
        "checkpoint_sha256": file_sha256(checkpoint),
        "environment": {
            "compute_capability": list(torch.cuda.get_device_capability("cuda")),
            "torch": torch.__version__,
            "triton": triton_status()[1].removeprefix("Triton "),
        },
    }
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(report))
    model = RWKV7Small(32, 8, 1, 4, {}).to("cuda", torch.bfloat16)
    adopt_megakernel_receipt(model, receipt, checkpoint)
    assert model._megakernel_adopted
    checkpoint.write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="does not match"):
        adopt_megakernel_receipt(model, receipt, checkpoint)


@pytest.mark.skipif(not triton_status()[0], reason=triton_status()[1])
def test_fused_rwkv_transition_matches_fp32_oracle():
    torch.manual_seed(71)
    batch, heads, width = 2, 3, 8
    r, gk, key, value, remove_a, remove_b = [
        torch.randn(batch, 1, heads, width, device="cuda", dtype=torch.bfloat16)
        for _ in range(6)
    ]
    gk = -gk.float().abs().to(torch.bfloat16)
    state = torch.randn(batch, heads, width, width, device="cuda", dtype=torch.float32)
    expected, expected_state = _rwkv7_python_ref(
        r, gk, key, value, remove_a, remove_b, initial_state=state)
    actual, actual_state = rwkv7_recurrent_step(
        r, gk, key, value, remove_a, remove_b, state)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(actual_state, expected_state, atol=2e-5, rtol=2e-5)


@pytest.mark.skipif(not triton_status()[0], reason=triton_status()[1])
def test_cuda_graph_plan_is_token_exact_across_replays(monkeypatch):
    monkeypatch.setenv("RWKV8_FORCE_PYREF", "1")
    torch.manual_seed(73)
    model = RWKV7Small(64, 16, 1, 8, {}).to("cuda", torch.bfloat16).eval()
    prompt = torch.tensor([[2, 3, 4]], device="cuda")
    tokens = [torch.tensor([[5]], device="cuda"), torch.tensor([[6]], device="cuda")]
    with torch.no_grad():
        _, state = model.forward_recurrent(prompt)
        expected = []
        for token in tokens:
            logits, state = model.forward_recurrent(token, state)
            expected.append(logits)
    backend = MegakernelBackend(model, device="cuda", compile_mode="default")
    backend.prefill(prompt)
    actual = [backend.step(token).clone() for token in tokens]
    torch.cuda.synchronize()
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, atol=8e-2, rtol=3e-2)
    receipt = backend.receipt()
    assert receipt["backend"] == "triton+inductor+cudagraph"
    assert len(receipt["plan_sha256"]) == 64
