from __future__ import annotations

import json
import pytest
import torch

from rwkv_lab.megakernel import (MegakernelBackend, StateCodec,
                                 _set_fused_state_step,
                                 adopt_megakernel_artifact,
                                 adopt_megakernel_receipt, file_sha256,
                                 finalize_megakernel_serving_embedding,
                                 megakernel_incompatibility,
                                 prefill_artifact_path,
                                 rwkv7_recurrent_step,
                                 rwkv7_recurrent_step_epilogue,
                                 rwkv7_time_mix_epilogue, triton_status)
from rwkv_lab.generate import sample_with_stats
from rwkv_lab.rwkv8_deltanet import _rwkv7_python_ref
from rwkv_lab.rwkv_pretrain import RWKV7Small

pytestmark = pytest.mark.gpu


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


def test_serving_embedding_finalization_reclaims_duplicate_and_preserves_logits():
    torch.manual_seed(70)
    model = RWKV7Small(32, 8, 1, 4, {}).eval()
    ids = torch.tensor([[2, 3, 4]])
    expected = model(ids)
    model._megakernel_adopted = True
    receipt = finalize_megakernel_serving_embedding(model)
    assert receipt["reclaimed_duplicate_bytes"] == model.emb.weight.numel() * 4
    assert not hasattr(model, "_megakernel_folded_embedding")
    torch.testing.assert_close(model(ids), expected, atol=2e-5, rtol=2e-5)


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
    original_state = state.clone()
    expected, expected_state = _rwkv7_python_ref(
        r, gk, key, value, remove_a, remove_b, initial_state=state)
    actual, actual_state = rwkv7_recurrent_step(
        r, gk, key, value, remove_a, remove_b, state)
    assert actual_state.data_ptr() != state.data_ptr()
    torch.testing.assert_close(state, original_state)
    inplace_state = state.clone()
    with torch.no_grad():
        inplace, inplace_result = rwkv7_recurrent_step(
            r, gk, key, value, remove_a, remove_b, inplace_state, inplace=True)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(actual_state, expected_state, atol=2e-5, rtol=2e-5)
    assert inplace_result.data_ptr() == inplace_state.data_ptr()
    torch.testing.assert_close(inplace, expected, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(inplace_result, expected_state, atol=2e-5, rtol=2e-5)


@pytest.mark.skipif(not triton_status()[0], reason=triton_status()[1])
def test_fused_time_mix_epilogue_matches_groupnorm_bonus_gate():
    torch.manual_seed(72)
    batch, heads, width = 2, 4, 16
    channels = heads * width
    wkv, receptance, key, value, gate = [
        torch.randn(batch, 1, heads, width, device="cuda", dtype=torch.bfloat16)
        for _ in range(5)
    ]
    r_k = torch.randn(heads, width, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(channels, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(channels, device="cuda", dtype=torch.bfloat16)
    normalized = torch.nn.functional.group_norm(
        wkv.view(batch, channels), heads, weight, bias, 6.4e-4,
    ).view_as(wkv)
    bonus = ((receptance * key * r_k.view(1, 1, heads, width))
             .sum(-1, keepdim=True) * value)
    expected = (normalized + bonus) * gate
    actual = rwkv7_time_mix_epilogue(
        wkv, receptance, key, value, r_k, gate, weight, bias, eps=6.4e-4)
    torch.testing.assert_close(actual, expected, atol=7e-2, rtol=3e-2)


@pytest.mark.skipif(not triton_status()[0], reason=triton_status()[1])
def test_combined_state_epilogue_matches_two_stage_oracle():
    torch.manual_seed(75)
    batch, heads, width = 2, 3, 16
    r, gk, key, value, remove_a, remove_b, gate = [
        torch.randn(batch, 1, heads, width, device="cuda", dtype=torch.bfloat16)
        for _ in range(7)
    ]
    gk = -gk.float().abs().to(torch.bfloat16)
    state = torch.randn(batch, heads, width, width, device="cuda", dtype=torch.float32)
    r_k = torch.randn(heads, width, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(heads * width, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(heads * width, device="cuda", dtype=torch.bfloat16)
    wkv, expected_state = rwkv7_recurrent_step(
        r, gk, key, value, remove_a, remove_b, state)
    expected = rwkv7_time_mix_epilogue(
        wkv, r, key, value, r_k, gate, weight, bias, eps=6.4e-4)
    actual, actual_state = rwkv7_recurrent_step_epilogue(
        r, gk, key, value, remove_a, remove_b, state, r_k, gate,
        weight, bias, eps=6.4e-4)
    torch.testing.assert_close(actual, expected, atol=7e-2, rtol=3e-2)
    torch.testing.assert_close(actual_state, expected_state, atol=2e-5, rtol=2e-5)


@pytest.mark.skipif(not triton_status()[0], reason=triton_status()[1])
def test_albatross_folded_embedding_preserves_recurrent_model():
    torch.manual_seed(74)
    model = RWKV7Small(
        64, 32, 2, 8, {}, deepembed=True, de_dim=4,
        de_mode="hidden", de_shift=True, de_emb_res=True,
    ).to("cuda", torch.bfloat16).eval()
    prompt = torch.tensor([[2, 3, 4]], device="cuda")
    MegakernelBackend(model, device="cuda", compile_mode="default")
    with torch.no_grad():
        expected, expected_state = model.forward_recurrent(prompt)
        _set_fused_state_step(model, True, inplace=False)
        try:
            actual, actual_state = model.forward_recurrent(prompt)
        finally:
            _set_fused_state_step(model, False)
    torch.testing.assert_close(actual, expected, atol=8e-2, rtol=3e-2)
    expected_codec, expected_leaves = StateCodec.from_state(expected_state)
    actual_leaves = expected_codec.flatten(actual_state)
    for got, want in zip(actual_leaves, expected_leaves):
        torch.testing.assert_close(got, want, atol=8e-2, rtol=3e-2)


@pytest.mark.skipif(not triton_status()[0], reason=triton_status()[1])
def test_cuda_graph_plan_is_token_exact_across_replays(monkeypatch, tmp_path):
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
    model._megakernel_backend = backend
    backend.prefill(prompt)
    actual = [backend.step(token).clone() for token in tokens]
    torch.cuda.synchronize()
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, atol=8e-2, rtol=3e-2)
    receipt = backend.receipt()
    assert receipt["backend"] == "triton+inductor+cudagraph"
    assert len(receipt["plan_sha256"]) == 64
    assert receipt["cached_prefill_plans"] == 1

    reference, _ = sample_with_stats(
        model, [2, 3, 4], max_new=6, temperature=0, stop_at_sep=False,
        device="cuda", engine="recurrent")
    generated, stats = sample_with_stats(
        model, [2, 3, 4], max_new=6, temperature=0, stop_at_sep=False,
        device="cuda", engine="megakernel")
    assert generated == reference and stats["device_side_greedy"]

    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"artifact checkpoint")
    artifact = tmp_path / "decode.pt2"
    backend.prefill(prompt, greedy_feedback=True)
    manifest = backend.plan.export_aot(
        artifact, checkpoint_sha256=file_sha256(checkpoint))
    prefill_artifact = prefill_artifact_path(artifact, 1, prompt.shape[1])
    prefill_manifest = backend.prefill_plans[
        (1, prompt.shape[1], str(prompt.dtype))
    ].export_aot(prefill_artifact, checkpoint_sha256=file_sha256(checkpoint))
    restored = RWKV7Small(64, 16, 1, 8, {}).to("cuda", torch.bfloat16).eval()
    restored.load_state_dict(model.state_dict())
    adopt_megakernel_artifact(restored, artifact, checkpoint)
    loaded = MegakernelBackend(restored, device="cuda", compile_mode="default")
    assert torch.equal(loaded.generate_greedy(prompt, max_new=6),
                       backend.generate_greedy(prompt, max_new=6))
    assert loaded.receipt()["aot_loaded"]
    assert loaded.receipt()["prefill_aot_loaded"] == 1
    assert manifest["artifact_sha256"] == file_sha256(artifact)
    assert prefill_manifest["artifact_sha256"] == file_sha256(prefill_artifact)
