import json
from pathlib import Path

import pytest
import torch
from torch import nn

from rwkv_lab.adapters import (AdapterConfig, LoRALinear, active_adapters, base_fingerprint,
                               inject_lora, load_adapter, merge_adapter, save_adapter,
                               unmerge_adapter, unload_adapter)
from rwkv_lab.distributed import fully_shard_rwkv, load_checkpoint, save_checkpoint
from rwkv_lab.export_bundle import export_bundle, verify_bundle
from rwkv_lab.posttrain_data import (IGNORE_INDEX, PostTrainingExample, dataset_manifest,
                                     cache_tokenized, load_jsonl, pack_variants, render, tokenize,
                                     TokenizedExample, TokenizedVariant, version_dataset)
from rwkv_lab.posttrain_campaign import assess, split_audit
from rwkv_lab.posttrain_train import _build_reference_cache, _train_loss
from rwkv_lab.posttrain_kernels import score_preference_pairs
from rwkv_lab.preference import (dpo_loss, kto_loss, orpo_loss, process_reward_loss,
                                 binary_calibration, OutcomeRewardHead, ProcessRewardHead, reward_model_loss,
                                 sequence_logps, simpo_loss)
from rwkv_lab.quantization import (NF4Linear, dequantize_model_nf4, qualify_linear_qlora,
                                   qualify_accelerated_nf4, quantize_model_nf4,
                                   TorchAONF4Linear)


class CharTokenizer:
    def encode(self, text):
        return [ord(char) for char in text]


def test_structured_sft_preserves_roles_and_loss_mask(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps({
        "id": "s1", "kind": "sft", "split": "train",
        "messages": [{"role": "system", "content": "be terse"},
                     {"role": "user", "content": "2+2?"},
                     {"role": "assistant", "content": "4"}],
    }) + "\n")
    rows, digest = load_jsonl(path)
    value = tokenize(render(rows[0]), CharTokenizer())
    variant = value.variants["sft"]
    text = render(rows[0]).variants["sft"].text
    user_at = text.index("2+2?")
    answer_at = text.index("4")
    assert variant.labels[user_at] == IGNORE_INDEX
    assert variant.labels[answer_at] == ord("4")
    assert variant.roles[answer_at] == "assistant"
    manifest = dataset_manifest(path)
    assert manifest["sha256"] == digest and manifest["kinds"] == {"sft": 1}
    assert manifest["template_sha256"]


def test_preference_and_feedback_validation_and_left_truncation():
    pref = PostTrainingExample.from_dict({
        "id": "p1", "kind": "preference", "messages": [{"role": "user", "content": "x" * 30}],
        "chosen": "yes", "rejected": "no",
    })
    encoded = tokenize(render(pref), CharTokenizer(), max_length=20, truncate="left")
    assert set(encoded.variants) == {"chosen", "rejected"}
    assert encoded.variants["chosen"].truncated > 0
    assert any(label != IGNORE_INDEX for label in encoded.variants["chosen"].labels)
    with pytest.raises(ValueError, match="must differ"):
        PostTrainingExample.from_dict({"kind": "preference", "prompt": "x",
                                       "chosen": "same", "rejected": "same"})
    with pytest.raises(ValueError, match="boolean"):
        PostTrainingExample.from_dict({"kind": "feedback", "prompt": "x",
                                       "response": "y", "label": "yes"})


def test_tool_calls_prm_steps_cache_and_packing_audit(tmp_path):
    rows = [
        {"id": "tool", "kind": "sft", "split": "train", "messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "function": {"name": "weather", "arguments": {"city": "Austin"}}}]},
            {"role": "tool", "name": "weather", "tool_call_id": "c1", "content": "sunny"},
            {"role": "assistant", "content": "It is sunny."}]},
        {"id": "prm", "kind": "prm", "split": "eval", "prompt": "prove it",
         "steps": [{"text": "first", "label": True}, {"text": "bad", "label": False}],
         "adversarial_steps": [{"text": "bad first", "label": False}]},
    ]
    path = tmp_path / "structured.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    parsed, _ = load_jsonl(path)
    tool_text = render(parsed[0]).variants["sft"].text
    assert "<tool_call>" in tool_text and '"city":"Austin"' in tool_text
    prm = tokenize(render(parsed[1]), CharTokenizer())
    assert prm.variants["steps"].step_labels == (True, False)
    assert len(prm.variants["steps"].step_positions) == 2
    encoded, first = cache_tokenized(path, CharTokenizer(), tmp_path / "cache", max_length=512,
                                     tokenizer_fingerprint="char-v1")
    cached, second = cache_tokenized(path, CharTokenizer(), tmp_path / "cache", max_length=512,
                                    tokenizer_fingerprint="char-v1")
    assert encoded == cached and not first["cache_hit"] and second["cache_hit"]
    packed, audit = pack_variants([(row.id, next(iter(row.variants.values()))) for row in encoded], 512,
                                  separator_id=0)
    assert packed and audit["loss_masks_preserved"] and audit["qualification_required"]


def test_prompt_response_compatibility_and_split_leak_report(tmp_path):
    path = tmp_path / "overlap.jsonl"
    path.write_text("\n".join(json.dumps({"id": f"row-{i}", "kind": "sft", "split": split,
                                           "prompt": "question", "response": "answer"})
                                    for i, split in enumerate(("train", "eval"))) + "\n")
    rows, _ = load_jsonl(path)
    assert rows[0].messages[-1].role == "assistant"
    manifest = dataset_manifest(path)
    assert manifest["duplicates"] == 1 and manifest["split_overlaps"] == 1


def test_content_addressed_dataset_versions_merge_and_reject_leakage(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text(json.dumps({"id": "a", "kind": "sft", "split": "train",
                             "prompt": "one", "response": "1"}) + "\n")
    b.write_text(json.dumps({"id": "b", "kind": "sft", "split": "eval",
                             "prompt": "two", "response": "2"}) + "\n")
    first = version_dataset([a, b], tmp_path / "versions")
    second = version_dataset([a, b], tmp_path / "versions")
    assert first["version"] == second["version"]
    assert Path(first["dataset"]).is_file() and first["examples"] == 2
    b.write_text(json.dumps({"id": "b", "kind": "sft", "split": "eval",
                             "prompt": "one", "response": "1"}) + "\n")
    with pytest.raises(ValueError, match="across train/eval/test"):
        version_dataset([a, b], tmp_path / "versions")


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 3, bias=False)

    def forward(self, x):
        return self.proj(x)


def test_named_adapter_identity_grad_merge_and_artifact_roundtrip(tmp_path):
    torch.manual_seed(3)
    model = TinyModel()
    original = {key: value.clone() for key, value in model.state_dict().items()}
    x = torch.randn(5, 4)
    baseline = model(x).detach()
    targets = inject_lora(model, AdapterConfig("trial", rank=2, alpha=4, targets=("proj",)))
    assert targets == ["proj"] and torch.equal(model(x), baseline)
    assert isinstance(model.proj, LoRALinear) and not model.proj.base.weight.requires_grad
    model(x).sum().backward()
    assert model.proj.adapters["trial"].B.grad is not None
    with torch.no_grad():
        model.proj.adapters["trial"].B.normal_(0, 0.1)
    adapted = model(x).detach()
    merge_adapter(model, "trial")
    assert torch.allclose(model(x), adapted, atol=1e-6)
    unmerge_adapter(model, "trial")
    assert torch.allclose(model(x), adapted, atol=1e-6)
    artifact = tmp_path / "adapter"
    manifest = save_adapter(model, artifact, "trial", parent_checkpoint="parent.pt")
    fresh = TinyModel()
    fresh.load_state_dict(original)
    assert base_fingerprint(fresh) == manifest["base_sha256"]
    load_adapter(fresh, artifact)
    assert torch.allclose(fresh(x), adapted, atol=1e-6)
    with active_adapters(fresh, ()):
        assert torch.equal(fresh(x), baseline)

    mixed = TinyModel().to(torch.bfloat16)
    inject_lora(mixed, AdapterConfig("fp32", rank=2, targets=("proj",)))
    assert mixed.proj.adapters["fp32"].A.dtype == torch.float32
    assert mixed(torch.randn(2, 4, dtype=torch.bfloat16)).dtype == torch.bfloat16


def test_preference_losses_have_expected_direction_and_gradients():
    pc = torch.tensor([-1.0, -2.0], requires_grad=True)
    pr = torch.tensor([-2.0, -1.0], requires_grad=True)
    rc = torch.tensor([-1.5, -1.5])
    rr = torch.tensor([-1.5, -1.5])
    dpo = dpo_loss(pc, pr, rc, rr, beta=1.0)
    assert dpo.loss[0] < dpo.loss[1] and dpo.margin.tolist() == pytest.approx([1.0, -1.0])
    dpo.loss.mean().backward()
    assert pc.grad is not None and pr.grad is not None
    simpo = simpo_loss(pc.detach(), pr.detach(), beta=1.0, gamma=0.0)
    assert simpo.loss[0] < simpo.loss[1]
    orpo = orpo_loss(pc.detach(), pr.detach(), -pc.detach(), beta=0.1)
    assert torch.isfinite(orpo.loss).all()
    kto = kto_loss(pc.detach(), rc, torch.tensor([True, False]), beta=0.1)
    assert kto.shape == pc.shape and torch.isfinite(kto).all()
    assert reward_model_loss(torch.tensor([2.0]), torch.tensor([1.0])) < reward_model_loss(
        torch.tensor([1.0]), torch.tensor([2.0]))
    prm = process_reward_loss(torch.tensor([[2.0, -2.0]]), torch.tensor([[1.0, 0.0]]),
                              torch.tensor([[True, True]]))
    assert prm < 0.2
    hidden = torch.randn(2, 5, 4)
    outcome = OutcomeRewardHead(4)(hidden, torch.tensor([[False, True, True, False, False],
                                                         [False, False, True, True, False]]))
    process = ProcessRewardHead(4)(hidden, torch.tensor([[1, 2], [2, 3]]))
    assert outcome.shape == (2,) and process.shape == (2, 2)
    calibration = binary_calibration(torch.tensor([5.0, -5.0]), torch.tensor([1.0, 0.0]))
    assert calibration["accuracy"] == 1.0 and calibration["ece"] < 0.01


def test_native_nf4_storage_gradient_and_dense_merge_parity():
    torch.manual_seed(23)
    linear = nn.Linear(64, 32)
    sample = torch.randn(3, 64)
    quantized = NF4Linear(linear)
    assert quantized.packed_weight.dtype == torch.uint8
    assert quantized.storage_bytes() < linear.weight.numel() * linear.weight.element_size()
    report = qualify_linear_qlora(linear, sample, base_tolerance=2.0)
    assert report.gradient_finite and report.gradient_nonzero
    assert report.zero_init_max_abs < 1e-6 and report.merged_max_abs < 1e-5
    assert report.compression_ratio > 1.5 and report.passed
    model = TinyModel()
    quantize_model_nf4(model, exclude=())
    inject_lora(model, AdapterConfig("q", rank=2, alpha=2, targets=("proj",)))
    with torch.no_grad():
        model.proj.adapters["q"].B.normal_(std=0.02)
    x = torch.randn(2, 4)
    expected = model(x)
    unload_adapter(model, "q", merge=True)
    dequantize_model_nf4(model)
    assert isinstance(model.proj, nn.Linear)
    assert torch.allclose(model(x), expected, atol=1e-5)


def test_torchao_nf4_matches_portable_output_and_input_gradients():
    pytest.importorskip("torchao")
    torch.manual_seed(29)
    linear = nn.Linear(64, 32, bias=False).to(torch.bfloat16)
    sample = torch.randn(4, 64, dtype=torch.bfloat16)
    accelerated = TorchAONF4Linear(linear)
    assert accelerated.storage_bytes() < linear.weight.numel() * linear.weight.element_size()
    report = qualify_accelerated_nf4(linear, sample, repeats=2, minimum_speedup=0.0)
    assert report["available"] and report["parity_passed"]
    assert report["output_max_abs"] <= report["tolerance"]
    assert report["input_gradient_max_abs"] <= report["tolerance"]
    class Wide(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(64, 32, bias=False)

        def forward(self, value):
            return self.proj(value)

    model = Wide().to(torch.bfloat16)
    quantize_model_nf4(model, exclude=(), backend="torchao")
    inject_lora(model, AdapterConfig("ao", rank=2, alpha=2, targets=("proj",)))
    value = model(torch.randn(3, 64, dtype=torch.bfloat16)).float().sum()
    value.backward()
    assert model.proj.adapters["ao"].B.grad is not None
    assert len(base_fingerprint(model)) == 64


def test_sequence_logps_respects_causal_mask():
    logits = torch.zeros(1, 4, 5)
    logits[0, 1, 3] = 5
    labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 3, IGNORE_INDEX]])
    result = sequence_logps(logits, labels)
    assert result.shape == (1,) and result.item() > -0.1


def test_batched_recurrent_preference_scoring_matches_two_forwards():
    class Scorer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(16, 5)
            self.head = nn.Linear(5, 16)

        def forward(self, ids):
            return self.head(self.embedding(ids))

    model = Scorer()
    chosen = torch.tensor([[1, 2, 3, 4], [1, 5, 6, 7]])
    rejected = torch.tensor([[1, 2, 8, 9], [1, 5, 10, 11]])
    chosen_labels, rejected_labels = chosen.clone(), rejected.clone()
    chosen_labels[:, :2] = IGNORE_INDEX
    rejected_labels[:, :2] = IGNORE_INDEX
    expected = (sequence_logps(model(chosen), chosen_labels),
                sequence_logps(model(rejected), rejected_labels))
    actual = score_preference_pairs(model, chosen, chosen_labels, rejected, rejected_labels)
    assert torch.allclose(actual[0], expected[0]) and torch.allclose(actual[1], expected[1])


def test_executable_posttraining_objective_steps_are_finite():
    class ToyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(8, 4)
            self.head = nn.Linear(4, 8)

        def forward(self, ids, return_hidden=False):
            hidden = self.emb(ids)
            logits = self.head(hidden)
            return (logits, hidden) if return_hidden else logits

    variant_a = TokenizedVariant((1, 2, 3, 4), (IGNORE_INDEX, IGNORE_INDEX, 3, 4), ("",) * 4)
    variant_b = TokenizedVariant((1, 2, 5, 6), (IGNORE_INDEX, IGNORE_INDEX, 5, 6), ("",) * 4)
    model = ToyLM()
    reward_head = OutcomeRewardHead(4)
    cases = {
        "sft": ([TokenizedExample("s", "sft", "train", {"sft": variant_a}, {})], None),
        "dpo": ([TokenizedExample("p", "preference", "train",
                                  {"chosen": variant_a, "rejected": variant_b}, {})], None),
        "orpo": ([TokenizedExample("p", "preference", "train",
                                   {"chosen": variant_a, "rejected": variant_b}, {})], None),
        "simpo": ([TokenizedExample("p", "preference", "train",
                                    {"chosen": variant_a, "rejected": variant_b}, {})], None),
        "kto": ([TokenizedExample("g", "feedback", "train", {"response": variant_a}, {"label": True}),
                 TokenizedExample("b", "feedback", "train", {"response": variant_b}, {"label": False})], None),
        "reward": ([TokenizedExample("r", "preference", "train",
                                     {"chosen": variant_a, "rejected": variant_b}, {})], reward_head),
        "prm": ([TokenizedExample("q", "prm", "train",
                    {"steps": TokenizedVariant((1, 2, 3, 4), (IGNORE_INDEX,) * 4, ("",) * 4,
                                               step_positions=(1, 3), step_labels=(True, False))}, {})],
                ProcessRewardHead(4)),
    }
    for objective, (batch, head) in cases.items():
        model.zero_grad()
        reward_head.zero_grad()
        loss, _ = _train_loss(model, head, batch, objective, "none", "cpu", 0.1, 1.0)
        assert torch.isfinite(loss), objective
        loss.backward()


def test_dpo_frozen_reference_is_scored_once_then_reused():
    class CountingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(16, 8)
            self.head = nn.Linear(8, 16)
            self.calls = 0

        def forward(self, ids):
            self.calls += 1
            return self.head(self.embedding(ids))

    chosen = TokenizedVariant((1, 2, 3, 4), (IGNORE_INDEX, IGNORE_INDEX, 3, 4), ("",) * 4)
    rejected = TokenizedVariant((1, 2, 5, 6), (IGNORE_INDEX, IGNORE_INDEX, 5, 6), ("",) * 4)
    row = TokenizedExample("p", "preference", "train",
                           {"chosen": chosen, "rejected": rejected}, {})
    model = CountingModel()
    cache = _build_reference_cache(model, [row], "dpo", "cpu", 1)
    assert model.calls == 1 and cache["p"]
    loss, _ = _train_loss(model, None, [row], "dpo", "none", "cpu", .1, 1,
                          reference_cache=cache, collect_metrics=False)
    assert torch.isfinite(loss) and model.calls == 2


def test_posttrain_campaign_split_and_promotion_gates(tmp_path):
    train = tmp_path / "train.jsonl"
    heldout = tmp_path / "eval.jsonl"
    train.write_text(json.dumps({"id": "train", "kind": "sft", "split": "train",
                                 "prompt": "train", "response": "ok"}) + "\n")
    heldout.write_text("\n".join(json.dumps({"id": f"e{i}", "kind": "sft", "split": "eval",
                                              "prompt": f"heldout {i}", "response": "ok",
                                              "metadata": {"family": "math"}}) for i in range(4)) + "\n")
    audit = split_audit(str(train), str(heldout))
    assert audit["passed"]
    initial = [{"id": f"e{i}", "loss": 1.0, "family": "math"} for i in range(4)]
    final = [{"id": f"e{i}", "loss": 0.5, "family": "math"} for i in range(4)]
    result = {"objective": "sft", "steps": 5, "train_tokens": 105,
              "initial_eval": {"per_example": initial}, "eval": {"per_example": final}}
    promotion = assess(result, audit=audit, minimum_delta=0.1, maximum_family_regression=0,
                       confidence=0.95, bootstrap_samples=200, seed=1, token_budget=100,
                       budget_slack=10)
    assert promotion["eligible"] and promotion["gates"]["equal_token_budget"]


def test_distributed_checkpoint_single_rank_exact_resume(tmp_path):
    torch.manual_seed(9)
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss = model(torch.randn(4, 3)).square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    expected = {key: value.clone() for key, value in model.state_dict().items()}
    path = tmp_path / "dcp"
    save_checkpoint(path, model, optimizer, extra={"step": 7, "rng": torch.arange(4)})
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    extra = load_checkpoint(path, model, optimizer)
    assert extra["step"] == 7 and torch.equal(extra["rng"], torch.arange(4))
    for key, value in model.state_dict().items():
        assert torch.equal(value, expected[key])


def test_fsdp2_wrap_forward_and_dcp_resume(tmp_path):
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed unavailable")
    rendezvous = tmp_path / "gloo-init"
    torch.distributed.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=0, world_size=1)
    try:
        class TinyRWKV(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
                self.head = nn.Linear(4, 2)

            def forward(self, value):
                for block in self.blocks:
                    value = torch.relu(block(value))
                return self.head(value)

        torch.manual_seed(11)
        model = fully_shard_rwkv(TinyRWKV())
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        x = torch.randn(3, 4)
        model(x).square().mean().backward()
        optimizer.step()
        optimizer.zero_grad()
        expected = model(x).detach().clone()
        checkpoint = tmp_path / "fsdp-dcp"
        save_checkpoint(checkpoint, model, optimizer, extra={"step": 3})
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        load_checkpoint(checkpoint, model, optimizer)
        assert torch.allclose(model(x), expected)
    finally:
        torch.distributed.destroy_process_group()


def test_safe_export_bundle_and_receipts(tmp_path):
    checkpoint = tmp_path / "model.pt"
    torch.save({"model": {"head.weight": torch.randn(3, 4)}, "arch": {"d_model": 4},
                "step": 12, "config": "tiny"}, checkpoint)
    data_receipt = tmp_path / "dataset.json"
    data_receipt.write_text(json.dumps({"schema": "rwkv-lab.posttrain.v1", "sha256": "abc"}))
    promotion = tmp_path / "promotion.json"
    promotion.write_text(json.dumps({"schema": "rwkv-lab.promotion.v1", "status": "passed"}))
    output = tmp_path / "export"
    result = export_bundle(checkpoint, output, dataset_manifest=data_receipt,
                           promotion_receipt=promotion)
    assert result["promotion"]["status"] == "passed"
    assert verify_bundle(output)["step"] == 12
    with (output / "model.safetensors").open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_bundle(output)
