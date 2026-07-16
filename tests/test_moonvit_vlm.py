import json
import os
import stat
import struct
import threading
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from rwkv_lab.moonvit import (MoonViT, MoonViTPrefixProjector, _Block, _resize,
                              _resize_geometry, feature_cache_key,
                              valid_pooled_feature,
                              valid_torch_archive_storages)
from rwkv_lab.vision_cache import cache_entry_valid
from rwkv_lab.vision_train import (EpochBatchSampler, _BoundedFeatureCache,
                                   _FEATURE_MEMORY_CACHE,
                                   _acquire_run_lock,
                                   _archive_fresh_run_artifacts,
                                   _durable_replace,
                                   _final_checkpoint_required,
                                   _initialize_adapters, _load_raw_tensor_archive,
                                   _image_file_identity,
                                   _best_checkpoint_path,
                                   _budget_resume_differences,
                                   _fail_nonterminal_status,
                                   _invalidate_step_evaluation,
                                   _loop_lr_resume_difference,
                                   _loop_runtime_scale,
                                   _pending_eval_work,
                                   _preserve_loop_reset_outcome,
                                   _publish_eval_contract_reset,
                                   _publish_eval_due,
                                   _quarantine_best,
                                   _quarantine_future_best, _require_finite_metric,
                                   _resume_contract_changed,
                                   _resume_checkpoint_publication_required,
                                   _resume_invalidates_step_evaluation,
                                   _resume_requires_best_quarantine,
                                   _resumed_last_checkpoint_step,
                                   _trim_log,
                                   _trainer_run_artifact_paths,
                                   _promote_checkpoint,
                                   cached_features, load_examples, preload_feature_cache,
                                   image_metadata_fingerprint,
                                   filter_eval_sample_indices, prepare_examples,
                                   select_eval_sample_indices,
                                   split_examples, supervised_positions,
                                   write_eval_samples)
from rwkv_lab.generate import SEP, WorldVocab
from rwkv_lab.deep_vision import DeepVisionInjector
from rwkv_lab.fused_ce import (logits_cross_entropy, masked_token_mean,
                               weighted_logits_cross_entropy)
from rwkv_lab.vision_grounding import ImageTextContrastiveHead, early_token_weights
from rwkv_lab.vision_caption import checkpoint_runtime_scales


def test_projector_has_fixed_prefix_and_gradients():
    projector = MoonViTPrefixProjector(rwkv_hidden=32, prefix_tokens=5)
    out = projector([torch.randn(7, 4, 1152), torch.randn(11, 4, 1152)])
    assert out.shape == (2, 5, 32)
    out.square().mean().backward()
    assert projector.project[0].weight.grad is not None


def test_learned_resampler_is_exact_noop_then_receives_gradients():
    torch.manual_seed(4)
    baseline = MoonViTPrefixProjector(rwkv_hidden=32, prefix_tokens=5)
    enhanced = MoonViTPrefixProjector(
        rwkv_hidden=32, prefix_tokens=5, resampler_layers=2,
        resampler_width=16, resampler_heads=4)
    info = enhanced.load_state_dict(baseline.state_dict(), strict=False)
    assert info.missing_keys and all(
        key.startswith("resampler.") for key in info.missing_keys)
    features = [torch.randn(5, 4, 1152), torch.randn(5, 4, 1152)]
    expected = baseline(features)
    actual = enhanced(features)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().mean().backward()
    assert enhanced.resampler.output_projection.weight.grad is not None
    assert enhanced.resampler.output_projection.weight.grad.abs().sum() > 0


def test_deep_vision_is_exact_noop_and_reinjection_backpropagates():
    class Layer(torch.nn.Module):
        def forward(self, hidden_states, **_kwargs):
            return hidden_states + 0.25

    layers = torch.nn.ModuleList([Layer() for _ in range(4)])
    injector = DeepVisionInjector(8, [1, 3], rank=4)
    injector.install(layers)
    prefix = torch.randn(2, 3, 8, requires_grad=True)
    hidden = torch.randn(2, 7, 8)

    def run(value):
        for layer in layers:
            value = layer(value)
        return value

    expected = run(hidden)
    with injector.use_prefix(prefix):
        actual = run(hidden)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().mean().backward()
    for adapter in injector.adapters.values():
        assert adapter.up.weight.grad is not None
        assert adapter.up.weight.grad.abs().sum() > 0
    injector.close()


def test_grounding_objectives_weight_opening_and_use_batch_negatives():
    labels = torch.tensor([
        [-100, -100, 4, 5, 6],
        [-100, 7, 8, -100, -100],
    ])
    batches = torch.tensor([0, 0, 0, 1, 1])
    causal_positions = torch.tensor([1, 2, 3, 0, 1])
    weights = early_token_weights(
        labels, batches, causal_positions, token_count=2, weight=3.0)
    torch.testing.assert_close(weights, torch.tensor([3., 3., 1., 3., 3.]))

    head = ImageTextContrastiveHead(8, width=4, temperature=0.2)
    prefix = torch.randn(3, 5, 8, requires_grad=True)
    targets = torch.randn(9, 8)
    target_batches = torch.arange(3).repeat_interleave(3)
    loss, accuracy = head(prefix, targets, target_batches)
    assert loss.isfinite() and 0 <= accuracy <= 1
    loss.backward()
    assert prefix.grad is not None and prefix.grad.abs().sum() > 0


def test_selected_logit_ce_portable_fallback_matches_pytorch():
    logits = torch.randn(7, 19, requires_grad=True)
    labels = torch.randint(0, 19, (7,))
    expected = torch.nn.functional.cross_entropy(logits.float(), labels)
    actual = logits_cross_entropy(logits, labels, fused=False)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert logits.grad is not None


def test_weighted_selected_logit_ce_matches_manual_reduction():
    logits = torch.randn(5, 13, requires_grad=True)
    labels = torch.randint(0, 13, (5,))
    weights = torch.tensor([3., 3., 1., 1., 1.])
    actual, raw = weighted_logits_cross_entropy(
        logits, labels, weights, fused=False)
    losses = torch.nn.functional.cross_entropy(
        logits.float(), labels, reduction="none")
    torch.testing.assert_close(actual, (losses * weights).sum() / weights.sum())
    torch.testing.assert_close(raw, losses.mean())


def test_masked_token_mean_matches_cross_entropy_ignore_semantics():
    torch.manual_seed(0)
    logits = torch.randn(6, 11)
    labels = torch.tensor([1, -100, 3, -100, 5, 6])
    # reduction="none" zeroes ignored rows, exactly like the flash CE kernel,
    # so this exercises the fused path's denominator math on CPU.
    losses = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    expected = torch.nn.functional.cross_entropy(logits, labels)
    torch.testing.assert_close(masked_token_mean(losses, labels), expected)
    all_ignored = torch.full((6,), -100)
    assert torch.isnan(masked_token_mean(torch.zeros(6), all_ignored))
    assert torch.isnan(torch.nn.functional.cross_entropy(logits, all_ignored))


def test_weighted_ce_denominators_exclude_ignored_positions():
    torch.manual_seed(1)
    logits = torch.randn(5, 7)
    labels = torch.tensor([1, 2, -100, 4, 5])
    weights = torch.tensor([3., 1., 100., 1., 1.])
    actual, raw = weighted_logits_cross_entropy(
        logits, labels, weights, fused=False)
    losses = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    keep = labels != -100
    torch.testing.assert_close(
        actual, (losses[keep] * weights[keep]).sum() / weights[keep].sum())
    torch.testing.assert_close(raw, losses[keep].mean())
    unweighted, unweighted_raw = weighted_logits_cross_entropy(
        logits, labels, None, fused=False)
    torch.testing.assert_close(
        unweighted, torch.nn.functional.cross_entropy(logits, labels))
    torch.testing.assert_close(unweighted_raw, losses[keep].mean())


def test_feature_memory_cache_evicts_oldest_and_refreshes_on_hit():
    entry = lambda: torch.zeros(4, dtype=torch.float32)  # 16 bytes each
    cache = _BoundedFeatureCache(max_bytes=48)
    a, b, c, d = (Path(f"/cache/{name}.pt") for name in "abcd")
    cache[a], cache[b], cache[c] = entry(), entry(), entry()
    assert len(cache) == 3 and cache.total_bytes == 48
    assert cache.get(a) is not None  # hit refreshes recency: b is now oldest
    cache[d] = entry()
    assert b not in cache
    assert a in cache and c in cache and d in cache
    assert cache.total_bytes == 48

    tuples = _BoundedFeatureCache(max_bytes=32)
    tuples[a] = (entry(), entry())  # fusion-style tuple entry: 32 bytes
    tuples[b] = entry()
    assert a not in tuples and b in tuples
    assert tuples.total_bytes == 16

    unbounded = _BoundedFeatureCache(max_bytes=0)
    for key in (a, b, c, d):
        unbounded[key] = entry()
    assert len(unbounded) == 4

    existing = entry()
    lru = _BoundedFeatureCache(max_bytes=48)
    lru[a], lru[b], lru[c] = existing, entry(), entry()
    assert lru.setdefault(a, entry()) is existing  # setdefault is also a hit
    lru[d] = entry()
    assert a in lru and b not in lru


def test_loop_runtime_scale_starts_small_and_saturates():
    assert _loop_runtime_scale(249, start_step=250, ramp_steps=1000) == 0.0
    assert _loop_runtime_scale(250, start_step=250, ramp_steps=1000) == 0.001
    assert _loop_runtime_scale(1249, start_step=250, ramp_steps=1000) == 1.0
    assert _loop_runtime_scale(250, start_step=250, ramp_steps=0) == 1.0


def test_nonfinite_metrics_are_rejected_before_logging_or_checkpointing():
    assert _require_finite_metric("loss", torch.tensor(1.25)) == 1.25
    for value in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(FloatingPointError, match="non-finite loss"):
            _require_finite_metric("loss", value)


def test_eval_obligations_resume_only_the_missing_phase(tmp_path: Path):
    log = tmp_path / "train.jsonl"
    log.write_text("\n".join((
        json.dumps({"kind": "train", "step": 100}),
        json.dumps({"kind": "eval_due", "step": 100}),
    )) + "\n")
    assert _pending_eval_work(log, 100) == ("loss", None)
    with log.open("a") as handle:
        handle.write(json.dumps({
            "kind": "eval", "step": 100, "loss": 1.0, "ppl": 4.0,
            "sample_artifact": str(tmp_path / "step_00000100.json"),
        }) + "\n")
    phase, prior = _pending_eval_work(log, 100)
    assert phase == "captions" and prior["ppl"] == 4.0
    with log.open("a") as handle:
        handle.write(json.dumps({"kind": "eval_artifact", "step": 100}) + "\n")
    assert _pending_eval_work(log, 100) is None
    assert _pending_eval_work(log, 99) is None


def test_scheduled_checkpoint_implies_eval_if_due_record_was_not_published(
        tmp_path: Path):
    log = tmp_path / "train.jsonl"
    assert _pending_eval_work(log, 100, eval_expected=True) == ("loss", None)
    log.write_text(json.dumps({"kind": "train", "step": 100}) + "\n")
    assert _pending_eval_work(log, 100, eval_expected=True) == ("loss", None)
    log.write_text(json.dumps({
        "kind": "eval", "step": 100, "loss": 1.0, "ppl": 2.0,
        "sample_artifact": None, "qualitative_complete": True,
    }) + "\n")
    assert _pending_eval_work(log, 100, eval_expected=True) is None


def test_same_step_eval_artifact_cannot_clear_mutated_resume_obligation(
        tmp_path: Path):
    log = tmp_path / "train.jsonl"
    log.write_text("\n".join((
        json.dumps({"kind": "eval", "step": 99, "ppl": 3.0}),
        json.dumps({"kind": "train", "step": 100}),
        json.dumps({"kind": "checkpoint", "step": 100,
                    "reason": "best_eval_promoted"}),
        json.dumps({"kind": "eval", "step": 100, "ppl": 2.0,
                    "sample_artifact": None, "qualitative_complete": True}),
        json.dumps({"kind": "eval_artifact", "step": 100}),
    )) + "\n")
    assert _pending_eval_work(log, 100, eval_expected=True) is None

    assert _resume_invalidates_step_evaluation(
        text_limit_migrated=True, unrelated_branch=False,
        loop_reset_pending=False)
    assert _resume_invalidates_step_evaluation(
        text_limit_migrated=False, unrelated_branch=True,
        loop_reset_pending=False)
    assert _resume_invalidates_step_evaluation(
        text_limit_migrated=False, unrelated_branch=False,
        loop_reset_pending=True)
    assert not _resume_invalidates_step_evaluation(
        text_limit_migrated=False, unrelated_branch=False,
        loop_reset_pending=False)
    assert _invalidate_step_evaluation(log, 100)
    assert _pending_eval_work(log, 100, eval_expected=True) == ("loss", None)
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert {record["kind"] for record in records} == {"eval", "train"}
    assert any(record.get("step") == 99 for record in records)


def test_eval_publication_orders_train_checkpoint_and_obligation(
        tmp_path: Path, monkeypatch):
    events = []

    class Log:
        def write(self, value):
            events.append(json.loads(value)["kind"])

    monkeypatch.setattr(
        "rwkv_lab.vision_train._sync_log", lambda handle: events.append("sync"))
    _publish_eval_due(
        Log(), step=100, checkpoint_path=tmp_path / "last.pt",
        train_record={"kind": "train", "step": 100},
        save_checkpoint=lambda: events.append("save"),
    )
    assert events == [
        "train", "sync", "save", "checkpoint", "eval_due", "sync",
    ]


def test_legacy_complete_eval_artifact_does_not_regenerate(tmp_path: Path):
    artifact = tmp_path / "step_00000100.json"
    artifact.write_text(json.dumps({
        "step": 100, "ppl": 2.7, "items": [{"caption": "done"}],
    }))
    log = tmp_path / "train.jsonl"
    scalar = {
        "kind": "eval", "step": 100, "loss": 1.0, "ppl": 2.7,
        "sample_artifact": str(artifact),
    }
    log.write_text(json.dumps(scalar) + "\n")
    assert _pending_eval_work(log, 100) is None
    artifact.write_text(json.dumps({
        "step": 100, "ppl": 2.7, "complete": False, "items": [],
    }))
    assert _pending_eval_work(log, 100) == ("captions", scalar)


def test_log_trim_preserves_malformed_lines_byte_for_byte(tmp_path: Path):
    log = tmp_path / "train.jsonl"
    malformed = "{not json"
    unstepped = json.dumps({"kind": "train", "step": None})
    kept_record = json.dumps({"kind": "train", "step": 10})
    log.write_text("\n".join((
        kept_record,
        malformed,
        unstepped,
        json.dumps({"kind": "train", "step": 12}),
    )) + "\n")
    _trim_log(log, 10)
    # Matches _invalidate_step_evaluation: only records provably newer than
    # the checkpoint are dropped; everything else survives byte-identically.
    assert log.read_text().splitlines() == [kept_record, malformed, unstepped]


def test_nonterminal_exit_guard_preserves_terminal_states(tmp_path: Path):
    status = tmp_path / "status.json"
    status.write_text(json.dumps({"state": "loading_rwkv", "step": 4}))
    assert _fail_nonterminal_status(status, reason="test", error="boom")
    failed = json.loads(status.read_text())
    assert failed["state"] == "failed" and failed["previous_state"] == "loading_rwkv"
    assert failed["error"] == "boom"
    status.write_text(json.dumps({"state": "paused", "step": 5}))
    assert not _fail_nonterminal_status(status, reason="test")
    assert json.loads(status.read_text())["state"] == "paused"


def test_run_lock_rejects_overlapping_trainers(tmp_path: Path):
    first = _acquire_run_lock(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="already owns run"):
            _acquire_run_lock(tmp_path)
    finally:
        first.close()
    replacement = _acquire_run_lock(tmp_path)
    replacement.close()


def test_future_branch_best_is_preserved_but_no_longer_advertised(tmp_path: Path):
    best = tmp_path / "best"
    best.mkdir()
    (best / "best.json").write_text(json.dumps({
        "step": 200, "ppl": 3.0, "checkpoint": "ckpt.pt",
    }))
    (best / "ckpt.pt").write_bytes(b"checkpoint")

    quarantined = _quarantine_future_best(best, checkpoint_step=100)

    assert quarantined is not None and quarantined.is_dir()
    assert not best.exists()
    assert json.loads((quarantined / "best.json").read_text())["step"] == 200


def test_current_or_older_best_remains_advertised(tmp_path: Path):
    best = tmp_path / "best"
    best.mkdir()
    (best / "best.json").write_text(json.dumps({
        "step": 100, "ppl": 3.0, "checkpoint": "ckpt.pt",
    }))
    (best / "ckpt.pt").write_bytes(b"checkpoint")

    assert _quarantine_future_best(best, checkpoint_step=100) is None
    assert best.is_dir()


def test_unrelated_explicit_resume_quarantines_even_an_older_active_best(
        tmp_path: Path):
    last = tmp_path / "last.pt"
    external = tmp_path / "external-step-100.pt"
    best = tmp_path / "best"
    best.mkdir()
    winner = best / "ckpt_step_00000050.pt"
    last.write_bytes(b"current branch")
    external.write_bytes(b"unrelated branch at a later step")
    winner.write_bytes(b"winner from current branch")
    (best / "best.json").write_text(json.dumps({
        "step": 50, "ppl": 2.0, "checkpoint": winner.name,
    }))

    assert _resume_requires_best_quarantine(external, last, winner)
    quarantined = _quarantine_best(best, "before-explicit-resume-step-100")
    assert quarantined is not None and quarantined.is_dir()
    assert not best.exists()


def test_explicit_resume_of_advertised_best_preserves_that_best(tmp_path: Path):
    last = tmp_path / "last.pt"
    winner = tmp_path / "winner.pt"
    last.write_bytes(b"later model")
    winner.write_bytes(b"best model")
    assert not _resume_requires_best_quarantine(winner, last, winner)


def test_intentionally_changed_resume_contract_quarantines_current_best(tmp_path: Path):
    best = tmp_path / "best"
    best.mkdir()
    (best / "best.json").write_text(json.dumps({"step": 100, "ppl": 3.0}))

    quarantined = _quarantine_best(best, "before-loop-reset-step-100")

    assert quarantined == tmp_path / "best.before-loop-reset-step-100"
    assert quarantined.is_dir() and not best.exists()


def test_eval_contract_reset_receipt_covers_missing_best_and_fresh_start(
        tmp_path: Path):
    receipt = tmp_path / "eval_contract_reset.json"
    _publish_eval_contract_reset(
        receipt, step=100,
        reasons=("text_limit_migration", "loop_reset", "loop_reset"))
    assert json.loads(receipt.read_text()) == {
        "schema": 1,
        "reset": True,
        "step": 100,
        "reasons": ["text_limit_migration", "loop_reset"],
    }

    _publish_eval_contract_reset(
        receipt, step=0, reasons=("fresh",))
    assert json.loads(receipt.read_text()) == {
        "schema": 1,
        "reset": True,
        "step": 0,
        "reasons": ["fresh"],
    }


def test_fresh_start_archives_all_active_trainer_artifacts_but_not_locks(
        tmp_path: Path):
    files = (
        "train.jsonl", "last.pt", "pre_loop.pt", "loop_rw.json",
        "config.json", "status.json", "eval_contract_reset.json",
        "operator_step_000001.json.gz", "operator_step_000001.txt",
        "overnight_caption_smoke.json", "overnight_caption_smoke.json.tmp",
        "overnight_inference.log", "overnight_caption_smoke.failed-123",
    )
    for name in files:
        (tmp_path / name).write_text(name)
    for name in ("best", "eval_samples"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "owned").write_text(name)
    for name in (".trainer.lock", ".launcher.lock", "watchdog.log", "notes.txt"):
        (tmp_path / name).write_text(name)

    active = {path.name for path in _trainer_run_artifact_paths(tmp_path)}
    assert "status.json" in active and "best" in active
    assert ".trainer.lock" not in active and ".launcher.lock" not in active

    archive = _archive_fresh_run_artifacts(tmp_path, stamp="fixed")

    assert archive == tmp_path / ".fresh-backup-fixed"
    assert archive is not None
    assert {path.name for path in archive.iterdir()} == {
        *files, "best", "eval_samples",
    }
    assert not any((tmp_path / name).exists() for name in files)
    assert not (tmp_path / "best").exists()
    assert not (tmp_path / "eval_samples").exists()
    for name in (".trainer.lock", ".launcher.lock", "watchdog.log", "notes.txt"):
        assert (tmp_path / name).read_text() == name

    (tmp_path / "last.pt").write_text("next generation")
    second = _archive_fresh_run_artifacts(tmp_path, stamp="fixed")
    assert second == tmp_path / ".fresh-backup-fixed.1"
    assert (second / "last.pt").read_text() == "next generation"


def test_caption_runtime_matches_checkpoint_ramps():
    args = {
        "loop_count": 2, "loop_start_step": 250, "loop_ramp_steps": 1000,
        "engram_warmup_steps": 1000,
    }
    assert checkpoint_runtime_scales(args, 249) == (False, 0.0, 0.249)
    enabled, loop_scale, engram_scale = checkpoint_runtime_scales(args, 250)
    assert enabled and loop_scale == 0.001 and engram_scale == 0.25
    assert checkpoint_runtime_scales(args, 1249) == (True, 1.0, 1.0)


def test_manifest_loader_requires_a_real_image_and_caption(tmp_path: Path):
    image = tmp_path / "x.jpg"
    Image.new("RGB", (4, 4)).save(image)
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in [
        {"image": "x.jpg", "text": "caption"}, {"image": "missing.jpg", "text": "no"}, {"image": "x.jpg", "text": ""},
    ]))
    rows = load_examples(path, root=tmp_path)
    assert len(rows) == 1 and rows[0]["image"] == image


def test_image_metadata_fingerprint_detects_replaced_training_input(tmp_path: Path):
    image = tmp_path / "x.jpg"
    Image.new("RGB", (4, 4)).save(image)
    manifest = tmp_path / "rows.jsonl"
    manifest.write_text(json.dumps({"image": "x.jpg", "text": "caption"}) + "\n")
    before = image_metadata_fingerprint(load_examples(manifest, root=tmp_path))
    Image.new("RGB", (8, 8)).save(image)
    after = image_metadata_fingerprint(load_examples(manifest, root=tmp_path))
    assert before != after


def test_image_split_identity_collapses_hard_links(tmp_path: Path):
    original = tmp_path / "original.jpg"
    alias = tmp_path / "alias.jpg"
    other = tmp_path / "other.jpg"
    Image.new("RGB", (4, 4), "red").save(original)
    os.link(original, alias)
    Image.new("RGB", (4, 4), "blue").save(other)
    manifest = tmp_path / "rows.jsonl"
    manifest.write_text("\n".join(json.dumps({"image": path.name, "text": path.name})
                                  for path in (original, alias, other)))
    rows = load_examples(manifest, root=tmp_path)

    assert _image_file_identity(rows[0]) == _image_file_identity(rows[1])
    train, val = split_examples(rows, val_fraction=0.1)
    assert ({0, 1} <= set(train)) != ({0, 1} <= set(val))


def test_caption_tokens_are_bounded_and_end_in_eod():
    vocab = WorldVocab()
    rows = [{"image": Path("x"), "text": "word " * 100}]
    prepared, lengths = prepare_examples(rows, vocab, prompt="Describe:\n", max_text_tokens=12)
    assert lengths == [12]
    assert prepared[0]["tokens"][-1] == SEP
    assert prepared[0]["truncated"]


def test_row_prompt_override_keeps_tag_task_distinct():
    vocab = WorldVocab()
    rows = [
        {"image": Path("a"), "text": "A red fox."},
        {"image": Path("b"), "text": "red_fur, fox", "prompt": "List visible tags:\n"},
    ]
    prepared, _ = prepare_examples(rows, vocab, prompt="Describe this image:\n",
                                   max_text_tokens=64)
    assert prepared[0]["prompt"] == "Describe this image:\n"
    assert prepared[1]["prompt"] == "List visible tags:\n"
    assert prepared[0]["tokens"][:prepared[0]["prompt_len"]] != \
        prepared[1]["tokens"][:prepared[1]["prompt_len"]]


def test_supervised_positions_align_caption_targets_after_prefix():
    rows = [
        {"tokens": [10, 11, 12, 13], "prompt_len": 2},
        {"tokens": [20, 21, 22], "prompt_len": 1},
    ]
    actual = supervised_positions(rows, 5, device="cpu")
    assert actual.tolist() == [[0, 6], [0, 7], [1, 5], [1, 6]]


def test_content_split_is_stable_under_reordering():
    rows = [{"image": Path(f"/{i}.jpg"), "text": f"caption {i}"} for i in range(100)]
    train, val = split_examples(rows, val_fraction=.1)
    reversed_rows = list(reversed(rows))
    train2, val2 = split_examples(reversed_rows, val_fraction=.1)
    assert {rows[i]["text"] for i in train} == {reversed_rows[i]["text"] for i in train2}
    assert {rows[i]["text"] for i in val} == {reversed_rows[i]["text"] for i in val2}


def test_content_split_keeps_all_captions_for_an_image_together():
    rows = [
        {"image": Path(f"/{image}.jpg"), "text": f"caption {variant}"}
        for image in range(20) for variant in range(3)
    ]
    train, val = split_examples(rows, val_fraction=.1)
    train_images = {rows[index]["image"] for index in train}
    val_images = {rows[index]["image"] for index in val}
    assert train_images.isdisjoint(val_images)
    assert train_images | val_images == {row["image"] for row in rows}


def test_qualitative_eval_samples_are_source_stratified():
    rows = [
        {"stage1_source": source, "image": Path(f"/{source}-{i}.jpg"), "text": "x"}
        for source in ("joy", "midjourney", "pexels") for i in range(20)
    ]
    chosen = select_eval_sample_indices(rows, list(range(len(rows))), 8)
    counts = {}
    for index in chosen:
        source = rows[index]["stage1_source"]
        counts[source] = counts.get(source, 0) + 1
    assert len(chosen) == len(set(chosen)) == 8
    assert set(counts) == {"joy", "midjourney", "pexels"}
    assert max(counts.values()) - min(counts.values()) <= 1


def test_qualitative_eval_source_filter_does_not_change_scalar_indices():
    rows = [
        {"stage1_source": source, "image": Path(f"/{i}.jpg"), "text": "x"}
        for i, source in enumerate((
            "eval_i1_pexels", "eval_i1_midjourneyv6", "eval_joy_caption",
            "civitai", "NSFW_manga"))
    ]
    scalar_indices = list(range(len(rows)))
    chosen = filter_eval_sample_indices(
        rows, scalar_indices, ["joy", "civitai", "nsfw", "porn", "manga"])
    assert chosen == [0, 1]
    assert scalar_indices == list(range(len(rows)))


def test_qualitative_eval_persists_partial_artifact_on_interrupt(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    rows = [{"image": tmp_path / "heldout.jpg", "text": "reference",
             "stage1_source": "heldout"}]

    def interrupted_features(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("rwkv_lab.vision_train.cached_features", interrupted_features)
    with pytest.raises(KeyboardInterrupt):
        write_eval_samples(
            rows, [0], step=100, ppl=4.5, rwkv=None, projector=None,
            vision=None, engram=None, cache_dir=None, vocab=WorldVocab(),
            prompt="Describe this image:\n", out=tmp_path, count=1, max_new=16)

    artifact = json.loads(
        (tmp_path / "eval_samples" / "step_00000100.json").read_text())
    assert artifact["complete"] is False
    assert artifact["generation_steps"] == 0
    assert artifact["items"][0]["reference"] == "reference"
    assert artifact["items"][0]["caption"] == ""


def test_epoch_sampler_resumes_exactly_without_replacement():
    original = EpochBatchSampler(range(19), list(range(19)), batch_size=4, seed=7)
    first = original.next_batch()
    state = original.state_dict()
    expected = original.next_batch()
    recovered = EpochBatchSampler(range(19), list(range(19)), batch_size=4, seed=999)
    recovered.load_state_dict(state)
    assert recovered.next_batch() == expected
    all_rows = first + expected
    while recovered.epoch == 0:
        all_rows += recovered.next_batch()
    assert len(set(all_rows[:19])) == 19


def test_token_budget_sampler_uses_more_short_rows_without_replacement():
    lengths = [10] * 12 + [100] * 12
    sampler = EpochBatchSampler(range(24), lengths, batch_size=2, seed=3, bucket_batches=32)
    seen, sizes = [], []
    while len(seen) < 24:
        batch = sampler.next_budget_batch(lengths, target_tokens=60, min_items=2, max_items=6)
        seen.extend(batch)
        sizes.append(len(batch))
    assert sorted(seen) == list(range(24))
    assert max(sizes) > min(sizes)


def test_token_budget_uses_true_max_when_crossing_a_bucket_boundary():
    costs = [100, 100, 10, 10, 10, 10]
    sampler = EpochBatchSampler(range(6), costs, batch_size=2, seed=1)
    sampler.order = list(range(6))
    sampler.position = 0
    batch = sampler.next_budget_batch(costs, target_tokens=250, min_items=2, max_items=6)
    assert batch == [0, 1]
    assert len(batch) * max(costs[i] for i in batch) <= 250


def test_budget_peek_matches_next_without_advancing_checkpoint_state():
    costs = [12, 15, 18, 40, 44, 48]
    sampler = EpochBatchSampler(range(6), costs, batch_size=2, seed=4)
    state = sampler.state_dict()
    peeked = sampler.peek_budget_batch(
        costs, target_tokens=100, min_items=2, max_items=5)
    assert sampler.position == state["position"]
    assert torch.equal(sampler.generator.get_state(), state["generator_state"])
    actual = sampler.next_budget_batch(
        costs, target_tokens=100, min_items=2, max_items=5)
    assert actual == peeked


def test_budget_batch_is_consumed_only_when_committed():
    costs = [12, 15, 18, 40, 44, 48]
    sampler = EpochBatchSampler(range(6), costs, batch_size=2, seed=4)
    first = sampler.peek_budget_batch(
        costs, target_tokens=100, min_items=2, max_items=5)
    assert sampler.peek_budget_batch(
        costs, target_tokens=100, min_items=2, max_items=5) == first
    assert sampler.position == 0
    second = sampler.peek_budget_batch(
        costs, target_tokens=100, min_items=2, max_items=5,
        position_offset=len(first))
    sampler.commit_batch(first)
    assert sampler.position == len(first)
    assert sampler.peek_budget_batch(
        costs, target_tokens=100, min_items=2, max_items=5) == second
    with pytest.raises(ValueError, match="not the sampler's current prefix"):
        sampler.commit_batch(first)


def test_fixed_token_budget_uses_configured_item_count_not_base_batch():
    costs = [10] * 24
    sampler = EpochBatchSampler(range(24), costs, batch_size=2, seed=4)
    batch = sampler.peek_budget_batch(
        costs, target_tokens=1_000, min_items=6, max_items=6)
    assert len(batch) == 6


def test_resume_only_marks_last_checkpoint_current_for_same_file(tmp_path: Path):
    last = tmp_path / "last.pt"
    external = tmp_path / "older.pt"
    alias = tmp_path / "last-alias.pt"
    last.write_bytes(b"newer branch")
    external.write_bytes(b"older branch")
    os.link(last, alias)

    # Auto-resume loads last.pt itself. An explicit hard-link alias represents
    # the same durable checkpoint, while a distinct external resume must force
    # a final save even if its step lands exactly on the checkpoint cadence.
    assert _resumed_last_checkpoint_step(last, last, 200) == 200
    assert _resumed_last_checkpoint_step(alias, last, 200) == 200
    external_step = _resumed_last_checkpoint_step(external, last, 100)
    assert external_step is None
    assert _resume_checkpoint_publication_required(external, external_step)
    assert _final_checkpoint_required(100, external_step)
    auto_step = _resumed_last_checkpoint_step(last, last, 200)
    assert not _resume_checkpoint_publication_required(last, auto_step)
    assert not _final_checkpoint_required(200, auto_step)
    assert not _resume_checkpoint_publication_required(None, None)
    assert _resumed_last_checkpoint_step(None, last, 0) is None


def test_text_limit_migration_forces_same_file_checkpoint_publication(
        tmp_path: Path):
    last = tmp_path / "last.pt"
    last.write_bytes(b"old text contract")
    changed = _resume_contract_changed(
        text_limit_migrated=True, budget_differences=[])
    known_step = _resumed_last_checkpoint_step(
        last, last, 100, contract_changed=changed)

    assert known_step is None
    assert _resume_checkpoint_publication_required(last, known_step)
    assert _final_checkpoint_required(100, known_step)


def test_allowed_base_batch_resize_forces_publication_then_becomes_exact(
        tmp_path: Path):
    last = tmp_path / "last.pt"
    last.write_bytes(b"batch-8 contract")
    saved = {
        "batch": 8, "min_batch": 4, "max_batch": 32,
        "target_batch_tokens": 4096, "loop_token_budget_scale": 0.5,
    }
    resized = SimpleNamespace(
        batch=16, min_batch=4, max_batch=32,
        target_batch_tokens=4096, loop_token_budget_scale=0.5)
    differences = _budget_resume_differences(saved, resized)
    assert differences == ["batch"]
    changed = _resume_contract_changed(
        text_limit_migrated=False, budget_differences=differences)
    known_step = _resumed_last_checkpoint_step(
        last, last, 100, contract_changed=changed)
    assert known_step is None
    assert _resume_checkpoint_publication_required(last, known_step)

    # save_last_checkpoint serializes vars(args), so the republished contract
    # compares cleanly on the following auto-resume without an override flag.
    republished = {name: getattr(resized, name) for name in saved}
    assert _budget_resume_differences(republished, resized) == []


def test_sampler_state_remains_valid_when_base_batch_size_changes():
    original = EpochBatchSampler(range(32), list(range(32)), batch_size=8, seed=9)
    original.next_batch()
    state = original.state_dict()
    resized = EpochBatchSampler(range(32), list(range(32)), batch_size=16, seed=1)
    resized.load_state_dict(state)
    assert resized.order == state["order"] and resized.position == state["position"]
    assert len(resized.peek_budget_batch(
        list(range(32)), target_tokens=0, min_items=16, max_items=16)) == 16


def test_committed_loop_reset_cannot_waive_another_loop_lr_change():
    requested = SimpleNamespace(reset_loop_on_resume=True, loop_lr=2e-5)
    already_committed = {"loop_reset_committed": True, "loop_lr": 1e-5}
    reset_pending = {"loop_reset_committed": False, "loop_lr": 1e-5}

    # _load_checkpoint appends loop_lr to incompatible settings in the first
    # case. Only a reset that will actually execute may replace the group LR.
    assert _loop_lr_resume_difference(already_committed, requested)
    assert not _loop_lr_resume_difference(reset_pending, requested)
    assert not _loop_lr_resume_difference(
        already_committed,
        SimpleNamespace(reset_loop_on_resume=True, loop_lr=1e-5))


def test_committed_loop_reset_receipt_survives_descendant_checkpoints():
    args = SimpleNamespace(reset_loop_on_resume=False)
    _preserve_loop_reset_outcome(args, committed=True)
    assert vars(args)["loop_reset_committed"] is True

    descendant = SimpleNamespace(reset_loop_on_resume=True)
    _preserve_loop_reset_outcome(descendant, committed=True)
    assert vars(descendant)["loop_reset_committed"] is True


def test_recovery_trims_only_uncheckpointed_log_records(tmp_path: Path):
    log = tmp_path / "train.jsonl"
    log.write_text("\n".join(json.dumps({"kind": "train", "step": i}) for i in range(1, 6)) + "\n")
    _trim_log(log, 3)
    assert [json.loads(line)["step"] for line in log.read_text().splitlines()] == [1, 2, 3]


def test_log_trim_syncs_rewritten_payload_before_parent_directory(
        tmp_path: Path, monkeypatch):
    log = tmp_path / "train.jsonl"
    log.write_text("\n".join(
        json.dumps({"kind": "train", "step": step}) for step in range(1, 5)
    ) + "\n")
    synced = []
    real_fsync = os.fsync

    def record_fsync(fd):
        mode = os.fstat(fd).st_mode
        synced.append("directory" if stat.S_ISDIR(mode) else "file")
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", record_fsync)
    _trim_log(log, 2)
    assert synced == ["file", "directory"]
    assert [json.loads(line)["step"] for line in log.read_text().splitlines()] == [1, 2]


def test_best_quarantine_syncs_parent_after_rename(tmp_path: Path, monkeypatch):
    best = tmp_path / "best"
    best.mkdir()
    candidate = tmp_path / "best.audit"
    events = []

    def record_directory_sync(path):
        events.append((path, best.exists(), candidate.exists()))

    monkeypatch.setattr(
        "rwkv_lab.vision_train._fsync_directory", record_directory_sync)
    assert _quarantine_best(best, "audit") == candidate
    assert events == [(tmp_path, False, True)]


def test_durable_replace_syncs_payload_before_directory_entry(tmp_path: Path,
                                                              monkeypatch):
    temporary = tmp_path / "new.tmp"
    target = tmp_path / "target.pt"
    temporary.write_bytes(b"new checkpoint")
    synced = []
    real_fsync = os.fsync

    def record_fsync(fd):
        mode = os.fstat(fd).st_mode
        synced.append("directory" if stat.S_ISDIR(mode) else "file")
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", record_fsync)
    _durable_replace(temporary, target)
    assert target.read_bytes() == b"new checkpoint"
    assert synced == ["file", "directory"]


def test_durable_replace_preserves_old_target_when_payload_sync_fails(
        tmp_path: Path, monkeypatch):
    temporary = tmp_path / "new.tmp"
    target = tmp_path / "target.pt"
    temporary.write_bytes(b"new checkpoint")
    target.write_bytes(b"old checkpoint")

    def fail_sync(_fd):
        raise OSError("disk")

    monkeypatch.setattr(os, "fsync", fail_sync)
    with pytest.raises(OSError, match="disk"):
        _durable_replace(temporary, target)
    assert target.read_bytes() == b"old checkpoint"


def test_best_checkpoint_promotion_preserves_exact_saved_inode(tmp_path: Path):
    latest = tmp_path / "last.pt"
    latest.write_bytes(b"checkpoint-at-eval")
    _promote_checkpoint(latest, tmp_path / "best", step=1200, loss=2.0)
    best = tmp_path / "best" / "ckpt.pt"
    info = json.loads((tmp_path / "best" / "best.json").read_text())
    immutable = tmp_path / "best" / info["checkpoint"]
    assert best.read_bytes() == b"checkpoint-at-eval"
    assert best.stat().st_ino == latest.stat().st_ino
    assert immutable.stat().st_ino == latest.stat().st_ino
    assert _best_checkpoint_path(tmp_path / "best") == immutable
    assert info["step"] == 1200 and info["loss"] == 2.0


def test_best_manifest_switch_cleans_old_immutable_winner(tmp_path: Path):
    best_dir = tmp_path / "best"
    first = tmp_path / "last.pt"
    first.write_bytes(b"first")
    _promote_checkpoint(first, best_dir, step=100, loss=2.0)
    old_target = _best_checkpoint_path(best_dir)
    assert old_target is not None

    second = tmp_path / "last2.pt"
    second.write_bytes(b"second")
    _promote_checkpoint(second, best_dir, step=200, loss=1.5)

    new_target = _best_checkpoint_path(best_dir)
    assert new_target is not None and new_target.read_bytes() == b"second"
    assert not old_target.exists()
    assert (best_dir / "ckpt.pt").read_bytes() == b"second"


def test_legacy_best_checkpoint_remains_resolvable(tmp_path: Path):
    best = tmp_path / "best"
    best.mkdir()
    legacy = best / "ckpt.pt"
    legacy.write_bytes(b"legacy")
    assert _best_checkpoint_path(best) == legacy


@pytest.mark.parametrize("manifest", (
    "{malformed", json.dumps({}), json.dumps([]), json.dumps({"step": 10}),
    json.dumps({"step": -1, "ppl": 3.0}),
    json.dumps({"step": 10.5, "ppl": 3.0}),
    json.dumps({"step": 10, "loss": -0.1}),
    json.dumps({"step": 10, "ppl": 0.0}),
    json.dumps({"step": 10, "loss": float("inf")}),
    json.dumps({"step": 10, "ppl": 3.0, "checkpoint": None}),
    json.dumps({"checkpoint": "ckpt.pt"}),
    json.dumps({"step": 10, "ppl": 3.0, "checkpoint": "../ckpt.pt"}),
    json.dumps({"step": 10, "ppl": 3.0, "checkpoint": "missing.pt"}),
))
def test_present_invalid_best_manifest_never_falls_back_to_legacy_alias(
        tmp_path: Path, manifest: str):
    best = tmp_path / "best"
    best.mkdir()
    (best / "ckpt.pt").write_bytes(b"potentially unrelated legacy winner")
    (best / "best.json").write_text(manifest)
    assert _best_checkpoint_path(best) is None


def test_best_manifest_rejects_non_checkpoint_extension(tmp_path: Path):
    best = tmp_path / "best"
    best.mkdir()
    payload = best / "winner.bin"
    payload.write_bytes(b"not a torch checkpoint")
    (best / "best.json").write_text(json.dumps({
        "step": 10, "ppl": 3.0, "checkpoint": payload.name,
    }))
    assert _best_checkpoint_path(best) is None


def test_live_legacy_best_manifest_shape_resolves_hardlinked_alias(
        tmp_path: Path):
    best = tmp_path / "best"
    best.mkdir()
    source = tmp_path / "last-at-17600.pt"
    source.write_bytes(b"phase2 winner")
    legacy = best / "ckpt.pt"
    os.link(source, legacy)
    (best / "best.json").write_text(json.dumps({
        "step": 17600, "loss": 0.9245, "ppl": 2.5206,
    }))

    assert _best_checkpoint_path(best) == legacy
    assert legacy.stat().st_ino == source.stat().st_ino


def test_new_best_manifest_rejects_symlink_that_escapes_best_directory(
        tmp_path: Path):
    best = tmp_path / "best"
    best.mkdir()
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"not an immutable winner")
    (best / "winner.pt").symlink_to(outside)
    (best / "best.json").write_text(json.dumps({
        "step": 10, "loss": 1.0, "ppl": 2.718,
        "checkpoint": "winner.pt",
    }))
    assert _best_checkpoint_path(best) is None


def test_legacy_best_manifest_rejects_symlink_alias(tmp_path: Path):
    best = tmp_path / "best"
    best.mkdir()
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"not a legacy hardlink")
    (best / "ckpt.pt").symlink_to(outside)
    (best / "best.json").write_text(json.dumps({
        "step": 17600, "loss": 0.9245, "ppl": 2.5206,
    }))
    assert _best_checkpoint_path(best) is None


def _adapter_init_fixture(tmp_path: Path):
    source_projector = torch.nn.Linear(3, 4)
    source_nextlat = torch.nn.Linear(4, 4)
    source_engram = torch.nn.Linear(4, 2)
    args = SimpleNamespace(
        rwkv_fingerprint="rwkv", moonvit_fingerprint="moonvit", prefix_tokens=64,
        max_input_patches=1024, nextlat_hidden=1024, loop_count=2,
        loop_index=True, loop_gate_cap=0.25,
        engram=True, engram_sites="3,15", engram_drow=128, engram_rows=65536,
        engram_boundary_id=0,
    )
    checkpoint = tmp_path / "source.pt"
    torch.save({
        "schema": 3, "step": 17600, "args": vars(args),
        "projector": source_projector.state_dict(),
        "nextlat": source_nextlat.state_dict(),
        "engram": source_engram.state_dict(), "loops": [],
    }, checkpoint)
    return checkpoint, args, source_projector, source_nextlat, source_engram


def test_phase_init_preserves_existing_engram(tmp_path: Path):
    checkpoint, args, source_projector, source_nextlat, source_engram = \
        _adapter_init_fixture(tmp_path)
    projector = torch.nn.Linear(3, 4)
    nextlat = torch.nn.Linear(4, 4)
    engram = torch.nn.Linear(4, 2)

    step = _initialize_adapters(
        checkpoint, projector=projector, nextlat=nextlat, engram=engram,
        wrappers=[], args=args)

    assert step == 17600
    for destination, source in (
        (projector, source_projector), (nextlat, source_nextlat),
        (engram, source_engram),
    ):
        assert all(torch.equal(a, b) for a, b in
                   zip(destination.state_dict().values(), source.state_dict().values()))


def test_phase_init_refuses_to_silently_discard_engram(tmp_path: Path):
    checkpoint, args, _, _, _ = _adapter_init_fixture(tmp_path)
    with pytest.raises(ValueError, match="discard the source Engram"):
        _initialize_adapters(
            checkpoint, projector=torch.nn.Linear(3, 4),
            nextlat=torch.nn.Linear(4, 4), engram=None, wrappers=[], args=args)


def test_phase_init_can_add_zero_init_resampler_to_legacy_projector(tmp_path: Path):
    source = MoonViTPrefixProjector(32, 5)
    checkpoint = tmp_path / "legacy.pt"
    saved_args = {
        "rwkv_fingerprint": "rwkv", "moonvit_fingerprint": "moonvit",
        "prefix_tokens": 5, "max_input_patches": 1024,
        "nextlat_hidden": 1024, "loop_count": 1, "loop_index": True,
        "loop_gate_cap": 0.25, "engram": False,
    }
    torch.save({
        "schema": 3, "step": 50, "args": saved_args,
        "projector": source.state_dict(), "nextlat": None,
        "engram": None, "loops": [],
    }, checkpoint)
    args = SimpleNamespace(
        **saved_args, vision_resampler_layers=1,
        vision_resampler_width=16, vision_resampler_heads=4)
    destination = MoonViTPrefixProjector(
        32, 5, resampler_layers=1, resampler_width=16,
        resampler_heads=4)
    assert _initialize_adapters(
        checkpoint, projector=destination, nextlat=None, engram=None,
        wrappers=[], args=args) == 50
    features = [torch.randn(5, 4, 1152)]
    torch.testing.assert_close(destination(features), source(features), rtol=0, atol=0)


def test_pooled_feature_cache_avoids_a_second_vision_call(tmp_path: Path):
    class Vision:
        max_input_patches = 1024
        cache_fingerprint = "test-weights"
        patch_embed = type("Patch", (), {"proj": type("Proj", (), {"weight": torch.empty(1)})()})()
        def __init__(self):
            self.calls = 0

        def encode_many(self, images):
            self.calls += 1
            return [torch.ones(70, 4, 1152) for _ in images]
    image = tmp_path / "x.jpg"
    Image.new("RGB", (4, 4)).save(image)
    projector, vision = MoonViTPrefixProjector(32, 5), Vision()
    rows = [{"image": image}]
    first = cached_features(rows, vision, projector, tmp_path / "cache")
    second = cached_features(rows, vision, projector, tmp_path / "cache")
    assert vision.calls == 1 and first[0].shape == (5, 4, 1152) and torch.equal(first[0], second[0])
    assert (tmp_path / "cache" / feature_cache_key(
        image, max_input_patches=1024, prefix_tokens=5,
        vision_fingerprint="test-weights")).exists()


def test_incompatible_pooled_feature_cache_is_regenerated(tmp_path: Path):
    class Vision:
        max_input_patches = 1024
        cache_fingerprint = "bad-cache-test"
        patch_embed = type(
            "Patch", (), {"proj": type("Proj", (), {"weight": torch.empty(1)})()})()
        def __init__(self):
            self.calls = 0

        def encode_many(self, images):
            self.calls += 1
            return [torch.ones(70, 4, 1152) for _ in images]
    image = tmp_path / "x.jpg"
    Image.new("RGB", (4, 4)).save(image)
    cache = tmp_path / "cache"
    cache.mkdir()
    projector, vision = MoonViTPrefixProjector(32, 5), Vision()
    key = cache / feature_cache_key(
        image, max_input_patches=1024, prefix_tokens=5,
        vision_fingerprint="bad-cache-test")
    torch.save(torch.ones(1), key)
    actual = cached_features([{"image": image}], vision, projector, cache)[0]
    assert vision.calls == 1 and actual.shape == (5, 4, 1152)
    assert torch.load(key, weights_only=True).shape == (5, 4, 1152)


def test_nonfinite_pooled_feature_cache_is_regenerated(tmp_path: Path):
    class Vision:
        max_input_patches = 1024
        cache_fingerprint = "nonfinite-cache-test"
        patch_embed = type(
            "Patch", (), {"proj": type("Proj", (), {"weight": torch.empty(1)})()})()

        def __init__(self):
            self.calls = 0

        def encode_many(self, images):
            self.calls += 1
            return [torch.ones(70, 4, 1152) for _ in images]

    image = tmp_path / "x.jpg"
    Image.new("RGB", (4, 4)).save(image)
    cache = tmp_path / "cache"
    cache.mkdir()
    projector, vision = MoonViTPrefixProjector(32, 5), Vision()
    key = cache / feature_cache_key(
        image, max_input_patches=1024, prefix_tokens=5,
        vision_fingerprint="nonfinite-cache-test")
    poisoned = torch.ones(5, 4, 1152)
    poisoned[0, 0, 0] = float("nan")
    torch.save(poisoned, key)

    actual = cached_features([{"image": image}], vision, projector, cache)[0]

    assert vision.calls == 1 and torch.isfinite(actual).all()
    assert torch.isfinite(torch.load(key, weights_only=True)).all()


def test_cache_prefill_rejects_corrupt_or_structurally_wrong_entries(tmp_path: Path):
    path = tmp_path / "feature.pt"
    path.write_bytes(b"not a torch archive")
    assert not cache_entry_valid(path, 5)
    torch.save(torch.ones(1), path)
    assert not cache_entry_valid(path, 5)
    poisoned = torch.ones(5, 4, 1152, dtype=torch.bfloat16)
    poisoned[0, 0, 0] = float("inf")
    torch.save(poisoned, path)
    assert not cache_entry_valid(path, 5)
    torch.save(torch.ones(5, 4, 1152, dtype=torch.bfloat16), path)
    assert cache_entry_valid(path, 5)
    assert valid_pooled_feature(torch.ones(5, 4, 1152), 5)


def _flip_first_torch_storage_byte(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        member = next(info for info in archive.infolist()
                      if info.filename.endswith("/data/0"))
    with path.open("r+b") as handle:
        handle.seek(member.header_offset)
        header = handle.read(30)
        name_length, extra_length = struct.unpack_from("<HH", header, 26)
        payload_offset = member.header_offset + 30 + name_length + extra_length
        handle.seek(payload_offset)
        original = handle.read(1)
        handle.seek(payload_offset)
        handle.write(bytes([original[0] ^ 0x01]))


def test_cache_crc_rejects_a_finite_payload_bit_flip(tmp_path: Path):
    path = tmp_path / "feature.pt"
    expected = torch.ones(5, 4, 1152, dtype=torch.bfloat16)
    torch.save(expected, path)
    _flip_first_torch_storage_byte(path)

    # PyTorch itself accepts the altered finite payload, which is why the
    # archive checksum must be checked separately.
    silently_altered = torch.load(path, map_location="cpu", weights_only=True)
    assert torch.isfinite(silently_altered).all()
    assert not torch.equal(silently_altered, expected)
    assert not cache_entry_valid(path, 5)
    with pytest.raises(zipfile.BadZipFile):
        _load_raw_tensor_archive(
            path, shape=tuple(expected.shape), stride=tuple(expected.stride()),
            storage_offset=expected.storage_offset(), dtype=expected.dtype,
            storage_bytes=expected.untyped_storage().nbytes())


def test_checkpoint_crc_uses_loaded_storages_without_a_second_payload_read(
        tmp_path: Path):
    path = tmp_path / "checkpoint.pt"
    shared = torch.arange(32, dtype=torch.float32)
    torch.save({"model": {"left": shared, "right": shared.view(4, 8)},
                "step": 7}, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    assert valid_torch_archive_storages(path, loaded)

    _flip_first_torch_storage_byte(path)
    silently_altered = torch.load(path, map_location="cpu", weights_only=False)
    assert torch.isfinite(silently_altered["model"]["left"]).all()
    assert not valid_torch_archive_storages(path, silently_altered)


def test_raw_cache_fast_path_never_reinterprets_another_two_byte_dtype(tmp_path: Path):
    path = tmp_path / "fp16.pt"
    expected = torch.linspace(-2, 2, 5 * 4 * 1152, dtype=torch.float16).reshape(5, 4, 1152)
    torch.save(expected, path)
    actual = _load_raw_tensor_archive(
        path, shape=tuple(expected.shape), stride=tuple(expected.stride()),
        storage_offset=expected.storage_offset(), dtype=torch.bfloat16,
        storage_bytes=expected.untyped_storage().nbytes())
    assert actual.dtype == torch.float16
    torch.testing.assert_close(actual, expected)


def test_preloaded_feature_cache_survives_removing_backing_file(tmp_path: Path):
    class Vision:
        max_input_patches = 1024
        cache_fingerprint = "preload-test"
        patch_embed = type("Patch", (), {"proj": type("Proj", (), {"weight": torch.empty(1)})()})()
        def __init__(self):
            self.calls = 0

        def encode_many(self, images):
            self.calls += 1
            return [torch.ones(70, 4, 1152) for _ in images]
    image = tmp_path / "x.jpg"
    Image.new("RGB", (4, 4)).save(image)
    cache = tmp_path / "cache"
    projector, vision = MoonViTPrefixProjector(32, 5), Vision()
    rows = [{"image": image}]
    expected = cached_features(rows, vision, projector, cache)[0]
    loaded, resident = preload_feature_cache(rows, vision, projector, cache)
    key = cache / feature_cache_key(image, max_input_patches=1024, prefix_tokens=5,
                                    vision_fingerprint="preload-test")
    key.unlink()
    actual = cached_features(rows, vision, projector, cache)[0]
    assert loaded == 1 and resident == expected.numel() * expected.element_size()
    assert vision.calls == 1 and torch.equal(actual, expected)
    _FEATURE_MEMORY_CACHE.pop(key, None)


def test_background_feature_preload_honors_shutdown_before_queueing(tmp_path: Path):
    class Vision:
        max_input_patches = 1024
        cache_fingerprint = "preload-stop"

    image = tmp_path / "x.jpg"
    Image.new("RGB", (4, 4)).save(image)
    cache = tmp_path / "cache"
    cache.mkdir()
    projector = MoonViTPrefixProjector(32, 5)
    key = cache / feature_cache_key(
        image, max_input_patches=1024, prefix_tokens=5,
        vision_fingerprint="preload-stop")
    torch.save(torch.ones(5, 4, 1152), key)
    stop = threading.Event()
    stop.set()
    loaded, resident = preload_feature_cache(
        [{"image": image}], Vision(), projector, cache, stop_event=stop)
    assert (loaded, resident) == (0, 0)


def test_moonvit_attention_batch_matches_individual_images():
    torch.manual_seed(1)
    block = _Block(hidden=16, heads=4, intermediate=24).eval()
    x = torch.randn(3, 4, 16)
    grid = torch.tensor([[1, 2, 2]])
    from rwkv_lab.moonvit import _rope
    freqs = _rope(grid, 4, x.device)
    together = block(x, freqs)
    separate = torch.cat([block(row, freqs).unsqueeze(0) for row in x])
    torch.testing.assert_close(together, separate, atol=2e-6, rtol=2e-6)


def test_variable_grid_encoder_matches_individual_encoder():
    # Use one tiny block while exercising the real padding/position/RoPE path.
    torch.manual_seed(2)
    model = MoonViT(max_input_patches=64)
    model.encoder.blocks = torch.nn.ModuleList([_Block()])
    prepared = [
        (torch.randn(16, 3, 14, 14), torch.tensor([[1, 4, 4]])),
        (torch.randn(24, 3, 14, 14), torch.tensor([[1, 4, 6]])),
    ]
    with torch.no_grad():
        separate = [model._encode_patches(patches, grid).squeeze(0) for patches, grid in prepared]
        together = model._encode_variable(prepared)
    for expected, actual in zip(separate, together):
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)


def test_resize_obeys_hard_patch_cap():
    patches, _ = _resize(Image.new("RGB", (4000, 500)), max_input_patches=128)
    assert len(patches) <= 128


def test_header_only_resize_geometry_matches_decoded_resize():
    image = Image.new("RGB", (733, 411))
    patches, grid = _resize(image, max_input_patches=256)
    new_w, new_h, pad_w, pad_h = _resize_geometry(*image.size, max_input_patches=256)
    assert len(patches) == ((new_w + pad_w) // 14) * ((new_h + pad_h) // 14)
    assert tuple(grid[0, 1:].tolist()) == ((new_h + pad_h) // 14, (new_w + pad_w) // 14)


def test_single_tap_staged_features_pass_validation_end_to_end(tmp_path: Path):
    from rwkv_lab.moonvit import pool_features

    class Vision:
        max_input_patches = 1024
        cache_fingerprint = "single-tap-test"
        tap_layers = (8,)
        feature_stages = 1
        view_mode = "full"
        patch_embed = type(
            "Patch", (), {"proj": type("Proj", (), {"weight": torch.empty(1)})()})()

        def __init__(self):
            self.calls = 0

        def encode_many(self, images):
            self.calls += 1
            # One tap always yields the staged [stages, groups, 4, 1152] shape.
            return [torch.ones(1, 70, 4, 1152) for _ in images]

    image = tmp_path / "x.jpg"
    Image.new("RGB", (4, 4)).save(image)
    projector, vision = MoonViTPrefixProjector(32, 5), Vision()
    rows = [{"image": image}]
    cache = tmp_path / "cache"
    first = cached_features(rows, vision, projector, cache)
    second = cached_features(rows, vision, projector, cache)
    assert vision.calls == 1
    assert first[0].shape == (1, 5, 4, 1152)
    assert torch.equal(first[0], second[0])
    key = cache / feature_cache_key(
        image, max_input_patches=1024, prefix_tokens=5,
        vision_fingerprint="single-tap-test", tap_layers=(8,))
    assert key.is_file()
    assert cache_entry_valid(key, 5, 1)
    # Zero stages is the unstaged 3-dim contract; positive counts are staged.
    pooled = pool_features(torch.ones(1, 70, 4, 1152), 5).squeeze(0)
    assert valid_pooled_feature(pooled, 5, 1)
    assert not valid_pooled_feature(pooled, 5)
    assert valid_pooled_feature(torch.ones(5, 4, 1152), 5)
    assert not valid_pooled_feature(torch.ones(5, 4, 1152), 5, 1)


def test_moonvit_feature_stages_zero_means_unstaged():
    assert MoonViT(max_input_patches=64).feature_stages == 0
    assert MoonViT(max_input_patches=64, tap_layers=(8,)).feature_stages == 1
    assert MoonViT(max_input_patches=64, tap_layers=(3, 9)).feature_stages == 2


def test_caption_layer_vision_path_injects_prefix_width_features():
    from rwkv_lab.deep_vision import LayerMatchedVisionInjector
    from rwkv_lab.moonvit import pool_features

    class Vision:
        def __call__(self, images):
            # Raw staged MoonViT output: [stages, groups, 4, 1152] per image
            # with a group count wider than the trained prefix width.
            return [torch.randn(2, 11, 4, 1152) for _ in images]

    class Layer(torch.nn.Module):
        def forward(self, hidden_states, **_kwargs):
            return hidden_states

    prefix_tokens = 5
    projector = MoonViTPrefixProjector(16, prefix_tokens)
    vision = Vision()
    raw_features = vision([object()])
    # Mirror rwkv_lab.vision_caption.caption: pool to the cacheable training
    # contract before the projector and the layer-matched injector.
    features = [pool_features(item, projector.prefix_tokens).squeeze(0)
                for item in raw_features]
    prefix = projector(features)
    assert prefix.shape == (1, prefix_tokens, 16)

    layers = torch.nn.ModuleList([Layer(), Layer()])
    injector = LayerMatchedVisionInjector(16, (0, 1), rank=4)
    for adapter in injector.adapters.values():
        torch.nn.init.ones_(adapter.up.weight)
    injector.install(layers)
    start = 2
    hidden = torch.zeros(1, 9, 16)
    with injector.use_features(torch.stack(features), (start,)):
        output = hidden
        for layer in layers:
            output = layer(output)
    changed = (output != hidden).any(-1)[0]
    # Exactly prefix_tokens positions, beginning at the visual span start,
    # receive the injected residual.
    assert int(changed.sum()) == prefix_tokens
    assert changed[start:start + prefix_tokens].all()
    # The raw unpooled stack violates the trained width contract and must not
    # silently inject a wider span.
    injection_width = torch.stack(raw_features).shape[2]
    assert injection_width != prefix_tokens
    injector.close()


def test_contrastive_head_detaches_caption_targets_itself():
    head = ImageTextContrastiveHead(hidden_size=8, width=4)
    prefix = torch.randn(2, 3, 8, requires_grad=True)
    targets = torch.randn(4, 8, requires_grad=True)
    positions = torch.tensor([0, 0, 1, 1])
    loss, _ = head(prefix, targets, positions)
    loss.backward()
    # The documented invariant: the auxiliary can never improve by rewriting
    # the language model's embeddings, even if a caller forgets to detach.
    assert targets.grad is None
    assert prefix.grad is not None
