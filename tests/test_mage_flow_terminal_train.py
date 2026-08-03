from __future__ import annotations

import itertools
import json
from types import SimpleNamespace

import pytest
import torch

from rwkv_lab.mage_flow_optimizations import FP32MasterAdamW
from rwkv_lab.mage_flow_terminal_train import (
    ResidentExpertOptimizerBank,
    TerminalExpertTrainConfig,
    _architecture_migration_required,
    _architecture_resume_change,
    _backward_training_objective,
    _best_evaluation_improved,
    _cache_span_should_stop,
    _contract_fingerprint,
    _epoch_batches,
    _evaluation_optimization_loss,
    _export_checkpoint_experts,
    _objective_resume_change,
    _promote_best_checkpoint,
    _rapid_alternating_batches,
    _repa_reset_resume_change,
    _resume_batch_index,
    _resume_domain_positions,
    _runtime_only_resume_change,
    _save_checkpoint,
    domain_window_schedule_report,
    plan_cache_span,
    prepare_cache_span,
    prepare_run,
    resolved_worker_component_contract,
    weighted_domain_windows,
)
from rwkv_lab.mage_flow_training_objectives import (
    VAERepresentationAlignment,
    apply_immiscible_noise_assignment,
    effective_flow_loss_weights,
    flow_min_snr_weights,
    load_repa_projection,
    minimum_cost_assignment,
    rectified_flow_loss_per_example,
    save_repa_projection,
    velocity_direction_loss_per_example,
    weighted_rectified_flow_loss,
    weighted_velocity_direction_loss,
)
from rwkv_lab.mage_flow_tread_looping import TreadLoopConfig


def _row(index: int, domain: str) -> dict:
    return {
        "image": f"/images/{index}.png",
        "image_id": f"id-{index}",
        "domain": domain,
        "source": "test",
        "caption": f"Caption {index}",
        "conditioning_text": f"Caption {index}",
        "conditioning_kind": "human",
        "is_captioned": True,
        "training_scope": "expert_and_selected_shared",
        "train_width": 512,
        "train_height": 512,
        "latent_tokens": 1024,
    }


def _write_manifest(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_terminal_config_selects_one_resident_domain(tmp_path):
    manifest = tmp_path / "train.jsonl"
    _write_manifest(
        manifest,
        [_row(0, "photo"), _row(1, "animation"), _row(2, "photo")],
    )
    expert = tmp_path / "photo.safetensors"
    expert.touch()
    config = TerminalExpertTrainConfig(
        # Explicit: the field default is an absolute path on one host.
        model_path=None,
        domain="photo",
        train_manifest=str(manifest),
        expert_checkpoint=str(expert),
        output_dir=str(tmp_path / "output"),
        max_steps=2,
    )
    config.validate()
    assert config.backbone_learning_rate_multiplier == pytest.approx(0.5)
    assert config.train_backbone_final_fraction == pytest.approx(1 / 3)
    assert config.repa_enabled
    assert config.immiscible_enabled
    assert config.flow_loss_weighting == "soft_min_snr"
    receipt = prepare_run(config, tmp_path / "plan")
    assert receipt["domain"] == "photo"
    assert receipt["resident_experts"] == 1
    assert receipt["train_examples"] == 2
    assert receipt["training_objectives"] == {
        "vae_repa": True,
        "immiscible_diffusion": True,
        "flow_loss_weighting": "soft_min_snr",
        "timestep_sampling": "uniform",
        "velocity_direction_loss_weight": 0.0,
    }
    assert "mage_flow_terminal_train" in (tmp_path / "plan/launch.sh").read_text()
    config.attention_backend = "flash4"
    prepare_run(config, tmp_path / "fa4-plan")
    assert ".venv-mage-flow-fa4" in (
        tmp_path / "fa4-plan/launch.sh"
    ).read_text()


def test_terminal_worker_components_are_exact_and_enter_resume_identity(tmp_path):
    manifest = tmp_path / "train.jsonl"
    _write_manifest(manifest, [_row(0, "photo")])
    expert = tmp_path / "photo.safetensors"
    expert.touch()
    config = TerminalExpertTrainConfig(
        domain="photo",
        train_manifest=str(manifest),
        expert_checkpoint=str(expert),
        output_dir=str(tmp_path / "output"),
        model_path=None,
        max_steps=20,
        warmup_steps=2,
    )
    configurations = {
        "optimizer": {
            "learning_rate": config.learning_rate,
            "beta1": config.adam_beta1,
            "beta2": config.adam_beta2,
            "epsilon": config.adam_epsilon,
            "foreach": True,
            "fused": False,
        },
        "weight_decay": {"weight_decay": config.weight_decay},
        "parameter_router": {
            "shared_backbone_multiplier": config.backbone_learning_rate_multiplier,
            "repa_projection_multiplier": config.repa_learning_rate_multiplier,
        },
        "learning_rate": {
            "warmup_steps": config.warmup_steps,
            "max_steps": config.max_steps,
            "minimum_ratio": config.min_learning_rate_ratio,
        },
        "gradient_clipping": {
            "max_norm": config.max_grad_norm,
            "norm_type": 2.0,
            "error_if_nonfinite": False,
        },
        "loop_gate_gradient_clipping": {
            "max_norm": config.loop_gate_max_grad_norm,
            "norm_type": 2.0,
            "error_if_nonfinite": False,
        },
    }

    class Components:
        composition = SimpleNamespace(composition_digest="sha256:" + "b" * 64)

        def configuration(self, slot, *, category):
            assert category in {
                "optimizer",
                "parameter_router",
                "learning_rate_schedule",
                "weight_decay_schedule",
                "gradient_clipping",
            }
            return configurations[slot]

        def evidence(self):
            return {
                slot: {"category": slot, "implementation": slot, "descriptor_digest": slot}
                for slot in configurations
            }

    components = Components()
    learning_rate, evidence, composition_digest = resolved_worker_component_contract(
        config, components
    )
    assert learning_rate == config.learning_rate
    assert set(evidence) == set(configurations)
    assert composition_digest == components.composition.composition_digest
    assert _contract_fingerprint(config) != _contract_fingerprint(
        config, component_composition_digest=composition_digest
    )

    config.learning_rate *= 0.25
    controls = SimpleNamespace(
        effective_values={"learning_rate": config.learning_rate}
    )
    live_learning_rate, _evidence, _digest = resolved_worker_component_contract(
        config, components, controls
    )
    assert live_learning_rate == pytest.approx(config.learning_rate)
    changed_controls = TerminalExpertTrainConfig(
        **{
            **config.__dict__,
            "learning_rate": config.learning_rate * 0.5,
            "eval_every": config.eval_every * 2,
            "caption_dropout": 0.25,
        }
    )
    mutable = ("learning_rate", "eval_every", "caption_dropout")
    assert _contract_fingerprint(
        config, mutable_control_keys=mutable
    ) == _contract_fingerprint(
        changed_controls, mutable_control_keys=mutable
    )
    assert _contract_fingerprint(config) != _contract_fingerprint(changed_controls)
    config.learning_rate = configurations["optimizer"]["learning_rate"]

    configurations["parameter_router"] = {
        **configurations["parameter_router"],
        "shared_backbone_multiplier": 1.0,
    }
    with pytest.raises(ValueError, match="parameter_router composition disagrees"):
        resolved_worker_component_contract(config, components)


def test_terminal_config_rejects_neutral_or_missing_domain(tmp_path):
    manifest = tmp_path / "train.jsonl"
    _write_manifest(manifest, [_row(0, "animation")])
    expert = tmp_path / "expert.safetensors"
    expert.touch()
    general = TerminalExpertTrainConfig(
        # Explicit: the field default is an absolute path on one host.
        model_path=None,
        domain="general",
        train_manifest=str(manifest),
        expert_checkpoint=str(expert),
        output_dir=str(tmp_path / "output"),
    )
    with pytest.raises(ValueError, match="photo or animation"):
        general.validate()
    photo = TerminalExpertTrainConfig(
        # Explicit: the field default is an absolute path on one host.
        model_path=None,
        domain="photo",
        train_manifest=str(manifest),
        expert_checkpoint=str(expert),
        output_dir=str(tmp_path / "output"),
    )
    with pytest.raises(ValueError, match="no photo"):
        photo.validate()


def test_domain_window_batches_are_deterministic_and_cover_once():
    first = _epoch_batches(11, batch_size=3, seed=42, epoch=0)
    second = _epoch_batches(11, batch_size=3, seed=42, epoch=0)
    later = _epoch_batches(11, batch_size=3, seed=42, epoch=1)
    assert first == second
    assert first != later
    assert sorted(index for batch in first for index in batch) == list(range(11))
    assert [len(batch) for batch in first] == [3, 3, 3, 2]


def test_balanced_epoch_batches_preserve_each_optimizer_window():
    tokens = [1, 2, 3, 4, 10, 20, 30, 40, 5, 6, 7, 8, 50, 60, 70, 80]
    original = _epoch_batches(16, batch_size=1, seed=42, epoch=0)
    balanced = _epoch_batches(
        16,
        batch_size=2,
        seed=42,
        epoch=0,
        accumulation_steps=4,
        token_counts=tokens,
        balance_accumulation_window=True,
    )

    original_indices = list(itertools.chain.from_iterable(original))
    balanced_indices = list(itertools.chain.from_iterable(balanced))
    for start in range(0, 16, 8):
        assert set(balanced_indices[start : start + 8]) == set(
            original_indices[start : start + 8]
        )
    assert all(len(batch) == 2 for batch in balanced)
    for start in range(0, len(balanced), 4):
        loads = [
            sum(tokens[index] for index in batch)
            for batch in balanced[start : start + 4]
        ]
        assert max(loads) - min(loads) < max(
            tokens[index]
            for batch in balanced[start : start + 4]
            for index in batch
        )


def test_equal_effective_batch_resume_translates_data_position(tmp_path):
    manifest = tmp_path / "train.jsonl"
    _write_manifest(manifest, [_row(index, "animation") for index in range(16)])
    expert = tmp_path / "animation.safetensors"
    expert.touch()
    output = tmp_path / "output"
    output.mkdir()
    checkpoint = output / "checkpoint-00000002"
    checkpoint.mkdir()
    old = TerminalExpertTrainConfig(
        domain="animation",
        train_manifest=str(manifest),
        expert_checkpoint=str(expert),
        output_dir=str(output),
        max_steps=10,
        microbatch_size=1,
        gradient_accumulation_steps=8,
    )
    old_fingerprint = "old"
    (output / "run_contract.json").write_text(
        json.dumps(
            {
                "fingerprint": old_fingerprint,
                "config": old.__dict__,
            }
        )
    )
    (checkpoint / "checkpoint.json").write_text(
        json.dumps({"fingerprint": old_fingerprint})
    )
    current = TerminalExpertTrainConfig(
        **(
            old.__dict__
            | {
                "resume_from": str(checkpoint),
                "microbatch_size": 2,
                "gradient_accumulation_steps": 4,
                "prefetch_batches": 4,
            }
        )
    )

    assert _runtime_only_resume_change(checkpoint, current)
    assert _resume_batch_index(checkpoint, current, 16) == 8
    assert _resume_domain_positions(
        checkpoint,
        current,
        {
            "photo": {"epoch": 1, "batch_start": 12},
            "animation": {"epoch": 2, "batch_start": 8},
        },
    ) == {
        "photo": {"epoch": 1, "batch_start": 6},
        "animation": {"epoch": 2, "batch_start": 4},
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rapid_expert_alternation", True),
        ("loop_gate_update_multiplier", 2.0),
        ("loop_gate_max_grad_norm", 0.5),
        ("balance_accumulation_window", False),
    ],
)
def test_resume_rejects_training_semantic_changes(tmp_path, field, value):
    manifest = tmp_path / "train.jsonl"
    _write_manifest(manifest, [_row(index, "animation") for index in range(16)])
    expert = tmp_path / "animation.safetensors"
    expert.touch()
    output = tmp_path / "output"
    output.mkdir()
    checkpoint = output / "checkpoint-00000002"
    checkpoint.mkdir()
    old = TerminalExpertTrainConfig(
        domain="animation",
        train_manifest=str(manifest),
        expert_checkpoint=str(expert),
        output_dir=str(output),
        max_steps=10,
        balance_accumulation_window=True,
    )
    (output / "run_contract.json").write_text(
        json.dumps({"fingerprint": "old", "config": old.__dict__})
    )
    (checkpoint / "checkpoint.json").write_text(
        json.dumps({"fingerprint": "old"})
    )
    current = TerminalExpertTrainConfig(
        **(old.__dict__ | {"resume_from": str(checkpoint), field: value})
    )

    assert not _runtime_only_resume_change(checkpoint, current)


def test_explicit_repa_reset_allows_only_repa_migration(tmp_path):
    manifest = tmp_path / "train.jsonl"
    _write_manifest(manifest, [_row(index, "animation") for index in range(16)])
    expert = tmp_path / "animation.safetensors"
    expert.touch()
    old_run = tmp_path / "old-run"
    old_run.mkdir()
    checkpoint = old_run / "checkpoint-00000002"
    checkpoint.mkdir()
    old = TerminalExpertTrainConfig(
        domain="animation",
        train_manifest=str(manifest),
        expert_checkpoint=str(expert),
        output_dir=str(old_run),
        max_steps=10,
        repa_hidden_layer=12,
        repa_loss_weight=0.5,
        repa_student_normalization="none",
        repa_per_example_loss_cap=None,
        repa_use_posterior_mean=False,
        repa_exclude_loop_gate_gradients=False,
    )
    (old_run / "run_contract.json").write_text(
        json.dumps({"fingerprint": "old", "config": old.__dict__})
    )
    (checkpoint / "checkpoint.json").write_text(
        json.dumps({"fingerprint": "old"})
    )
    current = TerminalExpertTrainConfig(
        **(
            old.__dict__
            | {
                "output_dir": str(tmp_path / "new-run"),
                "resume_from": str(checkpoint),
                "repa_hidden_layer": 1,
                "repa_loss_weight": 0.1,
                "repa_student_normalization": "token_rms",
                "repa_per_example_loss_cap": 5.0,
                "repa_use_posterior_mean": True,
                "repa_exclude_loop_gate_gradients": True,
                "repa_reset_projection_on_resume": True,
                "microbatch_size": 4,
                "gradient_accumulation_steps": 2,
            }
        )
    )

    assert _repa_reset_resume_change(checkpoint, current)
    current.weight_decay = 0.5
    assert not _repa_reset_resume_change(checkpoint, current)


def test_explicit_objective_migration_allows_lightning_recipe_only(tmp_path):
    manifest = tmp_path / "train.jsonl"
    _write_manifest(manifest, [_row(index, "animation") for index in range(16)])
    expert = tmp_path / "animation.safetensors"
    expert.touch()
    old_run = tmp_path / "old-run"
    old_run.mkdir()
    checkpoint = old_run / "checkpoint-00000002"
    checkpoint.mkdir()
    old = TerminalExpertTrainConfig(
        domain="animation",
        train_manifest=str(manifest),
        expert_checkpoint=str(expert),
        output_dir=str(old_run),
        max_steps=10,
        microbatch_size=8,
        gradient_accumulation_steps=1,
        repa_reset_projection_on_resume=True,
    )
    (old_run / "run_contract.json").write_text(
        json.dumps({"fingerprint": "old", "config": old.__dict__})
    )
    (checkpoint / "checkpoint.json").write_text(
        json.dumps({"fingerprint": "old"})
    )
    current = TerminalExpertTrainConfig(
        **(
            old.__dict__
            | {
                "output_dir": str(tmp_path / "new-run"),
                "resume_from": str(checkpoint),
                "microbatch_size": 4,
                "gradient_accumulation_steps": 2,
                "timestep_sampling": "logit_normal",
                "velocity_direction_loss_weight": 0.1,
                "repa_reset_projection_on_resume": False,
                "allow_objective_migration_on_resume": True,
            }
        )
    )

    assert _objective_resume_change(checkpoint, current)
    current.weight_decay = 0.5
    assert not _objective_resume_change(checkpoint, current)


def test_explicit_lightning_block_migration_requires_optimizer_reset(tmp_path):
    manifest = tmp_path / "train.jsonl"
    _write_manifest(manifest, [_row(index, "photo") for index in range(16)])
    expert = tmp_path / "photo.safetensors"
    expert.touch()
    old_run = tmp_path / "old-run"
    old_run.mkdir()
    checkpoint = old_run / "checkpoint-00000002"
    checkpoint.mkdir()
    old = TerminalExpertTrainConfig(
        # Explicit: the field default is an absolute path on one host.
        model_path=None,
        domain="photo",
        train_manifest=str(manifest),
        expert_checkpoint=str(expert),
        output_dir=str(old_run),
        max_steps=10,
    )
    (old_run / "run_contract.json").write_text(
        json.dumps({"fingerprint": "old", "config": old.__dict__})
    )
    (checkpoint / "checkpoint.json").write_text(
        json.dumps({"fingerprint": "old"})
    )
    current = TerminalExpertTrainConfig(
        **(
            old.__dict__
            | {
                "output_dir": str(tmp_path / "new-run"),
                "resume_from": str(checkpoint),
                "lightning_swiglu": True,
                "lightning_rmsnorm": True,
                "reset_optimizer_on_architecture_migration": True,
            }
        )
    )

    current.validate()
    assert _architecture_migration_required(checkpoint, current)
    assert _architecture_resume_change(checkpoint, current)
    current.weight_decay = 0.5
    assert not _architecture_resume_change(checkpoint, current)


def test_lightning_resume_does_not_repeat_architecture_optimizer_reset(tmp_path):
    manifest = tmp_path / "train.jsonl"
    _write_manifest(manifest, [_row(index, "photo") for index in range(16)])
    expert = tmp_path / "photo.safetensors"
    expert.touch()
    run = tmp_path / "run"
    run.mkdir()
    checkpoint = run / "checkpoint-00000002"
    checkpoint.mkdir()
    migrated = TerminalExpertTrainConfig(
        # Explicit: the field default is an absolute path on one host.
        model_path=None,
        domain="photo",
        train_manifest=str(manifest),
        expert_checkpoint=str(expert),
        output_dir=str(run),
        max_steps=10,
        lightning_swiglu=True,
        lightning_rmsnorm=True,
        reset_optimizer_on_architecture_migration=True,
    )
    (run / "run_contract.json").write_text(
        json.dumps({"fingerprint": "migrated", "config": migrated.__dict__})
    )
    (checkpoint / "checkpoint.json").write_text(
        json.dumps({"fingerprint": "migrated"})
    )
    resumed = TerminalExpertTrainConfig(
        **(
            migrated.__dict__
            | {
                "resume_from": str(checkpoint),
            }
        )
    )

    resumed.validate()
    assert not _architecture_migration_required(checkpoint, resumed)
    assert not _architecture_resume_change(checkpoint, resumed)


def test_tread_controller_migration_requires_one_explicit_optimizer_reset(tmp_path):
    manifest = tmp_path / "train.jsonl"
    _write_manifest(manifest, [_row(index, "animation") for index in range(16)])
    expert = tmp_path / "animation.safetensors"
    expert.touch()
    controller = tmp_path / "controller.safetensors"
    controller.touch()
    old_run = tmp_path / "old-run"
    old_run.mkdir()
    checkpoint = old_run / "checkpoint-00000002"
    checkpoint.mkdir()
    old = TerminalExpertTrainConfig(
        model_path=None,
        domain="animation",
        train_manifest=str(manifest),
        expert_checkpoint=str(expert),
        output_dir=str(old_run),
        max_steps=10,
    )
    old_document = dict(old.__dict__)
    old_document["tread_factored_looping"] = TreadLoopConfig().to_dict()
    (checkpoint / "run_contract.json").write_text(
        json.dumps({"fingerprint": "old", "config": old_document})
    )
    (checkpoint / "checkpoint.json").write_text(
        json.dumps({"fingerprint": "old", "run_contract": "run_contract.json"})
    )
    current = TerminalExpertTrainConfig(
        **(
            old.__dict__
            | {
                "output_dir": str(tmp_path / "new-run"),
                "resume_from": str(checkpoint),
                "tread_loop_checkpoint": str(controller),
                "reset_optimizer_on_architecture_migration": True,
            }
        )
    )

    current.validate()
    assert _architecture_migration_required(checkpoint, current)
    assert _architecture_resume_change(checkpoint, current)
    current.reset_optimizer_on_architecture_migration = False
    assert not _architecture_resume_change(checkpoint, current)


def test_terminal_checkpoint_persists_complete_tread_controller(
    tmp_path, monkeypatch
):
    output = tmp_path / "run"
    output.mkdir()
    (output / "run_contract.json").write_text(
        json.dumps({"fingerprint": "sha256:" + "a" * 64}),
        encoding="utf-8",
    )
    tread_controller = object()
    transformer = SimpleNamespace(tread_loop_controller=tread_controller)

    def save_expert(_controller, path):
        path.write_bytes(b"expert")

    def save_shared(_transformer, _controller, path):
        path.write_bytes(b"shared")
        return path

    observed = []

    def save_tread(controller, path):
        observed.append(controller)
        path.write_bytes(b"complete-tread-state")
        return {"tensor_count": 3}

    monkeypatch.setattr(
        "rwkv_lab.mage_flow_terminal_train.save_terminal_expert", save_expert
    )
    monkeypatch.setattr(
        "rwkv_lab.mage_flow_terminal_train.save_terminal_shared_backbone",
        save_shared,
    )
    monkeypatch.setattr(
        "rwkv_lab.mage_flow_terminal_train.save_tread_loop_controller",
        save_tread,
    )
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", list)
    final = _save_checkpoint(
        controller=object(),
        transformer=transformer,
        repa=None,
        optimizer=SimpleNamespace(state_dict=lambda: {"optimizer": 1}),
        scheduler=SimpleNamespace(state_dict=lambda: {"scheduler": 1}),
        output_dir=output,
        domain="animation",
        step=9,
        epoch=1,
        batch_index=4,
        fingerprint="sha256:" + "a" * 64,
        keep=2,
    )

    contract = json.loads((final / "checkpoint.json").read_text())
    assert observed == [tread_controller]
    assert contract["tread_loop_controller"] == (
        "mageflow-tread-loop-controller.safetensors"
    )
    assert contract["run_contract"] == "run_contract.json"
    assert json.loads((final / "run_contract.json").read_text())[
        "fingerprint"
    ] == "sha256:" + "a" * 64
    assert (final / contract["tread_loop_controller"]).read_bytes() == (
        b"complete-tread-state"
    )


def test_rapid_alternating_batch_stream_persists_and_groups_accumulation():
    rows = {
        domain: [_row(offset + index, domain) for index in range(8)]
        for domain, offset in (("photo", 0), ("animation", 100))
    }
    stream = _rapid_alternating_batches(
        rows,
        {
            "photo": {"epoch": 0, "batch_start": 0},
            "animation": {"epoch": 0, "batch_start": 0},
        },
        starting_domain="photo",
        batch_size=1,
        accumulation_steps=2,
        seed=42,
        balance_accumulation_window=True,
    )
    observed = [next(stream) for _ in range(6)]

    assert [domain for domain, *_rest in observed] == [
        "photo",
        "photo",
        "animation",
        "animation",
        "photo",
        "photo",
    ]
    assert [batch for _domain, _epoch, batch, _indices in observed] == [
        0,
        1,
        0,
        1,
        2,
        3,
    ]

    bounded = _rapid_alternating_batches(
        rows,
        {
            "photo": {"epoch": 0, "batch_start": 0},
            "animation": {"epoch": 0, "batch_start": 0},
        },
        starting_domain="photo",
        batch_size=1,
        accumulation_steps=2,
        seed=42,
        balance_accumulation_window=True,
        optimizer_steps=2,
    )
    assert len(list(bounded)) == 4


def test_final_cache_boundary_runs_completion_and_exports_checkpoint_experts(
    tmp_path,
):
    assert _cache_span_should_stop(
        step=5,
        max_steps=10,
        covered_until_step=5,
    )
    assert not _cache_span_should_stop(
        step=10,
        max_steps=10,
        covered_until_step=10,
    )

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    for domain in ("photo", "animation"):
        (checkpoint / f"{domain}.safetensors").write_bytes(domain.encode())
    (checkpoint / "checkpoint.json").write_text(
        json.dumps(
            {
                "domain": "photo",
                "expert": "photo.safetensors",
                "experts": {
                    "photo": "photo.safetensors",
                    "animation": "animation.safetensors",
                },
            }
        )
    )
    output = tmp_path / "output"
    output.mkdir()
    exported = _export_checkpoint_experts(checkpoint, output)

    assert exported["photo"].read_bytes() == b"photo"
    assert exported["animation"].read_bytes() == b"animation"


def test_eval_optimization_metric_keeps_all_training_objectives():
    assert _evaluation_optimization_loss(
        1.0,
        velocity_direction_loss=5.0,
        velocity_direction_weight=0.2,
        loop_auxiliary_loss=2.0,
        loop_auxiliary_weight=0.1,
        loop_ponder_loss=3.0,
        loop_ponder_weight=0.01,
        repa_loss=4.0,
        repa_weight=0.5,
    ) == pytest.approx(4.23)


def test_cache_span_follows_resume_position_and_domain_windows(tmp_path):
    train_manifest = tmp_path / "train.jsonl"
    rows = [
        *[_row(index, "animation") for index in range(20)],
        *[_row(100 + index, "photo") for index in range(20)],
    ]
    _write_manifest(train_manifest, rows)
    schedule = {
        "schema": "rwkv-lab.mage-flow-domain-window-schedule.v1",
        "train_manifest": str(train_manifest.resolve()),
        "max_steps": 6,
        "windows": [
            {"index": 0, "domain": "animation", "start_step": 0, "end_step": 3},
            {"index": 1, "domain": "photo", "start_step": 3, "end_step": 5},
            {"index": 2, "domain": "animation", "start_step": 5, "end_step": 6},
        ],
    }
    schedule_path = tmp_path / "schedule.json"
    schedule_path.write_text(json.dumps(schedule))
    animation = tmp_path / "animation.safetensors"
    photo = tmp_path / "photo.safetensors"
    animation.touch()
    photo.touch()
    checkpoint = tmp_path / "checkpoint-00000001"
    checkpoint.mkdir()
    (checkpoint / "checkpoint.json").write_text(
        json.dumps(
            {
                "schema": "rwkv-lab.mage-flow-terminal-train.v3",
                "domain": "animation",
                "step": 1,
                "epoch": 0,
                "batch_index": 2,
                "fingerprint": "test",
            }
        )
    )
    config = TerminalExpertTrainConfig(
        domain="animation",
        train_manifest=str(train_manifest),
        expert_checkpoint=str(animation),
        expert_checkpoints={
            "animation": str(animation),
            "photo": str(photo),
        },
        domain_window_schedule=str(schedule_path),
        output_dir=str(tmp_path / "run"),
        max_steps=6,
        microbatch_size=1,
        gradient_accumulation_steps=2,
    )

    selected, report = plan_cache_span(
        config,
        checkpoint,
        optimizer_steps=4,
    )

    animation_order = [
        index
        for batch in _epoch_batches(20, batch_size=1, seed=42, epoch=0)
        for index in batch
    ]
    photo_order = [
        index
        for batch in _epoch_batches(20, batch_size=1, seed=42, epoch=0)
        for index in batch
    ]
    assert [row["image_id"] for row in selected] == [
        *[f"id-{index}" for index in animation_order[2:6]],
        *[f"id-{100 + index}" for index in photo_order[:4]],
    ]
    assert report["start_step"] == 1
    assert report["end_step"] == 5
    assert report["example_occurrences"] == 8
    assert report["unique_examples"] == 8
    assert report["domain_occurrences"] == {"photo": 4, "animation": 4}
    assert report["ending_domain"] == "animation"

    prepared = prepare_cache_span(
        config,
        checkpoint,
        optimizer_steps=4,
        output_dir=tmp_path / "cache-span",
    )
    build = json.loads(
        (tmp_path / "cache-span/cache_build_config.json").read_text()
    )
    resume = json.loads(
        (tmp_path / "cache-span/resume_config.json").read_text()
    )
    assert prepared["unique_examples"] == 8
    assert build["encoder_cache_mode"] == "read_write"
    assert build["domain_window_schedule"] is None
    assert resume["encoder_cache_mode"] == "read_only"
    assert resume["offload_cached_encoders"] is True
    assert resume["encoder_cache_covered_until_step"] == 5


def test_cache_span_follows_strict_per_step_expert_alternation(tmp_path):
    train_manifest = tmp_path / "train.jsonl"
    rows = [
        *[_row(index, "animation") for index in range(20)],
        *[_row(100 + index, "photo") for index in range(20)],
    ]
    _write_manifest(train_manifest, rows)
    schedule = {
        "schema": "rwkv-lab.mage-flow-domain-window-schedule.v1",
        "train_manifest": str(train_manifest.resolve()),
        "max_steps": 6,
        "windows": [
            {"index": 0, "domain": "animation", "start_step": 0, "end_step": 3},
            {"index": 1, "domain": "photo", "start_step": 3, "end_step": 6},
        ],
    }
    schedule_path = tmp_path / "schedule.json"
    schedule_path.write_text(json.dumps(schedule))
    animation = tmp_path / "animation.safetensors"
    photo = tmp_path / "photo.safetensors"
    animation.touch()
    photo.touch()
    checkpoint = tmp_path / "checkpoint-00000001"
    checkpoint.mkdir()
    (checkpoint / "checkpoint.json").write_text(
        json.dumps(
            {
                "schema": "rwkv-lab.mage-flow-terminal-train.v3",
                "domain": "animation",
                "step": 1,
                "epoch": 0,
                "batch_index": 2,
                "fingerprint": "test",
                "domain_positions": {
                    "animation": {"epoch": 0, "batch_start": 2},
                    "photo": {"epoch": 0, "batch_start": 0},
                },
            }
        )
    )
    config = TerminalExpertTrainConfig(
        domain="animation",
        train_manifest=str(train_manifest),
        expert_checkpoint=str(animation),
        expert_checkpoints={
            "animation": str(animation),
            "photo": str(photo),
        },
        domain_window_schedule=str(schedule_path),
        output_dir=str(tmp_path / "run"),
        max_steps=6,
        microbatch_size=1,
        gradient_accumulation_steps=2,
        rapid_expert_alternation=True,
        expert_optimizer_state_device="cuda",
    )

    _selected, report = plan_cache_span(
        config,
        checkpoint,
        optimizer_steps=4,
    )

    assert report["domain_occurrences"] == {"photo": 4, "animation": 4}
    assert report["ending_domain"] == "photo"
    assert report["ending_positions"]["animation"]["batch"] == 6
    assert report["ending_positions"]["photo"]["batch"] == 4


def test_weighted_domain_windows_alternate_and_follow_manifest_ratio():
    counts = {"photo": 41_292, "animation": 56_527}
    windows = weighted_domain_windows(
        counts,
        max_steps=12_228,
        minimum_steps=500,
        maximum_steps=1_000,
        seed=42,
    )

    assert windows == weighted_domain_windows(
        counts,
        max_steps=12_228,
        minimum_steps=500,
        maximum_steps=1_000,
        seed=42,
    )
    assert windows[0].start_step == 0
    assert windows[-1].end_step == 12_228
    assert all(
        left.domain != right.domain
        for left, right in itertools.pairwise(windows)
    )
    assert all(
        500 <= window.steps <= 1_000 for window in windows[:-1]
    )
    animation_steps = sum(
        window.steps for window in windows if window.domain == "animation"
    )
    assert animation_steps / 12_228 == pytest.approx(
        counts["animation"] / sum(counts.values()),
        abs=0.02,
    )


def test_domain_window_schedule_report_restarts_epoch(tmp_path):
    manifest = tmp_path / "train.jsonl"
    _write_manifest(
        manifest,
        [
            *[_row(index, "photo") for index in range(4)],
            *[_row(100 + index, "animation") for index in range(6)],
        ],
    )
    report = domain_window_schedule_report(
        manifest,
        max_steps=1_600,
        minimum_steps=500,
        maximum_steps=1_000,
    )

    assert report["fresh_data_epoch"] is True
    assert report["domain_counts"] == {"photo": 4, "animation": 6}
    assert report["domain_weights"]["photo"] == pytest.approx(0.4)
    assert report["domain_weights"]["animation"] == pytest.approx(0.6)
    assert report["windows"][0]["domain"] == "animation"
    assert report["windows"][1]["domain"] == "photo"


def test_terminal_config_accepts_two_checkpoint_alternating_schedule(tmp_path):
    train_manifest = tmp_path / "train.jsonl"
    eval_manifest = tmp_path / "eval.jsonl"
    rows = [
        *[_row(index, "photo") for index in range(4)],
        *[_row(100 + index, "animation") for index in range(6)],
    ]
    _write_manifest(train_manifest, rows)
    _write_manifest(eval_manifest, rows)
    schedule = domain_window_schedule_report(
        train_manifest,
        max_steps=1_600,
    )
    schedule_path = tmp_path / "schedule.json"
    schedule_path.write_text(json.dumps(schedule))
    photo = tmp_path / "photo.safetensors"
    animation = tmp_path / "animation.safetensors"
    photo.touch()
    animation.touch()
    config = TerminalExpertTrainConfig(
        # Explicit: the field default is an absolute path on one host.
        model_path=None,
        domain="animation",
        train_manifest=str(train_manifest),
        eval_manifest=str(eval_manifest),
        expert_checkpoint=str(animation),
        expert_checkpoints={
            "photo": str(photo),
            "animation": str(animation),
        },
        domain_window_schedule=str(schedule_path),
        output_dir=str(tmp_path / "output"),
        max_steps=1_600,
    )

    config.validate()
    receipt = prepare_run(config, tmp_path / "plan")
    assert receipt["alternating_domain_windows"] is True
    assert receipt["train_examples"] == 10
    assert receipt["eval_examples"] == 10
    assert receipt["resident_experts"] == 1


def test_weighted_domain_windows_reject_impossible_strict_switch_ratio():
    with pytest.raises(ValueError, match="cannot be represented"):
        weighted_domain_windows(
            {"photo": 1, "animation": 99},
            max_steps=2_000,
            minimum_steps=500,
            maximum_steps=1_000,
        )


def test_resident_expert_optimizer_bank_swaps_only_expert_state():
    expert = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.bfloat16))
    shared = torch.nn.Parameter(torch.tensor([3.0], dtype=torch.bfloat16))
    optimizer = FP32MasterAdamW(
        [
            {"params": [expert], "lr": 1e-3, "group_name": "terminal_expert"},
            {"params": [shared], "lr": 5e-4, "group_name": "shared_backbone"},
        ]
    )
    pairs = {
        id(model): master for model, master, _independent in optimizer._model_master_pairs
    }
    expert_master = pairs[id(expert)]
    shared_master = pairs[id(shared)]
    optimizer.state[expert_master] = {
        "step": torch.tensor(7.0),
        "exp_avg": torch.tensor([0.1, 0.2]),
        "exp_avg_sq": torch.tensor([0.3, 0.4]),
    }
    optimizer.state[shared_master] = {
        "step": torch.tensor(11.0),
        "exp_avg": torch.tensor([0.5]),
        "exp_avg_sq": torch.tensor([0.6]),
    }
    bank = ResidentExpertOptimizerBank(optimizer, [expert])

    bank.park("photo")
    with torch.no_grad():
        expert.copy_(torch.tensor([8.0, 9.0], dtype=torch.bfloat16))
    bank.activate("animation", initialize_from_model=True)
    optimizer.state[expert_master] = {
        "step": torch.tensor(2.0),
        "exp_avg": torch.tensor([0.8, 0.9]),
        "exp_avg_sq": torch.tensor([1.0, 1.1]),
    }
    bank.park("animation")
    assert bank.activate("photo") is True

    assert torch.equal(expert.float(), torch.tensor([1.0, 2.0]))
    assert optimizer.state[expert_master]["step"].item() == 7
    assert torch.equal(
        optimizer.state[expert_master]["exp_avg"], torch.tensor([0.1, 0.2])
    )
    assert optimizer.state[shared_master]["step"].item() == 11
    assert torch.equal(
        optimizer.state[shared_master]["exp_avg"], torch.tensor([0.5])
    )
    assert bank.inactive_domains == ("animation",)


def test_resident_expert_optimizer_bank_round_trip():
    expert = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.bfloat16))
    optimizer = FP32MasterAdamW([{"params": [expert], "lr": 1e-3}])
    bank = ResidentExpertOptimizerBank(optimizer, [expert])
    bank.park("animation")
    state = bank.state_dict()

    restored = ResidentExpertOptimizerBank(optimizer, [expert])
    restored.load_state_dict(state)
    assert restored.inactive_domains == ("animation",)


def test_resident_expert_optimizer_bank_exchanges_same_device_state():
    expert = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = FP32MasterAdamW([{"params": [expert], "lr": 1e-3}])
    master = optimizer._model_master_pairs[0][1]
    optimizer.state[master] = {
        "step": torch.tensor(7.0),
        "exp_avg": torch.tensor([0.1]),
        "exp_avg_sq": torch.tensor([0.2]),
    }
    bank = ResidentExpertOptimizerBank(
        optimizer,
        [expert],
        storage_device="cpu",
    )
    bank.park("photo")

    with torch.no_grad():
        expert.fill_(9.0)
        master.fill_(9.0)
    optimizer.state[master] = {
        "step": torch.tensor(3.0),
        "exp_avg": torch.tensor([0.8]),
        "exp_avg_sq": torch.tensor([0.9]),
    }
    bank.exchange("animation", "photo")

    assert expert.item() == 1.0
    assert master.item() == 1.0
    assert optimizer.state[master]["step"].item() == 7
    assert bank.inactive_domains == ("animation",)

    bank.exchange("photo", "animation")
    assert expert.item() == 9.0
    assert master.item() == 9.0
    assert optimizer.state[master]["step"].item() == 3
    assert bank.inactive_domains == ("photo",)


def test_hungarian_assignment_finds_nonidentity_minimum():
    assignment = minimum_cost_assignment(
        [
            [10.0, 1.0, 8.0],
            [1.0, 10.0, 8.0],
            [8.0, 8.0, 1.0],
        ]
    )
    assert assignment == [1, 0, 2]


def test_immiscible_assignment_spans_separate_microbatch_flows():
    first_clean = torch.tensor([[[-1.0], [-1.0]]])
    second_clean = torch.tensor([[[1.0], [1.0]]])
    first_noise = torch.tensor([[[0.9], [0.9]]])
    second_noise = torch.tensor([[[-0.9], [-0.9]]])
    flows = [
        {
            "clean": first_clean,
            "noise": first_noise,
            "timesteps": torch.tensor([0.25]),
            "image_lens": [2],
        },
        {
            "clean": second_clean,
            "noise": second_noise,
            "timesteps": torch.tensor([0.75]),
            "image_lens": [2],
        },
    ]
    report = apply_immiscible_noise_assignment(flows)

    assert report["examples"] == 2
    assert report["reassigned_examples"] == 2
    assert report["cost_reduction"] > 0
    assert torch.equal(flows[0]["noise"], second_noise)
    assert torch.equal(flows[1]["noise"], first_noise)
    assert torch.allclose(
        flows[0]["img"],
        0.75 * first_clean + 0.25 * second_noise,
    )
    assert torch.equal(flows[0]["velocity"], second_noise - first_clean)


def test_vae_repa_projection_aligns_packed_clean_features(tmp_path):
    objective = VAERepresentationAlignment(
        model_dim=4,
        vae_dim=2,
        hidden_dim=3,
        smooth_l1_beta=0.5,
    )
    hidden = torch.randn(1, 5, 4, requires_grad=True)
    target = torch.randn(1, 5, 2)
    loss = objective(hidden, target)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert hidden.grad is not None
    assert all(parameter.grad is not None for parameter in objective.parameters())

    checkpoint = save_repa_projection(objective, tmp_path / "repa.safetensors")
    restored = VAERepresentationAlignment(4, 2, hidden_dim=3)
    assert load_repa_projection(restored, checkpoint) == len(
        objective.state_dict()
    )
    assert all(
        torch.equal(original, loaded)
        for original, loaded in zip(
            objective.state_dict().values(),
            restored.state_dict().values(),
            strict=True,
        )
    )


def test_vae_repa_is_per_example_normalized_and_outlier_bounded():
    objective = VAERepresentationAlignment(
        model_dim=4,
        vae_dim=2,
        hidden_dim=3,
        student_normalization="token_rms",
        per_example_loss_cap=5.0,
    )
    hidden = torch.zeros(1, 4, 4, requires_grad=True)
    hidden.data[0, 0, 0] = 1.0e20
    target = torch.zeros(1, 4, 2)
    target[0, 0] = 1.0e6
    loss = objective(hidden, target, image_lens=[1, 3])
    loss.backward()

    assert torch.isfinite(loss)
    assert 0 <= loss <= 5.0
    assert objective.last_metrics["raw_loss_max"] > loss
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()


def test_vae_repa_averages_native_resolution_examples_equally():
    objective = VAERepresentationAlignment(
        model_dim=2,
        vae_dim=1,
        hidden_dim=2,
        student_normalization="none",
        per_example_loss_cap=None,
        smooth_l1_beta=1.0,
    )
    with torch.no_grad():
        for parameter in objective.parameters():
            parameter.zero_()
    hidden = torch.zeros(1, 4, 2)
    target = torch.tensor([[[2.0], [0.0], [0.0], [0.0]]])

    loss = objective(hidden, target, image_lens=[1, 3])

    assert float(loss.detach()) == pytest.approx(0.75)


def test_repa_tap_before_looping_cannot_steer_loop_gates():
    model_weight = torch.nn.Parameter(torch.tensor(2.0))
    loop_gate = torch.nn.Parameter(torch.tensor(3.0))
    base = model_weight.square() + loop_gate.square()
    # A pre-route REPA tap shares the upstream model path but has no graph edge
    # to a downstream loop gate.
    repa = 5.0 * model_weight

    total = _backward_training_objective(base, repa)

    assert float(total.detach()) == pytest.approx(23.0)
    assert float(model_weight.grad) == pytest.approx(9.0)
    assert float(loop_gate.grad) == pytest.approx(6.0)


def test_best_checkpoint_survives_rolling_retention(tmp_path):
    source = tmp_path / "checkpoint-00000100"
    source.mkdir()
    (source / "checkpoint.json").write_text("{}")
    payload = source / "trainer_state.pt"
    payload.write_bytes(b"best-state")

    assert _best_evaluation_improved(
        tmp_path,
        {"eval/primary_loss": 1.5},
    )
    promoted = _promote_best_checkpoint(
        source,
        tmp_path,
        step=100,
        loss=1.5,
    )
    source.rename(tmp_path / "rotated-away")

    assert (promoted / "trainer_state.pt").read_bytes() == b"best-state"
    assert not _best_evaluation_improved(
        tmp_path,
        {"eval/primary_loss": 1.6},
    )
    assert _best_evaluation_improved(
        tmp_path,
        {"eval/primary_loss": 1.4},
    )


def test_flow_min_snr_is_translated_into_velocity_space():
    timesteps = torch.tensor([0.1, 0.5, 0.9])
    hard = flow_min_snr_weights(
        timesteps,
        weighting="min_snr",
        gamma=5.0,
    )
    soft = flow_min_snr_weights(
        timesteps,
        weighting="soft_min_snr",
        gamma=5.0,
    )

    assert torch.allclose(hard, torch.tensor([0.05, 0.25, 0.01]))
    expected_soft = (
        5.0
        * timesteps.square()
        * (1.0 - timesteps).square()
        / ((1.0 - timesteps).square() + 5.0 * timesteps.square())
    )
    assert torch.allclose(soft, expected_soft)
    assert soft[0] < hard[0]
    assert soft[1] < hard[1]


def test_flow_weights_normalize_across_accumulated_microbatches():
    batches, report = effective_flow_loss_weights(
        [torch.tensor([0.1]), torch.tensor([0.5]), torch.tensor([0.9])],
        weighting="soft_min_snr",
        gamma=5.0,
        normalize=True,
    )
    combined = torch.cat(batches)

    assert combined.mean() == pytest.approx(1.0)
    assert report["normalized_mean"] == pytest.approx(1.0)
    assert combined[1] > combined[0] > combined[2]


def test_weighted_flow_loss_preserves_effective_batch_weighting():
    prediction = torch.tensor([[[1.0], [1.0], [3.0], [3.0]]])
    target = torch.zeros_like(prediction)
    per_example = rectified_flow_loss_per_example(
        prediction,
        target,
        [2, 2],
    )
    contribution, observed = weighted_rectified_flow_loss(
        prediction,
        target,
        [2, 2],
        torch.tensor([0.5, 1.5]),
        effective_example_count=2,
    )

    assert torch.equal(per_example, torch.tensor([1.0, 9.0]))
    assert observed == pytest.approx(5.0)
    assert contribution == pytest.approx((0.5 * 1.0 + 1.5 * 9.0) / 2.0)

    # The same result must survive microbatch-one accumulation. Normalizing
    # inside each microbatch would incorrectly cancel both weights.
    first, _ = weighted_rectified_flow_loss(
        prediction[:, :2],
        target[:, :2],
        [2],
        torch.tensor([0.5]),
        effective_example_count=2,
    )
    second, _ = weighted_rectified_flow_loss(
        prediction[:, 2:],
        target[:, 2:],
        [2],
        torch.tensor([1.5]),
        effective_example_count=2,
    )
    assert first + second == pytest.approx(contribution.item())


def test_velocity_direction_loss_is_channel_cosine_and_image_balanced():
    prediction = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]],
        requires_grad=True,
    )
    target = torch.tensor(
        [[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]]
    )

    per_example = velocity_direction_loss_per_example(
        prediction,
        target,
        [1, 2],
    )
    contribution, observed = weighted_velocity_direction_loss(
        prediction,
        target,
        [1, 2],
        torch.tensor([0.5, 1.5]),
        effective_example_count=2,
    )
    contribution.backward()

    assert torch.allclose(per_example, torch.tensor([0.0, 1.5]))
    assert observed == pytest.approx(0.75)
    assert float(contribution.detach()) == pytest.approx(1.125)
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_default_model_path_is_portable(tmp_path, monkeypatch):
    """The default must not be one machine's absolute path.

    Regression for the defect that surfaced as five CI failures the first time
    the suite ran off the maintainer's host: the dataclass default was a bare
    absolute path, so every config built without an explicit model_path
    inherited it and failed validation on a path the caller never chose.
    """
    from rwkv_lab import mage_flow_terminal_train as trainer

    monkeypatch.delenv(trainer.MAGE_FLOW_BASE_LOCAL_PATH_ENV, raising=False)

    # A host that does not cache the weights resolves to None, not to a path
    # that happens to exist only somewhere else. Asserted on a CONSTRUCTED
    # config, not on the resolver: the defect was in the dataclass field
    # default, so a test that only calls the helper passes even when the field
    # still hardcodes an absolute path.
    monkeypatch.setattr(
        trainer, "MAGE_FLOW_BASE_HISTORICAL_PATH", str(tmp_path / "absent")
    )
    assert trainer.default_model_path() is None

    unconfigured = trainer.TerminalExpertTrainConfig(
        domain="photo",
        train_manifest=str(tmp_path / "train.jsonl"),
        expert_checkpoint=str(tmp_path / "expert.pt"),
        output_dir=str(tmp_path / "out"),
    )
    assert unconfigured.model_path is None, (
        "a config built without model_path inherited a host-specific path"
    )

    # An explicit environment override wins over everything.
    monkeypatch.setenv(trainer.MAGE_FLOW_BASE_LOCAL_PATH_ENV, str(tmp_path))
    assert trainer.default_model_path() == str(tmp_path)
    monkeypatch.delenv(trainer.MAGE_FLOW_BASE_LOCAL_PATH_ENV)

    # A host that DOES cache them at the historical path keeps using it, so
    # the maintainer's existing runs are unchanged.
    cached = tmp_path / "Mage-Flow-Base"
    cached.mkdir()
    monkeypatch.setattr(trainer, "MAGE_FLOW_BASE_HISTORICAL_PATH", str(cached))
    assert trainer.default_model_path() == str(cached)


def test_missing_explicit_model_path_names_the_configuration(tmp_path):
    """A supplied-but-missing path is still an error, with an actionable message."""
    from rwkv_lab import mage_flow_terminal_train as trainer

    config = trainer.TerminalExpertTrainConfig(
        domain="photo",
        train_manifest=str(tmp_path / "train.jsonl"),
        expert_checkpoint=str(tmp_path / "expert.pt"),
        output_dir=str(tmp_path / "out"),
        model_path=str(tmp_path / "not-here"),
    )
    with pytest.raises(ValueError) as failure:
        config.validate()
    message = str(failure.value)
    assert "model_path" in message
    assert trainer.MAGE_FLOW_BASE_LOCAL_PATH_ENV in message
