from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from rwkv_lab.trainvm_adapters import AdapterComponentError, WorkerTrainingComponents
from rwkv_lab.trainvm_worker import load_resolved_training_composition


def canonical(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def composition(*, frozen_evaluation: bool = False,
                no_pipeline: bool = False,
                train_selection_for_evaluation: bool = False):
    root = Path(__file__).resolve().parents[1]
    registry = json.loads(
        (root / "docs/experiment-vm/examples/training-components.v1.json").read_text(
            encoding="utf-8"
        )
    )
    requested = {
        "activation": (
            "squared_relu",
            {},
        ),
        "curriculum": (
            "context_length",
            {
                "maximum_sequence_length": 512,
                "base_batch_size": 16,
                "stages": "0:128,0.5:512",
            },
        ),
        "optimizer": (
            "torch_adamw_no_decay",
            {
                "learning_rate": 1e-3,
                "beta1": 0.9,
                "beta2": 0.999,
                "epsilon": 1e-8,
                "foreach": True,
                "fused": False,
            },
        ),
        "weight_decay": (
            "constant",
            {"weight_decay": 0.01},
        ),
        "learning_rate": (
            "linear_warmup_cosine",
            {"warmup_steps": 0, "max_steps": 4, "minimum_ratio": 0.1},
        ),
        "gradient_clipping": (
            "global_norm",
            {
                "max_norm": 1.0,
                "norm_type": 2.0,
                "error_if_nonfinite": True,
            },
        ),
        "gradient_accumulation": (
            "fixed",
            {"microbatches_per_optimizer_step": 4},
        ),
        "objective": (
            "linear_head_cross_entropy",
            {"chunk_size": 2, "prefer_fused": False},
        ),
        "normalization": (
            "layer_norm",
            {"epsilon": 1e-5},
        ),
        "precision": (
            "bf16_parameters_fp32_reductions",
            {
                "parameter_dtype": "bfloat16",
                "compute_dtype": "bfloat16",
                "reduction_dtype": "float32",
                "gradient_scaling": False,
            },
        ),
        "split": (
            "deterministic_holdout",
            {"seed": 7, "held_out_count": 2, "selection": "train"},
        ),
        "evaluation_split": (
            "frozen_named" if frozen_evaluation else "deterministic_holdout",
            (
                {"selection": "validation"}
                if frozen_evaluation
                else {"seed": 7, "held_out_count": 2, "selection": "held_out"}
            ),
        ),
        "evaluator": (
            "scalar_loss",
            {
                "split_slot": "evaluation_split",
                "metrics": ["loss"],
                "reduction": "mean",
                "maximum_examples": 0,
            },
        ),
        "evaluation_schedule": (
            "launch_gate_periodic",
            {
                "launch_gate_examples": 2,
                "full_step_zero": True,
                "qualitative_every_steps": 25,
                "full_every_steps": 100,
                "defer_full_scalar": True,
                "final": True,
            },
        ),
        "qualitative_samples": (
            "fixed_held_out",
            {
                "identity_field": "sample_id",
                "identities_digest": "sha256:" + "a" * 64,
                "selector_digest": "sha256:" + "b" * 64,
                "sample_count": 2,
            },
        ),
        "artifact_renderer": (
            "evidence_envelope",
            {"modality": "text", "schema": "trainvm.eval-evidence.v1"},
        ),
        "checkpoint_policy": (
            "atomic_retained",
            {
                "every_steps": 100,
                "keep_last": 2,
                "keep_best": 0,
                "publish_final": True,
                "resume_grade": "exact",
            },
        ),
    }
    if frozen_evaluation:
        requested = {
            slot: requested[slot] for slot in ("evaluation_split", "evaluator")
        }
    if train_selection_for_evaluation:
        requested["evaluation_split"] = (
            "deterministic_holdout",
            {"seed": 7, "held_out_count": 2, "selection": "train"},
        )
    if no_pipeline:
        # What every family whose evaluation corpus is named in its own trainer
        # configuration resolves to: an evaluator, and no split selector for it
        # to point at. `validate_evaluation_checkpoint_relationships` REQUIRES
        # an empty `split_slot` here -- a split-selector view is part of the
        # declarative data pipeline and is unresolvable without one.
        requested = {
            "evaluator": (
                "scalar_loss",
                {
                    "split_slot": "",
                    "metrics": ["eval.loss"],
                    "reduction": "weighted_mean",
                    "maximum_examples": 0,
                },
            ),
        }
    components = {}
    for slot, (name, configuration) in requested.items():
        category = {
            "evaluation_split": "split_selector",
            "learning_rate": "learning_rate_schedule",
            "qualitative_samples": "qualitative_sample",
            "split": "split_selector",
            "weight_decay": "weight_decay_schedule",
        }.get(slot, slot)
        descriptor = next(
            item
            for item in registry["components"]
            if item["key"]["category"] == category and item["key"]["name"] == name
        )
        components[slot] = {
            "configuration": configuration,
            "descriptor": descriptor,
            "descriptor_digest": digest(canonical(descriptor)),
        }
    body = {
        "api_version": "trainvm.resolved-training-composition/v1",
        "components": components,
        "model_family": (
            "transformer" if (frozen_evaluation or no_pipeline) else "rwkv"
        ),
        "registry_digest": digest(canonical(registry)),
    }
    return load_resolved_training_composition(
        {**body, "composition_digest": digest(canonical(body))}
    )


def test_worker_component_bridge_builds_optimizer_and_schedule() -> None:
    runtime = WorkerTrainingComponents(composition(), "rwkv")
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = runtime.optimizer([parameter])
    decay = runtime.weight_decay_schedule(optimizer)
    schedule = runtime.learning_rate_schedule(optimizer)
    parameter.grad = torch.tensor([4.0])
    gradient_norm = runtime.gradient_clipping([parameter])
    accumulation = runtime.gradient_accumulation()
    objective = runtime.objective()
    precision = runtime.precision()
    activation = runtime.activation()
    curriculum = runtime.curriculum()
    normalization = runtime.normalization()
    evaluator = runtime.evaluator()
    evaluation_schedule = runtime.evaluation_schedule()
    qualitative_samples = runtime.qualitative_samples()
    artifact_renderer = runtime.artifact_renderer()
    checkpoint_policy = runtime.checkpoint_policy()

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.01)
    assert decay.state_dict() == {}
    assert isinstance(schedule, torch.optim.lr_scheduler.LRScheduler)
    assert schedule.get_last_lr() == pytest.approx([1e-3])
    assert gradient_norm == pytest.approx(4.0)
    assert parameter.grad == pytest.approx(torch.tensor([1.0]))
    assert accumulation.microbatches_per_optimizer_step == 4
    assert accumulation.scale_loss(torch.tensor(8.0)) == pytest.approx(2.0)
    hidden = torch.randn(1, 2, 3)
    head = torch.nn.Linear(3, 5)
    labels = torch.tensor([[1, 2]])
    torch.testing.assert_close(
        objective(hidden, head, labels),
        torch.nn.functional.cross_entropy(
            head(hidden).reshape(-1, 5), labels.reshape(-1)
        ),
    )
    assert precision.parameter_dtype is torch.bfloat16
    assert precision.reduce(torch.ones(1, dtype=torch.bfloat16)).dtype is torch.float32
    torch.testing.assert_close(
        activation(torch.tensor([-2.0, 3.0])), torch.tensor([0.0, 9.0])
    )
    assert normalization(3).eps == pytest.approx(1e-5)
    assert curriculum.for_step(0, 100) == (128, 64)
    assert curriculum.for_step(50, 100) == (512, 16)
    assert evaluator.reduce((2.0, 4.0)) == pytest.approx(3.0)
    assert evaluation_schedule.for_step(0).launch_gate
    assert qualitative_samples.configuration.sample_count == 2
    assert artifact_renderer.configuration.modality == "text"
    assert checkpoint_policy.due(100)
    assert runtime.evidence()["optimizer"]["category"] == "optimizer"
    assert (
        runtime.require_implementation(
            "optimizer",
            category="optimizer",
            allowed=frozenset({"rwkv_lab.optimizer.torch_adamw_no_decay.v2"}),
        )
        == "rwkv_lab.optimizer.torch_adamw_no_decay.v2"
    )
    with pytest.raises(AdapterComponentError, match="outside the adapter allowlist"):
        runtime.require_implementation(
            "optimizer",
            category="optimizer",
            allowed=frozenset({"rwkv_lab.optimizer.torch_sparse_adam.v1"}),
        )


def test_worker_component_bridge_rejects_family_and_slot_category_confusion() -> None:
    resolved = composition()
    with pytest.raises(AdapterComponentError, match="different model family"):
        WorkerTrainingComponents(resolved, "transformer")

    runtime = WorkerTrainingComponents(resolved, "rwkv")
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    with pytest.raises(ValueError, match="not 'optimizer'"):
        runtime.optimizer([parameter], slot="learning_rate")


def test_worker_evaluator_bridge_accepts_production_frozen_validation_split() -> None:
    runtime = WorkerTrainingComponents(
        composition(frozen_evaluation=True), "transformer"
    )

    evaluator = runtime.evaluator()

    assert evaluator.configuration.split_slot == "evaluation_split"
    assert evaluator.reduce((2.0, 4.0)) == pytest.approx(3.0)


def test_worker_evaluator_bridge_accepts_an_adapter_owned_held_out_corpus() -> None:
    """The registry's no-pipeline branch, mirrored by the Python accessor.

    Twelve of the registry's stateful routes -- the four vision ones and the
    eight Transformer MLA ones -- name their evaluation corpus in the adapter's
    own configuration rather than through a `data_source`, and
    `validate_evaluation_checkpoint_relationships` requires their evaluator to
    name an EMPTY `split_slot` in that case. Resolving `""` as a slot raises,
    so without this branch the accessor refuses every one of those routes --
    and the accessor is what reads the evaluator provenance an armed route must
    publish, so the refusal would land after a model load and before any
    evidence.
    """

    runtime = WorkerTrainingComponents(composition(no_pipeline=True), "transformer")

    evaluator = runtime.evaluator()

    assert evaluator.configuration.split_slot == ""
    assert tuple(evaluator.configuration.metrics) == ("eval.loss",)


def test_worker_evaluator_bridge_still_refuses_a_split_that_is_not_held_out() -> None:
    """The relaxation above must not weaken the case it does not cover."""

    runtime = WorkerTrainingComponents(
        composition(train_selection_for_evaluation=True), "rwkv"
    )
    with pytest.raises(
        AdapterComponentError, match="does not select its validation partition"
    ):
        runtime.evaluator()


def test_worker_optimizer_and_schedule_resume_the_same_next_update() -> None:
    runtime = WorkerTrainingComponents(composition(), "rwkv")
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = runtime.optimizer([parameter])
    schedule = runtime.learning_rate_schedule(optimizer)
    for _ in range(2):
        parameter.grad = torch.tensor([0.25])
        optimizer.step()
        schedule.step()

    resumed_parameter = torch.nn.Parameter(parameter.detach().clone())
    resumed_optimizer = runtime.optimizer([resumed_parameter])
    resumed_schedule = runtime.learning_rate_schedule(resumed_optimizer)
    resumed_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    resumed_schedule.load_state_dict(copy.deepcopy(schedule.state_dict()))
    assert resumed_schedule.last_epoch == schedule.last_epoch
    assert resumed_schedule.get_last_lr() == pytest.approx(schedule.get_last_lr())

    parameter.grad = torch.tensor([0.5])
    resumed_parameter.grad = torch.tensor([0.5])
    optimizer.step()
    schedule.step()
    resumed_optimizer.step()
    resumed_schedule.step()
    assert resumed_parameter.detach() == pytest.approx(parameter.detach())
    assert resumed_schedule.get_last_lr() == pytest.approx(schedule.get_last_lr())
