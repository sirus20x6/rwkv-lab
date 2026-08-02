import pytest
import torch

from rwkv_lab.training_components import (
    PowerCoolConfiguration,
    ScheduleImplementation,
    build_registered_schedule,
)
from rwkv_lab.trainvm_adapters.qwen_controls import (
    QwenControlError,
    QwenMutableControls,
    lower_initial_qwen_controls,
)
from rwkv_lab.trainvm_worker import WorkerControlRuntime


class EmptySession:
    @staticmethod
    def poll_commands(maximum=None):
        return ()

    @staticmethod
    def acknowledge_controls(*_args, **_kwargs):
        raise AssertionError("no command should be acknowledged")


def schedule(base: float = 5.0e-5):
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=base)
    value = build_registered_schedule(
        ScheduleImplementation.POWERCOOL_V1,
        optimizer,
        PowerCoolConfiguration(warmup_steps=1, max_steps=10, minimum_ratio=0.1),
    )
    return optimizer, value


def test_qwen_initial_controls_preserve_frozen_config_and_schedule_ratio():
    runtime = WorkerControlRuntime(
        EmptySession(), {"learning_rate": 1.0e-5, "eval_every": 50}, 3
    )
    # SimpleNamespace is intentionally replaced by a tiny frozen dataclass-like
    # object for the actual lowering path.
    from rwkv_lab.qwen_ao3_cpt import QwenAO3Config

    frozen = QwenAO3Config(
        model_dir="model",
        train_pack_dir="train",
        eval_pack_dir="eval",
        run_dir="run",
    )
    lowered = lower_initial_qwen_controls(frozen, runtime)
    assert frozen.learning_rate == pytest.approx(5.0e-5)
    assert lowered.learning_rate == pytest.approx(1.0e-5)
    assert lowered.min_learning_rate == pytest.approx(1.0e-6)
    assert lowered.eval_every == 50


def test_qwen_live_controls_rebase_schedule_without_resetting_phase():
    optimizer, scheduler = schedule()
    runtime = WorkerControlRuntime(
        EmptySession(), {"learning_rate": 1.0e-5, "eval_every": 50}, 3
    )
    controls = QwenMutableControls(
        scheduler,
        runtime,
        learning_rate=1.0e-5,
        eval_every=50,
        constructed_base_learning_rate=5.0e-5,
    )
    assert scheduler.base_lrs == pytest.approx([1.0e-5])
    old_epoch = scheduler.last_epoch

    controls.apply(
        {"learning_rate": 2.0e-6, "eval_every": 20},
        {"learning_rate": 2.0e-6, "eval_every": 20},
    )

    assert controls.learning_rate == pytest.approx(2.0e-6)
    assert controls.eval_every == 20
    assert scheduler.last_epoch == old_epoch
    assert scheduler.base_lrs == pytest.approx([2.0e-6])
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2.0e-6)


def test_qwen_controls_reject_unknown_and_invalid_values_before_mutation():
    _optimizer, scheduler = schedule()
    runtime = WorkerControlRuntime(
        EmptySession(), {"learning_rate": 5.0e-5, "eval_every": 500}, 0
    )
    controls = QwenMutableControls(
        scheduler,
        runtime,
        learning_rate=5.0e-5,
        eval_every=500,
        constructed_base_learning_rate=5.0e-5,
    )

    with pytest.raises(QwenControlError, match="positive integer"):
        controls.apply(
            {"learning_rate": 1.0e-5, "eval_every": 0},
            {"learning_rate": 1.0e-5, "eval_every": 0},
        )
    assert controls.learning_rate == pytest.approx(5.0e-5)
    assert scheduler.base_lrs == pytest.approx([5.0e-5])

    with pytest.raises(QwenControlError, match="unknown key"):
        QwenMutableControls(
            scheduler,
            WorkerControlRuntime(EmptySession(), {"hidden_switch": True}, 0),
            learning_rate=5.0e-5,
            eval_every=500,
            constructed_base_learning_rate=5.0e-5,
        )
