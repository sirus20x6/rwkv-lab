from types import SimpleNamespace

import pytest
import torch

from rwkv_lab.training_components import (
    LinearWarmupCosineConfiguration,
    ScheduleImplementation,
    build_registered_schedule,
)
from rwkv_lab.trainvm_adapters.mageflow_controls import (
    MageFlowControlError,
    MageFlowMutableControls,
    lower_initial_mageflow_controls,
)
from rwkv_lab.trainvm_worker import WorkerControlRuntime


class EmptySession:
    def poll_commands(self, maximum=None):
        return ()

    def acknowledge_controls(self, *_args, **_kwargs):
        raise AssertionError("no command should be acknowledged")


def fixture(initial=None):
    config = SimpleNamespace(
        learning_rate=1.0e-4,
        eval_every=500,
        caption_dropout=0.1,
        mixed_precision="bf16",
    )
    expert = torch.nn.Parameter(torch.tensor([1.0]))
    backbone = torch.nn.Parameter(torch.tensor([2.0]))
    optimizer = torch.optim.AdamW(
        [
            {"params": [expert], "lr": 1.0e-4},
            {"params": [backbone], "lr": 5.0e-5},
        ]
    )
    scheduler = build_registered_schedule(
        ScheduleImplementation.LINEAR_WARMUP_COSINE_V1,
        optimizer,
        LinearWarmupCosineConfiguration(
            warmup_steps=0, max_steps=1000, minimum_ratio=0.1
        ),
    )
    runtime = WorkerControlRuntime(
        EmptySession(),
        initial
        or {
            "learning_rate": 1.0e-4,
            "eval_every": 500,
            "caption_dropout": 0.1,
            "mixed_precision": "bf16",
        },
        0,
    )
    return config, optimizer, scheduler, runtime


def test_mageflow_controls_update_config_and_rebase_group_rates_atomically():
    config, optimizer, scheduler, runtime = fixture()
    adapter = MageFlowMutableControls(config, scheduler, runtime)

    adapter.apply(
        {
            "learning_rate": 2.0e-5,
            "eval_every": 20,
            "caption_dropout": 0.25,
            "mixed_precision": "bf16",
        },
        {
            "learning_rate": 2.0e-5,
            "eval_every": 20,
            "caption_dropout": 0.25,
        },
    )

    assert config.learning_rate == pytest.approx(2.0e-5)
    assert config.eval_every == 20
    assert config.caption_dropout == pytest.approx(0.25)
    assert scheduler.base_lrs == pytest.approx([2.0e-5, 1.0e-5])
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(
        [2.0e-5, 1.0e-5]
    )


def test_mageflow_control_validation_precedes_every_mutation():
    config, optimizer, scheduler, runtime = fixture()
    adapter = MageFlowMutableControls(config, scheduler, runtime)
    original_rates = [group["lr"] for group in optimizer.param_groups]

    with pytest.raises(MageFlowControlError, match="caption_dropout"):
        adapter.apply(
            {
                "learning_rate": 2.0e-5,
                "eval_every": 500,
                "caption_dropout": 1.0,
                "mixed_precision": "bf16",
            },
            {"learning_rate": 2.0e-5, "caption_dropout": 1.0},
        )

    assert config.learning_rate == pytest.approx(1.0e-4)
    assert config.caption_dropout == pytest.approx(0.1)
    assert [group["lr"] for group in optimizer.param_groups] == original_rates


def test_restart_only_precision_and_unknown_or_mismatched_catalogs_fail_closed():
    config, _optimizer, scheduler, runtime = fixture()
    adapter = MageFlowMutableControls(config, scheduler, runtime)
    with pytest.raises(MageFlowControlError, match="replacement"):
        adapter.apply(
            {
                "learning_rate": 1.0e-4,
                "eval_every": 500,
                "caption_dropout": 0.1,
                "mixed_precision": "fp16",
            },
            {"mixed_precision": "fp16"},
        )

    with pytest.raises(MageFlowControlError, match="unknown key"):
        MageFlowMutableControls(
            config,
            scheduler,
            WorkerControlRuntime(EmptySession(), {"secret_switch": True}, 0),
        )
    with pytest.raises(MageFlowControlError, match="disagree"):
        MageFlowMutableControls(
            config,
            scheduler,
            WorkerControlRuntime(EmptySession(), {"learning_rate": 3.0e-4}, 0),
        )


def test_initial_authority_controls_lower_before_runtime_construction():
    config, _optimizer, _scheduler, _runtime = fixture()
    runtime = WorkerControlRuntime(
        EmptySession(),
        {
            "learning_rate": 3.0e-5,
            "eval_every": 40,
            "caption_dropout": 0.2,
            "mixed_precision": "bf16",
        },
        7,
    )

    lower_initial_mageflow_controls(config, runtime)

    assert config.learning_rate == pytest.approx(3.0e-5)
    assert config.eval_every == 40
    assert config.caption_dropout == pytest.approx(0.2)


def test_cached_conditioning_cannot_enable_a_null_embedding_that_was_not_built():
    config, _optimizer, scheduler, runtime = fixture(
        {
            "learning_rate": 1.0e-4,
            "eval_every": 500,
            "caption_dropout": 0.0,
            "mixed_precision": "bf16",
        }
    )
    config.caption_dropout = 0.0
    config.encoder_cache_mode = "read_only"
    adapter = MageFlowMutableControls(config, scheduler, runtime)

    with pytest.raises(MageFlowControlError, match="cached null condition"):
        adapter.apply(
            {
                "learning_rate": 1.0e-4,
                "eval_every": 500,
                "caption_dropout": 0.2,
                "mixed_precision": "bf16",
            },
            {"caption_dropout": 0.2},
        )
