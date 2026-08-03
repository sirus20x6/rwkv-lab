from __future__ import annotations

import pytest

from rwkv_lab.trainvm_adapters.metric_decision import ScalarMetricDecisionConfig


def test_scalar_metric_decision_config_is_closed_and_canonical() -> None:
    config = ScalarMetricDecisionConfig(
        metric="eval.loss",
        direction="minimize",
        left_subject="moonvit",
        right_subject="compressor",
        absolute_tolerance=1,
    )

    assert config.absolute_tolerance == 1.0


@pytest.mark.parametrize(
    "change",
    [
        {"metric": "eval loss"},
        {"direction": "smallest"},
        {"right_subject": "moonvit"},
        {"absolute_tolerance": -1.0},
        {"absolute_tolerance": float("nan")},
    ],
)
def test_scalar_metric_decision_config_rejects_open_or_ambiguous_values(
    change,
) -> None:
    values = {
        "metric": "eval.loss",
        "direction": "minimize",
        "left_subject": "moonvit",
        "right_subject": "compressor",
        "absolute_tolerance": 0.0,
    }
    values.update(change)

    with pytest.raises(ValueError):
        ScalarMetricDecisionConfig(**values)
