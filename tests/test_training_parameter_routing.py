import pytest
import torch

from rwkv_lab.training_parameter_routing import (
    ParameterRoute,
    route_trainable_parameters,
)


def _ids(*parameters: torch.nn.Parameter) -> frozenset[int]:
    return frozenset(id(parameter) for parameter in parameters)


def test_routes_every_trainable_tensor_once_and_preserves_aliases():
    expert = torch.nn.Parameter(torch.ones(2))
    backbone = torch.nn.Parameter(torch.ones(3))
    frozen = torch.nn.Parameter(torch.ones(4), requires_grad=False)
    result = route_trainable_parameters(
        [
            ("expert.weight", expert),
            ("expert.alias", expert),
            ("backbone.weight", backbone),
            ("frozen.weight", frozen),
        ],
        [
            ParameterRoute("expert", 1.0, _ids(expert), required=True),
            ParameterRoute("shared_backbone", 0.5, _ids(backbone, frozen)),
        ],
        base_learning_rate=2.0e-5,
    )
    assert [group["group_name"] for group in result.groups] == [
        "expert",
        "shared_backbone",
    ]
    assert [group["lr"] for group in result.groups] == pytest.approx([2.0e-5, 1.0e-5])
    assert result.report["trainable_tensor_count"] == 2
    assert result.report["trainable_parameter_count"] == 5
    assert result.report["frozen_tensor_count"] == 1
    assert result.report["aliases"] == [
        {"canonical_name": "expert.weight", "aliases": ["expert.alias"]}
    ]


def test_unclaimed_and_overlapping_trainable_parameters_fail_closed():
    first = torch.nn.Parameter(torch.ones(1))
    second = torch.nn.Parameter(torch.ones(1))
    with pytest.raises(RuntimeError, match="unclaimed"):
        route_trainable_parameters(
            [("first", first), ("second", second)],
            [ParameterRoute("expert", 1.0, _ids(first))],
            base_learning_rate=1.0e-4,
        )
    with pytest.raises(RuntimeError, match="overlaps"):
        route_trainable_parameters(
            [("first", first)],
            [
                ParameterRoute("expert", 1.0, _ids(first)),
                ParameterRoute("backbone", 0.5, _ids(first)),
            ],
            base_learning_rate=1.0e-4,
        )


def test_required_empty_route_and_invalid_contracts_fail_closed():
    frozen = torch.nn.Parameter(torch.ones(1), requires_grad=False)
    trainable = torch.nn.Parameter(torch.ones(1))
    with pytest.raises(RuntimeError, match="required parameter route"):
        route_trainable_parameters(
            [("frozen", frozen), ("trainable", trainable)],
            [
                ParameterRoute("expert", 1.0, _ids(frozen), required=True),
                ParameterRoute("backbone", 1.0, _ids(trainable)),
            ],
            base_learning_rate=1.0e-4,
        )
    with pytest.raises(ValueError, match="names must be unique"):
        route_trainable_parameters(
            [("trainable", trainable)],
            [
                ParameterRoute("same", 1.0, _ids(trainable)),
                ParameterRoute("same", 1.0, frozenset()),
            ],
            base_learning_rate=1.0e-4,
        )
    with pytest.raises(ValueError, match="base learning rate"):
        route_trainable_parameters(
            [("trainable", trainable)],
            [ParameterRoute("trainable", 1.0, _ids(trainable))],
            base_learning_rate=float("nan"),
        )
