import pytest
import torch

from rwkv_lab.rwkv8_deltanet import RWKV8ChannelMixDeltaNet
from rwkv_lab.training_components import (
    ActivationImplementation,
    build_registered_activation,
)


def test_rwkv_channelmix_consumes_declarative_activation_without_state_drift():
    module = RWKV8ChannelMixDeltaNet(4, 8)
    value = torch.randn(2, 3, 4)
    squared = module(value)
    policy = build_registered_activation(ActivationImplementation.SILU_V1)
    policy.install(module)
    observed = module(value)
    expected = module.value(torch.nn.functional.silu(module.key(value)))
    torch.testing.assert_close(observed, expected)
    assert module.activation_name == "silu"
    assert not torch.equal(squared, observed)
    assert "_activation" not in module.state_dict()


def test_rwkv_channelmix_rejects_squared_relu_only_fusion_under_silu():
    module = RWKV8ChannelMixDeltaNet(4, 8, activation="silu")
    with pytest.raises(ValueError, match="only squared_relu"):
        module.enable_fused_training()
