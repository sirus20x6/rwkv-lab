import pytest
import torch

from rwkv_lab.rwkv_pretrain import build_optimizer


def test_rwkv_adamw_path_uses_registered_scalar_cpu_backend():
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.LayerNorm(4))
    optimizer = build_optimizer(
        model.named_parameters(),
        "adamw",
        lr=3.0e-4,
        wd=0.1,
    )
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults["lr"] == pytest.approx(3.0e-4)
    assert optimizer.defaults["betas"] == pytest.approx((0.9, 0.95))
    assert optimizer.defaults["weight_decay"] == pytest.approx(0.1)
    assert optimizer.defaults["foreach"] is False
    assert optimizer.defaults["fused"] is False
