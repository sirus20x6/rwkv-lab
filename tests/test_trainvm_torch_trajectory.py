from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rwkv_lab.training_optimizers import FP32MasterAdamW
from rwkv_lab.trainvm_worker import state_fingerprint, torch_trajectory_state


def test_torch_trajectory_identity_detects_data_writes_and_restoration() -> None:
    torch.manual_seed(7)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    rng = np.random.default_rng(11)

    def snapshot() -> str:
        return state_fingerprint(
            torch_trajectory_state(
                model,
                optimizer,
                optimizer_step=0,
                numpy_rng=rng,
                extra={"data_cursor": 0},
            )
        )

    original_weights = model.weight.detach().clone()
    before = snapshot()
    assert snapshot() == before

    # Tensor.data mutations evade PyTorch's version counter.  The phase proof
    # hashes content and must still catch them.
    model.weight.data.add_(1)
    assert snapshot() != before
    model.weight.data.copy_(original_weights)
    assert snapshot() == before


def test_torch_trajectory_identity_covers_optimizer_and_rng_state() -> None:
    torch.manual_seed(3)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)

    before = state_fingerprint(
        torch_trajectory_state(model, optimizer, optimizer_step=0)
    )
    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    after = state_fingerprint(
        torch_trajectory_state(model, optimizer, optimizer_step=1)
    )
    assert after != before


def test_torch_trajectory_identity_covers_multiple_optimizer_modules() -> None:
    backbone = torch.nn.Linear(2, 2)
    auxiliary = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(
        [*backbone.parameters(), *auxiliary.parameters()], lr=1.0e-3
    )
    before = state_fingerprint(
        torch_trajectory_state(
            {"backbone": backbone, "auxiliary": auxiliary},
            optimizer,
            optimizer_step=0,
        )
    )
    auxiliary.bias.data.add_(1)
    after = state_fingerprint(
        torch_trajectory_state(
            {"backbone": backbone, "auxiliary": auxiliary},
            optimizer,
            optimizer_step=0,
        )
    )
    assert after != before


def test_torch_trajectory_identity_covers_independent_fp32_masters() -> None:
    model = torch.nn.Linear(2, 2).to(dtype=torch.bfloat16)
    optimizer = FP32MasterAdamW(model.parameters(), lr=1.0e-3)
    before = state_fingerprint(
        torch_trajectory_state(model, optimizer, optimizer_step=0)
    )
    independent_masters = [
        master
        for _model, master, independent in optimizer._model_master_pairs
        if independent
    ]
    assert independent_masters
    independent_masters[0].data.add_(1)
    after = state_fingerprint(
        torch_trajectory_state(model, optimizer, optimizer_step=0)
    )
    assert after != before
