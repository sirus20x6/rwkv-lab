from types import SimpleNamespace

import pytest
import torch

from rwkv_lab.rwkv_pretrain import (
    build_optimizer,
    resolved_worker_component_contract,
)
from rwkv_lab.training_components import PowerCoolConfiguration, ScheduleImplementation


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


def test_rwkv_worker_components_drive_powercool_and_optimizer() -> None:
    schedule = PowerCoolConfiguration(
        warmup_steps=10,
        max_steps=100,
        minimum_ratio=0.1,
        cooldown_fraction=0.2,
        power=2.0,
    )

    class Components:
        composition = SimpleNamespace(composition_digest="sha256:" + "e" * 64)

        def configuration(self, slot, *, category):
            if (slot, category) == (
                "gradient_clipping",
                "gradient_clipping",
            ):
                return {
                    "max_norm": 1.0,
                    "norm_type": 2.0,
                    "error_if_nonfinite": False,
                }
            if (slot, category) == (
                "weight_decay",
                "weight_decay_schedule",
            ):
                return {"weight_decay": 0.1}
            if (slot, category) == (
                "gradient_accumulation",
                "gradient_accumulation",
            ):
                return {"microbatches_per_optimizer_step": 1}
            if (slot, category) == ("objective", "objective"):
                return {"chunk_size": 2048, "prefer_fused": True}
            if (slot, category) == ("precision", "precision"):
                return {
                    "parameter_dtype": "bfloat16",
                    "compute_dtype": "bfloat16",
                    "reduction_dtype": "float32",
                    "gradient_scaling": False,
                }
            assert (slot, category) == ("optimizer", "optimizer")
            return {
                "learning_rate": 3.0e-4,
                "beta1": 0.9,
                "beta2": 0.95,
                "epsilon": 1.0e-8,
                "foreach": False,
                "fused": False,
            }

        def weight_decay_schedule(self, optimizer):
            optimizer.param_groups[0]["weight_decay"] = 0.1
            return SimpleNamespace(step=lambda _step: None)

        def learning_rate_configuration(self):
            return ScheduleImplementation.POWERCOOL_V1, schedule

        def gradient_clipping(self, parameters):
            return torch.nn.utils.clip_grad_norm_(parameters, 1.0)

        def gradient_accumulation(self):
            return SimpleNamespace(
                microbatches_per_optimizer_step=1,
                microbatch_indices=lambda: range(1),
                scale_loss=lambda loss: loss,
            )

        def objective(self):
            return lambda hidden, head, labels, **_kwargs: torch.nn.functional.cross_entropy(
                head(hidden).reshape(-1, head.out_features), labels.reshape(-1)
            )

        def precision(self):
            return SimpleNamespace(
                parameter_dtype=torch.bfloat16,
                convert_module=lambda module, device: module.to(
                    device=device, dtype=torch.bfloat16
                ),
            )

        def evidence(self):
            return {
                "optimizer": {"implementation": "test"},
                "gradient_clipping": {"implementation": "test"},
            }

        def optimizer(self, parameters):
            return torch.optim.AdamW(parameters, lr=3.0e-4)

    components = Components()
    args = SimpleNamespace(
        optimizer="adamw",
        lr_schedule="powercool",
        u_mup_base_width=0,
        lr=3.0e-4,
        weight_decay=0.1,
        grad_clip=1.0,
        grad_accum=1,
        distributed="none",
    )
    resolved, evidence, composition_digest = resolved_worker_component_contract(
        args, schedule, components
    )
    assert resolved == schedule
    assert evidence == components.evidence()
    assert composition_digest == components.composition.composition_digest

    model = torch.nn.Linear(3, 4)
    optimizer = build_optimizer(
        model.named_parameters(),
        "adamw",
        lr=args.lr,
        wd=args.weight_decay,
        worker_components=components,
    )
    assert isinstance(optimizer, torch.optim.AdamW)

    args.lr_schedule = "cosine"
    with pytest.raises(ValueError, match="requires PowerCool"):
        resolved_worker_component_contract(args, schedule, components)
