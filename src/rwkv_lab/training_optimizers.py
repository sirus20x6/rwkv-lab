"""Reusable optimizer implementations independent of model family."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch


class FP32MasterAdamW(torch.optim.AdamW):
    """AdamW with FP32 masters and moments for lower-precision live weights."""

    master_state_key = "_rwkv_lab_fp32_master_weights_v1"

    def __init__(self, params, **kwargs: Any) -> None:
        supplied = list(params)
        if not supplied:
            raise ValueError("FP32MasterAdamW requires parameters")
        if isinstance(supplied[0], Mapping):
            model_groups = [dict(group) for group in supplied]
        else:
            model_groups = [{"params": supplied}]

        master_groups = []
        self._model_master_pairs: list[
            tuple[torch.nn.Parameter, torch.nn.Parameter, bool]
        ] = []
        for model_group in model_groups:
            model_parameters = list(model_group["params"])
            master_parameters = []
            for model_parameter in model_parameters:
                if not isinstance(model_parameter, torch.nn.Parameter):
                    raise TypeError("optimizer parameters must be nn.Parameter")
                if not model_parameter.is_floating_point():
                    raise TypeError("FP32MasterAdamW supports floating parameters only")
                independent = model_parameter.dtype != torch.float32
                master_parameter = (
                    torch.nn.Parameter(
                        model_parameter.detach().to(dtype=torch.float32),
                        requires_grad=True,
                    )
                    if independent
                    else model_parameter
                )
                master_parameters.append(master_parameter)
                self._model_master_pairs.append(
                    (model_parameter, master_parameter, independent)
                )
            master_group = {
                key: value for key, value in model_group.items() if key != "params"
            }
            master_group["params"] = master_parameters
            master_groups.append(master_group)
        super().__init__(master_groups, **kwargs)

    @torch.no_grad()
    def _stage_model_gradients(self) -> None:
        for model, master, independent in self._model_master_pairs:
            if not independent:
                continue
            if model.grad is None:
                master.grad = None
                continue
            if master.grad is None:
                master.grad = torch.empty_like(master)
            master.grad.copy_(model.grad.detach())

    @torch.no_grad()
    def _copy_masters_to_model(self) -> None:
        for model, master, independent in self._model_master_pairs:
            if independent:
                model.copy_(master.to(device=model.device, dtype=model.dtype))

    @torch.no_grad()
    def sync_masters_from_model(self) -> None:
        """Initialize masters from model weights for legacy optimizer resumes."""
        for model, master, independent in self._model_master_pairs:
            if independent:
                master.copy_(model.to(device=master.device, dtype=torch.float32))

    @torch.no_grad()
    def reset_parameter_state(
        self,
        parameters: Sequence[torch.nn.Parameter],
    ) -> int:
        """Reset selected masters and Adam moments from live model tensors."""

        selected = {id(parameter) for parameter in parameters}
        reset = 0
        for model, master, _independent in self._model_master_pairs:
            if id(model) not in selected:
                continue
            master.copy_(model.detach().to(device=master.device, dtype=master.dtype))
            self.state.pop(master, None)
            model.grad = None
            master.grad = None
            reset += 1
        if reset != len(selected):
            raise ValueError("optimizer could not reset every selected parameter")
        return reset

    def step(
        self,
        closure=None,
        *,
        parameter_update_scales: Mapping[torch.nn.Parameter, float] | None = None,
    ):
        """Apply one AdamW step, optionally scaling selected updates."""

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._stage_model_gradients()
        scaled_updates: list[tuple[torch.nn.Parameter, torch.Tensor, float]] = []
        if parameter_update_scales:
            scales_by_id = {
                id(model): float(scale)
                for model, scale in parameter_update_scales.items()
            }
            if any(
                not math.isfinite(scale) or scale <= 0
                for scale in scales_by_id.values()
            ):
                raise ValueError("parameter update scales must be finite and positive")
            for model, master, _independent in self._model_master_pairs:
                scale = scales_by_id.get(id(model), 1.0)
                if scale != 1.0 and master.grad is not None:
                    scaled_updates.append((master, master.detach().clone(), scale))
        observed = super().step()
        with torch.no_grad():
            for master, previous, scale in scaled_updates:
                master.copy_(previous + scale * (master - previous))
        self._copy_masters_to_model()
        return loss if closure is not None else observed

    def zero_grad(self, set_to_none: bool = True) -> None:
        super().zero_grad(set_to_none=set_to_none)
        with torch.no_grad():
            for model, _master, independent in self._model_master_pairs:
                if not independent or model.grad is None:
                    continue
                if set_to_none:
                    model.grad = None
                else:
                    model.grad.zero_()

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state[self.master_state_key] = [
            master.detach()
            for _model, master, independent in self._model_master_pairs
            if independent
        ]
        return state

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        restored = dict(state_dict)
        master_weights = restored.pop(self.master_state_key, None)
        super().load_state_dict(restored)
        independent_pairs = [
            (model, master)
            for model, master, independent in self._model_master_pairs
            if independent
        ]
        if master_weights is None:
            self.sync_masters_from_model()
            return
        if len(master_weights) != len(independent_pairs):
            raise ValueError(
                "FP32 master checkpoint parameter count does not match optimizer"
            )
        with torch.no_grad():
            for (model, master), saved in zip(
                independent_pairs,
                master_weights,
                strict=True,
            ):
                if saved.shape != model.shape:
                    raise ValueError(
                        "FP32 master checkpoint parameter shape does not match model"
                    )
                master.copy_(saved.to(device=master.device, dtype=torch.float32))
                model.copy_(master.to(device=model.device, dtype=model.dtype))

    def precision_report(self) -> dict[str, Any]:
        independent = [
            (model, master)
            for model, master, is_independent in self._model_master_pairs
            if is_independent
        ]
        return {
            "name": type(self).__name__,
            "model_parameter_dtypes": sorted(
                {
                    str(model.dtype).removeprefix("torch.")
                    for model, _master in independent
                }
            ),
            "master_parameter_dtype": "float32",
            "moment_dtype": "float32",
            "independent_master_parameter_count": sum(
                master.numel() for _model, master in independent
            ),
            "exact_master_resume": True,
        }


__all__ = ["FP32MasterAdamW"]
