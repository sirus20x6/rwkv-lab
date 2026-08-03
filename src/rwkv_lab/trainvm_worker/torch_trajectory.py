"""Content-complete trajectory identities for disposable Torch phases."""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Mapping, Sequence
from typing import Any


class TorchTrajectoryStateError(ValueError):
    pass


def _digest_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _tensor_digest(tensor: Any, *, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    """Hash every tensor byte while bounding device-to-host staging memory."""

    import torch

    value = tensor.detach()
    if value.is_sparse:
        value = value.to_dense()
    if value.is_quantized:
        value = value.int_repr()
    value = value.contiguous().reshape(-1).view(torch.uint8)
    digest = hashlib.sha256()
    elements_per_chunk = max(1, chunk_bytes)
    for offset in range(0, value.numel(), elements_per_chunk):
        chunk = value[offset : offset + elements_per_chunk].to(
            device="cpu", non_blocking=False
        )
        digest.update(chunk.numpy().tobytes())
    return "sha256:" + digest.hexdigest()


def _tensor_identity(tensor: Any) -> dict[str, Any]:
    return {
        "content": _tensor_digest(tensor),
        "device": str(tensor.device),
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
    }


def _json_state(value: Any) -> Any:
    """Convert optimizer/RNG state to a closed, content-addressed JSON tree."""

    try:
        import torch
    except ImportError:  # pragma: no cover - the caller already requires Torch
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        return {"tensor": _tensor_identity(value)}
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TorchTrajectoryStateError("trajectory state contains nonfinite data")
        return value
    if isinstance(value, Mapping):
        encoded: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, (str, int)):
                raise TorchTrajectoryStateError(
                    "trajectory state mapping key is not a string or integer"
                )
            name = str(key)
            if name in encoded:
                raise TorchTrajectoryStateError(
                    "trajectory state mapping keys are ambiguous"
                )
            encoded[name] = _json_state(item)
        return encoded
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_state(item) for item in value]
    raise TorchTrajectoryStateError(
        f"unsupported trajectory state value: {type(value).__name__}"
    )


def torch_trajectory_state(
    model: Any,
    optimizer: Any,
    *,
    optimizer_step: int,
    numpy_rng: Any | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Return a JSON-safe identity covering trainable state and all RNGs.

    Tensor contents are SHA-256 hashed in bounded chunks.  This is deliberately
    more expensive than comparing object versions: ``Tensor.data`` writes do
    not reliably increment PyTorch's version counter and therefore cannot
    prove that a disposable warmup restored the trajectory.
    """

    import torch

    if isinstance(optimizer_step, bool) or not isinstance(optimizer_step, int):
        raise TorchTrajectoryStateError("optimizer step is invalid")

    named_parameters = dict(model.named_parameters())
    parameter_names = {id(value): name for name, value in named_parameters.items()}
    model_state = {
        "parameters": {
            name: {
                **_tensor_identity(value),
                "requires_grad": bool(value.requires_grad),
                "gradient": (
                    None if value.grad is None else _tensor_identity(value.grad)
                ),
            }
            for name, value in named_parameters.items()
        },
        "buffers": {
            name: _tensor_identity(value) for name, value in model.named_buffers()
        },
        "training": bool(model.training),
    }

    groups: list[dict[str, Any]] = []
    for group in optimizer.param_groups:
        names: list[str] = []
        for parameter in group.get("params", ()):
            name = parameter_names.get(id(parameter))
            if name is None:
                raise TorchTrajectoryStateError(
                    "optimizer owns a parameter outside the declared model"
                )
            names.append(name)
        values = {
            str(key): _json_state(value)
            for key, value in group.items()
            if key != "params"
        }
        groups.append({"parameters": names, "values": values})

    optimizer_state: dict[str, Any] = {}
    for parameter, value in optimizer.state.items():
        name = parameter_names.get(id(parameter))
        if name is None:
            raise TorchTrajectoryStateError(
                "optimizer state refers to a parameter outside the declared model"
            )
        optimizer_state[name] = _json_state(value)

    cpu_rng = torch.get_rng_state().contiguous().numpy().tobytes()
    cuda_rng = (
        [
            _digest_bytes(value.contiguous().cpu().numpy().tobytes())
            for value in torch.cuda.get_rng_state_all()
        ]
        if torch.cuda.is_available()
        else []
    )
    rng_state: dict[str, Any] = {
        "python": _digest_bytes(repr(random.getstate()).encode("utf-8")),
        "torch": _digest_bytes(cpu_rng),
        "cuda": cuda_rng,
    }
    if numpy_rng is not None:
        rng_state["numpy"] = _json_state(numpy_rng.bit_generator.state)

    return {
        "schema": "trainvm.torch-trajectory-state/v1",
        "optimizer_step": optimizer_step,
        "model": model_state,
        "optimizer": {"groups": groups, "state": optimizer_state},
        "rng": rng_state,
        "extra": _json_state(dict(extra or {})),
    }


__all__ = [
    "TorchTrajectoryStateError",
    "torch_trajectory_state",
]
