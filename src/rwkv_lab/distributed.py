"""FSDP2 and exact-resume distributed checkpoint helpers.

The implementation follows PyTorch's composable ``fully_shard`` and Distributed Checkpoint
contracts rather than importing a second launcher/runtime:

* FSDP2 API: https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html
* Distributed Checkpoint recipe:
  https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html

Adamaton owns allocation, leases, and worker orchestration.  This module begins at the model-state
boundary: rank discovery, bottom-up RWKV sharding, optional activation checkpointing, collective
gradient clipping, and reshardable model/optimizer/RNG state.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
from typing import Any

import torch
from torch import nn


DCP_SCHEMA = "rwkv-lab.dcp.v1"


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    device: str
    initialized_here: bool = False

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


def initialize(backend: str = "auto") -> DistributedContext:
    """Initialize from torchrun environment variables; remain a no-op at world size one."""
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    device = f"cuda:{local}" if torch.cuda.is_available() else "cpu"
    initialized_here = False
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
    if world > 1 and not torch.distributed.is_initialized():
        selected = ("nccl" if torch.cuda.is_available() else "gloo") if backend == "auto" else backend
        torch.distributed.init_process_group(selected, init_method="env://")
        initialized_here = True
    return DistributedContext(rank, world, local, device, initialized_here)


def barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def fully_shard_rwkv(model: nn.Module, *, cpu_offload: bool = False,
                     reshard_after_forward: bool = True) -> nn.Module:
    """Apply FSDP2 bottom-up to RWKV blocks and then the root model in place."""
    if not torch.distributed.is_initialized():
        raise RuntimeError("FSDP2 requires a torchrun process group")
    from torch.distributed.fsdp import CPUOffloadPolicy, OffloadPolicy, fully_shard

    offload = CPUOffloadPolicy() if cpu_offload else OffloadPolicy()
    blocks = getattr(model, "blocks", ())
    for block in blocks:
        fully_shard(block, reshard_after_forward=reshard_after_forward,
                    offload_policy=offload)
    fully_shard(model, reshard_after_forward=reshard_after_forward, offload_policy=offload)
    return model


def checkpoint_rwkv_blocks(model: nn.Module) -> nn.Module:
    """Wrap each RWKV block with non-reentrant activation checkpointing."""
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl, checkpoint_wrapper)

    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise ValueError("activation checkpointing expects model.blocks")
    for index, block in enumerate(list(blocks)):
        blocks[index] = checkpoint_wrapper(block, checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                                           preserve_rng_state=True)
    return model


def clip_grad_norm(model: nn.Module, max_norm: float) -> torch.Tensor:
    """Use FSDP2's collective norm when available, ordinary clipping otherwise."""
    method = getattr(model, "clip_grad_norm_", None)
    return method(max_norm) if callable(method) else torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def save_checkpoint(path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer,
                    *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Collectively save model/optimizer plus JSON and tensor extras into an atomic DCP directory."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict

    destination = Path(path)
    temporary = destination.with_name(destination.name + ".tmp")
    primary = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    if primary:
        shutil.rmtree(temporary, ignore_errors=True)
        temporary.mkdir(parents=True, exist_ok=True)
    barrier()
    model_state, optim_state = get_state_dict(
        model, optimizer, options=StateDictOptions(full_state_dict=False, cpu_offload=True))
    json_extra, local_tensors = _split_extra(extra or {})
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    tensor_extra = {f"rank_{rank}.{key}": value for key, value in local_tensors.items()}
    local_spec = {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                  for key, value in local_tensors.items()}
    rank_payloads = _all_gather_objects({"json": json_extra, "tensors": local_spec})
    state = {"model": model_state, "optimizer": optim_state, "extra": tensor_extra}
    dcp.save(state, checkpoint_id=str(temporary))
    manifest = {"schema": DCP_SCHEMA, "world_size": _world_size(),
                "rank_extras": rank_payloads}
    if primary:
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    barrier()
    if primary:
        shutil.rmtree(destination, ignore_errors=True)
        temporary.replace(destination)
    barrier()
    return manifest


def load_checkpoint(path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer,
                    *, strict: bool = True) -> dict[str, Any]:
    """Collectively load a DCP checkpoint; resharding across world sizes is handled by DCP."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (StateDictOptions, get_state_dict,
                                                         set_state_dict)

    source = Path(path)
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest.get("schema") != DCP_SCHEMA:
        raise ValueError("unsupported distributed checkpoint schema")
    options = StateDictOptions(full_state_dict=False, cpu_offload=True, strict=strict)
    model_state, optim_state = get_state_dict(model, optimizer, options=options)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    saved_world = int(manifest.get("world_size", 1))
    source_rank = rank if rank < saved_world else 0
    payload = manifest.get("rank_extras", [{}])[source_rank]
    tensor_extra = {f"rank_{source_rank}.{key}": torch.empty(
                        spec["shape"], dtype=_dtype(spec["dtype"]))
                    for key, spec in payload.get("tensors", {}).items()}
    state = {"model": model_state, "optimizer": optim_state, "extra": tensor_extra}
    dcp.load(state, checkpoint_id=str(source))
    set_state_dict(model, optimizer, model_state_dict=state["model"],
                   optim_state_dict=state["optimizer"], options=options)
    loaded_tensors = {key.split(".", 1)[1]: value for key, value in state["extra"].items()}
    return {**payload.get("json", {}), **loaded_tensors}


def _split_extra(extra: dict[str, Any]) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    json_values: dict[str, Any] = {}
    tensors: dict[str, torch.Tensor] = {}
    for key, value in extra.items():
        if isinstance(value, torch.Tensor):
            tensors[key] = value.detach().cpu().contiguous()
        else:
            try:
                json.dumps(value)
            except TypeError as exc:
                raise TypeError(f"distributed checkpoint extra {key!r} is neither tensor nor JSON") from exc
            json_values[key] = value
    return json_values, tensors


def _dtype(name: str) -> torch.dtype:
    value = name.removeprefix("torch.")
    dtype = getattr(torch, value, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unsupported checkpoint dtype {name!r}")
    return dtype


def _world_size() -> int:
    return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1


def _all_gather_objects(value: Any) -> list[Any]:
    if not torch.distributed.is_initialized():
        return [value]
    gathered = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered, value)
    return gathered
