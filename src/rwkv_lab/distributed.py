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
                     reshard_after_forward: bool = True,
                     prefetch_depth: int = 0,
                     ignored_params: set[nn.Parameter] | None = None) -> nn.Module:
    """Apply FSDP2 bottom-up to RWKV blocks and then the root model in place."""
    if not torch.distributed.is_initialized():
        raise RuntimeError("FSDP2 requires a torchrun process group")
    from torch.distributed.fsdp import CPUOffloadPolicy, OffloadPolicy, fully_shard

    offload = CPUOffloadPolicy() if cpu_offload else OffloadPolicy()
    ignored_params = set(ignored_params or ())
    blocks = getattr(model, "blocks", ())
    for block in blocks:
        block_params = set(block.parameters())
        fully_shard(block, reshard_after_forward=reshard_after_forward,
                    offload_policy=offload,
                    ignored_params=ignored_params & block_params)
    fully_shard(model, reshard_after_forward=reshard_after_forward,
                offload_policy=offload, ignored_params=ignored_params)
    configure_fsdp_prefetch(model, depth=prefetch_depth)
    return model


def configure_fsdp_prefetch(model: nn.Module, *, depth: int = 1) -> None:
    """Explicitly overlap adjacent RWKV block all-gathers with computation.

    Forward executes blocks in increasing order and backward in reverse order,
    so each block prefetches its next consumers in the corresponding direction.
    FSDP2 keeps this opt-in because explicit ordering assumes a static traversal.
    """
    if depth < 0:
        raise ValueError("FSDP2 prefetch depth must be non-negative")
    if depth == 0:
        return
    blocks = list(getattr(model, "blocks", ()))
    for index, block in enumerate(blocks):
        forward = blocks[index + 1:index + 1 + depth]
        backward = list(reversed(blocks[max(0, index - depth):index]))
        set_forward = getattr(block, "set_modules_to_forward_prefetch", None)
        set_backward = getattr(block, "set_modules_to_backward_prefetch", None)
        if forward and callable(set_forward):
            set_forward(forward)
        if backward and callable(set_backward):
            set_backward(backward)
    root_forward = getattr(model, "set_modules_to_forward_prefetch", None)
    if blocks and callable(root_forward):
        root_forward(blocks[:depth])


def set_requires_gradient_sync(model: nn.Module, required: bool) -> None:
    """Toggle FSDP2 reduction, used to avoid collectives on accumulation microsteps."""
    method = getattr(model, "set_requires_gradient_sync", None)
    if callable(method):
        method(bool(required), recurse=True)


@torch.no_grad()
def sparse_sync_parameter_rows(param: nn.Parameter, rows: torch.Tensor) -> int:
    """Average only touched rows of a replicated embedding-style parameter.

    ``param`` must be excluded from FSDP sharding. Every rank gathers compact
    ``(row_index, row_gradient)`` payloads, reconstructs the same averaged
    sparse gradient locally, and can then run an ordinary replicated optimizer.

    ``rows`` is a *hint*, not a contract: the transmitted set is the union of it
    and the rows the gradient actually reaches, so an incomplete or approximate
    index mapping costs bandwidth rather than silently dropping gradient. Rows
    are reduced in fp32 regardless of the parameter dtype.

    Returns the maximum transmitted row count (useful for diagnostics).
    """
    if param.grad is None:
        return 0
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return int(_touched_rows(param, rows).numel())
    world = torch.distributed.get_world_size()
    if world == 1:
        return int(_touched_rows(param, rows).numel())
    rows = _touched_rows(param, rows)
    count = torch.tensor([rows.numel()], dtype=torch.int64, device=param.grad.device)
    counts = [torch.empty_like(count) for _ in range(world)]
    torch.distributed.all_gather(counts, count)
    max_rows = max(int(value.item()) for value in counts)
    if max_rows == 0:
        param.grad.zero_()
        return 0

    index_payload = torch.full(
        (max_rows,), -1, dtype=torch.int64, device=param.grad.device
    )
    # Reduce in fp32 regardless of the parameter dtype. Summing `world` bf16 row
    # gradients in bf16 loses roughly log2(world) mantissa bits before the mean
    # is taken, which is exactly the accumulation error DDP avoids.
    value_payload = torch.zeros(
        (max_rows, *param.grad.shape[1:]),
        dtype=torch.float32,
        device=param.grad.device,
    )
    if rows.numel():
        index_payload[: rows.numel()] = rows
        value_payload[: rows.numel()] = param.grad.index_select(0, rows).float()
    gathered_indices = [torch.empty_like(index_payload) for _ in range(world)]
    gathered_values = [torch.empty_like(value_payload) for _ in range(world)]
    torch.distributed.all_gather(gathered_indices, index_payload)
    torch.distributed.all_gather(gathered_values, value_payload)

    reduced = torch.zeros_like(param.grad, dtype=torch.float32)
    for indices, values in zip(gathered_indices, gathered_values):
        valid = indices >= 0
        if valid.any():
            reduced.index_add_(0, indices[valid], values[valid])
    reduced.div_(world)
    param.grad.copy_(reduced)
    return max_rows


def _touched_rows(param: nn.Parameter, hint: torch.Tensor) -> torch.Tensor:
    """Every row this rank must transmit: the caller's hint plus real nonzeros.

    Deriving the nonzero set from the gradient itself is what makes this safe.
    An earlier version trusted ``hint`` alone and then zeroed ``param.grad``, so
    any gradient reaching a row the caller did not know about — a tied parameter,
    a second lookup path, a table whose index mapping the caller approximated —
    was discarded with no error at all. The scan is one reduction over a
    gradient that is about to be gathered anyway.
    """
    grad = param.grad
    hint = hint.to(device=grad.device, dtype=torch.int64).reshape(-1)
    hint = hint[(hint >= 0) & (hint < param.shape[0])]
    nonzero = grad.reshape(grad.shape[0], -1).ne(0).any(dim=1).nonzero().reshape(-1)
    return torch.unique(torch.cat((hint, nonzero)))


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
    if rank < saved_world:
        payload = manifest.get("rank_extras", [{}])[rank]
        tensor_extra = {f"rank_{rank}.{key}": torch.empty(
                            spec["shape"], dtype=_dtype(spec["dtype"]))
                        for key, spec in payload.get("tensors", {}).items()}
    else:
        # Resuming into a LARGER world: this rank has no saved per-rank state.
        # Take only the (shared) JSON extras from rank 0; do NOT copy rank-0's
        # per-rank tensor extras (e.g. RNG state), which would give duplicated /
        # correlated random streams — leave this rank's fresh harness-seeded RNG.
        payload = {"json": manifest.get("rank_extras", [{}])[0].get("json", {}),
                   "tensors": {}}
        tensor_extra = {}
        print(f"[rwkv_lab.distributed] rank {rank} >= saved world_size {saved_world}: "
              "loading shared JSON extras from rank 0 and skipping per-rank tensor "
              "extras (fresh per-rank RNG kept)", flush=True)
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
