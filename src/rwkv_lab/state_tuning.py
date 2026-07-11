"""Trainable recurrent-state adapters for RWKV.

Inspired by the RWKV community's state-tuning and RLHF implementations:

* https://github.com/wc2395082443-del/rwkv-rlhf
* https://github.com/OpenMOSE/RWKV-LM-RLHF
* https://discord.com/channels/992359628979568762/1178684958353661994/1522904879528673301

Unlike LoRA, a state adapter leaves every model weight frozen and learns the
constant-size initial WKV and token-shift states.  Artifacts are named,
content-addressed safetensors with an explicit base-checkpoint fingerprint.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import nn


STATE_ADAPTER_SCHEMA = "rwkv-lab.state-adapter.v1"
STATE_OFFSET_SCHEMA = "rwkv-lab.state-offset-adapter.v1"


class RecurrentStateAdapter(nn.Module):
    def __init__(self, model: nn.Module, *, name: str = "default", init_std: float = 0.0):
        super().__init__()
        blocks = list(getattr(model, "blocks", ()))
        if not blocks:
            raise ValueError("state tuning requires model.blocks")
        self.name = str(name)
        self.wkv = nn.ParameterList()
        self.att_shift = nn.ParameterList()
        self.ffn_shift = nn.ParameterList()
        for block in blocks:
            core = getattr(block, "att", None)
            if hasattr(core, "core"):
                core = core.core
            heads = int(getattr(core, "num_heads", 0))
            head_size = int(getattr(core, "head_size", 0))
            channels = int(getattr(core, "hidden_size", 0))
            if not heads or not head_size or not channels:
                raise ValueError("state tuning supports native RWKV time-mix blocks only")
            self.wkv.append(nn.Parameter(torch.zeros(1, heads, head_size, head_size)))
            self.att_shift.append(nn.Parameter(torch.zeros(1, 1, channels)))
            self.ffn_shift.append(nn.Parameter(torch.zeros(1, 1, channels)))
        if init_std:
            for parameter in self.parameters():
                nn.init.normal_(parameter, std=float(init_std))

    def expanded(self, batch: int, *, detach: bool = False) -> list[dict[str, torch.Tensor]]:
        if batch < 1:
            raise ValueError("state-adapter batch must be positive")
        result = []
        for wkv, att, ffn in zip(self.wkv, self.att_shift, self.ffn_shift):
            values = {"wkv": wkv.expand(batch, -1, -1, -1),
                      "att_shift": att.expand(batch, -1, -1),
                      "ffn_shift": ffn.expand(batch, -1, -1)}
            result.append({key: value.detach() for key, value in values.items()} if detach else values)
        return result

    def regularization(self, *, magnitude: float = 0.0, drift: float = 0.0,
                       parent: "RecurrentStateAdapter | None" = None) -> torch.Tensor:
        params = list(self.parameters())
        loss = sum(parameter.float().square().mean() for parameter in params) * float(magnitude)
        if drift:
            if parent is None:
                raise ValueError("state drift regularization requires a parent adapter")
            other = list(parent.parameters())
            if len(params) != len(other):
                raise ValueError("state adapters have different geometry")
            loss = loss + float(drift) * sum(
                (left.float() - right.detach().float()).square().mean()
                for left, right in zip(params, other))
        return loss


class StateOffsetAdapter(RecurrentStateAdapter):
    """FP32 state offsets injected on every selected recurrent timestep.

    Kang et al., *State-offset Tuning* (ACL 2025),
    https://arxiv.org/abs/2503.03499, directly modifies an SSM's state at each
    timestep.  This RWKV adaptation covers matrix and token-shift carries and
    keeps an exact no-op zero initialization. ``interval`` permits controlled
    scheduled-offset ablations; one is the paper-faithful default.
    """
    def __init__(self, model: nn.Module, *, name: str = "default", init_std: float = 0.0,
                 interval: int = 1):
        if interval < 1:
            raise ValueError("state-offset interval must be positive")
        super().__init__(model, name=name, init_std=init_std)
        self.interval = int(interval)

    def apply(self, states, batch: int, *, step: int = 0):
        if step % self.interval:
            return states
        offsets = self.expanded(batch)
        states = states if states is not None else [None] * len(offsets)
        if len(states) != len(offsets):
            raise ValueError("state offsets do not match recurrent depth")
        result = []
        for state, delta in zip(states, offsets):
            state = state or {}
            merged = dict(state)
            merged.update({key: (state[key].float() + value
                                 if state.get(key) is not None else value)
                           for key, value in delta.items()})
            result.append(merged)
        return result


def install_state_adapter(model: nn.Module, *, name: str = "default",
                          freeze_base: bool = True, init_std: float = 0.0) -> RecurrentStateAdapter:
    if freeze_base:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    adapter = RecurrentStateAdapter(model, name=name, init_std=init_std)
    reference = next(model.parameters())
    adapter.to(device=reference.device, dtype=torch.float32)
    model.state_adapter = adapter
    return adapter


def install_state_offset_adapter(model: nn.Module, *, name: str = "default",
                                 freeze_base: bool = True, init_std: float = 0.0,
                                 interval: int = 1) -> StateOffsetAdapter:
    if freeze_base:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    adapter = StateOffsetAdapter(model, name=name, init_std=init_std, interval=interval)
    reference = next(model.parameters())
    adapter.to(device=reference.device, dtype=torch.float32)
    model.state_offset_adapter = adapter
    return adapter


def state_adapter_parameters(model: nn.Module):
    adapter = getattr(model, "state_adapter", None)
    if not isinstance(adapter, RecurrentStateAdapter):
        raise ValueError("model has no recurrent state adapter")
    yield from adapter.parameters()


def _base_geometry(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if name.startswith("state_adapter."):
            continue
        digest.update(name.encode() + str(tuple(tensor.shape)).encode() + str(tensor.dtype).encode())
    return digest.hexdigest()


def save_state_adapter(model: nn.Module, directory: str | Path,
                       *, parent_checkpoint: str = "") -> dict:
    from safetensors.torch import save_file
    adapter = getattr(model, "state_adapter", None)
    if not isinstance(adapter, RecurrentStateAdapter):
        raise ValueError("model has no recurrent state adapter")
    root = Path(directory); root.mkdir(parents=True, exist_ok=True)
    tensors = {name: value.detach().cpu().contiguous()
               for name, value in adapter.state_dict().items()}
    weights = root / "state_adapter.safetensors"
    save_file(tensors, str(weights), metadata={"schema": STATE_ADAPTER_SCHEMA})
    sha = hashlib.sha256(weights.read_bytes()).hexdigest()
    manifest = {"schema": STATE_ADAPTER_SCHEMA, "name": adapter.name,
                "layers": len(adapter.wkv), "base_geometry_sha256": _base_geometry(model),
                "parent_checkpoint": parent_checkpoint, "weights": weights.name,
                "weights_sha256": sha}
    (root / "state_adapter.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_state_adapter(model: nn.Module, directory: str | Path, *, freeze_base: bool = True):
    from safetensors.torch import load_file
    root = Path(directory)
    manifest = json.loads((root / "state_adapter.json").read_text())
    if manifest.get("schema") != STATE_ADAPTER_SCHEMA:
        raise ValueError("unsupported state-adapter manifest")
    weights = root / manifest["weights"]
    if hashlib.sha256(weights.read_bytes()).hexdigest() != manifest.get("weights_sha256"):
        raise ValueError("state-adapter weights hash mismatch")
    if _base_geometry(model) != manifest.get("base_geometry_sha256"):
        raise ValueError("state-adapter base geometry mismatch")
    adapter = install_state_adapter(model, name=manifest["name"], freeze_base=freeze_base)
    adapter.load_state_dict(load_file(str(weights)))
    return manifest
