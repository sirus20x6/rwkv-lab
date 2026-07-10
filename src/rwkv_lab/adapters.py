"""Named, mergeable LoRA adapters for native RWKV and converted PyTorch models.

Primary references:

* LoRA, Hu et al. (2021), https://arxiv.org/abs/2106.09685.
* QLoRA, Dettmers et al. (2023), https://arxiv.org/abs/2305.14314.

This module implements the LoRA update ``W x + (alpha/r) B A x`` and the full artifact
lifecycle around it: frozen-base injection, multiple named adapters, activation, exact
merge/unmerge, safetensors persistence, and base-model fingerprints. Native NF4 quantization is
opt-in through ``quantization.py`` so unsupported RWKV kernels never get silently replaced.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Iterable, Iterator, Sequence

import torch
from torch import nn
import torch.nn.functional as F


ADAPTER_SCHEMA = "rwkv-lab.adapter.v1"
DEFAULT_RWKV_TARGETS = ("receptance", "key", "value", "output", "key_gate", "up", "down")


class LoRABranch(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float,
                 dropout: float = 0.0, *, dtype: torch.dtype | None = None,
                 device: torch.device | str | None = None):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.dropout = float(dropout)
        factory = {"dtype": dtype or torch.float32, "device": device}
        self.A = nn.Parameter(torch.empty(rank, in_features, **factory))
        self.B = nn.Parameter(torch.zeros(out_features, rank, **factory))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    @property
    def scale(self) -> float:
        return self.alpha / self.rank

    def delta(self) -> torch.Tensor:
        return (self.B @ self.A) * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x.to(self.A.dtype), p=self.dropout, training=self.training)
        return F.linear(F.linear(x, self.A), self.B) * self.scale


class LoRALinear(nn.Module):
    """Wrap a frozen linear-like module with independently named LoRA branches."""

    def __init__(self, base: nn.Module):
        super().__init__()
        if not hasattr(base, "weight") or not hasattr(base, "in_features") or not hasattr(base, "out_features"):
            raise TypeError("LoRALinear base must expose weight, in_features, and out_features")
        self.base = base
        self.adapters = nn.ModuleDict()
        self.active: tuple[str, ...] = ()
        self.merged: set[str] = set()
        self._unmerged_weight: torch.Tensor | None = None
        for parameter in base.parameters():
            parameter.requires_grad_(False)

    @property
    def in_features(self) -> int:
        return int(self.base.in_features)

    @property
    def out_features(self) -> int:
        return int(self.base.out_features)

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return getattr(self.base, "bias", None)

    def add_adapter(self, name: str, *, rank: int, alpha: float | None = None,
                    dropout: float = 0.0) -> None:
        _validate_name(name)
        if name in self.adapters:
            raise ValueError(f"adapter {name!r} already exists")
        weight = self.base.weight
        # Keep adapter optimizer state in fp32 even when the frozen base runs bf16/4-bit.
        dtype = torch.float32
        self.adapters[name] = LoRABranch(self.in_features, self.out_features, rank,
                                         float(alpha if alpha is not None else rank), dropout,
                                         dtype=dtype, device=weight.device)
        self.active = (*self.active, name)

    def set_active(self, names: Sequence[str]) -> None:
        unknown = set(names) - set(self.adapters)
        if unknown:
            raise ValueError(f"unknown adapters: {sorted(unknown)}")
        self.active = tuple(dict.fromkeys(names))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        for name in self.active:
            if name not in self.merged:
                out = out + self.adapters[name](x).to(out.dtype)
        return out

    @torch.no_grad()
    def merge(self, name: str) -> None:
        if name not in self.adapters:
            raise ValueError(f"unknown adapter {name!r}")
        if name in self.merged:
            return
        if getattr(self.base, "is_quantized_4bit", False):
            raise ValueError("cannot mutate a packed 4-bit base; materialize a dense merged model")
        if not self.merged:
            self._unmerged_weight = self.base.weight.detach().clone()
        self.merged.add(name)
        self._rebuild_merged_weight()

    @torch.no_grad()
    def unmerge(self, name: str) -> None:
        if name not in self.merged:
            return
        self.merged.remove(name)
        self._rebuild_merged_weight()
        if not self.merged:
            self._unmerged_weight = None

    @torch.no_grad()
    def _rebuild_merged_weight(self) -> None:
        if self._unmerged_weight is None:
            raise RuntimeError("merged LoRA base snapshot is missing")
        self.base.weight.copy_(self._unmerged_weight)
        for name in sorted(self.merged):
            delta = self.adapters[name].delta().to(device=self.base.weight.device,
                                                    dtype=self.base.weight.dtype)
            self.base.weight.add_(delta)


@dataclass(frozen=True)
class AdapterConfig:
    name: str = "default"
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.0
    targets: tuple[str, ...] = DEFAULT_RWKV_TARGETS


def inject_lora(model: nn.Module, config: AdapterConfig, *, freeze_base: bool = True,
                match: str = "suffix") -> list[str]:
    """Inject LoRA into matching linear-like modules and return their stable module paths.

    ``suffix`` matches the final dotted component against ``config.targets``. ``regex`` treats
    each target as a full-path regular expression.  Targeting is explicit because Transformer
    names such as ``q_proj`` are not meaningful defaults for RWKV time/channel-mix modules.
    """
    if freeze_base:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    candidates = list(model.named_modules())
    selected: list[str] = []
    for path, module in candidates:
        if not path or isinstance(module, LoRALinear) or not _linear_like(module):
            continue
        leaf = path.rsplit(".", 1)[-1]
        matches = (leaf in config.targets if match == "suffix"
                   else any(re.search(pattern, path) for pattern in config.targets))
        if not matches:
            continue
        parent, attr = _parent_module(model, path)
        wrapper = LoRALinear(module)
        wrapper.add_adapter(config.name, rank=config.rank, alpha=config.alpha,
                            dropout=config.dropout)
        setattr(parent, attr, wrapper)
        selected.append(path)
    if not selected:
        raise ValueError(f"no linear modules matched LoRA targets {config.targets}")
    return selected


def add_adapter(model: nn.Module, config: AdapterConfig, *, module_paths: Iterable[str] | None = None) -> None:
    wanted = set(module_paths or ())
    found = 0
    for path, module in iter_lora(model):
        if wanted and path not in wanted:
            continue
        module.add_adapter(config.name, rank=config.rank, alpha=config.alpha, dropout=config.dropout)
        found += 1
    if not found:
        raise ValueError("model has no matching LoRA-wrapped modules")


def iter_lora(model: nn.Module) -> Iterator[tuple[str, LoRALinear]]:
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            yield name, module


def set_active_adapters(model: nn.Module, names: Sequence[str]) -> None:
    for _, module in iter_lora(model):
        module.set_active(names)


@contextmanager
def active_adapters(model: nn.Module, names: Sequence[str]):
    previous = [(module, module.active) for _, module in iter_lora(model)]
    set_active_adapters(model, names)
    try:
        yield model
    finally:
        for module, active in previous:
            module.set_active(active)


def merge_adapter(model: nn.Module, name: str) -> None:
    for _, module in iter_lora(model):
        module.merge(name)


def unmerge_adapter(model: nn.Module, name: str) -> None:
    for _, module in iter_lora(model):
        module.unmerge(name)


def adapter_parameters(model: nn.Module, names: Sequence[str] | None = None) -> Iterator[nn.Parameter]:
    wanted = set(names or ())
    for _, module in iter_lora(model):
        for name, branch in module.adapters.items():
            if not wanted or name in wanted:
                yield from branch.parameters()


def base_fingerprint(model: nn.Module) -> str:
    """Hash frozen base tensors while normalizing wrapper-induced ``.base.`` path changes."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if ".adapters." in name:
            continue
        name = name.replace(".base.", ".")
        digest.update(name.encode() + b"\0")
        value = tensor.detach()
        if hasattr(value, "get_original_weight"):
            value = value.get_original_weight()
        value = value.cpu().contiguous()
        digest.update(str(value.dtype).encode() + str(tuple(value.shape)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def save_adapter(model: nn.Module, directory: str | Path, name: str, *,
                 parent_checkpoint: str = "", metadata: dict | None = None) -> dict:
    from safetensors.torch import save_file

    _validate_name(name)
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, torch.Tensor] = {}
    modules = []
    branch_config = None
    quantized = False
    for path, module in iter_lora(model):
        if name not in module.adapters:
            continue
        if name in module.merged:
            raise ValueError(f"unmerge adapter {name!r} before saving")
        branch = module.adapters[name]
        tensors[f"{path}.A"] = branch.A.detach().cpu().contiguous()
        tensors[f"{path}.B"] = branch.B.detach().cpu().contiguous()
        modules.append(path)
        current = {"rank": branch.rank, "alpha": branch.alpha, "dropout": branch.dropout}
        if branch_config is not None and current != branch_config:
            raise ValueError("one adapter name must use a consistent rank/alpha/dropout")
        branch_config = current
        quantized |= bool(getattr(module.base, "is_quantized_4bit", False))
    if not tensors or branch_config is None:
        raise ValueError(f"model has no adapter named {name!r}")
    weights = root / "adapter.safetensors"
    save_file(tensors, str(weights), metadata={"schema": ADAPTER_SCHEMA, "name": name})
    manifest = {"schema": ADAPTER_SCHEMA, "name": name, **branch_config, "modules": modules,
                "base_sha256": base_fingerprint(model), "parent_checkpoint": parent_checkpoint,
                "quantized_frozen_base": quantized, "weights": weights.name,
                "metadata": metadata or {}}
    manifest["weights_sha256"] = _sha256(weights)
    (root / "adapter.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_adapter(model: nn.Module, directory: str | Path, *, name: str | None = None,
                 verify_base: bool = True) -> dict:
    from safetensors.torch import load_file

    root = Path(directory)
    manifest = json.loads((root / "adapter.json").read_text())
    if manifest.get("schema") != ADAPTER_SCHEMA:
        raise ValueError("unsupported adapter manifest schema")
    if _sha256(root / manifest["weights"]) != manifest.get("weights_sha256"):
        raise ValueError("adapter weights hash mismatch")
    if verify_base and base_fingerprint(model) != manifest.get("base_sha256"):
        raise ValueError("adapter base-model fingerprint mismatch")
    adapter_name = name or manifest["name"]
    _validate_name(adapter_name)
    modules = dict(iter_lora(model))
    config = AdapterConfig(adapter_name, int(manifest["rank"]), float(manifest["alpha"]),
                           float(manifest.get("dropout", 0.0)), tuple(manifest["modules"]))
    for path in manifest["modules"]:
        if path not in modules:
            raw = dict(model.named_modules()).get(path)
            if raw is None or not _linear_like(raw):
                raise ValueError(f"adapter target {path!r} is absent from the model")
            parent, attr = _parent_module(model, path)
            setattr(parent, attr, LoRALinear(raw))
            modules = dict(iter_lora(model))
        modules[path].add_adapter(adapter_name, rank=config.rank, alpha=config.alpha,
                                  dropout=config.dropout)
    tensors = load_file(str(root / manifest["weights"]))
    for path in manifest["modules"]:
        branch = modules[path].adapters[adapter_name]
        branch.A.data.copy_(tensors[f"{path}.A"].to(branch.A))
        branch.B.data.copy_(tensors[f"{path}.B"].to(branch.B))
    return {**manifest, "loaded_as": adapter_name}


def _linear_like(module: nn.Module) -> bool:
    return (hasattr(module, "weight") and hasattr(module, "in_features") and
            hasattr(module, "out_features") and callable(getattr(module, "forward", None)))


@torch.no_grad()
def unload_adapter(model: nn.Module, name: str, *, merge: bool = True) -> list[str]:
    """Replace LoRA wrappers by dense linears, suitable for immutable parent checkpoints.

    Packed QLoRA bases are dequantized exactly once during materialization; the quantized parent
    itself remains immutable, matching the frozen-base contract in https://arxiv.org/abs/2305.14314.
    """
    replaced = []
    for path, wrapper in list(iter_lora(model)):
        if name not in wrapper.adapters:
            raise ValueError(f"adapter {name!r} is absent from {path}")
        base_weight = (wrapper.base.dequantized_weight(dtype=torch.float32)
                       if getattr(wrapper.base, "is_quantized_4bit", False)
                       else wrapper.base.weight.detach().float().clone())
        if merge:
            base_weight.add_(wrapper.adapters[name].delta().detach().float())
        bias = wrapper.bias
        dense = nn.Linear(wrapper.in_features, wrapper.out_features, bias=bias is not None,
                          device=base_weight.device, dtype=base_weight.dtype)
        dense.weight.copy_(base_weight)
        if bias is not None:
            dense.bias.copy_(bias.detach().to(dense.bias))
        parent, attr = _parent_module(model, path)
        setattr(parent, attr, dense)
        replaced.append(path)
    return replaced


def _parent_module(model: nn.Module, path: str) -> tuple[nn.Module, str]:
    parent_path, _, attr = path.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    return parent, attr


def _validate_name(name: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("adapter names may contain only letters, digits, '_' and '-'")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
