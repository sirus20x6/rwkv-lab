"""Gated deep visual-prefix reinjection for a frozen RWKV decoder.

The image prefix normally enters only before layer zero. These adapters add a
zero-initialized, low-rank residual derived from that same prefix before chosen
decoder layers. Only visual-token positions are modified, so causal text
semantics and the frozen backbone remain unchanged at initialization.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Sequence

import torch
from torch import nn


class _DeepVisionAdapter(nn.Module):
    def __init__(self, hidden_size: int, rank: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.act = nn.GELU()
        self.up = nn.Linear(rank, hidden_size, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, prefix: torch.Tensor) -> torch.Tensor:
        return self.up(self.act(self.down(self.norm(prefix))))


class DeepVisionInjector(nn.Module):
    """Inject a projected visual prefix at selected frozen decoder layers."""

    def __init__(self, hidden_size: int, layer_indices: Sequence[int], *, rank: int = 256):
        super().__init__()
        if rank < 1:
            raise ValueError("deep vision rank must be positive")
        sites = sorted({int(index) for index in layer_indices})
        if not sites:
            raise ValueError("deep vision needs at least one layer")
        self.layer_indices = tuple(sites)
        self.adapters = nn.ModuleDict({
            str(index): _DeepVisionAdapter(hidden_size, rank) for index in sites
        })
        self._prefix: torch.Tensor | None = None
        self._starts: tuple[int, ...] = ()
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._last_rms: dict[str, torch.Tensor] = {}

    def install(self, layers: Sequence[nn.Module]) -> None:
        if self._handles:
            raise RuntimeError("deep vision injector is already installed")
        invalid = [index for index in self.layer_indices
                   if not 0 <= index < len(layers)]
        if invalid:
            raise ValueError(f"deep vision layers out of range: {invalid}")

        for index in self.layer_indices:
            key = str(index)

            def inject(_module, args, kwargs, *, adapter_key=key):
                prefix = self._prefix
                if prefix is None:
                    return args, kwargs
                if args:
                    hidden = args[0]
                else:
                    hidden = kwargs.get("hidden_states")
                if hidden is None:
                    raise RuntimeError("deep vision hook received no hidden states")
                if hidden.shape[0] != prefix.shape[0] or hidden.shape[1] < prefix.shape[1]:
                    raise ValueError("deep vision prefix does not match decoder input")
                injection = self.adapters[adapter_key](prefix).to(hidden.dtype)
                self._last_rms[adapter_key] = injection.detach().float().square().mean().sqrt()
                starts = self._starts or (0,) * hidden.shape[0]
                if len(set(starts)) == 1:
                    start = starts[0]
                    end = start + prefix.shape[1]
                    if start < 0 or end > hidden.shape[1]:
                        raise ValueError("deep vision span falls outside decoder input")
                    replaced = torch.cat((hidden[:, :start],
                                          hidden[:, start:end] + injection,
                                          hidden[:, end:]), dim=1)
                else:
                    replaced = hidden.clone()
                    for batch, start in enumerate(starts):
                        end = start + prefix.shape[1]
                        if start < 0 or end > hidden.shape[1]:
                            raise ValueError("deep vision span falls outside decoder input")
                        replaced[batch, start:end] = hidden[batch, start:end] + injection[batch]
                if args:
                    return (replaced, *args[1:]), kwargs
                updated = dict(kwargs)
                updated["hidden_states"] = replaced
                return args, updated

            self._handles.append(layers[index].register_forward_pre_hook(
                inject, with_kwargs=True))

    @contextmanager
    def use_prefix(self, prefix: torch.Tensor,
                   starts: int | Sequence[int] = 0) -> Iterator[None]:
        if self._prefix is not None:
            raise RuntimeError("deep vision prefix contexts cannot be nested")
        if isinstance(starts, int):
            normalized = (int(starts),) * prefix.shape[0]
        else:
            normalized = tuple(int(value) for value in starts)
        if len(normalized) != prefix.shape[0]:
            raise ValueError("deep vision starts must have one entry per batch row")
        self._prefix = prefix
        self._starts = normalized
        try:
            yield
        finally:
            self._prefix = None
            self._starts = ()

    def injection_rms(self) -> torch.Tensor:
        if self._last_rms:
            return torch.stack(list(self._last_rms.values())).square().mean().sqrt()
        return next(self.parameters()).new_zeros((), dtype=torch.float32)

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


class _LayerFeatureAdapter(nn.Module):
    """Project one pooled MoonViT stage into one RWKV visual span."""

    def __init__(self, hidden_size: int, rank: int):
        super().__init__()
        source = 4 * 1152
        self.norm = nn.LayerNorm(source)
        self.down = nn.Linear(source, rank, bias=False)
        self.act = nn.GELU()
        self.up = nn.Linear(rank, hidden_size, bias=False)
        # Adding this experiment to an existing checkpoint is an exact no-op.
        nn.init.zeros_(self.up.weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        flat = features.flatten(-2)
        return self.up(self.act(self.down(self.norm(flat))))


class LayerMatchedVisionInjector(nn.Module):
    """Inject different cached MoonViT stages at matching RWKV depths.

    ``features`` has shape ``[B, stages, visual_tokens, 4, 1152]``.  Stage i is
    projected only at site i, avoiding the old behavior of replaying the same
    final-layer visual prefix at every decoder depth.
    """

    def __init__(self, hidden_size: int, layer_indices: Sequence[int], *,
                 rank: int = 256):
        super().__init__()
        sites = tuple(sorted({int(index) for index in layer_indices}))
        if not sites or rank < 1:
            raise ValueError("layer-matched vision needs sites and a positive rank")
        self.layer_indices = sites
        self.adapters = nn.ModuleDict({
            str(index): _LayerFeatureAdapter(hidden_size, rank) for index in sites
        })
        self._features: torch.Tensor | None = None
        self._starts: tuple[int, ...] = ()
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._last_rms: dict[str, torch.Tensor] = {}

    def install(self, layers: Sequence[nn.Module]) -> None:
        if self._handles:
            raise RuntimeError("layer-matched vision injector is already installed")
        invalid = [index for index in self.layer_indices if not 0 <= index < len(layers)]
        if invalid:
            raise ValueError(f"layer-matched vision layers out of range: {invalid}")
        for stage, index in enumerate(self.layer_indices):
            key = str(index)

            def inject(_module, args, kwargs, *, adapter_key=key, stage_index=stage):
                features = self._features
                if features is None:
                    return args, kwargs
                hidden = args[0] if args else kwargs.get("hidden_states")
                if hidden is None:
                    raise RuntimeError("layer-matched vision hook received no hidden states")
                if hidden.shape[0] != features.shape[0]:
                    raise ValueError("layer-matched vision features do not match decoder input")
                injection = self.adapters[adapter_key](features[:, stage]).to(hidden.dtype)
                self._last_rms[adapter_key] = injection.detach().float().square().mean().sqrt()
                starts = self._starts
                if len(set(starts)) == 1:
                    start = starts[0]
                    end = start + injection.shape[1]
                    if start < 0 or end > hidden.shape[1]:
                        raise ValueError("layer-matched visual span falls outside decoder input")
                    replaced = torch.cat((hidden[:, :start],
                                          hidden[:, start:end] + injection,
                                          hidden[:, end:]), dim=1)
                else:
                    replaced = hidden.clone()
                    for batch, start in enumerate(starts):
                        end = start + injection.shape[1]
                        if start < 0 or end > hidden.shape[1]:
                            raise ValueError("layer-matched visual span falls outside decoder input")
                        replaced[batch, start:end] = hidden[batch, start:end] + injection[batch]
                if args:
                    return (replaced, *args[1:]), kwargs
                updated = dict(kwargs)
                updated["hidden_states"] = replaced
                return args, updated

            self._handles.append(layers[index].register_forward_pre_hook(
                inject, with_kwargs=True))

    @contextmanager
    def use_features(self, features: torch.Tensor,
                     starts: int | Sequence[int] = 0) -> Iterator[None]:
        if self._features is not None:
            raise RuntimeError("layer-matched vision contexts cannot be nested")
        if features.ndim != 5 or features.shape[1] != len(self.layer_indices):
            raise ValueError(
                f"expected [B,{len(self.layer_indices)},T,4,1152] layer features, "
                f"got {tuple(features.shape)}")
        normalized = ((int(starts),) * features.shape[0]
                      if isinstance(starts, int)
                      else tuple(int(value) for value in starts))
        if len(normalized) != features.shape[0]:
            raise ValueError("layer-matched starts must have one entry per batch row")
        self._features, self._starts = features, normalized
        try:
            yield
        finally:
            self._features, self._starts = None, ()

    def injection_rms(self) -> torch.Tensor:
        if self._last_rms:
            return torch.stack(list(self._last_rms.values())).square().mean().sqrt()
        return next(self.parameters()).new_zeros((), dtype=torch.float32)

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
