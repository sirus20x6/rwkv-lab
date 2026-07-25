"""Factored recurrent-depth adapters for FLA RWKV-7 TimeMix modules.

The generic :class:`LoopedRWKV` contract is a TimeMix contract.  This adapter
keeps FLA's four-value attention return intact while applying recurrence only
to each block's TimeMix module—not to the complete language model stack.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from rwkv_lab.looped_rwkv import LoopedRWKV


class _RefinementCache:
    """The attention-only recurrent state for one refinement pass.

    FLA's model-level cache has one slot per decoder layer and that slot is
    shared by TimeMix and ChannelMix.  Refinement passes must not write there:
    doing so would both corrupt the pass-1/FFN state and make the next token's
    refinement start from the wrong recurrence.  This deliberately tiny cache
    implements the subset of FLA's cache protocol used by RWKV7Attention and
    owns only this wrapper's recurrent and token-shift (``conv_state``) state.
    """

    def __init__(self, layer_idx: int):
        self.layer_idx = int(layer_idx)
        self.state: dict | None = None
        self.seen_tokens = 0

    def __len__(self) -> int:
        return self.layer_idx + 1 if self.state is not None else 0

    def __getitem__(self, layer_idx: int) -> dict:
        if int(layer_idx) != self.layer_idx or self.state is None:
            raise KeyError(layer_idx)
        return self.state

    def update(self, *, recurrent_state=None, conv_state=None,
               layer_idx: int = 0, offset: int = 1, **_kwargs) -> dict:
        if int(layer_idx) != self.layer_idx:
            raise ValueError(
                f"refinement cache belongs to layer {self.layer_idx}, got {layer_idx}")
        if self.state is None:
            self.state = {
                "recurrent_state": None,
                "attn_state": None,
                "conv_state": None,
                "ffn_state": None,
            }
        if recurrent_state is not None:
            self.state["recurrent_state"] = recurrent_state
        if conv_state is not None:
            self.state["conv_state"] = conv_state
        self.seen_tokens += int(offset or 0)
        return self.state

    def reset(self) -> None:
        self.state = None
        self.seen_tokens = 0


def _stack_cache_values(values: Sequence[Any], *, location: str) -> Any:
    """Recursively concatenate recurrent-cache tensors on their batch axis."""
    if not values:
        raise ValueError("cannot stack an empty cache value sequence")
    first = values[0]
    if first is None:
        if any(value is not None for value in values):
            raise ValueError(f"cache values disagree at {location}")
        return None
    if torch.is_tensor(first):
        if not all(torch.is_tensor(value) for value in values):
            raise ValueError(f"cache value types disagree at {location}")
        if first.ndim < 1:
            raise ValueError(f"cache tensor has no batch dimension at {location}")
        expected = tuple(first.shape[1:])
        for value in values[1:]:
            if tuple(value.shape[1:]) != expected:
                raise ValueError(
                    f"cache tensor shapes disagree at {location}: "
                    f"{tuple(first.shape)} vs {tuple(value.shape)}")
            if value.dtype != first.dtype or value.device != first.device:
                raise ValueError(f"cache tensor dtype/device disagree at {location}")
        return torch.cat(tuple(values), dim=0)
    if isinstance(first, dict):
        keys = tuple(first)
        if any(not isinstance(value, dict) or tuple(value) != keys
               for value in values[1:]):
            raise ValueError(f"cache mappings disagree at {location}")
        return {
            key: _stack_cache_values(
                [value[key] for value in values], location=f"{location}.{key}")
            for key in keys
        }
    if isinstance(first, (tuple, list)):
        if any(not isinstance(value, type(first)) or len(value) != len(first)
               for value in values[1:]):
            raise ValueError(f"cache sequences disagree at {location}")
        stacked = [
            _stack_cache_values(
                [value[index] for value in values],
                location=f"{location}[{index}]")
            for index in range(len(first))
        ]
        return tuple(stacked) if isinstance(first, tuple) else stacked
    if any(value != first for value in values[1:]):
        raise ValueError(f"cache metadata disagree at {location}")
    return first


def stack_legacy_fla_caches(caches: Sequence[Any]) -> Any:
    """Batch independently-prefilled ``LegacyFLACache``-protocol objects.

    RWKV recurrent and token-shift states carry batch in dimension zero. The
    cache's scalar ``seen_tokens`` cannot represent variable prompt lengths;
    RWKV recurrence is positionless and does not consult it during a direct
    model forward, so the composed cache retains the maximum solely for API
    compatibility. This helper is therefore for RWKV-style recurrent caches,
    not positional attention caches.
    """
    if not caches:
        raise ValueError("cannot stack zero FLA caches")
    layer_count = len(caches[0])
    if any(len(cache) != layer_count for cache in caches[1:]):
        raise ValueError("FLA caches have different layer counts")
    states = [
        _stack_cache_values(
            [cache[layer] for cache in caches], location=f"layer[{layer}]")
        for layer in range(layer_count)
    ]
    seen = max(int(getattr(cache, "_seen_tokens", 0)) for cache in caches)
    factory = getattr(type(caches[0]), "from_legacy_cache", None)
    if factory is None:
        raise TypeError("FLA cache type does not provide from_legacy_cache")
    try:
        # Installed LegacyFLACache releases accept only ``list`` here (a tuple
        # is silently treated as an empty cache); newer FLACache accepts both.
        result = factory(list(states), seen_tokens=seen)
    except TypeError as error:
        raise TypeError("FLA cache type cannot restore stacked legacy states") from error
    if len(result) != layer_count:
        raise RuntimeError(
            "FLA cache factory discarded stacked layer states during restore")
    return result


@dataclass(frozen=True)
class RefinementCacheEntry:
    """Captured attention-only state for one layer and refinement pass."""

    layer_idx: int
    seen_tokens: int
    state: dict[str, Any] | None


@dataclass(frozen=True)
class LoopRefinementCacheSnapshot:
    """One snapshot row per factored TimeMix wrapper."""

    wrappers: tuple[tuple[RefinementCacheEntry, ...], ...]


def capture_loop_refinement_caches(
        wrappers: Sequence["FLAFactoredTimeMix"], *, owner: Any
        ) -> LoopRefinementCacheSnapshot:
    """Capture refinement states before prefilling the next independent row."""
    captured = []
    for index, wrapper in enumerate(wrappers):
        if wrapper._cache_owner is not owner:
            raise ValueError(
                f"factored wrapper {index} does not belong to the supplied outer cache")
        expected = max(0, wrapper.loop.n_loops - 1)
        if len(wrapper._refinement_caches) != expected:
            raise ValueError(f"factored wrapper {index} has incomplete refinement caches")
        captured.append(tuple(RefinementCacheEntry(
            layer_idx=cache.layer_idx,
            seen_tokens=cache.seen_tokens,
            state=cache.state,
        ) for cache in wrapper._refinement_caches))
    return LoopRefinementCacheSnapshot(tuple(captured))


def stack_loop_refinement_cache_snapshots(
        snapshots: Sequence[LoopRefinementCacheSnapshot]
        ) -> LoopRefinementCacheSnapshot:
    """Stack independently captured refinement streams along batch dimension."""
    if not snapshots:
        raise ValueError("cannot stack zero refinement-cache snapshots")
    wrapper_count = len(snapshots[0].wrappers)
    if any(len(snapshot.wrappers) != wrapper_count for snapshot in snapshots[1:]):
        raise ValueError("refinement snapshots have different wrapper counts")
    wrappers = []
    for wrapper_index in range(wrapper_count):
        pass_count = len(snapshots[0].wrappers[wrapper_index])
        if any(len(snapshot.wrappers[wrapper_index]) != pass_count
               for snapshot in snapshots[1:]):
            raise ValueError("refinement snapshots have different pass counts")
        passes = []
        for pass_index in range(pass_count):
            entries = [snapshot.wrappers[wrapper_index][pass_index]
                       for snapshot in snapshots]
            layer_idx = entries[0].layer_idx
            if any(entry.layer_idx != layer_idx for entry in entries[1:]):
                raise ValueError("refinement snapshots have different layer indices")
            state = _stack_cache_values(
                [entry.state for entry in entries],
                location=f"refinement[{wrapper_index}][{pass_index}]")
            passes.append(RefinementCacheEntry(
                layer_idx=layer_idx,
                seen_tokens=max(entry.seen_tokens for entry in entries),
                state=state,
            ))
        wrappers.append(tuple(passes))
    return LoopRefinementCacheSnapshot(tuple(wrappers))


def restore_loop_refinement_caches(
        wrappers: Sequence["FLAFactoredTimeMix"],
        snapshot: LoopRefinementCacheSnapshot, *, owner: Any) -> None:
    """Install a captured/stacked snapshot under its exact outer-cache owner."""
    if owner is None:
        raise ValueError("refinement caches require a non-null outer-cache owner")
    if len(wrappers) != len(snapshot.wrappers):
        raise ValueError("refinement snapshot does not match wrapper count")
    for index, (wrapper, entries) in enumerate(zip(wrappers, snapshot.wrappers)):
        expected = max(0, wrapper.loop.n_loops - 1)
        if len(entries) != expected:
            raise ValueError(
                f"refinement snapshot wrapper {index} has the wrong pass count")
        caches = []
        for entry in entries:
            cache = _RefinementCache(entry.layer_idx)
            cache.state = entry.state
            cache.seen_tokens = int(entry.seen_tokens)
            caches.append(cache)
        wrapper._cache_owner = owner
        wrapper._refinement_caches = caches


class _FLATimeMixCore(nn.Module):
    def __init__(self, inner: nn.Module, hidden_size: int, num_heads: int):
        super().__init__()
        self.inner = inner
        self.hidden_size = hidden_size
        self.num_heads = num_heads

    def forward(self, hidden_states: torch.Tensor, *, v_first=None,
                return_v_first: bool = False, **kwargs):
        out, _attn, _past, next_v_first = self.inner(
            hidden_states=hidden_states, v_first=v_first, **kwargs)
        return (out, next_v_first) if return_v_first else out


class FLAFactoredTimeMix(nn.Module):
    """Drop-in FLA attention module with zero-init factored refinement."""
    def __init__(self, inner: nn.Module, *, hidden_size: int, num_heads: int,
                 n_loops: int = 2, gate_cap: float = 0.25,
                 loop_index: bool = True):
        super().__init__()
        core = _FLATimeMixCore(inner, hidden_size, num_heads)
        self.loop = LoopedRWKV(core, n_loops=n_loops, hidden_size=hidden_size,
                               gate_mode="factored", gate_cap=gate_cap,
                               loop_index=loop_index).float_gates()
        # Multiplied into the effective refinement gates by the trainer.  A
        # delayed loop must be introduced gradually: Adam's first normalized
        # update can otherwise move every layer's zero-init gate at once.
        self.loop.runtime_scale = 1.0
        self.enabled = False
        self._cache_owner = None
        self._refinement_caches: list[_RefinementCache] = []

    @property
    def inner(self) -> nn.Module:
        return self.loop.core.inner

    def reset_inference_cache(self) -> None:
        """Drop refinement-pass recurrence before starting a new sequence.

        A different outer FLA cache resets this automatically.  The explicit
        hook is useful to callers that reuse/reset a cache object in place.
        The caller remains responsible for resetting that outer cache too.
        """
        self._cache_owner = None
        self._refinement_caches.clear()

    def _prepare_refinement_caches(self, outer_cache) -> list[_RefinementCache]:
        if outer_cache is None:
            raise ValueError(
                "factored TimeMix use_cache=True requires the outer FLA cache")
        if self._cache_owner is not outer_cache:
            self.reset_inference_cache()
            self._cache_owner = outer_cache
        count = max(0, self.loop.n_loops - 1)
        if len(self._refinement_caches) != count:
            layer_idx = int(getattr(self.inner, "layer_idx", 0))
            self._refinement_caches = [
                _RefinementCache(layer_idx) for _ in range(count)
            ]
        return self._refinement_caches

    def _cached_forward(self, hidden_states: torch.Tensor, *, attention_mask,
                        past_key_values, output_attentions: bool, v_first,
                        cu_seqlens, **kwargs):
        """Incremental equivalent of the plain LoopedRWKV refinement path."""
        refinement_caches = self._prepare_refinement_caches(past_key_values)

        # Pass 1 is the normal decoder attention call.  It alone mutates the
        # model-level FLA cache; ChannelMix will subsequently add its own state.
        out, attentions, outer_cache, next_v_first = self.inner(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=output_attentions,
            v_first=v_first,
            cu_seqlens=cu_seqlens,
            **kwargs,
        )

        # Every refinement is itself a recurrent stream.  Maintaining one
        # independent cache per pass makes prefill + one-token decoding equal
        # to evaluating the same factored loop over the complete sequence.
        self.loop._loop_v_first = v_first
        try:
            for i, cache in enumerate(refinement_caches, start=1):
                inp = hidden_states + out
                if self.loop.loop_index:
                    inp = inp + self.loop.loop_index_embed[i].to(inp.dtype)
                if self.loop.lora_rank:
                    self.loop._lora_pass = i
                inc, _attn, _cache, _vf = self.inner(
                    hidden_states=self.loop.iter_norm(inp),
                    attention_mask=attention_mask,
                    past_key_values=cache,
                    use_cache=True,
                    output_attentions=False,
                    v_first=v_first,
                    cu_seqlens=cu_seqlens,
                    **kwargs,
                )
                if self.loop.cart_anchor:
                    out = torch.sigmoid(self.loop.cart_gate).to(inc.dtype) * out \
                        + self.loop._gate(i).to(inc.dtype) * inc
                else:
                    out = out + self.loop._gate(i).to(inc.dtype) * inc
        finally:
            self.loop._lora_pass = 0
        return out, attentions, outer_cache, next_v_first

    def forward(self, hidden_states: torch.Tensor, attention_mask=None,
                past_key_values=None, use_cache: bool = False,
                output_attentions: bool = False, v_first=None,
                cu_seqlens=None, **kwargs):
        common = dict(attention_mask=attention_mask,
                      past_key_values=past_key_values, use_cache=use_cache,
                      output_attentions=output_attentions,
                      cu_seqlens=cu_seqlens, **kwargs)
        if not self.enabled or self.loop.n_loops <= 1:
            return self.inner(hidden_states=hidden_states, v_first=v_first, **common)
        if use_cache:
            return self._cached_forward(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                v_first=v_first,
                cu_seqlens=cu_seqlens,
                **kwargs,
            )
        if output_attentions:
            raise ValueError("factored TimeMix training does not support attention outputs")
        out, next_v_first = self.loop(hidden_states, v_first=v_first,
                                      return_v_first=True, **common)
        return out, None, past_key_values, next_v_first


def install_factored_timemix(model: nn.Module, *, n_loops: int = 2,
                             gate_cap: float = 0.25,
                             loop_index: bool = True) -> list[FLAFactoredTimeMix]:
    wrappers = []
    hidden_size = int(model.config.hidden_size)
    num_heads = int(model.config.num_heads)
    for layer in model.model.layers:
        inner = layer.attn
        device = next(inner.parameters()).device
        wrapper = FLAFactoredTimeMix(inner, hidden_size=hidden_size,
                                     num_heads=num_heads, n_loops=n_loops,
                                     gate_cap=gate_cap, loop_index=loop_index)
        # The base model has already moved to CUDA before adapters are
        # installed. Newly constructed nn.Parameters otherwise stay on CPU and
        # the disabled warmup path hides the mismatch until loop activation.
        wrapper.to(device=device)
        wrapper.loop.float_gates()
        layer.attn = wrapper
        wrappers.append(wrapper)
    return wrappers


def set_loop_enabled(wrappers: list[FLAFactoredTimeMix], enabled: bool) -> None:
    for wrapper in wrappers:
        wrapper.enabled = bool(enabled)


def set_loop_scale(wrappers: list[FLAFactoredTimeMix], scale: float) -> None:
    scale = max(0.0, min(1.0, float(scale)))
    for wrapper in wrappers:
        wrapper.loop.runtime_scale = scale


def reset_loop_inference_cache(wrappers: list[FLAFactoredTimeMix]) -> None:
    """Reset the attention-only caches owned by refinement passes."""
    for wrapper in wrappers:
        wrapper.reset_inference_cache()


@torch.no_grad()
def reset_loop_adapters(wrappers: list[FLAFactoredTimeMix]) -> None:
    """Restore the trainable refinement path to its exact no-op initialization."""
    for wrapper in wrappers:
        loop = wrapper.loop
        loop.residual_weight.zero_()
        if loop.gate_mode == "factored":
            loop.gate_chan.zero_()
        if loop.loop_index:
            loop.loop_index_embed.zero_()
        loop.iter_norm.weight.fill_(1.0)


def loop_adapter_state(wrappers: list[FLAFactoredTimeMix]) -> list[dict[str, torch.Tensor]]:
    """Save only trainable adapter tensors; frozen RWKV weights stay external."""
    return [{name: value.detach().cpu() for name, value in wrapper.loop.state_dict().items()
             if not name.startswith("core.")} for wrapper in wrappers]


def load_loop_adapter_state(wrappers: list[FLAFactoredTimeMix], states: list[dict[str, torch.Tensor]]) -> None:
    if len(wrappers) != len(states):
        raise ValueError(f"loop checkpoint has {len(states)} layers, model has {len(wrappers)}")
    for layer, (wrapper, saved) in enumerate(zip(wrappers, states)):
        current = wrapper.loop.state_dict()
        expected = {name for name in current if not name.startswith("core.")}
        actual = set(saved)
        if actual != expected:
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            raise ValueError(
                f"loop checkpoint layer {layer} key mismatch; "
                f"missing={missing[:3]}, unexpected={unexpected[:3]}"
            )
        for name, value in saved.items():
            if current[name].shape != value.shape:
                raise ValueError(f"incompatible loop tensor layer {layer} {name}")
            current[name].copy_(value.to(device=current[name].device, dtype=current[name].dtype))


@torch.no_grad()
def loop_training_metrics(wrappers: list[FLAFactoredTimeMix]) -> dict[str, float]:
    """Small scalar telemetry for JSONL charts; refinement pass zero is unused."""
    if not wrappers or wrappers[0].loop.n_loops <= 1:
        return {}
    effective = torch.cat([wrapper.loop.effective_rw()[1:].reshape(-1)
                           for wrapper in wrappers])
    absolute = effective.abs()
    cap = float(wrappers[0].loop.gate_cap)
    channel = torch.cat([wrapper.loop.gate_chan[1:].reshape(-1)
                         for wrapper in wrappers])
    index_parts = [wrapper.loop.loop_index_embed[1:].reshape(-1)
                   for wrapper in wrappers if wrapper.loop.loop_index]
    index = torch.cat(index_parts) if index_parts else None
    gate_mean = absolute.mean()
    gate_rms = effective.square().mean().sqrt()
    gate_max = absolute.max()
    active_frac = (absolute > 1e-3).float().mean()
    channel_rms = channel.square().mean().sqrt()
    index_rms = (index.square().mean().sqrt()
                 if index is not None and index.numel() else gate_rms.new_zeros(()))
    # One device-to-host transfer for the whole telemetry bundle. Calling
    # float(cuda_scalar) for each field serializes a train step behind a string
    # of tiny reductions and was visible as an avoidable GPU launch gap.
    mean, rms, maximum, active, channel_value, index_value = torch.stack((
        gate_mean, gate_rms, gate_max, active_frac, channel_rms, index_rms,
    )).float().tolist()
    return {
        "loop_gate_abs_mean": mean,
        "loop_gate_rms": rms,
        "loop_gate_max": maximum,
        "loop_gate_active_frac": active,
        "loop_gate_cap_utilization": (rms / cap if cap > 0 else 0.0),
        "loop_gate_max_cap_utilization": (maximum / cap if cap > 0 else 0.0),
        "loop_channel_delta_rms": channel_value,
        "loop_index_rms": index_value,
    }


def loop_telemetry_from_states(states: list[dict[str, torch.Tensor]], *,
                               loop_count: int, gate_cap: float, step: int,
                               channel_buckets: int = 64,
                               runtime_scale: float = 1.0) -> dict:
    """Build the dashboard's per-layer loop artifact from adapter state dicts."""
    runtime_scale = max(0.0, min(1.0, float(runtime_scale)))
    layers = []
    for layer, state in enumerate(states):
        residual = state["residual_weight"].float()
        channel = state.get("gate_chan")
        active_head = residual[1:]
        if channel is not None:
            active_channel = channel[1:].float()
            ch_per_head = active_channel.shape[-1] // residual.shape[-1]
            raw = active_head.repeat_interleave(ch_per_head, dim=-1) * (1 + active_channel)
        else:
            active_channel = None
            ch_per_head = 1
            raw = active_head.repeat_interleave(ch_per_head, dim=-1)
        effective = ((gate_cap * torch.tanh(raw / gate_cap)
                      if gate_cap > 0 else raw) * runtime_scale).abs()
        head_effective = ((gate_cap * torch.tanh(active_head / gate_cap)
                           if gate_cap > 0 else active_head) * runtime_scale).abs()
        buckets = min(channel_buckets, effective.shape[-1])
        while effective.shape[-1] % buckets:
            buckets -= 1
        bucket_width = effective.shape[-1] // buckets
        channel_abs = effective.reshape(
            effective.shape[0], buckets, bucket_width).mean(-1)
        maximum = float(effective.max()) if effective.numel() else 0.0
        layers.append({
            "layer": layer,
            "max_rw": maximum,
            "rw": [float(row.max()) for row in effective],
            "split": {
                "heads": residual.shape[-1],
                "channels": effective.shape[-1],
                "ch_per_head": ch_per_head,
                "channel_buckets": buckets,
                "head_abs": head_effective.tolist(),
                "channel_abs": channel_abs.tolist(),
            },
        })
    maxima = [layer["max_rw"] for layer in layers]
    pin = gate_cap * 0.98 if gate_cap > 0 else 0.245
    return {
        "step": int(step),
        "loop_count": int(loop_count),
        "n_layers": len(layers),
        "n_pinned": sum(value >= pin for value in maxima),
        "mean_max_rw": sum(maxima) / max(1, len(maxima)),
        "gate_mode": "factored",
        "runtime_scale": runtime_scale,
        "layers": layers,
    }


@torch.no_grad()
def write_loop_telemetry(path: str | Path, wrappers: list[FLAFactoredTimeMix],
                         *, step: int) -> None:
    states = [{name: value.detach().cpu() for name, value in wrapper.loop.state_dict().items()
               if not name.startswith("core.")} for wrapper in wrappers]
    first = wrappers[0].loop
    payload = loop_telemetry_from_states(states, loop_count=first.n_loops,
                                         gate_cap=first.gate_cap, step=step,
                                         runtime_scale=first.runtime_scale)
    target = Path(path)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload) + "\n")
    os.replace(temporary, target)
