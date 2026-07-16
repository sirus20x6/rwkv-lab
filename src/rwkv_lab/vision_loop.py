"""Factored recurrent-depth adapters for FLA RWKV-7 TimeMix modules.

The generic :class:`LoopedRWKV` contract is a TimeMix contract.  This adapter
keeps FLA's four-value attention return intact while applying recurrence only
to each block's TimeMix module—not to the complete language model stack.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from torch import nn

from rwkv_lab.looped_rwkv import LoopedRWKV


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

    @property
    def inner(self) -> nn.Module:
        return self.loop.core.inner

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
        if use_cache or output_attentions:
            raise ValueError("factored TimeMix training does not support cache/attention outputs")
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
