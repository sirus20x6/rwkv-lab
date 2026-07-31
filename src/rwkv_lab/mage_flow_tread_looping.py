"""TREAD plus RWKV-style learned recurrent depth for custom MageFlow.

The local architecture remains::

    complete original MageFlow backbone -> exactly one terminal domain expert

TREAD only lets selected image tokens bypass a contiguous backbone span.  It
does not replace any block.  Every original block still executes for active
tokens, and bypassed tokens are restored before the coda.

Every backbone and resident-expert block also receives a zero-initialized
RWKV-style loop adapter.  A configured loop count is a maximum.  Learned
per-layer/per-pass head gates factored with per-channel deltas decide how much
each refinement contributes. Optional PonderNet heads learn an expected depth.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def _resolve_index(index: int, depth: int) -> int:
    return index if index >= 0 else depth + index


@dataclass(frozen=True)
class TreadConfig:
    enabled: bool = False
    route_start: int = 2
    route_end: int = -4
    route_fraction: float = 0.5
    image_tokens_only: bool = True
    selection: str = "random"
    same_mask_across_loop_iterations: bool = True
    min_active_image_tokens: int = 1
    active_timestep_min: float | None = None
    active_timestep_max: float | None = None
    inference_enabled: bool = False


@dataclass(frozen=True)
class FactorizationConfig:
    enabled: bool = False
    gate_mode: str = "factored"
    gate_cap: float = 0.25
    adaptive_halt: bool = True
    ponder_prior: float = 0.1
    ponder_weight: float = 0.01
    inference_gate_threshold: float = 1e-4


@dataclass(frozen=True)
class LoopingConfig:
    enabled: bool = False
    loop_count: int = 3
    loop_embedding: bool = True
    factorization: FactorizationConfig = field(default_factory=FactorizationConfig)
    use_auxiliary_loss: bool = True
    auxiliary_weight: float = 0.1
    auxiliary_schedule: str = "increasing"


@dataclass(frozen=True)
class CombinedConfig:
    enabled: bool = False
    tread_mask_policy: str = "fixed_per_forward"


@dataclass(frozen=True)
class TreadLoopConfig:
    tread: TreadConfig = field(default_factory=TreadConfig)
    looping: LoopingConfig = field(default_factory=LoopingConfig)
    combined: CombinedConfig = field(default_factory=CombinedConfig)

    @classmethod
    def combined_training_preset(cls) -> TreadLoopConfig:
        return cls(
            tread=TreadConfig(enabled=True),
            looping=LoopingConfig(
                enabled=True,
                factorization=FactorizationConfig(enabled=True),
            ),
            combined=CombinedConfig(enabled=True),
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> TreadLoopConfig:
        if value is None:
            return cls()
        tread = TreadConfig(**value.get("tread", {}))
        looping_value = dict(value.get("looping", {}))
        factor = FactorizationConfig(**looping_value.pop("factorization", {}))
        looping = LoopingConfig(factorization=factor, **looping_value)
        combined = CombinedConfig(**value.get("combined", {}))
        return cls(tread=tread, looping=looping, combined=combined)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def resolve_and_validate(
        self,
        *,
        path_depth: int,
        hidden_width: int,
        attention_heads: int,
    ) -> tuple[int, int]:
        enabled = (
            self.tread.enabled,
            self.looping.enabled,
            self.combined.enabled,
        )
        if not any(enabled):
            return 0, 0
        if not all(enabled):
            raise ValueError(
                "the combined preset enables TREAD and learned looping together"
            )
        start = _resolve_index(self.tread.route_start, path_depth)
        end = _resolve_index(self.tread.route_end, path_depth)
        if not 0 <= start < end < path_depth:
            raise ValueError("TREAD route must be nonempty and leave a coda")
        if end > 12:
            raise ValueError("TREAD routing is restricted to the original backbone")
        if not 0 <= self.tread.route_fraction < 1:
            raise ValueError("route_fraction must be in [0, 1)")
        if self.tread.min_active_image_tokens < 1:
            raise ValueError("min_active_image_tokens must be positive")
        if not self.tread.image_tokens_only:
            raise ValueError("TREAD initially routes image tokens only")
        if self.tread.selection != "random":
            raise ValueError("only seeded random TREAD selection is supported")
        if not self.tread.same_mask_across_loop_iterations:
            raise ValueError("one fixed TREAD mask is required per forward")
        if self.looping.loop_count < 1:
            raise ValueError("loop_count must be positive")
        factor = self.looping.factorization
        if not factor.enabled or factor.gate_mode != "factored":
            raise ValueError("learned looping requires factored head/channel gates")
        if hidden_width % attention_heads:
            raise ValueError("attention heads must divide hidden width")
        if factor.gate_cap <= 0:
            raise ValueError("gate_cap must be positive")
        if not 0 < factor.ponder_prior < 1:
            raise ValueError("ponder_prior must be in (0, 1)")
        if factor.ponder_weight < 0:
            raise ValueError("ponder_weight must be nonnegative")
        if factor.inference_gate_threshold < 0:
            raise ValueError("inference_gate_threshold must be nonnegative")
        if self.combined.tread_mask_policy != "fixed_per_forward":
            raise ValueError("combined mode requires one route mask per forward")
        return start, end


@dataclass
class TreadRouteState:
    original_sequence_length: int
    active_indices: torch.Tensor
    bypass_indices: torch.Tensor
    bypass_tokens: torch.Tensor
    batch_or_segment_ids: torch.Tensor
    original_positions: torch.Tensor
    modality_ids: torch.Tensor
    original_cu_lens: torch.Tensor
    active_cu_lens: torch.Tensor
    original_rotary_emb: torch.Tensor


def _normalized_cu_lens(
    cu_lens: torch.Tensor | None,
    *,
    token_count: int,
    device: torch.device,
) -> torch.Tensor:
    if cu_lens is None:
        return torch.tensor([0, token_count], device=device, dtype=torch.int32)
    result = cu_lens.to(device=device, dtype=torch.int32)
    if result.ndim != 1 or result.numel() < 2:
        raise ValueError("packed cumulative lengths must be one-dimensional")
    if int(result[0]) != 0 or int(result[-1]) != token_count:
        raise ValueError("packed cumulative lengths do not cover image tokens")
    return result


def extract_tread_route(
    image_tokens: torch.Tensor,
    image_rotary_emb: torch.Tensor,
    image_cu_lens: torch.Tensor | None,
    *,
    bypass_fraction: float,
    min_active_image_tokens: int = 1,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, TreadRouteState]:
    if image_tokens.ndim != 3 or image_tokens.shape[0] != 1:
        raise ValueError("TREAD expects packed image tokens [1, tokens, channels]")
    if not 0 <= bypass_fraction < 1:
        raise ValueError("bypass_fraction must be in [0, 1)")
    token_count = image_tokens.shape[1]
    if image_rotary_emb.shape[0] != token_count:
        raise ValueError("image RoPE and image token counts differ")
    cu_lens = _normalized_cu_lens(
        image_cu_lens,
        token_count=token_count,
        device=image_tokens.device,
    )
    active_parts: list[torch.Tensor] = []
    bypass_parts: list[torch.Tensor] = []
    segment_parts: list[torch.Tensor] = []
    active_lengths: list[int] = []
    for segment, raw_length in enumerate(cu_lens[1:] - cu_lens[:-1]):
        length = int(raw_length)
        if length < min_active_image_tokens:
            raise ValueError("a sample has fewer than min_active_image_tokens")
        bypass_count = min(
            int(bypass_fraction * length),
            length - min_active_image_tokens,
        )
        order = torch.randperm(
            length,
            device=image_tokens.device,
            generator=generator,
        )
        offset = int(cu_lens[segment])
        bypass_parts.append((order[:bypass_count] + offset).sort().values)
        active = (order[bypass_count:] + offset).sort().values
        active_parts.append(active)
        active_lengths.append(active.numel())
        segment_parts.append(
            torch.full(
                (length,),
                segment,
                dtype=torch.int32,
                device=image_tokens.device,
            )
        )
    active_indices = torch.cat(active_parts)
    bypass_indices = torch.cat(bypass_parts)
    active_cu_lens = torch.cat(
        [
            torch.zeros(1, device=image_tokens.device, dtype=torch.int32),
            torch.tensor(active_lengths, device=image_tokens.device).cumsum(
                0, dtype=torch.int32
            ),
        ]
    )
    state = TreadRouteState(
        original_sequence_length=token_count,
        active_indices=active_indices,
        bypass_indices=bypass_indices,
        bypass_tokens=image_tokens.index_select(1, bypass_indices),
        batch_or_segment_ids=torch.cat(segment_parts),
        original_positions=torch.arange(token_count, device=image_tokens.device),
        modality_ids=torch.ones(
            token_count,
            dtype=torch.int8,
            device=image_tokens.device,
        ),
        original_cu_lens=cu_lens,
        active_cu_lens=active_cu_lens,
        original_rotary_emb=image_rotary_emb,
    )
    return (
        image_tokens.index_select(1, active_indices),
        image_rotary_emb.index_select(0, active_indices),
        state,
    )


def restore_tread_route(
    active_tokens: torch.Tensor,
    state: TreadRouteState,
) -> torch.Tensor:
    if active_tokens.shape[1] != state.active_indices.numel():
        raise ValueError("active token count does not match route metadata")
    restored = torch.empty(
        (
            active_tokens.shape[0],
            state.original_sequence_length,
            active_tokens.shape[-1],
        ),
        dtype=active_tokens.dtype,
        device=active_tokens.device,
    )
    restored.index_copy_(1, state.active_indices, active_tokens)
    restored.index_copy_(
        1,
        state.bypass_indices,
        state.bypass_tokens.to(
            device=active_tokens.device,
            dtype=active_tokens.dtype,
        ),
    )
    return restored


BlockRunner = Callable[
    [nn.Module, torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
]


class LearnedFactoredLoopAdapter(nn.Module):
    """RWKV-style learned loop rate for one existing MageFlow block."""

    def __init__(
        self,
        *,
        hidden_width: int,
        attention_heads: int,
        loop_count: int,
        loop_embedding: bool,
        gate_cap: float,
        adaptive_halt: bool,
        ponder_prior: float,
        inference_gate_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_width = hidden_width
        self.attention_heads = attention_heads
        self.channels_per_head = hidden_width // attention_heads
        self.loop_count = loop_count
        self.gate_cap = gate_cap
        self.adaptive_halt = adaptive_halt and loop_count > 1
        self.ponder_prior = ponder_prior
        self.inference_gate_threshold = inference_gate_threshold
        self.factor_count = math.gcd(4, attention_heads, hidden_width)
        if self.factor_count < 1:
            raise ValueError("learned loop factor count must be positive")
        refinements = max(0, loop_count - 1)
        self.residual_weight = nn.Parameter(
            torch.zeros(refinements, attention_heads, dtype=torch.float32)
        )
        self.gate_chan = nn.Parameter(
            torch.zeros(refinements, hidden_width, dtype=torch.float32)
        )
        if loop_embedding:
            self.loop_index_embed = nn.Parameter(
                torch.zeros(refinements, hidden_width, dtype=torch.float32)
            )
        else:
            self.register_parameter("loop_index_embed", None)
        self.halt_heads = nn.ModuleList(
            nn.Linear(hidden_width, attention_heads)
            for _ in range(refinements)
        )
        for halt in self.halt_heads:
            nn.init.zeros_(halt.weight)
            nn.init.constant_(halt.bias, -2.0)
        self.last_ponder: torch.Tensor | None = None
        self.last_expected_loops: torch.Tensor | None = None
        self.last_image_iterates: tuple[torch.Tensor, ...] = ()
        self.last_executed_refinements = 0
        self.last_executed_factors = 0
        # Updated from the once-per-step CPU telemetry snapshot. Branching on
        # a live CUDA gate here would synchronize the device for every block
        # and refinement during evaluation.
        self._inference_skip_refinements = [
            inference_gate_threshold > 0
        ] * refinements
        self._inference_active_factors = [
            (
                ()
                if inference_gate_threshold > 0
                else tuple(range(self.factor_count))
            )
            for _ in range(refinements)
        ]

    def float_controls(self) -> LearnedFactoredLoopAdapter:
        self.residual_weight.data = self.residual_weight.data.float()
        self.gate_chan.data = self.gate_chan.data.float()
        if self.loop_index_embed is not None:
            self.loop_index_embed.data = self.loop_index_embed.data.float()
        for halt in self.halt_heads:
            halt.float()
        return self

    @torch.no_grad()
    def reset_controls(self) -> None:
        self.residual_weight.zero_()
        self.gate_chan.zero_()
        if self.loop_index_embed is not None:
            self.loop_index_embed.zero_()
        for halt in self.halt_heads:
            halt.weight.zero_()
            halt.bias.fill_(-2.0)

    def _gate(self, refinement: int, dtype: torch.dtype) -> torch.Tensor:
        head = self.residual_weight[refinement].repeat_interleave(
            self.channels_per_head
        )
        gate = head * (1.0 + self.gate_chan[refinement])
        gate = self.gate_cap * torch.tanh(gate / self.gate_cap)
        return gate.to(dtype).view(1, 1, -1)

    def _ponder_stream(
        self,
        states: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(states) == 1:
            one = states[0].new_ones(())
            return states[0], one, states[0].new_zeros(())
        probabilities: list[torch.Tensor] = []
        remainder = torch.ones(
            (*states[0].shape[:-1], self.attention_heads),
            device=states[0].device,
            dtype=torch.float32,
        )
        for index, halt in enumerate(self.halt_heads):
            rate = torch.sigmoid(halt(states[index].float()))
            probability = remainder * rate
            probabilities.append(probability)
            remainder = remainder * (1.0 - rate)
        probabilities.append(remainder)
        # Difference form is exactly identity when every zero-gated iterate
        # equals pass one, avoiding a roundoff-only architecture change.
        base = states[0]
        result = base
        for index in range(1, len(states)):
            channel_probability = probabilities[index].repeat_interleave(
                self.channels_per_head,
                dim=-1,
            )
            result = result + channel_probability.to(base.dtype) * (
                states[index] - base
            )
        expected = sum(
            float(index + 1) * probability
            for index, probability in enumerate(probabilities)
        ).mean()
        distribution = torch.stack(probabilities, dim=0).clamp_min(1e-8)
        prior = torch.tensor(
            [
                self.ponder_prior * (1 - self.ponder_prior) ** index
                for index in range(len(states))
            ],
            device=distribution.device,
            dtype=distribution.dtype,
        )
        prior = (prior / prior.sum()).clamp_min(1e-8)
        ponder = (
            distribution
            * (
                distribution.log()
                - prior.view(-1, *([1] * (distribution.ndim - 1))).log()
            )
        ).sum(0).mean()
        return result, expected, ponder

    def forward(
        self,
        block: nn.Module,
        image: torch.Tensor,
        text: torch.Tensor,
        temb: torch.Tensor,
        runner: BlockRunner,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text, image = runner(block, image, text, temb)
        self.last_executed_refinements = 0
        self.last_executed_factors = 0
        image_states = [image]
        text_states = [text]
        for refinement in range(self.loop_count - 1):
            gate = self._gate(refinement, image.dtype)
            active_factors = (
                tuple(range(self.factor_count))
                if self.training or self.inference_gate_threshold <= 0
                else self._inference_active_factors[refinement]
            )
            if not active_factors:
                image_states.append(image)
                text_states.append(text)
                continue
            loop_temb = temb
            if self.loop_index_embed is not None:
                loop_temb = temb + self.loop_index_embed[refinement].to(
                    temb.dtype
                ).unsqueeze(0)
            factor_runner = getattr(runner, "run_factors", None)
            if factor_runner is None:
                candidate_text, candidate_image = runner(
                    block,
                    image,
                    text,
                    loop_temb,
                )
            else:
                candidate_text, candidate_image = factor_runner(
                    block,
                    image,
                    text,
                    loop_temb,
                    self.factor_count,
                    active_factors,
                )
            self.last_executed_refinements += 1
            self.last_executed_factors += len(active_factors)
            if len(active_factors) != self.factor_count:
                channel_width = self.hidden_width // self.factor_count
                hard_mask = torch.zeros(
                    self.hidden_width,
                    dtype=gate.dtype,
                    device=gate.device,
                )
                for factor in active_factors:
                    start = factor * channel_width
                    hard_mask[start : start + channel_width] = 1
                gate = gate * hard_mask.view(1, 1, -1)
            image = image + gate * (candidate_image - image)
            text = text + gate * (candidate_text - text)
            image_states.append(image)
            text_states.append(text)
        self.last_image_iterates = tuple(image_states)
        if self.adaptive_halt:
            image, image_expected, image_ponder = self._ponder_stream(image_states)
            text, text_expected, text_ponder = self._ponder_stream(text_states)
            self.last_expected_loops = 0.5 * (image_expected + text_expected)
            self.last_ponder = 0.5 * (image_ponder + text_ponder)
        else:
            self.last_expected_loops = image.new_tensor(float(self.loop_count))
            self.last_ponder = image.new_zeros((), dtype=torch.float32)
        return text, image

    def metrics(self) -> dict[str, torch.Tensor]:
        if not self.residual_weight.numel():
            gate_rms = self.residual_weight.new_zeros(())
        else:
            gates = torch.stack(
                [
                    self._gate(index, torch.float32).flatten()
                    for index in range(self.loop_count - 1)
                ]
            )
            gate_rms = gates.square().mean().sqrt().detach()
        return {
            "gate_rms": gate_rms,
            "expected_loops": (
                self.last_expected_loops.detach()
                if self.last_expected_loops is not None
                else self.residual_weight.new_ones(())
            ),
        }


class TreadFactoredLoopController(nn.Module):
    """Learned loop controls for all 12 backbone and three expert blocks."""

    def __init__(self, transformer, config: TreadLoopConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone_depth = len(transformer.transformer_blocks)
        self.expert_depth = len(transformer.terminal_expert_blocks)
        self.path_depth = self.backbone_depth + self.expert_depth
        self.hidden_width = int(transformer.inner_dim)
        self.attention_heads = int(transformer.num_attention_heads)
        self.route_start, self.route_end = config.resolve_and_validate(
            path_depth=self.path_depth,
            hidden_width=self.hidden_width,
            attention_heads=self.attention_heads,
        )
        factor = config.looping.factorization
        adapter_kwargs = {
            "hidden_width": self.hidden_width,
            "attention_heads": self.attention_heads,
            "loop_count": config.looping.loop_count,
            "loop_embedding": config.looping.loop_embedding,
            "gate_cap": factor.gate_cap,
            "adaptive_halt": factor.adaptive_halt,
            "ponder_prior": factor.ponder_prior,
            "inference_gate_threshold": factor.inference_gate_threshold,
        }
        self.backbone_adapters = nn.ModuleList(
            LearnedFactoredLoopAdapter(**adapter_kwargs)
            for _ in range(self.backbone_depth)
        )
        self.expert_adapters = nn.ModuleList(
            LearnedFactoredLoopAdapter(**adapter_kwargs)
            for _ in range(self.expert_depth)
        )
        self.last_aux_predictions: tuple[torch.Tensor, ...] = ()
        self.last_ponder_loss: torch.Tensor | None = None
        self.last_metrics: dict[str, Any] = {}
        self._ponder_terms: list[torch.Tensor] = []
        self._start_event: torch.cuda.Event | None = None
        self._end_event: torch.cuda.Event | None = None

    def float_controls(self) -> TreadFactoredLoopController:
        for adapter in (*self.backbone_adapters, *self.expert_adapters):
            adapter.float_controls()
        return self

    @torch.no_grad()
    def refresh_inference_skip_refinements(self) -> None:
        """Calibrate hard dead-pass shortcuts once, outside the block hot path."""
        adapters = (*self.backbone_adapters, *self.expert_adapters)
        if not adapters or self.config.looping.loop_count <= 1:
            return
        gates = [
            torch.stack(
                [
                    adapter._gate(refinement, torch.float32).flatten()
                    for refinement in range(adapter.loop_count - 1)
                ]
            )
            for adapter in adapters
        ]
        cpu_gates = [gate.detach().abs().cpu() for gate in gates]
        for adapter, rows in zip(adapters, cpu_gates, strict=True):
            channel_width = adapter.hidden_width // adapter.factor_count
            active_by_refinement = []
            for row in rows:
                active = tuple(
                    factor
                    for factor in range(adapter.factor_count)
                    if float(
                        row[
                            factor * channel_width : (factor + 1) * channel_width
                        ].max()
                    )
                    >= adapter.inference_gate_threshold
                )
                active_by_refinement.append(active)
            adapter._inference_active_factors = active_by_refinement
            adapter._inference_skip_refinements = [
                not active for active in active_by_refinement
            ]

    def begin_forward(self, image: torch.Tensor) -> None:
        self._ponder_terms = []
        self.last_aux_predictions = ()
        if image.is_cuda:
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()

    def _run(
        self,
        adapter: LearnedFactoredLoopAdapter,
        block: nn.Module,
        image: torch.Tensor,
        text: torch.Tensor,
        temb: torch.Tensor,
        runner: BlockRunner,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text, image = adapter(block, image, text, temb, runner)
        if adapter.last_ponder is not None:
            self._ponder_terms.append(adapter.last_ponder)
        return text, image

    def run_backbone(
        self,
        index: int,
        block: nn.Module,
        image: torch.Tensor,
        text: torch.Tensor,
        temb: torch.Tensor,
        runner: BlockRunner,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._run(
            self.backbone_adapters[index],
            block,
            image,
            text,
            temb,
            runner,
        )

    def run_expert(
        self,
        index: int,
        block: nn.Module,
        image: torch.Tensor,
        text: torch.Tensor,
        temb: torch.Tensor,
        runner: BlockRunner,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._run(
            self.expert_adapters[index],
            block,
            image,
            text,
            temb,
            runner,
        )

    def finish_forward(
        self,
        *,
        total_image_tokens: int,
        active_image_tokens: int,
        bypassed_image_tokens: int,
        text_tokens: int,
    ) -> None:
        self.last_ponder_loss = (
            torch.stack(self._ponder_terms).mean()
            if self._ponder_terms
            else None
        )
        adapters = (*self.backbone_adapters, *self.expert_adapters)
        adapter_metrics = [adapter.metrics() for adapter in adapters]
        maximum_refinement_calls = (
            self.config.looping.loop_count - 1
        ) * len(adapters)
        executed_refinement_calls = sum(
            adapter.last_executed_refinements for adapter in adapters
        )
        executed_factor_calls = sum(
            adapter.last_executed_factors for adapter in adapters
        )
        self.last_metrics = {
            "total_image_tokens": total_image_tokens,
            "active_tread_image_tokens": active_image_tokens,
            "bypassed_image_tokens": bypassed_image_tokens,
            "text_tokens": text_tokens,
            "original_backbone_blocks_executed": self.backbone_depth,
            "expert_blocks_executed": self.expert_depth,
            "maximum_loops_per_block": self.config.looping.loop_count,
            "maximum_refinement_block_calls": maximum_refinement_calls,
            "executed_refinement_block_calls": executed_refinement_calls,
            "factor_count": (
                adapters[0].factor_count if adapters else 0
            ),
            "maximum_refinement_factor_calls": (
                maximum_refinement_calls * adapters[0].factor_count
                if adapters
                else 0
            ),
            "executed_refinement_factor_calls": executed_factor_calls,
            # Keep CUDA scalars asynchronous here. They are materialized once
            # per optimizer step by performance_metrics(), rather than forcing
            # two device synchronizations for every block in every microbatch.
            "mean_gate_rms": torch.stack(
                [row["gate_rms"] for row in adapter_metrics]
            ).mean(),
            "mean_expected_loops": torch.stack(
                [row["expected_loops"] for row in adapter_metrics]
            ).mean(),
        }
        if self._end_event is not None:
            self._end_event.record()

    def performance_metrics(self, *, synchronize: bool = False) -> dict[str, Any]:
        metrics = dict(self.last_metrics)
        tensor_items = [
            (key, value)
            for key, value in metrics.items()
            if isinstance(value, torch.Tensor)
        ]
        if tensor_items:
            values = torch.stack(
                [value.detach().float().reshape(()) for _, value in tensor_items]
            ).cpu()
            for (key, _value), value in zip(tensor_items, values, strict=True):
                metrics[key] = float(value)
        if self._start_event is not None and self._end_event is not None:
            if synchronize:
                self._end_event.synchronize()
            if self._end_event.query():
                metrics["looped_path_cuda_ms"] = float(
                    self._start_event.elapsed_time(self._end_event)
                )
        return metrics

    def expert_parameters(self):
        yield from self.expert_adapters.parameters()

    def gate_parameters(self):
        """Yield only controls that scale recurrent residual updates."""
        for adapter in (*self.backbone_adapters, *self.expert_adapters):
            yield adapter.residual_weight
            yield adapter.gate_chan

    def reset_expert_controls(self) -> None:
        for adapter in self.expert_adapters:
            adapter.reset_controls()

    def report(self) -> dict[str, Any]:
        return {
            "schema": "rwkv-lab.mage-flow-tread-learned-loop.v2",
            "route_start": self.route_start,
            "route_end": self.route_end,
            "original_backbone_blocks_executed": self.backbone_depth,
            "resident_expert_blocks_executed": self.expert_depth,
            "replaced_backbone_blocks": [],
            "config": self.config.to_dict(),
            "adapter_parameter_count": sum(
                parameter.numel() for parameter in self.parameters()
            ),
            **self.last_metrics,
        }


def install_tread_factored_looping(
    transformer,
    config: TreadLoopConfig | dict[str, Any] | None = None,
    *,
    offload_replaced_source_blocks: bool = False,
) -> TreadFactoredLoopController:
    if offload_replaced_source_blocks:
        raise ValueError("the corrected architecture never replaces backbone blocks")
    if not hasattr(transformer, "terminal_expert_blocks"):
        raise ValueError("install the one terminal expert before learned looping")
    if hasattr(transformer, "tread_loop_controller"):
        raise ValueError("TREAD/learned looping is already installed")
    if not isinstance(config, TreadLoopConfig):
        config = TreadLoopConfig.from_dict(config)
    if not (
        config.tread.enabled
        and config.looping.enabled
        and config.combined.enabled
    ):
        raise ValueError("do not install the controller for a baseline-off config")
    controller = TreadFactoredLoopController(transformer, config)
    reference = next(transformer.parameters())
    controller.to(device=reference.device, dtype=reference.dtype)
    controller.float_controls()
    transformer.add_module("tread_loop_controller", controller)
    return controller


def auxiliary_loop_flow_loss(
    controller: TreadFactoredLoopController,
    target_velocity: torch.Tensor,
) -> torch.Tensor:
    predictions = controller.last_aux_predictions
    if not predictions:
        return target_velocity.new_zeros((), dtype=torch.float32)
    losses = [
        F.mse_loss(prediction.float(), target_velocity.float(), reduction="mean")
        for prediction in predictions
    ]
    count = len(losses)
    weights = [float(index + 1) / float(count + 1) for index in range(count)]
    return sum(weight * loss for weight, loss in zip(weights, losses, strict=True))


def learned_loop_ponder_loss(
    controller: TreadFactoredLoopController,
) -> torch.Tensor:
    if controller.last_ponder_loss is None:
        reference = next(controller.parameters())
        return reference.new_zeros((), dtype=torch.float32)
    return controller.last_ponder_loss


@torch.no_grad()
def mageflow_loop_telemetry_payload(
    controller: TreadFactoredLoopController,
    *,
    step: int,
    resident_domain: str,
    channel_buckets: int = 64,
) -> dict[str, Any]:
    """Build the dashboard's RWKV-compatible detailed loop artifact."""

    adapters = [
        *(("backbone", index, adapter)
          for index, adapter in enumerate(controller.backbone_adapters)),
        *(("expert", index, adapter)
          for index, adapter in enumerate(controller.expert_adapters)),
    ]
    residuals = torch.stack(
        [adapter.residual_weight.detach() for _, _, adapter in adapters]
    ).float().cpu()
    channels = torch.stack(
        [adapter.gate_chan.detach() for _, _, adapter in adapters]
    ).float().cpu()
    expected_values = torch.stack(
        [
            (
                adapter.last_expected_loops.detach()
                if adapter.last_expected_loops is not None
                else adapter.residual_weight.new_ones(())
            )
            for _, _, adapter in adapters
        ]
    ).float().cpu()
    layers = []
    expected_depths = []
    for adapter_offset, (path, index, adapter) in enumerate(adapters):
        residual = residuals[adapter_offset]
        channel = channels[adapter_offset]
        repeated = residual.repeat_interleave(
            adapter.channels_per_head,
            dim=-1,
        )
        raw = repeated * (1.0 + channel)
        effective = (
            adapter.gate_cap * torch.tanh(raw / adapter.gate_cap)
            if adapter.gate_cap > 0
            else raw
        ).abs()
        head_effective = (
            adapter.gate_cap * torch.tanh(residual / adapter.gate_cap)
            if adapter.gate_cap > 0
            else residual
        ).abs()
        buckets = min(channel_buckets, adapter.hidden_width)
        while adapter.hidden_width % buckets:
            buckets -= 1
        bucket_width = adapter.hidden_width // buckets
        channel_abs = effective.reshape(
            effective.shape[0],
            buckets,
            bucket_width,
        ).mean(-1)
        expected = float(expected_values[adapter_offset])
        expected_depths.append(expected)
        layers.append(
            {
                "layer": len(layers),
                "label": f"B{index}" if path == "backbone" else f"E{index}",
                "path": path,
                "max_rw": float(effective.max()) if effective.numel() else 0.0,
                "rw": [float(row.max()) for row in effective],
                "expected_loops": expected,
                "split": {
                    "heads": adapter.attention_heads,
                    "channels": adapter.hidden_width,
                    "ch_per_head": adapter.channels_per_head,
                    "channel_buckets": buckets,
                    "head_abs": head_effective.tolist(),
                    "channel_abs": channel_abs.tolist(),
                },
            }
        )
    maxima = [float(layer["max_rw"]) for layer in layers]
    gate_cap = controller.config.looping.factorization.gate_cap
    pin = gate_cap * 0.98 if gate_cap > 0 else 0.245
    metrics = controller.performance_metrics()
    total_image_tokens = int(metrics.get("total_image_tokens", 0))
    active_image_tokens = int(metrics.get("active_tread_image_tokens", 0))
    return {
        "schema": "rwkv-lab.mage-flow-loop-telemetry.v1",
        "model_family": "mageflow",
        "step": int(step),
        "loop_count": controller.config.looping.loop_count,
        "n_layers": len(layers),
        "n_pinned": sum(maximum >= pin for maximum in maxima),
        "mean_max_rw": sum(maxima) / max(1, len(maxima)),
        "mean_gate_rms": float(metrics.get("mean_gate_rms", 0.0)),
        "mean_expected_loops": sum(expected_depths) / max(1, len(expected_depths)),
        "executed_refinement_calls": int(
            metrics.get("executed_refinement_block_calls", 0)
        ),
        "maximum_refinement_calls": int(
            metrics.get("maximum_refinement_block_calls", 0)
        ),
        "factor_count": int(metrics.get("factor_count", 0)),
        "executed_refinement_factor_calls": int(
            metrics.get("executed_refinement_factor_calls", 0)
        ),
        "maximum_refinement_factor_calls": int(
            metrics.get("maximum_refinement_factor_calls", 0)
        ),
        "tread_active_fraction": (
            active_image_tokens / total_image_tokens
            if total_image_tokens
            else 1.0
        ),
        "resident_domain": resident_domain,
        "gate_mode": "executable-factored-head-channel",
        "layers": layers,
    }


def write_mageflow_loop_telemetry(
    path: str | Path,
    controller: TreadFactoredLoopController,
    *,
    step: int,
    resident_domain: str,
) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = mageflow_loop_telemetry_payload(
        controller,
        step=step,
        resident_domain=resident_domain,
    )
    temporary = target.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def save_tread_loop_controller(
    controller: TreadFactoredLoopController,
    path,
) -> dict[str, Any]:
    from safetensors.torch import save_file

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    tensors = {
        name: value.detach().cpu().contiguous()
        for name, value in controller.state_dict().items()
    }
    metadata = {
        "schema": "rwkv-lab.mage-flow-tread-learned-loop.v2",
        "config_json": json.dumps(controller.config.to_dict(), sort_keys=True),
    }
    temporary = target.with_name(target.name + ".tmp")
    save_file(tensors, str(temporary), metadata=metadata)
    os.replace(temporary, target)
    return {
        **controller.report(),
        "path": str(target),
        "tensor_count": len(tensors),
        "copied_tensors": 0,
        "sliced_tensors": 0,
        "newly_initialized_tensors": len(tensors),
        "missing_tensors": [],
        "unexpected_tensors": [],
        "functional_equivalence": "exact at zero loop gates",
    }


def load_tread_loop_controller(
    controller: TreadFactoredLoopController,
    path,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    from safetensors import safe_open
    from safetensors.torch import load_file

    source = Path(path).expanduser().resolve()
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    if metadata.get("schema") != "rwkv-lab.mage-flow-tread-learned-loop.v2":
        raise ValueError("not a learned-loop MageFlow checkpoint")
    expected = json.dumps(controller.config.to_dict(), sort_keys=True)
    if metadata.get("config_json") != expected:
        raise ValueError("learned-loop checkpoint configuration mismatch")
    result = controller.load_state_dict(
        load_file(str(source), device="cpu"),
        strict=strict,
    )
    controller.float_controls()
    controller.refresh_inference_skip_refinements()
    return {
        "path": str(source),
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
        "compatible": not result.missing_keys and not result.unexpected_keys,
    }
