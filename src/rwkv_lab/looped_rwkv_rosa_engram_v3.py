"""
looped_rwkv_rosa_engram_v3.py

Research scaffold for a depth-looped RWKV-style language model with:

- Persistent token-time state.
- Mutable loop-time memory.
- Ephemeral hidden-state recurrence.
- Optional Engram and ROSA memory channels.
- Per-loop logits and distillation losses.
- Residual/state pre-alignment hooks.
- Loop-dependent token-mixer scheduling.
- Adaptive halting hooks.
- A progressive training curriculum.
- SMT one-step memory supervision for stable recurrent conversion.
- DMT closed-loop rollout recovery for exposure-bias correction.
- Teacher/student block-conversion wrappers for DeltaNet -> RWKV surgery.
- State-norm, drift, Jacobian-proxy, and update-ratio diagnostics.

This is executable PyTorch mixed with deliberately explicit pseudocode. It is
not an optimized or faithful RWKV implementation yet. Replace RWKVLikeMixer
with the exact RWKV-7/8 + ROSA + Engram implementation used by the real model.

Every research-derived idea is annotated with a source tag. Source tags are
defined below and include the paper title and canonical URL.

===============================================================================
SOURCE REGISTRY
===============================================================================

[SRC-LT2]
    "LT2: Linear-Time Looped Transformers"
    https://arxiv.org/abs/2605.20670

    Ideas used here:
    - Reusing a shared block stack over multiple depth loops.
    - Loop-specific residual gating.
    - Linear/recurrent token mixers as a natural partner for depth looping.
    - Mutable recurrent memory refinement over loop iterations.
    - Hybrid token mixers along depth or loop axes.
    - Block-level residual-stream pre-alignment.
    - Top-k per-loop logit distillation.
    - Progressive per-loop supervision followed by final-loop emphasis.
    - Progressive context-length expansion.

[SRC-DEEPER]
    "Thinking Deeper, Not Longer:
     Depth-Recurrent Transformers for Compositional Generalization"
    https://arxiv.org/abs/2603.21676

    Ideas used here:
    - Shared-weight depth recurrence.
    - Final-step-only "silent thinking" as an optional objective.
    - Pre-normalization and LayerScale initialized near zero.
    - Identity-biased gated recurrence.
    - Learned depth/loop embeddings.
    - Variable recurrence counts and adaptive computation.

[SRC-PLT]
    "Parallel Loop Transformer for Efficient Test-Time Computation Scaling"
    https://arxiv.org/abs/2510.24824

    Ideas represented as future hooks:
    - Cross-loop parallel execution.
    - Reusing global memory/cache from the first loop.
    - Lightweight local refinement in later loops.
    - Input reinjection across loops.

    This reference scaffold remains sequential. The comments mark where a
    production implementation could introduce cross-loop pipelining.

[SRC-MODR]
    "MoDR: Mixture-of-Depth-Recurrent Transformers for Test-Time Reasoning"
    https://openreview.net/forum?id=9Pba4rcQbE

    Ideas represented here:
    - Multiple low-rank recurrent branches.
    - Hard routing between depth-loop branches.
    - Training only lightweight branches and their router.
    - Load-balancing hooks for branch routing.

[SRC-PROXYKD]
    "Knowledge Distillation of Black-Box Large Language Models"
    https://arxiv.org/abs/2401.07013

    Ideas used here:
    - Hard target-response NLL plus proxy soft-target KL.
    - Confidence/sample weighting.
    - LoRA-friendly proxy/student adaptation.
    - A white-box proxy standing between a black-box teacher and student.


[SRC-SMT]
    "Supervised Memory Training for Recurrent Models"
    https://arxiv.org/abs/2606.06479

    Ideas used here:
    - Train recurrent memory transitions from explicit teacher memory labels.
    - Replace long BPTT paths with independent one-step memory supervision.
    - Follow SMT with Dynamical Memory Training (DMT) on student rollouts.
    - Diagnose and correct exposure-bias-driven recurrent state drift.
    - Sample memory positions rather than storing every token state.
    - Keep a predictive memory decoder/teacher separate from the recurrent updater.

[SRC-RWKV7]
    "RWKV-7 / Goose family technical materials"
    https://github.com/BlinkDL/RWKV-LM

    Ideas used here:
    - Recurrent token-time state rather than an attention KV cache.
    - A linear-time token mixer suitable for streaming.
    - DPLR-like recurrent-memory updates are discussed explicitly by LT2,
      which cites RWKV7 as a frontier linear-attention architecture.

[SRC-ROSA]
    ROSA implementation or paper used by the user's prior RWKV experiments.

    NOTE:
    The exact public paper/repository identifier was not supplied in this
    conversation. ROSA is therefore isolated behind a narrow interface.
    Replace this registry entry with the exact citation used by the project.

    Idea used here:
    - A separate exact/suffix-style retrieval channel whose state is not
      conflated with the neural recurrent state.

[SRC-ENGRAM]
    Engram implementation or paper used by the user's prior RWKV experiments.

    NOTE:
    The exact public paper/repository identifier was not supplied in this
    conversation. Engram is therefore isolated behind a narrow interface.
    Replace this registry entry with the exact citation used by the project.

    Idea used here:
    - A cheap static lexical/n-gram memory axis alongside neural computation.

===============================================================================
ARCHITECTURAL INVARIANT
===============================================================================

There are three distinct state classes:

1. TokenTimeState
       Persistent across real tokens.
       Advances exactly once per real token.

2. LoopMemoryState
       Mutable across internal depth loops for the current token/chunk.
       Initialized from TokenTimeState, refined for K loops, then committed.

3. DepthHiddenState
       Ordinary hidden representations updated through the shared loop core.
       Reset for each token/chunk.

The central rule is:

    A depth loop must not pretend that the same token arrived multiple times.

Instead, it refines a temporary recurrent memory and hidden representation,
then performs a single token-time commit.

This separation is an original synthesis motivated by:
- [SRC-LT2]'s iterative recurrent-memory refinement, and
- the need to preserve RWKV token-time semantics from [SRC-RWKV7].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# Configuration
# =============================================================================


class MixerKind(str, Enum):
    """
    Mixer choices for loop/depth hybridization.

    [SRC-LT2] studies hybridization both across depth and across loop
    iterations. Here "FULL_ATTENTION" is only a placeholder hook because the
    reference implementation is token-recurrent rather than sequence-parallel.
    """

    RWKV = "rwkv"
    LOCAL_REFINEMENT = "local_refinement"
    FULL_ATTENTION = "full_attention_placeholder"


@dataclass
class ModelConfig:
    vocab_size: int = 65536
    d_model: int = 1024

    n_prelude_layers: int = 2
    n_loop_layers: int = 2
    n_coda_layers: int = 2

    max_loops: int = 8
    min_loops: int = 1

    # [SRC-DEEPER] Identity-biased recurrence uses a negative gate bias.
    recurrence_gate_bias: float = -2.0

    # [SRC-DEEPER] LayerScale near zero protects state during early training.
    layerscale_init: float = 1e-4

    # [SRC-LT2] Zero/near-zero loop residual gates preserve an initial identity
    # path while repeated computation is still untrained.
    loop_residual_init: float = 0.0

    # [SRC-DEEPER] Learned loop/depth embeddings identify recurrence step.
    use_loop_embeddings: bool = True

    # [SRC-PLT] Reinjection of original input/embedding at later loops helps
    # preserve token identity and supports parallel loop formulations.
    use_input_reinjection: bool = True

    # Adaptive halting remains a research option.
    halt_threshold: float = 0.90
    use_adaptive_halting: bool = True

    # [SRC-MODR] Optional LoRA-style branches selected by a hard router.
    num_depth_branches: int = 1
    branch_rank: int = 16
    branch_balance_weight: float = 1e-2

    # [SRC-LT2] Loop-level mixer schedules.
    #
    # Example:
    #   ("rwkv", "rwkv", "local_refinement", "rwkv")
    #
    # Missing entries fall back to RWKV.
    loop_mixer_schedule: Tuple[str, ...] = ()

    use_engram: bool = True
    use_rosa: bool = True

    engram_buckets: int = 1 << 18
    engram_order: int = 3

    dropout: float = 0.0

    # Training objective coefficients.
    hard_nll_weight: float = 1.0
    proxy_kl_weight: float = 1.0
    per_loop_kl_weight: float = 1.0
    hidden_alignment_weight: float = 0.0
    state_alignment_weight: float = 0.0
    compute_penalty_weight: float = 1e-3

    # Distillation temperature.
    kd_temperature: float = 1.0

    # [SRC-LT2] Top-k proxy distillation.
    proxy_top_k: int = 64

    # [SRC-DEEPER] Optional final-loop-only supervision.
    final_loop_only: bool = False

    # [SRC-SMT] One-step supervised memory transition objective.
    smt_memory_weight: float = 1.0

    # [SRC-SMT] Closed-loop DMT rollout recovery objective.
    dmt_rollout_weight: float = 1.0

    # Conversion-specific block matching. [SRC-LT2]
    teacher_block_weight: float = 1.0

    # Stability penalties for recurrent conversion. These are engineering
    # safeguards motivated by the drift/exposure-bias analysis in [SRC-SMT].
    state_norm_weight: float = 1e-4
    state_update_ratio_weight: float = 1e-4
    max_state_update_ratio: float = 0.25


@dataclass
class TrainingCurriculum:
    """
    Explicit training-stage controls.

    [SRC-LT2]
        Motivates:
        - Stage 1 residual/block pre-alignment.
        - Stage 2 per-loop top-k logit distillation.
        - Progressive loop supervision.
        - Final-loop emphasis late in training.
        - Progressive context-length expansion.

    [SRC-DEEPER]
        Motivates:
        - Final-step-only silent-thinking ablation.
        - Variable recurrence-depth training.
    """

    context_lengths: Tuple[int, ...] = (512, 2048, 8192, 32768)
    loop_choices: Tuple[int, ...] = (1, 2, 4, 8)

    # Fractions of total training used to shape per-loop supervision.
    loop_warmup_fraction: float = 0.20
    uniform_loop_fraction: float = 0.50

    # After warmup/uniform stages, most weight moves to the final loop.
    final_loop_mass: float = 0.90

    def loop_weights(
        self,
        step: int,
        total_steps: int,
        num_loops: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """
        Produce per-loop loss weights.

        [SRC-LT2]
            Progressively warm up loop supervision, supervise loops uniformly
            for a substantial middle stage, then emphasize final output.

        [SRC-DEEPER]
            Setting all mass on the final loop provides the silent-thinking
            ablation.
        """
        if num_loops < 1:
            raise ValueError("num_loops must be positive")

        progress = step / max(total_steps - 1, 1)

        if progress < self.loop_warmup_fraction:
            # Warm in earlier loops gradually, but keep final loop dominant.
            alpha = progress / max(self.loop_warmup_fraction, 1e-8)
            uniform = torch.full(
                (num_loops,),
                1.0 / num_loops,
                device=device,
                dtype=dtype,
            )
            final = torch.zeros(
                num_loops,
                device=device,
                dtype=dtype,
            )
            final[-1] = 1.0
            weights = alpha * uniform + (1.0 - alpha) * final
        elif progress < self.uniform_loop_fraction:
            weights = torch.full(
                (num_loops,),
                1.0 / num_loops,
                device=device,
                dtype=dtype,
            )
        else:
            remaining = (1.0 - self.final_loop_mass) / max(num_loops - 1, 1)
            weights = torch.full(
                (num_loops,),
                remaining,
                device=device,
                dtype=dtype,
            )
            weights[-1] = self.final_loop_mass

        return weights / weights.sum()

    def choose_loops(self, device: torch.device) -> int:
        valid = [x for x in self.loop_choices if x >= 1]
        index = torch.randint(0, len(valid), (1,), device=device).item()
        return valid[index]


# =============================================================================
# State objects
# =============================================================================


@dataclass
class TokenTimeState:
    """
    Persistent state from prior real tokens.

    [SRC-RWKV7]
        RWKV maintains recurrent state across token time.

    Design constraint:
        This state advances once per real token, not once per internal loop.
    """

    prelude: List[Tensor]
    loop_context: List[Tensor]
    coda: List[Tensor]

    @staticmethod
    def zeros(
        cfg: ModelConfig,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "TokenTimeState":
        def make(count: int) -> List[Tensor]:
            return [
                torch.zeros(
                    batch_size,
                    cfg.d_model,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(count)
            ]

        return TokenTimeState(
            prelude=make(cfg.n_prelude_layers),
            loop_context=make(cfg.n_loop_layers),
            coda=make(cfg.n_coda_layers),
        )

    def detach(self) -> "TokenTimeState":
        return TokenTimeState(
            prelude=[x.detach() for x in self.prelude],
            loop_context=[x.detach() for x in self.loop_context],
            coda=[x.detach() for x in self.coda],
        )


@dataclass
class LoopMemoryState:
    """
    Temporary recurrent memory refined over depth loops.

    [SRC-LT2]
        Looping linear/DPLR token mixers can iteratively refine recurrent
        memory. LT2 explicitly identifies recurrent-memory refinement as a
        benefit of looping.

    Original synthesis:
        This temporary memory is initialized from token-time state, modified
        through loops, and committed once. That avoids treating one token as
        multiple temporal arrivals.
    """

    layers: List[Tensor]

    @staticmethod
    def from_token_state(state: TokenTimeState) -> "LoopMemoryState":
        return LoopMemoryState(
            layers=[x.clone() for x in state.loop_context]
        )


@dataclass
class DepthHiddenState:
    """
    Ephemeral hidden representations repeatedly transformed by shared weights.

    [SRC-DEEPER] and [SRC-LT2]
        Both use shared-weight recurrence to increase effective depth.
    """

    layers: List[Tensor]

    @staticmethod
    def initialize(cfg: ModelConfig, seed: Tensor) -> "DepthHiddenState":
        return DepthHiddenState(
            layers=[seed.clone() for _ in range(cfg.n_loop_layers)]
        )


@dataclass
class ForwardDiagnostics:
    loops_executed: int
    halt_probs: Tensor
    per_loop_logits: Optional[Tensor] = None
    branch_probabilities: Optional[Tensor] = None
    hidden_states: Optional[Tensor] = None
    loop_memory_states: Optional[Tensor] = None


# =============================================================================
# Utility layers
# =============================================================================


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        scale = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * scale * self.weight


class LayerScale(nn.Module):
    """
    Per-channel near-zero residual scale.

    [SRC-DEEPER]
        Uses LayerScale initialized around 1e-4 to keep early recurrent
        computation close to identity and protect fragile latent state.
    """

    def __init__(self, d_model: int, initial_value: float):
        super().__init__()
        self.scale = nn.Parameter(
            torch.full((d_model,), initial_value)
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.scale


class EngramMemory(nn.Module):
    """
    Minimal hashed n-gram memory placeholder.

    [SRC-ENGRAM]
        Represents a cheap static memory axis alongside neural computation.

    Replace this implementation with the exact Engram module used by the
    project, then update SOURCE REGISTRY with its canonical citation.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.order = cfg.engram_order
        self.buckets = cfg.engram_buckets
        self.table = nn.Embedding(cfg.engram_buckets, cfg.d_model)
        self.gate = nn.Parameter(torch.tensor(-2.0))

        constants = [1_000_003, 1_000_033, 1_000_037, 1_000_081]
        self.register_buffer(
            "hash_constants",
            torch.tensor(constants[: self.order], dtype=torch.long),
            persistent=False,
        )

    def forward(self, token_window: Tensor) -> Tensor:
        if token_window.ndim != 2 or token_window.size(1) != self.order:
            raise ValueError(
                f"Expected [batch, {self.order}], got {tuple(token_window.shape)}"
            )

        index = torch.remainder(
            (token_window.long() * self.hash_constants).sum(dim=-1),
            self.buckets,
        )
        return torch.sigmoid(self.gate) * self.table(index)


class ROSAMemory(nn.Module):
    """
    Narrow retrieval interface.

    [SRC-ROSA]
        Represents exact or suffix-style recall outside the recurrent neural
        state.

    Replace this placeholder with the actual ROSA implementation and update
    SOURCE REGISTRY with its canonical citation.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.gate = nn.Parameter(torch.tensor(-2.0))

    def forward(
        self,
        x: Tensor,
        retrieved: Optional[Tensor],
    ) -> Tensor:
        if retrieved is None:
            return torch.zeros_like(x)

        if retrieved.shape != x.shape:
            raise ValueError(
                f"retrieved must be {tuple(x.shape)}, got {tuple(retrieved.shape)}"
            )

        return torch.sigmoid(self.gate) * self.proj(retrieved)


# =============================================================================
# Token mixers
# =============================================================================


class RWKVLikeMixer(nn.Module):
    """
    Placeholder recurrent mixer with separate read and proposed-write state.

    [SRC-RWKV7]
        Represents RWKV's token-recurrent linear-time mixer.

    [SRC-LT2]
        LT2 identifies RWKV7 as a DPLR-style frontier linear mixer and argues
        that repeated loops can increase the effective rank of recurrent-memory
        updates.

    Replace with the exact RWKV kernel. The production interface should remain:

        output, proposed_memory = mixer(hidden, memory)

    During a depth loop, proposed_memory updates LoopMemoryState, not the
    persistent TokenTimeState.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model

        self.norm = RMSNorm(d)
        self.hidden_proj = nn.Linear(d, d, bias=False)
        self.memory_proj = nn.Linear(d, d, bias=False)
        self.receptance = nn.Linear(d, d, bias=False)
        self.value = nn.Linear(d, d, bias=False)

        self.channel_up = nn.Linear(d, 4 * d, bias=False)
        self.channel_down = nn.Linear(4 * d, d, bias=False)

        self.time_scale = LayerScale(d, cfg.layerscale_init)
        self.channel_scale = LayerScale(d, cfg.layerscale_init)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        hidden: Tensor,
        memory: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        h = self.norm(hidden)

        mixed = self.hidden_proj(h) + self.memory_proj(memory)
        recurrent_update = (
            torch.sigmoid(self.receptance(mixed))
            * self.value(mixed)
        )

        channel_update = self.channel_down(
            F.silu(self.channel_up(h))
        )

        output = (
            hidden
            + self.dropout(self.time_scale(recurrent_update))
            + self.dropout(self.channel_scale(channel_update))
        )

        # Pseudocode stand-in for a real RWKV/DPLR state update.
        proposed_memory = 0.95 * memory + 0.05 * output

        return output, proposed_memory


class LocalRefinementMixer(nn.Module):
    """
    Cheap later-loop refinement placeholder.

    [SRC-PLT]
        Later loops combine shared global representation with lightweight local
        sliding-window refinement.

    [SRC-LT2]
        Sparse/local token mixers can be mixed with linear mixers across loop
        iterations.

    This single-token reference cannot perform a real sequence window. It
    therefore approximates the hook with a gated MLP. Replace it with a
    sequence-aware local mixer in the batched implementation.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.norm = RMSNorm(d)
        self.gate = nn.Linear(d, d)
        self.update = nn.Sequential(
            nn.Linear(d, 2 * d),
            nn.SiLU(),
            nn.Linear(2 * d, d),
        )
        self.scale = LayerScale(d, cfg.layerscale_init)

    def forward(
        self,
        hidden: Tensor,
        memory: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        h = self.norm(hidden)
        update = torch.sigmoid(self.gate(h)) * self.update(h)
        output = hidden + self.scale(update)

        # Local refinement does not need to overwrite global memory strongly.
        proposed_memory = 0.99 * memory + 0.01 * output
        return output, proposed_memory


class FullAttentionPlaceholder(nn.Module):
    """
    Explicit failure hook for future hybrid attention.

    [SRC-LT2]
        A small fraction of full-attention layers can improve recall/quality in
        a looped linear architecture.

    This token-by-token scaffold has no sequence tensor here, so exact full
    attention cannot be implemented honestly. The exception prevents silently
    pretending that an MLP is full attention.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()

    def forward(
        self,
        hidden: Tensor,
        memory: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError(
            "FULL_ATTENTION requires a sequence-level implementation."
        )


# =============================================================================
# Mixture-of-depth branches
# =============================================================================


class LowRankBranch(nn.Module):
    """
    LoRA-style branch delta.

    [SRC-MODR]
        MoDR adds multiple LoRA branches around a shared recurrent core and
        trains only the branches plus a hard router.
    """

    def __init__(self, d_model: int, rank: int):
        super().__init__()
        self.down = nn.Linear(d_model, rank, bias=False)
        self.up = nn.Linear(rank, d_model, bias=False)

        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.up(self.down(x))


class DepthBranchRouter(nn.Module):
    """
    Hard branch router with straight-through gradients.

    [SRC-MODR]
        Uses hard-gate routing and load balancing over lightweight recurrent
        branches.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_branches = cfg.num_depth_branches
        self.router = nn.Linear(cfg.d_model, self.num_branches)

    def forward(
        self,
        x: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        logits = self.router(x)
        probabilities = torch.softmax(logits, dim=-1)

        hard_index = probabilities.argmax(dim=-1)
        hard = F.one_hot(
            hard_index,
            num_classes=self.num_branches,
        ).to(probabilities.dtype)

        # Straight-through hard routing.
        assignment = hard + probabilities - probabilities.detach()
        return assignment, probabilities


# =============================================================================
# Shared depth core
# =============================================================================


class SharedDepthLayer(nn.Module):
    """
    One shared recurrent depth layer.

    Combines:
    - A selected token mixer.
    - Identity-biased gated recurrence.
    - Optional MoDR-style low-rank branches.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.mixers = nn.ModuleDict(
            {
                MixerKind.RWKV.value: RWKVLikeMixer(cfg),
                MixerKind.LOCAL_REFINEMENT.value: LocalRefinementMixer(cfg),
                MixerKind.FULL_ATTENTION.value: FullAttentionPlaceholder(cfg),
            }
        )

        # [SRC-DEEPER] Identity-biased gated recurrence.
        self.recurrence_gate = nn.Linear(cfg.d_model, cfg.d_model)
        nn.init.constant_(
            self.recurrence_gate.bias,
            cfg.recurrence_gate_bias,
        )

        # [SRC-LT2] Per-loop residual gate, initialized at or near zero.
        self.loop_residual_scale = nn.Parameter(
            torch.full((cfg.d_model,), cfg.loop_residual_init)
        )

        self.branches = nn.ModuleList(
            [
                LowRankBranch(cfg.d_model, cfg.branch_rank)
                for _ in range(cfg.num_depth_branches)
            ]
        )
        self.branch_router = (
            DepthBranchRouter(cfg)
            if cfg.num_depth_branches > 1
            else None
        )

    def forward(
        self,
        hidden: Tensor,
        loop_memory: Tensor,
        mixer_kind: MixerKind,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        candidate, proposed_memory = self.mixers[mixer_kind.value](
            hidden,
            loop_memory,
        )

        branch_probabilities = None

        if self.branch_router is not None:
            assignment, branch_probabilities = self.branch_router(hidden)
            branch_outputs = torch.stack(
                [branch(hidden) for branch in self.branches],
                dim=-2,
            )
            branch_delta = (
                assignment.unsqueeze(-1) * branch_outputs
            ).sum(dim=-2)
            candidate = candidate + branch_delta
        elif self.branches:
            candidate = candidate + self.branches[0](hidden)

        # [SRC-DEEPER] Identity-biased gate.
        gate = torch.sigmoid(self.recurrence_gate(hidden))
        gated = hidden + gate * (candidate - hidden)

        # [SRC-LT2] Additional learned per-loop residual.
        output = gated + self.loop_residual_scale * hidden

        return output, proposed_memory, branch_probabilities


class SharedDepthCore(nn.Module):
    """
    Shared recurrent stack run K times.

    [SRC-LT2] and [SRC-DEEPER]
        Repeatedly reuse the same parameters to trade compute for effective
        depth.

    [SRC-PLT]
        Input reinjection is kept explicit. A future sequence implementation
        can shift/pipeline loop states to enable cross-loop parallelism.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.layers = nn.ModuleList(
            [SharedDepthLayer(cfg) for _ in range(cfg.n_loop_layers)]
        )

        self.input_reinject = nn.ModuleList(
            [
                nn.Linear(cfg.d_model, cfg.d_model, bias=False)
                for _ in range(cfg.n_loop_layers)
            ]
        )

        self.loop_embedding = (
            nn.Embedding(cfg.max_loops, cfg.d_model)
            if cfg.use_loop_embeddings
            else None
        )

    def mixer_for_loop(self, loop_index: int) -> MixerKind:
        """
        [SRC-LT2]
            Supports loop-level heterogeneous mixer schedules.
        """
        if loop_index < len(self.cfg.loop_mixer_schedule):
            return MixerKind(self.cfg.loop_mixer_schedule[loop_index])
        return MixerKind.RWKV

    def forward_one_loop(
        self,
        depth_state: DepthHiddenState,
        loop_memory: LoopMemoryState,
        prelude_seed: Tensor,
        loop_index: int,
    ) -> Tuple[
        DepthHiddenState,
        LoopMemoryState,
        Optional[Tensor],
    ]:
        mixer_kind = self.mixer_for_loop(loop_index)

        x = depth_state.layers[0]
        new_hidden_layers: List[Tensor] = []
        new_memory_layers: List[Tensor] = []
        branch_probs: List[Tensor] = []

        for layer_index, layer in enumerate(self.layers):
            layer_input = x

            if self.cfg.use_input_reinjection:
                # [SRC-PLT] Original embedding/input reinjection across loops.
                layer_input = (
                    layer_input
                    + self.input_reinject[layer_index](prelude_seed)
                )

            if self.loop_embedding is not None:
                # [SRC-DEEPER] Learned depth embedding.
                layer_input = (
                    layer_input
                    + self.loop_embedding.weight[loop_index]
                )

            x, proposed_memory, probs = layer(
                layer_input,
                loop_memory.layers[layer_index],
                mixer_kind,
            )

            new_hidden_layers.append(x)
            new_memory_layers.append(proposed_memory)

            if probs is not None:
                branch_probs.append(probs)

        stacked_probs = (
            torch.stack(branch_probs, dim=0)
            if branch_probs
            else None
        )

        return (
            DepthHiddenState(new_hidden_layers),
            LoopMemoryState(new_memory_layers),
            stacked_probs,
        )


class AdaptiveHaltingHead(nn.Module):
    """
    Optional compute-allocation head.

    [SRC-DEEPER]
        Motivated by variable recurrence depth and adaptive computation.

    Hard halting is intended for inference. Training should first use sampled
    fixed loop counts or a differentiable ACT-style objective.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, 1)
        nn.init.constant_(self.head.bias, cfg.recurrence_gate_bias)

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(
            self.head(self.norm(x))
        ).squeeze(-1)


# =============================================================================
# Full model
# =============================================================================


class LoopedRWKVLanguageModel(nn.Module):
    """
    Prelude -> shared loop core -> coda language model.

    [SRC-LT2], [SRC-DEEPER]
        Shared depth recurrence.

    [SRC-RWKV7]
        Persistent recurrent token-time state.

    [SRC-ROSA], [SRC-ENGRAM]
        Independent retrieval/static-memory channels.

    [SRC-MODR]
        Optional routed low-rank depth branches.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)

        self.engram = EngramMemory(cfg) if cfg.use_engram else None
        self.rosa = ROSAMemory(cfg) if cfg.use_rosa else None
        # grokking diagnostics: when _grok_log is set, forward_token stashes the
        # ROSA/Engram injection magnitudes (relative to the token embedding) into
        # _grok_stats — the "is the recall path grokking on" signal a trainer emits.
        self._grok_log = False
        self._grok_stats: dict = {}

        self.memory_fusion = nn.Sequential(
            RMSNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

        self.prelude = nn.ModuleList(
            [RWKVLikeMixer(cfg) for _ in range(cfg.n_prelude_layers)]
        )
        self.depth_core = SharedDepthCore(cfg)
        self.halting = AdaptiveHaltingHead(cfg)
        self.coda = nn.ModuleList(
            [RWKVLikeMixer(cfg) for _ in range(cfg.n_coda_layers)]
        )

        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(
            cfg.d_model,
            cfg.vocab_size,
            bias=False,
        )
        self.lm_head.weight = self.embedding.weight

    def initial_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> TokenTimeState:
        return TokenTimeState.zeros(
            self.cfg,
            batch_size,
            device,
            dtype,
        )

    def _ngram_window(
        self,
        input_ids: Tensor,
        position: int,
    ) -> Tensor:
        assert self.engram is not None

        order = self.engram.order
        start = max(0, position - order + 1)
        window = input_ids[:, start : position + 1]

        if window.size(1) < order:
            pad = torch.zeros(
                window.size(0),
                order - window.size(1),
                device=window.device,
                dtype=window.dtype,
            )
            window = torch.cat([pad, window], dim=1)

        return window

    def pop_grok_stats(self) -> dict:
        """Return and clear the latest ROSA/Engram injection stats (populated by
        forward_token when self._grok_log is True). Empty dict when logging is off."""
        s = self._grok_stats
        self._grok_stats = {}
        return s

    def forward_token(
        self,
        token_id: Tensor,
        token_state: TokenTimeState,
        *,
        token_window: Optional[Tensor] = None,
        rosa_retrieved: Optional[Tensor] = None,
        forced_loops: Optional[int] = None,
        adaptive_inference: bool = False,
        retain_alignment_tensors: bool = False,
        return_loop_logits: bool = False,
    ) -> Tuple[
        Tensor,
        TokenTimeState,
        ForwardDiagnostics,
    ]:
        """
        Process one real token.

        State timeline:
            prior TokenTimeState
                -> provisional prelude writes
                -> initialize LoopMemoryState
                -> K shared depth loops
                -> one final token-time commit
                -> coda writes

        The mutable LoopMemoryState implements the main v2 change.
        """
        x = self.embedding(token_id)
        memory_terms = [x]
        eng_inj = ros_inj = None

        if self.engram is not None:
            if token_window is None:
                raise ValueError(
                    "token_window is required when Engram is enabled"
                )
            eng_inj = self.engram(token_window)
            memory_terms.append(eng_inj)

        if self.rosa is not None:
            ros_inj = self.rosa(x, rosa_retrieved)
            memory_terms.append(ros_inj)

        if self._grok_log:
            from . import grokking_metrics as gm
            self._grok_stats = gm.injection_stats(rosa_inj=ros_inj, engram_inj=eng_inj, ref=x)

        x = x + self.memory_fusion(
            torch.stack(memory_terms).sum(dim=0)
        )

        # Prelude advances once per real token. [SRC-RWKV7]
        next_prelude: List[Tensor] = []

        for layer, old_memory in zip(
            self.prelude,
            token_state.prelude,
        ):
            x, proposed_memory = layer(x, old_memory)
            next_prelude.append(proposed_memory)

        prelude_seed = x

        # Temporary mutable loop memory. [SRC-LT2]
        loop_memory = LoopMemoryState.from_token_state(token_state)

        # Ephemeral hidden state. [SRC-DEEPER]
        depth_state = DepthHiddenState.initialize(
            self.cfg,
            prelude_seed,
        )

        max_requested = (
            self.cfg.max_loops
            if forced_loops is None
            else forced_loops
        )
        max_requested = max(
            self.cfg.min_loops,
            min(max_requested, self.cfg.max_loops),
        )

        halt_probs: List[Tensor] = []
        per_loop_logits: List[Tensor] = []
        all_branch_probs: List[Tensor] = []
        hidden_snapshots: List[Tensor] = []
        memory_snapshots: List[Tensor] = []

        for loop_index in range(max_requested):
            (
                depth_state,
                loop_memory,
                branch_probs,
            ) = self.depth_core.forward_one_loop(
                depth_state,
                loop_memory,
                prelude_seed,
                loop_index,
            )

            loop_hidden = depth_state.layers[-1]

            # Per-loop vocabulary projection is useful for KD but dominates
            # ordinary inference. Keep it entirely out of the fast path.
            if return_loop_logits:
                per_loop_logits.append(self.lm_head(self.final_norm(loop_hidden)))

            halt_prob = self.halting(loop_hidden)
            halt_probs.append(halt_prob)

            if branch_probs is not None:
                all_branch_probs.append(branch_probs)

            if retain_alignment_tensors:
                # [SRC-LT2] Residual-stream pre-alignment.
                hidden_snapshots.append(
                    torch.stack(depth_state.layers, dim=0)
                )

                # Original extension for recurrent models:
                # align temporary recurrent state as well as hidden outputs.
                memory_snapshots.append(
                    torch.stack(loop_memory.layers, dim=0)
                )

            if (
                forced_loops is None
                and adaptive_inference
                and self.cfg.use_adaptive_halting
                and loop_index + 1 >= self.cfg.min_loops
                and bool(
                    torch.all(
                        halt_prob >= self.cfg.halt_threshold
                    ).item()
                )
            ):
                break

        final_hidden = depth_state.layers[-1]

        # Commit temporary loop memory once to token-time state.
        #
        # Original synthesis informed by [SRC-LT2] + [SRC-RWKV7].
        next_loop_context = loop_memory.layers

        # Coda advances once per token. [SRC-RWKV7]
        next_coda: List[Tensor] = []
        x = final_hidden

        for layer, old_memory in zip(
            self.coda,
            token_state.coda,
        ):
            x, proposed_memory = layer(x, old_memory)
            next_coda.append(proposed_memory)

        final_logits = self.lm_head(self.final_norm(x))

        next_state = TokenTimeState(
            prelude=next_prelude,
            loop_context=next_loop_context,
            coda=next_coda,
        )

        diagnostics = ForwardDiagnostics(
            loops_executed=len(halt_probs),
            halt_probs=torch.stack(halt_probs, dim=0),
            per_loop_logits=(torch.stack(per_loop_logits, dim=0)
                             if per_loop_logits else None),
            branch_probabilities=(
                torch.stack(all_branch_probs, dim=0)
                if all_branch_probs
                else None
            ),
            hidden_states=(
                torch.stack(hidden_snapshots, dim=0)
                if hidden_snapshots
                else None
            ),
            loop_memory_states=(
                torch.stack(memory_snapshots, dim=0)
                if memory_snapshots
                else None
            ),
        )

        return final_logits, next_state, diagnostics

    def forward(
        self,
        input_ids: Tensor,
        *,
        forced_loops: Optional[int] = None,
        adaptive_inference: bool = False,
        rosa_retrievals: Optional[Tensor] = None,
        initial_state: Optional[TokenTimeState] = None,
        retain_alignment_tensors: bool = False,
        return_loop_logits: bool = False,
    ) -> Tuple[
        Tensor,
        TokenTimeState,
        Dict[str, Tensor],
    ]:
        """
        Reference sequence implementation.

        [SRC-PLT]
            This remains sequential. A production path could pipeline loop k
            for token t-k alongside loop 1 for token t, but that requires a
            custom batching and recurrent-state scheduler.

        Returns:
            final_logits:
                [batch, sequence, vocab]

            final_state:
                TokenTimeState

            diagnostics:
                per_loop_logits [batch, sequence, loops, vocab]
                halt_probs     [batch, sequence, loops]
                mean_loops
                optional alignment tensors
        """
        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape [batch, sequence]"
            )

        batch_size, sequence_length = input_ids.shape

        token_state = initial_state or self.initial_state(
            batch_size,
            input_ids.device,
            self.embedding.weight.dtype,
        )

        final_logits: List[Tensor] = []
        token_diagnostics: List[ForwardDiagnostics] = []

        for position in range(sequence_length):
            token_window = (
                self._ngram_window(input_ids, position)
                if self.engram is not None
                else None
            )

            retrieved = (
                rosa_retrievals[:, position]
                if rosa_retrievals is not None
                else None
            )

            logits, token_state, diagnostics = self.forward_token(
                input_ids[:, position],
                token_state,
                token_window=token_window,
                rosa_retrieved=retrieved,
                forced_loops=forced_loops,
                adaptive_inference=adaptive_inference,
                retain_alignment_tensors=retain_alignment_tensors,
                return_loop_logits=return_loop_logits,
            )

            final_logits.append(logits)
            token_diagnostics.append(diagnostics)

        # Adaptive inference can produce different loop counts per position.
        # This reference pads to the maximum observed loop count.
        max_loops = max(
            item.loops_executed for item in token_diagnostics
        )

        def pad_loop_tensor(
            tensor: Tensor,
            target_loops: int,
            loop_dim: int,
        ) -> Tensor:
            current = tensor.size(loop_dim)
            if current == target_loops:
                return tensor

            pad_shape = list(tensor.shape)
            pad_shape[loop_dim] = target_loops - current
            pad = torch.zeros(
                *pad_shape,
                device=tensor.device,
                dtype=tensor.dtype,
            )
            return torch.cat([tensor, pad], dim=loop_dim)

        per_token_loop_logits = []
        per_token_halts = []

        for item in token_diagnostics:
            # Input shapes:
            #   logits [loops, batch, vocab]
            #   halts [loops, batch]
            if item.per_loop_logits is not None:
                loop_logits = pad_loop_tensor(
                    item.per_loop_logits,
                    max_loops,
                    loop_dim=0,
                ).permute(1, 0, 2)
                per_token_loop_logits.append(loop_logits)

            halts = pad_loop_tensor(
                item.halt_probs,
                max_loops,
                loop_dim=0,
            ).permute(1, 0)

            per_token_halts.append(halts)

        output_diagnostics: Dict[str, Tensor] = {
            "halt_probs": torch.stack(
                per_token_halts,
                dim=1,
            ),
            "mean_loops": torch.tensor(
                sum(x.loops_executed for x in token_diagnostics)
                / len(token_diagnostics),
                device=input_ids.device,
            ),
        }
        if per_token_loop_logits:
            output_diagnostics["per_loop_logits"] = torch.stack(
                per_token_loop_logits, dim=1
            )

        return (
            torch.stack(final_logits, dim=1),
            token_state,
            output_diagnostics,
        )


# =============================================================================
# Losses
# =============================================================================


def top_k_kl_divergence(
    student_logits: Tensor,
    teacher_logits: Tensor,
    *,
    top_k: int,
    temperature: float,
) -> Tensor:
    """
    Top-k KL distillation.

    [SRC-LT2]
        Distills teacher/student logits at every loop using the teacher's top-k
        token set with renormalization.

    Shapes:
        student_logits, teacher_logits:
            [..., vocab]

    Returns:
        KL tensor with shape [...]
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            "student_logits and teacher_logits must have equal shape"
        )

    k = min(top_k, teacher_logits.size(-1))

    teacher_values, teacher_indices = torch.topk(
        teacher_logits,
        k=k,
        dim=-1,
    )
    student_values = torch.gather(
        student_logits,
        dim=-1,
        index=teacher_indices,
    )

    teacher_log_probs = F.log_softmax(
        teacher_values / temperature,
        dim=-1,
    )
    student_log_probs = F.log_softmax(
        student_values / temperature,
        dim=-1,
    )

    teacher_probs = teacher_log_probs.exp()

    kl = (
        teacher_probs
        * (teacher_log_probs - student_log_probs)
    ).sum(dim=-1)

    return kl * (temperature ** 2)


def load_balance_loss(
    branch_probabilities: Optional[Tensor],
) -> Tensor:
    """
    Simple soft load-balancing loss.

    [SRC-MODR]
        MoDR uses an auxiliary-loss-free balancing strategy. This scaffold uses
        a conventional differentiable approximation as a placeholder and marks
        it clearly rather than claiming exact reproduction.
    """
    if branch_probabilities is None:
        return torch.tensor(0.0)

    # Average over every dimension except branch.
    reduction_dims = tuple(
        range(branch_probabilities.ndim - 1)
    )
    mean_usage = branch_probabilities.mean(dim=reduction_dims)

    target = torch.full_like(
        mean_usage,
        1.0 / mean_usage.numel(),
    )
    return F.mse_loss(mean_usage, target)


def composite_training_loss(
    *,
    cfg: ModelConfig,
    curriculum: TrainingCurriculum,
    step: int,
    total_steps: int,
    final_logits: Tensor,
    labels: Tensor,
    diagnostics: Dict[str, Tensor],
    proxy_final_logits: Optional[Tensor] = None,
    proxy_per_loop_logits: Optional[Tensor] = None,
    sample_weights: Optional[Tensor] = None,
    student_hidden_states: Optional[Tensor] = None,
    teacher_hidden_states: Optional[Tensor] = None,
    student_loop_memory: Optional[Tensor] = None,
    teacher_loop_memory: Optional[Tensor] = None,
    branch_probabilities: Optional[Tensor] = None,
    ignore_index: int = -100,
) -> Dict[str, Tensor]:
    """
    Combined training objective.

    [SRC-PROXYKD]
        Hard response NLL + proxy soft-target KL + optional sample weighting.

    [SRC-LT2]
        Per-loop top-k KL with a progressive loop-weight schedule.
        Optional block/residual pre-alignment.

    [SRC-DEEPER]
        final_loop_only=True gives the silent-thinking ablation.

    Original extension:
        State-alignment loss for recurrent memory, because RWKV exposes a
        meaningful persistent/temporary state absent from standard transformers.
    """
    hard_per_token = F.cross_entropy(
        final_logits.reshape(-1, final_logits.size(-1)),
        labels.reshape(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).view_as(labels)

    valid_mask = labels.ne(ignore_index)

    if sample_weights is not None:
        # [SRC-PROXYKD] Confidence/sample weighting.
        if sample_weights.shape != labels.shape:
            raise ValueError(
                "sample_weights must match labels [batch, sequence]"
            )
        hard_per_token = hard_per_token * sample_weights

    hard_nll = hard_per_token[valid_mask].mean()

    total = cfg.hard_nll_weight * hard_nll
    losses: Dict[str, Tensor] = {
        "hard_nll": hard_nll.detach(),
    }

    if proxy_final_logits is not None:
        final_kl_per_token = top_k_kl_divergence(
            final_logits,
            proxy_final_logits,
            top_k=cfg.proxy_top_k,
            temperature=cfg.kd_temperature,
        )

        if sample_weights is not None:
            final_kl_per_token = (
                final_kl_per_token * sample_weights
            )

        final_kl = final_kl_per_token[valid_mask].mean()
        total = total + cfg.proxy_kl_weight * final_kl
        losses["proxy_final_kl"] = final_kl.detach()

    if proxy_per_loop_logits is not None:
        student_per_loop = diagnostics.get("per_loop_logits")
        if student_per_loop is None:
            raise ValueError("proxy_per_loop_logits requires model(..., return_loop_logits=True)")

        if student_per_loop.shape != proxy_per_loop_logits.shape:
            raise ValueError(
                "Student and proxy per-loop logits must have equal shape. "
                "For a non-looped proxy, repeat or schedule proxy targets "
                "explicitly before calling this function."
            )

        num_loops = student_per_loop.size(2)

        if cfg.final_loop_only:
            # [SRC-DEEPER] Silent-thinking objective.
            loop_weights = torch.zeros(
                num_loops,
                device=student_per_loop.device,
                dtype=student_per_loop.dtype,
            )
            loop_weights[-1] = 1.0
        else:
            # [SRC-LT2] Progressive per-loop supervision.
            loop_weights = curriculum.loop_weights(
                step,
                total_steps,
                num_loops,
                student_per_loop.device,
                student_per_loop.dtype,
            )

        per_loop_kl = top_k_kl_divergence(
            student_per_loop,
            proxy_per_loop_logits,
            top_k=cfg.proxy_top_k,
            temperature=cfg.kd_temperature,
        )
        # [batch, sequence, loops]

        valid_loop_mask = valid_mask.unsqueeze(-1)
        weighted_loop_kl = (
            per_loop_kl
            * loop_weights.view(1, 1, -1)
        )

        if sample_weights is not None:
            weighted_loop_kl = (
                weighted_loop_kl
                * sample_weights.unsqueeze(-1)
            )

        loop_kl = weighted_loop_kl[
            valid_loop_mask.expand_as(weighted_loop_kl)
        ].mean()

        total = total + cfg.per_loop_kl_weight * loop_kl
        losses["per_loop_kl"] = loop_kl.detach()

    if (
        student_hidden_states is not None
        and teacher_hidden_states is not None
    ):
        # [SRC-LT2] Residual-stream block pre-alignment.
        hidden_alignment = F.mse_loss(
            student_hidden_states,
            teacher_hidden_states,
        )
        total = (
            total
            + cfg.hidden_alignment_weight * hidden_alignment
        )
        losses["hidden_alignment"] = hidden_alignment.detach()

    if (
        student_loop_memory is not None
        and teacher_loop_memory is not None
    ):
        # Original recurrent extension to [SRC-LT2]'s block alignment.
        state_alignment = F.mse_loss(
            student_loop_memory,
            teacher_loop_memory,
        )
        total = (
            total
            + cfg.state_alignment_weight * state_alignment
        )
        losses["state_alignment"] = state_alignment.detach()

    branch_balance = load_balance_loss(branch_probabilities)
    if branch_balance.device != total.device:
        branch_balance = branch_balance.to(total.device)

    total = total + cfg.branch_balance_weight * branch_balance
    losses["branch_balance"] = branch_balance.detach()

    compute_penalty = (
        diagnostics["mean_loops"] / cfg.max_loops
    )
    total = (
        total
        + cfg.compute_penalty_weight * compute_penalty
    )
    losses["compute_penalty"] = compute_penalty.detach()

    losses["loss"] = total
    return losses


# =============================================================================
# Training-stage helpers
# =============================================================================


def make_repeated_proxy_loop_targets(
    proxy_final_logits: Tensor,
    num_loops: int,
) -> Tensor:
    """
    Repeat one proxy final distribution across all student loops.

    This is useful when the white-box proxy is not itself looped.

    This is a practical engineering fallback, not a claim from the papers.
    [SRC-LT2] uses corresponding loop outputs from a looped teacher. A stronger
    future approach would construct loop-specific targets by:
    - using a looped proxy,
    - temperature scheduling,
    - hidden-state teachers,
    - or self-distillation from the student's final loop.
    """
    return proxy_final_logits.unsqueeze(2).expand(
        *proxy_final_logits.shape[:2],
        num_loops,
        proxy_final_logits.size(-1),
    )


def recommended_stage(
    progress: float,
) -> str:
    """
    Human-readable multi-stage recipe.

    [SRC-LT2]
        Stage 1 block pre-alignment.
        Stage 2 hybrid/per-loop logit distillation.
        Stage 3 longer-context continuation.

    [SRC-PROXYKD]
        Hard black-box responses remain part of the objective.

    [SRC-DEEPER]
        Final-loop-only supervision is an explicit ablation, not assumed best
        for every task.
    """
    if progress < 0.10:
        return "block_and_state_prealignment"
    if progress < 0.60:
        return "uniform_per_loop_distillation"
    if progress < 0.85:
        return "final_loop_emphasis"
    return "long_context_recovery"


# =============================================================================
# DeltaNet -> RWKV conversion and SMT/DMT training helpers
# =============================================================================


@dataclass
class MemoryTransitionBatch:
    """
    One-step recurrent transition supervision.

    [SRC-SMT]
        Each example is independent:

            (teacher_memory_t, token_or_hidden_t+1) -> teacher_memory_t+1

        This removes the need to backpropagate through the complete prefix.

    Shapes are intentionally generic because a real Qwen Gated DeltaNet state
    and a real RWKV state may be vectors, matrices, or tuples. The first draft
    uses flat tensors after codec/projection.
    """

    memory_t: Tensor
    layer_input_t1: Tensor
    memory_t1: Tensor
    teacher_block_output_t1: Optional[Tensor] = None
    teacher_logits_t1: Optional[Tensor] = None
    sample_weight: Optional[Tensor] = None


@dataclass
class RolloutBatch:
    """
    Short closed-loop trajectory for Dynamical Memory Training.

    [SRC-SMT]
        SMT trains on teacher states. DMT then rolls the student on its own
        states and teaches it to recover toward the teacher trajectory.
    """

    initial_teacher_memory: Tensor
    layer_inputs: Tensor
    teacher_memories: Tensor
    teacher_block_outputs: Optional[Tensor] = None


class PredictiveMemoryCodec(nn.Module):
    """
    Canonical memory bottleneck shared between teacher state and RWKV state.

    [SRC-SMT]
        The paper trains a memory encoder/decoder so the memory is predictive of
        future observations instead of merely copying an arbitrary hidden state.

    For direct DeltaNet conversion, teacher_encoder can first be trained to
    encode the original DeltaNet state. Later, a future decoder can refine the
    bottleneck toward predictive sufficiency.
    """

    def __init__(
        self,
        teacher_state_dim: int,
        memory_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.teacher_encoder = nn.Sequential(
            nn.Linear(teacher_state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, memory_dim),
        )
        self.teacher_decoder = nn.Sequential(
            nn.Linear(memory_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, teacher_state_dim),
        )

    def encode_teacher_state(self, teacher_state: Tensor) -> Tensor:
        return self.teacher_encoder(teacher_state)

    def reconstruct_teacher_state(self, memory: Tensor) -> Tensor:
        return self.teacher_decoder(memory)


class RWKVReplacementCell(nn.Module):
    """
    Adapter around an RWKV-like mixer for replacing one Gated DeltaNet layer.

    [SRC-LT2]
        Block-level alignment trains a replacement token mixer to reproduce the
        original block's residual-stream output before global training.

    [SRC-SMT]
        The recurrent updater is additionally trained against explicit memory
        transition labels rather than relying only on downstream LM loss.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.input_norm = RMSNorm(cfg.d_model)
        self.cell = RWKVLikeMixer(cfg)
        self.output_gate = nn.Parameter(torch.tensor(-2.0))

    def forward(
        self,
        layer_input: Tensor,
        memory: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        normalized = self.input_norm(layer_input)
        candidate, next_memory = self.cell(normalized, memory)

        # Conservative residual interpolation prevents the newly inserted cell
        # from immediately perturbing the surrounding pretrained network.
        gate = torch.sigmoid(self.output_gate)
        output = layer_input + gate * (candidate - layer_input)
        return output, next_memory


class ConvertedLayerStudent(nn.Module):
    """
    Minimal training wrapper for one-at-a-time layer conversion.

    The full Qwen model should call this module at a selected layer index while
    every unrelated layer remains frozen. The untouched Qwen model supplies:
    - original block outputs for [SRC-LT2] alignment;
    - original DeltaNet states or learned predictive memories for [SRC-SMT];
    - final logits for ordinary distillation.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.replacement = RWKVReplacementCell(cfg)

    def one_step(
        self,
        layer_input_t1: Tensor,
        memory_t: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        return self.replacement(layer_input_t1, memory_t)

    def rollout(
        self,
        layer_inputs: Tensor,
        initial_memory: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Closed-loop rollout used by DMT.

        Args:
            layer_inputs: [batch, time, d_model]
            initial_memory: [batch, memory_dim]

        Returns:
            outputs: [batch, time, d_model]
            memories: [batch, time, memory_dim]
        """
        memory = initial_memory
        outputs: List[Tensor] = []
        memories: List[Tensor] = []

        for position in range(layer_inputs.size(1)):
            output, memory = self.one_step(
                layer_inputs[:, position],
                memory,
            )
            outputs.append(output)
            memories.append(memory)

        return (
            torch.stack(outputs, dim=1),
            torch.stack(memories, dim=1),
        )


@dataclass
class RecurrentStabilityMetrics:
    """
    Metrics intended to detect the "looks fine, then explodes" failure mode.

    The exact thresholds are engineering choices, not paper claims.
    Their motivation is the rollout drift and chaotic recurrent dynamics
    discussed in [SRC-SMT].
    """

    state_rms: Tensor
    state_max_abs: Tensor
    update_ratio: Tensor
    rollout_error_by_step: Optional[Tensor]
    finite_fraction: Tensor


def recurrent_stability_metrics(
    memories: Tensor,
    *,
    teacher_memories: Optional[Tensor] = None,
    eps: float = 1e-8,
) -> RecurrentStabilityMetrics:
    """
    Args:
        memories: [batch, time, memory_dim]
    """
    if memories.ndim < 3:
        raise ValueError("memories must be [batch, time, ...]")

    flat = memories.flatten(start_dim=2)
    state_rms = flat.pow(2).mean(dim=-1).sqrt()
    state_max_abs = flat.abs().amax(dim=-1)

    if memories.size(1) > 1:
        delta = flat[:, 1:] - flat[:, :-1]
        base = flat[:, :-1].norm(dim=-1).clamp_min(eps)
        update_ratio = delta.norm(dim=-1) / base
    else:
        update_ratio = torch.zeros(
            memories.size(0),
            0,
            device=memories.device,
            dtype=memories.dtype,
        )

    rollout_error = None
    if teacher_memories is not None:
        if teacher_memories.shape != memories.shape:
            raise ValueError("teacher_memories must match memories")
        rollout_error = (
            memories - teacher_memories
        ).flatten(start_dim=2).pow(2).mean(dim=-1).sqrt()

    finite_fraction = torch.isfinite(memories).float().mean()

    return RecurrentStabilityMetrics(
        state_rms=state_rms,
        state_max_abs=state_max_abs,
        update_ratio=update_ratio,
        rollout_error_by_step=rollout_error,
        finite_fraction=finite_fraction,
    )


def smt_transition_loss(
    *,
    cfg: ModelConfig,
    student: ConvertedLayerStudent,
    batch: MemoryTransitionBatch,
) -> Dict[str, Tensor]:
    """
    Independent one-step supervised-memory objective.

    [SRC-SMT]
        This is the main time-parallel training path. No long sequence graph is
        required once teacher memories have been generated.

    [SRC-LT2]
        Optional teacher block-output MSE pre-aligns the replacement layer.
    """
    student_output, predicted_memory_t1 = student.one_step(
        batch.layer_input_t1,
        batch.memory_t,
    )

    memory_error = (
        predicted_memory_t1 - batch.memory_t1
    ).pow(2).mean(dim=-1)

    block_error = torch.zeros_like(memory_error)
    if batch.teacher_block_output_t1 is not None:
        block_error = (
            student_output - batch.teacher_block_output_t1
        ).pow(2).mean(dim=-1)

    state_rms = predicted_memory_t1.flatten(start_dim=1).pow(2).mean(dim=-1).sqrt()
    prior_norm = batch.memory_t.flatten(start_dim=1).norm(dim=-1).clamp_min(1e-8)
    update_norm = (
        predicted_memory_t1 - batch.memory_t
    ).flatten(start_dim=1).norm(dim=-1)
    update_ratio = update_norm / prior_norm

    excessive_update = F.relu(
        update_ratio - cfg.max_state_update_ratio
    ).pow(2)

    weights = batch.sample_weight
    if weights is None:
        weights = torch.ones_like(memory_error)

    memory_loss = (memory_error * weights).mean()
    block_loss = (block_error * weights).mean()
    state_norm_penalty = state_rms.pow(2).mean()
    update_penalty = excessive_update.mean()

    total = (
        cfg.smt_memory_weight * memory_loss
        + cfg.teacher_block_weight * block_loss
        + cfg.state_norm_weight * state_norm_penalty
        + cfg.state_update_ratio_weight * update_penalty
    )

    return {
        "loss": total,
        "smt_memory": memory_loss.detach(),
        "teacher_block": block_loss.detach(),
        "state_norm_penalty": state_norm_penalty.detach(),
        "state_update_penalty": update_penalty.detach(),
        "mean_update_ratio": update_ratio.mean().detach(),
    }


def dmt_rollout_loss(
    *,
    cfg: ModelConfig,
    student: ConvertedLayerStudent,
    batch: RolloutBatch,
    discount: float = 1.0,
) -> Dict[str, Tensor]:
    """
    Closed-loop Dynamical Memory Training recovery.

    [SRC-SMT]
        The student begins from a teacher memory but then consumes its own
        predicted states. This explicitly trains against exposure-bias drift.
    """
    outputs, memories = student.rollout(
        batch.layer_inputs,
        batch.initial_teacher_memory,
    )

    if memories.shape != batch.teacher_memories.shape:
        raise ValueError("Student and teacher rollout memories must match")

    per_step_memory_error = (
        memories - batch.teacher_memories
    ).flatten(start_dim=2).pow(2).mean(dim=-1)

    steps = memories.size(1)
    step_weights = torch.pow(
        torch.tensor(
            discount,
            device=memories.device,
            dtype=memories.dtype,
        ),
        torch.arange(
            steps,
            device=memories.device,
            dtype=memories.dtype,
        ),
    )
    step_weights = step_weights / step_weights.sum()

    memory_loss = (
        per_step_memory_error
        * step_weights.view(1, -1)
    ).mean()

    block_loss = torch.zeros(
        (),
        device=memories.device,
        dtype=memories.dtype,
    )
    if batch.teacher_block_outputs is not None:
        block_loss = F.mse_loss(
            outputs,
            batch.teacher_block_outputs,
        )

    metrics = recurrent_stability_metrics(
        memories,
        teacher_memories=batch.teacher_memories,
    )

    if metrics.update_ratio.numel() > 0:
        update_penalty = F.relu(
            metrics.update_ratio - cfg.max_state_update_ratio
        ).pow(2).mean()
    else:
        update_penalty = torch.zeros_like(memory_loss)

    state_norm_penalty = metrics.state_rms.pow(2).mean()

    total = (
        cfg.dmt_rollout_weight * memory_loss
        + cfg.teacher_block_weight * block_loss
        + cfg.state_norm_weight * state_norm_penalty
        + cfg.state_update_ratio_weight * update_penalty
    )

    return {
        "loss": total,
        "dmt_memory": memory_loss.detach(),
        "teacher_block": block_loss.detach(),
        "state_norm_penalty": state_norm_penalty.detach(),
        "state_update_penalty": update_penalty.detach(),
        "finite_fraction": metrics.finite_fraction.detach(),
        "final_rollout_error": (
            metrics.rollout_error_by_step[:, -1].mean().detach()
            if metrics.rollout_error_by_step is not None
            else torch.zeros_like(memory_loss.detach())
        ),
    }


class LayerConversionSchedule:
    """
    Bookkeeping for incremental Gated DeltaNet -> RWKV surgery.

    This schedule is an engineering recommendation based on the user's prior
    one-layer-at-a-time experiments and the stabilization lessons from
    [SRC-LT2] and [SRC-SMT]. It is not directly prescribed by either paper.
    """

    def __init__(
        self,
        convertible_layer_indices: Sequence[int],
        consolidate_every: int = 3,
    ):
        if not convertible_layer_indices:
            raise ValueError("At least one convertible layer is required")
        self.convertible = list(convertible_layer_indices)
        self.consolidate_every = consolidate_every
        self.converted: List[int] = []

    def next_layer(self) -> Optional[int]:
        remaining = [
            index
            for index in reversed(self.convertible)
            if index not in self.converted
        ]
        return remaining[0] if remaining else None

    def mark_converted(self, layer_index: int) -> None:
        if layer_index not in self.convertible:
            raise ValueError("Layer is not marked convertible")
        if layer_index not in self.converted:
            self.converted.append(layer_index)

    def needs_consolidation(self) -> bool:
        return (
            bool(self.converted)
            and len(self.converted) % self.consolidate_every == 0
        )


def recommended_conversion_stage(
    *,
    smt_validation_error: float,
    dmt_final_error: float,
    finite_fraction: float,
    update_ratio: float,
    cfg: ModelConfig,
) -> str:
    """
    Conservative checkpoint gate before replacing another layer.

    Thresholds should be calibrated empirically. This function intentionally
    refuses progression on non-finite or high-update trajectories.
    """
    if finite_fraction < 1.0:
        return "rollback_nonfinite"
    if update_ratio > cfg.max_state_update_ratio:
        return "continue_dmt_update_instability"
    if dmt_final_error > smt_validation_error * 2.0:
        return "continue_dmt_rollout_drift"
    return "ready_for_neighborhood_consolidation"


# =============================================================================
# Smoke test
# =============================================================================


def smoke_test() -> None:
    """
    Tiny shape/gradient test.

    Run:
        python looped_rwkv_rosa_engram_v2.py

    The reference implementation is intentionally slow because it loops over
    sequence positions and depth in Python.
    """
    torch.manual_seed(0)

    cfg = ModelConfig(
        vocab_size=256,
        d_model=64,
        n_prelude_layers=1,
        n_loop_layers=1,
        n_coda_layers=1,
        max_loops=2,
        num_depth_branches=2,
        branch_rank=8,
        engram_buckets=1024,
        loop_mixer_schedule=("rwkv", "local_refinement"),
    )
    curriculum = TrainingCurriculum(
        context_lengths=(4,),
        loop_choices=(1, 2),
    )

    model = LoopedRWKVLanguageModel(cfg)

    input_ids = torch.randint(
        0,
        cfg.vocab_size,
        (1, 4),
    )
    labels = torch.roll(
        input_ids,
        shifts=-1,
        dims=1,
    )
    labels[:, -1] = -100

    loops = 2

    final_logits, _, diagnostics = model(
        input_ids,
        forced_loops=loops,
        return_loop_logits=True,
    )

    # Dummy proxy targets for plumbing verification only.
    proxy_final = final_logits.detach() + 0.01 * torch.randn_like(
        final_logits
    )
    proxy_per_loop = make_repeated_proxy_loop_targets(
        proxy_final,
        num_loops=loops,
    )

    losses = composite_training_loss(
        cfg=cfg,
        curriculum=curriculum,
        step=0,
        total_steps=100,
        final_logits=final_logits,
        labels=labels,
        diagnostics=diagnostics,
        proxy_final_logits=proxy_final,
        proxy_per_loop_logits=proxy_per_loop,
    )

    losses["loss"].backward()

    print("final logits:", tuple(final_logits.shape))
    print(
        "per-loop logits:",
        tuple(diagnostics["per_loop_logits"].shape),
    )
    print("mean loops:", float(diagnostics["mean_loops"]))
    print("loss:", float(losses["loss"]))


if __name__ == "__main__":
    smoke_test()
