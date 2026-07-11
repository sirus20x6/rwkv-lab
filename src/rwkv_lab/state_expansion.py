"""StateX post-training state-expansion utilities.

StateX merges linear-attention heads so formerly disconnected off-diagonal
state blocks become usable, then post-trains only a uniformly selected subset
of recurrent layers.  See Shen et al., *StateX* (2026),
https://arxiv.org/abs/2509.22630 and the community lead
https://discord.com/channels/992359628979568762/992359629419991142/1524895529341816847

This module is a correctness-first transformation oracle.  RWKV's production
kernel fixes head geometry at construction, so applying it to a checkpoint
still requires a compatible expanded kernel; we do not silently relabel an
unchanged state as expanded.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class StateXPlan:
    n_layers: int
    expanded_layers: tuple[int, ...]
    old_heads: int
    merge_factor: int

    @property
    def new_heads(self) -> int:
        return self.old_heads // self.merge_factor

    @property
    def state_expansion(self) -> int:
        # Merging m square heads creates m-times as many state elements:
        # H*d^2 -> (H/m)*(m*d)^2.
        return self.merge_factor


def uniform_statex_plan(n_layers: int, old_heads: int, *, expanded_count: int = 4,
                        merge_factor: int | None = None) -> StateXPlan:
    """Select uniformly spaced layers, starting at layer zero, as in StateX §4.4."""
    if n_layers < 1 or old_heads < 1 or expanded_count < 1:
        raise ValueError("n_layers, old_heads, and expanded_count must be positive")
    expanded_count = min(expanded_count, n_layers)
    factor = old_heads if merge_factor is None else merge_factor
    if factor < 1 or old_heads % factor:
        raise ValueError("merge_factor must divide old_heads")
    layers = tuple((i * n_layers) // expanded_count for i in range(expanded_count))
    return StateXPlan(n_layers, layers, old_heads, factor)


def block_diagonal_merged_state(state: torch.Tensor, merge_factor: int) -> torch.Tensor:
    """Embed pretrained per-head states into merged dense heads without changing outputs.

    Input is ``[..., H, D, D]`` and output ``[..., H/m, mD, mD]``.  New
    cross-head blocks start at zero; post-training can make them useful through
    the merged Q/K projections.  This is the StateX linear-attention geometry.
    """
    if state.ndim < 3 or merge_factor < 1 or state.shape[-3] % merge_factor:
        raise ValueError("state must be [..., heads, d, d] and merge_factor divide heads")
    *prefix, heads, d1, d2 = state.shape
    if d1 != d2:
        raise ValueError("StateX head merge expects square per-head states")
    groups = heads // merge_factor
    src = state.reshape(*prefix, groups, merge_factor, d1, d1)
    out = state.new_zeros(*prefix, groups, merge_factor * d1, merge_factor * d1)
    for lane in range(merge_factor):
        sl = slice(lane * d1, (lane + 1) * d1)
        out[..., sl, sl] = src[..., lane, :, :]
    return out


def statex_receipt(plan: StateXPlan, head_dim: int) -> dict:
    old = plan.old_heads * head_dim * head_dim
    new = plan.new_heads * (head_dim * plan.merge_factor) ** 2
    return {"schema": "rwkv-lab.statex-plan.v1", "expanded_layers": list(plan.expanded_layers),
            "old_heads": plan.old_heads, "new_heads": plan.new_heads,
            "head_dim": head_dim, "state_elements_per_expanded_layer_before": old,
            "state_elements_per_expanded_layer_after": new,
            "expansion_factor": new / old,
            "paper_parameter_initialization": "reinitialize recurrent mixer; inherit FFN and embeddings",
            "requires_production_kernel_qualification": True}


def apply_statex_rwkv(model, *, expanded_count: int = 4) -> StateXPlan:
    """Replace selected native RWKV TimeMix blocks with single-head expanded blocks.

    The replacement realizes StateX's linear-attention head merge in the actual
    recurrent kernel: H states of shape D×D become one state of shape HD×HD.
    Following the paper's best GLA initialization, selected recurrent mixers are
    reinitialized while FFNs, embeddings, norms, and unselected layers remain
    untouched. Looped/wrapped or conversion-specific TimeMix variants are
    rejected rather than partially transformed.
    """
    from rwkv_lab.rwkv8_deltanet import RWKV8TimeMixDeltaNet

    if not getattr(model, "blocks", None):
        raise TypeError("expected a native RWKV model with blocks")
    first = model.blocks[0].att
    if not isinstance(first, RWKV8TimeMixDeltaNet):
        raise ValueError("StateX requires unwrapped native RWKV TimeMix blocks")
    plan = uniform_statex_plan(len(model.blocks), first.num_heads,
                               expanded_count=expanded_count)
    for index in plan.expanded_layers:
        old = model.blocks[index].att
        if not isinstance(old, RWKV8TimeMixDeltaNet):
            raise ValueError("StateX cannot partially transform wrapped TimeMix blocks")
        if old.num_heads != plan.old_heads or old.hidden_size != first.hidden_size:
            raise ValueError("StateX expects uniform native head geometry")
        if old.use_rope or old.comba_decouple or old.allow_neg_eigval:
            raise ValueError("StateX oracle currently supports clean native TimeMix only")
        parameter = next(old.parameters())
        expanded = RWKV8TimeMixDeltaNet(
            old.hidden_size, num_heads=1, head_size=old.hidden_size, layer_idx=index,
            depth_layer_id=index, depth_n_layer=max(len(model.blocks), 2),
            is_first_rwkv_layer=old.is_first_rwkv_layer, out_correct=old.out_correct,
            balance_state=old.balance_state, decay_cap_delta=old.decay_cap_delta,
        ).to(device=parameter.device, dtype=parameter.dtype)
        expanded.train(old.training)
        model.blocks[index].att = expanded
    model.statex_plan = plan
    return plan
