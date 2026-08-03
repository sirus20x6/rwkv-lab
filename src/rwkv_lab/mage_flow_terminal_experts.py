"""Terminal, hard-routed experts for the pinned Microsoft Mage-Flow backbone.

The deployed topology is deliberately simple::

    complete released Mage-Flow transformer (12 blocks)
                         |
                 exactly one domain
                    /          \
          photo tail          animation tail
             (3 blocks)          (3 blocks)
                    \\          /
                 released output head

Only one tail is resident in accelerator memory.  The other expert is an
independent safetensors checkpoint kept in host memory or on disk.  This module
also contains the one-way migration helper for the earlier residual-FFN beta:
late residual image MLPs can seed the image MLP portion of the new full blocks,
while all other terminal-block tensors start from the released Mage-Flow tail.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import MethodType
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import nn

from rwkv_lab.mage_flow_adaptation import (
    EXPERT_DOMAINS,
    MAGE_FLOW_BASE_ID,
    MAGE_FLOW_BASE_REVISION,
)
from rwkv_lab.mage_flow_tread_looping import (
    extract_tread_route,
    restore_tread_route,
)
from rwkv_lab.training_components import (
    ParameterRouterImplementation,
    TerminalExpertRoutingConfiguration,
    build_registered_parameter_routing,
)
from rwkv_lab.training_parameter_routing import ParameterRoutingResult

if TYPE_CHECKING:
    from rwkv_lab.trainvm_adapters import WorkerTrainingComponents

TERMINAL_EXPERT_SCHEMA = "rwkv-lab.mage-flow-terminal-expert.v1"
TERMINAL_EXPERT_DEPTH = 3
LIGHTNING_BLOCK_SCHEMA = "rwkv-lab.mage-flow-lightning-blocks.v1"

_ANIMATION_CUE = re.compile(
    r"\b(?:anime|manga|manhwa|cartoon|comic|illustration|illustrated|"
    r"drawing|sketch|line[ -]?art|cel[ -]?shaded|animation|animated|"
    r"watercolou?r|gouache|oil painting|digital art|concept art|"
    r"pixel art|vector art|storybook|graphic novel|2d art|3d render|cgi)\b",
    re.IGNORECASE,
)
_PHOTO_CUE = re.compile(
    r"\b(?:photo|photograph|photography|photographic|photorealistic|"
    r"photo-realistic|realistic photo|real life|dslr|mirrorless|"
    r"shot on|camera|lens|film still|cinematic still|studio portrait|"
    r"documentary|editorial photo|product shot|macro photo)\b",
    re.IGNORECASE,
)


def _contiguous_factor_ranges(
    active_factors: tuple[int, ...],
) -> tuple[tuple[int, int], ...]:
    """Coalesce adjacent active factors without executing inactive slices."""

    if not active_factors:
        return ()
    ordered = tuple(sorted(set(active_factors)))
    ranges = []
    start = previous = ordered[0]
    for factor in ordered[1:]:
        if factor == previous + 1:
            previous = factor
            continue
        ranges.append((start, previous + 1))
        start = previous = factor
    ranges.append((start, previous + 1))
    return tuple(ranges)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


class LightningRMSNorm(nn.Module):
    """LightningDiT-compatible learnable RMSNorm with FP32 statistics."""

    def __init__(
        self,
        dim: int,
        *,
        eps: float = 1.0e-6,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if dim < 1 or eps <= 0:
            raise ValueError("RMSNorm dimension and epsilon must be positive")
        self.eps = float(eps)
        self.weight = nn.Parameter(
            torch.ones(dim, device=device, dtype=dtype)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = inputs.float() * torch.rsqrt(
            inputs.float().square().mean(dim=-1, keepdim=True) + self.eps
        )
        return normalized.to(dtype=inputs.dtype) * self.weight


def _legacy_swiglu_tensors(
    first_weight: torch.Tensor,
    first_bias: torch.Tensor | None,
    second_weight: torch.Tensor,
    second_bias: torch.Tensor | None,
    *,
    hidden_features: int,
) -> dict[str, torch.Tensor]:
    """Convert a dense GELU MLP into a parameter-matched SwiGLU initializer."""

    expansion = int(first_weight.shape[0])
    if (
        first_weight.ndim != 2
        or second_weight.ndim != 2
        or second_weight.shape[1] != expansion
        or hidden_features < 1
        or hidden_features > expansion
    ):
        raise ValueError("legacy MLP tensors are incompatible with SwiGLU")
    incoming = torch.linalg.vector_norm(
        first_weight, dim=1, dtype=torch.float32
    )
    outgoing = torch.linalg.vector_norm(
        second_weight, dim=0, dtype=torch.float32
    )
    selected = torch.topk(
        incoming * outgoing,
        k=hidden_features,
        largest=True,
        sorted=False,
    ).indices.sort().values
    gate_weight = first_weight.index_select(0, selected)
    value_weight = torch.zeros_like(gate_weight)
    w12_weight = torch.cat((gate_weight, value_weight), dim=0)
    if first_bias is None:
        w12_bias = None
    else:
        gate_bias = first_bias.index_select(0, selected)
        value_bias = torch.ones_like(gate_bias)
        w12_bias = torch.cat((gate_bias, value_bias), dim=0)
    return {
        "w12.weight": w12_weight,
        **({"w12.bias": w12_bias} if w12_bias is not None else {}),
        "w3.weight": second_weight.index_select(1, selected),
        **({"w3.bias": second_bias} if second_bias is not None else {}),
    }


class LightningSwiGLU(nn.Module):
    """Parameter-matched SwiGLU used by LightningDiT."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        *,
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.hidden_features = int(hidden_features)
        self.out_features = int(out_features)
        self.w12 = nn.Linear(
            in_features,
            2 * hidden_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.w3 = nn.Linear(
            hidden_features,
            out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )

    @classmethod
    def from_feed_forward(cls, module: nn.Module) -> LightningSwiGLU:
        linears = [
            child for child in module.modules() if isinstance(child, nn.Linear)
        ]
        if len(linears) != 2:
            raise ValueError(
                "Mage FeedForward conversion requires exactly two projections"
            )
        first, second = linears
        hidden_features = int((2 * first.out_features) / 3)
        converted = cls(
            first.in_features,
            hidden_features,
            second.out_features,
            bias=first.bias is not None and second.bias is not None,
            device=first.weight.device,
            dtype=first.weight.dtype,
        )
        tensors = _legacy_swiglu_tensors(
            first.weight.detach(),
            first.bias.detach() if first.bias is not None else None,
            second.weight.detach(),
            second.bias.detach() if second.bias is not None else None,
            hidden_features=hidden_features,
        )
        with torch.no_grad():
            converted.load_state_dict(tensors, strict=True)
        return converted

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        legacy_prefix = prefix + "net."
        legacy_names = {
            "first_weight": legacy_prefix + "0.proj.weight",
            "first_bias": legacy_prefix + "0.proj.bias",
            "second_weight": legacy_prefix + "2.weight",
            "second_bias": legacy_prefix + "2.bias",
        }
        if (
            prefix + "w12.weight" not in state_dict
            and legacy_names["first_weight"] in state_dict
        ):
            converted = _legacy_swiglu_tensors(
                state_dict[legacy_names["first_weight"]],
                state_dict.get(legacy_names["first_bias"]),
                state_dict[legacy_names["second_weight"]],
                state_dict.get(legacy_names["second_bias"]),
                hidden_features=self.hidden_features,
            )
            for name, value in converted.items():
                state_dict[prefix + name] = value
            for name in legacy_names.values():
                state_dict.pop(name, None)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        gate, value = self.w12(inputs).chunk(2, dim=-1)
        return self.w3(F.silu(gate) * value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class PromptRoute:
    domain: str
    reason: str
    animation_cues: tuple[str, ...] = ()
    photo_cues: tuple[str, ...] = ()


def route_prompt(
    prompt: str,
    *,
    override: str | None = None,
    default_domain: str = "photo",
) -> PromptRoute:
    """Select exactly one expert before Mage-Flow is placed on the GPU.

    Explicit route metadata wins.  With conflicting medium cues, the final
    explicit cue wins because image prompts conventionally put the desired
    rendering medium near the end.  Prompts without a medium cue use the
    configured deployment default; there is intentionally no neutral result.
    """
    if override is not None:
        normalized = override.strip().lower()
        if normalized not in EXPERT_DOMAINS:
            raise ValueError(f"unsupported explicit expert route {override!r}")
        return PromptRoute(normalized, "explicit_override")
    if default_domain not in EXPERT_DOMAINS:
        raise ValueError(f"unsupported default expert route {default_domain!r}")
    if not isinstance(prompt, str) or not prompt.strip():
        return PromptRoute(default_domain, "empty_prompt_default")

    animation = tuple(match.group(0) for match in _ANIMATION_CUE.finditer(prompt))
    photo = tuple(match.group(0) for match in _PHOTO_CUE.finditer(prompt))
    if animation and not photo:
        return PromptRoute("animation", "animation_medium_cue", animation, photo)
    if photo and not animation:
        return PromptRoute("photo", "photo_medium_cue", animation, photo)
    if animation and photo:
        last_animation = max(match.end() for match in _ANIMATION_CUE.finditer(prompt))
        last_photo = max(match.end() for match in _PHOTO_CUE.finditer(prompt))
        domain = "animation" if last_animation > last_photo else "photo"
        return PromptRoute(domain, "last_explicit_medium_cue", animation, photo)
    return PromptRoute(default_domain, "no_medium_cue_default", animation, photo)


@dataclass(frozen=True)
class TerminalExpertConfig:
    depth: int
    source_block_indices: tuple[int, ...]
    expert_parameter_count: int
    base_parameter_count: int
    active_parameter_count: int
    total_checkpoint_parameter_count: int
    resident_expert_count: int = 1
    domains: tuple[str, ...] = EXPERT_DOMAINS


def _run_block(
    transformer,
    block: nn.Module,
    *,
    block_index: int,
    img: torch.Tensor,
    txt: torch.Tensor,
    temb: torch.Tensor,
    ms_pe: torch.Tensor,
    txt_cu_seqlens: torch.Tensor | None,
    img_cu_seqlens: torch.Tensor | None,
    attention_kwargs: Mapping[str, Any],
    terminal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    should_checkpoint = bool(
        transformer.training
        and transformer.checkpoint
        and (
            terminal
            or transformer.checkpoint_block_indices is None
            or block_index in transformer.checkpoint_block_indices
        )
    )
    if should_checkpoint:
        checkpoint_kwargs: dict[str, Any] = {"use_reentrant": False}
        if transformer.checkpoint_context_fn is not None:
            checkpoint_kwargs["context_fn"] = transformer.checkpoint_context_fn
        return torch.utils.checkpoint.checkpoint(
            block,
            img,
            txt,
            temb,
            ms_pe,
            txt_cu_seqlens,
            img_cu_seqlens,
            **checkpoint_kwargs,
        )
    return block(
        hidden_states=img,
        encoder_hidden_states=txt,
        txt_cu_lens=txt_cu_seqlens,
        img_cu_lens=img_cu_seqlens,
        temb=temb,
        image_rotary_emb=ms_pe,
        joint_attention_kwargs=dict(attention_kwargs),
    )


def _factored_linear_sum(
    inputs: torch.Tensor,
    first: nn.Linear,
    second: nn.Linear,
    *,
    factor_count: int,
    active_factors: tuple[int, ...],
    activation: str,
    dropout: nn.Module | None,
) -> torch.Tensor:
    """Execute only selected MLP expansion slices and sum their output."""

    expansion = first.out_features
    if expansion % factor_count:
        raise ValueError("Mage MLP expansion width is not factor divisible")
    factor_width = expansion // factor_count
    output = torch.zeros(
        (*inputs.shape[:-1], second.out_features),
        device=inputs.device,
        dtype=inputs.dtype,
    )
    for factor_start, factor_end in _contiguous_factor_ranges(active_factors):
        start = factor_start * factor_width
        end = factor_end * factor_width
        hidden = F.linear(
            inputs,
            first.weight[start:end],
            first.bias[start:end] if first.bias is not None else None,
        )
        hidden = F.gelu(hidden, approximate=activation)
        if dropout is not None:
            hidden = dropout(hidden)
        output = output + F.linear(
            hidden,
            second.weight[:, start:end],
            None,
        )
    if second.bias is not None:
        output = output + second.bias
    return output


def _factored_swiglu_sum(
    inputs: torch.Tensor,
    module: LightningSwiGLU,
    *,
    factor_count: int,
    active_factors: tuple[int, ...],
) -> torch.Tensor:
    """Execute only selected SwiGLU expansion factors."""

    expansion = module.hidden_features
    if expansion % factor_count:
        raise ValueError("Mage SwiGLU width is not factor divisible")
    factor_width = expansion // factor_count
    output = torch.zeros(
        (*inputs.shape[:-1], module.out_features),
        device=inputs.device,
        dtype=inputs.dtype,
    )
    for factor_start, factor_end in _contiguous_factor_ranges(active_factors):
        start = factor_start * factor_width
        end = factor_end * factor_width
        gate = F.linear(
            inputs,
            module.w12.weight[start:end],
            module.w12.bias[start:end] if module.w12.bias is not None else None,
        )
        value_start = expansion + start
        value_end = expansion + end
        value = F.linear(
            inputs,
            module.w12.weight[value_start:value_end],
            (
                module.w12.bias[value_start:value_end]
                if module.w12.bias is not None
                else None
            ),
        )
        hidden = F.silu(gate) * value
        output = output + F.linear(
            hidden,
            module.w3.weight[:, start:end],
            None,
        )
    if module.w3.bias is not None:
        output = output + module.w3.bias
    return output


def _factored_joint_attention(
    block: nn.Module,
    image: torch.Tensor,
    text: torch.Tensor,
    *,
    image_rotary_emb: torch.Tensor,
    txt_cu_lens: torch.Tensor,
    img_cu_lens: torch.Tensor,
    factor_count: int,
    active_factors: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute disjoint attention-head groups without full-width QKV GEMMs."""

    from mage_flow.models.modules._attn_backend import flash_attn_varlen_func
    from mage_flow.models.modules.mage_layers import apply_rotary_emb_mageflow

    attn = block.attn
    heads = int(attn.heads)
    head_dim = int(attn.inner_dim // heads)
    if heads % factor_count:
        raise ValueError("Mage attention heads are not factor divisible")
    heads_per_factor = heads // factor_count
    img_lens = img_cu_lens[1:] - img_cu_lens[:-1]
    txt_lens = txt_cu_lens[1:] - txt_cu_lens[:-1]
    joint_lens = txt_lens + img_lens
    joint_cu_lens = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32, device=joint_lens.device),
            torch.cumsum(joint_lens, dim=0, dtype=torch.int32),
        )
    )
    sample_indices = torch.arange(len(txt_lens), device=joint_lens.device)
    txt_sample_ids = torch.repeat_interleave(sample_indices, txt_lens)
    img_sample_ids = torch.repeat_interleave(sample_indices, img_lens)
    txt_intra_pos = (
        torch.arange(text.shape[1], device=text.device)
        - txt_cu_lens[txt_sample_ids]
    )
    img_intra_pos = (
        torch.arange(image.shape[1], device=image.device)
        - img_cu_lens[img_sample_ids]
    )
    txt_dest = joint_cu_lens[txt_sample_ids] + txt_intra_pos
    img_dest = joint_cu_lens[img_sample_ids] + txt_lens[img_sample_ids] + img_intra_pos
    total_tokens = int(joint_cu_lens[-1])
    img_output = torch.zeros_like(image)
    txt_output = torch.zeros_like(text)

    projections = (
        (attn.to_q, image),
        (attn.to_k, image),
        (attn.to_v, image),
        (attn.add_q_proj, text),
        (attn.add_k_proj, text),
        (attn.add_v_proj, text),
    )
    for factor_start, factor_end in _contiguous_factor_ranges(active_factors):
        head_start = factor_start * heads_per_factor
        head_end = factor_end * heads_per_factor
        active_heads = head_end - head_start
        channel_start = head_start * head_dim
        channel_end = head_end * head_dim
        projected = []
        for projection, source in projections:
            value = F.linear(
                source,
                projection.weight[channel_start:channel_end],
                (
                    projection.bias[channel_start:channel_end]
                    if projection.bias is not None
                    else None
                ),
            )
            projected.append(
                value.unflatten(-1, (active_heads, head_dim)).flatten(0, 1)
            )
        img_q, img_k, img_v, txt_q, txt_k, txt_v = projected
        if attn.norm_q is not None:
            img_q = attn.norm_q(img_q)
        if attn.norm_k is not None:
            img_k = attn.norm_k(img_k)
        if attn.norm_added_q is not None:
            txt_q = attn.norm_added_q(txt_q)
        if attn.norm_added_k is not None:
            txt_k = attn.norm_added_k(txt_k)
        img_q = apply_rotary_emb_mageflow(img_q, image_rotary_emb)
        img_k = apply_rotary_emb_mageflow(img_k, image_rotary_emb)

        shape = (total_tokens, active_heads, head_dim)
        joint_q = torch.empty(shape, dtype=img_q.dtype, device=img_q.device)
        joint_k = torch.empty_like(joint_q)
        joint_v = torch.empty_like(joint_q)
        joint_q[txt_dest], joint_q[img_dest] = txt_q, img_q
        joint_k[txt_dest], joint_k[img_dest] = txt_k, img_k
        joint_v[txt_dest], joint_v[img_dest] = txt_v, img_v
        attended = flash_attn_varlen_func(
            joint_q,
            joint_k,
            joint_v,
            cu_seqlens_q=joint_cu_lens,
            cu_seqlens_k=joint_cu_lens,
            max_seqlen_q=int(joint_lens.max()),
            max_seqlen_k=int(joint_lens.max()),
            dropout_p=0.0,
            softmax_scale=None,
            causal=False,
        )
        img_heads = attended[img_dest].flatten(1, 2).unsqueeze(0)
        txt_heads = attended[txt_dest].flatten(1, 2).unsqueeze(0)
        img_output = img_output + F.linear(
            img_heads,
            attn.to_out[0].weight[:, channel_start:channel_end],
            None,
        )
        txt_output = txt_output + F.linear(
            txt_heads,
            attn.to_add_out.weight[:, channel_start:channel_end],
            None,
        )
    if attn.to_out[0].bias is not None:
        img_output = img_output + attn.to_out[0].bias
    if len(attn.to_out) > 1:
        img_output = attn.to_out[1](img_output)
    if attn.to_add_out.bias is not None:
        txt_output = txt_output + attn.to_add_out.bias
    return img_output, txt_output


def _factored_mage_block_forward(
    block: nn.Module,
    image: torch.Tensor,
    text: torch.Tensor,
    temb: torch.Tensor,
    image_rotary_emb: torch.Tensor,
    txt_cu_lens: torch.Tensor,
    img_cu_lens: torch.Tensor,
    *,
    factor_count: int,
    active_factors: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference factor execution; all active factors reconstruct the dense block."""

    img_mod1, img_mod2 = block.img_mod(temb).chunk(2, dim=-1)
    txt_mod1, txt_mod2 = block.txt_mod(temb).chunk(2, dim=-1)
    img_modulated, img_gate1 = block._modulate(
        block.img_norm1(image),
        img_mod1,
        cu_lens=img_cu_lens,
    )
    txt_modulated, txt_gate1 = block._modulate(
        block.txt_norm1(text),
        txt_mod1,
        cu_lens=txt_cu_lens,
    )
    img_attention, txt_attention = _factored_joint_attention(
        block,
        img_modulated,
        txt_modulated,
        image_rotary_emb=image_rotary_emb,
        txt_cu_lens=txt_cu_lens,
        img_cu_lens=img_cu_lens,
        factor_count=factor_count,
        active_factors=active_factors,
    )
    image = image + img_gate1 * img_attention
    text = text + txt_gate1 * txt_attention

    img_modulated2, img_gate2 = block._modulate(
        block.img_norm2(image),
        img_mod2,
        cu_lens=img_cu_lens,
    )
    txt_modulated2, txt_gate2 = block._modulate(
        block.txt_norm2(text),
        txt_mod2,
        cu_lens=txt_cu_lens,
    )
    if isinstance(block.img_mlp, LightningSwiGLU):
        if not isinstance(block.txt_mlp, LightningSwiGLU):
            raise TypeError("image/text MLP architectures must match")
        image_mlp = _factored_swiglu_sum(
            img_modulated2,
            block.img_mlp,
            factor_count=factor_count,
            active_factors=active_factors,
        )
        text_mlp = _factored_swiglu_sum(
            txt_modulated2,
            block.txt_mlp,
            factor_count=factor_count,
            active_factors=active_factors,
        )
    else:
        img_activation = block.img_mlp.net[0]
        txt_activation = block.txt_mlp.net[0]
        image_mlp = _factored_linear_sum(
            img_modulated2,
            img_activation.proj,
            block.img_mlp.net[2],
            factor_count=factor_count,
            active_factors=active_factors,
            activation=img_activation.approximate,
            dropout=block.img_mlp.net[1],
        )
        text_mlp = _factored_linear_sum(
            txt_modulated2,
            txt_activation.proj,
            block.txt_mlp.net[2],
            factor_count=factor_count,
            active_factors=active_factors,
            activation=txt_activation.approximate,
            dropout=block.txt_mlp.net[1],
        )
    image = image + img_gate2 * image_mlp
    text = text + txt_gate2 * text_mlp
    if image.dtype == torch.float16:
        image = image.clip(-65504, 65504)
    if text.dtype == torch.float16:
        text = text.clip(-65504, 65504)
    return text, image


def _run_factored_block(
    transformer,
    block: nn.Module,
    *,
    block_index: int,
    img: torch.Tensor,
    txt: torch.Tensor,
    temb: torch.Tensor,
    ms_pe: torch.Tensor,
    txt_cu_seqlens: torch.Tensor,
    img_cu_seqlens: torch.Tensor,
    terminal: bool,
    factor_count: int,
    active_factors: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    should_checkpoint = bool(
        transformer.training
        and transformer.checkpoint
        and (
            terminal
            or transformer.checkpoint_block_indices is None
            or block_index in transformer.checkpoint_block_indices
        )
    )

    def forward(image, text, conditioning, rope, txt_lens, img_lens):
        return _factored_mage_block_forward(
            block,
            image,
            text,
            conditioning,
            rope,
            txt_lens,
            img_lens,
            factor_count=factor_count,
            active_factors=active_factors,
        )

    if should_checkpoint:
        checkpoint_kwargs: dict[str, Any] = {"use_reentrant": False}
        if transformer.checkpoint_context_fn is not None:
            checkpoint_kwargs["context_fn"] = transformer.checkpoint_context_fn
        return torch.utils.checkpoint.checkpoint(
            forward,
            img,
            txt,
            temb,
            ms_pe,
            txt_cu_seqlens,
            img_cu_seqlens,
            **checkpoint_kwargs,
        )
    return forward(img, txt, temb, ms_pe, txt_cu_seqlens, img_cu_seqlens)


def _terminal_expert_forward(
    transformer,
    img: torch.Tensor,
    txt: torch.Tensor,
    timesteps: torch.Tensor,
    img_shapes=None,
    img_cu_seqlens: torch.Tensor | None = None,
    txt_cu_seqlens: torch.Tensor | None = None,
    attention_kwargs: dict[str, Any] | None = None,
    return_hidden_layer: int | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Pinned Mage-Flow forward with one terminal expert before the output head."""
    if img.ndim != 3 or txt.ndim != 3:
        raise ValueError("Input img and txt tensors must have 3 dimensions.")
    if getattr(transformer, "_terminal_expert_domain", None) not in EXPERT_DOMAINS:
        raise RuntimeError("no photo or animation terminal expert is resident")
    path_depth = len(transformer.transformer_blocks) + len(
        transformer.terminal_expert_blocks
    )
    if return_hidden_layer is not None and not 0 <= return_hidden_layer < path_depth:
        raise ValueError(
            f"return_hidden_layer must be in [0, {path_depth - 1}], "
            f"got {return_hidden_layer}"
        )

    ms_pe = transformer.pos_embed(img_shapes, device=img.device)
    img = transformer.img_in(img)
    txt = transformer.txt_norm(txt)
    timesteps = timesteps.to(img.dtype)
    temb = transformer.time_text_embed(timesteps, img)
    txt = transformer.txt_in(txt)
    temb = temb + torch.zeros(
        txt.shape[0],
        transformer.inner_dim,
        dtype=txt.dtype,
        device=txt.device,
    )
    attention_kwargs = attention_kwargs or {}
    captured: torch.Tensor | None = None
    loop_controller = getattr(transformer, "tread_loop_controller", None)
    original_img_cu_lens = img_cu_seqlens
    if original_img_cu_lens is None:
        original_img_cu_lens = torch.tensor(
            [0, img.shape[1]], device=img.device, dtype=torch.int32
        )
    active_txt_cu_lens = txt_cu_seqlens
    if active_txt_cu_lens is None:
        active_txt_cu_lens = torch.tensor(
            [0, txt.shape[1]], device=txt.device, dtype=torch.int32
        )
    total_image_tokens = int(img.shape[1])
    route_state = None
    active_ms_pe = ms_pe
    active_img_cu_lens = original_img_cu_lens
    if loop_controller is not None:
        loop_controller.begin_forward(img)

    for index, block in enumerate(transformer.transformer_blocks):
        if loop_controller is not None and index == loop_controller.route_start:
            tread = loop_controller.config.tread
            timestep_active = True
            if tread.active_timestep_min is not None:
                timestep_active = timestep_active and bool(
                    torch.all(timesteps >= tread.active_timestep_min)
                )
            if tread.active_timestep_max is not None:
                timestep_active = timestep_active and bool(
                    torch.all(timesteps <= tread.active_timestep_max)
                )
            route_enabled = bool(
                tread.enabled
                and timestep_active
                and (transformer.training or tread.inference_enabled)
            )
            if route_enabled and tread.route_fraction > 0:
                img, active_ms_pe, route_state = extract_tread_route(
                    img,
                    ms_pe,
                    original_img_cu_lens,
                    bypass_fraction=tread.route_fraction,
                    min_active_image_tokens=tread.min_active_image_tokens,
                )
                active_img_cu_lens = route_state.active_cu_lens

        def run_backbone_block(
            current_block,
            current_img,
            current_txt,
            current_temb,
            _index=index,
            _ms_pe=active_ms_pe,
            _img_cu_lens=active_img_cu_lens,
        ):
            return _run_block(
                transformer,
                current_block,
                block_index=_index,
                img=current_img,
                txt=current_txt,
                temb=current_temb,
                ms_pe=_ms_pe,
                txt_cu_seqlens=active_txt_cu_lens,
                img_cu_seqlens=_img_cu_lens,
                attention_kwargs=attention_kwargs,
                terminal=False,
            )

        def run_backbone_factors(
            current_block,
            current_img,
            current_txt,
            current_temb,
            factor_count,
            active_factors,
            _index=index,
            _ms_pe=active_ms_pe,
            _img_cu_lens=active_img_cu_lens,
        ):
            if not hasattr(current_block, "img_mod"):
                return run_backbone_block(
                    current_block,
                    current_img,
                    current_txt,
                    current_temb,
                )
            return _run_factored_block(
                transformer,
                current_block,
                block_index=_index,
                img=current_img,
                txt=current_txt,
                temb=current_temb,
                ms_pe=_ms_pe,
                txt_cu_seqlens=active_txt_cu_lens,
                img_cu_seqlens=_img_cu_lens,
                terminal=False,
                factor_count=factor_count,
                active_factors=active_factors,
            )

        run_backbone_block.run_factors = run_backbone_factors

        if loop_controller is None:
            txt, img = _run_block(
                transformer,
                block,
                block_index=index,
                img=img,
                txt=txt,
                temb=temb,
                ms_pe=active_ms_pe,
                txt_cu_seqlens=txt_cu_seqlens,
                img_cu_seqlens=active_img_cu_lens,
                attention_kwargs=attention_kwargs,
                terminal=False,
            )
        else:
            txt, img = loop_controller.run_backbone(
                index,
                block,
                img,
                txt,
                temb,
                run_backbone_block,
            )

        if (
            loop_controller is not None
            and index + 1 == loop_controller.route_end
            and route_state is not None
        ):
            img = restore_tread_route(img, route_state)
            active_ms_pe = ms_pe
            active_img_cu_lens = original_img_cu_lens
        if return_hidden_layer == index:
            captured = (
                restore_tread_route(img, route_state)
                if route_state is not None
                and index + 1 < loop_controller.route_end
                else img
            )

    backbone_depth = len(transformer.transformer_blocks)
    for index, block in enumerate(transformer.terminal_expert_blocks):
        def run_expert_block(
            current_block,
            current_img,
            current_txt,
            current_temb,
            _index=index,
        ):
            return _run_block(
                transformer,
                current_block,
                block_index=_index,
                img=current_img,
                txt=current_txt,
                temb=current_temb,
                ms_pe=ms_pe,
                txt_cu_seqlens=active_txt_cu_lens,
                img_cu_seqlens=original_img_cu_lens,
                attention_kwargs=attention_kwargs,
                terminal=True,
            )

        def run_expert_factors(
            current_block,
            current_img,
            current_txt,
            current_temb,
            factor_count,
            active_factors,
            _index=index,
        ):
            if not hasattr(current_block, "img_mod"):
                return run_expert_block(
                    current_block,
                    current_img,
                    current_txt,
                    current_temb,
                )
            return _run_factored_block(
                transformer,
                current_block,
                block_index=_index,
                img=current_img,
                txt=current_txt,
                temb=current_temb,
                ms_pe=ms_pe,
                txt_cu_seqlens=active_txt_cu_lens,
                img_cu_seqlens=original_img_cu_lens,
                terminal=True,
                factor_count=factor_count,
                active_factors=active_factors,
            )

        run_expert_block.run_factors = run_expert_factors

        if loop_controller is None:
            txt, img = run_expert_block(block, img, txt, temb)
        else:
            txt, img = loop_controller.run_expert(
                index,
                block,
                img,
                txt,
                temb,
                run_expert_block,
            )
        if return_hidden_layer == backbone_depth + index:
            captured = img

    if loop_controller is not None:
        last_adapter = loop_controller.expert_adapters[-1]
        if loop_controller.config.looping.use_auxiliary_loss:
            loop_controller.last_aux_predictions = tuple(
                transformer.proj_out(
                    transformer.norm_out(
                        hidden,
                        temb,
                        cu_seqlens=original_img_cu_lens,
                    )
                )
                for hidden in last_adapter.last_image_iterates[:-1]
            )
        loop_controller.finish_forward(
            total_image_tokens=total_image_tokens,
            active_image_tokens=(
                int(route_state.active_indices.numel())
                if route_state is not None
                else total_image_tokens
            ),
            bypassed_image_tokens=(
                int(route_state.bypass_indices.numel())
                if route_state is not None
                else 0
            ),
            text_tokens=int(txt.shape[1]),
        )

    img = transformer.norm_out(img, temb, cu_seqlens=img_cu_seqlens)
    prediction = transformer.proj_out(img)
    if return_hidden_layer is None:
        return prediction
    if captured is None:
        raise RuntimeError("requested terminal-path representation was not captured")
    return prediction, captured


class TerminalExpertController:
    """Own the single resident expert attached to one Mage-Flow transformer."""

    def __init__(
        self,
        transformer,
        config: TerminalExpertConfig,
        *,
        resident_domain: str,
        base_model: str,
        base_revision: str,
    ):
        if resident_domain not in EXPERT_DOMAINS:
            raise ValueError(f"unsupported resident expert {resident_domain!r}")
        self.transformer = transformer
        self.config = config
        self.base_model = base_model
        self.base_revision = base_revision
        self.transformer._terminal_expert_domain = resident_domain

    @property
    def resident_domain(self) -> str:
        return str(self.transformer._terminal_expert_domain)

    @property
    def blocks(self) -> nn.ModuleList:
        return self.transformer.terminal_expert_blocks

    def parameters(self) -> Iterator[nn.Parameter]:
        yield from self.blocks.parameters()

    def parameter_count(self, domain: str | None = None) -> int:
        if domain is not None and domain not in EXPERT_DOMAINS:
            raise ValueError(f"unsupported expert domain {domain!r}")
        return sum(parameter.numel() for parameter in self.parameters())

    @contextmanager
    def route(self, domain: str) -> Iterator[None]:
        if domain not in EXPERT_DOMAINS:
            raise ValueError(f"unsupported expert domain {domain!r}")
        if domain != self.resident_domain:
            raise RuntimeError(
                f"{domain} expert requested while {self.resident_domain} is resident; "
                "load the requested expert before executing Mage-Flow"
            )
        yield

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> TerminalExpertController:
        self.blocks.to(device=device, dtype=dtype)
        return self


class TerminalExpertRuntime:
    """Prompt router plus a two-file expert store with one GPU-resident tail."""

    def __init__(
        self,
        controller: TerminalExpertController,
        checkpoints: Mapping[str, Path | str],
        *,
        default_domain: str = "photo",
    ):
        if set(checkpoints) != set(EXPERT_DOMAINS):
            raise ValueError(
                f"terminal expert store requires exactly {list(EXPERT_DOMAINS)}"
            )
        if default_domain not in EXPERT_DOMAINS:
            raise ValueError(f"unsupported default domain {default_domain!r}")
        self.controller = controller
        self.checkpoints = {
            domain: Path(path).expanduser().resolve()
            for domain, path in checkpoints.items()
        }
        missing = [
            str(path) for path in self.checkpoints.values() if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(f"terminal expert checkpoints missing: {missing}")
        self.default_domain = default_domain

    def select(
        self,
        prompt: str,
        *,
        override: str | None = None,
    ) -> PromptRoute:
        decision = route_prompt(
            prompt,
            override=override,
            default_domain=self.default_domain,
        )
        if decision.domain != self.controller.resident_domain:
            load_terminal_expert(
                self.controller,
                decision.domain,
                self.checkpoints[decision.domain],
            )
        return decision

    @contextmanager
    def route(
        self,
        prompt: str,
        *,
        override: str | None = None,
    ) -> Iterator[PromptRoute]:
        decision = self.select(prompt, override=override)
        with self.controller.route(decision.domain):
            yield decision


def reset_terminal_expert_from_backbone(
    controller: TerminalExpertController,
    domain: str,
) -> None:
    """Reset the resident module from the released late-block initialization."""
    if domain not in EXPERT_DOMAINS:
        raise ValueError(f"unsupported expert domain {domain!r}")
    source_blocks = [
        controller.transformer.transformer_blocks[index]
        for index in controller.config.source_block_indices
    ]
    with torch.no_grad():
        for target, source in zip(controller.blocks, source_blocks, strict=True):
            target.load_state_dict(source.state_dict(), strict=True)
    controller.transformer._terminal_expert_domain = domain


def install_terminal_expert(
    transformer,
    resident_domain: str,
    *,
    depth: int = TERMINAL_EXPERT_DEPTH,
    base_model: str = MAGE_FLOW_BASE_ID,
    base_revision: str = MAGE_FLOW_BASE_REVISION,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> TerminalExpertController:
    """Append one three-block expert after the complete released backbone.

    New blocks begin as copies of the released final blocks.  This is a
    meaningful pretrained initialization, not an identity or zero-output tail.
    """
    if resident_domain not in EXPERT_DOMAINS:
        raise ValueError(f"unsupported resident expert {resident_domain!r}")
    if hasattr(transformer, "terminal_expert_blocks"):
        raise ValueError("terminal expert is already installed")
    blocks = transformer.transformer_blocks
    if depth < 1 or depth > len(blocks):
        raise ValueError("terminal expert depth is outside the backbone")

    base_parameter_count = sum(parameter.numel() for parameter in transformer.parameters())
    source_indices = tuple(range(len(blocks) - depth, len(blocks)))
    expert_blocks = nn.ModuleList(copy.deepcopy(blocks[index]) for index in source_indices)
    if device is not None or dtype is not None:
        expert_blocks.to(device=device, dtype=dtype)
    transformer.add_module("terminal_expert_blocks", expert_blocks)
    transformer._terminal_expert_domain = resident_domain
    transformer._terminal_original_forward = transformer.forward
    transformer.forward = MethodType(_terminal_expert_forward, transformer)

    expert_parameter_count = sum(
        parameter.numel() for parameter in expert_blocks.parameters()
    )
    config = TerminalExpertConfig(
        depth=depth,
        source_block_indices=source_indices,
        expert_parameter_count=expert_parameter_count,
        base_parameter_count=base_parameter_count,
        active_parameter_count=base_parameter_count + expert_parameter_count,
        total_checkpoint_parameter_count=base_parameter_count
        + len(EXPERT_DOMAINS) * expert_parameter_count,
    )
    return TerminalExpertController(
        transformer,
        config,
        resident_domain=resident_domain,
        base_model=base_model,
        base_revision=base_revision,
    )


def convert_terminal_path_to_lightning_blocks(
    transformer,
    controller: TerminalExpertController,
    *,
    use_swiglu: bool,
    use_rmsnorm: bool,
) -> dict[str, Any]:
    """Convert every executed backbone/expert block to LightningDiT primitives."""

    if not use_swiglu and not use_rmsnorm:
        return {
            "schema": LIGHTNING_BLOCK_SCHEMA,
            "enabled": False,
            "use_swiglu": False,
            "use_rmsnorm": False,
            "converted_blocks": 0,
            "parameter_delta": 0,
        }
    if use_swiglu != use_rmsnorm:
        raise ValueError(
            "the qualified Lightning block migration enables SwiGLU and "
            "RMSNorm together"
        )

    def convert_norm(norm: nn.Module) -> nn.Module:
        if isinstance(norm, LightningRMSNorm):
            return norm
        if not isinstance(norm, nn.LayerNorm):
            raise TypeError(f"unsupported Mage normalization {type(norm)!r}")
        shape = norm.normalized_shape
        if len(shape) != 1:
            raise ValueError("Mage block normalization must have one dimension")
        reference = next(norm.parameters(), None)
        if reference is None:
            device = next(transformer.parameters()).device
            dtype = next(transformer.parameters()).dtype
        else:
            device = reference.device
            dtype = reference.dtype
        return LightningRMSNorm(
            int(shape[0]),
            eps=float(norm.eps),
            device=device,
            dtype=dtype,
        )

    def convert_block(block: nn.Module) -> int:
        before = sum(parameter.numel() for parameter in block.parameters())
        block.img_norm1 = convert_norm(block.img_norm1)
        block.img_norm2 = convert_norm(block.img_norm2)
        block.txt_norm1 = convert_norm(block.txt_norm1)
        block.txt_norm2 = convert_norm(block.txt_norm2)
        if not isinstance(block.img_mlp, LightningSwiGLU):
            block.img_mlp = LightningSwiGLU.from_feed_forward(block.img_mlp)
        if not isinstance(block.txt_mlp, LightningSwiGLU):
            block.txt_mlp = LightningSwiGLU.from_feed_forward(block.txt_mlp)
        after = sum(parameter.numel() for parameter in block.parameters())
        return after - before

    backbone_delta = sum(
        convert_block(block) for block in transformer.transformer_blocks
    )
    expert_delta = sum(convert_block(block) for block in controller.blocks)
    final_norm_delta = 0
    if hasattr(transformer.norm_out, "norm"):
        old_final = transformer.norm_out.norm
        before = sum(parameter.numel() for parameter in old_final.parameters())
        transformer.norm_out.norm = convert_norm(old_final)
        after = sum(
            parameter.numel()
            for parameter in transformer.norm_out.norm.parameters()
        )
        final_norm_delta = after - before

    base_parameter_count = (
        controller.config.base_parameter_count
        + backbone_delta
        + final_norm_delta
    )
    expert_parameter_count = (
        controller.config.expert_parameter_count + expert_delta
    )
    controller.config = replace(
        controller.config,
        expert_parameter_count=expert_parameter_count,
        base_parameter_count=base_parameter_count,
        active_parameter_count=base_parameter_count + expert_parameter_count,
        total_checkpoint_parameter_count=(
            base_parameter_count
            + len(EXPERT_DOMAINS) * expert_parameter_count
        ),
    )
    transformer._lightning_swiglu = True
    transformer._lightning_rmsnorm = True
    return {
        "schema": LIGHTNING_BLOCK_SCHEMA,
        "enabled": True,
        "use_swiglu": True,
        "use_rmsnorm": True,
        "backbone_blocks": len(transformer.transformer_blocks),
        "expert_blocks": len(controller.blocks),
        "converted_blocks": (
            len(transformer.transformer_blocks) + len(controller.blocks)
        ),
        "backbone_parameter_delta": backbone_delta + final_norm_delta,
        "expert_parameter_delta": expert_delta,
        "parameter_delta": backbone_delta + final_norm_delta + expert_delta,
        "swiglu_hidden_features": int(
            transformer.transformer_blocks[0].img_mlp.hidden_features
        ),
        "rmsnorm_final_output": isinstance(
            transformer.norm_out.norm, LightningRMSNorm
        ),
    }


def _terminal_tensors(controller: TerminalExpertController) -> dict[str, torch.Tensor]:
    tensors = {
        f"blocks.{name}": value.detach().cpu().contiguous()
        for name, value in controller.blocks.state_dict().items()
    }
    loop_controller = getattr(
        controller.transformer, "tread_loop_controller", None
    )
    if loop_controller is not None:
        tensors.update(
            {
                f"loop_adapters.{name}": value.detach().cpu().contiguous()
                for name, value in loop_controller.expert_adapters.state_dict().items()
            }
        )
    return tensors


def save_terminal_expert(
    controller: TerminalExpertController,
    path: Path,
    *,
    dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    """Save only the resident terminal expert, never the shared backbone."""
    from safetensors.torch import save_file

    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = _terminal_tensors(controller)
    if dtype is not None:
        tensors = {
            name: value.to(dtype=dtype).contiguous()
            if value.is_floating_point()
            else value
            for name, value in tensors.items()
        }
    metadata = {
        "schema": TERMINAL_EXPERT_SCHEMA,
        "domain": controller.resident_domain,
        "base_model": controller.base_model,
        "base_revision": controller.base_revision,
        "config_json": json.dumps(asdict(controller.config), sort_keys=True),
    }
    temporary = path.with_name(path.name + ".tmp")
    save_file(tensors, str(temporary), metadata=metadata)
    os.replace(temporary, path)
    manifest = {
        **metadata,
        "config": asdict(controller.config),
        "path": str(path),
        "sha256": _file_sha256(path),
        "tensor_count": len(tensors),
    }
    manifest.pop("config_json")
    _atomic_json(path.with_suffix(path.suffix + ".json"), manifest)
    return manifest


def load_terminal_expert(
    controller: TerminalExpertController,
    domain: str,
    path: Path,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Replace the one resident tail with a selected domain checkpoint."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    if domain not in EXPERT_DOMAINS:
        raise ValueError(f"unsupported expert domain {domain!r}")
    path = path.expanduser().resolve()
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    expected_metadata = {
        "schema": TERMINAL_EXPERT_SCHEMA,
        "domain": domain,
        "base_model": controller.base_model,
        "base_revision": controller.base_revision,
    }
    mismatches = {
        key: (metadata.get(key), expected)
        for key, expected in expected_metadata.items()
        if metadata.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"terminal expert metadata mismatch: {mismatches}")

    tensors = load_file(str(path), device="cpu")
    state = {
        name.removeprefix("blocks."): value
        for name, value in tensors.items()
        if name.startswith("blocks.")
    }
    legacy_lightning_conversion = any(
        name.endswith("net.0.proj.weight") for name in state
    )
    if legacy_lightning_conversion:
        for name, module in controller.blocks.named_modules():
            if isinstance(module, LightningRMSNorm):
                state.setdefault(
                    f"{name}.weight",
                    module.weight.detach().cpu().clone(),
                )
    result = controller.blocks.load_state_dict(state, strict=strict)
    loop_state = {
        name.removeprefix("loop_adapters."): value
        for name, value in tensors.items()
        if name.startswith("loop_adapters.")
    }
    loop_result = None
    if loop_state:
        loop_controller = getattr(
            controller.transformer, "tread_loop_controller", None
        )
        if loop_controller is None:
            raise ValueError(
                "expert checkpoint contains learned loop adapters; install "
                "TREAD/looping before loading it"
            )
        loop_result = loop_controller.expert_adapters.load_state_dict(
            loop_state,
            strict=strict,
        )
        loop_controller.float_controls()
    else:
        loop_controller = getattr(
            controller.transformer, "tread_loop_controller", None
        )
        if loop_controller is not None:
            loop_controller.reset_expert_controls()
    if loop_controller is not None:
        loop_controller.refresh_inference_skip_refinements()
    controller.transformer._terminal_expert_domain = domain
    return {
        "domain": domain,
        "path": str(path),
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
        "loop_missing_keys": (
            list(loop_result.missing_keys) if loop_result is not None else []
        ),
        "loop_unexpected_keys": (
            list(loop_result.unexpected_keys) if loop_result is not None else []
        ),
        "compatible": (
            not result.missing_keys
            and not result.unexpected_keys
            and (
                loop_result is None
                or (
                    not loop_result.missing_keys
                    and not loop_result.unexpected_keys
                )
            )
        ),
    }


def _feed_forward_linears(module: nn.Module) -> tuple[nn.Linear, nn.Linear]:
    """Return the two projections from Mage's diffusers FeedForward module."""
    linears = [child for child in module.modules() if isinstance(child, nn.Linear)]
    if len(linears) != 2:
        raise ValueError(
            f"expected exactly two image-MLP projections, found {len(linears)}"
        )
    return linears[0], linears[1]


def initialize_from_residual_expert(
    controller: TerminalExpertController,
    residual_path: Path,
    *,
    source_block_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Seed terminal image MLPs from the trained residual beta checkpoint.

    The source MLP is slightly wider (12,544 versus 12,288).  For each of the
    final three residual blocks, select neurons by the product of incoming and
    outgoing weight norms, retain the strongest target-width neurons, and fold
    the learned scalar into the output projection.  Attention, text MLPs,
    modulation, and norms remain copied from the corresponding released block.
    """
    from safetensors import safe_open

    residual_path = residual_path.expanduser().resolve()
    if source_block_indices is None:
        source_block_indices = controller.config.source_block_indices
    source_block_indices = tuple(int(index) for index in source_block_indices)
    if len(source_block_indices) != len(controller.blocks):
        raise ValueError("one residual source block is required per terminal block")

    reports: list[dict[str, Any]] = []
    with safe_open(str(residual_path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if metadata.get("domain") != controller.resident_domain:
            raise ValueError(
                f"residual domain {metadata.get('domain')!r} does not match "
                f"resident {controller.resident_domain!r}"
            )
        available = set(handle.keys())
        for terminal_index, source_index in enumerate(source_block_indices):
            prefix = f"blocks.{source_index}.expert."
            names = {
                "fc1_weight": prefix + "fc1.weight",
                "fc1_bias": prefix + "fc1.bias",
                "fc2_weight": prefix + "fc2.weight",
                "fc2_bias": prefix + "fc2.bias",
                "scale": f"blocks.{source_index}.scale",
            }
            missing = sorted(set(names.values()) - available)
            if missing:
                raise ValueError(f"residual checkpoint is missing {missing}")

            fc1_weight = handle.get_tensor(names["fc1_weight"])
            fc1_bias = handle.get_tensor(names["fc1_bias"])
            fc2_weight = handle.get_tensor(names["fc2_weight"])
            fc2_bias = handle.get_tensor(names["fc2_bias"])
            scale = handle.get_tensor(names["scale"]).float().reshape(())
            target_in, target_out = _feed_forward_linears(
                controller.blocks[terminal_index].img_mlp
            )
            target_width = target_in.out_features
            source_width = fc1_weight.shape[0]
            if (
                fc1_weight.shape[1] != target_in.in_features
                or fc2_weight.shape[0] != target_out.out_features
                or fc2_weight.shape[1] != source_width
                or target_out.in_features != target_width
            ):
                raise ValueError(
                    f"incompatible residual/terminal MLP at source block {source_index}"
                )
            if source_width < target_width:
                raise ValueError("residual MLP is narrower than terminal image MLP")

            importance = (
                torch.linalg.vector_norm(fc1_weight.float(), dim=1)
                * torch.linalg.vector_norm(fc2_weight.float(), dim=0)
            )
            selected = torch.topk(
                importance, k=target_width, largest=True, sorted=False
            ).indices.sort().values
            with torch.no_grad():
                target_in.weight.copy_(
                    fc1_weight[selected].to(
                        device=target_in.weight.device,
                        dtype=target_in.weight.dtype,
                    )
                )
                target_in.bias.copy_(
                    fc1_bias[selected].to(
                        device=target_in.bias.device,
                        dtype=target_in.bias.dtype,
                    )
                )
                target_out.weight.copy_(
                    (fc2_weight[:, selected] * scale).to(
                        device=target_out.weight.device,
                        dtype=target_out.weight.dtype,
                    )
                )
                target_out.bias.copy_(
                    (fc2_bias * scale).to(
                        device=target_out.bias.device,
                        dtype=target_out.bias.dtype,
                    )
                )
            reports.append(
                {
                    "terminal_block": terminal_index,
                    "source_residual_block": source_index,
                    "source_width": source_width,
                    "retained_width": target_width,
                    "retained_fraction": target_width / source_width,
                    "scale": float(scale),
                }
            )
    return {
        "schema": "rwkv-lab.mage-flow-residual-to-terminal-migration.v1",
        "domain": controller.resident_domain,
        "source": str(residual_path),
        "policy": "top-importance residual image MLP; released weights elsewhere",
        "blocks": reports,
    }


def terminal_architecture_report(
    controller: TerminalExpertController,
) -> dict[str, Any]:
    """Return an auditable topology and residency receipt."""
    transformer = controller.transformer
    residual_wrappers = [
        index
        for index, block in enumerate(transformer.transformer_blocks)
        if hasattr(block.img_mlp, "experts") or hasattr(block.img_mlp, "shared_ffn")
    ]
    loop_controller = getattr(transformer, "tread_loop_controller", None)
    report = {
        "schema": "rwkv-lab.mage-flow-terminal-architecture.v1",
        "base_block_count": len(transformer.transformer_blocks),
        "base_blocks_execute_before_expert": True,
        "terminal_expert_depth": len(controller.blocks),
        "resident_domain": controller.resident_domain,
        "resident_expert_count": 1,
        "neutral_route": False,
        "residual_expert_blocks": residual_wrappers,
        "expert_parameter_count": controller.config.expert_parameter_count,
        "base_parameter_count": controller.config.base_parameter_count,
        "active_parameter_count": controller.config.active_parameter_count,
        "total_checkpoint_parameter_count": (
            controller.config.total_checkpoint_parameter_count
        ),
        "passed": (
            len(transformer.transformer_blocks) == 12
            and len(controller.blocks) == TERMINAL_EXPERT_DEPTH
            and not residual_wrappers
            and controller.resident_domain in EXPERT_DOMAINS
        ),
    }
    if loop_controller is not None:
        report["tread_factored_looping"] = loop_controller.report()
        report["independent_backbone_blocks_executed"] = list(
            range(len(transformer.transformer_blocks))
        )
        report["independent_backbone_blocks_replaced_by_recurrent_core"] = []
        report["executed_backbone_contract"] = (
            "complete original backbone with token-only TREAD bypass -> "
            "exactly one learned-loop terminal expert"
        )
    return report


def configure_terminal_training_scope(
    transformer,
    controller: TerminalExpertController,
    *,
    train_backbone_final_fraction: float = 1 / 3,
) -> dict[str, Any]:
    """Train the resident expert and only the approved original backbone tail."""
    if not 0 <= train_backbone_final_fraction <= 1:
        raise ValueError("train_backbone_final_fraction must be in [0, 1]")
    transformer.requires_grad_(False)
    for parameter in controller.parameters():
        parameter.requires_grad_(True)
    loop_controller = getattr(transformer, "tread_loop_controller", None)
    if loop_controller is not None:
        loop_controller.requires_grad_(True)

    blocks = transformer.transformer_blocks
    base_indices: tuple[int, ...] = ()
    if train_backbone_final_fraction > 0:
        selected_count = max(1, round(len(blocks) * train_backbone_final_fraction))
        base_indices = tuple(range(len(blocks) - selected_count, len(blocks)))
        for index in base_indices:
            blocks[index].requires_grad_(True)

    expert_ids = {id(parameter) for parameter in controller.parameters()}
    loop_controller = getattr(transformer, "tread_loop_controller", None)
    if loop_controller is not None:
        expert_ids.update(
            id(parameter) for parameter in loop_controller.expert_parameters()
        )
    expert_names = []
    backbone_names = []
    for name, parameter in transformer.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in expert_ids:
            expert_names.append(name)
        else:
            backbone_names.append(name)
    return {
        "resident_domain": controller.resident_domain,
        "expert_block_count": len(controller.blocks),
        "backbone_block_indices": list(base_indices),
        "expert_trainable_parameter_names": expert_names,
        "backbone_trainable_parameter_names": backbone_names,
        "expert_trainable_parameter_count": sum(
            parameter.numel()
            for parameter in transformer.parameters()
            if parameter.requires_grad and id(parameter) in expert_ids
        ),
        "backbone_trainable_parameter_count": sum(
            parameter.numel()
            for name, parameter in transformer.named_parameters()
            if parameter.requires_grad and id(parameter) not in expert_ids
        ),
    }


def terminal_optimizer_parameter_groups(
    transformer,
    controller: TerminalExpertController,
    *,
    expert_learning_rate: float,
    backbone_learning_rate_multiplier: float = 0.5,
    repa_projection: nn.Module | None = None,
    repa_learning_rate_multiplier: float = 1.0,
) -> list[dict[str, Any]]:
    """Create the required 1.0x expert and 0.5x backbone optimizer groups."""
    return list(
        terminal_optimizer_parameter_routing(
            transformer,
            controller,
            expert_learning_rate=expert_learning_rate,
            backbone_learning_rate_multiplier=backbone_learning_rate_multiplier,
            repa_projection=repa_projection,
            repa_learning_rate_multiplier=repa_learning_rate_multiplier,
        ).groups
    )


def terminal_optimizer_parameter_routing(
    transformer,
    controller: TerminalExpertController,
    *,
    expert_learning_rate: float,
    backbone_learning_rate_multiplier: float = 0.5,
    repa_projection: nn.Module | None = None,
    repa_learning_rate_multiplier: float = 1.0,
    worker_components: WorkerTrainingComponents | None = None,
) -> ParameterRoutingResult:
    """Prove terminal-expert/backbone/REPA ownership before grouping."""

    expert_ids = {id(parameter) for parameter in controller.parameters()}
    loop_controller = getattr(transformer, "tread_loop_controller", None)
    if loop_controller is not None:
        # The controller has its own complete, metadata-bearing checkpoint.
        # Excluding it here avoids duplicate partial state and makes one file
        # authoritative for both backbone and expert loop controls.
        expert_ids.update(id(parameter) for parameter in loop_controller.parameters())
    repa_ids = (
        frozenset(id(parameter) for parameter in repa_projection.parameters())
        if repa_projection is not None
        else frozenset()
    )
    named_parameters = [
        (f"transformer.{name}", parameter)
        for name, parameter in transformer.named_parameters(remove_duplicate=False)
    ]
    if repa_projection is not None:
        named_parameters.extend(
            (f"repa.{name}", parameter)
            for name, parameter in repa_projection.named_parameters(
                remove_duplicate=False
            )
        )
    if worker_components is not None:
        return worker_components.parameter_routing(
            named_parameters,
            {"expert": frozenset(expert_ids), "repa": repa_ids},
            base_learning_rate=expert_learning_rate,
        )
    return build_registered_parameter_routing(
        ParameterRouterImplementation.MAGEFLOW_TERMINAL_EXPERT_V1,
        named_parameters,
        {"expert": frozenset(expert_ids), "repa": repa_ids},
        base_learning_rate=expert_learning_rate,
        configuration=TerminalExpertRoutingConfiguration(
            shared_backbone_multiplier=backbone_learning_rate_multiplier,
            repa_projection_multiplier=repa_learning_rate_multiplier,
        ),
    )


def save_terminal_shared_backbone(
    transformer,
    controller: TerminalExpertController,
    path: Path,
    *,
    dtype: torch.dtype | None = None,
) -> Path | None:
    """Save only trainable original-backbone tensors, excluding the expert."""
    from safetensors.torch import save_file

    expert_ids = {id(parameter) for parameter in controller.parameters()}
    loop_controller = getattr(transformer, "tread_loop_controller", None)
    if loop_controller is not None:
        expert_ids.update(
            id(parameter) for parameter in loop_controller.expert_parameters()
        )
    tensors = {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in transformer.named_parameters()
        if parameter.requires_grad and id(parameter) not in expert_ids
    }
    if not tensors:
        return None
    if dtype is not None:
        tensors = {
            name: value.to(dtype=dtype).contiguous()
            if value.is_floating_point()
            else value
            for name, value in tensors.items()
        }
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    save_file(tensors, str(temporary))
    os.replace(temporary, path)
    return path


def _translate_legacy_swiglu_state(
    tensors: dict[str, torch.Tensor],
    modules: Mapping[str, nn.Module],
) -> int:
    """Translate legacy FeedForward keys for converted SwiGLU modules."""

    first_suffix = ".net.0.proj.weight"
    prefixes = sorted(
        name[: -len(first_suffix)]
        for name in tensors
        if name.endswith(first_suffix)
    )
    converted_count = 0
    for prefix in prefixes:
        module = modules.get(prefix)
        if not isinstance(module, LightningSwiGLU):
            continue
        names = {
            "first_weight": prefix + ".net.0.proj.weight",
            "first_bias": prefix + ".net.0.proj.bias",
            "second_weight": prefix + ".net.2.weight",
            "second_bias": prefix + ".net.2.bias",
        }
        converted = _legacy_swiglu_tensors(
            tensors[names["first_weight"]],
            tensors.get(names["first_bias"]),
            tensors[names["second_weight"]],
            tensors.get(names["second_bias"]),
            hidden_features=module.hidden_features,
        )
        for name in names.values():
            tensors.pop(name, None)
        for name, value in converted.items():
            tensors[f"{prefix}.{name}"] = value
        converted_count += 1
    return converted_count


def load_terminal_shared_backbone(transformer, path: Path) -> int:
    """Load a shared-backbone replacement from terminal or residual training.

    Residual-expert checkpoints wrapped the native image MLP as ``shared_ffn``.
    The terminal architecture removes that wrapper, so translate only that
    historical namespace back to the native Mage-Flow parameter path.
    """
    from safetensors.torch import load_file

    path = path.expanduser().resolve()
    loaded = load_file(str(path), device="cpu")
    tensors = {}
    for name, value in loaded.items():
        normalized = name.replace(".img_mlp.shared_ffn.", ".img_mlp.")
        if normalized in tensors:
            raise ValueError(
                f"shared-backbone checkpoint has duplicate normalized key: {normalized}"
            )
        tensors[normalized] = value
    _translate_legacy_swiglu_state(
        tensors,
        dict(transformer.named_modules()),
    )
    parameters = dict(transformer.named_parameters())
    unexpected = sorted(set(tensors) - set(parameters))
    if unexpected:
        raise ValueError(f"shared-backbone checkpoint has unexpected keys: {unexpected}")
    with torch.no_grad():
        for name, value in tensors.items():
            target = parameters[name]
            if target.shape != value.shape:
                raise ValueError(
                    f"shared-backbone tensor shape mismatch for {name}: "
                    f"{tuple(value.shape)} != {tuple(target.shape)}"
                )
            target.copy_(value.to(device=target.device, dtype=target.dtype))
    return len(tensors)
