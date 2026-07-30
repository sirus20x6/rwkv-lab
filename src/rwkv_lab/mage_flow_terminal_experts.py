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
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MethodType
from typing import Any

import torch
from torch import nn

from rwkv_lab.mage_flow_adaptation import (
    EXPERT_DOMAINS,
    MAGE_FLOW_BASE_ID,
    MAGE_FLOW_BASE_REVISION,
)

TERMINAL_EXPERT_SCHEMA = "rwkv-lab.mage-flow-terminal-expert.v1"
TERMINAL_EXPERT_DEPTH = 3

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


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


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


def _terminal_expert_forward(
    transformer,
    img: torch.Tensor,
    txt: torch.Tensor,
    timesteps: torch.Tensor,
    img_shapes=None,
    img_cu_seqlens: torch.Tensor | None = None,
    txt_cu_seqlens: torch.Tensor | None = None,
    attention_kwargs: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Pinned Mage-Flow forward with one terminal expert before the output head."""
    if img.ndim != 3 or txt.ndim != 3:
        raise ValueError("Input img and txt tensors must have 3 dimensions.")
    if getattr(transformer, "_terminal_expert_domain", None) not in EXPERT_DOMAINS:
        raise RuntimeError("no photo or animation terminal expert is resident")

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

    for index, block in enumerate(transformer.transformer_blocks):
        txt, img = _run_block(
            transformer,
            block,
            block_index=index,
            img=img,
            txt=txt,
            temb=temb,
            ms_pe=ms_pe,
            txt_cu_seqlens=txt_cu_seqlens,
            img_cu_seqlens=img_cu_seqlens,
            attention_kwargs=attention_kwargs,
            terminal=False,
        )
    for index, block in enumerate(transformer.terminal_expert_blocks):
        txt, img = _run_block(
            transformer,
            block,
            block_index=index,
            img=img,
            txt=txt,
            temb=temb,
            ms_pe=ms_pe,
            txt_cu_seqlens=txt_cu_seqlens,
            img_cu_seqlens=img_cu_seqlens,
            attention_kwargs=attention_kwargs,
            terminal=True,
        )

    img = transformer.norm_out(img, temb, cu_seqlens=img_cu_seqlens)
    return transformer.proj_out(img)


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


def _terminal_tensors(controller: TerminalExpertController) -> dict[str, torch.Tensor]:
    return {
        f"blocks.{name}": value.detach().cpu().contiguous()
        for name, value in controller.blocks.state_dict().items()
    }


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
    result = controller.blocks.load_state_dict(state, strict=strict)
    controller.transformer._terminal_expert_domain = domain
    return {
        "domain": domain,
        "path": str(path),
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
        "compatible": not result.missing_keys and not result.unexpected_keys,
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
    return {
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

    blocks = transformer.transformer_blocks
    base_indices: tuple[int, ...] = ()
    if train_backbone_final_fraction > 0:
        selected_count = max(1, round(len(blocks) * train_backbone_final_fraction))
        base_indices = tuple(range(len(blocks) - selected_count, len(blocks)))
        for index in base_indices:
            blocks[index].requires_grad_(True)

    expert_ids = {id(parameter) for parameter in controller.parameters()}
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
            for parameter in controller.parameters()
            if parameter.requires_grad
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
) -> list[dict[str, Any]]:
    """Create the required 1.0x expert and 0.5x backbone optimizer groups."""
    if expert_learning_rate <= 0:
        raise ValueError("expert_learning_rate must be positive")
    if backbone_learning_rate_multiplier <= 0:
        raise ValueError("backbone_learning_rate_multiplier must be positive")
    expert_ids = {id(parameter) for parameter in controller.parameters()}
    experts = [
        parameter
        for parameter in transformer.parameters()
        if parameter.requires_grad and id(parameter) in expert_ids
    ]
    backbone = [
        parameter
        for parameter in transformer.parameters()
        if parameter.requires_grad and id(parameter) not in expert_ids
    ]
    if not experts:
        raise RuntimeError("the resident terminal expert is not trainable")
    groups = [
        {
            "params": experts,
            "lr": expert_learning_rate,
            "initial_lr": expert_learning_rate,
            "group_name": "terminal_expert",
        }
    ]
    if backbone:
        backbone_lr = expert_learning_rate * backbone_learning_rate_multiplier
        groups.append(
            {
                "params": backbone,
                "lr": backbone_lr,
                "initial_lr": backbone_lr,
                "group_name": "shared_backbone",
            }
        )
    return groups


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


def load_terminal_shared_backbone(transformer, path: Path) -> int:
    """Load an exact shared-backbone replacement saved by the terminal trainer."""
    from safetensors.torch import load_file

    path = path.expanduser().resolve()
    tensors = load_file(str(path), device="cpu")
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
