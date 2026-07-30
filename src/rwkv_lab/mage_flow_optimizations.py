"""Opt-in, checkpoint-compatible runtime optimizations for Mage-Flow training.

The helpers in this module do not change the flow-matching objective or the
serialized parameter names.  They are intentionally separate from
representation-alignment work so runtime and convergence experiments can be
qualified independently.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

ACTIVATION_CHECKPOINT_MODES = frozenset(
    {"full", "trainable", "selective", "none"}
)
FLOAT8_RECIPES = frozenset(
    {"tensorwise", "rowwise", "rowwise_with_gw_hp"}
)
ENCODER_CACHE_MODES = frozenset({"off", "read_only", "read_write"})


def configure_activation_checkpointing(
    transformer: torch.nn.Module,
    mode: str,
) -> dict[str, Any]:
    """Configure all-block, trainable-block, selective, or disabled AC.

    ``trainable`` avoids recomputing frozen prefix blocks. ``selective`` also
    keeps expensive GEMM outputs while allowing cheaper pointwise operations
    to be recomputed. The released ``full`` behavior remains the default.
    """
    if mode not in ACTIVATION_CHECKPOINT_MODES:
        raise ValueError(
            f"unsupported activation checkpoint mode {mode!r}; "
            f"expected one of {sorted(ACTIVATION_CHECKPOINT_MODES)}"
        )
    blocks = getattr(transformer, "transformer_blocks", None)
    if blocks is None:
        raise ValueError("Mage-Flow transformer has no transformer_blocks")

    transformer.checkpoint_context_fn = None
    if mode == "none":
        transformer.checkpoint = False
        transformer.checkpoint_block_indices = frozenset()
    elif mode == "full":
        transformer.checkpoint = True
        transformer.checkpoint_block_indices = None
    else:
        trainable = frozenset(
            index
            for index, block in enumerate(blocks)
            if any(parameter.requires_grad for parameter in block.parameters())
        )
        transformer.checkpoint = bool(trainable)
        transformer.checkpoint_block_indices = trainable
        if mode == "selective" and trainable:
            from torch.utils.checkpoint import create_selective_checkpoint_contexts

            aten = torch.ops.aten
            saved_ops = [
                aten.mm.default,
                aten.addmm.default,
                aten.bmm.default,
            ]
            transformer.checkpoint_context_fn = functools.partial(
                create_selective_checkpoint_contexts,
                saved_ops,
            )

    indices = transformer.checkpoint_block_indices
    return {
        "mode": mode,
        "enabled": bool(transformer.checkpoint),
        "block_indices": (
            list(range(len(blocks))) if indices is None else sorted(indices)
        ),
        "selective_saved_ops": (
            ["aten.mm", "aten.addmm", "aten.bmm"]
            if transformer.checkpoint_context_fn is not None
            else []
        ),
    }


def compile_transformer_blocks(
    transformer: torch.nn.Module,
    *,
    enabled: bool,
    mode: str = "default",
    dynamic: bool = True,
    backend: str | None = None,
) -> dict[str, Any]:
    """Apply lazy regional compilation to each repeated DiT block in-place."""
    blocks = getattr(transformer, "transformer_blocks", None)
    if blocks is None:
        raise ValueError("Mage-Flow transformer has no transformer_blocks")
    if not enabled:
        return {
            "enabled": False,
            "block_indices": [],
            "mode": mode,
            "dynamic": dynamic,
            "backend": backend,
        }
    regions = [
        (f"transformer_blocks.{index}", block)
        for index, block in enumerate(blocks)
    ]
    terminal_blocks = getattr(transformer, "terminal_expert_blocks", ())
    regions.extend(
        (f"terminal_expert_blocks.{index}", block)
        for index, block in enumerate(terminal_blocks)
    )
    compiled = []
    for region, block in regions:
        if getattr(block, "_compiled_call_impl", None) is None:
            kwargs: dict[str, Any] = {"mode": mode, "dynamic": dynamic}
            if backend is not None:
                kwargs["backend"] = backend
            block.compile(**kwargs)
        compiled.append(region)
    return {
        "enabled": True,
        "block_indices": list(range(len(blocks))),
        "regions": compiled,
        "mode": mode,
        "dynamic": dynamic,
        "backend": backend,
    }


def _float8_candidate_names(transformer: torch.nn.Module) -> list[str]:
    """Return trainable image-FFN GEMMs on the qualified FP8 allowlist."""
    names = []
    for name, module in transformer.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        # Attention, modulation, text FFNs, projections, and norms stay BF16.
        if ".img_mlp." not in f".{name}.":
            continue
        if not any(parameter.requires_grad for parameter in module.parameters()):
            continue
        if module.in_features % 16 or module.out_features % 16:
            continue
        names.append(name)
    return names


def convert_trainable_image_ffns_to_float8(
    transformer: torch.nn.Module,
    *,
    enabled: bool,
    recipe: str = "tensorwise",
) -> dict[str, Any]:
    """Convert only trainable image-FFN linears to TorchAO Float8Linear."""
    if recipe not in FLOAT8_RECIPES:
        raise ValueError(
            f"unsupported float8 recipe {recipe!r}; "
            f"expected one of {sorted(FLOAT8_RECIPES)}"
        )
    if not enabled:
        return {"enabled": False, "recipe": recipe, "modules": []}
    if not torch.cuda.is_available():
        raise RuntimeError("float8 training requires CUDA")
    capability = torch.cuda.get_device_capability()
    if capability < (8, 9):
        raise RuntimeError(
            f"float8 training requires compute capability >= 8.9, got {capability}"
        )
    try:
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training
    except ImportError as error:
        raise RuntimeError(
            "float8 training requires torchao>=0.17 in the Mage-Flow environment"
        ) from error

    candidates = _float8_candidate_names(transformer)
    if not candidates:
        raise RuntimeError("no trainable image-FFN Linear modules qualified for FP8")
    selected = frozenset(candidates)
    config = Float8LinearConfig.from_recipe_name(recipe)
    convert_to_float8_training(
        transformer,
        config=config,
        module_filter_fn=lambda module, fqn: fqn in selected,
    )
    return {
        "enabled": True,
        "recipe": recipe,
        "modules": candidates,
        "master_parameter_dtypes": sorted(
            {
                str(parameter.dtype).removeprefix("torch.")
                for name, parameter in transformer.named_parameters()
                if any(
                    name == candidate + ".weight"
                    or name == candidate + ".bias"
                    for candidate in selected
                )
            }
        ),
    }


class FrozenEncoderCache:
    """Content-addressed cache for frozen Qwen states and VAE moments.

    VAE posterior *moments* are stored instead of sampled latents, preserving a
    fresh posterior draw on every training visit. Cache entries are safetensors
    and are published atomically.
    """

    schema = "rwkv-lab.mage-flow-frozen-encoder-cache.v1"

    def __init__(
        self,
        root: str | Path,
        *,
        mode: str,
        model_id: str,
        model_revision: str,
    ):
        if mode not in ENCODER_CACHE_MODES - {"off"}:
            raise ValueError("FrozenEncoderCache mode must be read_only or read_write")
        self.root = Path(root).expanduser().resolve()
        self.mode = mode
        self.model_id = model_id
        self.model_revision = model_revision
        self.root.mkdir(parents=True, exist_ok=True)
        contract = {
            "schema": self.schema,
            "model_id": model_id,
            "model_revision": model_revision,
        }
        contract_path = self.root / "cache_contract.json"
        if contract_path.exists():
            existing = json.loads(contract_path.read_text(encoding="utf-8"))
            if existing != contract:
                raise ValueError("frozen-encoder cache contract does not match")
        elif mode == "read_only":
            raise ValueError("read-only frozen-encoder cache has no contract")
        else:
            self._atomic_json(contract_path, contract)

    @staticmethod
    def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)

    @staticmethod
    def _digest(payload: Mapping[str, Any]) -> str:
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _entry(self, kind: str, digest: str) -> Path:
        return self.root / kind / digest[:2] / f"{digest}.safetensors"

    def text_digest(self, prompt: str, template: str, drop_idx: int) -> str:
        return self._digest(
            {
                "kind": "qwen_text",
                "model_id": self.model_id,
                "model_revision": self.model_revision,
                "prompt": prompt,
                "template": template,
                "drop_idx": int(drop_idx),
            }
        )

    def moments_digest(self, row: Mapping[str, Any]) -> str:
        image = Path(str(row["image"])).expanduser().resolve()
        stat = image.stat()
        return self._digest(
            {
                "kind": "vae_posterior_moments",
                "model_id": self.model_id,
                "model_revision": self.model_revision,
                "image": str(image),
                "image_size": stat.st_size,
                "image_mtime_ns": stat.st_mtime_ns,
                "train_width": int(row["train_width"]),
                "train_height": int(row["train_height"]),
            }
        )

    def load_text(
        self,
        prompt: str,
        template: str,
        drop_idx: int,
    ) -> torch.Tensor | None:
        path = self._entry(
            "text", self.text_digest(prompt, template, drop_idx)
        )
        if not path.is_file():
            return None
        return load_file(path, device="cpu")["txt"]

    def has_text(self, prompt: str, template: str, drop_idx: int) -> bool:
        return self._entry(
            "text", self.text_digest(prompt, template, drop_idx)
        ).is_file()

    def save_text(
        self,
        prompt: str,
        template: str,
        drop_idx: int,
        txt: torch.Tensor,
    ) -> Path:
        if self.mode != "read_write":
            raise RuntimeError("cannot populate a read-only encoder cache")
        path = self._entry(
            "text", self.text_digest(prompt, template, drop_idx)
        )
        return self._save(path, {"txt": txt.detach().cpu().contiguous()})

    def load_moments(
        self,
        row: Mapping[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        path = self._entry("moments", self.moments_digest(row))
        if not path.is_file():
            return None
        tensors = load_file(path, device="cpu")
        return tensors["mean"], tensors["logvar"]

    def has_moments(self, row: Mapping[str, Any]) -> bool:
        return self._entry("moments", self.moments_digest(row)).is_file()

    def save_moments(
        self,
        row: Mapping[str, Any],
        mean: torch.Tensor,
        logvar: torch.Tensor,
    ) -> Path:
        if self.mode != "read_write":
            raise RuntimeError("cannot populate a read-only encoder cache")
        path = self._entry("moments", self.moments_digest(row))
        return self._save(
            path,
            {
                "mean": mean.detach().cpu().contiguous(),
                "logvar": logvar.detach().cpu().contiguous(),
            },
        )

    @staticmethod
    def sample_moments(
        mean: torch.Tensor,
        logvar: torch.Tensor,
        *,
        sample_posterior: bool,
    ) -> torch.Tensor:
        if sample_posterior:
            return mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
        return mean

    @staticmethod
    def _save(path: Path, tensors: Mapping[str, torch.Tensor]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file():
            return path
        temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
        save_file(dict(tensors), temporary)
        os.replace(temporary, path)
        return path


def cache_coverage(
    cache: FrozenEncoderCache,
    rows: Sequence[Mapping[str, Any]],
    *,
    prompts: Sequence[str],
    template: str,
    drop_idx: int,
    require_null_prompt: bool,
) -> dict[str, Any]:
    """Report whether a read-only run can avoid both frozen encoders."""
    unique_prompts = set(prompts)
    if require_null_prompt:
        unique_prompts.add(" ")
    missing_text = sum(
        cache.load_text(prompt, template, drop_idx) is None
        for prompt in unique_prompts
    )
    missing_moments = sum(cache.load_moments(row) is None for row in rows)
    return {
        "rows": len(rows),
        "unique_prompts": len(unique_prompts),
        "missing_text": missing_text,
        "missing_moments": missing_moments,
        "complete": missing_text == 0 and missing_moments == 0,
    }


__all__ = [
    "ACTIVATION_CHECKPOINT_MODES",
    "ENCODER_CACHE_MODES",
    "FLOAT8_RECIPES",
    "FrozenEncoderCache",
    "cache_coverage",
    "compile_transformer_blocks",
    "configure_activation_checkpointing",
    "convert_trainable_image_ffns_to_float8",
]
