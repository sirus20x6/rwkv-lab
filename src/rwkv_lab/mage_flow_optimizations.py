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
import math
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
    loop_controller = getattr(transformer, "tread_loop_controller", None)
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
        terminal_blocks = getattr(transformer, "terminal_expert_blocks", ())
        trainable_terminal = frozenset(
            index
            for index, block in enumerate(terminal_blocks)
            if any(parameter.requires_grad for parameter in block.parameters())
        )
        transformer.checkpoint = bool(trainable or trainable_terminal)
        transformer.checkpoint_block_indices = trainable
        if mode == "selective" and transformer.checkpoint:
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
    terminal_blocks = getattr(transformer, "terminal_expert_blocks", ())
    terminal_indices = (
        list(range(len(terminal_blocks)))
        if transformer.checkpoint and mode == "full"
        else [
            index
            for index, block in enumerate(terminal_blocks)
            if transformer.checkpoint
            and any(parameter.requires_grad for parameter in block.parameters())
        ]
    )
    if loop_controller is not None:
        loop_controller.checkpoint = mode != "none"
        loop_controller.checkpoint_context_fn = transformer.checkpoint_context_fn
    return {
        "mode": mode,
        "enabled": bool(transformer.checkpoint),
        "block_indices": (
            list(range(len(blocks))) if indices is None else sorted(indices)
        ),
        "terminal_block_indices": terminal_indices,
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
    dynamic: bool = False,
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
    # Coverage is a metadata check. Loading every tensor here made a read-only
    # training restart scan the complete (potentially 100+ GiB) cache before
    # the first optimizer step.
    missing_text = sum(
        not cache.has_text(prompt, template, drop_idx)
        for prompt in unique_prompts
    )
    missing_moments = sum(not cache.has_moments(row) for row in rows)
    return {
        "rows": len(rows),
        "unique_prompts": len(unique_prompts),
        "missing_text": missing_text,
        "missing_moments": missing_moments,
        "complete": missing_text == 0 and missing_moments == 0,
    }


class FP32MasterAdamW(torch.optim.AdamW):
    """AdamW with FP32 masters and moments for lower-precision model weights.

    The model parameters remain BF16 so Mage-Flow attention receives a
    supported dtype. AdamW operates on independent FP32 master parameters and
    copies each completed update back to the model. Master weights are included
    in ``state_dict`` so resume does not round them through the BF16 export.
    """

    master_state_key = "_rwkv_lab_fp32_master_weights_v1"

    def __init__(self, params, **kwargs: Any) -> None:
        supplied = list(params)
        if not supplied:
            raise ValueError("FP32MasterAdamW requires parameters")
        if isinstance(supplied[0], Mapping):
            model_groups = [dict(group) for group in supplied]
        else:
            model_groups = [{"params": supplied}]

        master_groups = []
        self._model_master_pairs: list[
            tuple[torch.nn.Parameter, torch.nn.Parameter, bool]
        ] = []
        for model_group in model_groups:
            model_parameters = list(model_group["params"])
            master_parameters = []
            for model_parameter in model_parameters:
                if not isinstance(model_parameter, torch.nn.Parameter):
                    raise TypeError("optimizer parameters must be nn.Parameter")
                if not model_parameter.is_floating_point():
                    raise TypeError("FP32MasterAdamW supports floating parameters only")
                independent = model_parameter.dtype != torch.float32
                master_parameter = (
                    torch.nn.Parameter(
                        model_parameter.detach().to(dtype=torch.float32),
                        requires_grad=True,
                    )
                    if independent
                    else model_parameter
                )
                master_parameters.append(master_parameter)
                self._model_master_pairs.append(
                    (model_parameter, master_parameter, independent)
                )
            master_group = {
                key: value for key, value in model_group.items() if key != "params"
            }
            master_group["params"] = master_parameters
            master_groups.append(master_group)
        super().__init__(master_groups, **kwargs)

    @torch.no_grad()
    def _stage_model_gradients(self) -> None:
        for model, master, independent in self._model_master_pairs:
            if not independent:
                continue
            if model.grad is None:
                master.grad = None
                continue
            if master.grad is None:
                master.grad = torch.empty_like(master)
            master.grad.copy_(model.grad.detach())

    @torch.no_grad()
    def _copy_masters_to_model(self) -> None:
        for model, master, independent in self._model_master_pairs:
            if independent:
                model.copy_(master.to(device=model.device, dtype=model.dtype))

    @torch.no_grad()
    def sync_masters_from_model(self) -> None:
        """Initialize masters from model weights for legacy optimizer resumes."""
        for model, master, independent in self._model_master_pairs:
            if independent:
                master.copy_(model.to(device=master.device, dtype=torch.float32))

    @torch.no_grad()
    def reset_parameter_state(
        self,
        parameters: Sequence[torch.nn.Parameter],
    ) -> int:
        """Reset selected masters and Adam moments from their live model tensors."""

        selected = {id(parameter) for parameter in parameters}
        reset = 0
        for model, master, _independent in self._model_master_pairs:
            if id(model) not in selected:
                continue
            master.copy_(model.detach().to(device=master.device, dtype=master.dtype))
            self.state.pop(master, None)
            model.grad = None
            master.grad = None
            reset += 1
        if reset != len(selected):
            raise ValueError("optimizer could not reset every selected parameter")
        return reset

    def step(
        self,
        closure=None,
        *,
        parameter_update_scales: Mapping[torch.nn.Parameter, float] | None = None,
    ):
        """Apply one AdamW step, optionally scaling selected parameter updates.

        Scaling the completed update instead of the input gradient preserves
        Adam's moments while providing a true per-parameter effective learning
        rate. This is useful for small control tensors that share an optimizer
        group with much larger model weights.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._stage_model_gradients()
        scaled_updates: list[tuple[torch.nn.Parameter, torch.Tensor, float]] = []
        if parameter_update_scales:
            scales_by_id = {
                id(model): float(scale)
                for model, scale in parameter_update_scales.items()
            }
            if any(not math.isfinite(scale) or scale <= 0 for scale in scales_by_id.values()):
                raise ValueError("parameter update scales must be finite and positive")
            for model, master, _independent in self._model_master_pairs:
                scale = scales_by_id.get(id(model), 1.0)
                if scale != 1.0 and master.grad is not None:
                    scaled_updates.append((master, master.detach().clone(), scale))
        observed = super().step()
        with torch.no_grad():
            for master, previous, scale in scaled_updates:
                master.copy_(previous + scale * (master - previous))
        self._copy_masters_to_model()
        return loss if closure is not None else observed

    def zero_grad(self, set_to_none: bool = True) -> None:
        super().zero_grad(set_to_none=set_to_none)
        with torch.no_grad():
            for model, _master, independent in self._model_master_pairs:
                if not independent or model.grad is None:
                    continue
                if set_to_none:
                    model.grad = None
                else:
                    model.grad.zero_()

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state[self.master_state_key] = [
            master.detach()
            for _model, master, independent in self._model_master_pairs
            if independent
        ]
        return state

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        restored = dict(state_dict)
        master_weights = restored.pop(self.master_state_key, None)
        super().load_state_dict(restored)
        independent_pairs = [
            (model, master)
            for model, master, independent in self._model_master_pairs
            if independent
        ]
        if master_weights is None:
            self.sync_masters_from_model()
            return
        if len(master_weights) != len(independent_pairs):
            raise ValueError(
                "FP32 master checkpoint parameter count does not match optimizer"
            )
        with torch.no_grad():
            for (model, master), saved in zip(
                independent_pairs,
                master_weights,
                strict=True,
            ):
                if saved.shape != model.shape:
                    raise ValueError(
                        "FP32 master checkpoint parameter shape does not match model"
                    )
                master.copy_(saved.to(device=master.device, dtype=torch.float32))
                model.copy_(master.to(device=model.device, dtype=model.dtype))

    def precision_report(self) -> dict[str, Any]:
        independent = [
            (model, master)
            for model, master, is_independent in self._model_master_pairs
            if is_independent
        ]
        return {
            "name": type(self).__name__,
            "model_parameter_dtypes": sorted(
                {str(model.dtype).removeprefix("torch.") for model, _master in independent}
            ),
            "master_parameter_dtype": "float32",
            "moment_dtype": "float32",
            "independent_master_parameter_count": sum(
                master.numel() for _model, master in independent
            ),
            "exact_master_resume": True,
        }


__all__ = [
    "ACTIVATION_CHECKPOINT_MODES",
    "ENCODER_CACHE_MODES",
    "FLOAT8_RECIPES",
    "FP32MasterAdamW",
    "FrozenEncoderCache",
    "cache_coverage",
    "compile_transformer_blocks",
    "configure_activation_checkpointing",
    "convert_trainable_image_ffns_to_float8",
]
