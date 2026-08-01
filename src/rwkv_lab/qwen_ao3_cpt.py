"""MoE-aware continued pretraining for Qwen3.6-35B-A3B on packed AO3 prose.

This trainer uses a hybrid QLoRA layout dictated by the checkpoint geometry:
ordinary language-model linears use NF4 frozen bases, while the fused 3-D MoE
expert parameters remain frozen BF16 and receive per-expert rsLoRA updates.
Standard bitsandbytes QLoRA does not quantize those 3-D parameters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shlex
import signal
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from rwkv_lab.training_components import (
    AdamWConfiguration,
    OptimizerImplementation,
    PowerCoolConfiguration,
    build_registered_optimizer,
    powercool_multiplier,
)

if TYPE_CHECKING:
    from rwkv_lab.trainvm_adapters import WorkerTrainingComponents
    from rwkv_lab.trainvm_worker import WorkerStepProfiler

SCHEMA = "rwkv-lab.qwen-ao3-cpt.v1"
ROUTER_TARGET = "mlp.gate.weight"
DENSE_TARGETS = (
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "embed_tokens",
    "lm_head",
)
EXPERT_ADAPTER_PREFIX = "expert_adapter_"


class GroupedExpertLoRA(torch.nn.Module):
    """Apply per-expert LoRA with grouped GEMMs and no dense delta weights.

    PEFT's generic ``target_parameters`` implementation forms ``W + B @ A``
    for the complete 3-D expert bank on every forward.  A single Qwen layer's
    expert banks are about 1.5 GiB, so that path would defeat both the fit and
    performance goals.  This wrapper keeps the frozen BF16 banks untouched and
    applies the low-rank factors directly to routed token/expert pairs.
    """

    def __init__(self, base_layer: torch.nn.Module, rank: int, alpha: int):
        super().__init__()
        if rank < 1:
            raise ValueError("expert rank must be positive")
        self.base_layer = base_layer
        self.num_experts = int(base_layer.num_experts)
        hidden = int(base_layer.hidden_dim)
        intermediate = int(base_layer.intermediate_dim)
        self.rank = int(rank)
        self.scaling = float(alpha) / math.sqrt(rank)
        self.expert_adapter_gate_up_A = torch.nn.Parameter(
            torch.empty(self.num_experts, rank, hidden, dtype=torch.float32)
        )
        self.expert_adapter_gate_up_B = torch.nn.Parameter(
            torch.zeros(self.num_experts, 2 * intermediate, rank, dtype=torch.float32)
        )
        self.expert_adapter_down_A = torch.nn.Parameter(
            torch.empty(self.num_experts, rank, intermediate, dtype=torch.float32)
        )
        self.expert_adapter_down_B = torch.nn.Parameter(
            torch.zeros(self.num_experts, hidden, rank, dtype=torch.float32)
        )
        torch.nn.init.uniform_(
            self.expert_adapter_gate_up_A,
            -1.0 / math.sqrt(hidden),
            1.0 / math.sqrt(hidden),
        )
        torch.nn.init.uniform_(
            self.expert_adapter_down_A,
            -1.0 / math.sqrt(intermediate),
            1.0 / math.sqrt(intermediate),
        )
        self.to(device=base_layer.gate_up_proj.device)

    @staticmethod
    def _compute_offsets(
        expert_ids: torch.Tensor, num_experts: int
    ) -> torch.Tensor:
        histc_input = (
            expert_ids.float() if expert_ids.device.type == "cpu" else expert_ids.int()
        )
        counts = torch.histc(
            histc_input, bins=num_experts, min=0, max=num_experts - 1
        )
        return torch.cumsum(counts, dim=0, dtype=torch.int32)

    @staticmethod
    def _grouped_linear(
        inputs: torch.Tensor, weights: torch.Tensor, offsets: torch.Tensor
    ) -> torch.Tensor:
        from transformers.integrations.moe import _grouped_linear

        return _grouped_linear(
            inputs, weights, offsets, bias=None, is_transposed=False
        )

    def _adapter_linear(
        self,
        inputs: torch.Tensor,
        factor_a: torch.Tensor,
        factor_b: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        # Keep master adapter parameters in FP32, but execute their two small
        # grouped GEMMs in the model compute dtype. PyTorch's grouped GEMM
        # requires BF16 dimensions/strides aligned to 16 bytes, so pad the
        # compute rank to eight without adding trainable parameters.
        dtype = inputs.dtype
        factor_a_compute = factor_a.to(dtype)
        factor_b_compute = factor_b.to(dtype)
        padded_rank = math.ceil(self.rank / 8) * 8
        if padded_rank != self.rank:
            padding = padded_rank - self.rank
            factor_a_compute = torch.nn.functional.pad(
                factor_a_compute, (0, 0, 0, padding)
            )
            factor_b_compute = torch.nn.functional.pad(
                factor_b_compute, (0, padding)
            )
        low_rank = self._grouped_linear(inputs, factor_a_compute, offsets)
        return self._grouped_linear(low_rank, factor_b_compute, offsets)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        device = hidden_states.device
        num_top_k = top_k_index.size(-1)
        num_tokens = hidden_states.size(0)
        hidden_dim = hidden_states.size(-1)
        token_idx = (
            torch.arange(num_tokens, device=device)
            .unsqueeze(1)
            .expand(-1, num_top_k)
            .reshape(-1)
        )
        expert_ids = top_k_index.reshape(-1)
        permutation = torch.argsort(expert_ids)
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(permutation.numel(), device=device)
        grouped_experts = expert_ids[permutation]
        grouped_weights = top_k_weights.reshape(-1)[permutation]
        grouped_hidden = hidden_states[token_idx[permutation]]
        offsets = self._compute_offsets(grouped_experts, self.num_experts)

        gate_up = self._grouped_linear(
            grouped_hidden, self.base_layer.gate_up_proj, offsets
        )
        gate_up = gate_up + self.scaling * self._adapter_linear(
            grouped_hidden,
            self.expert_adapter_gate_up_A,
            self.expert_adapter_gate_up_B,
            offsets,
        )
        activated = self.base_layer._apply_gate(gate_up)
        projected = self._grouped_linear(
            activated, self.base_layer.down_proj, offsets
        )
        projected = projected + self.scaling * self._adapter_linear(
            activated,
            self.expert_adapter_down_A,
            self.expert_adapter_down_B,
            offsets,
        )
        weighted = projected * grouped_weights.unsqueeze(-1)
        weighted = weighted[inverse]
        return weighted.view(num_tokens, num_top_k, hidden_dim).sum(dim=1).to(
            hidden_states.dtype
        )


@dataclass(frozen=True)
class QwenAO3Config:
    model_dir: str
    train_pack_dir: str
    eval_pack_dir: str
    run_dir: str
    context_length: int = 8192
    gradient_accumulation_steps: int = 8
    dense_rank: int = 64
    dense_alpha: int = 128
    expert_rank: int = 4
    expert_alpha: int = 8
    router_rank: int = 8
    router_alpha: int = 8
    learning_rate: float = 5.0e-5
    min_learning_rate: float = 5.0e-6
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1.0e-8
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.01
    powercool_cooldown_fraction: float = 0.20
    powercool_power: float = 2.0
    router_aux_loss_coef: float = 0.001
    max_steps: int = 0
    eval_every: int = 500
    eval_batches: int = 16
    save_every: int = 500
    log_every: int = 10
    seed: int = 20260727
    cuda_index: int = 0
    experts_implementation: str = "grouped_mm"
    attention_implementation: str = "flash_attention_2"
    minimum_free_vram_gib: float = 92.0
    smoke_context: int = 512
    resume: str = ""
    auto_resume: bool = True

    def validate(self, *, require_paths: bool = True) -> None:
        if self.context_length < 128:
            raise ValueError("context_length must be at least 128")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        for name in ("dense_rank", "expert_rank", "router_rank"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.learning_rate <= 0 or not 0 <= self.min_learning_rate <= self.learning_rate:
            raise ValueError("require 0 <= min_learning_rate <= learning_rate")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if self.max_steps < 0:
            raise ValueError("max_steps cannot be negative")
        if self.eval_batches < 1 or self.log_every < 1:
            raise ValueError("eval_batches and log_every must be positive")
        if self.smoke_context < 32 or self.smoke_context > self.context_length:
            raise ValueError("smoke_context must be in [32, context_length]")
        if require_paths:
            for name in ("model_dir", "train_pack_dir", "eval_pack_dir"):
                if not Path(getattr(self, name)).exists():
                    raise FileNotFoundError(getattr(self, name))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_shapes(model_dir: Path) -> dict[str, tuple[int, ...]]:
    from safetensors import safe_open

    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    by_file: dict[str, list[str]] = defaultdict(list)
    for name, filename in index["weight_map"].items():
        by_file[filename].append(name)
    shapes = {}
    for filename, names in by_file.items():
        with safe_open(model_dir / filename, framework="pt", device="cpu") as handle:
            for name in names:
                shapes[name] = tuple(handle.get_slice(name).get_shape())
    return shapes


def checkpoint_fit_audit(
    model_dir: str | Path,
    *,
    dense_rank: int = 64,
    expert_rank: int = 4,
    router_rank: int = 8,
    activation_reserve_gib: float = 16.0,
    runtime_reserve_gib: float = 4.0,
) -> dict[str, Any]:
    """Estimate resident/training storage from checkpoint tensor geometry."""
    model_dir = Path(model_dir)
    shapes = _checkpoint_shapes(model_dir)
    groups: Counter[str] = Counter()
    dense_adapter = expert_adapter = router_adapter = 0
    dense_suffixes = tuple(f".{name}.weight" for name in DENSE_TARGETS)
    for name, shape in shapes.items():
        count = math.prod(shape)
        if name.startswith("model.language_model.visual"):
            group = "vision"
        elif name.startswith("mtp."):
            group = "mtp"
        elif ".mlp.experts." in name:
            group = "fused_experts"
        elif name.startswith("model.language_model.embed_tokens"):
            group = "embedding"
        elif name.startswith("lm_head"):
            group = "lm_head"
        elif name.startswith("model.language_model"):
            group = "language_other"
        else:
            group = "other"
        groups[group] += count * 2

        is_language_weight = name.startswith(
            (
                "model.language_model.layers.",
                "model.language_model.embed_tokens.",
                "lm_head.",
            )
        )
        if is_language_weight and ".mlp.experts." in name and len(shape) == 3:
            experts, out_features, in_features = shape
            expert_adapter += experts * expert_rank * (out_features + in_features)
        elif is_language_weight and name.endswith(".mlp.gate.weight") and len(shape) == 2:
            out_features, in_features = shape
            router_adapter += router_rank * (out_features + in_features)
        elif is_language_weight and len(shape) == 2 and (
            name.endswith(dense_suffixes) or name == "lm_head.weight"
        ):
            out_features, in_features = shape
            dense_adapter += dense_rank * (out_features + in_features)

    # NF4 is approximately 0.5 byte/value plus block scales.  Use 0.28 of
    # BF16 storage as a conservative allowance for the language linears.
    quantized_other = groups["language_other"] * 0.28
    resident_bytes = (
        groups["fused_experts"]
        + groups["embedding"]
        + groups["lm_head"]
        + quantized_other
    )
    adapter_parameters = dense_adapter + expert_adapter + router_adapter
    # FP32 adapter parameter + gradient + two Adam moments = 16 bytes/value.
    adapter_training_bytes = adapter_parameters * 16
    estimated_peak = (
        resident_bytes
        + adapter_training_bytes
        + (activation_reserve_gib + runtime_reserve_gib) * 2**30
    )
    language_bf16 = (
        groups["fused_experts"]
        + groups["embedding"]
        + groups["lm_head"]
        + groups["language_other"]
    )
    # Full AdamW CPT: BF16 parameter + BF16 gradient + FP32 m/v.
    full_training_floor = language_bf16 * 6
    return {
        "schema": SCHEMA,
        "checkpoint": str(model_dir.resolve()),
        "checkpoint_groups_gib": {
            key: value / 2**30 for key, value in sorted(groups.items())
        },
        "adapter_parameters": adapter_parameters,
        "adapter_breakdown": {
            "dense": dense_adapter,
            "experts": expert_adapter,
            "routers": router_adapter,
        },
        "estimated_resident_base_gib": resident_bytes / 2**30,
        "estimated_adapter_training_gib": adapter_training_bytes / 2**30,
        "activation_reserve_gib": activation_reserve_gib,
        "runtime_reserve_gib": runtime_reserve_gib,
        "estimated_peak_gib": estimated_peak / 2**30,
        "full_adamw_floor_gib": full_training_floor / 2**30,
        "standard_bnb_quantizes_fused_experts": False,
        "expert_adapter_implementation": "routed_grouped_low_rank",
        "materializes_dense_expert_delta": False,
    }


class PackedRows:
    def __init__(self, directory: str | Path, expected_context: int):
        directory = Path(directory)
        manifest_path = directory / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text())
        context = int(self.manifest["context_length"])
        if context != expected_context:
            raise ValueError(f"packed context {context} != configured {expected_context}")
        self.context = context
        self.rows = int(self.manifest["rows"])
        self.path = directory / self.manifest["packed_file"]
        expected_bytes = self.rows * self.context * np.dtype(np.uint32).itemsize
        if self.path.stat().st_size != expected_bytes:
            raise ValueError(f"{self.path}: packed size does not match manifest")
        self.values = np.memmap(
            self.path, mode="r", dtype=np.uint32, shape=(self.rows, self.context)
        )
        self.manifest_sha256 = _sha256(manifest_path)

    def tensor(self, row: int, device: torch.device, *, length: int | None = None) -> torch.Tensor:
        value = np.array(self.values[row, : length or self.context], copy=True)
        return torch.from_numpy(value.astype(np.int64, copy=False)).unsqueeze(0).to(
            device, non_blocking=True
        )


def _free_vram_gib(cuda_index: int) -> tuple[float, float]:
    with torch.cuda.device(cuda_index):
        free, total = torch.cuda.mem_get_info()
    return free / 2**30, total / 2**30


def _base_model(model):
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def _language_parts(model):
    base = _base_model(model)
    return base.model.language_model, base.lm_head


def _adapter_config(config: QwenAO3Config):
    from peft import LoraConfig, TaskType

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.dense_rank,
        lora_alpha=config.dense_alpha,
        lora_dropout=0.0,
        use_rslora=True,
        bias="none",
        target_modules=list(DENSE_TARGETS),
        target_parameters=[ROUTER_TARGET],
        rank_pattern={
            "mlp.gate.weight": config.router_rank,
        },
        alpha_pattern={
            "mlp.gate.weight": config.router_alpha,
        },
        init_lora_weights=True,
    )


def _install_grouped_expert_adapters(base, config: QwenAO3Config) -> None:
    layers = base.model.language_model.layers
    for layer in layers:
        experts = layer.mlp.experts
        if isinstance(experts, GroupedExpertLoRA):
            raise TypeError("expert adapters were installed more than once")
        layer.mlp.experts = GroupedExpertLoRA(
            experts, rank=config.expert_rank, alpha=config.expert_alpha
        )


def _set_expert_adapters_trainable(model) -> None:
    found = 0
    for name, parameter in model.named_parameters():
        if EXPERT_ADAPTER_PREFIX in name:
            parameter.requires_grad_(True)
            found += 1
    if not found:
        raise RuntimeError("no grouped expert adapter parameters were installed")


def _save_expert_adapters(model, path: Path) -> None:
    from safetensors.torch import save_file

    state = {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in model.named_parameters()
        if EXPERT_ADAPTER_PREFIX in name
    }
    if not state:
        raise RuntimeError("no grouped expert adapters to save")
    save_file(state, path, metadata={"schema": SCHEMA})


def _load_expert_adapters(model, path: Path) -> None:
    from safetensors import safe_open

    if not path.is_file():
        raise FileNotFoundError(path)
    expected = {
        name: parameter
        for name, parameter in model.named_parameters()
        if EXPERT_ADAPTER_PREFIX in name
    }
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        if keys != set(expected):
            missing = sorted(set(expected) - keys)
            unexpected = sorted(keys - set(expected))
            raise RuntimeError(
                f"expert adapter state mismatch; missing={missing[:3]}, "
                f"unexpected={unexpected[:3]}"
            )
        with torch.no_grad():
            for name, parameter in expected.items():
                value = handle.get_tensor(name)
                if tuple(value.shape) != tuple(parameter.shape):
                    raise RuntimeError(f"expert adapter shape mismatch for {name}")
                parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def load_hybrid_qlora(config: QwenAO3Config):
    """Load without PEFT's k-bit helper, which would cast BF16 experts to FP32."""
    from peft import PeftModel, get_peft_model
    from transformers import BitsAndBytesConfig, Qwen3_5MoeForConditionalGeneration

    free_gib, total_gib = _free_vram_gib(config.cuda_index)
    if free_gib < config.minimum_free_vram_gib:
        raise RuntimeError(
            f"need at least {config.minimum_free_vram_gib:.1f} GiB free VRAM before model load; "
            f"found {free_gib:.1f}/{total_gib:.1f} GiB"
        )
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.uint8,
    )
    with torch.cuda.device(config.cuda_index):
        base = Qwen3_5MoeForConditionalGeneration.from_pretrained(
            config.model_dir,
            local_files_only=True,
            dtype=torch.bfloat16,
            quantization_config=quantization,
            device_map={"": config.cuda_index},
            low_cpu_mem_usage=True,
            attn_implementation=config.attention_implementation,
            experts_implementation=config.experts_implementation,
        )
    # This is text-only CPT.  The visual tower was loaded for checkpoint
    # compatibility but must not consume resident training memory.
    base.model.visual = None
    base.config.use_cache = False
    base.config.text_config.use_cache = False
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    _install_grouped_expert_adapters(base, config)
    base.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    if config.resume:
        adapter_dir = Path(config.resume) / "adapter"
        model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=True)
    else:
        model = get_peft_model(base, _adapter_config(config), autocast_adapter_dtype=True)
    _set_expert_adapters_trainable(model)
    if config.resume:
        _load_expert_adapters(
            model, Path(config.resume) / "expert-adapter.safetensors"
        )
    return model


def model_coverage_report(model, expected_layers: int = 40) -> dict[str, Any]:
    from bitsandbytes.nn import Params4bit

    base = _base_model(model)
    named = dict(base.named_parameters())
    expert_bf16 = {
        name: parameter
        for name, parameter in named.items()
        if ".mlp.experts." in name
        and name.endswith(("gate_up_proj", "down_proj"))
    }
    if len(expert_bf16) != expected_layers * 2:
        raise RuntimeError(
            f"expected {expected_layers * 2} fused expert bases, found {len(expert_bf16)}"
        )
    wrong_experts = [
        name
        for name, parameter in expert_bf16.items()
        if parameter.dtype != torch.bfloat16 or parameter.requires_grad
    ]
    if wrong_experts:
        raise RuntimeError(f"fused experts must be frozen BF16: {wrong_experts[:4]}")
    trainable = {
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    expert_adapters = [
        name for name in trainable if EXPERT_ADAPTER_PREFIX in name
    ]
    router_adapters = [name for name in trainable if ".mlp.gate." in name]
    if len(expert_adapters) != expected_layers * 4:
        raise RuntimeError("expert LoRA coverage is incomplete")
    wrong_adapter_dtype = [
        name for name in expert_adapters if trainable[name].dtype != torch.float32
    ]
    if wrong_adapter_dtype:
        raise RuntimeError(
            f"expert adapter masters must be FP32: {wrong_adapter_dtype[:3]}"
        )
    if len(router_adapters) < expected_layers * 2:
        raise RuntimeError("router LoRA coverage is incomplete")
    quantized = [
        name for name, parameter in named.items() if isinstance(parameter, Params4bit)
    ]
    if not quantized:
        raise RuntimeError("no ordinary language linears were NF4-quantized")
    materializing_expert_wrappers = [
        name
        for name, module in base.named_modules()
        if ".mlp.experts" in name and module.__class__.__name__ == "ParamWrapper"
    ]
    if materializing_expert_wrappers:
        raise RuntimeError(
            "generic PEFT expert wrappers would materialize dense deltas: "
            f"{materializing_expert_wrappers[:2]}"
        )
    storage = sum(p.numel() * p.element_size() for p in base.parameters())
    trainable_count = sum(p.numel() for p in trainable.values())
    return {
        "schema": SCHEMA,
        "frozen_expert_tensors": len(expert_bf16),
        "expert_adapter_tensors": len(expert_adapters),
        "expert_adapter_implementation": "routed_grouped_low_rank",
        "materializing_expert_wrappers": 0,
        "router_adapter_tensors": len(router_adapters),
        "nf4_parameter_tensors": len(quantized),
        "trainable_parameters": trainable_count,
        "parameter_storage_gib": storage / 2**30,
        "cuda_allocated_gib": torch.cuda.memory_allocated() / 2**30,
    }


def _causal_loss(
    model,
    tokens: torch.Tensor,
    *,
    router_aux_coef: float,
    training: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    from rwkv_lab.fused_ce import HAS_FUSED_CE, lmhead_cross_entropy

    if not HAS_FUSED_CE and tokens.is_cuda:
        raise RuntimeError("flash-attn Triton cross entropy is required for this vocabulary")
    text_model, lm_head = _language_parts(model)
    inputs = tokens[:, :-1]
    labels = tokens[:, 1:]
    outputs = text_model(
        input_ids=inputs,
        use_cache=False,
        output_router_logits=bool(training and router_aux_coef),
    )
    lm_loss = lmhead_cross_entropy(
        outputs.last_hidden_state,
        lm_head,
        labels,
        fused=True,
    )
    aux_loss = None
    total = lm_loss
    if training and router_aux_coef:
        if not outputs.router_logits:
            raise RuntimeError("router auxiliary logits were requested but not returned")
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            load_balancing_loss_func,
        )

        text_config = _base_model(model).config.text_config
        aux_loss = load_balancing_loss_func(
            outputs.router_logits,
            text_config.num_experts,
            text_config.num_experts_per_tok,
            attention_mask=None,
        )
        total = total + router_aux_coef * aux_loss.to(total.device)
    return total, lm_loss.detach(), None if aux_loss is None else aux_loss.detach()


@torch.no_grad()
def evaluate(model, rows: PackedRows, config: QwenAO3Config) -> dict[str, float]:
    model.eval()
    device = torch.device(f"cuda:{config.cuda_index}")
    loss_total = torch.zeros((), device=device, dtype=torch.float32)
    batches = min(rows.rows, config.eval_batches)
    for row in range(batches):
        tokens = rows.tensor(row, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, lm_loss, _ = _causal_loss(
                model, tokens, router_aux_coef=0.0, training=False
            )
        loss_total.add_(lm_loss.float())
    mean = float(loss_total / batches)
    model.train()
    return {"loss": mean, "perplexity": math.exp(min(mean, 20.0)), "batches": batches}


def _optimizer_configuration(config: QwenAO3Config) -> AdamWConfiguration:
    return AdamWConfiguration(
        learning_rate=config.learning_rate,
        beta1=config.adam_beta1,
        beta2=config.adam_beta2,
        epsilon=config.adam_epsilon,
        weight_decay=config.weight_decay,
        foreach=False,
        fused=True,
    )


def _learning_rate_schedule(
    config: QwenAO3Config, total_steps: int
) -> PowerCoolConfiguration:
    return PowerCoolConfiguration(
        warmup_steps=max(1, round(total_steps * config.warmup_ratio)),
        max_steps=total_steps,
        minimum_ratio=config.min_learning_rate / config.learning_rate,
        cooldown_fraction=config.powercool_cooldown_fraction,
        power=config.powercool_power,
    )


def _optimizer(
    model,
    config: QwenAO3Config,
    worker_components: WorkerTrainingComponents | None = None,
):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("model has no trainable adapter parameters")
    if worker_components is not None:
        return worker_components.optimizer(parameters)
    return build_registered_optimizer(
        OptimizerImplementation.TORCH_ADAMW_V1,
        parameters,
        _optimizer_configuration(config),
    )


def _resume_contract(config: QwenAO3Config | dict[str, Any]) -> dict[str, Any]:
    value = asdict(config) if isinstance(config, QwenAO3Config) else dict(config)
    value.pop("resume", None)
    value.pop("auto_resume", None)
    # Telemetry cadence does not affect model or optimizer state and may need
    # adjustment after real step latency is known.
    value.pop("log_every", None)
    return value


def resolved_worker_component_contract(
    config: QwenAO3Config,
    total_steps: int,
    worker_components: WorkerTrainingComponents | None,
) -> tuple[dict[str, dict[str, str]] | None, str | None]:
    if worker_components is None:
        return None, None
    expected = {
        "optimizer": asdict(_optimizer_configuration(config)),
        "learning_rate": asdict(_learning_rate_schedule(config, total_steps)),
    }
    categories = {
        "optimizer": "optimizer",
        "learning_rate": "learning_rate_schedule",
    }
    for slot, configuration in expected.items():
        actual = dict(
            worker_components.configuration(slot, category=categories[slot])
        )
        if actual != configuration:
            raise ValueError(
                f"authority {categories[slot]} composition disagrees with "
                "Qwen training configuration"
            )
    return (
        dict(worker_components.evidence()),
        worker_components.composition.composition_digest,
    )


def _write_status(
    run_dir: Path,
    *,
    state: str,
    step: int,
    cursor: int,
    total_steps: int,
) -> None:
    """Publish an atomic heartbeat during long full-context microbatches."""
    path = run_dir / "status.json"
    temporary = run_dir / ".status.json.tmp"
    temporary.write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "state": state,
                "step": step,
                "cursor": cursor,
                "total_steps": total_steps,
                "updated_at": time.time(),
            }
        )
        + "\n"
    )
    temporary.replace(path)


def _save_checkpoint(
    model,
    optimizer,
    scheduler,
    config: QwenAO3Config,
    step: int,
    cursor: int,
    run_dir: Path,
    metrics: dict[str, Any],
    component_composition_digest: str | None = None,
) -> Path:
    final = run_dir / f"step-{step:06d}"
    temporary = run_dir / f".step-{step:06d}.tmp-{time.time_ns()}"
    temporary.mkdir(parents=True)
    # Embeddings and lm_head are LoRA targets, so PEFT otherwise auto-saves
    # their frozen BF16 base weights as "embedding layers".  The immutable
    # base checkpoint already owns those ~1.9 GiB; store only adapter deltas.
    model.save_pretrained(
        temporary / "adapter",
        safe_serialization=True,
        save_embedding_layers=False,
    )
    _save_expert_adapters(model, temporary / "expert-adapter.safetensors")
    state = {
        "schema": SCHEMA,
        "step": step,
        "cursor": cursor,
        "config": _resume_contract(config),
        "optimizer": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(config.cuda_index),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
    }
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if component_composition_digest is not None:
        state["component_composition_digest"] = component_composition_digest
    torch.save(state, temporary / "trainer-state.pt")
    (temporary / "state.json").write_text(
        json.dumps(
            {"schema": SCHEMA, "step": step, "cursor": cursor, "metrics": metrics},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    temporary.rename(final)
    latest = run_dir / "latest.json"
    latest.write_text(
        json.dumps({"schema": SCHEMA, "step": step, "checkpoint": str(final.resolve())})
        + "\n"
    )
    return final


def _restore_state(
    optimizer,
    scheduler,
    config: QwenAO3Config,
    *,
    component_composition_digest: str | None = None,
) -> tuple[int, int]:
    if not config.resume:
        return 0, 0
    state = torch.load(
        Path(config.resume) / "trainer-state.pt",
        map_location=f"cuda:{config.cuda_index}",
        weights_only=False,
    )
    if state.get("schema") != SCHEMA:
        raise ValueError("resume checkpoint schema mismatch")
    if _resume_contract(state.get("config", {})) != _resume_contract(config):
        raise ValueError("resume checkpoint training contract mismatch")
    if component_composition_digest is not None:
        if state.get("component_composition_digest") != component_composition_digest:
            raise ValueError("resume training-component composition mismatch")
        if scheduler is None or "scheduler" not in state:
            raise ValueError("resume checkpoint has no exact LR-schedule state")
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["torch_rng"].cpu())
    torch.cuda.set_rng_state(state["cuda_rng"].cpu(), config.cuda_index)
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    return int(state["step"]), int(state["cursor"])


def _resolve_auto_resume(config: QwenAO3Config, run_dir: Path) -> QwenAO3Config:
    if config.resume or not config.auto_resume:
        return config
    latest = run_dir / "latest.json"
    if latest.is_file():
        value = json.loads(latest.read_text())
        if value.get("schema") != SCHEMA:
            raise ValueError("latest checkpoint schema mismatch")
        checkpoint = Path(str(value.get("checkpoint") or ""))
    else:
        # Recover the narrow crash window after the atomic checkpoint rename
        # but before latest.json is written.
        candidates = sorted(
            path
            for path in run_dir.glob("step-*")
            if path.is_dir()
            and (path / "trainer-state.pt").is_file()
            and (path / "state.json").is_file()
        )
        if not candidates:
            return config
        checkpoint = candidates[-1]
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    print(f"auto-resuming from {checkpoint}", flush=True)
    return replace(config, resume=str(checkpoint.resolve()))


def _smoke_backward(model, rows: PackedRows, config: QwenAO3Config) -> dict[str, Any]:
    device = torch.device(f"cuda:{config.cuda_index}")
    tokens = rows.tensor(0, device, length=config.smoke_context)
    model.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, lm_loss, aux_loss = _causal_loss(
            model,
            tokens,
            router_aux_coef=config.router_aux_loss_coef,
            training=True,
        )
    loss.backward()
    named_gradients = {
        name: parameter.grad
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    }
    gradients = list(named_gradients.values())
    finite = bool(gradients) and all(torch.isfinite(value).all() for value in gradients)
    nonzero = any(torch.count_nonzero(value).item() for value in gradients)
    families = {
        "experts": [
            value
            for name, value in named_gradients.items()
            if EXPERT_ADAPTER_PREFIX in name
        ],
        "routers": [
            value
            for name, value in named_gradients.items()
            if ".mlp.gate." in name
        ],
        "dense": [
            value
            for name, value in named_gradients.items()
            if "lora_" in name and ".mlp.gate." not in name
        ],
    }
    disconnected = [
        family
        for family, values in families.items()
        if not values or not any(torch.count_nonzero(value).item() for value in values)
    ]
    model.zero_grad(set_to_none=True)
    if not finite or not nonzero or disconnected:
        raise RuntimeError(
            f"adapter smoke backward produced invalid gradients; "
            f"disconnected={disconnected}"
        )
    return {
        "loss": float(lm_loss),
        "router_aux_loss": None if aux_loss is None else float(aux_loss),
        "gradient_tensors": len(gradients),
        "gradient_family_tensors": {
            family: len(values) for family, values in families.items()
        },
        "finite": finite,
        "nonzero": nonzero,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
    }


def train(
    config: QwenAO3Config,
    *,
    worker_components: WorkerTrainingComponents | None = None,
    worker_step_profiler: WorkerStepProfiler | None = None,
) -> dict[str, Any]:
    run_dir = Path(config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    complete_path = run_dir / "complete.json"
    if config.auto_resume and not config.resume and complete_path.is_file():
        result = json.loads(complete_path.read_text())
        result["status"] = "already_complete"
        return result
    config = _resolve_auto_resume(config, run_dir)
    config.validate()
    train_rows = PackedRows(config.train_pack_dir, config.context_length)
    eval_rows = PackedRows(config.eval_pack_dir, config.context_length)
    receipt_path = run_dir / "run-receipt.json"
    if config.resume:
        if not receipt_path.is_file():
            raise FileNotFoundError(
                f"resume requires the original run receipt: {receipt_path}"
            )
        original_receipt = json.loads(receipt_path.read_text())
        if original_receipt.get("schema") != SCHEMA:
            raise ValueError("resume run receipt schema mismatch")
        if original_receipt.get("train_pack_manifest_sha256") != train_rows.manifest_sha256:
            raise ValueError("resume train pack fingerprint mismatch")
        if original_receipt.get("eval_pack_manifest_sha256") != eval_rows.manifest_sha256:
            raise ValueError("resume eval pack fingerprint mismatch")
        if _resume_contract(original_receipt.get("config", {})) != _resume_contract(config):
            raise ValueError("resume run configuration mismatch")
    available_steps = math.ceil(
        train_rows.rows / config.gradient_accumulation_steps
    )
    total_steps = min(config.max_steps or available_steps, available_steps)
    if total_steps < 1:
        raise ValueError("packed train corpus has no complete optimizer step")
    learning_rate_schedule = _learning_rate_schedule(config, total_steps)
    component_evidence, component_digest = resolved_worker_component_contract(
        config, total_steps, worker_components
    )
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    model = load_hybrid_qlora(config)
    coverage = model_coverage_report(model)
    optimizer = _optimizer(model, config, worker_components)
    scheduler = (
        worker_components.learning_rate_schedule(optimizer)
        if worker_components is not None
        else None
    )
    step, cursor = _restore_state(
        optimizer,
        scheduler,
        config,
        component_composition_digest=component_digest,
    )
    order = np.random.default_rng(config.seed).permutation(train_rows.rows)
    expected_cursor = min(
        step * config.gradient_accumulation_steps, train_rows.rows
    )
    if cursor != expected_cursor:
        raise ValueError("resume cursor is inconsistent with optimizer step")
    if not config.resume:
        smoke = _smoke_backward(model, train_rows, config)
    else:
        smoke = {"skipped": "resume"}
    initial_eval = evaluate(model, eval_rows, config)
    receipt = {
        "schema": SCHEMA,
        "config": asdict(config),
        "coverage": coverage,
        "smoke": smoke,
        "initial_eval": initial_eval,
        "train_pack_manifest_sha256": train_rows.manifest_sha256,
        "eval_pack_manifest_sha256": eval_rows.manifest_sha256,
        "available_steps": available_steps,
        "total_steps": total_steps,
        "learning_rate_schedule": {
            "implementation": "rwkv_lab.schedule.powercool.v1",
            "configuration": asdict(learning_rate_schedule),
            "step_domain": "optimizer_step",
        },
        "training_components": component_evidence,
        "component_composition_digest": component_digest,
        "dropped_rows": train_rows.rows
        - min(
            train_rows.rows,
            total_steps * config.gradient_accumulation_steps,
        ),
    }
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    # trainboard ingests the repository-wide runs/*/train.jsonl protocol.
    # Keep the canonical dashboard aliases alongside the descriptive names so
    # both generic cards and trainer-specific metric charts work.
    log = (run_dir / "train.jsonl").open("a", encoding="utf-8")
    log.write(
        json.dumps(
            {
                "kind": "eval",
                "step": step,
                **initial_eval,
                "ppl": initial_eval["perplexity"],
            }
        )
        + "\n"
    )
    log.flush()
    _write_status(
        run_dir,
        state="training",
        step=step,
        cursor=cursor,
        total_steps=total_steps,
    )

    interrupted = {"value": False}

    def handle_signal(signum, _frame):
        interrupted["value"] = True
        print(f"received signal {signum}; saving after current optimizer step", flush=True)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    device = torch.device(f"cuda:{config.cuda_index}")
    model.train()
    last_metrics: dict[str, Any] = {}
    latest_checkpoint = Path(config.resume) if config.resume else None
    latest_checkpoint_step = step if config.resume else -1
    started = time.perf_counter()
    tokens_since_log = 0
    while step < total_steps:
        optimizer.zero_grad(set_to_none=True)
        lm_total = torch.zeros((), device=device, dtype=torch.float32)
        aux_total = torch.zeros((), device=device, dtype=torch.float32)
        micro_batches = min(
            config.gradient_accumulation_steps,
            train_rows.rows - cursor,
        )
        if micro_batches < 1:
            raise RuntimeError("optimizer step has no packed rows remaining")
        for _ in range(micro_batches):
            row = int(order[cursor])
            cursor += 1
            tokens = train_rows.tensor(row, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, lm_loss, aux_loss = _causal_loss(
                    model,
                    tokens,
                    router_aux_coef=config.router_aux_loss_coef,
                    training=True,
                )
            (loss / micro_batches).backward()
            lm_total = lm_total + lm_loss.float()
            if aux_loss is not None:
                aux_total = aux_total + aux_loss.float()
            tokens_since_log += config.context_length
            _write_status(
                run_dir,
                state="training",
                step=step,
                cursor=cursor,
                total_steps=total_steps,
            )
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            config.max_grad_norm,
            error_if_nonfinite=True,
        )
        if scheduler is None:
            lr = config.learning_rate * powercool_multiplier(
                step, learning_rate_schedule
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
        else:
            lr = float(scheduler.get_last_lr()[0])
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        step += 1
        if worker_step_profiler is not None:
            worker_step_profiler.step(step)
        should_log = step % config.log_every == 0
        should_save = bool(config.save_every and step % config.save_every == 0)
        # Converting CUDA scalars to Python values synchronizes the device.
        # Keep the hot path asynchronous and pay for that synchronization only
        # when a metric is actually consumed.
        if should_log or should_save or interrupted["value"]:
            torch.cuda.synchronize(device)
            last_metrics = {
                "kind": "train",
                "step": step,
                "loss": float(lm_total / micro_batches),
                "router_aux_loss": float(aux_total / micro_batches),
                "lr": lr,
                "grad_norm": float(grad_norm),
                "gnorm": float(grad_norm),
            }
        if should_log:
            elapsed = time.perf_counter() - started
            last_metrics["tokens_per_second"] = tokens_since_log / max(elapsed, 1e-9)
            last_metrics["tok_per_sec"] = last_metrics["tokens_per_second"]
            last_metrics["cuda_allocated_gib"] = torch.cuda.memory_allocated() / 2**30
            last_metrics["cuda_reserved_gib"] = torch.cuda.memory_reserved() / 2**30
            log.write(json.dumps(last_metrics) + "\n")
            log.flush()
            print(json.dumps(last_metrics), flush=True)
            started = time.perf_counter()
            tokens_since_log = 0
        if config.eval_every and step % config.eval_every == 0:
            metrics = evaluate(model, eval_rows, config)
            log.write(
                json.dumps(
                    {
                        "kind": "eval",
                        "step": step,
                        **metrics,
                        "ppl": metrics["perplexity"],
                    }
                )
                + "\n"
            )
            log.flush()
        if should_save:
            latest_checkpoint = _save_checkpoint(
                model,
                optimizer,
                scheduler,
                config,
                step,
                cursor,
                run_dir,
                last_metrics,
                component_composition_digest=component_digest,
            )
            latest_checkpoint_step = step
        if interrupted["value"]:
            checkpoint = latest_checkpoint
            if checkpoint is None or latest_checkpoint_step != step:
                checkpoint = _save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    config,
                    step,
                    cursor,
                    run_dir,
                    last_metrics,
                    component_composition_digest=component_digest,
                )
            log.close()
            _write_status(
                run_dir,
                state="interrupted",
                step=step,
                cursor=cursor,
                total_steps=total_steps,
            )
            return {"status": "interrupted", "step": step, "checkpoint": str(checkpoint)}

    final_eval = evaluate(model, eval_rows, config)
    checkpoint = latest_checkpoint
    if checkpoint is None or latest_checkpoint_step != step:
        checkpoint = _save_checkpoint(
            model,
            optimizer,
            scheduler,
            config,
            step,
            cursor,
            run_dir,
            final_eval,
            component_composition_digest=component_digest,
        )
    log.write(
        json.dumps(
            {
                "kind": "eval",
                "step": step,
                **final_eval,
                "ppl": final_eval["perplexity"],
            }
        )
        + "\n"
    )
    log.close()
    result = {
        "schema": SCHEMA,
        "status": "complete",
        "step": step,
        "checkpoint": str(checkpoint),
        "initial_eval": initial_eval,
        "final_eval": final_eval,
    }
    (run_dir / "complete.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    _write_status(
        run_dir,
        state="complete",
        step=step,
        cursor=cursor,
        total_steps=total_steps,
    )
    return result


def prepare_run(config: QwenAO3Config) -> dict[str, Any]:
    config.validate()
    run_dir = Path(config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "train-config.json"
    config_path.write_text(json.dumps(asdict(config), indent=2, sort_keys=True) + "\n")
    root = Path(__file__).resolve().parents[2]
    launcher = run_dir / "launch.sh"
    launcher.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        f"ROOT={shlex.quote(str(root))}\n"
        'VENV="${AO3_CPT_VENV:-$ROOT/.venv-ao3-cpt}"\n'
        'export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"\n'
        'export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"\n'
        'exec "$VENV/bin/python" -m rwkv_lab.qwen_ao3_cpt train '
        '--config "$RUN_DIR/train-config.json"\n'
    )
    launcher.chmod(0o755)
    receipt = {
        "schema": SCHEMA,
        "config": str(config_path.resolve()),
        "launcher": str(launcher.resolve()),
        "fit_audit": checkpoint_fit_audit(
            config.model_dir,
            dense_rank=config.dense_rank,
            expert_rank=config.expert_rank,
            router_rank=config.router_rank,
        ),
    }
    (run_dir / "preparation-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return receipt


def _config_from_json(path: Path) -> QwenAO3Config:
    return QwenAO3Config(**json.loads(path.read_text()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--model-dir", type=Path, required=True)
    audit.add_argument("--dense-rank", type=int, default=64)
    audit.add_argument("--expert-rank", type=int, default=4)
    audit.add_argument("--router-rank", type=int, default=8)
    prepare = subparsers.add_parser("prepare-run")
    prepare.add_argument("--config", type=Path, required=True)
    run = subparsers.add_parser("train")
    run.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "audit":
        result = checkpoint_fit_audit(
            args.model_dir,
            dense_rank=args.dense_rank,
            expert_rank=args.expert_rank,
            router_rank=args.router_rank,
        )
    elif args.action == "prepare-run":
        result = prepare_run(_config_from_json(args.config))
    else:
        result = train(_config_from_json(args.config))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
